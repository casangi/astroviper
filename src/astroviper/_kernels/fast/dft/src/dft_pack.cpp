// Path S's preparation of one model plane: which real planes it sums and where
// (analyse_plane), and the packed, folded weights of its GEMM (pack_plane).

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <unordered_map>

#include "dft_fast.hpp"

namespace dft_fast {

namespace {

template <typename T>
constexpr bool kComplex = false;
template <typename T>
constexpr bool kComplex<std::complex<T>> = true;

// Part 0 (real) or 1 (imaginary) of a model value, as the reference reads it.
template <typename ModelT>
double part_of(const ModelT& value, int part) {
    if constexpr (kComplex<ModelT>) {
        return static_cast<double>(part == 0 ? value.real() : value.imag());
    } else {
        return static_cast<double>(value);
    }
}

// n = sqrt(1 - l^2 - m^2) exactly as the reference's direction_n forms it.
double direction_n(double l, double m) {
    const double r2 = l * l + m * m;
    return std::sqrt(1.0 - r2);
}

// No referenced value may exceed 2^960 / (the number of referenced values):
// their sum S is then at most 2^960 and, as n >= 2^-26.5 in front of the
// horizon (1 - l^2 - m^2 >= 2^-53), every partial sum and product of path S
// is at most about 2 S / n < 2^988, far from overflow. NaN and infinity fail.
constexpr double kMaxSumOfValues = 0x1p960;

constexpr std::uint64_t kSignBit = std::uint64_t{1} << 63;

// Model values per worker of the analysis at least: a smaller plane is
// analysed by fewer workers than the call's threads (a thread's start costs
// about as much as reading this many values).
constexpr i64 kAnalysisGrain = i64{1} << 16;

// Index of the pair of candidates a < b.
i64 pair_index(i64 a, i64 b) {
    return b * (b - 1) / 2 + a;
}

// What the analysis needs of one row of a candidate (part `part` of n values):
// whether a value is not finite or above limit, which values are non-zero
// (ORed into kept) and the OR of the values' bits (0: every value is +0.0).
// Returns whether a value is non-zero.
template <typename ModelT>
bool scan_row(const ModelT* __restrict row, int part, i64 n, double limit,
              unsigned char* __restrict kept, bool& too_large, std::uint64_t& bits) {
    unsigned char large = 0, any = 0;
    std::uint64_t all = 0;
    for (i64 j = 0; j < n; ++j) {
        const double x = part_of(row[j], part);
        large |= !(std::abs(x) <= limit);
        const unsigned char nonzero = x != 0.0;
        kept[j] |= nonzero;
        any |= nonzero;
        all |= std::bit_cast<std::uint64_t>(x);
    }
    too_large |= large != 0;
    bits = all;
    return any != 0;
}

// The bits of two rows of candidates compared value by value: the OR over the
// row of (x XOR y) and of (x XOR y XOR sign), zero where the row of b is the
// row of a bitwise, or its bitwise negation.
template <typename ModelT>
void compare_rows(const ModelT* __restrict a, int part_a, const ModelT* __restrict b, int part_b,
                  i64 n, std::uint64_t& differ, std::uint64_t& differ_negated) {
    std::uint64_t to_equal = 0, to_negated = 0;
    for (i64 j = 0; j < n; ++j) {
        const auto x = std::bit_cast<std::uint64_t>(part_of(b[j], part_b));
        const auto y = std::bit_cast<std::uint64_t>(part_of(a[j], part_a));
        to_equal |= x ^ y;
        to_negated |= x ^ y ^ kSignBit;
    }
    differ = to_equal;
    differ_negated = to_negated;
}

// Adds a row's kept pixels to the kept columns; returns their number.
i64 merge_row(const unsigned char* __restrict kept, i64 n, unsigned char* __restrict columns) {
    i64 count = 0;
    for (i64 j = 0; j < n; ++j) {
        count += kept[j];
        columns[j] |= kept[j];
    }
    return count;
}

}  // namespace

template <typename ModelT>
bool analyse_plane(const ModelT* plane, i64 n_l, i64 n_m, const i64* pol_map, i64 n_pol,
                   const double* m, int threads, PlaneLayout& layout) {
    using Component = PlaneLayout::Component;
    constexpr int kParts = kComplex<ModelT> ? 2 : 1;
    const i64 n_pixels = n_l * n_m;

    // The candidates: every part of each referenced polarization, in the order
    // of the first reference.
    std::vector<Component> candidates;
    for (i64 p = 0; p < n_pol; ++p) {
        const bool seen = std::any_of(candidates.begin(), candidates.end(),
                                      [&](const Component& c) { return c.pol == pol_map[p]; });
        if (seen) continue;
        for (int part = 0; part < kParts; ++part) candidates.push_back({pol_map[p], part});
    }
    const auto n_candidates = static_cast<i64>(candidates.size());
    const i64 n_pairs = n_candidates * (n_candidates - 1) / 2;
    auto values = [&](const Component& c, i64 pixel) {
        return plane + c.pol * n_pixels + pixel;
    };

    // One pass over the rows, shared by the workers: the magnitude bound, the
    // non-zero candidates, rows, columns and pixels, and for each pair of
    // candidates whether one is the other bitwise, or its bitwise negation, on
    // every pixel. A worker stops comparing a pair once it differs both ways,
    // and two rows of +0.0 only are equal (not negated) without a comparison.
    // Each worker reduces its rows with OR, AND and integer sums, so the layout
    // does not depend on how the rows are shared. Equal or negated candidates
    // have the same non-zero pixels, so the kept rows and columns are those of
    // the distinct components.
    const double limit =
        kMaxSumOfValues / static_cast<double>(std::max<i64>(n_pixels * n_candidates, 1));
    struct Partial {
        bool too_large = false;
        std::vector<unsigned char> nonzero, equal, negated, column_kept, pixel_kept;
        std::vector<std::uint64_t> bits;  // of the current row, per candidate
        i64 nonzero_pixels = 0;
    };
    const int workers =
        resolve_threads(threads, std::min(n_l, n_pixels * n_candidates / kAnalysisGrain + 1));
    std::vector<Partial> partial(static_cast<std::size_t>(workers));
    for (Partial& w : partial) {
        w.nonzero.assign(static_cast<std::size_t>(n_candidates), 0);
        w.equal.assign(static_cast<std::size_t>(n_pairs), 1);
        w.negated.assign(static_cast<std::size_t>(n_pairs), 1);
        w.column_kept.assign(static_cast<std::size_t>(n_m), 0);
        w.pixel_kept.resize(static_cast<std::size_t>(n_m));
        w.bits.resize(static_cast<std::size_t>(n_candidates));
    }
    std::vector<unsigned char> row_kept(static_cast<std::size_t>(n_l), 0);
    parallel_for(workers, n_l, [&](int worker, i64 i) {
        Partial& w = partial[static_cast<std::size_t>(worker)];
        std::fill(w.pixel_kept.begin(), w.pixel_kept.end(), 0);
        for (i64 b = 0; b < n_candidates; ++b) {
            const ModelT* row_b = values(candidates[b], i * n_m);
            const int part_b = candidates[b].part;
            if (scan_row(row_b, part_b, n_m, limit, w.pixel_kept.data(), w.too_large, w.bits[b]))
                w.nonzero[b] = 1;
            for (i64 a = 0; a < b; ++a) {
                const i64 k = pair_index(a, b);
                if (!w.equal[k] && !w.negated[k]) continue;
                if ((w.bits[a] | w.bits[b]) == 0) {  // +0.0 everywhere in both
                    w.negated[k] = 0;
                    continue;
                }
                std::uint64_t differ, differ_negated;
                compare_rows(values(candidates[a], i * n_m), candidates[a].part, row_b, part_b,
                             n_m, differ, differ_negated);
                w.equal[k] &= differ == 0;
                w.negated[k] &= differ_negated == 0;
            }
        }
        const i64 count = merge_row(w.pixel_kept.data(), n_m, w.column_kept.data());
        w.nonzero_pixels += count;
        row_kept[i] = count > 0;
    });
    bool too_large = false;
    std::vector<unsigned char> nonzero(static_cast<std::size_t>(n_candidates), 0);
    std::vector<unsigned char> equal(static_cast<std::size_t>(n_pairs), 1);
    std::vector<unsigned char> negated(static_cast<std::size_t>(n_pairs), 1);
    std::vector<unsigned char> column_kept(static_cast<std::size_t>(n_m), 0);
    layout.nonzero_pixels = 0;
    for (const Partial& w : partial) {
        too_large |= w.too_large;
        for (i64 c = 0; c < n_candidates; ++c) nonzero[c] |= w.nonzero[c];
        for (i64 k = 0; k < n_pairs; ++k) {
            equal[k] &= w.equal[k];
            negated[k] &= w.negated[k];
        }
        for (i64 j = 0; j < n_m; ++j) column_kept[j] |= w.column_kept[j];
        layout.nonzero_pixels += w.nonzero_pixels;
    }
    if (too_large) return false;

    // The distinct components, in candidate order, and the sign of every
    // candidate relative to its component: bitwise equal (+1) or bitwise
    // negated (-1). Equality up to the sign is an equivalence, so a candidate
    // belongs to the component of the first earlier non-zero candidate it
    // matches, with the product of the two signs.
    std::vector<int> unique(static_cast<std::size_t>(n_candidates), -1);
    std::vector<double> sign(static_cast<std::size_t>(n_candidates), 1.0);
    layout.components.clear();
    for (i64 b = 0; b < n_candidates; ++b) {
        if (!nonzero[b]) continue;
        for (i64 a = 0; a < b && unique[b] < 0; ++a) {
            const i64 k = pair_index(a, b);
            if (!nonzero[a] || !(equal[k] || negated[k])) continue;
            unique[b] = unique[a];
            sign[b] = equal[k] ? sign[a] : -sign[a];
        }
        if (unique[b] < 0) {
            unique[b] = static_cast<int>(layout.components.size());
            layout.components.push_back(candidates[b]);
        }
    }
    layout.correlations.assign(static_cast<std::size_t>(n_pol), {});
    for (i64 p = 0; p < n_pol; ++p) {
        for (i64 c = 0; c < n_candidates; ++c) {
            if (candidates[c].pol != pol_map[p]) continue;
            PlaneLayout::Correlation& correlation = layout.correlations[p];
            (candidates[c].part == 0 ? correlation.re : correlation.im) = unique[c];
            (candidates[c].part == 0 ? correlation.re_sign : correlation.im_sign) = sign[c];
        }
    }

    layout.rows.clear();
    for (i64 i = 0; i < n_l; ++i) {
        if (row_kept[i]) layout.rows.push_back(i);
    }

    // Folding: columns with m_b = -m_a bitwise have B_b = conj(B_a) bit for bit
    // (column_phasors), so sum_j W_j B_j over the pair is (W_a + W_b) cos + i
    // (W_a - W_b) sin. Each kept column is in at most one pair (the first
    // column of each negative value), the pairs in the column order of their
    // positive member, then the unpaired columns in column order; anything
    // else (an axis without exact mirror pairs) stays unfolded, still exact.
    std::vector<i64> kept;
    for (i64 j = 0; j < n_m; ++j) {
        if (column_kept[j]) kept.push_back(j);
    }
    std::unordered_map<std::uint64_t, std::size_t> negative;  // bits of m < 0 -> index in kept
    for (std::size_t k = 0; k < kept.size(); ++k) {
        if (m[kept[k]] < 0.0) negative.emplace(std::bit_cast<std::uint64_t>(m[kept[k]]), k);
    }
    std::vector<unsigned char> paired(kept.size(), 0);
    layout.column_a.clear();
    layout.column_b.clear();
    layout.m_a.clear();
    for (std::size_t k = 0; k < kept.size(); ++k) {
        const double m_k = m[kept[k]];
        if (!(m_k > 0.0)) continue;
        const auto partner = negative.find(std::bit_cast<std::uint64_t>(-m_k));
        if (partner == negative.end() || paired[partner->second]) continue;
        paired[k] = paired[partner->second] = 1;
        layout.column_a.push_back(kept[k]);
        layout.column_b.push_back(kept[partner->second]);
        layout.m_a.push_back(m_k);
    }
    for (std::size_t k = 0; k < kept.size(); ++k) {
        if (paired[k]) continue;
        layout.column_a.push_back(kept[k]);
        layout.column_b.push_back(-1);
        layout.m_a.push_back(m[kept[k]]);
    }
    return true;
}

template <typename ModelT>
void pack_plane(const ModelT* plane, i64 n_l, i64 n_m, const PlaneLayout& layout,
                const double* l, const double* m, int threads, std::vector<Scratch>& scratch,
                double* p, double* q) {
    const i64 n_pixels = n_l * n_m;
    const auto n_components = static_cast<i64>(layout.components.size());
    const auto n_rows = static_cast<i64>(layout.rows.size());
    const auto n_columns = static_cast<i64>(layout.column_a.size());
    const i64* column_a = layout.column_a.data();
    const i64* column_b = layout.column_b.data();

    // One item per kept row: its n are computed once for all its components.
    parallel_for(resolve_threads(threads, n_rows), n_rows, [&](int worker, i64 kept_row) {
        const i64 i = layout.rows[kept_row];
        double* n_a = scratch[worker].n_a.get(n_columns);
        double* n_b = scratch[worker].n_b.get(n_columns);
        for (i64 f = 0; f < n_columns; ++f) {
            n_a[f] = direction_n(l[i], m[column_a[f]]);
            n_b[f] = column_b[f] >= 0 ? direction_n(l[i], m[column_b[f]]) : 1.0;
        }
        for (i64 c = 0; c < n_components; ++c) {
            const PlaneLayout::Component& component = layout.components[c];
            const ModelT* row = plane + component.pol * n_pixels + i * n_m;
            const i64 r = kept_row * n_components + c;
            const i64 offset = (r / kTileRows) * n_columns * kTileRows + r % kTileRows;
            double* p_r = p + offset;
            double* q_r = q + offset;
            for (i64 f = 0; f < n_columns; ++f) {
                const double w_a = part_of(row[column_a[f]], component.part) / n_a[f];
                if (column_b[f] >= 0) {
                    const double w_b = part_of(row[column_b[f]], component.part) / n_b[f];
                    p_r[f * kTileRows] = w_a + w_b;
                    q_r[f * kTileRows] = w_a - w_b;
                } else {
                    p_r[f * kTileRows] = w_a;
                    q_r[f * kTileRows] = w_a;
                }
            }
        }
    });

    // The padding rows of the last micro-panel take part in the GEMM (their
    // results are not used): zeros.
    const i64 n_packed = n_rows * n_components;
    const i64 padded = (n_packed + kTileRows - 1) / kTileRows * kTileRows;
    for (i64 r = n_packed; r < padded; ++r) {
        const i64 offset = (r / kTileRows) * n_columns * kTileRows + r % kTileRows;
        for (i64 f = 0; f < n_columns; ++f) {
            p[offset + f * kTileRows] = 0.0;
            q[offset + f * kTileRows] = 0.0;
        }
    }
}

#define DFT_FAST_INSTANTIATE(MODELT)                                                              \
    template bool analyse_plane<MODELT>(const MODELT*, i64, i64, const i64*, i64, const double*, \
                                        int, PlaneLayout&);                                      \
    template void pack_plane<MODELT>(const MODELT*, i64, i64, const PlaneLayout&, const double*, \
                                     const double*, int, std::vector<Scratch>&, double*, double*);

DFT_FAST_INSTANTIATE(float)
DFT_FAST_INSTANTIATE(double)
DFT_FAST_INSTANTIATE(std::complex<float>)
DFT_FAST_INSTANTIATE(std::complex<double>)
#undef DFT_FAST_INSTANTIATE

}  // namespace dft_fast
