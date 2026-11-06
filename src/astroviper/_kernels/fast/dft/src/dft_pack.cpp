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

}  // namespace

template <typename ModelT>
bool analyse_plane(const ModelT* plane, i64 n_l, i64 n_m, const i64* pol_map, i64 n_pol,
                   const double* m, PlaneLayout& layout) {
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
    auto values = [&](const Component& c, i64 pixel) {
        return plane + c.pol * n_pixels + pixel;
    };

    // One pass: the magnitude bound, the non-zero candidates, rows, columns and
    // pixels. Equal or negated candidates have the same non-zero pixels, so the
    // kept rows and columns are those of the distinct components.
    const double limit =
        kMaxSumOfValues / static_cast<double>(std::max<i64>(n_pixels * n_candidates, 1));
    bool too_large = false;
    std::vector<unsigned char> nonzero(static_cast<std::size_t>(n_candidates), 0);
    std::vector<unsigned char> row_kept(static_cast<std::size_t>(n_l), 0);
    std::vector<unsigned char> column_kept(static_cast<std::size_t>(n_m), 0);
    std::vector<unsigned char> pixel_kept(static_cast<std::size_t>(n_m));
    layout.nonzero_pixels = 0;
    for (i64 i = 0; i < n_l; ++i) {
        std::fill(pixel_kept.begin(), pixel_kept.end(), 0);
        for (i64 c = 0; c < n_candidates; ++c) {
            const ModelT* row = values(candidates[c], i * n_m);
            const int part = candidates[c].part;
            unsigned char any = 0;
            for (i64 j = 0; j < n_m; ++j) {
                const double x = part_of(row[j], part);
                too_large |= !(std::abs(x) <= limit);
                const unsigned char kept = x != 0.0;
                pixel_kept[j] |= kept;
                any |= kept;
            }
            nonzero[c] |= any;
        }
        i64 count = 0;
        for (i64 j = 0; j < n_m; ++j) {
            count += pixel_kept[j];
            column_kept[j] |= pixel_kept[j];
        }
        layout.nonzero_pixels += count;
        row_kept[i] = count > 0;
    }
    if (too_large) return false;

    // The distinct components and the sign of every candidate relative to its
    // component: bitwise equal (+1) or bitwise negated (-1).
    std::vector<int> unique(static_cast<std::size_t>(n_candidates), -1);
    std::vector<double> sign(static_cast<std::size_t>(n_candidates), 1.0);
    layout.components.clear();
    for (i64 c = 0; c < n_candidates; ++c) {
        if (!nonzero[c]) continue;
        for (std::size_t u = 0; u < layout.components.size() && unique[c] < 0; ++u) {
            const Component& a = candidates[c];
            const Component& b = layout.components[u];
            bool equal = true, negated = true;
            for (i64 i = 0; i < n_l && (equal || negated); ++i) {
                const ModelT* row_a = values(a, i * n_m);
                const ModelT* row_b = values(b, i * n_m);
                for (i64 j = 0; j < n_m; ++j) {
                    const auto x = std::bit_cast<std::uint64_t>(part_of(row_a[j], a.part));
                    const auto y = std::bit_cast<std::uint64_t>(part_of(row_b[j], b.part));
                    equal &= x == y;
                    negated &= x == (y ^ kSignBit);
                }
            }
            if (equal || negated) {
                unique[c] = static_cast<int>(u);
                sign[c] = equal ? 1.0 : -1.0;
            }
        }
        if (unique[c] < 0) {
            unique[c] = static_cast<int>(layout.components.size());
            layout.components.push_back(candidates[c]);
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
                                        PlaneLayout&);                                           \
    template void pack_plane<MODELT>(const MODELT*, i64, i64, const PlaneLayout&, const double*, \
                                     const double*, int, std::vector<Scratch>&, double*, double*);

DFT_FAST_INSTANTIATE(float)
DFT_FAST_INSTANTIATE(double)
DFT_FAST_INSTANTIATE(std::complex<float>)
DFT_FAST_INSTANTIATE(std::complex<double>)
#undef DFT_FAST_INSTANTIATE

}  // namespace dft_fast
