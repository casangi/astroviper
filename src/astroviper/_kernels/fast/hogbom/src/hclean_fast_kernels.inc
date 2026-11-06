// The code of the fast Hogbom routines shared by every ISA build
// (hclean_fast_kernels.hpp). Included inside an anonymous namespace by
// hclean_fast_generic.cpp and hclean_fast_avx2.cpp, after <cmath>, <cstdint>,
// hclean_fast_kernels.hpp and the unit's Simd<double> and Simd<float>: W
// lanes with load, store, splat, mul and sub, and for W > 1 also zero, iota,
// add, magnitude, greater, select, store_squares, and keep and clear (the
// lanes the mask searches). Everything here has internal linkage, so the
// AVX2 unit defines no weak symbol.
//
// The bits are the reference's (hclean.cpp: clean<T>, the cube kernels and
// scan_row):
//  * a subtraction r = r - pv * psf is a rounded product, then a rounded
//    difference (both units are built with -ffp-contract=off, and -mavx2 does
//    not enable FMA), pixel by pixel, so the order of the lanes does not
//    matter, nor whether a pixel that is not searched is subtracted at once
//    or later, in the same order;
//  * the sum of squares of a row is scan_row's: the sequential double sum, in
//    column order, of (double)v * (double)v over the searched pixels,
//    starting at +0.0. Every row keeps its own chain; the chains of kRows
//    rows are interleaved only to hide the latency of the additions. A pixel
//    of a segment that is not searched, or a row that has run out of pixels,
//    adds +0.0, which leaves the sum (>= +0, NaN or +inf) unchanged;
//  * the largest |v| of a row and its first column are exact: each lane keeps
//    its first strict maximum, and the first column holding the row's largest
//    |v| is the smallest column among the lanes that hold it. NaN is above
//    nothing, as in scan_row, and a pixel that is not searched counts as 0.

using hclean_fast::Component;
using hclean_fast::ExteriorFlush;
using hclean_fast::Run;
using hclean_fast::SearchPass;
using hclean_fast::SearchRow;

constexpr int kChunk = 256;  // pixels per row and step of the interleaved sums
constexpr int kRows = 8;     // rows whose sums are interleaved
constexpr int kBlock = 8;    // vectors per register block of the deferred subtraction
constexpr int kAhead = 6;    // components ahead whose PSF samples are prefetched

inline int min_int(int a, int b) {
    return b < a ? b : a;
}

inline int max_int(int a, int b) {
    return a < b ? b : a;
}

// Pixels [begin, end) of a row, end - begin <= kChunk: with Update, each
// first loses pv x psf[x - begin]; then its square (in double) goes to
// squares[x - begin] and it is folded into the row's running maximum (peak at
// column peak_x), the first strict maximum winning. With Masked, a pixel
// whose mask byte is 0 is updated but not searched (its square is +0.0).
template <typename T, bool Update, bool Masked>
inline void span(T* __restrict row, const T* __restrict psf, const std::uint8_t* __restrict mask,
                 T pv, int begin, int end, double* __restrict squares, T& peak, int& peak_x) {
    using S = Simd<T>;
    int x = begin;
    if constexpr (S::W > 1) {
        using V = typename S::V;
        constexpr int W = S::W;
        if (end - begin >= W) {
            // Two accumulators of (maximum, offset from begin), so that the
            // compare-and-select chain of one does not bound the loop. The
            // offsets are below kChunk, exact in T.
            const V vpv = S::splat(pv);
            const V step = S::splat(static_cast<T>(2 * W));
            V max0 = S::zero(), max1 = S::zero(), at0 = S::zero(), at1 = S::zero();
            V offset0 = S::iota(0), offset1 = S::iota(W);
            const auto fold = [&](int at, V& vmax, V& vat, const V& offset) {
                V v = S::load(row + at);
                if constexpr (Update) {
                    v = S::sub(v, S::mul(vpv, S::load(psf + (at - begin))));
                    S::store(row + at, v);
                }
                V magnitude = S::magnitude(v);
                if constexpr (Masked) {
                    const auto keep = S::keep(mask + at);
                    S::store_squares(squares + (at - begin), v, keep);
                    magnitude = S::clear(magnitude, keep);
                } else {
                    S::store_squares(squares + (at - begin), v);
                }
                const auto above = S::greater(magnitude, vmax);
                vmax = S::select(vmax, magnitude, above);
                vat = S::select(vat, offset, above);
            };
            for (; x + 2 * W <= end; x += 2 * W) {
                fold(x, max0, at0, offset0);
                fold(x + W, max1, at1, offset1);
                offset0 = S::add(offset0, step);
                offset1 = S::add(offset1, step);
            }
            if (x + W <= end) {  // offset0 holds the offsets of x
                fold(x, max0, at0, offset0);
                x += W;
            }
            alignas(64) T lane_max[2 * W];
            alignas(64) T lane_at[2 * W];
            S::store(lane_max, max0);
            S::store(lane_max + W, max1);
            S::store(lane_at, at0);
            S::store(lane_at + W, at1);
            T largest = lane_max[0];
            for (int lane = 1; lane < 2 * W; ++lane) {
                if (lane_max[lane] > largest) largest = lane_max[lane];
            }
            if (largest > peak) {  // then every lane holding it found it at least once
                int first = kChunk;
                for (int lane = 0; lane < 2 * W; ++lane) {
                    if (lane_max[lane] == largest) {
                        first = min_int(first, static_cast<int>(lane_at[lane]));
                    }
                }
                peak = largest;
                peak_x = begin + first;
            }
        }
    }
    for (; x < end; ++x) {
        T v = row[x];
        if constexpr (Update) {
            v = v - pv * psf[x - begin];
            row[x] = v;
        }
        if constexpr (Masked) {
            if (mask[x] == 0) {
                squares[x - begin] = 0.0;
                continue;
            }
        }
        squares[x - begin] = static_cast<double>(v) * static_cast<double>(v);
        const T magnitude = std::abs(v);
        if (magnitude > peak) {
            peak = magnitude;
            peak_x = x;
        }
    }
}

// The progress of the fused pass along one searched row.
template <typename T>
struct Cursor {
    T* row = nullptr;                    // the residual row
    const T* psf = nullptr;              // in the footprint: the PSF sample of its column x1
    const std::uint8_t* mask = nullptr;  // the mask row of a masked row
    int run = 0, run_end = 0;            // the segments of the row not yet finished
    int x = 0;                           // the next column of runs[run]
    int x1 = 0, x2 = 0;                  // the footprint's columns [x1, x2) (none off it)
    T peak = 0;
    int peak_x = 0;
};

// Pixels [begin, end) of a cursor's row: span() with the row's kind.
template <typename T, bool Update>
inline void span_of(Cursor<T>& c, const T* psf, T pv, int begin, int end, double* squares) {
    if (c.mask != nullptr) {
        span<T, Update, true>(c.row, psf, c.mask, pv, begin, end, squares, c.peak, c.peak_x);
    } else {
        span<T, Update, false>(c.row, psf, nullptr, pv, begin, end, squares, c.peak, c.peak_x);
    }
}

// The next (up to kChunk) pixels of a searched row: subtracted inside the
// footprint, squared into squares, folded into the row's maximum. Returns how
// many (0 once the row is done).
template <typename T>
inline int chunk(const Run* runs, Cursor<T>& c, T pv, double* squares) {
    int filled = 0;
    while (filled < kChunk && c.run < c.run_end) {
        const int run_end = runs[c.run].end;
        const int begin = c.x;
        const int end = min_int(run_end, begin + (kChunk - filled));
        double* out = squares + filled;
        const int u0 = max_int(begin, c.x1), u1 = min_int(end, c.x2);  // the part in the footprint
        if (u0 < u1) {
            if (begin < u0) span_of<T, false>(c, nullptr, pv, begin, u0, out);
            span_of<T, true>(c, c.psf + (u0 - c.x1), pv, u0, u1, out + (u0 - begin));
            if (u1 < end) span_of<T, false>(c, nullptr, pv, u1, end, out + (u1 - begin));
        } else {
            span_of<T, false>(c, nullptr, pv, begin, end, out);
        }
        filled += end - begin;
        if (end < run_end) {
            c.x = end;
        } else if (++c.run < c.run_end) {
            c.x = runs[c.run].begin;
        }
    }
    return filled;
}

template <typename T>
void search(const SearchPass<T>& pass, int row_begin, int row_end) {
    alignas(64) double squares[kRows][kChunk];
    const Component<T>* component = pass.component;
    const T pv = component != nullptr ? component->pv : static_cast<T>(0);
    for (int first = row_begin; first < row_end; first += kRows) {
        const int count = min_int(kRows, row_end - first);
        Cursor<T> cursor[kRows];
        for (int g = 0; g < count; ++g) {
            const SearchRow& searched = pass.rows[first + g];
            Cursor<T>& c = cursor[g];
            const std::size_t row_offset = static_cast<std::size_t>(searched.y) * pass.nx;
            c.row = pass.residual + row_offset;
            if (searched.masked) c.mask = pass.mask + row_offset;
            c.run = searched.run_begin;
            c.run_end = searched.run_end;
            c.x = pass.runs[searched.run_begin].begin;
            c.peak_x = pass.scan_begin;
            if (component != nullptr && searched.y >= component->y1 && searched.y <= component->y2) {
                c.x1 = component->x1;
                c.x2 = component->x2 + 1;
                c.psf = pass.psf + (component->offset +
                                    static_cast<std::ptrdiff_t>(searched.y) * pass.psf_nx +
                                    component->x1);
            }
        }
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
        for (;;) {
            int filled[kRows] = {};
            int longest = 0;
            for (int g = 0; g < count; ++g) {
                filled[g] = chunk<T>(pass.runs, cursor[g], pv, squares[g]);
                longest = max_int(longest, filled[g]);
            }
            if (longest == 0) break;
            for (int g = 0; g < kRows; ++g) {
                for (int i = filled[g]; i < longest; ++i) squares[g][i] = 0.0;
            }
            // kRows chains, each in its row's column order.
            for (int i = 0; i < longest; ++i) {
                s0 += squares[0][i];
                s1 += squares[1][i];
                s2 += squares[2][i];
                s3 += squares[3][i];
                s4 += squares[4][i];
                s5 += squares[5][i];
                s6 += squares[6][i];
                s7 += squares[7][i];
            }
        }
        const double sums[kRows] = {s0, s1, s2, s3, s4, s5, s6, s7};
        for (int g = 0; g < count; ++g) {
            pass.row_sum[first + g] = sums[g];
            pass.row_max[first + g] = cursor[g].peak;
            pass.row_ix[first + g] = cursor[g].peak_x;
        }
    }
}

// Prefetches the PSF samples that component c will subtract from the block
// [x0, x0 + kWidth) of a row (row_offset = the row times psf_nx) if its
// footprint covers the block, as it mostly does. The address is computed as
// an integer and only prefetched: where the footprint does not cover the
// block it may lie outside the PSF, and a prefetch never faults (testing the
// footprint first costs more than the prefetch saves, M).
template <typename T, int kWidth>
inline void prefetch(const ExteriorFlush<T>& exterior, const Component<T>& c,
                     std::ptrdiff_t row_offset, int x0) {
    const std::uintptr_t first =
        reinterpret_cast<std::uintptr_t>(exterior.psf) +
        static_cast<std::uintptr_t>(c.offset + row_offset + x0) * sizeof(T);
    constexpr int kLine = 64 / static_cast<int>(sizeof(T));
    for (int i = 0; i < kWidth; i += kLine) {
        __builtin_prefetch(reinterpret_cast<const void*>(first + i * sizeof(T)), 0, 3);
    }
    __builtin_prefetch(reinterpret_cast<const void*>(first + (kWidth - 1) * sizeof(T)), 0, 3);
}

// Columns [x0, x0 + NV * W) of row y: the components in order, the pixels held
// in registers.
template <typename T, int NV>
inline void flush_block(const ExteriorFlush<T>& exterior, T* __restrict row, int y,
                        std::ptrdiff_t row_offset, int x0) {
    using S = Simd<T>;
    using V = typename S::V;
    constexpr int W = S::W;
    constexpr int kWidth = NV * W;
    const int last = x0 + kWidth - 1;
    V acc[NV];
    for (int j = 0; j < NV; ++j) acc[j] = S::load(row + x0 + j * W);
    for (int k = 0; k < exterior.count; ++k) {
        if (kAhead > 0 && k + kAhead < exterior.count) {
            prefetch<T, kWidth>(exterior, exterior.components[k + kAhead], row_offset, x0);
        }
        const Component<T>& c = exterior.components[k];
        if (y < c.y1 || y > c.y2 || last < c.x1 || x0 > c.x2) continue;
        if (x0 >= c.x1 && last <= c.x2) {
            const T* samples = exterior.psf + (c.offset + row_offset + x0);
            const V vpv = S::splat(c.pv);
            for (int j = 0; j < NV; ++j) acc[j] = S::sub(acc[j], S::mul(vpv, S::load(samples + j * W)));
        } else {  // the block straddles an edge of the footprint: the pixels inside, one by one
            alignas(64) T values[kWidth];
            for (int j = 0; j < NV; ++j) S::store(values + j * W, acc[j]);
            const int lo = max_int(x0, c.x1), hi = min_int(last, c.x2);
            const T* samples = exterior.psf + (c.offset + row_offset + lo);
            for (int x = lo; x <= hi; ++x) values[x - x0] = values[x - x0] - c.pv * samples[x - lo];
            for (int j = 0; j < NV; ++j) acc[j] = S::load(values + j * W);
        }
    }
    for (int j = 0; j < NV; ++j) S::store(row + x0 + j * W, acc[j]);
}

// Columns [x0, x1) of row y, one by one.
template <typename T>
inline void flush_pixels(const ExteriorFlush<T>& exterior, T* row, int y, std::ptrdiff_t row_offset,
                         int x0, int x1) {
    for (int x = x0; x < x1; ++x) {
        T v = row[x];
        for (int k = 0; k < exterior.count; ++k) {
            const Component<T>& c = exterior.components[k];
            if (y < c.y1 || y > c.y2 || x < c.x1 || x > c.x2) continue;
            v = v - c.pv * exterior.psf[c.offset + row_offset + x];
        }
        row[x] = v;
    }
}

template <typename T>
void flush_rows(const ExteriorFlush<T>& exterior, int y_begin, int y_end) {
    constexpr int W = Simd<T>::W;
    if (exterior.count <= 0) return;
    for (int y = y_begin; y < y_end; ++y) {
        T* row = exterior.residual + static_cast<std::size_t>(y) * exterior.nx;
        const std::ptrdiff_t row_offset = static_cast<std::ptrdiff_t>(y) * exterior.psf_nx;
        for (int r = exterior.row_start[y]; r < exterior.row_start[y + 1]; ++r) {
            const int end = exterior.runs[r].end;
            int x = exterior.runs[r].begin;
            for (; x + kBlock * W <= end; x += kBlock * W) {
                flush_block<T, kBlock>(exterior, row, y, row_offset, x);
            }
            // The rest in halving blocks, so that a short tail pays the cost
            // per component a few times only.
            for (; x + 4 * W <= end; x += 4 * W) flush_block<T, 4>(exterior, row, y, row_offset, x);
            if (x + 2 * W <= end) {
                flush_block<T, 2>(exterior, row, y, row_offset, x);
                x += 2 * W;
            }
            if (x + W <= end) {
                flush_block<T, 1>(exterior, row, y, row_offset, x);
                x += W;
            }
            if (x < end) flush_pixels<T>(exterior, row, y, row_offset, x, end);
        }
    }
}

void search_f64(const SearchPass<double>& pass, int row_begin, int row_end) {
    search<double>(pass, row_begin, row_end);
}

void search_f32(const SearchPass<float>& pass, int row_begin, int row_end) {
    search<float>(pass, row_begin, row_end);
}

void flush_f64(const ExteriorFlush<double>& exterior, int y_begin, int y_end) {
    flush_rows<double>(exterior, y_begin, y_end);
}

void flush_f32(const ExteriorFlush<float>& exterior, int y_begin, int y_end) {
    flush_rows<float>(exterior, y_begin, y_end);
}
