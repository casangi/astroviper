// The code of the fast Hogbom routines shared by every ISA build
// (hclean_fast_kernels.hpp). Included inside an anonymous namespace by
// hclean_fast_generic.cpp and hclean_fast_avx2.cpp, after <cstdint>,
// hclean_fast_kernels.hpp and the unit's Simd<double> and Simd<float>: W
// lanes with load, store, splat, mul, sub, zero, magnitude, greater, select,
// keep and clear (the lanes the mask searches), and sums of squares in double
// (Sums, no_sums, add_squares and store_sums); for W > 1 also iota, add and
// store_squares, and largest and first (the reduction of the lanes' maxima).
// Everything here has internal linkage and calls no function of a standard
// header, so the AVX2 unit defines no weak symbol, even at -O0.
//
// The bits are the reference's (hclean.cpp: clean<T>, the cube kernels and
// scan_row) for every value that is not a NaN; a NaN result is a NaN, whose
// sign and payload depend on the operand order the compiler picks for a
// commutative operation, in the reference as here:
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
// A short row (short_row) does the same operations pixel by pixel. The direct
// pass over a small plane subtracts each component from its whole footprint
// before it scans the box, so it too makes the reference's operations on
// every pixel in their order, and scans the rows as the fused pass does.

using hclean_fast::ColumnPass;
using hclean_fast::Component;
using hclean_fast::DirectPass;
using hclean_fast::DirectScan;
using hclean_fast::ExteriorFlush;
using hclean_fast::Run;
using hclean_fast::SearchPass;
using hclean_fast::SearchRow;

constexpr int kChunk = 256;  // pixels per row and step of the interleaved sums
constexpr int kRows = 8;     // rows whose sums are interleaved
constexpr int kBlock = 8;    // vectors per register block of the deferred subtraction
constexpr int kAhead = 6;    // components ahead whose PSF samples are prefetched

// A searched row of at most kShortRow pixels is scanned pixel by pixel
// (short_row): on so few pixels the setup and the reduction of the lanes cost
// more than they save (M).
constexpr int kShortRow = 48;

inline int min_int(int a, int b) {
    return b < a ? b : a;
}

inline int max_int(int a, int b) {
    return a < b ? b : a;
}

// |v|, inline at every optimisation level (std::abs of <cmath> is a weak
// symbol at -O0).
inline double magnitude_of(double v) {
    return __builtin_fabs(v);
}

inline float magnitude_of(float v) {
    return __builtin_fabsf(v);
}

// Folds the searched pixel v at column x into a row's sum of squares and its
// first strict maximum of |v|, without a branch on the data.
template <typename T>
inline void fold_pixel(T v, int x, double& sum, T& peak, int& peak_x) {
    sum += static_cast<double>(v) * static_cast<double>(v);
    const T magnitude = magnitude_of(v);
    const bool above = magnitude > peak;  // false for NaN
    peak_x = above ? x : peak_x;
    peak = above ? magnitude : peak;
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
            // The lanes' maxima are >= +0 and never NaN.
            const T largest = S::largest(max0, max1);
            if (largest > peak) {  // then every lane holding it found it at least once
                peak = largest;
                peak_x = begin + S::first(max0, at0, max1, at1, largest);
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
        const T magnitude = magnitude_of(v);
        const bool above = magnitude > peak;
        peak_x = above ? x : peak_x;
        peak = above ? magnitude : peak;
    }
}

// Pixels [begin, end) of a short row: span() and the row's chain in one loop.
template <typename T, bool Update, bool Masked>
inline void short_span(T* __restrict row, const T* __restrict psf, const std::uint8_t* __restrict mask,
                       T pv, int begin, int end, double& sum, T& peak, int& peak_x) {
    for (int x = begin; x < end; ++x) {
        T v = row[x];
        if constexpr (Update) {
            v = v - pv * psf[x - begin];
            row[x] = v;
        }
        if constexpr (Masked) {
            if (mask[x] == 0) continue;
        }
        fold_pixel(v, x, sum, peak, peak_x);
    }
}

// The segment [begin, end) of a short row: subtracted inside the footprint
// [x1, x2) (psf: the PSF sample of column x1), with the row's kind.
template <typename T, bool Masked>
inline void short_segment(T* row, const T* psf, const std::uint8_t* mask, T pv, int x1, int x2,
                          int begin, int end, double& sum, T& peak, int& peak_x) {
    const int u0 = max_int(begin, x1), u1 = min_int(end, x2);
    if (u0 >= u1) {
        short_span<T, false, Masked>(row, nullptr, mask, pv, begin, end, sum, peak, peak_x);
        return;
    }
    short_span<T, false, Masked>(row, nullptr, mask, pv, begin, u0, sum, peak, peak_x);
    short_span<T, true, Masked>(row, psf + (u0 - x1), mask, pv, u0, u1, sum, peak, peak_x);
    short_span<T, false, Masked>(row, nullptr, mask, pv, u1, end, sum, peak, peak_x);
}

// Searched row `index` of a pass, of at most kShortRow pixels.
template <typename T>
inline void short_row(const SearchPass<T>& pass, int index, T pv) {
    const SearchRow& searched = pass.rows[index];
    const std::size_t row_offset = static_cast<std::size_t>(searched.y) * pass.nx;
    T* const row = pass.residual + row_offset;
    const Component<T>* const component = pass.component;
    int x1 = 0, x2 = 0;  // the footprint's columns in this row, none by default
    const T* psf = nullptr;
    if (component != nullptr && searched.y >= component->y1 && searched.y <= component->y2) {
        x1 = component->x1;
        x2 = component->x2 + 1;
        psf = pass.psf + (component->offset + static_cast<std::ptrdiff_t>(searched.y) * pass.psf_nx +
                          component->x1);
    }
    double sum = 0.0;
    T peak = 0;
    int peak_x = pass.scan_begin;
    for (int r = searched.run_begin; r < searched.run_end; ++r) {
        const Run run = pass.runs[r];
        if (searched.masked) {
            short_segment<T, true>(row, psf, pass.mask + row_offset, pv, x1, x2, run.begin, run.end,
                                   sum, peak, peak_x);
        } else {
            short_segment<T, false>(row, psf, nullptr, pv, x1, x2, run.begin, run.end, sum, peak,
                                    peak_x);
        }
    }
    pass.row_sum[index] = sum;
    pass.row_max[index] = peak;
    pass.row_ix[index] = peak_x;
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

// The searched rows index[0, count), count <= kRows, their chains interleaved.
template <typename T>
void search_rows(const SearchPass<T>& pass, const int* index, int count, T pv) {
    alignas(64) double squares[kRows][kChunk];
    const Component<T>* component = pass.component;
    Cursor<T> cursor[kRows];
    for (int g = 0; g < count; ++g) {
        const SearchRow& searched = pass.rows[index[g]];
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
        pass.row_sum[index[g]] = sums[g];
        pass.row_max[index[g]] = cursor[g].peak;
        pass.row_ix[index[g]] = cursor[g].peak_x;
    }
}

template <typename T>
void search(const SearchPass<T>& pass, int row_begin, int row_end) {
    const T pv = pass.component != nullptr ? pass.component->pv : static_cast<T>(0);
    int group[kRows];
    int count = 0;
    for (int i = row_begin; i < row_end; ++i) {
        if (pass.rows[i].pixels <= kShortRow) {
            short_row(pass, i, pv);
            continue;
        }
        group[count++] = i;
        if (count == kRows) {
            search_rows(pass, group, count, pv);
            count = 0;
        }
    }
    if (count > 0) search_rows(pass, group, count, pv);
}

// The deferred subtraction of components[0, count), in their order, from the
// exterior runs of rows [y_begin, y_end) (flush_rows), in two ways that make
// the same operations on every pixel:
//  * a long run (kBlockedRun pixels or more) in blocks of kBlock vectors held
//    in registers while every component passes over them (flush_block): the
//    fewest loads and stores where the footprints cover long spans of a row
//    (M: HL Tau's call at 6 threads takes 9 % longer the second way);
//  * a short run, and the tail of a long one, a chunk of at most kFlushChunk
//    pixels at a time, the components passing in their order, each over the
//    part of the chunk inside its footprint (flush_span): no block straddles a
//    footprint's edge and no pixel loops over the components alone (M: on
//    planes of at most 160 x 160 pixels, blocks took 1.25 times as long in
//    the geometric mean, and up to 7 times).
constexpr int kBlockedRun = 256;   // pixels
constexpr int kFlushChunk = 1024;  // pixels, in the first-level cache meanwhile (8 KiB of double)

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

// row[0, count) loses pv x samples[0, count), pixel by pixel: a rounded
// product, then a rounded difference.
template <typename T>
inline void subtract_span(T* __restrict row, const T* __restrict samples, T pv, int count) {
    using S = Simd<T>;
    int i = 0;
    if constexpr (S::W > 1) {
        using V = typename S::V;
        constexpr int W = S::W;
        const V vpv = S::splat(pv);
        for (; i + 2 * W <= count; i += 2 * W) {
            const V a = S::sub(S::load(row + i), S::mul(vpv, S::load(samples + i)));
            const V b = S::sub(S::load(row + i + W), S::mul(vpv, S::load(samples + i + W)));
            S::store(row + i, a);
            S::store(row + i + W, b);
        }
        if (i + W <= count) {
            S::store(row + i, S::sub(S::load(row + i), S::mul(vpv, S::load(samples + i))));
            i += W;
        }
    }
    for (; i < count; ++i) row[i] = row[i] - pv * samples[i];
}

// Columns [begin, end) of row y, chunk by chunk, each component in turn over
// the part of the chunk inside its footprint.
template <typename T>
inline void flush_span(const ExteriorFlush<T>& exterior, T* row, int y, std::ptrdiff_t row_offset,
                       int begin, int end) {
    for (int x0 = begin, x1; x0 < end; x0 = x1) {
        x1 = end - x0 > kFlushChunk ? x0 + kFlushChunk : end;
        for (int k = 0; k < exterior.count; ++k) {
            const Component<T>& c = exterior.components[k];
            if (y < c.y1 || y > c.y2) continue;
            const int lo = max_int(x0, c.x1), hi = min_int(x1, c.x2 + 1);
            if (lo < hi) {
                subtract_span(row + lo, exterior.psf + (c.offset + row_offset + lo), c.pv, hi - lo);
            }
        }
    }
}

template <typename T>
void flush_rows(const ExteriorFlush<T>& exterior, int y_begin, int y_end) {
    constexpr int kBlockPixels = kBlock * Simd<T>::W;
    if (exterior.count <= 0) return;
    for (int y = y_begin; y < y_end; ++y) {
        T* const row = exterior.residual + static_cast<std::size_t>(y) * exterior.nx;
        const std::ptrdiff_t row_offset = static_cast<std::ptrdiff_t>(y) * exterior.psf_nx;
        for (int r = exterior.row_start[y]; r < exterior.row_start[y + 1]; ++r) {
            const Run run = exterior.runs[r];
            int x = run.begin;
            if (run.end - run.begin >= kBlockedRun) {
                for (; run.end - x >= kBlockPixels; x += kBlockPixels) {
                    flush_block<T, kBlock>(exterior, row, y, row_offset, x);
                }
            }
            if (x < run.end) flush_span(exterior, row, y, row_offset, x, run.end);
        }
    }
}

// ------------------------------------------------------------------ the direct pass

// The bits of v where `keep` is all ones, +0.0 where it is 0: a pixel of a
// box row that the mask leaves out adds +0.0 to the row's sum, which leaves it
// unchanged (it is >= +0, NaN or +inf), and its |value| counts as 0, without a
// branch on the mask (a random mask would mispredict half of them).
inline double kept(double v, std::uint64_t keep) {
    return __builtin_bit_cast(double, __builtin_bit_cast(std::uint64_t, v) & keep);
}

inline float kept(float v, std::uint64_t keep) {
    const auto bits = __builtin_bit_cast(std::uint32_t, v) & static_cast<std::uint32_t>(keep);
    return __builtin_bit_cast(float, bits);
}

// All ones for a mask byte that is not 0 (searched, as the reference tests
// it), else 0.
inline std::uint64_t keep_bits(std::uint8_t byte) {
    return std::uint64_t{0} - static_cast<std::uint64_t>(byte != 0);
}

// Pixels [begin, end) of a box row of at most kShortRow pixels, in one chain:
// scan_row's sum of squares and first strict maximum of |value|.
template <typename T, bool Masked>
inline void scan_short(const T* __restrict row, const std::uint8_t* __restrict mask, int begin, int end,
                       double& sum, T& peak, int& peak_x) {
    for (int x = begin; x < end; ++x) {
        const T v = row[x];
        if constexpr (Masked) {
            const std::uint64_t keep = keep_bits(mask[x]);
            sum += kept(static_cast<double>(v) * static_cast<double>(v), keep);
            const T magnitude = kept(magnitude_of(v), keep);
            const bool above = magnitude > peak;  // false for NaN
            peak_x = above ? x : peak_x;
            peak = above ? magnitude : peak;
        } else {
            fold_pixel(v, x, sum, peak, peak_x);
        }
    }
}

template <typename T>
inline T* direct_row(const DirectPass<T>& pass, int y) {
    return pass.residual + static_cast<std::size_t>(y) * pass.nx;
}

// The mask of row y of a pass with a mask (Masked), else null.
template <typename T, bool Masked>
inline const std::uint8_t* direct_mask(const DirectPass<T>& pass, int y) {
    return Masked ? pass.mask + static_cast<std::size_t>(y) * pass.nx : nullptr;
}

// Row y of component c's footprint loses pv x PSF over the footprint's columns.
template <typename T>
inline void subtract_row(const DirectPass<T>& pass, const Component<T>& c, int y) {
    const std::ptrdiff_t at = c.offset + static_cast<std::ptrdiff_t>(y) * pass.psf_nx + c.x1;
    subtract_span(direct_row(pass, y) + c.x1, pass.psf + at, c.pv, c.x2 - c.x1 + 1);
}

// A box row's sum of squares and peak, folded into the plane's in row order:
// the first row with the largest peak wins, as in clean<T>.
template <typename T>
inline void fold_row(DirectScan<T>& scan, double sum, T peak, int peak_x, int y) {
    scan.sum_squares += sum;
    if (peak > scan.peak) {
        scan.peak = peak;
        scan.x = peak_x;
        scan.y = y;
    }
}

// The box rows of a direct pass with more than kShortRow pixels each: in
// groups of kRows rows, subtracted (in the footprint), then scanned a chunk at
// a time with lanes, their chains interleaved as search_rows interleaves them.
template <typename T, bool Masked>
void scan_long_rows(const DirectPass<T>& pass, DirectScan<T>& scan) {
    alignas(64) double squares[kRows][kChunk];
    const Component<T>* const c = pass.component;
    for (int y0 = pass.ybeg; y0 < pass.yend; y0 += kRows) {
        const int count = min_int(kRows, pass.yend - y0);
        T peak[kRows];
        int peak_x[kRows];
        for (int g = 0; g < count; ++g) {
            peak[g] = static_cast<T>(0);
            peak_x[g] = pass.xbeg;
            if (c != nullptr && y0 + g >= c->y1 && y0 + g <= c->y2) subtract_row(pass, *c, y0 + g);
        }
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
        for (int x0 = pass.xbeg; x0 < pass.xend; x0 += kChunk) {
            const int x1 = min_int(pass.xend, x0 + kChunk), n = x1 - x0;
            for (int g = 0; g < count; ++g) {
                const int y = y0 + g;
                span<T, false, Masked>(direct_row(pass, y), nullptr, direct_mask<T, Masked>(pass, y),
                                       static_cast<T>(0), x0, x1, squares[g], peak[g], peak_x[g]);
            }
            for (int g = count; g < kRows; ++g) {
                for (int i = 0; i < n; ++i) squares[g][i] = 0.0;
            }
            // kRows chains, each in its row's column order.
            for (int i = 0; i < n; ++i) {
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
        for (int g = 0; g < count; ++g) fold_row(scan, sums[g], peak[g], peak_x[g], y0 + g);
    }
}

template <typename T, bool Masked>
void direct_pass(const DirectPass<T>& pass, DirectScan<T>& scan) {
    const Component<T>* const c = pass.component;
    if (c != nullptr) {  // the footprint's rows above and below the box
        const int above = min_int(c->y2 + 1, pass.ybeg);
        for (int y = c->y1; y < above; ++y) subtract_row(pass, *c, y);
        for (int y = max_int(c->y1, pass.yend); y <= c->y2; ++y) subtract_row(pass, *c, y);
    }
    scan.sum_squares = 0.0;
    scan.peak = static_cast<T>(0);
    scan.x = pass.xbeg;
    scan.y = pass.ybeg;
    if (pass.xend - pass.xbeg > kShortRow) {
        scan_long_rows<T, Masked>(pass, scan);
        return;
    }
    for (int y = pass.ybeg; y < pass.yend; ++y) {
        if (c != nullptr && y >= c->y1 && y <= c->y2) subtract_row(pass, *c, y);
        double sum = 0.0;
        T peak = static_cast<T>(0);
        int peak_x = pass.xbeg;
        scan_short<T, Masked>(direct_row(pass, y), direct_mask<T, Masked>(pass, y), pass.xbeg, pass.xend,
                              sum, peak, peak_x);
        fold_row(scan, sum, peak, peak_x, y);
    }
}

template <typename T>
void direct(const DirectPass<T>& pass, DirectScan<T>& scan) {
    if (pass.mask != nullptr) {
        direct_pass<T, true>(pass, scan);
    } else {
        direct_pass<T, false>(pass, scan);
    }
}

// The direct pass over a plane held column by column (ColumnPass): the
// footprint subtracted a column at a time, then the box rows scanned W at a
// time, lane g holding row y0 + g: its sum of squares chained in column order,
// its first strict maximum of |value| and that column, as scan_row forms them.
template <typename T, bool Masked>
void column_pass(const ColumnPass<T>& pass, DirectScan<T>& scan) {
    using S = Simd<T>;
    using V = typename S::V;
    constexpr int W = S::W;
    const Component<T>* const c = pass.component;
    if (c != nullptr) {
        for (int x = c->x1; x <= c->x2; ++x) {
            const std::ptrdiff_t at = c->offset + static_cast<std::ptrdiff_t>(x) * pass.psf_ny + c->y1;
            subtract_span(pass.columns + static_cast<std::size_t>(x) * pass.ny + c->y1, pass.psf + at, c->pv,
                          c->y2 - c->y1 + 1);
        }
    }
    scan.sum_squares = 0.0;
    scan.peak = static_cast<T>(0);
    scan.x = pass.xbeg;
    scan.y = pass.ybeg;
    for (int y0 = pass.ybeg; y0 < pass.yend; y0 += W) {
        typename S::Sums sums = S::no_sums();
        V peak = S::zero();
        V first = S::splat(static_cast<T>(pass.xbeg));
        for (int x = pass.xbeg; x < pass.xend; ++x) {
            const std::size_t i = static_cast<std::size_t>(x) * pass.ny + static_cast<std::size_t>(y0);
            const V v = S::load(pass.columns + i);
            V magnitude = S::magnitude(v);
            if constexpr (Masked) {
                const auto keep = S::keep(pass.mask + i);
                S::add_squares(sums, v, keep);
                magnitude = S::clear(magnitude, keep);
            } else {
                S::add_squares(sums, v);
            }
            const auto above = S::greater(magnitude, peak);  // false for NaN
            peak = S::select(peak, magnitude, above);
            first = S::select(first, S::splat(static_cast<T>(x)), above);  // exact: x < 2^24
        }
        alignas(64) double sum[W];
        alignas(64) T largest[W], at[W];
        S::store_sums(sum, sums);
        S::store(largest, peak);
        S::store(at, first);
        const int rows = min_int(W, pass.yend - y0);
        for (int g = 0; g < rows; ++g) fold_row(scan, sum[g], largest[g], static_cast<int>(at[g]), y0 + g);
    }
}

template <typename T>
void columns(const ColumnPass<T>& pass, DirectScan<T>& scan) {
    if (pass.mask != nullptr) {
        column_pass<T, true>(pass, scan);
    } else {
        column_pass<T, false>(pass, scan);
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

void direct_f64(const DirectPass<double>& pass, DirectScan<double>& scan) {
    direct<double>(pass, scan);
}

void direct_f32(const DirectPass<float>& pass, DirectScan<float>& scan) {
    direct<float>(pass, scan);
}

void columns_f64(const ColumnPass<double>& pass, DirectScan<double>& scan) {
    columns<double>(pass, scan);
}

void columns_f32(const ColumnPass<float>& pass, DirectScan<float>& scan) {
    columns<float>(pass, scan);
}
