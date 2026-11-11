#pragma once

// The routines of the fast Hogbom minor cycle that are built once per
// instruction set: the fused pass over the searched pixels, the deferred
// subtraction from the other pixels (hclean_fast.hpp explains the split) and
// the direct pass over a small plane.
// src/hclean_fast_kernels.inc holds their code; src/hclean_fast_generic.cpp
// (baseline ISA, portable C++) and src/hclean_fast_avx2.cpp (AVX2, x86-64
// only) each compile it once, with lanes of their own.
//
// Every version performs the reference's rounded operations pixel by pixel,
// in its order and without FMA, so every version gives the same bits (a NaN
// result is a NaN in every version, its sign and payload unspecified).
//
// This header is included by the AVX2 unit and therefore declares only
// aggregates (some of them templates) without member functions, and plain
// functions: an AVX2 unit must define no weak symbol (no inline or template
// code is emitted in it), see cmake/AstroviperKernels.cmake.

#include <cstddef>
#include <cstdint>

namespace hclean_fast {

// Columns [begin, end) of one image row.
struct Run {
    int begin, end;
};

// A row of the searched pixels: row y, its segments runs[run_begin, run_end)
// in column order (at least one), `pixels` pixels in all. The segments are
// the runs of the mask merged across short gaps: in a `masked` row, the
// pixels of a segment that the mask leaves out are updated with the others
// but not searched.
struct SearchRow {
    int y, run_begin, run_end, pixels;
    bool masked;
};

// A subtracted component: the residual loses pv x PSF over the footprint
// [x1, x2] x [y1, y2] (inclusive, the reference's clipped one), and the PSF
// sample subtracted at pixel (y, x) of the footprint is
// psf[offset + y * psf_nx + x].
template <typename T>
struct Component {
    T pv;
    int x1, x2, y1, y2;
    std::ptrdiff_t offset;
};

// The fused pass over the searched rows [row_begin, row_end) of a plane: it
// subtracts `component` (when not null) from the pixels of their segments
// inside its footprint, then gives each row the sum of the squares of its
// searched pixels, their largest |value| (0 when none is above 0) and the
// first column holding it (scan_begin when none is above 0): what the
// reference's scan_row finds on the subtracted residual.
template <typename T>
struct SearchPass {
    T* residual;  // the plane, nx columns per row
    int nx;
    const T* psf;  // the plane's PSF, psf_nx columns per row
    int psf_nx;
    const std::uint8_t* mask;  // the plane's mask, nx columns per row (read in masked rows)
    const SearchRow* rows;
    const Run* runs;
    const Component<T>* component;
    double* row_sum;  // per searched row
    T* row_max;
    int* row_ix;
    int scan_begin;
};

// The deferred subtraction of components[0, count), in that order, from the
// exterior runs of the rows [y_begin, y_end) of a plane.
template <typename T>
struct ExteriorFlush {
    T* residual;
    int nx;
    const T* psf;
    int psf_nx;
    const Run* runs;       // the exterior runs of row y:
    const int* row_start;  // runs[row_start[y], row_start[y + 1])
    const Component<T>* components;
    int count;
};

// The pass of an iteration over a small plane, which no split defers
// (hclean_fast.hpp): it subtracts `component` (when not null) from every pixel
// of its footprint, then scans the box [xbeg, xend) x [ybeg, yend) (clean<T>'s
// box) as the reference's peak search does: the sum of the squares of the
// searched pixels (those whose mask byte is not 0, all of them without a mask),
// row by row and the rows in order, and their largest |value| at its first
// pixel in row order.
template <typename T>
struct DirectPass {
    T* residual;  // the plane, nx columns per row
    int nx;
    const T* psf;  // the plane's PSF, psf_nx columns per row
    int psf_nx;
    const std::uint8_t* mask;  // the plane's mask, nx columns per row, or null
    int xbeg, xend, ybeg, yend;
    const Component<T>* component;
};

// What a direct pass finds: the sum of the squares and the largest |value|
// (0 when no searched value is above 0) at pixel (x, y), set only when it is
// above 0.
template <typename T>
struct DirectScan {
    double sum_squares;
    T peak;
    int x, y;
};

// The direct pass over a plane held column by column: pixel (y, x) of the
// plane is columns[x * ny + y], its mask byte mask[x * ny + y] (when the mask
// is not null) and the PSF sample (row r, column s) psf[s * psf_ny + r]. The
// columns and the mask continue for 8 values and bytes after the plane's last,
// which the scan reads and ignores. It makes the operations of DirectPass: it
// subtracts `component` (when not null) from its footprint, a column at a time
// (the PSF sample subtracted at pixel (y, x) is psf[offset + x * psf_ny + y]),
// then scans the box a lane per row, each lane taking its row's columns in
// order.
template <typename T>
struct ColumnPass {
    T* columns;
    int ny;
    const T* psf;
    int psf_ny;
    const std::uint8_t* mask;
    int xbeg, xend, ybeg, yend;
    const Component<T>* component;
};

struct Kernels {
    const char* isa;  // "generic" or "avx2"
    void (*search_f64)(const SearchPass<double>& pass, int row_begin, int row_end);
    void (*search_f32)(const SearchPass<float>& pass, int row_begin, int row_end);
    void (*flush_f64)(const ExteriorFlush<double>& flush, int y_begin, int y_end);
    void (*flush_f32)(const ExteriorFlush<float>& flush, int y_begin, int y_end);
    void (*direct_f64)(const DirectPass<double>& pass, DirectScan<double>& scan);
    void (*direct_f32)(const DirectPass<float>& pass, DirectScan<float>& scan);
    void (*columns_f64)(const ColumnPass<double>& pass, DirectScan<double>& scan);
    void (*columns_f32)(const ColumnPass<float>& pass, DirectScan<float>& scan);
};

// The AVX2 routines exist where the build adds the AVX2 unit (AV_HAVE_AVX2)
// and the target is x86-64, the condition of that unit's body.
#if defined(AV_HAVE_AVX2) && (defined(__x86_64__) || defined(_M_X64))
#define HCLEAN_FAST_AVX2 1
#endif

Kernels generic_kernels();
#if defined(HCLEAN_FAST_AVX2)
Kernels avx2_kernels();
#endif

}  // namespace hclean_fast
