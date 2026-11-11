#pragma once

// The routines of path S that are built once per instruction set: the
// phasors (the reference's cos_sin_of_turns, of the exact phases u l and v m
// reduced by whole turns and rounded once), the micro-kernel of the GEMM and
// the sums over the model rows. src/dft_s_kernels.inc holds the code they
// share; src/dft_s_generic.cpp (baseline ISA) and src/dft_s_avx2.cpp (AVX2,
// x86-64 only) each compile it once, with their own micro-kernel.
//
// Every version performs the same rounded operations element by element, in
// the same order and without FMA, so every version gives the same bits.
//
// This header is included by the AVX2 unit and therefore declares only plain
// types and functions: an AVX2 unit must define no weak symbol (no inline or
// template code is instantiated in it), see cmake/AstroviperKernels.cmake.

#include <cstdint>

namespace dft_fast {

using i64 = std::int64_t;

// The micro-kernel's tile: kTileRows packed model rows by kTileSamples samples.
constexpr int kTileRows = 6;
constexpr int kTileSamples = 8;

struct Kernels {
    const char* isa;  // "generic" or "avx2"

    // cos(2 pi t) and sin(2 pi t) of count angles t in turns (the reference's
    // cos_sin_of_turns, which the phasors below evaluate).
    void (*cos_sin_turns)(const double* turns, i64 count, double* cos_values,
                          double* sin_values);

    // The phasors of the columns for a block of count samples, as micro-panels
    // of kTileSamples samples: for panel p, column f and lane s,
    //     cos_panels[(p * columns + f) * kTileSamples + s] = cos(2 pi v[k] m[f])
    // with k = p * kTileSamples + s, and sin_panels likewise; v[k] is taken as
    // 0 for k >= count (the lanes past the block's end).
    void (*column_phasors)(const double* v, i64 count, const double* m, i64 columns,
                           double* cos_panels, double* sin_panels);

    // cos(2 pi u[k] l) and sin(2 pi u[k] l) for k < count.
    void (*row_phasors)(const double* u, i64 count, double l, double* cos_values,
                        double* sin_values);

    // x += (cos + i sin) (t_re + i t_im), element by element, for count elements.
    void (*add_row)(const double* cos_values, const double* sin_values, const double* t_re,
                    const double* t_im, i64 count, double* x_re, double* x_im);

    // The kTileRows x kTileSamples tile c (row stride ldc) set to
    //     (accumulate ? c : 0) + sum_k a[k * kTileRows + r] b[k * kTileSamples + s]
    // over k = 0 .. depth - 1 in that order, each step a rounded product and a
    // rounded sum. b is 32-byte aligned.
    void (*multiply)(i64 depth, const double* a, const double* b, double* c, i64 ldc,
                     bool accumulate);
};

Kernels generic_kernels();
#if defined(AV_HAVE_AVX2)
Kernels avx2_kernels();
#endif

}  // namespace dft_fast
