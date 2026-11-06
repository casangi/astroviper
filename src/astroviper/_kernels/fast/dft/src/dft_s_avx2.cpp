// Path S routines for AVX2 (dft_s_kernels.hpp), compiled with -mavx2 and
// without LTO, and used only when the CPU has AVX2. -mavx2 does not enable FMA,
// so the shared code is the same rounded operations as the generic build, and
// the micro-kernel below multiplies and adds separately like the generic one.
// Everything is internal to this unit except the table it exports.

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <cmath>

#include "dft_s_kernels.hpp"

namespace {

#include "dft_s_kernels.inc"

using dft_fast::kTileRows;

// The tile in 2 x kTileRows registers of 4 samples; b is 32-byte aligned.
void multiply(i64 depth, const double* __restrict a, const double* __restrict b,
              double* __restrict c, i64 ldc, bool accumulate) {
    __m256d low[kTileRows], high[kTileRows];
#pragma GCC unroll 6
    for (int r = 0; r < kTileRows; ++r) {
        low[r] = accumulate ? _mm256_loadu_pd(c + r * ldc) : _mm256_setzero_pd();
        high[r] = accumulate ? _mm256_loadu_pd(c + r * ldc + 4) : _mm256_setzero_pd();
    }
    for (i64 k = 0; k < depth; ++k) {
        const __m256d b_low = _mm256_load_pd(b + k * kTileSamples);
        const __m256d b_high = _mm256_load_pd(b + k * kTileSamples + 4);
#pragma GCC unroll 6
        for (int r = 0; r < kTileRows; ++r) {
            const __m256d a_r = _mm256_broadcast_sd(a + k * kTileRows + r);
            low[r] = _mm256_add_pd(low[r], _mm256_mul_pd(a_r, b_low));
            high[r] = _mm256_add_pd(high[r], _mm256_mul_pd(a_r, b_high));
        }
    }
#pragma GCC unroll 6
    for (int r = 0; r < kTileRows; ++r) {
        _mm256_storeu_pd(c + r * ldc, low[r]);
        _mm256_storeu_pd(c + r * ldc + 4, high[r]);
    }
}

}  // namespace

dft_fast::Kernels dft_fast::avx2_kernels() {
    return {"avx2", cos_sin_turns, column_phasors, row_phasors, add_row, multiply};
}

#endif  // defined(__x86_64__) || defined(_M_X64)
