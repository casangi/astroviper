// Path S routines for the baseline instruction set (dft_s_kernels.hpp).

#include <cmath>

#include "dft_s_kernels.hpp"

namespace {

#include "dft_s_kernels.inc"

using dft_fast::kTileRows;

void multiply(i64 depth, const double* __restrict a, const double* __restrict b,
              double* __restrict c, i64 ldc, bool accumulate) {
    double tile[kTileRows][kTileSamples];
    for (int r = 0; r < kTileRows; ++r) {
        for (int s = 0; s < kTileSamples; ++s) tile[r][s] = accumulate ? c[r * ldc + s] : 0.0;
    }
    for (i64 k = 0; k < depth; ++k) {
        const double* b_k = b + k * kTileSamples;
        const double* a_k = a + k * kTileRows;
        for (int r = 0; r < kTileRows; ++r) {
            const double a_r = a_k[r];
            for (int s = 0; s < kTileSamples; ++s) tile[r][s] = tile[r][s] + a_r * b_k[s];
        }
    }
    for (int r = 0; r < kTileRows; ++r) {
        for (int s = 0; s < kTileSamples; ++s) c[r * ldc + s] = tile[r][s];
    }
}

}  // namespace

dft_fast::Kernels dft_fast::generic_kernels() {
    return {"generic", cos_sin_turns, column_phasors, row_phasors, add_row, multiply};
}
