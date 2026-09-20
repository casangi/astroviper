#pragma once

// FFT primitives of the Asp (Adaptive Scale Pixel) deconvolver.
//
// The transforms are computed by pocketfft (include/pocketfft_hdronly.h, the
// header-only C++ FFT that powers NumPy and SciPy; BSD-3-Clause, vendored
// verbatim with its licence text). It handles arbitrary lengths (mixed radix
// plus Bluestein for large primes) in double precision and is roughly ten
// times faster than the textbook radix-2 / Bluestein code it replaced, which
// mattered because the deconvolver's cost is almost entirely FFTs.
//
// Conventions (what the Asp algorithm relies on):
//   forward : F[k] = sum_n f[n] exp(-2*pi*i*k*n/N)          (unnormalized)
//   inverse : f[n] = (1/N) sum_k F[k] exp(+2*pi*i*k*n/N)    (normalized)
// so a forward followed by an inverse is the identity, and
// ifft2(fft2(a) * fft2(b)) is the standard circular convolution.
//
// Images are stored row-major as data[i + j*nx], i in [0, nx) the fast ("x")
// axis and j in [0, ny); spectra use the same layout with kx fast.

#include <complex>
#include <cstddef>
#include <vector>

#ifndef POCKETFFT_CACHE_SIZE
// Cache the last few transform plans: every minor-cycle iteration transforms
// the same (nx, ny) shape many times over.
#define POCKETFFT_CACHE_SIZE 16
#endif
#include "pocketfft_hdronly.h"

namespace aspfft {

using cd = std::complex<double>;

namespace detail {

// In-place 2-D complex transform of an [ny][nx] array.
inline void c2c_2d(cd* data, int nx, int ny, bool forward) {
    const pocketfft::shape_t shape{static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(nx * sizeof(cd)),
                                     static_cast<std::ptrdiff_t>(sizeof(cd))};
    const pocketfft::shape_t axes{0, 1};
    const double fct = forward ? 1.0 : 1.0 / (static_cast<double>(nx) * static_cast<double>(ny));
    pocketfft::c2c(shape, stride, stride, axes, forward, data, data, fct, 1);
}

// In-place 1-D complex transform of a length-n vector.
inline void c2c_1d(cd* data, int n, bool forward) {
    const pocketfft::shape_t shape{static_cast<std::size_t>(n)};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(cd))};
    const pocketfft::shape_t axes{0};
    const double fct = forward ? 1.0 : 1.0 / static_cast<double>(n);
    pocketfft::c2c(shape, stride, stride, axes, forward, data, data, fct, 1);
}

}  // namespace detail

// 1-D forward transform of a real vector (unnormalized).
inline std::vector<cd> fft1d_forward(const std::vector<double>& x) {
    std::vector<cd> spec(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) spec[i] = cd(x[i], 0.0);
    if (!spec.empty()) detail::c2c_1d(spec.data(), static_cast<int>(spec.size()), true);
    return spec;
}

// 2-D forward transform of a real image, unnormalized.
inline std::vector<cd> rfft2_forward(const double* data, int nx, int ny) {
    std::vector<cd> spec(static_cast<std::size_t>(nx) * ny);
    for (std::size_t i = 0; i < spec.size(); ++i) spec[i] = cd(data[i], 0.0);
    detail::c2c_2d(spec.data(), nx, ny, true);
    return spec;
}

// 2-D forward transform of a complex image, unnormalized.
inline std::vector<cd> cfft2_forward(std::vector<cd> spec, int nx, int ny) {
    detail::c2c_2d(spec.data(), nx, ny, true);
    return spec;
}

// 2-D inverse transform returning the full complex result (normalized by
// nx*ny). Used to recover two real convolutions from one transform: if A and B
// are the spectra of real images a and b, ifft2(A + i B) = a + i b.
inline std::vector<cd> cfft2_inverse(std::vector<cd> spec, int nx, int ny) {
    detail::c2c_2d(spec.data(), nx, ny, false);
    return spec;
}

// 2-D inverse transform, real part written into `out` (normalized by nx*ny).
inline void irfft2_inverse(std::vector<cd> spec, int nx, int ny, double* out) {
    detail::c2c_2d(spec.data(), nx, ny, false);
    for (std::size_t k = 0; k < spec.size(); ++k) out[k] = spec[k].real();
}

}  // namespace aspfft
