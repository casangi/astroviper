#pragma once

// FFTs of the C++ kernels, in float and double precision, computed by
// pocketfft (pocketfft_hdronly.h beside this header, the header-only C++ FFT
// that powers NumPy and SciPy; BSD-3-Clause, vendored verbatim with its
// licence text). It handles arbitrary lengths (mixed radix plus Bluestein for
// large primes).
//
// Conventions:
//   forward : F[k] = sum_n f[n] exp(-2*pi*i*k*n/N)          (unnormalized)
//   inverse : f[n] = (1/N) sum_k F[k] exp(+2*pi*i*k*n/N)    (normalized)
// so a forward followed by an inverse is the identity, and
// ifft2(fft2(a) * fft2(b)) is the standard circular convolution.
//
// Images are stored row-major as data[i + j*nx], i in [0, nx) the fast ("x")
// axis and j in [0, ny); spectra use the same layout with kx fast. The
// spectrum of a real image is kept as its half spectrum (kx in [0, nx/2], so
// rows of nx/2 + 1 values); the other half is its complex conjugate.
//
// Every transform takes the number of threads pocketfft may use; a count
// below 1 runs one thread. pocketfft gives each thread whole lines of a
// transform, so the results do not depend on the count.
//
// pocketfft has one configuration for all kernels, set in
// cmake/AstroviperPybind.cmake (the one-definition rule): a plan cache of 16
// and threads compiled in. This header refuses a build without threads or
// with another cache size.
#if defined(POCKETFFT_NO_MULTITHREADING) || !defined(POCKETFFT_CACHE_SIZE) || POCKETFFT_CACHE_SIZE != 16
#error "pocketfft is configured in cmake/AstroviperPybind.cmake only: POCKETFFT_CACHE_SIZE=16, threads on"
#endif

#include "scalar_buffer.hpp"
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <vector>

#include "pocketfft_hdronly.h"

namespace astroviper::fft {

namespace detail {

inline std::size_t pocketfft_threads(int threads) {
    return static_cast<std::size_t>(std::max(1, threads));
}

// In-place 2-D complex transform of an [ny][nx] array.
template <typename T>
inline void c2c_2d(std::complex<T>* data, int nx, int ny, bool forward, int threads) {
    const pocketfft::shape_t shape{static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(nx * sizeof(std::complex<T>)),
                                     static_cast<std::ptrdiff_t>(sizeof(std::complex<T>))};
    const pocketfft::shape_t axes{0, 1};
    const T fct = forward ? T(1) : T(1) / (static_cast<T>(nx) * static_cast<T>(ny));
    pocketfft::c2c(shape, stride, stride, axes, forward, data, data, fct, pocketfft_threads(threads));
}

// In-place 1-D complex transform of a length-n vector.
template <typename T>
inline void c2c_1d(std::complex<T>* data, int n, bool forward, int threads) {
    const pocketfft::shape_t shape{static_cast<std::size_t>(n)};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(std::complex<T>))};
    const pocketfft::shape_t axes{0};
    const T fct = forward ? T(1) : T(1) / static_cast<T>(n);
    pocketfft::c2c(shape, stride, stride, axes, forward, data, data, fct, pocketfft_threads(threads));
}

}  // namespace detail

// 1-D forward transform of a real vector (unnormalized).
template <typename T>
inline std::vector<std::complex<T>> fft1d_forward(const std::vector<T>& x, int threads) {
    std::vector<std::complex<T>> spec(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) spec[i] = std::complex<T>(x[i], T(0));
    if (!spec.empty()) detail::c2c_1d(spec.data(), static_cast<int>(spec.size()), true, threads);
    return spec;
}

// 2-D forward transform of a real image, unnormalized.
template <typename T>
inline std::vector<std::complex<T>> rfft2_forward(const T* data, int nx, int ny, int threads) {
    std::vector<std::complex<T>> spec(static_cast<std::size_t>(nx) * ny);
    for (std::size_t i = 0; i < spec.size(); ++i) spec[i] = std::complex<T>(astroviper::scalar_load(data, i), T(0));
    detail::c2c_2d(spec.data(), nx, ny, true, threads);
    return spec;
}

// 2-D forward transform of a complex image, unnormalized.
template <typename T>
inline std::vector<std::complex<T>> cfft2_forward(std::vector<std::complex<T>> spec, int nx, int ny, int threads) {
    detail::c2c_2d(spec.data(), nx, ny, true, threads);
    return spec;
}

// 2-D inverse transform returning the full complex result (normalized by
// nx*ny). Used to recover two real convolutions from one transform: if A and B
// are the spectra of real images a and b, ifft2(A + i B) = a + i b.
template <typename T>
inline std::vector<std::complex<T>> cfft2_inverse(std::vector<std::complex<T>> spec, int nx, int ny, int threads) {
    detail::c2c_2d(spec.data(), nx, ny, false, threads);
    return spec;
}

// Half spectrum [ny][nx/2 + 1] of a real [ny][nx] image, unnormalized, into
// `spec` (resized; its storage is reused).
template <typename T>
inline void rfft2_half(const T* data, int nx, int ny, std::vector<std::complex<T>>& spec, int threads) {
    const int hx = nx / 2 + 1;
    spec.resize(static_cast<std::size_t>(hx) * ny);
    const pocketfft::shape_t shape{static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    const pocketfft::stride_t stride_in{static_cast<std::ptrdiff_t>(nx * sizeof(T)),
                                        static_cast<std::ptrdiff_t>(sizeof(T))};
    const pocketfft::stride_t stride_out{static_cast<std::ptrdiff_t>(hx * sizeof(std::complex<T>)),
                                         static_cast<std::ptrdiff_t>(sizeof(std::complex<T>))};
    const std::size_t nthreads = detail::pocketfft_threads(threads);
    if (astroviper::naturally_aligned<T>(data)) {
        pocketfft::r2c(shape, stride_in, stride_out, pocketfft::shape_t{0, 1}, pocketfft::FORWARD,
                       data, spec.data(), T(1), nthreads);
    } else {
        // Reuse the required spectrum and pack only one FFT row at a time.
        std::vector<T> row(nx);
        for (int y = 0; y < ny; ++y) {
            std::memcpy(row.data(), reinterpret_cast<const unsigned char*>(data) + static_cast<std::size_t>(y) * nx * sizeof(T), nx * sizeof(T));
            pocketfft::r2c(pocketfft::shape_t{static_cast<std::size_t>(nx)}, pocketfft::stride_t{sizeof(T)},
                           pocketfft::stride_t{sizeof(std::complex<T>)}, std::size_t{0}, pocketfft::FORWARD,
                           row.data(), spec.data() + static_cast<std::size_t>(y) * hx, T(1), nthreads);
        }
        pocketfft::c2c(pocketfft::shape_t{static_cast<std::size_t>(ny), static_cast<std::size_t>(hx)},
                       stride_out, stride_out, pocketfft::shape_t{0}, pocketfft::FORWARD,
                       spec.data(), spec.data(), T(1), nthreads);
    }
}

template <typename T>
inline std::vector<std::complex<T>> rfft2_half(const T* data, int nx, int ny, int threads) {
    std::vector<std::complex<T>> spec;
    rfft2_half(data, nx, ny, spec, threads);
    return spec;
}

// Real [ny][nx] image of a half spectrum [ny][nx/2 + 1], normalized by nx*ny.
// The spectrum is overwritten (the transform along y runs in place on it),
// so no temporary of the spectrum's size is allocated, as pocketfft's own
// multi-axis c2r would do on every call.
template <typename T>
inline void irfft2_half(std::complex<T>* spec, int nx, int ny, T* out, int threads) {
    const int hx = nx / 2 + 1;
    const pocketfft::shape_t half{static_cast<std::size_t>(ny), static_cast<std::size_t>(hx)};
    const pocketfft::stride_t stride_half{static_cast<std::ptrdiff_t>(hx * sizeof(std::complex<T>)),
                                          static_cast<std::ptrdiff_t>(sizeof(std::complex<T>))};
    const std::size_t nthreads = detail::pocketfft_threads(threads);
    pocketfft::c2c(half, stride_half, stride_half, pocketfft::shape_t{0}, pocketfft::BACKWARD,
                   spec, spec, T(1), nthreads);
    const pocketfft::shape_t shape{static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    const pocketfft::stride_t stride_out{static_cast<std::ptrdiff_t>(nx * sizeof(T)),
                                         static_cast<std::ptrdiff_t>(sizeof(T))};
    const T fct = T(1) / (static_cast<T>(nx) * static_cast<T>(ny));
    pocketfft::c2r(shape, stride_half, stride_out, std::size_t{1}, pocketfft::BACKWARD, spec, out,
                   fct, nthreads);
}

// Half spectrum (n/2 + 1 values) of a real vector, unnormalized.
template <typename T>
inline std::vector<std::complex<T>> rfft1d_half(const std::vector<T>& x, int threads) {
    std::vector<std::complex<T>> spec(x.size() / 2 + 1);
    if (x.empty()) return spec;
    const pocketfft::shape_t shape{x.size()};
    const pocketfft::stride_t stride_in{static_cast<std::ptrdiff_t>(sizeof(T))};
    const pocketfft::stride_t stride_out{static_cast<std::ptrdiff_t>(sizeof(std::complex<T>))};
    pocketfft::r2c(shape, stride_in, stride_out, std::size_t{0}, pocketfft::FORWARD, x.data(),
                   spec.data(), T(1), detail::pocketfft_threads(threads));
    return spec;
}

// 2-D inverse transform, real part written into `out` (normalized by nx*ny).
template <typename T>
inline void irfft2_inverse(std::vector<std::complex<T>> spec, int nx, int ny, T* out, int threads) {
    detail::c2c_2d(spec.data(), nx, ny, false, threads);
    for (std::size_t k = 0; k < spec.size(); ++k) out[k] = spec[k].real();
}

}  // namespace astroviper::fft
