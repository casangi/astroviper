// Port of CASA stdcleaner/StdFFT.cc onto the vendored PocketFFT. See mt_fft.hpp
// for the conventions reproduced here.

#include "../include/mt_fft.hpp"

// Header-only PocketFFT, single-threaded: deterministic results and no extra
// thread pool inside a kernel that may itself be run from many Dask workers.
#define POCKETFFT_NO_MULTITHREADING
#include "../include/thirdparty/pocketfft_hdronly.h"

#include <vector>

namespace mtmfs {

namespace {
using pocketfft::shape_t;
using pocketfft::stride_t;

template <typename T>
inline stride_t real_strides(int nx) {
    return stride_t{static_cast<std::ptrdiff_t>(nx) * static_cast<std::ptrdiff_t>(sizeof(T)),
                    static_cast<std::ptrdiff_t>(sizeof(T))};
}

template <typename T>
inline stride_t spec_strides(int ncx) {
    return stride_t{
        static_cast<std::ptrdiff_t>(ncx) * static_cast<std::ptrdiff_t>(sizeof(std::complex<T>)),
        static_cast<std::ptrdiff_t>(sizeof(std::complex<T>))};
}
}  // namespace

template <typename T>
void forward_r2c(const T* real, std::complex<T>* spec, int nx, int ny) {
    const std::size_t sny = static_cast<std::size_t>(ny);
    const std::size_t snx = static_cast<std::size_t>(nx);
    const std::size_t sncx = static_cast<std::size_t>(rfft_width(nx));
    const stride_t sr = real_strides<T>(nx);
    const stride_t sc = spec_strides<T>(rfft_width(nx));

    // Real->complex along the fast (nx) axis, then complex->complex along the
    // slow (ny) axis. Forward direction, no scaling (fft0 convention).
    pocketfft::r2c(shape_t{sny, snx}, sr, sc, /*axis=*/1, /*forward=*/true, real, spec,
                   static_cast<T>(1));
    pocketfft::c2c(shape_t{sny, sncx}, sc, sc, shape_t{0}, /*forward=*/true, spec, spec,
                   static_cast<T>(1));
}

template <typename T>
void backward_c2r(const std::complex<T>* spec, T* real, int nx, int ny) {
    const std::size_t sny = static_cast<std::size_t>(ny);
    const std::size_t snx = static_cast<std::size_t>(nx);
    const std::size_t sncx = static_cast<std::size_t>(rfft_width(nx));
    const stride_t sr = real_strides<T>(nx);
    const stride_t sc = spec_strides<T>(rfft_width(nx));

    // Work on a copy so the caller's spectrum (a cached scale/PSF transform)
    // is preserved. Backward direction; total 1/N scaling on the c2r step.
    std::vector<std::complex<T>> tmp(spec, spec + sny * sncx);
    pocketfft::c2c(shape_t{sny, sncx}, sc, sc, shape_t{0}, /*forward=*/false, tmp.data(),
                   tmp.data(), static_cast<T>(1));
    pocketfft::c2r(shape_t{sny, snx}, sc, sr, /*axis=*/1, /*forward=*/false, tmp.data(), real,
                   static_cast<T>(1) / static_cast<T>(static_cast<std::size_t>(nx) * ny));
}

template <typename T>
void flip_quadrants(T* real, int nx, int ny) {
    // fftshift: move [0,0] to the centre [ny/2, nx/2] (casacore flip, toZero=false).
    const int hx = nx / 2;
    const int hy = ny / 2;
    std::vector<T> tmp(static_cast<std::size_t>(nx) * ny);
    for (int iy = 0; iy < ny; ++iy) {
        const int sy = (iy + hy) % ny;
        for (int ix = 0; ix < nx; ++ix) {
            const int sx = (ix + hx) % nx;
            tmp[static_cast<std::size_t>(sy) * nx + sx] = real[static_cast<std::size_t>(iy) * nx + ix];
        }
    }
    for (std::size_t i = 0; i < tmp.size(); ++i) real[i] = tmp[i];
}

template void forward_r2c<float>(const float*, std::complex<float>*, int, int);
template void forward_r2c<double>(const double*, std::complex<double>*, int, int);
template void backward_c2r<float>(const std::complex<float>*, float*, int, int);
template void backward_c2r<double>(const std::complex<double>*, double*, int, int);
template void flip_quadrants<float>(float*, int, int);
template void flip_quadrants<double>(double*, int, int);

}  // namespace mtmfs
