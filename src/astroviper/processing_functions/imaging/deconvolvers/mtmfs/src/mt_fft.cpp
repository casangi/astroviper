// Port of CASA stdcleaner/StdFFT.cc onto the vendored PocketFFT. See mt_fft.hpp
// for the conventions reproduced here.

#include "../include/mt_fft.hpp"

// Header-only PocketFFT, compiled single-threaded: deterministic results and
// no internal thread pool inside a kernel that may itself run on many Dask
// worker threads at once.
#define POCKETFFT_NO_MULTITHREADING
#include "../include/thirdparty/pocketfft_hdronly.h"

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
    return stride_t{static_cast<std::ptrdiff_t>(ncx) * static_cast<std::ptrdiff_t>(sizeof(std::complex<T>)),
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

    // Real -> complex along the fast (nx) axis, then complex -> complex along
    // the slow (ny) axis. Forward, unscaled (fft0 convention).
    pocketfft::r2c(shape_t{sny, snx}, sr, sc, /*axis=*/1, /*forward=*/true, real, spec, static_cast<T>(1));
    pocketfft::c2c(shape_t{sny, sncx}, sc, sc, shape_t{0}, /*forward=*/true, spec, spec, static_cast<T>(1));
}

template <typename T>
void backward_c2r(std::complex<T>* spec, T* real, int nx, int ny) {
    const std::size_t sny = static_cast<std::size_t>(ny);
    const std::size_t snx = static_cast<std::size_t>(nx);
    const std::size_t sncx = static_cast<std::size_t>(rfft_width(nx));
    const stride_t sr = real_strides<T>(nx);
    const stride_t sc = spec_strides<T>(rfft_width(nx));

    // Column pass in place on the caller's spectrum, then the c2r row pass with
    // the total 1/N scaling.
    pocketfft::c2c(shape_t{sny, sncx}, sc, sc, shape_t{0}, /*forward=*/false, spec, spec, static_cast<T>(1));
    pocketfft::c2r(shape_t{sny, snx}, sc, sr, /*axis=*/1, /*forward=*/false, spec, real,
                   static_cast<T>(1) / static_cast<T>(static_cast<std::size_t>(nx) * ny));
}

template <typename T>
void flip_quadrants(T* real, int nx, int ny, T* scratch) {
    // fftshift: move [0,0] to the centre [ny/2, nx/2] (casacore flip, toZero=false).
    const int hx = nx / 2;
    const int hy = ny / 2;
    for (int iy = 0; iy < ny; ++iy) {
        const int sy = (iy + hy) % ny;
        for (int ix = 0; ix < nx; ++ix) {
            const int sx = (ix + hx) % nx;
            scratch[static_cast<std::size_t>(sy) * nx + sx] = real[static_cast<std::size_t>(iy) * nx + ix];
        }
    }
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    for (std::size_t i = 0; i < n; ++i) real[i] = scratch[i];
}

template void forward_r2c<float>(const float*, std::complex<float>*, int, int);
template void forward_r2c<double>(const double*, std::complex<double>*, int, int);
template void backward_c2r<float>(std::complex<float>*, float*, int, int);
template void backward_c2r<double>(std::complex<double>*, double*, int, int);
template void flip_quadrants<float>(float*, int, int, float*);
template void flip_quadrants<double>(double*, int, int, double*);

}  // namespace mtmfs
