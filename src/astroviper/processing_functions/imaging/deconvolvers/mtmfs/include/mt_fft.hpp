#pragma once

// 2-D real/complex FFT primitives for the multi-term (MTMFS) deconvolver,
// built on the vendored header-only PocketFFT (BSD-3, thirdparty/).
//
// Port of CASA's casacore-free `stdcleaner/StdFFT.{h,cc}`, which reproduces the
// exact conventions of casacore FFTServer<Float,Complex>::fft0 that the
// MultiTermMatrixCleaner relies on:
//   * forward (real -> complex): exponent sign negative, NO output scaling;
//   * backward (complex -> real): exponent sign positive, scaled by 1/(nx*ny),
//     so forward_r2c followed by backward_c2r is the identity.
// Images are row-major with logical shape (ny, nx) (element (ix, iy) lives at
// index iy*nx + ix); the half-complex spectrum is row-major (ny, nx/2 + 1).
//
// Templated on the image value type so the engine can run on float32 or
// float64 buffers without any copy/conversion at the Python boundary.

#include <complex>
#include <cstddef>

namespace mtmfs {

// Number of complex columns in the half-complex spectrum of an nx-wide image.
inline int rfft_width(int nx) { return nx / 2 + 1; }

// Forward real->complex transform (fft0 convention: origin at [0,0], no scaling).
//   real : input,  row-major (ny, nx)
//   spec : output, row-major (ny, nx/2 + 1)
template <typename T>
void forward_r2c(const T* real, std::complex<T>* spec, int nx, int ny);

// Backward complex->real transform (fft0 convention: origin at [0,0], scaled by
// 1/(nx*ny)). The input spectrum is left untouched (it is usually a cached
// PSF/scale transform), so an internal spectrum-sized scratch copy is made.
//   spec : input,  row-major (ny, nx/2 + 1)
//   real : output, row-major (ny, nx)
template <typename T>
void backward_c2r(const std::complex<T>* spec, T* real, int nx, int ny);

// In-place quadrant swap (fftshift) of a real (ny, nx) image, reproducing
// casacore FFTServer::flip used to centre a convolution result.
template <typename T>
void flip_quadrants(T* real, int nx, int ny);

extern template void forward_r2c<float>(const float*, std::complex<float>*, int, int);
extern template void forward_r2c<double>(const double*, std::complex<double>*, int, int);
extern template void backward_c2r<float>(const std::complex<float>*, float*, int, int);
extern template void backward_c2r<double>(const std::complex<double>*, double*, int, int);
extern template void flip_quadrants<float>(float*, int, int);
extern template void flip_quadrants<double>(double*, int, int);

}  // namespace mtmfs
