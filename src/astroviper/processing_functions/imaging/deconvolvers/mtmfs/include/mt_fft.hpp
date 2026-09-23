#pragma once

// 2-D real <-> half-complex FFT primitives for the MTMFS deconvolver, built on
// the vendored header-only PocketFFT (BSD-3, thirdparty/).
//
// Port of CASA's casacore-free `stdcleaner/StdFFT.{h,cc}`, which reproduces the
// casacore FFTServer<Float,Complex>::fft0 conventions that
// MultiTermMatrixCleaner relies on:
//   * forward (real -> complex): negative exponent, NO scaling;
//   * backward (complex -> real): positive exponent, scaled by 1/(nx*ny),
//     so forward followed by backward is the identity.
//
// Images are row-major with logical shape (ny, nx): element (ix, iy) lives at
// index iy*nx + ix. The half-complex spectrum is row-major (ny, nx/2 + 1).
//
// None of these functions allocate: callers own every buffer, including the
// scratch needed by the in-place quadrant swap. The kernels are pure functions
// of their arguments and may be called concurrently on distinct buffers.

#include <complex>
#include <cstddef>

namespace mtmfs {

// Number of complex columns in the half-complex spectrum of an nx-wide image.
inline int rfft_width(int nx) { return nx / 2 + 1; }

// Forward real -> complex transform (fft0 convention: origin at [0,0], unscaled).
//   real : input,  row-major (ny, nx)
//   spec : output, row-major (ny, nx/2 + 1)
template <typename T>
void forward_r2c(const T* real, std::complex<T>* spec, int nx, int ny);

// Backward complex -> real transform (fft0 convention, scaled by 1/(nx*ny)).
// The input spectrum is used as in-place scratch for the column pass and is
// NOT preserved; callers pass the product spectrum they just formed.
//   spec : input/scratch, row-major (ny, nx/2 + 1)
//   real : output,        row-major (ny, nx)
template <typename T>
void backward_c2r(std::complex<T>* spec, T* real, int nx, int ny);

// In-place quadrant swap (fftshift) of a real (ny, nx) image, reproducing
// casacore FFTServer::flip(toZero=false) used to centre a convolution result.
// `scratch` must hold nx*ny elements.
template <typename T>
void flip_quadrants(T* real, int nx, int ny, T* scratch);

extern template void forward_r2c<float>(const float*, std::complex<float>*, int, int);
extern template void forward_r2c<double>(const double*, std::complex<double>*, int, int);
extern template void backward_c2r<float>(std::complex<float>*, float*, int, int);
extern template void backward_c2r<double>(std::complex<double>*, double*, int, int);
extern template void flip_quadrants<float>(float*, int, int, float*);
extern template void flip_quadrants<double>(double*, int, int, double*);

}  // namespace mtmfs
