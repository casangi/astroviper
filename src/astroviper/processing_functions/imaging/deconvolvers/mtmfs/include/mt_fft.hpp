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
// Callers own every input and output buffer, including the scratch needed to
// recenter a convolution; PocketFFT allocates its plan and scratch internally
// on each transform. The kernels are pure functions
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

// Remove the origin offsets introduced by centered kernels in a convolution.
// Each centered kernel adds floor(n/2) pixels on an axis of length n. Roll the
// inverse FFT by -centered_kernel_count*floor(n/2), independently on each axis.
// Use 1 for image*kernel and 2 for PSF*scale1*scale2 (the PSF stays centered).
// For one kernel this is ifftshift, not fftshift: they differ on odd axes.
// Even-axis two-kernel offsets cancel exactly; a zero shift is a no-op.
// `scratch` must hold nx*ny elements when the shift is nonzero and must not
// alias `real`. Both buffers belong to the caller.
template <typename T>
void recenter_convolution(T* real, int nx, int ny, T* scratch, int centered_kernel_count);

extern template void forward_r2c<float>(const float*, std::complex<float>*, int, int);
extern template void forward_r2c<double>(const double*, std::complex<double>*, int, int);
extern template void backward_c2r<float>(std::complex<float>*, float*, int, int);
extern template void backward_c2r<double>(std::complex<double>*, double*, int, int);
extern template void recenter_convolution<float>(float*, int, int, float*, int);
extern template void recenter_convolution<double>(double*, int, int, double*, int);

}  // namespace mtmfs
