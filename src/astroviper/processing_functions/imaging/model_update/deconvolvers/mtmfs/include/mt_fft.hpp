#pragma once

// FFT helpers of the MTMFS deconvolver. Its transforms are those of
// processing_functions/include/fft.hpp, whose conventions are the casacore
// FFTServer<Float,Complex>::fft0 conventions that MultiTermMatrixCleaner
// relies on (CASA's casacore-free `stdcleaner/StdFFT.{h,cc}` reproduces them).
//
// Callers own every input and output buffer, including the scratch needed to
// recenter a convolution.

#include <cstddef>

namespace mtmfs {

// Number of complex columns in the half-complex spectrum of an nx-wide image.
inline int rfft_width(int nx) { return nx / 2 + 1; }

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

extern template void recenter_convolution<float>(float*, int, int, float*, int);
extern template void recenter_convolution<double>(double*, int, int, double*, int);

}  // namespace mtmfs
