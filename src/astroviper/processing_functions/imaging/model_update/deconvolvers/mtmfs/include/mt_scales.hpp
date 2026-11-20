#pragma once

// Multi-scale CLEAN scale-image primitives for the MTMFS deconvolver.
//
// Port of CASA's casacore-free `stdcleaner/StdScales.{h,cc}`, itself a faithful
// re-implementation of casacore synthesis::MatrixCleaner::spheroidal() and
// makeScale() (the Cornwell tapered-parabola scale images). Pure functions.

namespace mtmfs {

// Prolate-spheroidal taper used to shape a scale image (MatrixCleaner::spheroidal).
float spheroidal(float nu);

// Build the Cornwell multi-scale kernel of the given scale size (in pixels)
// into a row-major (ny, nx) image (MatrixCleaner::makeScale). scale_size == 0
// yields a unit delta at the image centre (nx/2, ny/2); otherwise a tapered
// parabola of radius scale_size centred there and normalised to unit sum.
template <typename T>
void make_scale(T* scale, int nx, int ny, float scale_size);

extern template void make_scale<float>(float*, int, int, float);
extern template void make_scale<double>(double*, int, int, float);

}  // namespace mtmfs
