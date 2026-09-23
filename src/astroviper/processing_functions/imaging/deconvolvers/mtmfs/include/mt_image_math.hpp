#pragma once

// Small image reductions shared by the MTMFS kernel and its Python binding.
//
// Port of CASA's casacore-free `stdcleaner/StdImageMath.{h,cc}`, which replaces
// the casacore ArrayMath sum()/max(abs()) helpers and MatrixCleaner's
// findMaxAbs()/findMaxAbsMask(). All functions operate on flat, contiguous
// row-major (ny, nx) storage, never allocate, and hold no state.

#include <cstddef>

namespace mtmfs {

// Sum of all n elements of `a` (casacore sum(Array): accumulate in storage
// order with an accumulator of the array's own type).
template <typename T>
T sum_array(const T* a, std::size_t n);

// max over i of |a[i]|.
template <typename T>
T peak_abs(const T* a, std::size_t n);

// max over i of |a[i] * mask[i]|  ==  casacore max(abs(residual * mask)).
template <typename T>
T peak_abs_masked(const T* a, const T* mask, std::size_t n);

// Signed value and (ix, iy) position of the largest-magnitude element,
// matching casacore MatrixCleaner::findMaxAbs()/findMaxAbsMask(). Ties resolve
// to the first element in storage order (as casacore minMax()/minMaxMasked()).
template <typename T>
struct PeakResult {
    T value = static_cast<T>(0);  // signed value at the extremum
    int ix = 0;
    int iy = 0;
};

// Extremum (by magnitude) of a.
template <typename T>
PeakResult<T> find_max_abs(const T* a, int nx, int ny);

// Extremum (by magnitude) of the weighted array a*mask (value is that of a*mask).
// The product is evaluated on the fly; no temporary image is built.
template <typename T>
PeakResult<T> find_max_abs_mask(const T* a, const T* mask, int nx, int ny);

extern template float sum_array<float>(const float*, std::size_t);
extern template double sum_array<double>(const double*, std::size_t);
extern template float peak_abs<float>(const float*, std::size_t);
extern template double peak_abs<double>(const double*, std::size_t);
extern template float peak_abs_masked<float>(const float*, const float*, std::size_t);
extern template double peak_abs_masked<double>(const double*, const double*, std::size_t);
extern template PeakResult<float> find_max_abs<float>(const float*, int, int);
extern template PeakResult<double> find_max_abs<double>(const double*, int, int);
extern template PeakResult<float> find_max_abs_mask<float>(const float*, const float*, int, int);
extern template PeakResult<double> find_max_abs_mask<double>(const double*, const double*, int, int);

}  // namespace mtmfs
