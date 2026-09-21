// Port of CASA stdcleaner/StdImageMath.cc. The masked extremum is evaluated on
// the fly (a[k]*mask[k]) rather than through a temporary product image; the
// comparisons are identical, so the result (value, position, tie-breaking) is
// unchanged while no allocation is made.

#include "../include/mt_image_math.hpp"

#include <cmath>

namespace mtmfs {

template <typename T>
T sum_array(const T* a, std::size_t n) {
    T total = static_cast<T>(0);
    for (std::size_t i = 0; i < n; ++i) total += a[i];
    return total;
}

template <typename T>
T peak_abs_masked(const T* a, const T* mask, std::size_t n) {
    T peak = static_cast<T>(0);
    for (std::size_t i = 0; i < n; ++i) {
        const T w = std::abs(a[i] * mask[i]);
        if (w > peak) peak = w;
    }
    return peak;
}

namespace {
// Shared min/max scan matching casacore minMax(): seed from element 0, strict
// updates so ties keep the first element in storage order. `weight(k)` yields
// the k-th value of the (possibly masked) image.
template <typename T, typename F>
PeakResult<T> extremum_by_magnitude(F weight, int nx, int ny) {
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    T minv = weight(0), maxv = minv;
    std::size_t minp = 0, maxp = 0;
    for (std::size_t k = 1; k < n; ++k) {
        const T w = weight(k);
        if (w < minv) {
            minv = w;
            minp = k;
        }
        if (w > maxv) {
            maxv = w;
            maxp = k;
        }
    }
    PeakResult<T> r;
    if (std::abs(minv) > std::abs(maxv)) {
        r.value = minv;
        r.ix = static_cast<int>(minp % nx);
        r.iy = static_cast<int>(minp / nx);
    } else {
        r.value = maxv;
        r.ix = static_cast<int>(maxp % nx);
        r.iy = static_cast<int>(maxp / nx);
    }
    return r;
}
}  // namespace

template <typename T>
PeakResult<T> find_max_abs(const T* a, int nx, int ny) {
    return extremum_by_magnitude<T>([a](std::size_t k) { return a[k]; }, nx, ny);
}

template <typename T>
PeakResult<T> find_max_abs_mask(const T* a, const T* mask, int nx, int ny) {
    return extremum_by_magnitude<T>([a, mask](std::size_t k) { return a[k] * mask[k]; }, nx, ny);
}

template float sum_array<float>(const float*, std::size_t);
template double sum_array<double>(const double*, std::size_t);
template float peak_abs_masked<float>(const float*, const float*, std::size_t);
template double peak_abs_masked<double>(const double*, const double*, std::size_t);
template PeakResult<float> find_max_abs<float>(const float*, int, int);
template PeakResult<double> find_max_abs<double>(const double*, int, int);
template PeakResult<float> find_max_abs_mask<float>(const float*, const float*, int, int);
template PeakResult<double> find_max_abs_mask<double>(const double*, const double*, int, int);

}  // namespace mtmfs
