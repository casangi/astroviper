// Port of CASA stdcleaner/StdScales.cc (casacore MatrixCleaner::spheroidal /
// makeScale). Arithmetic follows the image working precision so the float64
// imaging path does not quantise its scale kernels to float32. Indexing is
// row-major pointer arithmetic.

#include "../include/mt_scales.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mtmfs {

namespace {
inline double square(double x) { return x * x; }

template <typename T>
T spheroidal_impl(T nu) {
    if (nu <= static_cast<T>(0)) return static_cast<T>(1);
    if (nu >= static_cast<T>(1)) return static_cast<T>(0);

    // Coefficients as [row][part] with part 0 for nu in [0,0.75), 1 otherwise.
    static const T p[5][2] = {{static_cast<T>(8.203343e-2), static_cast<T>(4.028559e-3)},
                              {static_cast<T>(-3.644705e-1), static_cast<T>(-3.697768e-2)},
                              {static_cast<T>(6.278660e-1), static_cast<T>(1.021332e-1)},
                              {static_cast<T>(-5.335581e-1), static_cast<T>(-1.201436e-1)},
                              {static_cast<T>(2.312756e-1), static_cast<T>(6.412774e-2)}};
    static const T q[3][2] = {{static_cast<T>(1.0000000e0), static_cast<T>(1.0000000e0)},
                              {static_cast<T>(8.212018e-1), static_cast<T>(9.599102e-1)},
                              {static_cast<T>(2.078043e-1), static_cast<T>(2.918724e-1)}};
    int part = 0;
    T nuend = static_cast<T>(0);
    if (nu >= static_cast<T>(0) && nu < static_cast<T>(0.75)) {
        part = 0;
        nuend = static_cast<T>(0.75);
    } else if (nu >= static_cast<T>(0.75) && nu <= static_cast<T>(1)) {
        part = 1;
        nuend = static_cast<T>(1);
    }

    T top = p[0][part];
    const T delnusq = static_cast<T>(std::pow(static_cast<double>(nu), 2.0) -
                                     std::pow(static_cast<double>(nuend), 2.0));
    for (int k = 1; k < 5; ++k) top += p[k][part] * std::pow(delnusq, static_cast<T>(k));
    T bot = q[0][part];
    for (int k = 1; k < 3; ++k) bot += q[k][part] * std::pow(delnusq, static_cast<T>(k));

    return (bot != static_cast<T>(0)) ? (top / bot) : static_cast<T>(0);
}
}  // namespace

float spheroidal(float nu) { return spheroidal_impl<float>(nu); }

template <typename T>
void make_scale(T* scale, int nx, int ny, float scale_size) {
    const std::size_t nimg = static_cast<std::size_t>(nx) * ny;
    for (std::size_t k = 0; k < nimg; ++k) scale[k] = static_cast<T>(0);

    const double refi = nx / 2;  // integer division, as in casacore
    const double refj = ny / 2;

    if (scale_size == 0.0f) {
        scale[static_cast<std::size_t>(ny / 2) * nx + (nx / 2)] = static_cast<T>(1);
        return;
    }

    const int mini = std::max(0, static_cast<int>(refi - scale_size));
    const int maxi = std::min(nx - 1, static_cast<int>(refi + scale_size));
    const int minj = std::max(0, static_cast<int>(refj - scale_size));
    const int maxj = std::min(ny - 1, static_cast<int>(refj + scale_size));

    T volume = static_cast<T>(0);
    for (int j = minj; j <= maxj; ++j) {
        const T ypart = static_cast<T>(square((refj - double(j)) / scale_size));
        for (int i = mini; i <= maxi; ++i) {
            const T rad2 = static_cast<T>(ypart + square((refi - double(i)) / scale_size));
            T& px = scale[static_cast<std::size_t>(j) * nx + i];
            if (rad2 < static_cast<T>(1)) {
                const T rad = (rad2 <= static_cast<T>(0)) ? static_cast<T>(0) : std::sqrt(rad2);
                // Retain the original float-path promotion while allowing the
                // double path to keep its full working precision.
                const T v = static_cast<T>((1.0 - static_cast<double>(rad2)) *
                                           static_cast<double>(spheroidal_impl<T>(rad)));
                px = v;
                volume += v;
            } else {
                px = static_cast<T>(0);
            }
        }
    }
    for (std::size_t k = 0; k < nimg; ++k) scale[k] /= volume;
}

template void make_scale<float>(float*, int, int, float);
template void make_scale<double>(double*, int, int, float);

}  // namespace mtmfs
