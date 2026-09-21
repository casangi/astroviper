// Port of CASA stdcleaner/StdScales.cc (casacore MatrixCleaner::spheroidal /
// makeScale). The floating-point promotions (double reference coordinates,
// float image values) mirror the casacore original so results match to float
// precision. Indexing is plain row-major pointer arithmetic (no mdspan).

#include "../include/mt_scales.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mtmfs {

namespace {
inline double square(double x) { return x * x; }
}  // namespace

float spheroidal(float nu) {
    if (nu <= 0.0f) {
        return 1.0f;
    } else if (nu >= 1.0f) {
        return 0.0f;
    } else {
        // Coefficients as [row][part] with part 0 for nu in [0,0.75), 1 otherwise.
        static const float p[5][2] = {{8.203343e-2f, 4.028559e-3f},
                                      {-3.644705e-1f, -3.697768e-2f},
                                      {6.278660e-1f, 1.021332e-1f},
                                      {-5.335581e-1f, -1.201436e-1f},
                                      {2.312756e-1f, 6.412774e-2f}};
        static const float q[3][2] = {{1.0000000e0f, 1.0000000e0f},
                                      {8.212018e-1f, 9.599102e-1f},
                                      {2.078043e-1f, 2.918724e-1f}};
        int part = 0;
        float nuend = 0.0f;
        if (nu >= 0.0f && nu < 0.75f) {
            part = 0;
            nuend = 0.75f;
        } else if (nu >= 0.75f && nu <= 1.00f) {
            part = 1;
            nuend = 1.0f;
        }

        float top = p[0][part];
        const float delnusq = static_cast<float>(std::pow(static_cast<double>(nu), 2.0) -
                                                 std::pow(static_cast<double>(nuend), 2.0));
        for (int k = 1; k < 5; ++k) top += p[k][part] * std::pow(delnusq, static_cast<float>(k));
        float bot = q[0][part];
        for (int k = 1; k < 3; ++k) bot += q[k][part] * std::pow(delnusq, static_cast<float>(k));

        return (bot != 0.0f) ? (top / bot) : 0.0f;
    }
}

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

    float volume = 0.0f;
    for (int j = minj; j <= maxj; ++j) {
        const float ypart = static_cast<float>(square((refj - double(j)) / scale_size));
        for (int i = mini; i <= maxi; ++i) {
            const float rad2 = static_cast<float>(ypart + square((refi - double(i)) / scale_size));
            T& px = scale[static_cast<std::size_t>(j) * nx + i];
            if (rad2 < 1.0f) {
                const float rad = (rad2 <= 0.0f) ? 0.0f : std::sqrt(rad2);
                const float v = static_cast<float>((1.0 - rad2) * spheroidal(rad));
                px = static_cast<T>(v);
                volume += v;
            } else {
                px = static_cast<T>(0);
            }
        }
    }
    for (std::size_t k = 0; k < nimg; ++k) scale[k] /= static_cast<T>(volume);
}

template void make_scale<float>(float*, int, int, float);
template void make_scale<double>(double*, int, int, float);

}  // namespace mtmfs
