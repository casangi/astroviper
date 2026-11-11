#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace anti_alias {

/**
 * Kernel interpolation of the sampled gridding kernel table.
 *
 * The table holds the half kernel, ``oversampling`` samples per uv cell from
 * the centre out to ``support / 2 + 1`` cells (the last cell zero).
 */
enum KernelInterpolation : int {
    KERNEL_NEAREST = 0,  ///< the table entry nearest to the stencil point
    KERNEL_LINEAR  = 1,  ///< linear interpolation between the two neighbours
};

/**
 * Kernel values at the ``support`` stencil points of one visibility.
 *
 * The stencil point ``i`` (``0 <= i < support``) is the grid point
 * ``i - support / 2`` cells from the grid point nearest to the visibility,
 * and ``offset`` is that grid point minus the visibility position (cells, in
 * (-0.5, 0.5]), so the stencil point lies ``|i - support / 2 + offset|``
 * cells from the visibility. With ``KERNEL_NEAREST`` the table entry nearest
 * to that distance is taken, in exactly the arithmetic the gridders always
 * used (the offset is rounded to table samples once, so the whole stencil
 * shares one rounding); with ``KERNEL_LINEAR`` the two neighbouring entries
 * are interpolated.
 *
 * @param cgk_1D         The kernel table, ``oversampling * (support / 2 + 1)`` values.
 * @param support        Grid points of the stencil along one axis (odd).
 * @param oversampling   Table samples per uv cell.
 * @param interpolation  ``KERNEL_NEAREST`` or ``KERNEL_LINEAR``.
 * @param offset         Nearest grid point minus the visibility position, cells.
 * @param values         Output, ``support`` kernel values.
 */
inline void stencil_kernel_values(
    const double* cgk_1D,
    int support, int oversampling, int interpolation,
    double offset,
    double* values
) {
    const int support_center = support / 2;
    if (interpolation == KERNEL_LINEAR) {
        const int    last   = oversampling * (support_center + 1) - 1;
        const double center = offset * oversampling;
        for (int i = 0; i < support; ++i) {
            const double pos = std::abs(oversampling * (i - support_center) + center);
            const int    k   = std::min(static_cast<int>(pos), last);
            const int    k1  = std::min(k + 1, last);
            const double w   = pos - static_cast<double>(k);
            values[i] = cgk_1D[k] * (1.0 - w) + cgk_1D[k1] * w;
        }
    } else {
        const int center_offset_indx =
            static_cast<int>(std::floor(offset * oversampling + 0.5));
        for (int i = 0; i < support; ++i) {
            values[i] = cgk_1D[std::abs(
                oversampling * (i - support_center) + center_offset_indx)];
        }
    }
}

} // namespace anti_alias
