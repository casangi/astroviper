#pragma once

#include <complex>
#include <cstdint>

namespace prolate_spheroidal {

/**
 * Sample (degrid) a model UV grid at visibility coordinates using a prolate
 * spheroidal convolution kernel.
 *
 * Inverse of ::prolate_spheroidal_grid. For each non-NaN visibility sample
 * whose convolution support falls fully inside the grid, writes
 *
 *     vis_data[i_time, i_baseline, i_chan, i_pol] =
 *         sum_over_support(conv * grid[a_time, a_chan, a_pol, u_indx, v_indx])
 *         / sum_over_support(conv)
 *
 * in place. Samples whose UV coordinate is NaN, whose support falls outside
 * the grid, or whose visibility value is NaN are left unchanged.
 *
 * The model grid is an image-domain array and may be single (``GridT=float``)
 * or double (``GridT=double``) precision; each grid cell is widened to
 * std::complex<double> before the accumulation, which is always performed in
 * std::complex<double>. The final value is cast to VisT on write so a complex64
 * vis_data buffer can be written in place without any intermediate copy. The
 * visibilities themselves are never down-cast by this routine.
 *
 * Thread safety: work is partitioned by (i_time, i_baseline), so each worker
 * writes to a disjoint slice of vis_data. No atomic operations are required.
 *
 * @tparam GridT Grid element type: float or double.
 * @tparam VisT  Visibility element type: std::complex<float> or std::complex<double>.
 */
template <typename GridT, typename VisT>
void prolate_spheroidal_degrid(
    const std::complex<GridT>* grid,
    VisT* vis_data,
    const double* uvw,
    const double* frequency_coord,
    const int64_t* frequency_map,
    const int64_t* time_map,
    const int64_t* pol_map,
    const double* cgk_1D,
    int m_time_g, int m_chan_g, int m_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol,
    double delta_l, double delta_m,
    int support, int oversampling,
    int num_threads
);

#define PS_DEGRID_EXTERN(GRIDT, VIST)                                          \
    extern template void prolate_spheroidal_degrid<GRIDT, VIST>(               \
        const std::complex<GRIDT>*, VIST*, const double*, const double*,       \
        const int64_t*, const int64_t*, const int64_t*, const double*,         \
        int, int, int, int, int, int, int, int, int,                           \
        double, double, int, int, int);

PS_DEGRID_EXTERN(double, std::complex<float>)
PS_DEGRID_EXTERN(double, std::complex<double>)
PS_DEGRID_EXTERN(float, std::complex<float>)
PS_DEGRID_EXTERN(float, std::complex<double>)

#undef PS_DEGRID_EXTERN

} // namespace prolate_spheroidal
