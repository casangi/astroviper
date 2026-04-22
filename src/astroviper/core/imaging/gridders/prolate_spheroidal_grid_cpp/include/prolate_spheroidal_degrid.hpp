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
 * Thread safety: work is partitioned by (i_time, i_baseline), so each worker
 * writes to a disjoint slice of vis_data. No atomic operations are required.
 *
 * @param grid             Complex UV grid (read only), shape (m_time, m_chan, m_pol, m_u, m_v)
 * @param vis_data         Visibility data buffer, shape (n_time, n_baseline, n_vis_chan, n_pol).
 *                         Modified in place: processed samples are overwritten;
 *                         NaN samples are left as-is.
 * @param uvw              UVW coordinates in metres, shape (n_time, n_baseline, 3)
 * @param frequency_coord  Channel frequencies in Hz, shape (n_vis_chan,)
 * @param frequency_map    Maps vis channel index to image channel index, shape (n_vis_chan,)
 * @param time_map         Maps vis time index to image time index, shape (n_time,)
 * @param pol_map          Maps vis pol index to image pol index, shape (n_pol,)
 * @param cgk_1D           Oversampled 1-D prolate spheroidal kernel
 * @param m_time_g         Grid time dimension
 * @param m_chan_g         Grid channel dimension
 * @param m_pol_g          Grid polarization dimension
 * @param m_u              Grid U dimension (== n_uv[0])
 * @param m_v              Grid V dimension (== n_uv[1])
 * @param n_time           Number of visibility time samples
 * @param n_baseline       Number of baselines
 * @param n_vis_chan       Number of visibility channels
 * @param n_pol            Number of polarizations
 * @param delta_l          Image cell size in l (radians)
 * @param delta_m          Image cell size in m (radians)
 * @param support          Full convolution support width in grid pixels
 * @param oversampling     Oversampling factor of cgk_1D
 * @param num_threads      Number of worker threads. Values <= 0 fall back to
 *                         std::thread::hardware_concurrency(); 1 runs serial.
 */
void prolate_spheroidal_degrid(
    const std::complex<double>* grid,
    std::complex<double>* vis_data,
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

} // namespace prolate_spheroidal
