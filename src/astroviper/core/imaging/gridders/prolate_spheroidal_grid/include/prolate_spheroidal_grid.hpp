#pragma once

#include <complex>

namespace prolate_spheroidal {

/**
 * Grid weighted visibilities onto a UV plane using a prolate spheroidal convolution kernel.
 *
 * Accumulates weighted, convolved visibility samples into grid and the
 * corresponding sum of convolution weights into normalization. Both arrays
 * are modified in place.
 *
 * @param grid             Complex UV grid accumulator, shape (m_time, m_chan, m_pol, m_u, m_v)
 * @param normalization    Per-cell weight accumulator, shape (m_time, m_chan, m_pol)
 * @param vis_data         Visibility data, shape (n_time, n_baseline, n_vis_chan, n_pol)
 * @param uvw              UVW coordinates in metres, shape (n_time, n_baseline, 3)
 * @param frequency_coord  Channel frequencies in Hz, shape (n_vis_chan,)
 * @param frequency_map    Maps vis channel index to image channel index, shape (n_vis_chan,)
 * @param time_map         Maps vis time index to image time index, shape (n_time,)
 * @param pol_map          Maps vis pol index to image pol index, shape (n_pol,)
 * @param weight           Imaging weights, shape (n_time, n_baseline, n_vis_chan, n_pol)
 * @param cgk_1D           Oversampled 1-D prolate spheroidal kernel
 * @param m_time_g         Grid time dimension
 * @param m_chan_g         Grid channel dimension
 * @param m_pol_g          Grid polarization dimension
 * @param m_u              Grid U dimension (== n_uv[0])
 * @param m_v              Grid V dimension (== n_uv[1])
 * @param n_time           Number of visibility time samples
 * @param n_baseline       Number of baselines
 * @param n_vis_chan        Number of visibility channels
 * @param n_pol            Number of polarizations
 * @param delta_l          Image cell size in l (radians)
 * @param delta_m          Image cell size in m (radians)
 * @param support          Full convolution support width in grid pixels
 * @param oversampling     Oversampling factor of cgk_1D
 */
void prolate_spheroidal_grid(
    std::complex<double>* grid,
    double* normalization,
    const std::complex<double>* vis_data,
    const double* uvw,
    const double* frequency_coord,
    const int* frequency_map,
    const int* time_map,
    const int* pol_map,
    const double* weight,
    const double* cgk_1D,
    int m_time_g, int m_chan_g, int m_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol,
    double delta_l, double delta_m,
    int support, int oversampling
);

/**
 * Grid imaging weights onto a UV plane to form the UV-sampling function (PSF numerator).
 *
 * PSF variant: grids imaging weights instead of weighted visibility data.
 *
 * @param grid             Complex UV-sampling grid accumulator, shape (m_time, m_chan, m_pol, m_u, m_v)
 * @param normalization    Per-cell weight accumulator, shape (m_time, m_chan, m_pol)
 * @param uvw              UVW coordinates in metres, shape (n_time, n_baseline, 3)
 * @param frequency_coord  Channel frequencies in Hz, shape (n_vis_chan,)
 * @param frequency_map    Maps vis channel index to image channel index, shape (n_vis_chan,)
 * @param time_map         Maps vis time index to image time index, shape (n_time,)
 * @param pol_map          Maps vis pol index to image pol index, shape (n_pol,)
 * @param weight           Imaging weights, shape (n_time, n_baseline, n_vis_chan, n_pol)
 * @param cgk_1D           Oversampled 1-D prolate spheroidal kernel
 * @param m_time_g         Grid time dimension
 * @param m_chan_g         Grid channel dimension
 * @param m_pol_g          Grid polarization dimension
 * @param m_u              Grid U dimension
 * @param m_v              Grid V dimension
 * @param n_time           Number of visibility time samples
 * @param n_baseline       Number of baselines
 * @param n_vis_chan        Number of visibility channels
 * @param n_pol            Number of polarizations
 * @param delta_l          Image cell size in l (radians)
 * @param delta_m          Image cell size in m (radians)
 * @param support          Full convolution support width in grid pixels
 * @param oversampling     Oversampling factor of cgk_1D
 */
void prolate_spheroidal_grid_uv_sampling(
    std::complex<double>* grid,
    double* normalization,
    const double* uvw,
    const double* frequency_coord,
    const int* frequency_map,
    const int* time_map,
    const int* pol_map,
    const double* weight,
    const double* cgk_1D,
    int m_time_g, int m_chan_g, int m_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol,
    double delta_l, double delta_m,
    int support, int oversampling
);

} // namespace prolate_spheroidal
