#pragma once

#include <cstdint>

namespace imaging_weighting {

// Grid per-visibility data weights onto a UV grid (nearest-pixel scatter with
// Hermitian-conjugate update), accumulating per-(channel, polarization) weight
// sums. The kernel is deliberately SERIAL so the floating-point accumulation
// order — and therefore the weight sums — are bit-reproducible across runs.
//
// grid            : (n_chan_g, n_pol_g, m_u, m_v)          GridT, updated in place
// sum_weight      : (n_chan_g, n_pol_g)                    float64, updated in place
// uvw             : (n_time, n_baseline, 3)                float64
// frequency_coord : (n_vis_chan,)                          float64
// chan_map        : (n_vis_chan,)                          int64
// data_weights    : (n_time, n_baseline, n_vis_chan, n_pol) float64
//
// Only polarization 0 is gridded (parallel-hand-equalized weights); this
// matches the Python wrapper's `n_pol < 3` contract.
template <typename GridT>
void grid_imaging_weights(
    GridT* grid,
    double* sum_weight,
    const double* uvw,
    const double* frequency_coord,
    const int64_t* chan_map,
    const double* data_weights,
    int n_chan_g, int n_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol,
    double delta_l, double delta_m, bool truncate_uv_cells);

// Sample a UV imaging-weight grid at each visibility's (u, v) and apply the
// Briggs/robust denominator briggs[0]*G + briggs[1]. Serial for the same
// bit-reproducibility reason as grid_imaging_weights.
//
// imaging_weight       : (n_time, n_baseline, n_vis_chan, n_pol) float64, written in place
// grid_imaging_weight  : (n_chan_g, n_pol_g, m_u, m_v)           GridT
// briggs_factors       : (2, n_chan_g, n_pol_g)                  float64
// uvw                  : (n_time, n_baseline, 3)                 float64
// frequency_coord      : (n_vis_chan,)                           float64
// chan_map             : (n_vis_chan,)                           int64
// pol_map              : (n_pol_out,)                            int64
// data_weight          : (n_time, n_baseline, n_vis_chan, n_pol) float64
//
// Throws std::invalid_argument (Python ValueError) on NaN Briggs factors.
template <typename GridT>
void degrid_imaging_weights(
    double* imaging_weight,
    const GridT* grid_imaging_weight,
    const double* briggs_factors,
    const double* uvw,
    const double* frequency_coord,
    const int64_t* chan_map,
    const int64_t* pol_map,
    const double* data_weight,
    int n_chan_g, int n_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol, int n_pol_out,
    double delta_l, double delta_m, bool truncate_uv_cells);

extern template void grid_imaging_weights<float>(
    float*, double*, const double*, const double*, const int64_t*,
    const double*, int, int, int, int, int, int, int, int, double, double, bool);
extern template void grid_imaging_weights<double>(
    double*, double*, const double*, const double*, const int64_t*,
    const double*, int, int, int, int, int, int, int, int, double, double, bool);

extern template void degrid_imaging_weights<float>(
    double*, const float*, const double*, const double*, const double*,
    const int64_t*, const int64_t*, const double*,
    int, int, int, int, int, int, int, int, int, double, double, bool);
extern template void degrid_imaging_weights<double>(
    double*, const double*, const double*, const double*, const double*,
    const int64_t*, const int64_t*, const double*,
    int, int, int, int, int, int, int, int, int, double, double, bool);

}  // namespace imaging_weighting
