#pragma once

#include <cstdint>

namespace imaging_weighting {

// Grid per-visibility data weights onto a UV grid, accumulating
// per-(channel, polarization) weight sums. Nearest-cell mode rounds the direct
// and conjugate coordinates and requires both cells to be in bounds. CASA
// continuum mode truncates the shifted coordinates, skips index 0 on either
// axis, and accepts the direct and conjugate cells independently. The kernel is
// deliberately SERIAL so the floating-point accumulation order — and therefore
// the weight sums — are bit-reproducible across runs.
//
// grid            : (n_chan_g, n_pol_g, m_u, m_v)          GridT, updated in place
// sum_weight      : (n_chan_g, n_pol_g)                    float64, updated in place
// uvw             : (n_time, n_baseline, 3)                float64
// frequency_coord : (n_vis_chan,)                          float64
// chan_map        : (n_vis_chan,)                          int64
// data_weights    : (n_time, n_baseline, n_vis_chan, n_pol) float64
//
// Buffers may be unaligned: dispatch uses scalar byte loads/stores without
// allocating array copies. Aligned buffers retain the typed fast path.
// Only polarization 0 is gridded (parallel-hand-equalized weights); this
// matches the Python wrapper's `n_pol < 3` contract.
template <typename GridT>
void grid_imaging_weights(
    void* grid,
    void* sum_weight,
    const void* uvw,
    const void* frequency_coord,
    const void* chan_map,
    const void* data_weights,
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
    void* imaging_weight,
    const void* grid_imaging_weight,
    const void* briggs_factors,
    const void* uvw,
    const void* frequency_coord,
    const void* chan_map,
    const void* pol_map,
    const void* data_weight,
    int n_chan_g, int n_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol, int n_pol_out,
    double delta_l, double delta_m, bool truncate_uv_cells);

extern template void grid_imaging_weights<float>(
    void*, void*, const void*, const void*, const void*,
    const void*, int, int, int, int, int, int, int, int, double, double, bool);
extern template void grid_imaging_weights<double>(
    void*, void*, const void*, const void*, const void*,
    const void*, int, int, int, int, int, int, int, int, double, double, bool);

extern template void degrid_imaging_weights<float>(
    void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*,
    int, int, int, int, int, int, int, int, int, double, double, bool);
extern template void degrid_imaging_weights<double>(
    void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*,
    int, int, int, int, int, int, int, int, int, double, double, bool);

}  // namespace imaging_weighting
