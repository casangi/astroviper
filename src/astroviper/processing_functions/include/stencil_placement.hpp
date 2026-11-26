#pragma once

// The uv scales and the stencil placement that the anti-alias gridders
// (anti_alias_grid, anti_alias_grid_uv_sampling) and the anti-alias degridder
// share. kernel_lookup.hpp is the anti-alias module's own: its include/
// directory comes first on the module's include path.

#include "kernel_lookup.hpp"
#include "scalar_buffer.hpp"

#include <cmath>

namespace anti_alias {

// The factors that turn the u and v of uvw (metres) into grid cells, per
// visibility channel:
//   uv_scale_u[i] = -(frequency_coord[i] * delta_l * m_u) / c
//   uv_scale_v[i] = -(frequency_coord[i] * delta_m * m_v) / c
inline void uv_scales(const double* frequency_coord, int n_vis_chan, double delta_l,
                      double delta_m, int m_u, int m_v, double* uv_scale_u,
                      double* uv_scale_v) {
    constexpr double c = 299792458.0;
    for (int i = 0; i < n_vis_chan; ++i) {
        uv_scale_u[i] = -(astroviper::scalar_load(frequency_coord, i) * delta_l * m_u) / c;
        uv_scale_v[i] = -(astroviper::scalar_load(frequency_coord, i) * delta_m * m_v) / c;
    }
}

// The stencil of one visibility, whose u, v, w (metres) uvw_baseline points
// to: false when the sample is skipped (u or v is NaN, or the stencil is not
// wholly inside the m_u x m_v grid); otherwise true, with the grid point
// nearest to the visibility (u_center_indx, v_center_indx) and the support
// kernel values of the stencil along u and v (conv_u, conv_v).
inline bool stencil_placement(const double* uvw_baseline, double uv_scale_u,
                              double uv_scale_v, int m_u, int m_v, const double* cgk_1D,
                              int support, int oversampling, int interpolation,
                              int& u_center_indx, int& v_center_indx, double* conv_u,
                              double* conv_v) {
    const double u = astroviper::scalar_load(uvw_baseline, 0) * uv_scale_u;
    const double v = astroviper::scalar_load(uvw_baseline, 1) * uv_scale_v;

    if (std::isnan(u) || std::isnan(v)) return false;

    const int support_center = support / 2;
    const int uv_center_u    = m_u / 2;
    const int uv_center_v    = m_v / 2;
    const double u_pos = u + uv_center_u;
    const double v_pos = v + uv_center_v;

    // Round to the nearest grid centre and bounds-check in double
    // precision *before* narrowing to int. Casting an out-of-range
    // double to int is undefined behaviour (it yields INT_MIN on
    // x86/ARM), and the `idx - support_center` guard would then
    // overflow, letting an extreme UV coordinate slip through and
    // corrupt memory (gridding) or read out of bounds (degridding).
    // Validating in double — where nothing overflows — guarantees the
    // centre lies in [support_center, m - support_center) and therefore
    // fits in int. For an in-bounds (non-negative) coordinate
    // std::floor(x + 0.5) matches the previous truncating cast exactly.
    const double u_center = std::floor(u_pos + 0.5);
    const double v_center = std::floor(v_pos + 0.5);

    if (u_center + support_center >= m_u) return false;
    if (v_center + support_center >= m_v) return false;
    if (u_center - support_center <  0)   return false;
    if (v_center - support_center <  0)   return false;

    u_center_indx = static_cast<int>(u_center);
    v_center_indx = static_cast<int>(v_center);

    stencil_kernel_values(cgk_1D, support, oversampling, interpolation,
                          u_center_indx - u_pos, conv_u);
    stencil_kernel_values(cgk_1D, support, oversampling, interpolation,
                          v_center_indx - v_pos, conv_v);
    return true;
}

}  // namespace anti_alias
