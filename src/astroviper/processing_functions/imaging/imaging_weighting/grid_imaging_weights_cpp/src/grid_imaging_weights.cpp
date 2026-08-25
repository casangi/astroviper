#include "../include/grid_imaging_weights.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace imaging_weighting {

namespace {
constexpr double kSpeedOfLight = 299792458.0;
}  // namespace

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
    double delta_l, double delta_m
) {
    (void)n_chan_g;

    // uv_scale maps baseline meters -> UV pixels for each frequency:
    //   u_pix = uvw_u * -(freq * delta_l * m_u) / c   (likewise for v).
    const int uv_center_u = m_u / 2;
    const int uv_center_v = m_v / 2;

    // Strides for grid shape (n_chan_g, n_pol_g, m_u, m_v):
    const int grid_pol_stride  = m_u * m_v;
    const int grid_chan_stride = n_pol_g * grid_pol_stride;

    // Strides for data_weights shape (n_time, n_baseline, n_vis_chan, n_pol):
    const int wt_chan_stride = n_pol;
    const int wt_bl_stride   = n_vis_chan * wt_chan_stride;
    const int wt_time_stride = n_baseline * wt_bl_stride;

    // Strides for uvw shape (n_time, n_baseline, 3):
    const int uvw_bl_stride   = 3;
    const int uvw_time_stride = n_baseline * uvw_bl_stride;

    for (int i_time = 0; i_time < n_time; ++i_time) {
        for (int i_baseline = 0; i_baseline < n_baseline; ++i_baseline) {
            const int uvw_off = i_time * uvw_time_stride + i_baseline * uvw_bl_stride;
            const int wt_off  = i_time * wt_time_stride + i_baseline * wt_bl_stride;

            for (int i_chan = 0; i_chan < n_vis_chan; ++i_chan) {
                const int a_chan = static_cast<int>(chan_map[i_chan]);

                const double uv_scale_u =
                    -(frequency_coord[i_chan] * delta_l * m_u) / kSpeedOfLight;
                const double uv_scale_v =
                    -(frequency_coord[i_chan] * delta_m * m_v) / kSpeedOfLight;

                const double u = uvw[uvw_off + 0] * uv_scale_u;
                const double v = uvw[uvw_off + 1] * uv_scale_v;

                if (std::isnan(u) || std::isnan(v)) continue;

                const double u_pos      = u + uv_center_u;
                const double v_pos      = v + uv_center_v;
                const double u_pos_conj = -u + uv_center_u;
                const double v_pos_conj = -v + uv_center_v;

                // Round to the nearest pixel and bounds-check in double
                // precision *before* narrowing to int: casting an out-of-range
                // double to int is undefined behaviour (see the identical
                // rationale in the prolate spheroidal gridder). For in-bounds,
                // non-negative coordinates std::floor(x + 0.5) matches the
                // historical Fortran/CASA int(x + 0.5) rounding exactly.
                const double u_center      = std::floor(u_pos + 0.5);
                const double v_center      = std::floor(v_pos + 0.5);
                const double u_center_conj = std::floor(u_pos_conj + 0.5);
                const double v_center_conj = std::floor(v_pos_conj + 0.5);

                if (u_center < 0 || u_center >= m_u) continue;
                if (v_center < 0 || v_center >= m_v) continue;
                // The conjugate pixel is guarded too (the historical kernel
                // only checked the direct pixel; an out-of-bounds conjugate
                // write would corrupt memory here).
                if (u_center_conj < 0 || u_center_conj >= m_u) continue;
                if (v_center_conj < 0 || v_center_conj >= m_v) continue;

                const int u_indx      = static_cast<int>(u_center);
                const int v_indx      = static_cast<int>(v_center);
                const int u_indx_conj = static_cast<int>(u_center_conj);
                const int v_indx_conj = static_cast<int>(v_center_conj);

                // Only polarization 0 is gridded (weights are parallel-hand
                // equalized upstream).
                const double weight = data_weights[wt_off + i_chan * wt_chan_stride + 0];
                if (std::isnan(weight)) continue;

                const int cell      = a_chan * grid_chan_stride + u_indx * m_v + v_indx;
                const int cell_conj =
                    a_chan * grid_chan_stride + u_indx_conj * m_v + v_indx_conj;

                // Accumulate in double and narrow on store (a float32 grid
                // cell promotes to double when added to the float64 weight),
                // matching NumPy's mixed-precision semantics exactly.
                grid[cell] =
                    static_cast<GridT>(static_cast<double>(grid[cell]) + weight);
                grid[cell_conj] =
                    static_cast<GridT>(static_cast<double>(grid[cell_conj]) + weight);

                // Factor 2 accounts for the conjugate update.
                sum_weight[a_chan * n_pol_g + 0] += 2.0 * weight;
            }
        }
    }
}

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
    double delta_l, double delta_m
) {
    const int uv_center_u = m_u / 2;
    const int uv_center_v = m_v / 2;

    // Strides for grid shape (n_chan_g, n_pol_g, m_u, m_v):
    const int grid_pol_stride  = m_u * m_v;
    const int grid_chan_stride = n_pol_g * grid_pol_stride;

    // Strides for briggs_factors shape (2, n_chan_g, n_pol_g):
    const int briggs_plane_stride = n_chan_g * n_pol_g;

    // Strides for data_weight / imaging_weight shape
    // (n_time, n_baseline, n_vis_chan, n_pol):
    const int wt_chan_stride = n_pol;
    const int wt_bl_stride   = n_vis_chan * wt_chan_stride;
    const int wt_time_stride = n_baseline * wt_bl_stride;

    // Strides for uvw shape (n_time, n_baseline, 3):
    const int uvw_bl_stride   = 3;
    const int uvw_time_stride = n_baseline * uvw_bl_stride;

    for (int i_time = 0; i_time < n_time; ++i_time) {
        for (int i_baseline = 0; i_baseline < n_baseline; ++i_baseline) {
            const int uvw_off = i_time * uvw_time_stride + i_baseline * uvw_bl_stride;
            const int wt_off  = i_time * wt_time_stride + i_baseline * wt_bl_stride;

            for (int i_chan = 0; i_chan < n_vis_chan; ++i_chan) {
                const int a_chan = static_cast<int>(chan_map[i_chan]);

                const double uv_scale_u =
                    -(frequency_coord[i_chan] * delta_l * m_u) / kSpeedOfLight;
                const double uv_scale_v =
                    -(frequency_coord[i_chan] * delta_m * m_v) / kSpeedOfLight;

                const double u = uvw[uvw_off + 0] * uv_scale_u;
                const double v = uvw[uvw_off + 1] * uv_scale_v;

                if (std::isnan(u) || std::isnan(v)) continue;

                // Same rounding / double-precision bounds-check convention as
                // grid_imaging_weights above.
                const double u_center = std::floor(u + uv_center_u + 0.5);
                const double v_center = std::floor(v + uv_center_v + 0.5);

                if (u_center < 0 || u_center >= m_u) continue;
                if (v_center < 0 || v_center >= m_v) continue;

                const int u_center_indx = static_cast<int>(u_center);
                const int v_center_indx = static_cast<int>(v_center);

                for (int i_pol = 0; i_pol < n_pol_out; ++i_pol) {
                    const int a_pol   = static_cast<int>(pol_map[i_pol]);
                    const int wt_idx  = wt_off + i_chan * wt_chan_stride + i_pol;
                    const double natural_weight = data_weight[wt_idx];

                    // Start from the natural weight; only reweight finite,
                    // non-zero samples that land on a finite, non-zero grid cell.
                    imaging_weight[wt_idx] = natural_weight;

                    if (std::isnan(natural_weight) || natural_weight == 0.0) continue;

                    const double gij = static_cast<double>(grid_imaging_weight[
                        a_chan * grid_chan_stride + a_pol * grid_pol_stride +
                        u_center_indx * m_v + v_center_indx]);

                    if (std::isnan(gij) || gij == 0.0) continue;

                    const double briggs_a =
                        briggs_factors[0 * briggs_plane_stride + a_chan * n_pol_g + a_pol];
                    const double briggs_b =
                        briggs_factors[1 * briggs_plane_stride + a_chan * n_pol_g + a_pol];

                    if (std::isnan(briggs_a) || std::isnan(briggs_b))
                        throw std::invalid_argument("NaN in briggs_factors");

                    // Briggs denominator: a * G + b.
                    imaging_weight[wt_idx] = natural_weight / (briggs_a * gij + briggs_b);
                }
            }
        }
    }
}

template void grid_imaging_weights<float>(
    float*, double*, const double*, const double*, const int64_t*,
    const double*, int, int, int, int, int, int, int, int, double, double);
template void grid_imaging_weights<double>(
    double*, double*, const double*, const double*, const int64_t*,
    const double*, int, int, int, int, int, int, int, int, double, double);

template void degrid_imaging_weights<float>(
    double*, const float*, const double*, const double*, const double*,
    const int64_t*, const int64_t*, const double*,
    int, int, int, int, int, int, int, int, int, double, double);
template void degrid_imaging_weights<double>(
    double*, const double*, const double*, const double*, const double*,
    const int64_t*, const int64_t*, const double*,
    int, int, int, int, int, int, int, int, int, double, double);

}  // namespace imaging_weighting
