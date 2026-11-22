#include "../include/grid_imaging_weights.hpp"

#include "scalar_buffer.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace imaging_weighting {

namespace {
constexpr double kSpeedOfLight = 299792458.0;
}  // namespace

template <typename GridT, bool Aligned>
void grid_imaging_weights_impl(
    void* grid_raw,
    void* sum_weight_raw,
    const void* uvw_raw,
    const void* frequency_coord_raw,
    const void* chan_map_raw,
    const void* data_weights_raw,
    int n_chan_g, int n_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol,
    double delta_l, double delta_m, bool truncate_uv_cells
) {
    astroviper::ScalarBuffer<GridT, Aligned> grid(grid_raw);
    astroviper::ScalarBuffer<double, Aligned> sum_weight(sum_weight_raw);
    astroviper::ScalarBuffer<const double, Aligned> uvw(uvw_raw);
    astroviper::ScalarBuffer<const double, Aligned> frequency_coord(frequency_coord_raw);
    astroviper::ScalarBuffer<const int64_t, Aligned> chan_map(chan_map_raw);
    astroviper::ScalarBuffer<const double, Aligned> data_weights(data_weights_raw);
    (void)n_chan_g;

    // uv_scale maps baseline meters -> UV pixels for each frequency:
    //   u_pix = uvw_u * -(freq * delta_l * m_u) / c   (likewise for v).
    const int uv_center_u = m_u / 2;
    const int uv_center_v = m_v / 2;

    // Strides for grid shape (n_chan_g, n_pol_g, m_u, m_v):
    const std::int64_t grid_pol_stride =
        static_cast<std::int64_t>(m_u) * m_v;
    const std::int64_t grid_chan_stride =
        static_cast<std::int64_t>(n_pol_g) * grid_pol_stride;

    // Strides for data_weights shape (n_time, n_baseline, n_vis_chan, n_pol):
    const std::int64_t wt_chan_stride = n_pol;
    const std::int64_t wt_bl_stride =
        static_cast<std::int64_t>(n_vis_chan) * wt_chan_stride;
    const std::int64_t wt_time_stride =
        static_cast<std::int64_t>(n_baseline) * wt_bl_stride;

    // Strides for uvw shape (n_time, n_baseline, 3):
    const std::int64_t uvw_bl_stride = 3;
    const std::int64_t uvw_time_stride =
        static_cast<std::int64_t>(n_baseline) * uvw_bl_stride;

    for (int i_time = 0; i_time < n_time; ++i_time) {
        for (int i_baseline = 0; i_baseline < n_baseline; ++i_baseline) {
            const std::int64_t uvw_off =
                i_time * uvw_time_stride + i_baseline * uvw_bl_stride;
            const std::int64_t wt_off =
                i_time * wt_time_stride + i_baseline * wt_bl_stride;

            for (int i_chan = 0; i_chan < n_vis_chan; ++i_chan) {
                const std::int64_t a_chan = chan_map[i_chan];

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

                // Cube weighting uses nearest-cell assignment. CASA continuum
                // weighting instead applies integer conversion after shifting
                // into the positive grid domain, i.e. truncation toward zero.
                // Keep the coordinates in double precision until after bounds
                // checks so narrowing an out-of-range value is never undefined.
                const double u_center = truncate_uv_cells
                    ? std::trunc(u_pos) : std::floor(u_pos + 0.5);
                const double v_center = truncate_uv_cells
                    ? std::trunc(v_pos) : std::floor(v_pos + 0.5);
                const double u_center_conj = truncate_uv_cells
                    ? std::trunc(u_pos_conj) : std::floor(u_pos_conj + 0.5);
                const double v_center_conj = truncate_uv_cells
                    ? std::trunc(v_pos_conj) : std::floor(v_pos_conj + 0.5);

                const bool direct_in_bounds = truncate_uv_cells
                    ? (u_center > 0 && u_center < m_u &&
                       v_center > 0 && v_center < m_v)
                    : (u_center >= 0 && u_center < m_u &&
                       v_center >= 0 && v_center < m_v);
                const bool conjugate_in_bounds = truncate_uv_cells
                    ? (u_center_conj > 0 && u_center_conj < m_u &&
                       v_center_conj > 0 && v_center_conj < m_v)
                    : (u_center_conj >= 0 && u_center_conj < m_u &&
                       v_center_conj >= 0 && v_center_conj < m_v);

                // Preserve main's nearest-cell behavior: both Hermitian cells
                // must be valid before either is accumulated. CASA continuum's
                // historical truncation path handles the two cells separately.
                if (!truncate_uv_cells &&
                    (!direct_in_bounds || !conjugate_in_bounds)) continue;
                if (truncate_uv_cells &&
                    !direct_in_bounds && !conjugate_in_bounds) continue;

                // Only polarization 0 is gridded (weights are parallel-hand
                // equalized upstream).
                const double weight = data_weights[wt_off + i_chan * wt_chan_stride + 0];
                if (std::isnan(weight)) continue;

                if (!truncate_uv_cells) {
                    const std::int64_t cell = a_chan * grid_chan_stride +
                        static_cast<std::int64_t>(u_center) * m_v +
                        static_cast<std::int64_t>(v_center);
                    const std::int64_t cell_conj = a_chan * grid_chan_stride +
                        static_cast<std::int64_t>(u_center_conj) * m_v +
                        static_cast<std::int64_t>(v_center_conj);
                    grid[cell] = static_cast<GridT>(
                        static_cast<double>(grid[cell]) + weight);
                    grid[cell_conj] = static_cast<GridT>(
                        static_cast<double>(grid[cell_conj]) + weight);
                    // Retain main's single addition and accumulation order.
                    sum_weight[a_chan * n_pol_g + 0] += 2.0 * weight;
                } else {
                    if (direct_in_bounds) {
                        const std::int64_t cell = a_chan * grid_chan_stride +
                            static_cast<std::int64_t>(u_center) * m_v +
                            static_cast<std::int64_t>(v_center);
                        grid[cell] = static_cast<GridT>(
                            static_cast<double>(grid[cell]) + weight);
                        sum_weight[a_chan * n_pol_g + 0] += weight;
                    }
                    if (conjugate_in_bounds) {
                        const std::int64_t cell_conj = a_chan * grid_chan_stride +
                            static_cast<std::int64_t>(u_center_conj) * m_v +
                            static_cast<std::int64_t>(v_center_conj);
                        grid[cell_conj] = static_cast<GridT>(
                            static_cast<double>(grid[cell_conj]) + weight);
                        sum_weight[a_chan * n_pol_g + 0] += weight;
                    }
                }
            }
        }
    }
}

template <typename GridT, bool Aligned>
void degrid_imaging_weights_impl(
    void* imaging_weight_raw,
    const void* grid_imaging_weight_raw,
    const void* briggs_factors_raw,
    const void* uvw_raw,
    const void* frequency_coord_raw,
    const void* chan_map_raw,
    const void* pol_map_raw,
    const void* data_weight_raw,
    int n_chan_g, int n_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol, int n_pol_out,
    double delta_l, double delta_m, bool truncate_uv_cells
) {
    astroviper::ScalarBuffer<double, Aligned> imaging_weight(imaging_weight_raw);
    astroviper::ScalarBuffer<const GridT, Aligned> grid_imaging_weight(grid_imaging_weight_raw);
    astroviper::ScalarBuffer<const double, Aligned> briggs_factors(briggs_factors_raw);
    astroviper::ScalarBuffer<const double, Aligned> uvw(uvw_raw);
    astroviper::ScalarBuffer<const double, Aligned> frequency_coord(frequency_coord_raw);
    astroviper::ScalarBuffer<const int64_t, Aligned> chan_map(chan_map_raw);
    astroviper::ScalarBuffer<const int64_t, Aligned> pol_map(pol_map_raw);
    astroviper::ScalarBuffer<const double, Aligned> data_weight(data_weight_raw);
    const int uv_center_u = m_u / 2;
    const int uv_center_v = m_v / 2;

    // Strides for grid shape (n_chan_g, n_pol_g, m_u, m_v):
    const std::int64_t grid_pol_stride =
        static_cast<std::int64_t>(m_u) * m_v;
    const std::int64_t grid_chan_stride =
        static_cast<std::int64_t>(n_pol_g) * grid_pol_stride;

    // Strides for briggs_factors shape (2, n_chan_g, n_pol_g):
    const std::int64_t briggs_plane_stride =
        static_cast<std::int64_t>(n_chan_g) * n_pol_g;

    // Strides for data_weight / imaging_weight shape
    // (n_time, n_baseline, n_vis_chan, n_pol):
    const std::int64_t wt_chan_stride = n_pol;
    const std::int64_t wt_bl_stride =
        static_cast<std::int64_t>(n_vis_chan) * wt_chan_stride;
    const std::int64_t wt_time_stride =
        static_cast<std::int64_t>(n_baseline) * wt_bl_stride;

    // Strides for uvw shape (n_time, n_baseline, 3):
    const std::int64_t uvw_bl_stride = 3;
    const std::int64_t uvw_time_stride =
        static_cast<std::int64_t>(n_baseline) * uvw_bl_stride;

    for (int i_time = 0; i_time < n_time; ++i_time) {
        for (int i_baseline = 0; i_baseline < n_baseline; ++i_baseline) {
            const std::int64_t uvw_off =
                i_time * uvw_time_stride + i_baseline * uvw_bl_stride;
            const std::int64_t wt_off =
                i_time * wt_time_stride + i_baseline * wt_bl_stride;

            for (int i_chan = 0; i_chan < n_vis_chan; ++i_chan) {
                const std::int64_t a_chan = chan_map[i_chan];

                const double uv_scale_u =
                    -(frequency_coord[i_chan] * delta_l * m_u) / kSpeedOfLight;
                const double uv_scale_v =
                    -(frequency_coord[i_chan] * delta_m * m_v) / kSpeedOfLight;

                const double u = uvw[uvw_off + 0] * uv_scale_u;
                const double v = uvw[uvw_off + 1] * uv_scale_v;

                if (std::isnan(u) || std::isnan(v)) continue;

                const double u_pos = u + uv_center_u;
                const double v_pos = v + uv_center_v;
                const double u_center = truncate_uv_cells
                    ? std::trunc(u_pos) : std::floor(u_pos + 0.5);
                const double v_center = truncate_uv_cells
                    ? std::trunc(v_pos) : std::floor(v_pos + 0.5);

                if (truncate_uv_cells) {
                    if (u_center <= 0 || u_center >= m_u) continue;
                    if (v_center <= 0 || v_center >= m_v) continue;
                } else {
                    // As gridded: both Hermitian cells must be on the grid.
                    const double u_center_conj = std::floor(-u + uv_center_u + 0.5);
                    const double v_center_conj = std::floor(-v + uv_center_v + 0.5);
                    if (u_center < 0 || u_center >= m_u) continue;
                    if (v_center < 0 || v_center >= m_v) continue;
                    if (u_center_conj < 0 || u_center_conj >= m_u) continue;
                    if (v_center_conj < 0 || v_center_conj >= m_v) continue;
                }

                const std::int64_t u_center_indx =
                    static_cast<std::int64_t>(u_center);
                const std::int64_t v_center_indx =
                    static_cast<std::int64_t>(v_center);

                for (int i_pol = 0; i_pol < n_pol_out; ++i_pol) {
                    const std::int64_t a_pol = pol_map[i_pol];
                    const std::int64_t wt_idx =
                        wt_off + i_chan * wt_chan_stride + i_pol;
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
    double delta_l, double delta_m, bool truncate_uv_cells
) {
    const bool aligned = astroviper::naturally_aligned<GridT>(grid) &&
        astroviper::naturally_aligned<double>(sum_weight) &&
        astroviper::naturally_aligned<double>(uvw) &&
        astroviper::naturally_aligned<double>(frequency_coord) &&
        astroviper::naturally_aligned<int64_t>(chan_map) &&
        astroviper::naturally_aligned<double>(data_weights);
    if (aligned)
        grid_imaging_weights_impl<GridT, true>(
            grid, sum_weight, uvw, frequency_coord, chan_map, data_weights, n_chan_g, n_pol_g,
            m_u, m_v, n_time, n_baseline, n_vis_chan, n_pol, delta_l, delta_m,
            truncate_uv_cells);
    else
        grid_imaging_weights_impl<GridT, false>(
            grid, sum_weight, uvw, frequency_coord, chan_map, data_weights, n_chan_g, n_pol_g,
            m_u, m_v, n_time, n_baseline, n_vis_chan, n_pol, delta_l, delta_m,
            truncate_uv_cells);
}

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
    double delta_l, double delta_m, bool truncate_uv_cells
) {
    const bool aligned = astroviper::naturally_aligned<double>(imaging_weight) &&
        astroviper::naturally_aligned<GridT>(grid_imaging_weight) &&
        astroviper::naturally_aligned<double>(briggs_factors) &&
        astroviper::naturally_aligned<double>(uvw) &&
        astroviper::naturally_aligned<double>(frequency_coord) &&
        astroviper::naturally_aligned<int64_t>(chan_map) &&
        astroviper::naturally_aligned<int64_t>(pol_map) &&
        astroviper::naturally_aligned<double>(data_weight);
    if (aligned)
        degrid_imaging_weights_impl<GridT, true>(
            imaging_weight, grid_imaging_weight, briggs_factors, uvw, frequency_coord,
            chan_map, pol_map, data_weight, n_chan_g, n_pol_g, m_u, m_v, n_time, n_baseline,
            n_vis_chan, n_pol, n_pol_out, delta_l, delta_m, truncate_uv_cells);
    else
        degrid_imaging_weights_impl<GridT, false>(
            imaging_weight, grid_imaging_weight, briggs_factors, uvw, frequency_coord,
            chan_map, pol_map, data_weight, n_chan_g, n_pol_g, m_u, m_v, n_time, n_baseline,
            n_vis_chan, n_pol, n_pol_out, delta_l, delta_m, truncate_uv_cells);
}
template void grid_imaging_weights<float>(
    void*, void*, const void*, const void*, const void*,
    const void*, int, int, int, int, int, int, int, int, double, double, bool);
template void grid_imaging_weights<double>(
    void*, void*, const void*, const void*, const void*,
    const void*, int, int, int, int, int, int, int, int, double, double, bool);

template void degrid_imaging_weights<float>(
    void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*,
    int, int, int, int, int, int, int, int, int, double, double, bool);
template void degrid_imaging_weights<double>(
    void*, const void*, const void*, const void*, const void*,
    const void*, const void*, const void*,
    int, int, int, int, int, int, int, int, int, double, double, bool);

}  // namespace imaging_weighting
