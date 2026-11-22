#include "../include/anti_alias_degrid.hpp"
#include "scalar_buffer.hpp"
#include "parallel_for.hpp"
#include "stencil_placement.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace anti_alias {

template <typename GridT, typename VisT>
void anti_alias_degrid(
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
    int support, int oversampling, int interpolation,
    int processing_function_threads
) {
    using VisElem = typename VisT::value_type;

    std::vector<double> uv_scale_u(n_vis_chan), uv_scale_v(n_vis_chan);
    uv_scales(frequency_coord, n_vis_chan, delta_l, delta_m, m_u, m_v,
              uv_scale_u.data(), uv_scale_v.data());

    const int support_center = support / 2;
    const int start_support  = -support_center;
    const int end_support    = support - support_center;

    // Grid strides for shape (m_time_g, m_chan_g, m_pol_g, m_u, m_v).
    const int grid_pol_stride  = m_u * m_v;
    const int grid_chan_stride = m_pol_g * grid_pol_stride;
    const int grid_time_stride = m_chan_g * grid_chan_stride;

    // Visibility strides for shape (n_time, n_baseline, n_vis_chan, n_pol).
    const int vis_chan_stride = n_pol;
    const int vis_bl_stride   = n_vis_chan * vis_chan_stride;
    const int vis_time_stride = n_baseline * vis_bl_stride;

    // UVW strides for shape (n_time, n_baseline, 3).
    const int uvw_bl_stride   = 3;
    const int uvw_time_stride = n_baseline * uvw_bl_stride;

    // Per-(time, baseline) kernel. Work is partitioned on (i_time, i_baseline),
    // so each worker writes to a disjoint slice of vis_data — no atomics
    // needed. The grid is read only.
    auto process_time_baseline = [&](int i_time, int i_baseline) {
        const int a_time    = astroviper::scalar_load(time_map, i_time);
        const int vis_t_off = i_time * vis_time_stride;
        const int uvw_t_off = i_time * uvw_time_stride;

        const int vis_b_off = vis_t_off + i_baseline * vis_bl_stride;
        const int uvw_b_off = uvw_t_off + i_baseline * uvw_bl_stride;

        // kernel values of the stencil along u and v, per visibility
        std::vector<double> conv_u(support), conv_v(support);

        for (int i_chan = 0; i_chan < n_vis_chan; ++i_chan) {
            const int a_chan = astroviper::scalar_load(frequency_map, i_chan);

            int u_center_indx, v_center_indx;
            if (!stencil_placement(uvw + uvw_b_off, uv_scale_u[i_chan], uv_scale_v[i_chan],
                                   m_u, m_v, cgk_1D, support, oversampling, interpolation,
                                   u_center_indx, v_center_indx, conv_u.data(),
                                   conv_v.data()))
                continue;

            const int vis_c_off = vis_b_off + i_chan * vis_chan_stride;

            for (int i_pol = 0; i_pol < n_pol; ++i_pol) {
                const int vis_idx = vis_c_off + i_pol;

                // Read existing sample (cast to double precision) to honour
                // the NaN-skip contract regardless of vis_data dtype.
                const VisT current_in = astroviper::scalar_load(vis_data, vis_idx);
                if (std::isnan(static_cast<double>(current_in.real())) ||
                    std::isnan(static_cast<double>(current_in.imag())))
                    continue;

                const int a_pol = astroviper::scalar_load(pol_map, i_pol);

                const int grid_base = a_time * grid_time_stride
                                    + a_chan  * grid_chan_stride
                                    + a_pol   * grid_pol_stride;

                std::complex<double> degrid_value(0.0, 0.0);
                double norm = 0.0;

                // Iterate with i_v innermost so the contiguous last axis of
                // `grid` (v, stride 1) is accessed with unit stride.
                for (int i_u = start_support; i_u < end_support; ++i_u) {
                    const int u_indx = u_center_indx + i_u;
                    const double kernel_u = conv_u[i_u - start_support];

                    const std::complex<GridT>* grid_row =
                        grid + grid_base + u_indx * m_v;

                    for (int i_v = start_support; i_v < end_support; ++i_v) {
                        const int v_indx = v_center_indx + i_v;
                        const double conv = kernel_u * conv_v[i_v - start_support];

                        // Widen the (possibly single-precision) grid cell to
                        // complex<double> so the accumulation stays double.
                        const std::complex<double> grid_value = astroviper::scalar_load(grid_row, v_indx);
                        degrid_value += conv * grid_value;
                        norm += conv;
                    }
                }

                if (norm > 0.0) {
                    const std::complex<double> result = degrid_value / norm;
                    astroviper::scalar_store(vis_data, vis_idx, VisT(
                        static_cast<VisElem>(result.real()),
                        static_cast<VisElem>(result.imag())));
                }
            }
        }
    };

    const int64_t total_work =
        static_cast<int64_t>(n_time) * static_cast<int64_t>(n_baseline);

    auto worker = [&](int64_t begin, int64_t end) {
        for (int64_t flat = begin; flat < end; ++flat) {
            const int i_time     = static_cast<int>(flat / n_baseline);
            const int i_baseline = static_cast<int>(flat % n_baseline);
            process_time_baseline(i_time, i_baseline);
        }
    };

    astroviper::parallel_for(processing_function_threads, total_work, worker);
}

#define AA_DEGRID_INSTANTIATE(GRIDT, VIST)                                     \
    template void anti_alias_degrid<GRIDT, VIST>(                      \
        const std::complex<GRIDT>*, VIST*, const double*, const double*,       \
        const int64_t*, const int64_t*, const int64_t*, const double*,         \
        int, int, int, int, int, int, int, int, int,                           \
        double, double, int, int, int, int);

AA_DEGRID_INSTANTIATE(double, std::complex<float>)
AA_DEGRID_INSTANTIATE(double, std::complex<double>)
AA_DEGRID_INSTANTIATE(float, std::complex<float>)
AA_DEGRID_INSTANTIATE(float, std::complex<double>)

#undef AA_DEGRID_INSTANTIATE

} // namespace anti_alias
