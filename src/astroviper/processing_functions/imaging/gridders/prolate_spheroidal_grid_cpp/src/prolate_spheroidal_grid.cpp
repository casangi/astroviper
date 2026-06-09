#include "../include/prolate_spheroidal_grid.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <thread>
#include <vector>

namespace prolate_spheroidal {

namespace {

// Lock-free atomic add on a non-atomic double. numpy allocates doubles
// with 8-byte alignment, which satisfies the lock-free requirement on all
// supported platforms.
//
// Prefer std::atomic_ref (C++20), but fall back to the GCC/Clang __atomic
// builtins on compilers that don't yet implement it (e.g. GCC < 10, which
// is still the system compiler on RHEL/Rocky 8). The builtins operate on
// the same suitably aligned plain double, so the two paths are equivalent.
inline void atomic_add_double(double* target, double value) {
#if defined(__cpp_lib_atomic_ref)
    std::atomic_ref<double> ref(*target);
    double current = ref.load(std::memory_order_relaxed);
    while (!ref.compare_exchange_weak(
        current, current + value, std::memory_order_relaxed)) {}
#else
    // Generic __atomic_* builtins (pointer args) work on plain double on
    // both GCC and Clang; the type-specialized __atomic_*_n forms reject
    // floating-point types under Clang.
    double current;
    __atomic_load(target, &current, __ATOMIC_RELAXED);
    double desired;
    do {
        desired = current + value;
    } while (!__atomic_compare_exchange(
        target, &current, &desired, /*weak=*/true,
        __ATOMIC_RELAXED, __ATOMIC_RELAXED));
#endif
}

// std::complex<double> is guaranteed to be layout-compatible with
// double[2], so we atomically add the real and imaginary parts
// independently.
inline void atomic_add_complex(std::complex<double>* target,
                               std::complex<double> value) {
    double* parts = reinterpret_cast<double*>(target);
    atomic_add_double(parts + 0, value.real());
    atomic_add_double(parts + 1, value.imag());
}

} // namespace

void prolate_spheroidal_grid(
    std::complex<double>* grid,
    double* normalization,
    const std::complex<double>* vis_data,
    const double* uvw,
    const double* frequency_coord,
    const int64_t* frequency_map,
    const int64_t* time_map,
    const int64_t* pol_map,
    const double* weight,
    const double* cgk_1D,
    int m_time_g, int m_chan_g, int m_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol,
    double delta_l, double delta_m,
    int support, int oversampling,
    int num_threads
) {
    constexpr double c = 299792458.0;

    // Precompute per-channel UV scaling factors:
    //   uv_scale[0, i] = -(freq[i] * delta_l * m_u) / c
    //   uv_scale[1, i] = -(freq[i] * delta_m * m_v) / c
    std::vector<double> uv_scale_u(n_vis_chan), uv_scale_v(n_vis_chan);
    for (int i = 0; i < n_vis_chan; ++i) {
        uv_scale_u[i] = -(frequency_coord[i] * delta_l * m_u) / c;
        uv_scale_v[i] = -(frequency_coord[i] * delta_m * m_v) / c;
    }

    const int support_center = support / 2;
    const int uv_center_u    = m_u / 2;
    const int uv_center_v    = m_v / 2;
    const int start_support  = -support_center;
    const int end_support    = support - support_center;

    // Grid strides for shape (m_time_g, m_chan_g, m_pol_g, m_u, m_v):
    //   grid[a_time, a_chan, a_pol, u_idx, v_idx] =
    //     grid[((a_time*m_chan_g + a_chan)*m_pol_g + a_pol)*m_u*m_v + u_idx*m_v + v_idx]
    const int grid_pol_stride  = m_u * m_v;
    const int grid_chan_stride  = m_pol_g * grid_pol_stride;
    const int grid_time_stride  = m_chan_g * grid_chan_stride;

    // Normalization strides for shape (m_time_g, m_chan_g, m_pol_g):
    //   norm[a_time, a_chan, a_pol] = norm[(a_time*m_chan_g + a_chan)*m_pol_g + a_pol]
    const int norm_chan_stride = m_pol_g;
    const int norm_time_stride = m_chan_g * norm_chan_stride;

    // Visibility / weight strides for shape (n_time, n_baseline, n_vis_chan, n_pol):
    //   arr[i_time, i_baseline, i_chan, i_pol] =
    //     arr[((i_time*n_baseline + i_baseline)*n_vis_chan + i_chan)*n_pol + i_pol]
    const int vis_pol_stride  = 1;
    const int vis_chan_stride  = n_pol;
    const int vis_bl_stride   = n_vis_chan * vis_chan_stride;
    const int vis_time_stride = n_baseline * vis_bl_stride;

    // UVW strides for shape (n_time, n_baseline, 3):
    const int uvw_bl_stride   = 3;
    const int uvw_time_stride = n_baseline * uvw_bl_stride;

    // Per-(time, baseline) kernel. Multiple workers may collide on the same
    // grid cell (baselines can sample overlapping UV coordinates), so every
    // write into grid / normalization goes through an atomic add.
    auto process_time_baseline = [&](int i_time, int i_baseline) {
        const int a_time    = time_map[i_time];
        const int vis_t_off = i_time * vis_time_stride;
        const int uvw_t_off = i_time * uvw_time_stride;

        const int vis_b_off = vis_t_off + i_baseline * vis_bl_stride;
        const int uvw_b_off = uvw_t_off + i_baseline * uvw_bl_stride;

        for (int i_chan = 0; i_chan < n_vis_chan; ++i_chan) {
            const int a_chan = frequency_map[i_chan];

            const double u = uvw[uvw_b_off + 0] * uv_scale_u[i_chan];
            const double v = uvw[uvw_b_off + 1] * uv_scale_v[i_chan];

            if (std::isnan(u) || std::isnan(v)) continue;

            const double u_pos = u + uv_center_u;
            const double v_pos = v + uv_center_v;

            // Round to the nearest grid centre and bounds-check in double
            // precision *before* narrowing to int. Casting an out-of-range
            // double to int is undefined behaviour (it yields INT_MIN on
            // x86/ARM), and the `idx - support_center` guard would then
            // overflow, letting an extreme UV coordinate slip through and
            // corrupt memory. Validating in double — where nothing overflows
            // — guarantees the centre lies in [support_center, m - support_center)
            // and therefore fits in int. For an in-bounds (non-negative)
            // coordinate std::floor(x + 0.5) matches the previous
            // truncating cast exactly.
            const double u_center = std::floor(u_pos + 0.5);
            const double v_center = std::floor(v_pos + 0.5);

            if (u_center + support_center >= m_u) continue;
            if (v_center + support_center >= m_v) continue;
            if (u_center - support_center <  0)   continue;
            if (v_center - support_center <  0)   continue;

            const int u_center_indx = static_cast<int>(u_center);
            const int v_center_indx = static_cast<int>(v_center);

            const double u_offset = u_center_indx - u_pos;
            const int    u_center_offset_indx =
                static_cast<int>(std::floor(u_offset * oversampling + 0.5));

            const double v_offset = v_center_indx - v_pos;
            const int    v_center_offset_indx =
                static_cast<int>(std::floor(v_offset * oversampling + 0.5));

            const int vis_c_off = vis_b_off + i_chan * vis_chan_stride;

            for (int i_pol = 0; i_pol < n_pol; ++i_pol) {
                const int vis_idx = vis_c_off + i_pol;

                const double sel_weight = weight[vis_idx];
                const std::complex<double> wd = vis_data[vis_idx] * sel_weight;

                if (std::isnan(wd.real()) || std::isnan(wd.imag())) continue;
                if (wd == std::complex<double>(0.0, 0.0)) continue;

                const int a_pol = pol_map[i_pol];

                const int grid_base = a_time * grid_time_stride
                                    + a_chan  * grid_chan_stride
                                    + a_pol   * grid_pol_stride;
                const int norm_idx  = a_time * norm_time_stride
                                    + a_chan  * norm_chan_stride
                                    + a_pol;

                double norm = 0.0;

                // Iterate with i_v innermost so the contiguous last axis of
                // `grid` (v, stride 1) is accessed with unit stride.
                for (int i_u = start_support; i_u < end_support; ++i_u) {
                    const int u_indx = u_center_indx + i_u;
                    const int u_offset_indx =
                        std::abs(oversampling * i_u + u_center_offset_indx);
                    const double conv_u = cgk_1D[u_offset_indx];

                    std::complex<double>* grid_row =
                        grid + grid_base + u_indx * m_v;

                    for (int i_v = start_support; i_v < end_support; ++i_v) {
                        const int v_indx = v_center_indx + i_v;
                        const int v_offset_indx =
                            std::abs(oversampling * i_v + v_center_offset_indx);
                        const double conv = conv_u * cgk_1D[v_offset_indx];

                        atomic_add_complex(grid_row + v_indx, conv * wd);
                        norm += conv;
                    }
                }

                atomic_add_double(normalization + norm_idx, sel_weight * norm);
            }
        }
    };

    const int64_t total_work =
        static_cast<int64_t>(n_time) * static_cast<int64_t>(n_baseline);
    if (total_work <= 0) return;

    int resolved_threads = num_threads;
    if (resolved_threads <= 0) {
        resolved_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (resolved_threads <= 0) resolved_threads = 1;
    }
    if (static_cast<int64_t>(resolved_threads) > total_work) {
        resolved_threads = static_cast<int>(total_work);
    }

    auto worker = [&](int64_t begin, int64_t end) {
        for (int64_t flat = begin; flat < end; ++flat) {
            const int i_time     = static_cast<int>(flat / n_baseline);
            const int i_baseline = static_cast<int>(flat % n_baseline);
            process_time_baseline(i_time, i_baseline);
        }
    };

    if (resolved_threads <= 1) {
        worker(0, total_work);
        return;
    }

    const int64_t chunk =
        (total_work + resolved_threads - 1) / resolved_threads;

    std::vector<std::thread> threads;
    threads.reserve(resolved_threads - 1);
    for (int t = 0; t < resolved_threads - 1; ++t) {
        const int64_t begin = static_cast<int64_t>(t) * chunk;
        const int64_t end   = std::min(total_work, begin + chunk);
        threads.emplace_back(worker, begin, end);
    }
    // Run the last chunk on the calling thread to save one join.
    worker(static_cast<int64_t>(resolved_threads - 1) * chunk, total_work);
    for (auto& th : threads) th.join();
}


void prolate_spheroidal_grid_uv_sampling(
    std::complex<double>* grid,
    double* normalization,
    const double* uvw,
    const double* frequency_coord,
    const int64_t* frequency_map,
    const int64_t* time_map,
    const int64_t* pol_map,
    const double* weight,
    const double* cgk_1D,
    int m_time_g, int m_chan_g, int m_pol_g, int m_u, int m_v,
    int n_time, int n_baseline, int n_vis_chan, int n_pol,
    double delta_l, double delta_m,
    int support, int oversampling,
    int num_threads
) {
    constexpr double c = 299792458.0;

    std::vector<double> uv_scale_u(n_vis_chan), uv_scale_v(n_vis_chan);
    for (int i = 0; i < n_vis_chan; ++i) {
        uv_scale_u[i] = -(frequency_coord[i] * delta_l * m_u) / c;
        uv_scale_v[i] = -(frequency_coord[i] * delta_m * m_v) / c;
    }

    const int support_center = support / 2;
    const int uv_center_u    = m_u / 2;
    const int uv_center_v    = m_v / 2;
    const int start_support  = -support_center;
    const int end_support    = support - support_center;

    const int grid_pol_stride  = m_u * m_v;
    const int grid_chan_stride  = m_pol_g * grid_pol_stride;
    const int grid_time_stride  = m_chan_g * grid_chan_stride;

    const int norm_chan_stride = m_pol_g;
    const int norm_time_stride = m_chan_g * norm_chan_stride;

    const int wt_chan_stride  = n_pol;
    const int wt_bl_stride    = n_vis_chan * wt_chan_stride;
    const int wt_time_stride  = n_baseline * wt_bl_stride;

    const int uvw_bl_stride   = 3;
    const int uvw_time_stride = n_baseline * uvw_bl_stride;

    // Per-(time, baseline) kernel; same threading/atomic strategy as
    // prolate_spheroidal_grid.
    auto process_time_baseline = [&](int i_time, int i_baseline) {
        const int a_time    = time_map[i_time];
        const int wt_t_off  = i_time * wt_time_stride;
        const int uvw_t_off = i_time * uvw_time_stride;

        const int wt_b_off  = wt_t_off  + i_baseline * wt_bl_stride;
        const int uvw_b_off = uvw_t_off + i_baseline * uvw_bl_stride;

        for (int i_chan = 0; i_chan < n_vis_chan; ++i_chan) {
            const int a_chan = frequency_map[i_chan];

            const double u = uvw[uvw_b_off + 0] * uv_scale_u[i_chan];
            const double v = uvw[uvw_b_off + 1] * uv_scale_v[i_chan];

            if (std::isnan(u) || std::isnan(v)) continue;

            const double u_pos = u + uv_center_u;
            const double v_pos = v + uv_center_v;

            // Round to the nearest grid centre and bounds-check in double
            // precision *before* narrowing to int. Casting an out-of-range
            // double to int is undefined behaviour (it yields INT_MIN on
            // x86/ARM), and the `idx - support_center` guard would then
            // overflow, letting an extreme UV coordinate slip through and
            // corrupt memory. Validating in double — where nothing overflows
            // — guarantees the centre lies in [support_center, m - support_center)
            // and therefore fits in int. For an in-bounds (non-negative)
            // coordinate std::floor(x + 0.5) matches the previous
            // truncating cast exactly.
            const double u_center = std::floor(u_pos + 0.5);
            const double v_center = std::floor(v_pos + 0.5);

            if (u_center + support_center >= m_u) continue;
            if (v_center + support_center >= m_v) continue;
            if (u_center - support_center <  0)   continue;
            if (v_center - support_center <  0)   continue;

            const int u_center_indx = static_cast<int>(u_center);
            const int v_center_indx = static_cast<int>(v_center);

            const double u_offset = u_center_indx - u_pos;
            const int    u_center_offset_indx =
                static_cast<int>(std::floor(u_offset * oversampling + 0.5));

            const double v_offset = v_center_indx - v_pos;
            const int    v_center_offset_indx =
                static_cast<int>(std::floor(v_offset * oversampling + 0.5));

            const int wt_c_off = wt_b_off + i_chan * wt_chan_stride;

            for (int i_pol = 0; i_pol < n_pol; ++i_pol) {
                const double weight_data = weight[wt_c_off + i_pol];

                if (std::isnan(weight_data) || weight_data == 0.0) continue;

                const int a_pol = pol_map[i_pol];

                const int grid_base = a_time * grid_time_stride
                                    + a_chan  * grid_chan_stride
                                    + a_pol   * grid_pol_stride;
                const int norm_idx  = a_time * norm_time_stride
                                    + a_chan  * norm_chan_stride
                                    + a_pol;

                double norm = 0.0;

                // Iterate with i_v innermost so the contiguous last axis of
                // `grid` (v, stride 1) is accessed with unit stride.
                for (int i_u = start_support; i_u < end_support; ++i_u) {
                    const int u_indx = u_center_indx + i_u;
                    const int u_offset_indx =
                        std::abs(oversampling * i_u + u_center_offset_indx);
                    const double conv_u = cgk_1D[u_offset_indx];

                    std::complex<double>* grid_row =
                        grid + grid_base + u_indx * m_v;

                    for (int i_v = start_support; i_v < end_support; ++i_v) {
                        const int v_indx = v_center_indx + i_v;
                        const int v_offset_indx =
                            std::abs(oversampling * i_v + v_center_offset_indx);
                        const double conv = conv_u * cgk_1D[v_offset_indx];

                        // The grid is real-valued here (we add a real weight),
                        // so only the real part of the complex cell needs an
                        // atomic add. The imag part is never touched.
                        double* parts = reinterpret_cast<double*>(
                            grid_row + v_indx);
                        atomic_add_double(parts + 0, conv * weight_data);
                        norm += conv;
                    }
                }

                atomic_add_double(normalization + norm_idx, weight_data * norm);
            }
        }
    };

    const int64_t total_work =
        static_cast<int64_t>(n_time) * static_cast<int64_t>(n_baseline);
    if (total_work <= 0) return;

    int resolved_threads = num_threads;
    if (resolved_threads <= 0) {
        resolved_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (resolved_threads <= 0) resolved_threads = 1;
    }
    if (static_cast<int64_t>(resolved_threads) > total_work) {
        resolved_threads = static_cast<int>(total_work);
    }

    auto worker = [&](int64_t begin, int64_t end) {
        for (int64_t flat = begin; flat < end; ++flat) {
            const int i_time     = static_cast<int>(flat / n_baseline);
            const int i_baseline = static_cast<int>(flat % n_baseline);
            process_time_baseline(i_time, i_baseline);
        }
    };

    if (resolved_threads <= 1) {
        worker(0, total_work);
        return;
    }

    const int64_t chunk =
        (total_work + resolved_threads - 1) / resolved_threads;

    std::vector<std::thread> threads;
    threads.reserve(resolved_threads - 1);
    for (int t = 0; t < resolved_threads - 1; ++t) {
        const int64_t begin = static_cast<int64_t>(t) * chunk;
        const int64_t end   = std::min(total_work, begin + chunk);
        threads.emplace_back(worker, begin, end);
    }
    worker(static_cast<int64_t>(resolved_threads - 1) * chunk, total_work);
    for (auto& th : threads) th.join();
}

} // namespace prolate_spheroidal
