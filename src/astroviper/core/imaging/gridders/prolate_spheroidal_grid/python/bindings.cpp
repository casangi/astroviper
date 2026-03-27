#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <stdexcept>
#include "../include/prolate_spheroidal_grid.hpp"

namespace py = pybind11;

using cdbl = std::complex<double>;

/**
 * Pybind11 wrapper for prolate_spheroidal_grid.
 *
 * Grids weighted visibilities onto a UV plane in-place.
 *
 * Parameters match the Python function prolate_spheroidal_grid_jit exactly.
 */
void prolate_spheroidal_grid_bind(
    py::array_t<cdbl,   py::array::c_style> grid,
    py::array_t<double, py::array::c_style> normalization,
    py::array_t<cdbl,   py::array::c_style | py::array::forcecast> vis_data,
    py::array_t<double, py::array::c_style | py::array::forcecast> uvw,
    py::array_t<double, py::array::c_style | py::array::forcecast> frequency_coord,
    py::array_t<int,    py::array::c_style | py::array::forcecast> frequency_map,
    py::array_t<int,    py::array::c_style | py::array::forcecast> time_map,
    py::array_t<int,    py::array::c_style | py::array::forcecast> pol_map,
    py::array_t<double, py::array::c_style | py::array::forcecast> weight,
    py::array_t<double, py::array::c_style | py::array::forcecast> cgk_1D,
    py::array_t<int,    py::array::c_style | py::array::forcecast> n_uv,
    py::array_t<double, py::array::c_style | py::array::forcecast> delta_lm,
    int support,
    int oversampling
) {
    // Validate shapes
    auto grid_info  = grid.request();
    auto norm_info  = normalization.request();
    auto vis_info   = vis_data.request();
    auto uvw_info   = uvw.request();
    auto wt_info    = weight.request();

    if (grid_info.ndim != 5)
        throw std::runtime_error("grid must be 5-D (m_time, m_chan, m_pol, m_u, m_v)");
    if (norm_info.ndim != 3)
        throw std::runtime_error("normalization must be 3-D (m_time, m_chan, m_pol)");
    if (vis_info.ndim != 4)
        throw std::runtime_error("vis_data must be 4-D (n_time, n_baseline, n_vis_chan, n_pol)");
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");

    // Extract grid dimensions
    const int m_time_g  = static_cast<int>(grid_info.shape[0]);
    const int m_chan_g  = static_cast<int>(grid_info.shape[1]);
    const int m_pol_g   = static_cast<int>(grid_info.shape[2]);
    const int m_u       = static_cast<int>(grid_info.shape[3]);
    const int m_v       = static_cast<int>(grid_info.shape[4]);

    // Extract visibility dimensions
    const int n_time     = static_cast<int>(uvw_info.shape[0]);
    const int n_baseline = static_cast<int>(uvw_info.shape[1]);
    const int n_vis_chan = static_cast<int>(frequency_map.shape(0));
    const int n_pol      = static_cast<int>(pol_map.shape(0));

    // Extract delta_lm
    auto dlm = delta_lm.unchecked<1>();
    const double delta_l = dlm(0);
    const double delta_m = dlm(1);

    prolate_spheroidal::prolate_spheroidal_grid(
        static_cast<cdbl*>(grid_info.ptr),
        static_cast<double*>(norm_info.ptr),
        static_cast<const cdbl*>(vis_info.ptr),
        static_cast<const double*>(uvw_info.ptr),
        frequency_coord.data(),
        frequency_map.data(),
        time_map.data(),
        pol_map.data(),
        static_cast<const double*>(wt_info.ptr),
        cgk_1D.data(),
        m_time_g, m_chan_g, m_pol_g, m_u, m_v,
        n_time, n_baseline, n_vis_chan, n_pol,
        delta_l, delta_m,
        support, oversampling
    );
}

/**
 * Pybind11 wrapper for prolate_spheroidal_grid_uv_sampling.
 *
 * Grids imaging weights onto a UV plane in-place (PSF / UV-sampling function).
 *
 * Parameters match the Python function prolate_spheroidal_grid_uv_sampling_jit exactly,
 * except vis_data is absent (weights are gridded directly).
 */
void prolate_spheroidal_grid_uv_sampling_bind(
    py::array_t<cdbl,   py::array::c_style> grid,
    py::array_t<double, py::array::c_style> normalization,
    py::array_t<double, py::array::c_style | py::array::forcecast> uvw,
    py::array_t<double, py::array::c_style | py::array::forcecast> frequency_coord,
    py::array_t<int,    py::array::c_style | py::array::forcecast> frequency_map,
    py::array_t<int,    py::array::c_style | py::array::forcecast> time_map,
    py::array_t<int,    py::array::c_style | py::array::forcecast> pol_map,
    py::array_t<double, py::array::c_style | py::array::forcecast> weight,
    py::array_t<double, py::array::c_style | py::array::forcecast> cgk_1D,
    py::array_t<int,    py::array::c_style | py::array::forcecast> n_uv,
    py::array_t<double, py::array::c_style | py::array::forcecast> delta_lm,
    int support,
    int oversampling
) {
    auto grid_info = grid.request();
    auto norm_info = normalization.request();
    auto uvw_info  = uvw.request();
    auto wt_info   = weight.request();

    if (grid_info.ndim != 5)
        throw std::runtime_error("grid must be 5-D (m_time, m_chan, m_pol, m_u, m_v)");
    if (norm_info.ndim != 3)
        throw std::runtime_error("normalization must be 3-D (m_time, m_chan, m_pol)");
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");

    const int m_time_g  = static_cast<int>(grid_info.shape[0]);
    const int m_chan_g  = static_cast<int>(grid_info.shape[1]);
    const int m_pol_g   = static_cast<int>(grid_info.shape[2]);
    const int m_u       = static_cast<int>(grid_info.shape[3]);
    const int m_v       = static_cast<int>(grid_info.shape[4]);

    const int n_time     = static_cast<int>(uvw_info.shape[0]);
    const int n_baseline = static_cast<int>(uvw_info.shape[1]);
    const int n_vis_chan = static_cast<int>(frequency_map.shape(0));
    const int n_pol      = static_cast<int>(pol_map.shape(0));

    auto dlm = delta_lm.unchecked<1>();
    const double delta_l = dlm(0);
    const double delta_m = dlm(1);

    prolate_spheroidal::prolate_spheroidal_grid_uv_sampling(
        static_cast<cdbl*>(grid_info.ptr),
        static_cast<double*>(norm_info.ptr),
        static_cast<const double*>(uvw_info.ptr),
        frequency_coord.data(),
        frequency_map.data(),
        time_map.data(),
        pol_map.data(),
        static_cast<const double*>(wt_info.ptr),
        cgk_1D.data(),
        m_time_g, m_chan_g, m_pol_g, m_u, m_v,
        n_time, n_baseline, n_vis_chan, n_pol,
        delta_l, delta_m,
        support, oversampling
    );
}

PYBIND11_MODULE(_prolate_spheroidal_grid_ext, m) {
    m.doc() = "C++ prolate spheroidal convolution gridder (UV gridding)";

    m.def("prolate_spheroidal_grid",
          &prolate_spheroidal_grid_bind,
          "Grid weighted visibilities onto a UV plane using a prolate spheroidal kernel (in-place).",
          py::arg("grid"),
          py::arg("normalization"),
          py::arg("vis_data"),
          py::arg("uvw"),
          py::arg("frequency_coord"),
          py::arg("frequency_map"),
          py::arg("time_map"),
          py::arg("pol_map"),
          py::arg("weight"),
          py::arg("cgk_1D"),
          py::arg("n_uv"),
          py::arg("delta_lm"),
          py::arg("support"),
          py::arg("oversampling"));

    m.def("prolate_spheroidal_grid_uv_sampling",
          &prolate_spheroidal_grid_uv_sampling_bind,
          "Grid imaging weights onto a UV plane (PSF / UV-sampling function, in-place).",
          py::arg("grid"),
          py::arg("normalization"),
          py::arg("uvw"),
          py::arg("frequency_coord"),
          py::arg("frequency_map"),
          py::arg("time_map"),
          py::arg("pol_map"),
          py::arg("weight"),
          py::arg("cgk_1D"),
          py::arg("n_uv"),
          py::arg("delta_lm"),
          py::arg("support"),
          py::arg("oversampling"));
}
