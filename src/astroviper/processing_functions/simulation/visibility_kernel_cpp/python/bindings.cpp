#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/visibility_kernel.hpp"

namespace py = pybind11;
using visibility_kernel::BeamModel;

namespace {

/**
 * Validate dtype / contiguity / dimensionality of a Python-owned array and
 * return its buffer (no copy, no implicit conversion).  Follows the AstroVIPER
 * Python<->C++ memory contract (AGENTS.md section 6).
 */
template <typename T>
py::buffer_info checked_buffer(const py::array& arr, const char* name, int ndim, bool writable) {
    if (!arr.dtype().is(py::dtype::of<T>())) {
        throw std::runtime_error(std::string(name) + " has wrong dtype; expected " +
                                 py::cast<std::string>(py::dtype::of<T>().attr("name")) + " but got " +
                                 py::cast<std::string>(arr.dtype().attr("name")));
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error(std::string(name) +
                                 " must be C-contiguous (use np.ascontiguousarray; no copy is made)");
    }
    if (writable && !arr.writeable()) {
        throw std::runtime_error(std::string(name) + " must be writeable (modified in place)");
    }
    py::buffer_info info = arr.request(writable);
    if (info.ndim != ndim) {
        throw std::runtime_error(std::string(name) + " must be a " + std::to_string(ndim) + "-D array");
    }
    return info;
}

template <typename T>
const T* ptr_of(const py::buffer_info& info) {
    return static_cast<const T*>(info.ptr);
}

BeamModel beam_model_from_dict(const py::dict& d, std::vector<py::buffer_info>& keep) {
    BeamModel bm;
    const std::string kind = py::cast<std::string>(d["kind"]);
    bm.max_rad_1GHz = py::cast<double>(d["max_rad_1GHz"]);
    if (kind == "analytic") {
        bm.kind = 0;
        const std::string func = py::cast<std::string>(d["func"]);
        if (func == "none") bm.func = 0;
        else if (func == "casa_airy") bm.func = 1;
        else if (func == "airy") bm.func = 2;
        else throw std::runtime_error("Unknown analytic beam function '" + func + "'");
        bm.dish_diameter = py::cast<double>(d["dish_diameter"]);
        bm.blockage_diameter = py::cast<double>(d["blockage_diameter"]);
    } else if (kind == "polynomial") {
        bm.kind = 1;
        py::array freq = py::cast<py::array>(d["frequency"]);
        py::array coef = py::cast<py::array>(d["coefficients"]);
        keep.push_back(checked_buffer<double>(freq, "beam polynomial frequency", 1, false));
        keep.push_back(checked_buffer<double>(coef, "beam polynomial coefficients", 2, false));
        const py::buffer_info& fi = keep[keep.size() - 2];
        const py::buffer_info& ci = keep[keep.size() - 1];
        bm.n_poly_frequency = fi.shape[0];
        bm.n_poly_coefficient = ci.shape[1];
        if (ci.shape[0] != bm.n_poly_frequency) {
            throw std::runtime_error("beam polynomial coefficients must have shape [n_frequency, n_coefficient]");
        }
        bm.poly_frequency = ptr_of<double>(fi);
        bm.poly_coefficients = ptr_of<double>(ci);
    } else if (kind == "jones_image") {
        bm.kind = 2;
        py::array jones = py::cast<py::array>(d["jones"]);
        py::array pa = py::cast<py::array>(d["parallactic_angle"]);
        py::array freq = py::cast<py::array>(d["frequency"]);
        py::array pol = py::cast<py::array>(d["polarization_index"]);
        keep.push_back(checked_buffer<std::complex<double>>(jones, "jones", 5, false));
        keep.push_back(checked_buffer<double>(pa, "jones parallactic_angle", 1, false));
        keep.push_back(checked_buffer<double>(freq, "jones frequency", 1, false));
        keep.push_back(checked_buffer<std::int64_t>(pol, "jones polarization_index", 1, false));
        const std::size_t n = keep.size();
        const py::buffer_info& ji = keep[n - 4];
        const py::buffer_info& pi = keep[n - 3];
        const py::buffer_info& fi = keep[n - 2];
        const py::buffer_info& li = keep[n - 1];
        bm.n_pa = ji.shape[0];
        bm.n_jones_frequency = ji.shape[1];
        bm.n_jones_pol = ji.shape[2];
        bm.n_l = ji.shape[3];
        bm.n_m = ji.shape[4];
        if (pi.shape[0] != bm.n_pa || fi.shape[0] != bm.n_jones_frequency || li.shape[0] != bm.n_jones_pol) {
            throw std::runtime_error("jones image coordinate arrays do not match the jones array shape");
        }
        bm.jones = ptr_of<std::complex<double>>(ji);
        bm.jones_parallactic_angle = ptr_of<double>(pi);
        bm.jones_frequency = ptr_of<double>(fi);
        bm.jones_polarization_index = ptr_of<std::int64_t>(li);
        bm.cell_size_l = py::cast<double>(d["cell_size_l"]);
        bm.cell_size_m = py::cast<double>(d["cell_size_m"]);
    } else {
        throw std::runtime_error("Unknown packed beam model kind '" + kind + "'");
    }
    return bm;
}

void calculate_visibilities_py(py::array visibility, py::array uvw, py::array antenna1, py::array antenna2,
                               py::array frequency, py::array polarization_index, py::array flux,
                               py::array k_vector, py::array inverse_n, py::array source_ra_dec,
                               py::array pointing_ra_dec, py::array beam_model_map, py::list beam_models,
                               py::array parallactic_angle, py::array mueller_selection, int n_threads) {
    auto vis_info = checked_buffer<std::complex<double>>(visibility, "visibility", 4, true);
    auto uvw_info = checked_buffer<double>(uvw, "uvw", 3, false);
    auto a1_info = checked_buffer<std::int64_t>(antenna1, "antenna1", 1, false);
    auto a2_info = checked_buffer<std::int64_t>(antenna2, "antenna2", 1, false);
    auto freq_info = checked_buffer<double>(frequency, "frequency", 1, false);
    auto pol_info = checked_buffer<std::int64_t>(polarization_index, "polarization_index", 1, false);
    auto flux_info = checked_buffer<std::complex<double>>(flux, "point_source_flux", 4, false);
    auto k_info = checked_buffer<double>(k_vector, "k_vector", 3, false);
    auto n_info = checked_buffer<double>(inverse_n, "inverse_n", 2, false);
    auto src_info = checked_buffer<double>(source_ra_dec, "point_source_ra_dec", 3, false);
    auto pt_info = checked_buffer<double>(pointing_ra_dec, "pointing_ra_dec", 3, false);
    auto map_info = checked_buffer<std::int64_t>(beam_model_map, "beam_model_map", 1, false);
    auto pa_info = checked_buffer<double>(parallactic_angle, "parallactic_angle", 1, false);
    auto mu_info = checked_buffer<std::int64_t>(mueller_selection, "mueller_selection", 1, false);

    const std::int64_t n_time = vis_info.shape[0];
    const std::int64_t n_baseline = vis_info.shape[1];
    const std::int64_t n_frequency = vis_info.shape[2];
    const std::int64_t n_polarization = vis_info.shape[3];
    const std::int64_t n_source = src_info.shape[1];
    const std::int64_t n_antenna = map_info.shape[0];

    auto require = [](bool ok, const char* msg) {
        if (!ok) throw std::runtime_error(msg);
    };
    require(uvw_info.shape[0] == n_time && uvw_info.shape[1] == n_baseline && uvw_info.shape[2] == 3,
            "uvw must have shape [n_time, n_baseline, 3]");
    require(a1_info.shape[0] == n_baseline && a2_info.shape[0] == n_baseline,
            "antenna1/antenna2 must have shape [n_baseline]");
    require(freq_info.shape[0] == n_frequency, "frequency must have shape [n_frequency]");
    require(pol_info.shape[0] == n_polarization, "polarization_index must have shape [n_polarization]");
    require(flux_info.shape[0] == n_source && flux_info.shape[3] == 4 &&
                (flux_info.shape[1] == 1 || flux_info.shape[1] == n_time) &&
                (flux_info.shape[2] == 1 || flux_info.shape[2] == n_frequency),
            "point_source_flux must have shape [n_source, n_time|1, n_frequency|1, 4]");
    require(k_info.shape[0] == n_time && k_info.shape[1] == n_source && k_info.shape[2] == 3,
            "k_vector must have shape [n_time, n_source, 3]");
    require(n_info.shape[0] == n_time && n_info.shape[1] == n_source, "inverse_n must have shape [n_time, n_source]");
    require((src_info.shape[0] == 1 || src_info.shape[0] == n_time) && src_info.shape[2] == 2,
            "point_source_ra_dec must have shape [n_time|1, n_source, 2]");
    require((pt_info.shape[0] == 1 || pt_info.shape[0] == n_time) &&
                (pt_info.shape[1] == 1 || pt_info.shape[1] == n_antenna) && pt_info.shape[2] == 2,
            "pointing_ra_dec must have shape [n_time|1, n_antenna|1, 2]");
    require(pa_info.shape[0] == n_time, "parallactic_angle must have shape [n_time]");
    for (std::int64_t p = 0; p < n_polarization; ++p) {
        const std::int64_t v = ptr_of<std::int64_t>(pol_info)[p];
        require(v >= 0 && v < 4, "polarization_index values must be in 0..3");
    }
    for (std::int64_t i = 0; i < mu_info.shape[0]; ++i) {
        const std::int64_t v = ptr_of<std::int64_t>(mu_info)[i];
        require(v >= 0 && v < 16, "mueller_selection values must be in 0..15");
    }
    for (std::int64_t b = 0; b < n_baseline; ++b) {
        require(ptr_of<std::int64_t>(a1_info)[b] >= 0 && ptr_of<std::int64_t>(a1_info)[b] < n_antenna &&
                    ptr_of<std::int64_t>(a2_info)[b] >= 0 && ptr_of<std::int64_t>(a2_info)[b] < n_antenna,
                "antenna indices must be in 0..n_antenna-1");
    }

    std::vector<py::buffer_info> keep;  // keeps beam-model buffers alive during compute
    std::vector<BeamModel> models;
    for (auto item : beam_models) models.push_back(beam_model_from_dict(py::cast<py::dict>(item), keep));
    for (std::int64_t a = 0; a < n_antenna; ++a) {
        const std::int64_t v = ptr_of<std::int64_t>(map_info)[a];
        require(v >= 0 && v < static_cast<std::int64_t>(models.size()), "beam_model_map must index beam_models");
    }

    auto* vis_ptr = static_cast<std::complex<double>*>(vis_info.ptr);
    {
        py::gil_scoped_release release;
        visibility_kernel::calculate_visibilities(
            vis_ptr, ptr_of<double>(uvw_info), ptr_of<std::int64_t>(a1_info), ptr_of<std::int64_t>(a2_info),
            ptr_of<double>(freq_info), ptr_of<std::int64_t>(pol_info), ptr_of<std::complex<double>>(flux_info),
            ptr_of<double>(k_info), ptr_of<double>(n_info), ptr_of<double>(src_info), ptr_of<double>(pt_info),
            ptr_of<std::int64_t>(map_info), models, ptr_of<double>(pa_info), ptr_of<std::int64_t>(mu_info), n_time,
            n_baseline, n_frequency, n_polarization, n_source, flux_info.shape[1], flux_info.shape[2],
            src_info.shape[0], pt_info.shape[0], pt_info.shape[1], n_antenna, mu_info.shape[0], n_threads);
    }
}

}  // namespace

PYBIND11_MODULE(_visibility_kernel_ext, m) {
    m.doc() = "Point-source visibility DFT kernel with direction-dependent antenna beams";
    m.def("calculate_visibilities", &calculate_visibilities_py, py::arg("visibility"), py::arg("uvw"),
          py::arg("antenna1"), py::arg("antenna2"), py::arg("frequency"), py::arg("polarization_index"),
          py::arg("point_source_flux"), py::arg("k_vector"), py::arg("inverse_n"), py::arg("point_source_ra_dec"),
          py::arg("pointing_ra_dec"), py::arg("beam_model_map"), py::arg("beam_models"),
          py::arg("parallactic_angle"), py::arg("mueller_selection"), py::arg("processing_function_threads") = 1,
          "Accumulate simulated visibilities into `visibility` in place (complex128, C-contiguous).");
    m.def("bessel_j1", &visibility_kernel::bessel_j1, py::arg("x"),
          "Bessel J1 (Cephes algorithm, identical to scipy.special.j1).");
}
