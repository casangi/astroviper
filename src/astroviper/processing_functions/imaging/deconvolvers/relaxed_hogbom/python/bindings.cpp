#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "relaxed_hogbom.hpp"

namespace py = pybind11;

template<typename T>
py::buffer_info validate_cube(py::array& array, const char* name,
                              int nt, int nf, int npol, int ny, int nx,
                              bool writable) {
    if (!array.dtype().is(py::dtype::of<T>()))
        throw std::runtime_error(std::string(name) + " has wrong dtype");
    if (!(array.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    if (writable && !array.writeable())
        throw std::runtime_error(std::string(name) + " must be writeable");
    auto info = array.request(writable);
    if (info.ndim != 5)
        throw std::runtime_error(std::string(name) + " must be a 5D array");
    if (info.shape[0] != nt || info.shape[1] != nf || info.shape[2] != npol ||
        info.shape[3] != ny || info.shape[4] != nx)
        throw std::runtime_error(std::string(name) + " has wrong shape");
    return info;
}

template<typename T>
py::buffer_info validate_planes(py::array& array, const char* name,
                                int nt, int nf, int npol) {
    if (!array.dtype().is(py::dtype::of<T>()))
        throw std::runtime_error(std::string(name) + " has wrong dtype");
    if (!(array.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    auto info = array.request(false);
    if (info.ndim != 3 || info.shape[0] != nt || info.shape[1] != nf ||
        info.shape[2] != npol)
        throw std::runtime_error(std::string(name) + " must have shape (nt, nf, npol)");
    return info;
}

template<typename T>
py::dict clean_cube_impl(
    py::array residual_cube, py::array psf_cube, py::array model_cube,
    py::array mask_cube, py::tuple clean_box, py::array max_iter_remaining,
    double gain, py::array threshold, py::array relaxation_active,
    int processing_function_threads,
    int max_iter_divergence, double divergence_rms_factor,
    double divergence_peak_factor, double constraint_radius_beams,
    int max_active_components, double minimum_improvement, bool adaptive_start,
    double rapid_progress_rms_ratio, double stalled_progress_rms_ratio,
    int rms_normalization_iterations, int optimizer_max_iterations) {
    if (residual_cube.ndim() != 5)
        throw std::runtime_error("residual_cube must be a 5D array");
    const int nt = static_cast<int>(residual_cube.shape(0));
    const int nf = static_cast<int>(residual_cube.shape(1));
    const int np_img = static_cast<int>(residual_cube.shape(2));
    const int ny = static_cast<int>(residual_cube.shape(3));
    const int nx = static_cast<int>(residual_cube.shape(4));
    auto residual_info = validate_cube<T>(residual_cube, "residual_cube", nt, nf, np_img, ny, nx, true);
    auto model_info = validate_cube<T>(model_cube, "model_cube", nt, nf, np_img, ny, nx, true);
    if (psf_cube.ndim() != 5) throw std::runtime_error("psf_cube must be a 5D array");
    const int np_psf = static_cast<int>(psf_cube.shape(2));
    if (np_psf != 1 && np_psf != np_img)
        throw std::runtime_error("psf_cube polarization axis must equal the image or be 1");
    auto psf_info = validate_cube<T>(psf_cube, "psf_cube", nt, nf, np_psf, ny, nx, false);

    int domask = 0;
    const bool* mask = nullptr;
    if (mask_cube.size() > 0) {
        auto mask_info = validate_cube<bool>(mask_cube, "mask_cube", nt, nf, np_img, ny, nx, false);
        domask = 1;
        mask = static_cast<const bool*>(mask_info.ptr);
    }
    auto iter_info = validate_planes<int>(max_iter_remaining, "max_iter_remaining", nt, nf, np_img);
    auto threshold_info = validate_planes<T>(threshold, "threshold", nt, nf, np_img);
    auto active_info = validate_planes<bool>(
        relaxation_active, "relaxation_active", nt, nf, np_img);

    int xbeg = 0, xend = nx, ybeg = 0, yend = ny;
    if (clean_box.size() == 4) {
        xbeg = py::cast<int>(clean_box[0]); xend = py::cast<int>(clean_box[1]);
        ybeg = py::cast<int>(clean_box[2]); yend = py::cast<int>(clean_box[3]);
        if (xbeg == -1) xbeg = 0; if (xend == -1) xend = nx;
        if (ybeg == -1) ybeg = 0; if (yend == -1) yend = ny;
        xbeg = std::clamp(xbeg, 0, nx - 1); xend = std::clamp(xend, xbeg + 1, nx);
        ybeg = std::clamp(ybeg, 0, ny - 1); yend = std::clamp(yend, ybeg + 1, ny);
    }

    const int nplanes = nt * nf * np_img;
    std::vector<int> iterations(nplanes), stop_reason(nplanes), component_count(nplanes);
    std::unique_ptr<bool[]> attempted(new bool[nplanes]{});
    std::unique_ptr<bool[]> accepted(new bool[nplanes]{});
    std::vector<double> initial_norm(nplanes), final_norm(nplanes), rms_ratio(nplanes);
    hclean::DivergenceControl<T> divergence;
    divergence.max_iter_divergence = max_iter_divergence;
    divergence.rms_factor = divergence_rms_factor;
    divergence.peak_factor = static_cast<T>(divergence_peak_factor);
    relaxed_hogbom::Controls controls;
    controls.constraint_radius_beams = constraint_radius_beams;
    controls.max_active_components = max_active_components;
    controls.minimum_improvement = minimum_improvement;
    controls.adaptive_start = adaptive_start;
    controls.rapid_progress_rms_ratio = rapid_progress_rms_ratio;
    controls.stalled_progress_rms_ratio = stalled_progress_rms_ratio;
    controls.rms_normalization_iterations = rms_normalization_iterations;
    controls.optimizer_max_iterations = optimizer_max_iterations;
    {
        py::gil_scoped_release release;
        relaxed_hogbom::clean_cube(
            static_cast<T*>(residual_info.ptr), static_cast<T*>(model_info.ptr),
            static_cast<const T*>(psf_info.ptr), domask, mask,
            nt, nf, np_img, np_psf, ny, nx, xbeg, xend, ybeg, yend,
            static_cast<const int*>(iter_info.ptr), static_cast<T>(gain),
            static_cast<const T*>(threshold_info.ptr), processing_function_threads,
            divergence, controls, static_cast<const bool*>(active_info.ptr),
            iterations.data(), stop_reason.data(),
            attempted.get(), accepted.get(), component_count.data(),
            initial_norm.data(), final_norm.data(), rms_ratio.data());
    }

    py::array_t<int> iterations_array({nt, nf, np_img});
    py::array_t<int> stop_array({nt, nf, np_img});
    py::array_t<bool> diverged_array({nt, nf, np_img});
    py::array_t<bool> attempted_array({nt, nf, np_img});
    py::array_t<bool> accepted_array({nt, nf, np_img});
    py::array_t<int> component_array({nt, nf, np_img});
    py::array_t<double> initial_array({nt, nf, np_img});
    py::array_t<double> final_array({nt, nf, np_img});
    py::array_t<double> ratio_array({nt, nf, np_img});
    std::memcpy(iterations_array.mutable_data(), iterations.data(), sizeof(int) * nplanes);
    std::memcpy(stop_array.mutable_data(), stop_reason.data(), sizeof(int) * nplanes);
    std::memcpy(component_array.mutable_data(), component_count.data(), sizeof(int) * nplanes);
    std::memcpy(initial_array.mutable_data(), initial_norm.data(), sizeof(double) * nplanes);
    std::memcpy(final_array.mutable_data(), final_norm.data(), sizeof(double) * nplanes);
    std::memcpy(ratio_array.mutable_data(), rms_ratio.data(), sizeof(double) * nplanes);
    for (int i = 0; i < nplanes; ++i) {
        diverged_array.mutable_data()[i] = stop_reason[i] == hclean::STOP_DIVERGED;
        attempted_array.mutable_data()[i] = attempted[i];
        accepted_array.mutable_data()[i] = accepted[i];
    }
    py::dict output;
    output["iterations_performed"] = iterations_array;
    output["stop_code"] = stop_array;
    output["diverged"] = diverged_array;
    output["relaxation_attempted"] = attempted_array;
    output["relaxation_accepted"] = accepted_array;
    output["relaxed_component_count"] = component_array;
    output["initial_residual_norm"] = initial_array;
    output["final_residual_norm"] = final_array;
    output["hogbom_rms_ratio"] = ratio_array;
    return output;
}

py::dict clean_cube_dispatch(py::array residual_cube, py::array psf_cube,
    py::array model_cube, py::array mask_cube, py::tuple clean_box,
    py::array max_iter_remaining, double gain, py::array threshold,
    py::array relaxation_active, int processing_function_threads,
    int max_iter_divergence,
    double divergence_rms_factor, double divergence_peak_factor,
    double constraint_radius_beams, int max_active_components,
    double minimum_improvement, bool adaptive_start,
    double rapid_progress_rms_ratio, double stalled_progress_rms_ratio,
    int rms_normalization_iterations, int optimizer_max_iterations) {
    if (residual_cube.dtype().is(py::dtype::of<float>()))
        return clean_cube_impl<float>(residual_cube, psf_cube, model_cube, mask_cube,
            clean_box, max_iter_remaining, gain, threshold, relaxation_active,
            processing_function_threads,
            max_iter_divergence, divergence_rms_factor, divergence_peak_factor,
            constraint_radius_beams, max_active_components, minimum_improvement,
            adaptive_start, rapid_progress_rms_ratio, stalled_progress_rms_ratio,
            rms_normalization_iterations, optimizer_max_iterations);
    if (residual_cube.dtype().is(py::dtype::of<double>()))
        return clean_cube_impl<double>(residual_cube, psf_cube, model_cube, mask_cube,
            clean_box, max_iter_remaining, gain, threshold, relaxation_active,
            processing_function_threads,
            max_iter_divergence, divergence_rms_factor, divergence_peak_factor,
            constraint_radius_beams, max_active_components, minimum_improvement,
            adaptive_start, rapid_progress_rms_ratio, stalled_progress_rms_ratio,
            rms_normalization_iterations, optimizer_max_iterations);
    throw std::runtime_error("residual_cube must be float32 or float64");
}

py::array_t<double> convolve_components_helper(py::array_t<double> model,
                                                py::array_t<double> psf) {
    auto model_info = model.request(false);
    auto psf_info = psf.request(false);
    if (model_info.ndim != 2 || psf_info.ndim != 2 ||
        model_info.shape[0] != psf_info.shape[0] ||
        model_info.shape[1] != psf_info.shape[1])
        throw std::runtime_error("model and psf must have the same 2D shape");
    if (!(model.flags() & py::array::c_style) ||
        !(psf.flags() & py::array::c_style))
        throw std::runtime_error("model and psf must be C-contiguous");
    const int ny = static_cast<int>(model_info.shape[0]);
    const int nx = static_cast<int>(model_info.shape[1]);
    py::array_t<double> output({ny, nx});
    relaxed_hogbom::convolve_components(
        static_cast<const double*>(model_info.ptr),
        static_cast<const double*>(psf_info.ptr), output.mutable_data(), nx, ny);
    return output;
}

PYBIND11_MODULE(_relaxed_hogbom_ext, module) {
    module.doc() = "Hogbom CLEAN followed by local point-component relaxation";
    module.def("clean_cube", &clean_cube_dispatch,
        py::arg("residual_cube"), py::arg("psf_cube"), py::arg("model_cube"),
        py::arg("mask_cube") = py::array(),
        py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1),
        py::arg("max_iter_remaining") = py::array(), py::arg("gain") = 0.1,
        py::arg("threshold") = py::array(),
        py::arg("relaxation_active") = py::array(),
        py::arg("processing_function_threads") = 1,
        py::arg("max_iter_divergence") = -1,
        py::arg("divergence_rms_factor") = 1.0,
        py::arg("divergence_peak_factor") = 1.0,
        py::arg("constraint_radius_beams") = 0.5,
        py::arg("max_active_components") = 100,
        py::arg("minimum_improvement") = 0.001,
        py::arg("adaptive_start") = true,
        py::arg("rapid_progress_rms_ratio") = 0.25,
        py::arg("stalled_progress_rms_ratio") = 0.78,
        py::arg("rms_normalization_iterations") = 100,
        py::arg("optimizer_max_iterations") = 100);
    module.def("_prefer_sparse_convolution",
               &relaxed_hogbom::prefer_sparse_convolution,
               py::arg("image_pixels"), py::arg("occupied_pixels"));
    module.def("_convolve_components", &convolve_components_helper,
               py::arg("model"), py::arg("psf"));
}
