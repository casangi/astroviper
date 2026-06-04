#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/asp_clean.hpp"

namespace py = pybind11;

// Validate that `arr` has dtype T, ndim dimensions, the expected shape and is
// C-contiguous. If `writable`, also require it writeable. Returns buffer_info
// so the caller can take the raw pointer with no copy.
template <typename T>
static py::buffer_info check_array(py::array& arr, const char* name, int ndim,
                                   const std::vector<py::ssize_t>& shape,
                                   bool writable) {
    if (!arr.dtype().is(py::dtype::of<T>()))
        throw std::runtime_error(std::string(name) + " has wrong dtype; expected " +
                                 py::cast<std::string>(py::dtype::of<T>().attr("name")));
    if (!(arr.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) +
                                 " must be C-contiguous (no copy is made)");
    if (writable && !arr.writeable())
        throw std::runtime_error(std::string(name) +
                                 " must be writeable (modified in place)");
    py::buffer_info info = arr.request(writable);
    if (info.ndim != ndim)
        throw std::runtime_error(std::string(name) + " must be a " +
                                 std::to_string(ndim) + "-D array");
    for (int d = 0; d < ndim; ++d)
        if (info.shape[d] != shape[d])
            throw std::runtime_error(std::string(name) + " has an unexpected shape");
    return info;
}

template <typename T>
static py::dict plane_impl(py::array residual, py::array psf, py::array model,
                           py::array mask, double gain, double threshold,
                           int niter, double fusedthreshold, double psf_width,
                           int largestscale, int stoppointmode, int norm_method,
                           bool verbose) {
    const int ny = static_cast<int>(residual.shape(0));
    const int nx = static_cast<int>(residual.shape(1));
    const std::vector<py::ssize_t> shp = {ny, nx};

    py::buffer_info ri = check_array<T>(residual, "residual", 2, shp, true);
    py::buffer_info mi = check_array<T>(model, "model", 2, shp, true);
    py::buffer_info pi = check_array<T>(psf, "psf", 2, shp, false);

    const T* mask_ptr = nullptr;
    py::buffer_info ki;
    if (mask.size() > 0) {
        ki = check_array<T>(mask, "mask", 2, shp, false);
        mask_ptr = static_cast<const T*>(ki.ptr);
    }

    aspclean::AspResult res;
    {
        py::gil_scoped_release release;  // C++ touches no Python objects
        res = aspclean::aspclean_plane<T>(
            static_cast<T*>(ri.ptr), static_cast<T*>(mi.ptr),
            static_cast<const T*>(pi.ptr), mask_ptr, nx, ny, gain, threshold,
            niter, fusedthreshold, psf_width, largestscale, stoppointmode,
            norm_method, verbose);
    }

    py::dict out;
    out["iterations_performed"] = res.iterations;
    out["retval"] = res.retval;
    out["converged"] = (res.retval == 1);
    out["peak_residual"] = res.peak_residual;
    out["model_flux"] = res.model_flux;
    out["switched_to_hogbom"] = res.switched_to_hogbom;
    out["psf_width"] = res.psf_width;
    return out;
}

static py::dict clean_dispatch(py::array residual, py::array psf, py::array model,
                               py::array mask, double gain, double threshold,
                               int niter, double fusedthreshold, double psf_width,
                               int largestscale, int stoppointmode,
                               int norm_method, bool verbose) {
    if (residual.ndim() != 2)
        throw std::runtime_error("residual must be a 2D array [ny, nx]");
    auto dt = residual.dtype();
    if (dt.is(py::dtype::of<float>()))
        return plane_impl<float>(residual, psf, model, mask, gain, threshold,
                                 niter, fusedthreshold, psf_width, largestscale,
                                 stoppointmode, norm_method, verbose);
    if (dt.is(py::dtype::of<double>()))
        return plane_impl<double>(residual, psf, model, mask, gain, threshold,
                                  niter, fusedthreshold, psf_width, largestscale,
                                  stoppointmode, norm_method, verbose);
    throw std::runtime_error("residual must be float32 or float64");
}

template <typename T>
static py::dict cube_impl(py::array residual, py::array psf, py::array model,
                          py::array mask, double gain, py::array threshold,
                          py::array niter, double fusedthreshold, double psf_width,
                          int largestscale, int stoppointmode, int norm_method,
                          int num_threads) {
    const int nt = static_cast<int>(residual.shape(0));
    const int nf = static_cast<int>(residual.shape(1));
    const int np_img = static_cast<int>(residual.shape(2));
    const int ny = static_cast<int>(residual.shape(3));
    const int nx = static_cast<int>(residual.shape(4));
    const std::vector<py::ssize_t> shp = {nt, nf, np_img, ny, nx};

    py::buffer_info ri = check_array<T>(residual, "residual", 5, shp, true);
    py::buffer_info mi = check_array<T>(model, "model", 5, shp, true);

    if (!psf.dtype().is(py::dtype::of<T>()))
        throw std::runtime_error("psf has wrong dtype");
    if (psf.ndim() != 5)
        throw std::runtime_error("psf must be a 5D array");
    if (psf.shape(0) != nt || psf.shape(1) != nf || psf.shape(3) != ny ||
        psf.shape(4) != nx)
        throw std::runtime_error("psf must match residual on (t, f, y, x)");
    const int np_psf = static_cast<int>(psf.shape(2));
    if (np_psf != np_img && np_psf != 1)
        throw std::runtime_error(
            "psf polarization axis must equal residual's or be 1 (broadcast)");
    const std::vector<py::ssize_t> pshp = {nt, nf, np_psf, ny, nx};
    py::buffer_info pi = check_array<T>(psf, "psf", 5, pshp, false);

    const T* mask_ptr = nullptr;
    py::buffer_info ki;
    if (mask.size() > 0) {
        ki = check_array<T>(mask, "mask", 5, shp, false);
        mask_ptr = static_cast<const T*>(ki.ptr);
    }

    // Per-plane iteration control: threshold (float64) and niter (int32) are
    // (nt, nf, np_img) arrays so every (time, frequency, polarization) plane
    // is cleaned with its own threshold and iteration limit.
    const std::vector<py::ssize_t> cshp = {nt, nf, np_img};
    py::buffer_info thr_info =
        check_array<double>(threshold, "threshold", 3, cshp, false);
    py::buffer_info nit_info =
        check_array<int>(niter, "niter", 3, cshp, false);
    const double* thr_ptr = static_cast<const double*>(thr_info.ptr);
    const int* nit_ptr = static_cast<const int*>(nit_info.ptr);

    const int nplanes = nt * nf * np_img;
    std::vector<aspclean::AspResult> results(nplanes);
    {
        py::gil_scoped_release release;
        aspclean::aspclean_cube<T>(
            static_cast<T*>(ri.ptr), static_cast<T*>(mi.ptr),
            static_cast<const T*>(pi.ptr), mask_ptr, nt, nf, np_img, np_psf, nx,
            ny, gain, thr_ptr, nit_ptr, fusedthreshold, psf_width, largestscale,
            stoppointmode, norm_method, num_threads, results.data());
    }

    py::array_t<int> iters({nt, nf, np_img});
    py::array_t<int> retval({nt, nf, np_img});
    py::array_t<bool> converged({nt, nf, np_img});
    py::array_t<double> peak({nt, nf, np_img});
    py::array_t<double> flux({nt, nf, np_img});
    int* ip = iters.mutable_data();
    int* rp = retval.mutable_data();
    bool* cp = converged.mutable_data();
    double* pp = peak.mutable_data();
    double* fp = flux.mutable_data();
    for (int i = 0; i < nplanes; ++i) {
        ip[i] = results[i].iterations;
        rp[i] = results[i].retval;
        cp[i] = (results[i].retval == 1);
        pp[i] = results[i].peak_residual;
        fp[i] = results[i].model_flux;
    }

    py::dict out;
    out["iterations_performed"] = iters;
    out["retval"] = retval;
    out["converged"] = converged;
    out["peak_residual"] = peak;
    out["model_flux"] = flux;
    return out;
}

static py::dict clean_cube_dispatch(py::array residual, py::array psf,
                                    py::array model, py::array mask, double gain,
                                    py::array threshold, py::array niter,
                                    double fusedthreshold, double psf_width,
                                    int largestscale, int stoppointmode,
                                    int norm_method, int num_threads) {
    if (residual.ndim() != 5)
        throw std::runtime_error(
            "residual must be a 5D array [time, frequency, polarization, y, x]");
    auto dt = residual.dtype();
    if (dt.is(py::dtype::of<float>()))
        return cube_impl<float>(residual, psf, model, mask, gain, threshold,
                                niter, fusedthreshold, psf_width, largestscale,
                                stoppointmode, norm_method, num_threads);
    if (dt.is(py::dtype::of<double>()))
        return cube_impl<double>(residual, psf, model, mask, gain, threshold,
                                 niter, fusedthreshold, psf_width, largestscale,
                                 stoppointmode, norm_method, num_threads);
    throw std::runtime_error("residual must be float32 or float64");
}

// Test helper: PSF Gaussian width estimate (mean FWHM in pixels).
static double psf_width_helper(
    py::array_t<double, py::array::c_style | py::array::forcecast> psf) {
    py::buffer_info info = psf.request();
    if (info.ndim != 2) throw std::runtime_error("psf must be a 2D array");
    return aspclean::psf_gaussian_width(static_cast<const double*>(info.ptr),
                                        static_cast<int>(info.shape[1]),
                                        static_cast<int>(info.shape[0]));
}

// Test helper: centred FFT-based circular convolution of two 2D images.
static py::array_t<double> convolve_helper(
    py::array_t<double, py::array::c_style | py::array::forcecast> a,
    py::array_t<double, py::array::c_style | py::array::forcecast> b) {
    py::buffer_info ai = a.request(), bi = b.request();
    if (ai.ndim != 2 || bi.ndim != 2)
        throw std::runtime_error("inputs must be 2D arrays");
    if (ai.shape[0] != bi.shape[0] || ai.shape[1] != bi.shape[1])
        throw std::runtime_error("inputs must have the same shape");
    const int ny = static_cast<int>(ai.shape[0]);
    const int nx = static_cast<int>(ai.shape[1]);
    py::array_t<double> out({ny, nx});
    aspclean::convolve_centered(static_cast<const double*>(ai.ptr),
                                static_cast<const double*>(bi.ptr), nx, ny,
                                out.mutable_data());
    return out;
}

PYBIND11_MODULE(_aspclean_ext, m) {
    m.doc() =
        "Adaptive Scale Pixel (Asp / AAspClean) deconvolution - dependency-free "
        "port of CASA's AspMatrixCleaner, operating in place on Python-owned "
        "numpy buffers with no copies.";

    m.def("clean", &clean_dispatch,
          "Run the Asp minor cycle on a single 2D plane in place. `residual` and "
          "`model` (float32 or float64, C-contiguous, writeable) are modified: "
          "the residual replaces the dirty image and Asp/Hogbom components are "
          "accumulated into the model. `psf` and optional `mask` share the dtype "
          "and shape. psf_width <= 0 estimates the PSF width internally.",
          py::arg("residual"), py::arg("psf"), py::arg("model"),
          py::arg("mask") = py::array(), py::arg("gain") = 0.1,
          py::arg("threshold") = 0.0, py::arg("niter") = 100,
          py::arg("fusedthreshold") = 0.0, py::arg("psf_width") = 0.0,
          py::arg("largestscale") = -1, py::arg("stoppointmode") = -1,
          py::arg("norm_method") = 1, py::arg("verbose") = false);

    m.def("clean_cube", &clean_cube_dispatch,
          "Run the Asp minor cycle independently on every (time, frequency, "
          "polarization) plane of a 5D cube in place, across num_threads worker "
          "threads. The PSF cube may be single-polarization (broadcast). "
          "Iteration control is independent per plane: threshold (float64) and "
          "niter (int32) are (nt, nf, np) arrays giving each plane its own "
          "threshold and iteration limit. Returns per-plane arrays of shape "
          "(nt, nf, np).",
          py::arg("residual"), py::arg("psf"), py::arg("model"),
          py::arg("mask") = py::array(), py::arg("gain") = 0.1,
          py::arg("threshold") = py::array(), py::arg("niter") = py::array(),
          py::arg("fusedthreshold") = 0.0, py::arg("psf_width") = 0.0,
          py::arg("largestscale") = -1, py::arg("stoppointmode") = -1,
          py::arg("norm_method") = 1, py::arg("num_threads") = 1);

    m.def("psf_gaussian_width", &psf_width_helper,
          "Estimate the PSF Gaussian width (mean FWHM in pixels) used to seed "
          "the Asp initial scale sizes.",
          py::arg("psf"));

    m.def("convolve_centered", &convolve_helper,
          "Centred FFT-based circular convolution of two equally-shaped 2D "
          "images (the convolution primitive used internally by the cleaner).",
          py::arg("a"), py::arg("b"));
}
