#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/asp_clean.hpp"
#include "../include/asp_lbfgs.hpp"

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
                           py::array mask, double gain, double threshold_per_cycle,
                           int max_iter_per_cycle_updated, double fusedthreshold, double psf_width,
                           int largestscale, int stoppointmode, int norm_method,
                           bool verbose, int max_iter_divergence,
                           double divergence_rms_factor, double divergence_peak_factor,
                           int processing_function_threads, bool stop_after_hogbom_run) {
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

    deconvolvers::DivergenceControl<double> divergence;
    divergence.max_iter_divergence = max_iter_divergence;
    divergence.rms_factor = divergence_rms_factor;
    divergence.peak_factor = divergence_peak_factor;
    aspclean::AspResult res;
    {
        py::gil_scoped_release release;  // C++ touches no Python objects
        res = aspclean::aspclean_plane<T>(
            static_cast<T*>(ri.ptr), static_cast<T*>(mi.ptr),
            static_cast<const T*>(pi.ptr), mask_ptr, nx, ny, gain, threshold_per_cycle,
            max_iter_per_cycle_updated, fusedthreshold, psf_width, largestscale, stoppointmode,
            norm_method, stop_after_hogbom_run, verbose, divergence, processing_function_threads);
    }

    py::dict out;
    out["iterations_performed"] = res.iterations;
    out["retval"] = res.retval;
    out["stop_code"] = res.stop_code;
    out["converged"] = (res.retval == 1);
    out["diverged"] = (res.retval == -3);
    out["peak_residual"] = res.peak_residual;
    out["model_flux"] = res.model_flux;
    out["switched_to_hogbom"] = res.switched_to_hogbom;
    out["psf_width"] = res.psf_width;
    py::dict timing;
    timing["setup"] = res.t_setup;
    timing["search"] = res.t_search;
    timing["fit"] = res.t_fit;
    timing["update_scale"] = res.t_update_scale;
    timing["update_point"] = res.t_update_point;
    timing["bookkeeping"] = res.t_bookkeeping;
    timing["n_scale_components"] = res.n_scale_components;
    timing["n_point_components"] = res.n_point_components;
    timing["n_fits"] = res.n_fits;
    timing["n_fit_evaluations"] = res.n_fit_evaluations;
    out["timing"] = timing;
    return out;
}

static py::dict clean_dispatch(py::array residual, py::array psf, py::array model,
                               py::array mask, double gain, double threshold_per_cycle,
                               int max_iter_per_cycle_updated, double fusedthreshold, double psf_width,
                               int largestscale, int stoppointmode,
                               int norm_method, bool verbose, int max_iter_divergence,
                               double divergence_rms_factor,
                               double divergence_peak_factor,
                               int processing_function_threads, bool stop_after_hogbom_run) {
    if (residual.ndim() != 2)
        throw std::runtime_error("residual must be a 2D array [ny, nx]");
    auto dt = residual.dtype();
    if (dt.is(py::dtype::of<float>()))
        return plane_impl<float>(residual, psf, model, mask, gain, threshold_per_cycle,
                                 max_iter_per_cycle_updated, fusedthreshold, psf_width, largestscale,
                                 stoppointmode, norm_method, verbose, max_iter_divergence,
                                 divergence_rms_factor, divergence_peak_factor,
                                 processing_function_threads, stop_after_hogbom_run);
    if (dt.is(py::dtype::of<double>()))
        return plane_impl<double>(residual, psf, model, mask, gain, threshold_per_cycle,
                                  max_iter_per_cycle_updated, fusedthreshold, psf_width, largestscale,
                                  stoppointmode, norm_method, verbose, max_iter_divergence,
                                  divergence_rms_factor, divergence_peak_factor,
                                  processing_function_threads, stop_after_hogbom_run);
    throw std::runtime_error("residual must be float32 or float64");
}

template <typename T>
static py::dict cube_impl(py::array residual, py::array psf, py::array model,
                          py::array mask, double gain, py::array threshold_per_cycle,
                          py::array max_iter_per_cycle_updated, double fusedthreshold, double psf_width,
                          int largestscale, int stoppointmode, int norm_method,
                          int processing_function_threads, int max_iter_divergence,
                          double divergence_rms_factor, double divergence_peak_factor,
                          bool stop_after_hogbom_run) {
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

    // Per-plane iteration control: threshold_per_cycle (float64) and
    // max_iter_per_cycle_updated (int32) are (nt, nf, np_img) arrays so every
    // (time, frequency, polarization) plane is cleaned with its own threshold
    // and iteration limit.
    const std::vector<py::ssize_t> cshp = {nt, nf, np_img};
    py::buffer_info thr_info =
        check_array<double>(threshold_per_cycle, "threshold_per_cycle", 3, cshp, false);
    py::buffer_info nit_info =
        check_array<int>(max_iter_per_cycle_updated, "max_iter_per_cycle_updated", 3, cshp, false);
    const double* thr_ptr = static_cast<const double*>(thr_info.ptr);
    const int* nit_ptr = static_cast<const int*>(nit_info.ptr);

    const int nplanes = nt * nf * np_img;
    std::vector<aspclean::AspResult> results(nplanes);
    deconvolvers::DivergenceControl<double> divergence;
    divergence.max_iter_divergence = max_iter_divergence;
    divergence.rms_factor = divergence_rms_factor;
    divergence.peak_factor = divergence_peak_factor;
    {
        py::gil_scoped_release release;
        aspclean::aspclean_cube<T>(
            static_cast<T*>(ri.ptr), static_cast<T*>(mi.ptr),
            static_cast<const T*>(pi.ptr), mask_ptr, nt, nf, np_img, np_psf, nx,
            ny, gain, thr_ptr, nit_ptr, fusedthreshold, psf_width, largestscale,
            stoppointmode, norm_method, stop_after_hogbom_run, processing_function_threads,
            divergence, results.data());
    }

    py::array_t<int> iters({nt, nf, np_img});
    py::array_t<int> retval({nt, nf, np_img});
    py::array_t<int> stop_code({nt, nf, np_img});
    py::array_t<bool> converged({nt, nf, np_img});
    py::array_t<bool> diverged({nt, nf, np_img});
    py::array_t<double> peak({nt, nf, np_img});
    py::array_t<double> flux({nt, nf, np_img});
    int* ip = iters.mutable_data();
    int* rp = retval.mutable_data();
    int* sp = stop_code.mutable_data();
    bool* cp = converged.mutable_data();
    bool* dp = diverged.mutable_data();
    double* pp = peak.mutable_data();
    double* fp = flux.mutable_data();
    for (int i = 0; i < nplanes; ++i) {
        ip[i] = results[i].iterations;
        rp[i] = results[i].retval;
        sp[i] = results[i].stop_code;
        cp[i] = (results[i].retval == 1);
        dp[i] = (results[i].retval == -3);
        pp[i] = results[i].peak_residual;
        fp[i] = results[i].model_flux;
    }

    py::dict out;
    out["iterations_performed"] = iters;
    out["retval"] = retval;
    out["stop_code"] = stop_code;
    out["converged"] = converged;
    out["diverged"] = diverged;
    out["peak_residual"] = peak;
    out["model_flux"] = flux;
    return out;
}

static py::dict clean_cube_dispatch(py::array residual, py::array psf,
                                    py::array model, py::array mask, double gain,
                                    py::array threshold_per_cycle, py::array max_iter_per_cycle_updated,
                                    double fusedthreshold, double psf_width,
                                    int largestscale, int stoppointmode,
                                    int norm_method, int processing_function_threads,
                                    int max_iter_divergence, double divergence_rms_factor,
                                    double divergence_peak_factor, bool stop_after_hogbom_run) {
    if (residual.ndim() != 5)
        throw std::runtime_error(
            "residual must be a 5D array [time, frequency, polarization, y, x]");
    auto dt = residual.dtype();
    if (dt.is(py::dtype::of<float>()))
        return cube_impl<float>(residual, psf, model, mask, gain, threshold_per_cycle,
                                max_iter_per_cycle_updated, fusedthreshold, psf_width, largestscale,
                                stoppointmode, norm_method, processing_function_threads,
                                max_iter_divergence, divergence_rms_factor,
                                divergence_peak_factor, stop_after_hogbom_run);
    if (dt.is(py::dtype::of<double>()))
        return cube_impl<double>(residual, psf, model, mask, gain, threshold_per_cycle,
                                 max_iter_per_cycle_updated, fusedthreshold, psf_width, largestscale,
                                 stoppointmode, norm_method, processing_function_threads,
                                 max_iter_divergence, divergence_rms_factor,
                                 divergence_peak_factor, stop_after_hogbom_run);
    throw std::runtime_error("residual must be float32 or float64");
}

// Test helper: PSF Gaussian width estimate (mean FWHM in pixels). The PSF is
// a full 2D image, so it is validated zero-copy (float64, C-contiguous) —
// never forcecast.
static double psf_width_helper(py::array psf) {
    if (psf.ndim() != 2) throw std::runtime_error("psf must be a 2D array");
    const std::vector<py::ssize_t> shp = {psf.shape(0), psf.shape(1)};
    py::buffer_info info = check_array<double>(psf, "psf", 2, shp, false);
    return aspclean::psf_gaussian_width(static_cast<const double*>(info.ptr),
                                        static_cast<int>(info.shape[1]),
                                        static_cast<int>(info.shape[0]));
}

// Test helper: centred FFT-based circular convolution of two 2D images. Both
// inputs are full 2D images, validated zero-copy (float64, C-contiguous) —
// never forcecast.
static py::array_t<double> convolve_helper(py::array a, py::array b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::runtime_error("inputs must be 2D arrays");
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::runtime_error("inputs must have the same shape");
    const std::vector<py::ssize_t> shp = {a.shape(0), a.shape(1)};
    py::buffer_info ai = check_array<double>(a, "a", 2, shp, false);
    py::buffer_info bi = check_array<double>(b, "b", 2, shp, false);
    const int ny = static_cast<int>(ai.shape[0]);
    const int nx = static_cast<int>(ai.shape[1]);
    py::array_t<double> out({ny, nx});
    aspclean::convolve_centered(static_cast<const double*>(ai.ptr),
                                static_cast<const double*>(bi.ptr), nx, ny,
                                out.mutable_data());
    return out;
}

// Test helper: run the deconvolver's L-BFGS on a Python objective
// ``fg(x) -> (f, grad)``. Returns (x, f, iterations, evaluations).
static py::tuple lbfgs_helper(py::function fg, std::vector<double> x, std::vector<double> scale,
                              int max_iters, double epsg, double epsf, double epsx) {
    if (x.size() != scale.size()) throw std::runtime_error("x and scale must have the same length");
    asplbfgs::Options opt;
    opt.max_iters = max_iters;
    opt.epsg = epsg;
    opt.epsf = epsf;
    opt.epsx = epsx;
    const asplbfgs::Report rep = asplbfgs::minimize(
        x, scale,
        [&fg](const std::vector<double>& xx, std::vector<double>& gg) {
            py::tuple r = fg(xx).cast<py::tuple>();
            gg = r[1].cast<std::vector<double>>();
            return r[0].cast<double>();
        },
        opt);
    return py::make_tuple(x, rep.f, rep.iterations, rep.evaluations);
}

PYBIND11_MODULE(_aspclean_ext, m) {
    m.doc() =
        "Adaptive Scale Pixel (Asp / AAspClean) deconvolution - dependency-free "
        "port of CASA's AspMatrixCleaner, operating in place on Python-owned "
        "numpy buffers with no copies.";

    m.def("clean", &clean_dispatch,
          "Run the Asp model update on a single 2D plane in place. `residual` and "
          "`model` (float32 or float64, C-contiguous, writeable) are modified: "
          "the residual replaces the dirty image and Asp/Hogbom components are "
          "accumulated into the model. `psf` and optional `mask` share the dtype "
          "and shape. psf_width <= 0 estimates the PSF width internally. The "
          "model update also stops as diverged (retval -3, diverged) on the "
          "divergence test shared with the other deconvolvers: the RMS of the "
          "masked residual above divergence_rms_factor x its lowest value for "
          "max_iter_divergence consecutive iterations, its peak above "
          "divergence_peak_factor x its starting peak, or a peak that is not "
          "finite (max_iter_divergence < 0 disables the first two tests). "
          "stop_code reports why the model update ended, as the other "
          "deconvolvers do: 0 did not run, 1 spent max_iter_per_cycle_updated, "
          "2 reached threshold_per_cycle, 4 diverged, 5 a test of Asp itself "
          "(optimum strength, end of a Hogbom run, stop point mode). "
          "processing_function_threads is the number of threads of every "
          "2-D transform. stop_after_hogbom_run ends the model update after a "
          "Hogbom run unless the RMS of the residual has fallen, by less than "
          "a factor 1.5, since its start (CASA and LibRA); off, the model "
          "update goes on to its threshold or iteration limit.",
          py::arg("residual"), py::arg("psf"), py::arg("model"),
          py::arg("mask") = py::array(), py::arg("gain") = 0.1,
          py::arg("threshold_per_cycle") = 0.0, py::arg("max_iter_per_cycle_updated") = 100,
          py::arg("fusedthreshold") = 0.0, py::arg("psf_width") = 0.0,
          py::arg("largestscale") = -1, py::arg("stoppointmode") = -1,
          py::arg("norm_method") = 1, py::arg("verbose") = false,
          py::arg("max_iter_divergence") = -1, py::arg("divergence_rms_factor") = 1.0,
          py::arg("divergence_peak_factor") = 1.0, py::arg("processing_function_threads") = 1,
          py::arg("stop_after_hogbom_run") = false);

    m.def("clean_cube", &clean_cube_dispatch,
          "Run the Asp model update independently on every (time, frequency, "
          "polarization) plane of a 5D cube in place, across processing_function_threads worker "
          "threads. The PSF cube may be single-polarization (broadcast). "
          "Iteration control is independent per plane: threshold_per_cycle "
          "(float64) and max_iter_per_cycle_updated (int32) are (nt, nf, np) "
          "arrays giving each plane its own threshold and iteration limit; the "
          "divergence test and stop_after_hogbom_run are those of clean. "
          "Returns per-plane arrays of shape (nt, nf, np), stop_code among "
          "them.",
          py::arg("residual"), py::arg("psf"), py::arg("model"),
          py::arg("mask") = py::array(), py::arg("gain") = 0.1,
          py::arg("threshold_per_cycle") = py::array(), py::arg("max_iter_per_cycle_updated") = py::array(),
          py::arg("fusedthreshold") = 0.0, py::arg("psf_width") = 0.0,
          py::arg("largestscale") = -1, py::arg("stoppointmode") = -1,
          py::arg("norm_method") = 1, py::arg("processing_function_threads") = 1,
          py::arg("max_iter_divergence") = -1, py::arg("divergence_rms_factor") = 1.0,
          py::arg("divergence_peak_factor") = 1.0, py::arg("stop_after_hogbom_run") = false);

    m.def("psf_gaussian_width", &psf_width_helper,
          "Estimate the PSF Gaussian width (mean FWHM in pixels) used to seed "
          "the Asp initial scale sizes.",
          py::arg("psf"));

    m.def("lbfgs_minimize", &lbfgs_helper,
          "Test helper: minimize a Python objective fg(x) -> (f, grad) with the "
          "deconvolver's L-BFGS (ALGLIB minlbfgs with one correction pair and a "
          "per-variable scale for the stopping tests, More-Thuente line search). "
          "Returns (x, f, iterations, evaluations).",
          py::arg("fg"), py::arg("x"), py::arg("scale"), py::arg("max_iters") = 5,
          py::arg("epsg") = 1e-3, py::arg("epsf") = 1e-3, py::arg("epsx") = 1e-3);

    m.def("convolve_centered", &convolve_helper,
          "Centred FFT-based circular convolution of two equally-shaped 2D "
          "images. The cleaner's residual update uses the linear convolution "
          "instead (no wrap-around).",
          py::arg("a"), py::arg("b"));
}
