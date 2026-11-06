// Python entries of the fast Asp model update: clean and clean_cube of the
// reference binding (deconvolvers/aspclean/python/bindings.cpp at e0c0d126),
// with its names, parameters, defaults, docstrings, checks, exceptions and
// results, calling aspclean_fast:: instead of aspclean::. The reference's test
// helpers (psf_gaussian_width, convolve_centered, lbfgs_minimize) are not
// hooked and not repeated here. Two changes, which change no value: a
// misaligned array is read (and the residual and the model written back)
// through an aligned copy (Aligned), and the components are copied only when
// there is one (memcpy from the null data() of an empty vector is undefined,
// even of 0 bytes).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "asp_clean_fast.hpp"
#include "isa.hpp"

namespace py = pybind11;

// Compare dtype values: pickle/Dask may reconstruct an equivalent NumPy dtype
// with a different descriptor object, while the underlying buffer is unchanged.

// Validate that `arr` has dtype T, ndim dimensions, the expected shape and is
// C-contiguous. If `writable`, also require it writeable. Returns buffer_info
// so the caller can take the raw pointer with no copy.
template <typename T>
static py::buffer_info check_array(py::array& arr, const char* name, int ndim,
                                   const std::vector<py::ssize_t>& shape,
                                   bool writable) {
    if (!arr.dtype().equal(py::dtype::of<T>()))
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

// The iteration history fields (record_iteration_history), with their records.
static const std::pair<const char*, std::vector<double> aspclean::AspResult::*>
    kIterationHistoryRecords[] = {
        {"iteration_peak_residual", &aspclean::AspResult::history_peak},
        {"iteration_residual_rms", &aspclean::AspResult::history_rms},
        {"iteration_component_l", &aspclean::AspResult::history_l},
        {"iteration_component_m", &aspclean::AspResult::history_m},
        {"iteration_component_scale", &aspclean::AspResult::history_scale},
        {"iteration_component_flux", &aspclean::AspResult::history_flux},
};

static py::array_t<double> history_array(const std::vector<double>& values) {
    return py::array_t<double>(static_cast<py::ssize_t>(values.size()), values.data());
}

// The data of a checked array, or an aligned copy of it when its address is
// not a multiple of alignof(T): Dask can hand a worker an array at any byte
// offset of a frame it received, and reading a T at such an address is
// undefined (the reference reads it in place, which x86-64 tolerates). The
// copy has the same bytes, so the results are the same; with `write_back` it
// goes back to the caller's buffer at the end of the call (the residual and
// the model are updated in place).
template <typename T>
class Aligned {
public:
    Aligned(const py::buffer_info& info, bool write_back)
        : caller_(info.ptr), count_(static_cast<std::size_t>(info.size)), write_back_(write_back) {
        if (count_ == 0 || reinterpret_cast<std::uintptr_t>(caller_) % alignof(T) == 0) {
            data_ = static_cast<T*>(caller_);
            return;
        }
        copy_.resize(count_);
        std::memcpy(copy_.data(), caller_, count_ * sizeof(T));
        data_ = copy_.data();
    }
    Aligned(const Aligned&) = delete;
    Aligned& operator=(const Aligned&) = delete;
    ~Aligned() {
        if (write_back_ && !copy_.empty()) std::memcpy(caller_, copy_.data(), count_ * sizeof(T));
    }
    T* get() const { return data_; }

private:
    void* caller_;
    std::size_t count_;
    bool write_back_;
    std::vector<T> copy_;
    T* data_ = nullptr;
};

template <typename T>
static py::dict plane_impl(py::array residual, py::array psf, py::array model,
                           py::array mask, double gain, double threshold_per_cycle,
                           int max_iter_per_cycle_updated, double asp_fused_threshold,
                           double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                           int asp_scale_normalization_method,
                           bool verbose, int max_iter_divergence,
                           double divergence_rms_factor, double divergence_peak_factor,
                           int processing_function_threads, bool asp_stop_after_hogbom_run,
                           const aspclean::AspOptions& options) {
    const int ny = static_cast<int>(residual.shape(0));
    const int nx = static_cast<int>(residual.shape(1));
    const std::vector<py::ssize_t> shp = {ny, nx};

    py::buffer_info ri = check_array<T>(residual, "residual", 2, shp, true);
    py::buffer_info mi = check_array<T>(model, "model", 2, shp, true);
    py::buffer_info pi = check_array<T>(psf, "psf", 2, shp, false);

    py::buffer_info ki;
    if (mask.size() > 0) ki = check_array<T>(mask, "mask", 2, shp, false);
    // No mask: a null pointer (ki holds none).
    const Aligned<T> r(ri, true), m(mi, true), p(pi, false), k(ki, false);

    deconvolvers::DivergenceControl<double> divergence;
    divergence.max_iter_divergence = max_iter_divergence;
    divergence.rms_factor = divergence_rms_factor;
    divergence.peak_factor = divergence_peak_factor;
    aspclean::AspResult res;
    {
        py::gil_scoped_release release;  // C++ touches no Python objects
        res = aspclean_fast::aspclean_plane<T>(
            r.get(), m.get(), p.get(), k.get(), nx, ny, gain, threshold_per_cycle,
            max_iter_per_cycle_updated, asp_fused_threshold, asp_psf_width, asp_largest_scale,
            asp_stop_point_mode, asp_scale_normalization_method, asp_stop_after_hogbom_run,
            verbose, divergence, processing_function_threads, options);
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
    timing["residual_components"] = res.t_residual_components;
    timing["n_residual_components"] = res.n_residual_components;
    timing["n_residual_fit_evaluations"] = res.n_residual_fit_evaluations;
    timing["significance"] = res.t_significance;
    timing["n_significance_components"] = res.n_significance_components;
    timing["n_hogbom_runs"] = res.n_hogbom_runs;
    timing["n_switch_no_change"] = res.n_switch_no_change;
    timing["n_switch_point_scales"] = res.n_switch_point_scales;
    timing["n_switch_fused"] = res.n_switch_fused;
    out["timing"] = timing;
    if (options.record_iteration_history)
        for (const auto& [name, record] : kIterationHistoryRecords) out[name] = history_array(res.*record);
    if (options.record_components) {
        const py::ssize_t n = static_cast<py::ssize_t>(res.components.size() / 4);
        py::array_t<double> components({n, static_cast<py::ssize_t>(4)});
        if (n) std::memcpy(components.mutable_data(), res.components.data(), res.components.size() * sizeof(double));
        out["components"] = components;
        const py::ssize_t m = static_cast<py::ssize_t>(res.residual_fits.size() / 6);
        py::array_t<double> fits({m, static_cast<py::ssize_t>(6)});
        if (m) std::memcpy(fits.mutable_data(), res.residual_fits.data(), res.residual_fits.size() * sizeof(double));
        out["residual_fits"] = fits;
    }
    if (options.asp_scale_significance) out["scale_noise"] = res.scale_noise;
    return out;
}

static py::dict clean_dispatch(py::array residual, py::array psf, py::array model,
                               py::array mask, double gain, double threshold_per_cycle,
                               int max_iter_per_cycle_updated, double asp_fused_threshold,
                               double asp_psf_width, int asp_largest_scale,
                               int asp_stop_point_mode, int asp_scale_normalization_method,
                               bool verbose, int max_iter_divergence,
                               double divergence_rms_factor,
                               double divergence_peak_factor,
                               int processing_function_threads, bool asp_stop_after_hogbom_run,
                               bool asp_scale_significance, double asp_significance,
                               bool record_components, bool asp_residual_components,
                               double asp_residual_component_noise_floor,
                               double asp_residual_component_min_fraction,
                               double asp_residual_component_max_epsilon_fraction,
                               double asp_residual_component_switch_factor,
                               bool asp_residual_component_floor_at_threshold,
                               bool asp_residual_component_positive_only,
                               double asp_significance_gain, double asp_no_change_tolerance,
                               bool record_iteration_history, double asp_min_mask_coverage) {
    aspclean::AspOptions options;
    options.asp_scale_significance = asp_scale_significance;
    options.asp_significance = asp_significance;
    options.record_components = record_components;
    options.asp_residual_components = asp_residual_components;
    options.asp_residual_component_noise_floor = asp_residual_component_noise_floor;
    options.asp_residual_component_min_fraction = asp_residual_component_min_fraction;
    options.asp_residual_component_max_epsilon_fraction = asp_residual_component_max_epsilon_fraction;
    options.asp_residual_component_switch_factor = asp_residual_component_switch_factor;
    options.asp_residual_component_floor_at_threshold = asp_residual_component_floor_at_threshold;
    options.asp_residual_component_positive_only = asp_residual_component_positive_only;
    options.asp_significance_gain = asp_significance_gain;
    options.asp_no_change_tolerance = asp_no_change_tolerance;
    options.record_iteration_history = record_iteration_history;
    options.asp_min_mask_coverage = asp_min_mask_coverage;
    if (residual.ndim() != 2)
        throw std::runtime_error("residual must be a 2D array [ny, nx]");
    auto dt = residual.dtype();
    if (dt.equal(py::dtype::of<float>()))
        return plane_impl<float>(residual, psf, model, mask, gain, threshold_per_cycle,
                                 max_iter_per_cycle_updated, asp_fused_threshold, asp_psf_width,
                                 asp_largest_scale, asp_stop_point_mode,
                                 asp_scale_normalization_method, verbose, max_iter_divergence,
                                 divergence_rms_factor, divergence_peak_factor,
                                 processing_function_threads, asp_stop_after_hogbom_run, options);
    if (dt.equal(py::dtype::of<double>()))
        return plane_impl<double>(residual, psf, model, mask, gain, threshold_per_cycle,
                                  max_iter_per_cycle_updated, asp_fused_threshold, asp_psf_width,
                                  asp_largest_scale, asp_stop_point_mode,
                                  asp_scale_normalization_method, verbose, max_iter_divergence,
                                  divergence_rms_factor, divergence_peak_factor,
                                  processing_function_threads, asp_stop_after_hogbom_run, options);
    throw std::runtime_error("residual must be float32 or float64");
}

template <typename T>
static py::dict cube_impl(py::array residual, py::array psf, py::array model,
                          py::array mask, double gain, py::array threshold_per_cycle,
                          py::array max_iter_per_cycle_updated, double asp_fused_threshold,
                          double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                          int asp_scale_normalization_method,
                          int processing_function_threads, int max_iter_divergence,
                          double divergence_rms_factor, double divergence_peak_factor,
                          bool asp_stop_after_hogbom_run, const aspclean::AspOptions& options) {
    const int nt = static_cast<int>(residual.shape(0));
    const int nf = static_cast<int>(residual.shape(1));
    const int np_img = static_cast<int>(residual.shape(2));
    const int ny = static_cast<int>(residual.shape(3));
    const int nx = static_cast<int>(residual.shape(4));
    const std::vector<py::ssize_t> shp = {nt, nf, np_img, ny, nx};

    py::buffer_info ri = check_array<T>(residual, "residual", 5, shp, true);
    py::buffer_info mi = check_array<T>(model, "model", 5, shp, true);

    if (!psf.dtype().equal(py::dtype::of<T>()))
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

    py::buffer_info ki;
    if (mask.size() > 0) ki = check_array<T>(mask, "mask", 5, shp, false);

    // Per-plane iteration control: threshold_per_cycle (float64) and
    // max_iter_per_cycle_updated (int32) are (nt, nf, np_img) arrays so every
    // (time, frequency, polarization) plane is cleaned with its own threshold
    // and iteration limit.
    const std::vector<py::ssize_t> cshp = {nt, nf, np_img};
    py::buffer_info thr_info =
        check_array<double>(threshold_per_cycle, "threshold_per_cycle", 3, cshp, false);
    py::buffer_info nit_info =
        check_array<int>(max_iter_per_cycle_updated, "max_iter_per_cycle_updated", 3, cshp, false);
    // No mask: a null pointer (ki holds none).
    const Aligned<T> r(ri, true), m(mi, true), p(pi, false), k(ki, false);
    const Aligned<double> thr(thr_info, false);
    const Aligned<int> nit(nit_info, false);

    const int nplanes = nt * nf * np_img;
    std::vector<aspclean::AspResult> results(nplanes);
    deconvolvers::DivergenceControl<double> divergence;
    divergence.max_iter_divergence = max_iter_divergence;
    divergence.rms_factor = divergence_rms_factor;
    divergence.peak_factor = divergence_peak_factor;
    {
        py::gil_scoped_release release;
        aspclean_fast::aspclean_cube<T>(
            r.get(), m.get(), p.get(), k.get(), nt, nf, np_img, np_psf, nx,
            ny, gain, thr.get(), nit.get(), asp_fused_threshold, asp_psf_width, asp_largest_scale,
            asp_stop_point_mode, asp_scale_normalization_method, asp_stop_after_hogbom_run,
            processing_function_threads, divergence, results.data(), options);
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
    // Components of each kind the model update of every plane subtracted.
    const std::pair<const char*, int aspclean::AspResult::*> counts[] = {
        {"n_residual_components", &aspclean::AspResult::n_residual_components},
        {"n_scale_components", &aspclean::AspResult::n_scale_components},
        {"n_point_components", &aspclean::AspResult::n_point_components},
        {"n_significance_components", &aspclean::AspResult::n_significance_components},
        {"n_hogbom_runs", &aspclean::AspResult::n_hogbom_runs},
        {"n_switch_no_change", &aspclean::AspResult::n_switch_no_change},
        {"n_switch_point_scales", &aspclean::AspResult::n_switch_point_scales},
        {"n_switch_fused", &aspclean::AspResult::n_switch_fused},
    };
    for (const auto& [name, member] : counts) {
        py::array_t<int> a({nt, nf, np_img});
        int* ap = a.mutable_data();
        for (int i = 0; i < nplanes; ++i) ap[i] = results[i].*member;
        out[name] = a;
    }
    // The iteration history: nested lists [t][f][p] of 1-D arrays, one value
    // per component of the plane's model update.
    if (options.record_iteration_history) {
        for (const auto& [name, record] : kIterationHistoryRecords) {
            py::list per_time;
            for (int tt = 0, i = 0; tt < nt; ++tt) {
                py::list per_frequency;
                for (int ff = 0; ff < nf; ++ff) {
                    py::list per_polarization;
                    for (int pp = 0; pp < np_img; ++pp, ++i)
                        per_polarization.append(history_array(results[i].*record));
                    per_frequency.append(per_polarization);
                }
                per_time.append(per_frequency);
            }
            out[name] = per_time;
        }
    }
    return out;
}

static py::dict clean_cube_dispatch(py::array residual, py::array psf,
                                    py::array model, py::array mask, double gain,
                                    py::array threshold_per_cycle, py::array max_iter_per_cycle_updated,
                                    double asp_fused_threshold, double asp_psf_width,
                                    int asp_largest_scale, int asp_stop_point_mode,
                                    int asp_scale_normalization_method,
                                    int processing_function_threads,
                                    int max_iter_divergence, double divergence_rms_factor,
                                    double divergence_peak_factor, bool asp_stop_after_hogbom_run,
                                    bool asp_scale_significance, double asp_significance,
                                    bool asp_residual_components,
                                    double asp_residual_component_noise_floor,
                                    double asp_residual_component_min_fraction,
                                    double asp_residual_component_max_epsilon_fraction,
                                    double asp_residual_component_switch_factor,
                                    bool asp_residual_component_floor_at_threshold,
                                    bool asp_residual_component_positive_only,
                                    double asp_significance_gain, double asp_no_change_tolerance,
                                    bool record_iteration_history, double asp_min_mask_coverage) {
    aspclean::AspOptions options;
    options.asp_scale_significance = asp_scale_significance;
    options.asp_significance = asp_significance;
    options.asp_residual_components = asp_residual_components;
    options.asp_residual_component_noise_floor = asp_residual_component_noise_floor;
    options.asp_residual_component_min_fraction = asp_residual_component_min_fraction;
    options.asp_residual_component_max_epsilon_fraction = asp_residual_component_max_epsilon_fraction;
    options.asp_residual_component_switch_factor = asp_residual_component_switch_factor;
    options.asp_residual_component_floor_at_threshold = asp_residual_component_floor_at_threshold;
    options.asp_residual_component_positive_only = asp_residual_component_positive_only;
    options.asp_significance_gain = asp_significance_gain;
    options.asp_no_change_tolerance = asp_no_change_tolerance;
    options.record_iteration_history = record_iteration_history;
    options.asp_min_mask_coverage = asp_min_mask_coverage;
    if (residual.ndim() != 5)
        throw std::runtime_error(
            "residual must be a 5D array [time, frequency, polarization, y, x]");
    auto dt = residual.dtype();
    if (dt.equal(py::dtype::of<float>()))
        return cube_impl<float>(residual, psf, model, mask, gain, threshold_per_cycle,
                                max_iter_per_cycle_updated, asp_fused_threshold, asp_psf_width,
                                asp_largest_scale, asp_stop_point_mode,
                                asp_scale_normalization_method, processing_function_threads,
                                max_iter_divergence, divergence_rms_factor,
                                divergence_peak_factor, asp_stop_after_hogbom_run, options);
    if (dt.equal(py::dtype::of<double>()))
        return cube_impl<double>(residual, psf, model, mask, gain, threshold_per_cycle,
                                 max_iter_per_cycle_updated, asp_fused_threshold, asp_psf_width,
                                 asp_largest_scale, asp_stop_point_mode,
                                 asp_scale_normalization_method, processing_function_threads,
                                 max_iter_divergence, divergence_rms_factor,
                                 divergence_peak_factor, asp_stop_after_hogbom_run, options);
    throw std::runtime_error("residual must be float32 or float64");
}

PYBIND11_MODULE(_aspclean_fast_ext, m) {
    m.doc() =
        "Fast Asp model update: clean and clean_cube of the reference "
        "(deconvolvers.aspclean), with its arguments and results, bit for bit.";
    // Every fast module reads ASTROVIPER_KERNELS_ISA at its import (an unknown
    // value fails it); this one has the baseline routines only.
    static_cast<void>(astroviper::fast::select_isa());
    m.attr("ISA") = "generic";

    m.def("clean", &clean_dispatch,
          "Run the Asp model update on a single 2D plane in place. `residual` and "
          "`model` (float32 or float64, C-contiguous, writeable) are modified: "
          "the residual replaces the dirty image and Asp/Hogbom components are "
          "accumulated into the model. `psf` and optional `mask` share the dtype "
          "and shape. asp_psf_width = -1 estimates the PSF width internally. The "
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
          "2-D transform. asp_stop_after_hogbom_run ends the model update after a "
          "Hogbom run unless the RMS of the residual has fallen, by less than "
          "a factor 1.5, since its start (CASA and LibRA); off, the model "
          "update goes on to its threshold or iteration limit. Options beyond "
          "CASA and LibRA (all off by default here but asp_min_mask_coverage, "
          "see AspOptions in asp_clean.hpp): asp_min_mask_coverage is the least "
          "share of the flux of a Gaussian component on the mask, in [0, 1). "
          "asp_residual_components cleans first with components "
          "shaped by the residual (Ma, Zhang & Zhang 2026), C = R tanh((sR - "
          "T) / eps) on the mask pixels with sR >= T, T = max(frac |R_peak|, "
          "asp_residual_component_noise_floor x noise, and the threshold with "
          "asp_residual_component_floor_at_threshold), (frac, eps) fitted within "
          "[asp_residual_component_min_fraction, 0.99] x [1e-3, "
          "asp_residual_component_max_epsilon_fraction] by least squares of the "
          "masked residual, down to asp_residual_component_switch_factor x "
          "threshold_per_cycle; then the Asp iterations. asp_scale_significance "
          "measures the noise of the residual smoothed with each initial scale "
          "and, with the peak below "
          "threshold_per_cycle, goes on cleaning the most significant smoothed "
          "residual while its significance reaches that of the threshold at "
          "scale 0 (or asp_significance, when > 0), with the gain "
          "asp_significance_gain (when > 0); scale_noise is returned. "
          "record_components returns the Asp components (i, j, scale, flux) "
          "and the residual-shaped fits (peak, frac, eps_frac, 0, 0, "
          "objective).",
          py::arg("residual"), py::arg("psf"), py::arg("model"),
          py::arg("mask") = py::array(), py::arg("gain") = 0.1,
          py::arg("threshold_per_cycle") = 0.0, py::arg("max_iter_per_cycle_updated") = 100,
          py::arg("asp_fused_threshold") = 0.0, py::arg("asp_psf_width") = -1.0,
          py::arg("asp_largest_scale") = -1, py::arg("asp_stop_point_mode") = -1,
          py::arg("asp_scale_normalization_method") = 1, py::arg("verbose") = false,
          py::arg("max_iter_divergence") = -1, py::arg("divergence_rms_factor") = 1.0,
          py::arg("divergence_peak_factor") = 1.0, py::arg("processing_function_threads") = 1,
          py::arg("asp_stop_after_hogbom_run") = false, py::arg("asp_scale_significance") = false,
          py::arg("asp_significance") = 0.0, py::arg("record_components") = false,
          py::arg("asp_residual_components") = false,
          py::arg("asp_residual_component_noise_floor") = 2.0,
          py::arg("asp_residual_component_min_fraction") = 0.3,
          py::arg("asp_residual_component_max_epsilon_fraction") = 10.0,
          py::arg("asp_residual_component_switch_factor") = 1.0,
          py::arg("asp_residual_component_floor_at_threshold") = true,
          py::arg("asp_residual_component_positive_only") = false,
          py::arg("asp_significance_gain") = 0.0, py::arg("asp_no_change_tolerance") = 0.0,
          py::arg("record_iteration_history") = false,
          py::arg("asp_min_mask_coverage") = 0.995);

    m.def("clean_cube", &clean_cube_dispatch,
          "Run the Asp model update independently on every (time, frequency, "
          "polarization) plane of a 5D cube in place, across processing_function_threads worker "
          "threads. The PSF cube may be single-polarization (broadcast). "
          "Iteration control is independent per plane: threshold_per_cycle "
          "(float64) and max_iter_per_cycle_updated (int32) are (nt, nf, np) "
          "arrays giving each plane its own threshold and iteration limit; the "
          "divergence test, asp_stop_after_hogbom_run and the options beyond "
          "CASA (asp_residual_components, asp_scale_significance, ...) are those of "
          "clean. "
          "Returns per-plane arrays of shape (nt, nf, np), stop_code among "
          "them.",
          py::arg("residual"), py::arg("psf"), py::arg("model"),
          py::arg("mask") = py::array(), py::arg("gain") = 0.1,
          py::arg("threshold_per_cycle") = py::array(), py::arg("max_iter_per_cycle_updated") = py::array(),
          py::arg("asp_fused_threshold") = 0.0, py::arg("asp_psf_width") = -1.0,
          py::arg("asp_largest_scale") = -1, py::arg("asp_stop_point_mode") = -1,
          py::arg("asp_scale_normalization_method") = 1, py::arg("processing_function_threads") = 1,
          py::arg("max_iter_divergence") = -1, py::arg("divergence_rms_factor") = 1.0,
          py::arg("divergence_peak_factor") = 1.0, py::arg("asp_stop_after_hogbom_run") = false,
          py::arg("asp_scale_significance") = false, py::arg("asp_significance") = 0.0,
          py::arg("asp_residual_components") = false,
          py::arg("asp_residual_component_noise_floor") = 2.0,
          py::arg("asp_residual_component_min_fraction") = 0.3,
          py::arg("asp_residual_component_max_epsilon_fraction") = 10.0,
          py::arg("asp_residual_component_switch_factor") = 1.0,
          py::arg("asp_residual_component_floor_at_threshold") = true,
          py::arg("asp_residual_component_positive_only") = false,
          py::arg("asp_significance_gain") = 0.0, py::arg("asp_no_change_tolerance") = 0.0,
          py::arg("record_iteration_history") = false,
          py::arg("asp_min_mask_coverage") = 0.995);
}
