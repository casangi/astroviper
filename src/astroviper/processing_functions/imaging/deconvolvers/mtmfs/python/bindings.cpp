#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/mtmfs_clean.hpp"

namespace py = pybind11;

namespace {

template <typename T>
py::buffer_info check_array(py::array& arr, const char* name, const std::vector<py::ssize_t>& shape, bool writable) {
    if (!py::isinstance<py::array_t<T>>(arr))
        throw std::runtime_error(std::string(name) + " has wrong dtype; expected " +
                                 py::cast<std::string>(py::dtype::of<T>().attr("name")) + " but got " +
                                 py::cast<std::string>(arr.dtype().attr("name")));
    if (!(arr.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous (no copy is made)");
    if (writable && !arr.writeable())
        throw std::runtime_error(std::string(name) + " must be writeable (modified in place)");
    py::buffer_info info = arr.request(writable);
    const int ndim = static_cast<int>(shape.size());
    if (info.ndim != ndim)
        throw std::runtime_error(std::string(name) + " must be a " + std::to_string(ndim) + "-D array");
    for (int d = 0; d < ndim; ++d)
        if (info.shape[d] != shape[d]) {
            std::string want = "(";
            for (int e = 0; e < ndim; ++e) want += std::to_string(shape[e]) + (e + 1 < ndim ? ", " : ")");
            throw std::runtime_error(std::string(name) + " has wrong shape; expected " + want);
        }
    return info;
}

template <typename T>
py::buffer_info check_strided_residual(py::array& arr, const std::vector<py::ssize_t>& shape) {
    if (!py::isinstance<py::array_t<T>>(arr))
        throw std::runtime_error("residual has wrong dtype; expected " +
                                 py::cast<std::string>(py::dtype::of<T>().attr("name")) + " but got " +
                                 py::cast<std::string>(arr.dtype().attr("name")));
    if (!arr.writeable()) throw std::runtime_error("residual must be writeable (modified in place)");
    py::buffer_info info = arr.request(true);
    if (info.ndim != 3) throw std::runtime_error("residual must be a 3-D array");
    for (int d = 0; d < 3; ++d)
        if (info.shape[d] != shape[d])
            throw std::runtime_error("residual has wrong shape; expected (" + std::to_string(shape[0]) + ", " +
                                     std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + ")");

    const py::ssize_t itemsize = static_cast<py::ssize_t>(sizeof(T));
    const py::ssize_t plane_bytes = shape[1] * shape[2] * itemsize;
    if (info.strides[2] != itemsize || info.strides[1] != shape[2] * itemsize ||
        info.strides[0] < plane_bytes || info.strides[0] % itemsize != 0)
        throw std::runtime_error(
            "residual spatial planes must be C-contiguous with a positive Taylor-term stride (no copy is made)");
    return info;
}

struct MemoryInterval {
    std::uintptr_t begin;
    std::uintptr_t end;
};

MemoryInterval memory_interval(const py::buffer_info& info) {
    std::uintptr_t extent = static_cast<std::uintptr_t>(info.itemsize);
    for (py::ssize_t d = 0; d < info.ndim; ++d) {
        if (info.strides[d] < 0) throw std::runtime_error("negative array strides are not supported");
        if (info.shape[d] > 0)
            extent += static_cast<std::uintptr_t>((info.shape[d] - 1) * info.strides[d]);
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(info.ptr);
    return {begin, begin + extent};
}

void stack_dims(const py::array& stack, const char* name, int& n, int& ny, int& nx, bool psf_stack = false) {
    if (stack.ndim() != 3) throw std::runtime_error(std::string(name) + " must be a 3-D array (nterms, ny, nx)");
    const py::ssize_t nplanes = stack.shape(0);
    if (nplanes < 1) throw std::runtime_error(std::string(name) + " must have at least one plane");
    if (psf_stack && nplanes % 2 == 0)
        throw std::runtime_error("psf must have an odd number of planes (2*nterms-1)");
    // Limit fitted Taylor terms, not the PSF's 2*nterms-1 planes.
    const py::ssize_t nterms = psf_stack ? nplanes / 2 + 1 : nplanes;
    if (nterms > mtmfs::kMaxTaylorTerms)
        throw std::runtime_error(std::string(name) + " has too many Taylor terms (max " +
                                 std::to_string(mtmfs::kMaxTaylorTerms) + ")");
    n = static_cast<int>(stack.shape(0));
    ny = static_cast<int>(stack.shape(1));
    nx = static_cast<int>(stack.shape(2));
    if (ny < 2 || nx < 2) throw std::runtime_error(std::string(name) + " image dimensions must be >= 2 pixels");
}

py::array_t<double> hessian_array(const std::vector<double>& m, int nscales, int nterms) {
    py::array_t<double> out({nscales, nterms, nterms});
    std::memcpy(out.mutable_data(), m.data(), sizeof(double) * m.size());
    return out;
}

py::array_t<double> vector_array(const std::vector<double>& values) {
    py::array_t<double> out(values.size());
    std::memcpy(out.mutable_data(), values.data(), sizeof(double) * values.size());
    return out;
}

template <typename T>
py::dict clean_impl(py::array residual, py::array psf, py::array model, py::object mask_obj,
                    const std::vector<float>& scales, float small_scale_bias, int max_iter_per_cycle_updated, double gain,
                    double threshold_per_cycle, double mask_threshold, int max_iter_divergence,
                    double divergence_rms_factor, double divergence_peak_factor) {
    int nterms, ny, nx;
    stack_dims(residual, "residual", nterms, ny, nx);
    const std::vector<py::ssize_t> stack = {nterms, ny, nx};
    const std::vector<py::ssize_t> psf_shape = {2 * nterms - 1, ny, nx};

    py::buffer_info ri = check_strided_residual<T>(residual, stack);
    py::buffer_info mi = check_array<T>(model, "model", stack, true);
    py::buffer_info pi = check_array<T>(psf, "psf", psf_shape, false);
    auto overlaps = [](const py::buffer_info& a, const py::buffer_info& b) {
        const MemoryInterval ai = memory_interval(a);
        const MemoryInterval bi = memory_interval(b);
        return ai.begin < bi.end && bi.begin < ai.end;
    };
    if (overlaps(ri, mi))
        throw std::runtime_error("residual and model must be distinct, non-overlapping arrays");
    if (overlaps(ri, pi) || overlaps(mi, pi))
        throw std::runtime_error("writable residual and model arrays must not overlap psf");

    const T* mask_ptr = nullptr;
    py::array mask_arr;
    py::buffer_info ki;
    if (!mask_obj.is_none()) {
        // No py::cast: it would silently copy a list or other array-like into a new array.
        if (!py::isinstance<py::array>(mask_obj)) throw py::type_error("mask must be a numpy.ndarray or None");
        mask_arr = py::reinterpret_borrow<py::array>(mask_obj);
        if (mask_arr.size() > 0) {
            ki = check_array<T>(mask_arr, "mask", {ny, nx}, false);
            mask_ptr = static_cast<const T*>(ki.ptr);
        }
    }
    if (mask_ptr != nullptr && (overlaps(ri, ki) || overlaps(mi, ki)))
        throw std::runtime_error("writable residual and model arrays must not overlap mask");

    T* res = static_cast<T*>(ri.ptr);
    T* mod = static_cast<T*>(mi.ptr);
    const T* psf_ptr = static_cast<const T*>(pi.ptr);

    deconvolvers::DivergenceControl<T> divergence;
    divergence.max_iter_divergence = max_iter_divergence;
    divergence.rms_factor = divergence_rms_factor;
    divergence.peak_factor = static_cast<T>(divergence_peak_factor);
    mtmfs::CleanResult<T> result;
    {
        py::gil_scoped_release release;
        result = mtmfs::clean<T>(res, mod, psf_ptr, mask_ptr, nterms, nx, ny, scales, small_scale_bias,
                                 max_iter_per_cycle_updated, static_cast<T>(gain), static_cast<T>(threshold_per_cycle),
                                 static_cast<T>(mask_threshold), divergence,
                                 ri.strides[0] / static_cast<py::ssize_t>(sizeof(T)));
    }

    const int nscales = static_cast<int>(result.scales.size());
    py::dict out;
    out["iterations_performed"] = result.iterations;
    out["peak_residual"] = result.peak_residual;
    out["model_flux"] = result.model_flux;
    out["converged"] = (result.stop_code == mtmfs::kStopThreshold);
    out["diverged"] = (result.stop_code == mtmfs::kStopDiverged);
    out["stop_code"] = result.stop_code;
    out["scales"] = result.scales;
    out["small_scale_bias"] = result.small_scale_bias;
    out["psf_support"] = result.psf_support;
    out["hessian"] = hessian_array(result.hessian, nscales, nterms);
    out["inverse_hessian"] = hessian_array(result.inverse_hessian, nscales, nterms);
    out["hessian_condition_number"] = vector_array(result.hessian_condition_number);
    return out;
}

static py::dict clean_dispatch(py::array residual, py::array psf, py::array model, py::object mask,
                               const std::vector<float>& scales, float small_scale_bias, int max_iter_per_cycle_updated,
                               double gain, double threshold_per_cycle, double mask_threshold,
                               int max_iter_divergence, double divergence_rms_factor,
                               double divergence_peak_factor) {
    if (py::isinstance<py::array_t<float>>(residual))
        return clean_impl<float>(residual, psf, model, mask, scales, small_scale_bias, max_iter_per_cycle_updated, gain,
                                 threshold_per_cycle, mask_threshold, max_iter_divergence,
                                 divergence_rms_factor, divergence_peak_factor);
    if (py::isinstance<py::array_t<double>>(residual))
        return clean_impl<double>(residual, psf, model, mask, scales, small_scale_bias, max_iter_per_cycle_updated, gain,
                                  threshold_per_cycle, mask_threshold, max_iter_divergence,
                                  divergence_rms_factor, divergence_peak_factor);
    throw std::runtime_error("residual must be float32 or float64");
}

template <typename T>
py::dict hessian_impl(py::array psf, const std::vector<float>& scales, float small_scale_bias) {
    int npsf, ny, nx;
    stack_dims(psf, "psf", npsf, ny, nx, true);
    const int nterms = npsf / 2 + 1;
    py::buffer_info pi = check_array<T>(psf, "psf", {npsf, ny, nx}, false);

    std::vector<float> eff = scales;
    std::vector<double> H, invH, condition_number;
    int support = 0;
    int rc;
    {
        py::gil_scoped_release release;
        rc = mtmfs::taylor_hessian<T>(static_cast<const T*>(pi.ptr), nterms, nx, ny, eff, small_scale_bias, H, invH,
                                      condition_number, support);
    }
    if (rc == mtmfs::kSingularHessian)
        throw std::runtime_error(
            "MT-Cleaner error : Non-invertible Hessian. Please check if the multi-frequency data "
            "selection is appropriate for a polynomial fit of the desired order.");

    py::dict out;
    out["hessian"] = hessian_array(H, static_cast<int>(eff.size()), nterms);
    out["inverse_hessian"] = hessian_array(invH, static_cast<int>(eff.size()), nterms);
    out["hessian_condition_number"] = vector_array(condition_number);
    out["scales"] = eff;
    out["small_scale_bias"] = mtmfs::clamp_small_scale_bias(small_scale_bias);
    out["psf_support"] = support;
    out["nterms"] = nterms;
    return out;
}

static py::dict hessian_dispatch(py::array psf, const std::vector<float>& scales, float small_scale_bias) {
    if (py::isinstance<py::array_t<float>>(psf)) return hessian_impl<float>(psf, scales, small_scale_bias);
    if (py::isinstance<py::array_t<double>>(psf)) return hessian_impl<double>(psf, scales, small_scale_bias);
    throw std::runtime_error("psf must be float32 or float64");
}

template <typename T>
void principal_impl(py::array residual, py::array inverse_hessian) {
    int nterms, ny, nx;
    stack_dims(residual, "residual", nterms, ny, nx);
    py::buffer_info ri = check_array<T>(residual, "residual", {nterms, ny, nx}, true);
    py::buffer_info hi = check_array<double>(inverse_hessian, "inverse_hessian", {nterms, nterms}, false);
    py::gil_scoped_release release;
    mtmfs::principal_solution<T>(static_cast<T*>(ri.ptr), static_cast<const double*>(hi.ptr), nterms, nx, ny);
}

static void principal_dispatch(py::array residual, py::array inverse_hessian) {
    if (py::isinstance<py::array_t<float>>(residual))
        principal_impl<float>(residual, inverse_hessian);
    else if (py::isinstance<py::array_t<double>>(residual))
        principal_impl<double>(residual, inverse_hessian);
    else
        throw std::runtime_error("residual must be float32 or float64");
}

}  // namespace

PYBIND11_MODULE(_mtmfs_ext, m) {
    m.doc() =
        "Multi-Term Multi-Frequency Synthesis (MTMFS) CLEAN -- casacore-free port of "
        "CASA SDAlgorithmMSMFS / MultiTermMatrixCleaner as stateless free functions. "
        "Python owns the arrays; residual and model are updated in place with no copies.";

    m.def("clean", &clean_dispatch,
          "One MTMFS model update in place, with 1 <= nterms <= 4. residual and model are (nterms, ny, nx) "
          "writeable arrays of the same dtype (float32 or float64). Each residual spatial plane must be "
          "C-contiguous; the Taylor-term axis may be strided so views into a larger image are updated without "
          "a copy. model must be fully C-contiguous. psf is "
          "(2*nterms-1, ny, nx), read-only. mask is an optional (ny, nx) array of the same "
          "dtype, or None. scales are pixel sizes (empty -> [0]; otherwise 0 is required; sorted and de-duplicated; "
          "sizes larger than half the image are dropped). small_scale_bias is clamped to "
          "[-1, 1]. gain <= 0 selects casacore adaptive gain. max_iter_per_cycle_updated [CASA cycleniter] caps "
          "the components added in this call. The call stops when max |residual[0] convolved with the smallest "
          "scale| over that scale's mask, divided by hessian[0, 0, 0] (the PSF peak when the smallest scale "
          "is 0), falls below threshold_per_cycle; that estimate is updated only inside the psf_support "
          "patch, so the returned residual can still exceed threshold_per_cycle. "
          "mask_threshold > 0 binarises each scale-convolved mask at 0.1 (its value is otherwise unused). The "
          "call also stops as diverged (stop_code 4) on the divergence test shared with the other "
          "deconvolvers, applied to that estimate and to the sum of the squares of the same masked image: "
          "its RMS above divergence_rms_factor times its lowest value for max_iter_divergence consecutive "
          "components, its peak above divergence_peak_factor times its starting value, or a peak that is "
          "not finite (max_iter_divergence < 0 disables the first two tests). The "
          "residual is updated at the end of the call by a circular FFT convolution of the new components "
          "with the PSF. Returns a dict with "
          "iterations_performed, peak_residual (max |residual[0]*mask|, not divided by the PSF peak), model_flux "
          "(sum of model[0]), converged (engine stop on threshold_per_cycle or empty search), diverged, "
          "stop_code (0 did not run, 1 spent max_iter_per_cycle_updated, 2 reached threshold_per_cycle or "
          "nothing left to clean, 4 diverged), scales, small_scale_bias, psf_support, hessian, inverse_hessian and the per-scale "
          "infinity-norm hessian_condition_number.",
          py::arg("residual"), py::arg("psf"), py::arg("model"), py::arg("mask") = py::none(),
          py::arg("scales") = std::vector<float>{}, py::arg("small_scale_bias") = 0.0f,
          py::arg("max_iter_per_cycle_updated") = 100, py::arg("gain") = 0.1, py::arg("threshold_per_cycle") = 0.0,
          py::arg("mask_threshold") = 0.9,
          py::arg("max_iter_divergence") = -1, py::arg("divergence_rms_factor") = 1.0,
          py::arg("divergence_peak_factor") = 1.0);

    m.def("hessian", &hessian_dispatch,
          "Taylor Hessians and inverses for a (2*nterms-1, ny, nx) PSF stack, with 1 <= nterms <= 4. Returns a dict "
          "with hessian and inverse_hessian of shape (nscales, nterms, nterms) float64, the per-scale "
          "infinity-norm hessian_condition_number, plus scales, small_scale_bias, psf_support and nterms.",
          py::arg("psf"), py::arg("scales") = std::vector<float>{}, py::arg("small_scale_bias") = 0.0f);

    m.def("principal_solution", &principal_dispatch,
          "For 1 <= nterms <= 4, replace residual in place by the principal solution "
          "residual[t1] = sum_t2 inverse_hessian[t1, t2] * residual[t2]. "
          "inverse_hessian is (nterms, nterms) float64 (the delta-function scale inverse).",
          py::arg("residual"), py::arg("inverse_hessian"));
}
