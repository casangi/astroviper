#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/mtmfs_clean.hpp"

namespace py = pybind11;

namespace {

template <typename T>
py::buffer_info check_array(py::array& arr, const char* name, const std::vector<py::ssize_t>& shape, bool writable) {
    if (!arr.dtype().is(py::dtype::of<T>()))
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

void stack_dims(const py::array& stack, const char* name, int& n, int& ny, int& nx) {
    if (stack.ndim() != 3) throw std::runtime_error(std::string(name) + " must be a 3-D array (nterms, ny, nx)");
    if (stack.shape(0) > mtmfs::kMaxTaylorTerms)
        throw std::runtime_error(std::string(name) + " has too many Taylor terms (max " +
                                 std::to_string(mtmfs::kMaxTaylorTerms) + ")");
    n = static_cast<int>(stack.shape(0));
    ny = static_cast<int>(stack.shape(1));
    nx = static_cast<int>(stack.shape(2));
    if (n < 1) throw std::runtime_error(std::string(name) + " must have at least one Taylor term");
    if (ny < 2 || nx < 2) throw std::runtime_error(std::string(name) + " image dimensions must be >= 2 pixels");
}

py::array_t<double> hessian_array(const std::vector<double>& m, int nscales, int nterms) {
    py::array_t<double> out({nscales, nterms, nterms});
    std::memcpy(out.mutable_data(), m.data(), sizeof(double) * m.size());
    return out;
}

py::array_t<double> matrix_array(const std::vector<double>& m, int n) {
    py::array_t<double> out({n, n});
    std::memcpy(out.mutable_data(), m.data(), sizeof(double) * m.size());
    return out;
}

template <typename T>
py::dict clean_impl(py::array residual, py::array psf, py::array model, py::object mask_obj,
                    const std::vector<float>& scales, float small_scale_bias, int niter, double gain, double threshold,
                    double stop_fraction, double mask_threshold) {
    int nterms, ny, nx;
    stack_dims(residual, "residual", nterms, ny, nx);
    const std::vector<py::ssize_t> stack = {nterms, ny, nx};
    const std::vector<py::ssize_t> psf_shape = {2 * nterms - 1, ny, nx};

    py::buffer_info ri = check_array<T>(residual, "residual", stack, true);
    py::buffer_info mi = check_array<T>(model, "model", stack, true);
    py::buffer_info pi = check_array<T>(psf, "psf", psf_shape, false);
    if (ri.ptr == mi.ptr) throw std::runtime_error("residual and model must be distinct arrays");

    const T* mask_ptr = nullptr;
    py::array mask_arr;
    py::buffer_info ki;
    if (!mask_obj.is_none()) {
        mask_arr = py::cast<py::array>(mask_obj);
        if (mask_arr.size() > 0) {
            ki = check_array<T>(mask_arr, "mask", {ny, nx}, false);
            mask_ptr = static_cast<const T*>(ki.ptr);
        }
    }

    T* res = static_cast<T*>(ri.ptr);
    T* mod = static_cast<T*>(mi.ptr);
    const T* psf_ptr = static_cast<const T*>(pi.ptr);

    mtmfs::CleanResult<T> result;
    {
        py::gil_scoped_release release;
        result = mtmfs::clean<T>(res, mod, psf_ptr, mask_ptr, nterms, nx, ny, scales, small_scale_bias, niter,
                                 static_cast<T>(gain), static_cast<T>(threshold), static_cast<T>(stop_fraction),
                                 static_cast<T>(mask_threshold));
    }

    const int nscales = static_cast<int>(result.scales.size());
    py::dict out;
    out["iterations_performed"] = result.iterations;
    out["peak_residual"] = result.peak_residual;
    out["model_flux"] = result.model_flux;
    out["converged"] = (result.stop_code == mtmfs::kStopThreshold || result.stop_code == mtmfs::kStopNothingToClean);
    out["stop_code"] = result.stop_code;
    out["scales"] = result.scales;
    out["small_scale_bias"] = result.small_scale_bias;
    out["psf_support"] = result.psf_support;
    out["hessian"] = hessian_array(result.hessian, nscales, nterms);
    out["inverse_hessian"] = hessian_array(result.inverse_hessian, nscales, nterms);
    return out;
}

static py::dict clean_dispatch(py::array residual, py::array psf, py::array model, py::object mask,
                               const std::vector<float>& scales, float small_scale_bias, int niter, double gain,
                               double threshold, double stop_fraction, double mask_threshold) {
    if (residual.dtype().is(py::dtype::of<float>()))
        return clean_impl<float>(residual, psf, model, mask, scales, small_scale_bias, niter, gain, threshold,
                                 stop_fraction, mask_threshold);
    if (residual.dtype().is(py::dtype::of<double>()))
        return clean_impl<double>(residual, psf, model, mask, scales, small_scale_bias, niter, gain, threshold,
                                  stop_fraction, mask_threshold);
    throw std::runtime_error("residual must be float32 or float64");
}

template <typename T>
py::dict hessian_impl(py::array psf, const std::vector<float>& scales, float small_scale_bias) {
    int npsf, ny, nx;
    stack_dims(psf, "psf", npsf, ny, nx);
    if (npsf % 2 == 0) throw std::runtime_error("psf must have an odd number of Taylor terms (2*nterms-1)");
    const int nterms = npsf / 2 + 1;
    py::buffer_info pi = check_array<T>(psf, "psf", {npsf, ny, nx}, false);

    std::vector<float> eff = scales;
    std::vector<double> H, invH;
    int support = 0;
    int rc;
    {
        py::gil_scoped_release release;
        rc = mtmfs::taylor_hessian<T>(static_cast<const T*>(pi.ptr), nterms, nx, ny, eff, small_scale_bias, H, invH,
                                      support);
    }
    if (rc == mtmfs::kSingularHessian)
        throw std::runtime_error(
            "MT-Cleaner error : Non-invertible Hessian. Please check if the multi-frequency data "
            "selection is appropriate for a polynomial fit of the desired order.");

    py::dict out;
    out["hessian"] = hessian_array(H, static_cast<int>(eff.size()), nterms);
    out["inverse_hessian"] = hessian_array(invH, static_cast<int>(eff.size()), nterms);
    out["scales"] = eff;
    out["small_scale_bias"] = mtmfs::clamp_small_scale_bias(small_scale_bias);
    out["psf_support"] = support;
    out["nterms"] = nterms;
    return out;
}

static py::dict hessian_dispatch(py::array psf, const std::vector<float>& scales, float small_scale_bias) {
    if (psf.dtype().is(py::dtype::of<float>())) return hessian_impl<float>(psf, scales, small_scale_bias);
    if (psf.dtype().is(py::dtype::of<double>())) return hessian_impl<double>(psf, scales, small_scale_bias);
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
    if (residual.dtype().is(py::dtype::of<float>()))
        principal_impl<float>(residual, inverse_hessian);
    else if (residual.dtype().is(py::dtype::of<double>()))
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
          "One MTMFS model-update cycle in place. residual and model are (nterms, ny, nx) "
          "C-contiguous writeable arrays of the same dtype (float32 or float64). psf is "
          "(2*nterms-1, ny, nx), read-only. mask is an optional (ny, nx) array of the same "
          "dtype, or None. scales are pixel sizes (empty -> [0]; sorted and de-duplicated; "
          "sizes larger than half the image are dropped). small_scale_bias is clamped to "
          "[-1, 1]. gain <= 0 selects casacore adaptive gain. Returns a dict with "
          "iterations_performed, peak_residual (max |residual[0]*mask|), model_flux "
          "(sum of model[0]), converged (engine stop on threshold or empty search), "
          "stop_code, scales, small_scale_bias, psf_support, hessian and inverse_hessian.",
          py::arg("residual"), py::arg("psf"), py::arg("model"), py::arg("mask") = py::none(),
          py::arg("scales") = std::vector<float>{}, py::arg("small_scale_bias") = 0.0f, py::arg("niter") = 100,
          py::arg("gain") = 0.1, py::arg("threshold") = 0.0, py::arg("stop_fraction") = 0.0,
          py::arg("mask_threshold") = 0.9);

    m.def("hessian", &hessian_dispatch,
          "Taylor Hessians and inverses for a (2*nterms-1, ny, nx) PSF stack. Returns a dict "
          "with hessian and inverse_hessian of shape (nscales, nterms, nterms) float64, plus "
          "scales, small_scale_bias, psf_support and nterms.",
          py::arg("psf"), py::arg("scales") = std::vector<float>{}, py::arg("small_scale_bias") = 0.0f);

    m.def("principal_solution", &principal_dispatch,
          "Replace residual in place by the principal solution "
          "residual[t1] = sum_t2 inverse_hessian[t1, t2] * residual[t2]. "
          "inverse_hessian is (nterms, nterms) float64 (the delta-function scale inverse).",
          py::arg("residual"), py::arg("inverse_hessian"));
}
