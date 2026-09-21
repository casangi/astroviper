#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "../include/mt_image_math.hpp"
#include "../include/mtmfs_clean.hpp"

namespace py = pybind11;

namespace {

// Validate that `arr` has dtype T, `ndim` dimensions, the expected shape and is
// C-contiguous. If `writable`, also require it writeable. Returns buffer_info so
// the caller can take the raw pointer with no copy. No forcecast is ever used:
// a wrong dtype/layout raises instead of silently allocating a converted copy.
template <typename T>
py::buffer_info check_array(py::array& arr, const char* name, const std::vector<py::ssize_t>& shape,
                            bool writable) {
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

const char* kSingularHessianMessage =
    "MT-Cleaner error : Non-invertible Hessian. Please check if the multi-frequency data "
    "selection is appropriate for a polynomial fit of the desired order.";

// dtype-erased holder: the engine is templated on the image value type and the
// dtype is fixed at construction, so every array crossing the boundary must
// match it exactly (float32 or float64) -- no implicit narrowing/widening.
class PyMultiTermCleaner {
   public:
    using Engine = std::variant<mtmfs::MultiTermCleaner<float>, mtmfs::MultiTermCleaner<double>>;

    PyMultiTermCleaner(int nterms, py::tuple shape, const std::vector<float>& scales, float small_scale_bias,
                       py::object dtype)
        : dtype_(py::dtype::from_args(dtype)), engine_(make_engine(nterms, scales, shape, small_scale_bias, dtype_)) {}

   private:
    // Dispatch on the concrete engine type. Defined before first use because
    // a deduced return type cannot be used before the definition is seen.
    template <typename F>
    auto visit(F&& f) const {
        return std::visit(std::forward<F>(f), engine_);
    }
    template <typename F>
    auto visit_mut(F&& f) {
        return std::visit(std::forward<F>(f), engine_);
    }

   public:

    // ---- properties -------------------------------------------------------
    int nterms() const { return visit([](auto& e) { return e.nterms(); }); }
    int npsf_terms() const { return visit([](auto& e) { return e.npsf_terms(); }); }
    int nscales() const { return visit([](auto& e) { return e.nscales(); }); }
    std::vector<float> scales() const { return visit([](auto& e) { return e.scales(); }); }
    py::tuple shape() const { return visit([](auto& e) { return py::make_tuple(e.ny(), e.nx()); }); }
    float small_scale_bias() const { return visit([](auto& e) { return e.small_scale_bias(); }); }
    int psf_support() const { return visit([](auto& e) { return e.psf_support(); }); }
    bool has_psf() const { return visit([](auto& e) { return e.has_psf(); }); }
    py::dtype dtype() const { return dtype_; }

    // ---- set_psf ----------------------------------------------------------
    void set_psf(py::array psf) {
        visit_mut([&](auto& e) {
            using T = typename std::decay_t<decltype(e)>::value_type;
            py::buffer_info pi = check_array<T>(psf, "psf", {e.npsf_terms(), e.ny(), e.nx()}, false);
            int rc;
            {
                py::gil_scoped_release release;  // C++ touches no Python objects
                rc = e.set_psf(static_cast<const T*>(pi.ptr));
            }
            if (rc == -2) throw std::runtime_error(kSingularHessianMessage);
        });
    }

    // ---- clean ------------------------------------------------------------
    py::dict clean(py::array residual, py::array model, py::array mask, int niter, double gain,
                   double threshold, double stop_fraction, double mask_threshold) {
        return visit_mut([&](auto& e) -> py::dict {
            using T = typename std::decay_t<decltype(e)>::value_type;
            if (!e.has_psf()) throw std::runtime_error("set_psf() must be called before clean()");
            const std::vector<py::ssize_t> stack = {e.nterms(), e.ny(), e.nx()};
            py::buffer_info ri = check_array<T>(residual, "residual", stack, true);
            py::buffer_info mi = check_array<T>(model, "model", stack, true);
            if (ri.ptr == mi.ptr) throw std::runtime_error("residual and model must be distinct arrays");

            const T* mask_ptr = nullptr;
            py::buffer_info ki;
            if (mask.size() > 0) {
                ki = check_array<T>(mask, "mask", {e.ny(), e.nx()}, false);
                mask_ptr = static_cast<const T*>(ki.ptr);
            }

            T* res = static_cast<T*>(ri.ptr);
            T* mod = static_cast<T*>(mi.ptr);
            const std::size_t nimg = static_cast<std::size_t>(e.nx()) * e.ny();

            int iters;
            T peak, flux;
            {
                py::gil_scoped_release release;  // buffers are Python-owned and outlive the call
                iters = e.clean(res, mod, mask_ptr, static_cast<T>(mask_threshold), niter,
                                static_cast<T>(stop_fraction), static_cast<T>(gain), static_cast<T>(threshold));
                // SDAlgorithmMSMFS::takeOneStep reductions: max|residual0 * mask|, sum(model0).
                peak = (mask_ptr != nullptr) ? mtmfs::peak_abs_masked<T>(res, mask_ptr, nimg)
                                             : std::abs(mtmfs::find_max_abs<T>(res, e.nx(), e.ny()).value);
                flux = mtmfs::sum_array<T>(mod, nimg);
            }
            if (iters == -2) throw std::runtime_error(kSingularHessianMessage);

            py::dict out;
            out["iterations_performed"] = iters;
            out["peak_residual"] = peak;
            out["model_flux"] = flux;
            out["converged"] = (static_cast<double>(peak) <= threshold);
            return out;
        });
    }

    // ---- principal solution ------------------------------------------------
    void compute_principal_solution(py::array residual) {
        visit_mut([&](auto& e) {
            using T = typename std::decay_t<decltype(e)>::value_type;
            if (!e.has_psf())
                throw std::runtime_error("set_psf() must be called before compute_principal_solution()");
            py::buffer_info ri = check_array<T>(residual, "residual", {e.nterms(), e.ny(), e.nx()}, true);
            py::gil_scoped_release release;
            e.compute_principal_solution(static_cast<T*>(ri.ptr));
        });
    }

    // ---- Hessian accessors (small (nterms, nterms) float64 copies) ---------
    py::array_t<double> hessian(int scale) const {
        return visit([&](auto& e) { return matrix_copy(e.hessian(scale), e.nterms()); });
    }
    py::array_t<double> inverse_hessian(int scale) const {
        return visit([&](auto& e) { return matrix_copy(e.inverse_hessian(scale), e.nterms()); });
    }

   private:
    static Engine make_engine(int nterms, const std::vector<float>& scales, py::tuple shape, float bias,
                              const py::dtype& dt) {
        if (shape.size() != 2) throw std::runtime_error("shape must be (ny, nx)");
        const int ny = py::cast<int>(shape[0]);
        const int nx = py::cast<int>(shape[1]);
        if (dt.is(py::dtype::of<float>()))
            return Engine(std::in_place_type<mtmfs::MultiTermCleaner<float>>, nterms, scales, nx, ny, bias);
        if (dt.is(py::dtype::of<double>()))
            return Engine(std::in_place_type<mtmfs::MultiTermCleaner<double>>, nterms, scales, nx, ny, bias);
        throw std::runtime_error("dtype must be float32 or float64");
    }

    static py::array_t<double> matrix_copy(const std::vector<double>& m, int n) {
        py::array_t<double> out({n, n});
        double* p = out.mutable_data();
        for (std::size_t k = 0; k < m.size(); ++k) p[k] = m[k];
        return out;
    }

    py::dtype dtype_;
    Engine engine_;
};

}  // namespace

PYBIND11_MODULE(_mtmfs_ext, m) {
    m.doc() =
        "Multi-Term Multi-Frequency Synthesis (MTMFS) CLEAN - casacore-free port of "
        "CASA's SDAlgorithmMSMFS / MultiTermMatrixCleaner minor cycle, operating in "
        "place on Python-owned numpy buffers with no copies.";

    py::class_<PyMultiTermCleaner>(m, "MultiTermCleaner",
                                   "Stateful MTMFS cleaner. Construct once per image geometry, call "
                                   "set_psf() once per PSF (builds the PSF/scale transforms and the "
                                   "Taylor Hessian), then call clean() after every residual-update "
                                   "cycle; residual and model stacks are modified in place.")
        .def(py::init<int, py::tuple, const std::vector<float>&, float, py::object>(),
             "Create a cleaner for `nterms` Taylor terms on images of `shape` (ny, nx).\n"
             "`scales` are multi-scale sizes in pixels (empty -> [0.0]; scales larger than\n"
             "half the image are dropped, as in CASA). `small_scale_bias` is clamped to\n"
             "[-1, 1]. `dtype` (float32 or float64) fixes the dtype every array passed to\n"
             "this object must have; no conversion or copy is ever made.",
             py::arg("nterms"), py::arg("shape"), py::arg("scales") = std::vector<float>{},
             py::arg("small_scale_bias") = 0.0f, py::arg("dtype") = py::dtype::of<float>())
        .def("set_psf", &PyMultiTermCleaner::set_psf,
             "Set the PSF Taylor-term stack, shape (2*nterms-1, ny, nx), dtype matching the\n"
             "cleaner. Read zero-copy; computes the PSF/scale transforms and the per-scale\n"
             "Taylor Hessians. Raises RuntimeError if a Hessian is not invertible.",
             py::arg("psf"))
        .def("clean", &PyMultiTermCleaner::clean,
             "Run one MTMFS minor cycle in place. `residual` and `model` are (nterms, ny, nx)\n"
             "C-contiguous writeable arrays of the cleaner's dtype: CLEAN components are added\n"
             "into `model` and `residual` is updated with the new components convolved with\n"
             "the PSF terms. `mask` is an optional (ny, nx) array of the same dtype\n"
             "(`mask_threshold` mirrors SDAlgorithmMSMFS::setMask). `niter`, `gain` and\n"
             "`threshold` are the minor-cycle iteration limit, loop gain and stopping\n"
             "threshold (`gain` <= 0 selects casacore's adaptive gain); `stop_fraction` is\n"
             "the mtclean() stop fraction (SDAlgorithmMSMFS passes 0). Returns a dict with\n"
             "iterations_performed, peak_residual (max |residual[0]*mask|), model_flux\n"
             "(sum of model[0]) and converged (peak_residual <= threshold).",
             py::arg("residual"), py::arg("model"), py::arg("mask") = py::array(), py::arg("niter") = 100,
             py::arg("gain") = 0.1, py::arg("threshold") = 0.0, py::arg("stop_fraction") = 0.0,
             py::arg("mask_threshold") = 0.9)
        .def("compute_principal_solution", &PyMultiTermCleaner::compute_principal_solution,
             "Replace the (nterms, ny, nx) residual stack in place by the principal solution\n"
             "residual[t1] = sum_t2 inverse_hessian(0)[t1, t2] * residual[t2] (the correction\n"
             "CASA applies to the Taylor residuals before restoration).",
             py::arg("residual"))
        .def("hessian", &PyMultiTermCleaner::hessian,
             "Taylor Hessian (nterms, nterms) float64 for the given scale index (a copy).",
             py::arg("scale") = 0)
        .def("inverse_hessian", &PyMultiTermCleaner::inverse_hessian,
             "Inverse Taylor Hessian (nterms, nterms) float64 for the given scale index (a copy).",
             py::arg("scale") = 0)
        .def_property_readonly("nterms", &PyMultiTermCleaner::nterms)
        .def_property_readonly("npsf_terms", &PyMultiTermCleaner::npsf_terms)
        .def_property_readonly("nscales", &PyMultiTermCleaner::nscales)
        .def_property_readonly("scales", &PyMultiTermCleaner::scales,
                               "Effective scale sizes (after dropping scales too large for the image).")
        .def_property_readonly("shape", &PyMultiTermCleaner::shape, "(ny, nx)")
        .def_property_readonly("small_scale_bias", &PyMultiTermCleaner::small_scale_bias)
        .def_property_readonly("psf_support", &PyMultiTermCleaner::psf_support,
                               "Side length of the PSF/scale patches used in the minor cycle (set by set_psf).")
        .def_property_readonly("has_psf", &PyMultiTermCleaner::has_psf)
        .def_property_readonly("dtype", &PyMultiTermCleaner::dtype);
}
