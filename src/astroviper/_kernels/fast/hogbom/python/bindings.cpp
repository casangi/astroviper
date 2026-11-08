// Python entries of the fast Hogbom minor cycle.
//
// clean_cube(*args, **kwargs) and clean_cube_many_threads(*args, **kwargs)
// take the reference's arguments (deconvolvers/hogbom/python/bindings.cpp)
// and return the reference's results, or None having written nothing: the
// caller (__init__.py) then runs the reference with the same arguments. They
// never raise for their arguments: they decline every call that the reference
// binding would not run exactly as given, so that pybind11's conversions, the
// defaults and every error message stay the reference's:
//
// * argument names, as pybind11 binds them to the reference's parameters;
// * types: an array argument that is not an ndarray (the reference converts
//   it), a clean_box that is not a tuple or whose four items do not convert
//   to int, a scalar that pybind11 cannot convert;
// * the reference binding's checks (dtypes, dimensions, shapes, C-contiguity,
//   a writeable residual and model), as predicates here;
// * a residual or model that shares memory with another array argument (the
//   reference would read values it has already overwritten), and image planes
//   without pixels (which the reference reads out of bounds);
// * a call that cannot allocate what it needs before its first write
//   (std::bad_alloc, or NumPy's MemoryError for the result arrays).
//
// One error is not the reference's: a keyword whose name cannot be encoded
// as UTF-8 (a lone surrogate) raises RuntimeError here, where the reference
// binding aborts the process while it formats its TypeError.
//
// A misaligned residual, model or PSF, which the reference reads in place
// (Dask can hand its workers arrays at any offset of the frames it received),
// is read through an aligned copy, as much memory again as the array, filled
// and written back without the GIL: the residual and the model are written
// back after the call, also when it raises after its first write, so that the
// caller sees what an aligned caller sees. The results are assembled as the
// reference binding assembles them, the same dtypes, shapes and keys in the
// same order.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hclean_fast.hpp"

namespace py = pybind11;

using hclean_fast::Entry;

namespace {

// The reference's parameters, in order; the first kRequired have no default.
constexpr std::array<const char*, 14> kParameters = {
    "residual_cube",
    "psf_cube",
    "model_cube",
    "mask_cube",
    "clean_box",
    "max_iter_per_cycle_updated",
    "gain",
    "threshold_per_cycle",
    "speedup",
    "processing_function_threads",
    "max_iter_divergence",
    "divergence_rms_factor",
    "divergence_peak_factor",
    "record_iteration_history"};
constexpr std::size_t kRequired = 3;

enum Parameter : std::size_t {
    kResidual,
    kPsf,
    kModel,
    kMask,
    kCleanBox,
    kMaxIter,
    kGain,
    kThreshold,
    kSpeedup,
    kThreads,
    kMaxIterDivergence,
    kRmsFactor,
    kPeakFactor,
    kHistory,
};

using Bound = std::array<py::handle, kParameters.size()>;

// The arguments matched to the parameters as pybind11 matches them, or false
// where the reference's call would fail.
bool match(const py::args& args, const py::kwargs& kwargs, Bound& bound) {
    if (args.size() > kParameters.size()) return false;
    for (std::size_t i = 0; i < args.size(); ++i) bound[i] = args[i].ptr();
    for (const auto& [key, value] : kwargs) {
        const auto name = key.cast<std::string>();
        const auto* found = std::find(kParameters.begin(), kParameters.end(), name);
        if (found == kParameters.end()) return false;
        py::handle& slot = bound[static_cast<std::size_t>(found - kParameters.begin())];
        if (slot) return false;  // given by position and by name
        slot = value;
    }
    return std::all_of(bound.begin(), bound.begin() + kRequired, [](py::handle h) { return bool(h); });
}

// NPY_ARRAY_ALIGNED of NumPy's C API (numpy/ndarraytypes.h, a stable ABI
// value); pybind11 names the C-contiguity flag (py::array::c_style) only.
constexpr int kNpyArrayAligned = 0x0100;

bool has_flag(const py::array& array, int flag) {
    return (array.flags() & flag) != 0;
}

// An argument the reference takes as a py::array as given: an ndarray (or a
// subclass). Anything else the reference converts to a new array, or rejects.
std::optional<py::array> ndarray(py::handle object) {
    if (!py::isinstance<py::array>(object)) return std::nullopt;
    return py::reinterpret_borrow<py::array>(object);
}

template <typename U>
bool of_type(const py::array& array) {
    return array.dtype().equal(py::dtype::of<U>());
}

bool shaped(const py::array& array, std::initializer_list<py::ssize_t> shape) {
    if (array.ndim() != static_cast<py::ssize_t>(shape.size())) return false;
    py::ssize_t axis = 0;
    for (const py::ssize_t length : shape) {
        if (array.shape(axis++) != length) return false;
    }
    return true;
}

// The reference binding's validate_inplace_cube, validate_per_plane_array and
// the checks around them, as predicates.
template <typename U>
bool valid_cube(const py::array& array, std::initializer_list<py::ssize_t> shape, bool writeable) {
    return of_type<U>(array) && has_flag(array, py::array::c_style) &&
           (!writeable || array.writeable()) && shaped(array, shape);
}

bool overlaps(const py::array& a, const py::array& b) {
    const auto* a_begin = static_cast<const char*>(a.data());
    const auto* b_begin = static_cast<const char*>(b.data());
    return a.nbytes() > 0 && b.nbytes() > 0 && a_begin < b_begin + b.nbytes() &&
           b_begin < a_begin + a.nbytes();
}

// A scalar argument as pybind11 converts it for the reference (its default
// when not given), or nothing where the reference's call would fail.
template <typename U>
std::optional<U> scalar(py::handle object, U fallback) {
    if (!object) return fallback;
    try {
        return py::cast<U>(object);
    } catch (const py::cast_error&) {
        return std::nullopt;
    }
}

// The iteration history fields of the results, with their records.
const std::pair<const char*, std::vector<double> hclean::IterationHistory::*>
    kIterationHistoryRecords[] = {
        {"iteration_peak_residual", &hclean::IterationHistory::peak},
        {"iteration_residual_rms", &hclean::IterationHistory::rms},
        {"iteration_component_l", &hclean::IterationHistory::l},
        {"iteration_component_m", &hclean::IterationHistory::m},
        {"iteration_component_flux", &hclean::IterationHistory::flux},
        {"iteration_component_scale", &hclean::IterationHistory::scale},
};

// A per-plane iteration record as the reference returns it: nested lists
// indexed [t][f][p] of 1-D float64 arrays.
py::list per_plane_iteration_history(const std::vector<hclean::IterationHistory>& history,
                                     std::vector<double> hclean::IterationHistory::*record, int nt,
                                     int nf, int np_img) {
    py::list per_time;
    std::size_t pl = 0;
    for (int tt = 0; tt < nt; ++tt) {
        py::list per_frequency;
        for (int nn = 0; nn < nf; ++nn) {
            py::list per_polarization;
            for (int pp = 0; pp < np_img; ++pp, ++pl) {
                const std::vector<double>& values = history[pl].*record;
                per_polarization.append(
                    py::array_t<double>(static_cast<py::ssize_t>(values.size()), values.data()));
            }
            per_frequency.append(per_polarization);
        }
        per_time.append(per_frequency);
    }
    return per_time;
}

// The data of an array as the fast path reads or writes it: in place, or an
// aligned copy of a misaligned array, allocated here with the GIL held and
// filled by fill() and written back by restore() without it (they touch the
// raw buffers only; the caller's references keep the array alive).
template <typename T>
class Aligned {
public:
    explicit Aligned(const py::array& array)
        : raw_(const_cast<void*>(array.data())), size_(static_cast<std::size_t>(array.size())) {
        if (!has_flag(array, kNpyArrayAligned) && size_ > 0) {
            copy_.reset(new T[size_]);  // every element is written by fill()
        }
    }
    void fill() {
        if (copy_) std::memcpy(copy_.get(), raw_, size_ * sizeof(T));
    }
    T* data() { return copy_ ? copy_.get() : static_cast<T*>(raw_); }
    void restore() {
        if (copy_) std::memcpy(raw_, copy_.get(), size_ * sizeof(T));
    }

private:
    void* raw_;  // the array's data, at any alignment (only memcpy reads or writes it)
    std::size_t size_;
    std::unique_ptr<T[]> copy_;
};

// What a call allocates before the fast path runs: the aligned data of the
// image cubes, the per-plane values (read once, through copies, at any
// alignment), the iteration histories and the per-plane result arrays,
// allocated as the reference allocates them.
template <typename T>
struct Buffers {
    Buffers(const py::array& residual_cube, const py::array& model_cube, const py::array& psf_cube,
            const py::array& max_iter, const py::array& threshold, int nt, int nf, int np_img,
            bool record)
        : residual(residual_cube),
          model(model_cube),
          psf(psf_cube),
          budgets(static_cast<std::size_t>(max_iter.size())),
          thresholds(static_cast<std::size_t>(threshold.size())),
          history(record ? budgets.size() : 0),
          iterations({nt, nf, np_img}),
          final_peak({nt, nf, np_img}),
          total_flux({nt, nf, np_img}),
          converged({nt, nf, np_img}),
          stop_code({nt, nf, np_img}),
          diverged({nt, nf, np_img}) {
        if (!budgets.empty()) {
            std::memcpy(budgets.data(), max_iter.data(), budgets.size() * sizeof(int));
            std::memcpy(thresholds.data(), threshold.data(), thresholds.size() * sizeof(T));
        }
    }

    // The aligned copies of the image cubes (without the GIL).
    void fill() {
        residual.fill();
        model.fill();
        psf.fill();
    }

    // The residual and model written back to misaligned caller arrays
    // (without the GIL).
    void restore() {
        residual.restore();
        model.restore();
    }

    Aligned<T> residual, model, psf;
    std::vector<int> budgets;
    std::vector<T> thresholds;
    std::vector<hclean::IterationHistory> history;
    py::array_t<int> iterations;
    py::array_t<T> final_peak;
    py::array_t<T> total_flux;
    py::array_t<bool> converged;
    py::array_t<int> stop_code;
    py::array_t<bool> diverged;
};

// The call with the bound arguments, or None; sets `written` before its
// first write.
template <typename T>
py::object run(Entry entry, const Bound& bound, bool& written) {
    const std::optional<py::array> residual = ndarray(bound[kResidual]);
    const std::optional<py::array> psf = ndarray(bound[kPsf]);
    const std::optional<py::array> model = ndarray(bound[kModel]);
    if (!residual || !psf || !model) return py::none();
    // max_iter_per_cycle_updated and threshold_per_cycle default to empty
    // float64 arrays, which the reference rejects.
    if (!bound[kMaxIter] || !bound[kThreshold]) return py::none();
    const std::optional<py::array> max_iter = ndarray(bound[kMaxIter]);
    const std::optional<py::array> threshold = ndarray(bound[kThreshold]);
    if (!max_iter || !threshold) return py::none();
    std::optional<py::array> mask;  // an empty mask (the default) is no mask
    if (bound[kMask]) {
        mask = ndarray(bound[kMask]);
        if (!mask) return py::none();
        if (mask->size() == 0) mask.reset();
    }

    // The residual's shape, which the reference reads as int.
    if (residual->ndim() != 5) return py::none();
    for (py::ssize_t axis = 0; axis < 5; ++axis) {
        if (residual->shape(axis) > INT_MAX) return py::none();
    }
    const int nt = static_cast<int>(residual->shape(0));
    const int nf = static_cast<int>(residual->shape(1));
    const int np_img = static_cast<int>(residual->shape(2));
    const int ny = static_cast<int>(residual->shape(3));
    const int nx = static_cast<int>(residual->shape(4));
    const std::initializer_list<py::ssize_t> image = {nt, nf, np_img, ny, nx};
    const std::initializer_list<py::ssize_t> planes = {nt, nf, np_img};
    if (!valid_cube<T>(*residual, image, true) || !valid_cube<T>(*model, image, true)) {
        return py::none();
    }
    if (psf->ndim() != 5 || psf->shape(3) > INT_MAX || psf->shape(4) > INT_MAX) return py::none();
    const int psf_ny = static_cast<int>(psf->shape(3));
    const int psf_nx = static_cast<int>(psf->shape(4));
    const int np_psf = static_cast<int>(psf->shape(2));
    if (!((psf_ny == ny && psf_nx == nx) || (psf_ny == 2 * ny && psf_nx == 2 * nx))) {
        return py::none();
    }
    if ((np_psf != np_img && np_psf != 1) ||
        !valid_cube<T>(*psf, {nt, nf, np_psf, psf_ny, psf_nx}, false)) {
        return py::none();
    }
    if (mask && !valid_cube<bool>(*mask, image, false)) return py::none();
    if (!valid_cube<int>(*max_iter, planes, false) || !valid_cube<T>(*threshold, planes, false)) {
        return py::none();
    }
    const long nplanes = static_cast<long>(nt) * nf * np_img;
    if (nplanes > INT_MAX || (nplanes > 0 && (ny == 0 || nx == 0))) return py::none();

    // Arrays the call writes may share no memory with another argument.
    for (const py::array* target : {&*residual, &*model}) {
        for (const py::array* other : {&*residual, &*model, &*psf, &*max_iter, &*threshold}) {
            if (target != other && overlaps(*target, *other)) return py::none();
        }
        if (mask && overlaps(*target, *mask)) return py::none();
    }

    // The clean box, as the reference parses it (-1: full extent).
    int xbeg = 0, xend = nx, ybeg = 0, yend = ny;
    if (bound[kCleanBox]) {
        if (!py::isinstance<py::tuple>(bound[kCleanBox])) return py::none();
        const auto box = py::reinterpret_borrow<py::tuple>(bound[kCleanBox]);
        if (box.size() == 4) {
            std::array<int, 4> values{};
            for (std::size_t i = 0; i < 4; ++i) {
                const std::optional<int> value = scalar<int>(box[i], 0);
                if (!value) return py::none();
                values[i] = *value;
            }
            xbeg = values[0];
            xend = values[1];
            ybeg = values[2];
            yend = values[3];
            if (xbeg == -1) xbeg = 0;
            if (xend == -1) xend = nx;
            if (ybeg == -1) ybeg = 0;
            if (yend == -1) yend = ny;
            xbeg = std::max(0, std::min(xbeg, nx - 1));
            xend = std::max(xbeg + 1, std::min(xend, nx));
            ybeg = std::max(0, std::min(ybeg, ny - 1));
            yend = std::max(ybeg + 1, std::min(yend, ny));
        }
    }

    const std::optional<double> gain = scalar<double>(bound[kGain], 0.1);
    const std::optional<double> speedup = scalar<double>(bound[kSpeedup], 0.0);
    const std::optional<int> threads = scalar<int>(bound[kThreads], 1);
    const std::optional<int> max_iter_divergence = scalar<int>(bound[kMaxIterDivergence], -1);
    const std::optional<double> rms_factor = scalar<double>(bound[kRmsFactor], 1.0);
    const std::optional<double> peak_factor = scalar<double>(bound[kPeakFactor], 1.0);
    const std::optional<bool> record = scalar<bool>(bound[kHistory], false);
    if (!gain || !speedup || !threads || !max_iter_divergence || !rms_factor || !peak_factor ||
        !record) {
        return py::none();
    }

    Buffers<T> b(*residual, *model, *psf, *max_iter, *threshold, nt, nf, np_img, *record);
    {
        hclean_fast::Cube<T> cube{};
        cube.entry = entry;
        cube.residual = b.residual.data();
        cube.model = b.model.data();
        cube.psf = b.psf.data();
        cube.mask = mask ? static_cast<const std::uint8_t*>(mask->data()) : nullptr;
        cube.nt = nt;
        cube.nf = nf;
        cube.np_img = np_img;
        cube.np_psf = np_psf;
        cube.ny = ny;
        cube.nx = nx;
        cube.psf_ny = psf_ny;
        cube.psf_nx = psf_nx;
        cube.xbeg = xbeg;
        cube.xend = xend;
        cube.ybeg = ybeg;
        cube.yend = yend;
        cube.max_iter = b.budgets.data();
        cube.threshold = b.thresholds.data();
        cube.gain = static_cast<T>(*gain);
        cube.speedup = static_cast<T>(*speedup);
        cube.divergence.max_iter_divergence = *max_iter_divergence;
        cube.divergence.rms_factor = *rms_factor;
        cube.divergence.peak_factor = static_cast<T>(*peak_factor);
        cube.threads = *threads;
        cube.iterations = b.iterations.mutable_data();
        cube.stop_code = b.stop_code.mutable_data();
        cube.history = *record ? b.history.data() : nullptr;
        cube.final_peak = b.final_peak.mutable_data();
        cube.total_flux = b.total_flux.mutable_data();
        cube.converged = b.converged.mutable_data();
        bool done = false;
        {
            py::gil_scoped_release release;
            b.fill();
            // clean_cube declines (false) before its first write, so anything
            // it throws comes after it.
            written = true;
            try {
                done = hclean_fast::clean_cube<T>(cube);
            } catch (...) {
                // After the first write (std::bad_alloc of a growing component
                // list or history): it propagates, as the reference's would,
                // with the cube partly cleaned, at any alignment.
                b.restore();
                throw;
            }
            if (done) b.restore();
        }
        if (!done) return py::none();
    }

    bool* dv = b.diverged.mutable_data();
    const int* stop = b.stop_code.data();
    for (long i = 0; i < nplanes; ++i) dv[i] = (stop[i] == hclean::STOP_DIVERGED);

    py::dict results;
    results["iterations_performed"] = b.iterations;
    results["final_peak"] = b.final_peak;
    results["total_flux_cleaned"] = b.total_flux;
    results["converged"] = b.converged;
    results["stop_code"] = b.stop_code;
    results["diverged"] = b.diverged;
    if (*record) {
        for (const auto& [name, member] : kIterationHistoryRecords) {
            results[name] = per_plane_iteration_history(b.history, member, nt, nf, np_img);
        }
    }
    return results;
}

// A call's result, or None. An allocation failure before the first write
// (`written`, which run() sets) declines the call, as an argument error does.
py::object call(Entry entry, const py::args& args, const py::kwargs& kwargs) {
    bool written = false;
    try {
        Bound bound{};
        if (!match(args, kwargs, bound)) return py::none();
        const std::optional<py::array> residual = ndarray(bound[kResidual]);
        if (!residual) return py::none();
        if (of_type<float>(*residual)) return run<float>(entry, bound, written);
        if (of_type<double>(*residual)) return run<double>(entry, bound, written);
        return py::none();
    } catch (const std::bad_alloc&) {
        if (written) throw;
        return py::none();
    } catch (const py::error_already_set& error) {
        if (written || !error.matches(PyExc_MemoryError)) throw;
        return py::none();
    }
}

}  // namespace

PYBIND11_MODULE(_hogbom_fast_ext, module) {
    module.doc() =
        "Fast Hogbom minor cycle: the results of the reference clean_cube and "
        "clean_cube_many_threads, bit for bit, declining every other call to the reference";
    module.attr("ISA") = hclean_fast::kernels().isa;  // chosen once, at this import
    module.def(
        "clean_cube",
        [](const py::args& args, const py::kwargs& kwargs) {
            return call(Entry::clean_cube, args, kwargs);
        },
        "clean_cube(*args, **kwargs) with the reference's arguments: cleans the cube in place and "
        "returns the reference's results, or writes nothing and returns None (the caller then "
        "runs the reference).");
    module.def(
        "clean_cube_many_threads",
        [](const py::args& args, const py::kwargs& kwargs) {
            return call(Entry::clean_cube_many_threads, args, kwargs);
        },
        "clean_cube_many_threads(*args, **kwargs), likewise.");
}
