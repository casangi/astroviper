// Python entry of the fast dft_predict.
//
// dft_predict(*args, **kwargs) takes the reference's arguments and returns the
// path it took: "S" (the visibilities are written) or "declined" (nothing was
// written, and the caller runs the reference binding with the same
// arguments). It never raises: it declines every call that the reference
// binding would not run exactly as given, so that pybind11's conversions, the
// defaults and every error message stay the reference's:
//
// * argument names, as pybind11 binds them to the reference's parameters
//   (too many positional ones, an unknown, repeated or missing name);
// * types and layouts: an array the reference would convert (another dtype,
//   a list, a non-contiguous map) or reject, or a processing_function_threads
//   that pybind11 cannot convert to int;
// * the reference binding's checks (shapes, C-contiguity, a writable
//   vis_data, maps within their image axes, finite direction cosines in front
//   of the horizon), as predicates here;
// * a vis_data that shares memory with an input (the reference would read
//   inputs that it has already overwritten).
//
// A misaligned input, which the reference reads in place (Dask hands its
// workers arrays at arbitrary offsets of the frames it received), is read
// through an aligned copy of its bytes; path S reads and writes the values of
// vis_data in place, as bytes, at any alignment. The path is then decided in
// C++ from the inputs (dft_separable.cpp), without the GIL, before anything
// is written.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
#include <utility>

#include "dft_fast.hpp"

namespace py = pybind11;

using dft_fast::Element;
using dft_fast::i64;

namespace {

constexpr const char* kSeparable = "S";
constexpr const char* kDeclined = "declined";

// The reference's parameters, in order; the last one has a default (1).
constexpr std::array<const char*, 10> kParameters = {
    "model",   "vis_data", "uvw", "frequency_coord", "frequency_map",
    "time_map", "pol_map", "l",   "m",               "processing_function_threads"};

// The arguments matched to the parameters as pybind11 matches them, or false
// where the reference's call would fail. Names are compared in place, so that
// matching allocates nothing and cannot raise (an unencodable name declines).
bool match(const py::args& args, const py::kwargs& kwargs,
           std::array<py::handle, kParameters.size()>& bound) {
    if (args.size() > kParameters.size()) return false;
    for (std::size_t i = 0; i < args.size(); ++i) bound[i] = args[i].ptr();
    for (const auto& [key, value] : kwargs) {
        const auto* found =
            std::find_if(kParameters.begin(), kParameters.end(), [&](const char* name) {
                return PyUnicode_Check(key.ptr()) &&
                       PyUnicode_CompareWithASCIIString(key.ptr(), name) == 0;
            });
        if (found == kParameters.end()) return false;
        py::handle& slot = bound[static_cast<std::size_t>(found - kParameters.begin())];
        if (slot) return false;  // given by position and by name
        slot = value;
    }
    return std::all_of(bound.begin(), bound.end() - 1, [](py::handle h) { return bool(h); });
}

// NPY_ARRAY_ALIGNED of NumPy's C API (numpy/ndarraytypes.h, a stable ABI
// value); pybind11 names the C-contiguity flag (py::array::c_style) only.
constexpr int kNpyArrayAligned = 0x0100;

bool has_flag(const py::array& array, int flag) {
    return (array.flags() & flag) != 0;
}

// array if NumPy flags it aligned, else an aligned C-contiguous copy of its
// bytes (same dtype), or nothing if the copy cannot be made.
std::optional<py::array> aligned(const py::array& array) {
    if (has_flag(array, kNpyArrayAligned)) return array;
    py::array copy = py::array::ensure(array, kNpyArrayAligned | py::array::c_style);
    if (!copy) return std::nullopt;
    return copy;
}

// An argument the reference takes as array_t<T, c_style>, when the reference
// reads it as given: an ndarray (or subclass) of element type T in native
// byte order and C-contiguous (anything else the reference converts to a
// copy, or rejects).
template <typename T>
std::optional<py::array> typed(py::handle object) {
    if (!py::isinstance<py::array>(object)) return std::nullopt;
    auto array = py::reinterpret_borrow<py::array>(object);
    if (!array.dtype().equal(py::dtype::of<T>()) || !has_flag(array, py::array::c_style))
        return std::nullopt;
    return array;
}

// Element type of the model or of vis_data, as the reference dispatches them.
std::optional<Element> element_of(const py::array& array) {
    const py::dtype dtype = array.dtype();
    if (dtype.equal(py::dtype::of<float>())) return Element::float32;
    if (dtype.equal(py::dtype::of<double>())) return Element::float64;
    if (dtype.equal(py::dtype::of<std::complex<float>>())) return Element::complex64;
    if (dtype.equal(py::dtype::of<std::complex<double>>())) return Element::complex128;
    return std::nullopt;
}

// require_c_contiguous of the reference binding: it checks the strides of the
// array's buffer, which NumPy exports in their canonical form for an array it
// flags C-contiguous (whatever the strides of axes of length 1), so it is
// NumPy's C-contiguous flag.
bool c_contiguous(const py::array& array) {
    return has_flag(array, py::array::c_style);
}

// require_map of the reference binding.
bool valid_map(const py::array& map, py::ssize_t length, i64 limit) {
    if (map.ndim() != 1 || map.shape(0) != length) return false;
    const auto* values = static_cast<const i64*>(map.data());
    for (py::ssize_t i = 0; i < length; ++i) {
        if (values[i] < 0 || values[i] >= limit) return false;
    }
    return true;
}

// require_direction_cosines of the reference binding.
bool valid_direction_cosines(const py::array& l, const py::array& m, py::ssize_t n_l,
                             py::ssize_t n_m) {
    if (l.ndim() != 1 || l.shape(0) != n_l) return false;
    if (m.ndim() != 1 || m.shape(0) != n_m) return false;
    double l2_max = 0.0, m2_max = 0.0;
    for (py::ssize_t i = 0; i < n_l; ++i) {
        const double value = static_cast<const double*>(l.data())[i];
        if (!std::isfinite(value)) return false;
        l2_max = std::max(l2_max, value * value);
    }
    for (py::ssize_t j = 0; j < n_m; ++j) {
        const double value = static_cast<const double*>(m.data())[j];
        if (!std::isfinite(value)) return false;
        m2_max = std::max(m2_max, value * value);
    }
    return l2_max + m2_max < 1.0;
}

bool overlaps(const py::array& a, const py::array& b) {
    const auto* a_begin = static_cast<const char*>(a.data());
    const auto* b_begin = static_cast<const char*>(b.data());
    return a.nbytes() > 0 && b.nbytes() > 0 && a_begin < b_begin + b.nbytes() &&
           b_begin < a_begin + a.nbytes();
}

// A call as the fast path runs it: its view for dft_fast::predict, and its
// input arrays (model, uvw, frequency_coord, the three maps, l, m), each
// aligned (a copy of a misaligned argument), held while it runs.
struct Accepted {
    dft_fast::Call call;
    std::array<py::array, 8> inputs;
};

// The call as the fast path runs it, or nothing if it declines it before the
// decision on the inputs' values.
std::optional<Accepted> accept(const py::args& args, const py::kwargs& kwargs) {
    std::array<py::handle, kParameters.size()> bound{};
    if (!match(args, kwargs, bound)) return std::nullopt;

    // model and vis_data: arrays of any of the reference's dtypes.
    if (!py::isinstance<py::array>(bound[0]) || !py::isinstance<py::array>(bound[1]))
        return std::nullopt;
    const auto model = py::reinterpret_borrow<py::array>(bound[0]);
    auto vis = py::reinterpret_borrow<py::array>(bound[1]);
    const std::optional<Element> model_type = element_of(model);
    const std::optional<Element> vis_type = element_of(vis);
    if (!model_type || !vis_type) return std::nullopt;
    if (*vis_type != Element::complex64 && *vis_type != Element::complex128) return std::nullopt;
    // The typed arguments (uvw is .noconvert(): the reference rejects another).
    const std::optional<py::array> uvw = typed<double>(bound[2]);
    const std::optional<py::array> frequency = typed<double>(bound[3]);
    const std::optional<py::array> frequency_map = typed<i64>(bound[4]);
    const std::optional<py::array> time_map = typed<i64>(bound[5]);
    const std::optional<py::array> pol_map = typed<i64>(bound[6]);
    const std::optional<py::array> l = typed<double>(bound[7]);
    const std::optional<py::array> m = typed<double>(bound[8]);
    if (!uvw || !frequency || !frequency_map || !time_map || !pol_map || !l || !m)
        return std::nullopt;
    int threads = 1;
    if (bound[9]) {
        try {  // pybind11's int caster with conversions, as for the reference
            threads = py::cast<int>(bound[9]);
        } catch (const py::cast_error&) {
            return std::nullopt;
        }
    }

    // The checks of dft_predict_bind on the shapes and layouts, in its order.
    if (model.ndim() != 5 || !c_contiguous(model)) return std::nullopt;
    if (uvw->ndim() != 3 || uvw->shape(2) != 3 || vis.ndim() != 4 ||
        vis.shape(0) != uvw->shape(0) || vis.shape(1) != uvw->shape(1))
        return std::nullopt;
    if (!c_contiguous(vis) || !c_contiguous(*uvw) || !vis.writeable()) return std::nullopt;
    const py::ssize_t n_time = vis.shape(0), n_baseline = vis.shape(1), n_vis_chan = vis.shape(2),
                      n_pol = vis.shape(3);
    const py::ssize_t m_time = model.shape(0), m_chan = model.shape(1), m_pol = model.shape(2),
                      n_l = model.shape(3), n_m = model.shape(4);
    if (frequency->ndim() != 1 || frequency->shape(0) != n_vis_chan) return std::nullopt;
    std::array<py::array, 8> inputs = {model,     *uvw,     *frequency, *frequency_map,
                                       *time_map, *pol_map, *l,         *m};
    for (const py::array& input : inputs) {
        if (overlaps(vis, input)) return std::nullopt;
    }

    // Then its checks of the values, which are read through aligned arrays.
    for (py::array& input : inputs) {
        std::optional<py::array> copy = aligned(input);
        if (!copy) return std::nullopt;
        input = std::move(*copy);
    }
    const auto& [model_in, uvw_in, frequency_in, frequency_map_in, time_map_in, pol_map_in, l_in,
                 m_in] = inputs;
    if (!valid_map(frequency_map_in, n_vis_chan, m_chan) ||
        !valid_map(time_map_in, n_time, m_time) || !valid_map(pol_map_in, n_pol, m_pol) ||
        !valid_direction_cosines(l_in, m_in, n_l, n_m))
        return std::nullopt;

    const dft_fast::Call call{
        model_in.data(),
        vis.mutable_data(),
        static_cast<const double*>(uvw_in.data()),
        static_cast<const double*>(frequency_in.data()),
        static_cast<const i64*>(frequency_map_in.data()),
        static_cast<const i64*>(time_map_in.data()),
        static_cast<const i64*>(pol_map_in.data()),
        static_cast<const double*>(l_in.data()),
        static_cast<const double*>(m_in.data()),
        m_time, m_chan, m_pol, n_l, n_m, n_time, n_baseline, n_vis_chan, n_pol,
        *model_type, *vis_type, threads};
    return Accepted{call, std::move(inputs)};
}

py::str dft_predict(const py::args& args, const py::kwargs& kwargs) {
    const std::optional<Accepted> accepted = accept(args, kwargs);
    if (!accepted) return kDeclined;
    dft_fast::Path path;
    {
        py::gil_scoped_release release;
        path = dft_fast::predict(accepted->call);
    }
    return path == dft_fast::Path::separable ? kSeparable : kDeclined;
}  // the inputs (and their aligned copies) are released here, with the GIL

py::tuple cos_sin_turns(py::array_t<double, py::array::c_style | py::array::forcecast> turns) {
    py::array_t<double> cos_values(turns.size());
    py::array_t<double> sin_values(turns.size());
    dft_fast::kernels().cos_sin_turns(turns.data(), static_cast<i64>(turns.size()),
                                      cos_values.mutable_data(), sin_values.mutable_data());
    return py::make_tuple(cos_values, sin_values);
}

}  // namespace

PYBIND11_MODULE(_dft_fast_ext, module) {
    module.doc() =
        "Fast dft_predict: the exact separable sum at w = 0 (path S), declining every "
        "other call to the reference";
    module.attr("ISA") = dft_fast::kernels().isa;  // chosen once, at this import
    module.attr("DECLINED") = kDeclined;
    module.def("dft_predict", &dft_predict,
               "dft_predict(*args, **kwargs) with the reference's arguments: writes the "
               "visibilities and returns 'S', or writes nothing and returns 'declined' (the "
               "caller then runs the reference).");
    module.def("cos_sin_turns", &cos_sin_turns,
               "cos(2 pi t) and sin(2 pi t) of angles t in turns, by the function of path S's "
               "phasors (the reference's cos_sin_of_turns, with the selected instruction set, "
               "which path S evaluates at its exact reduced phases): bitwise the reference's "
               "cos_sin_turns for finite turns (for NaN or infinite turns both give NaN, not "
               "always with the same sign bit or payload).",
               py::arg("turns"));
}
