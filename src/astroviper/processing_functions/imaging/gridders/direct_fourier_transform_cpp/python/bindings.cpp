#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "../include/direct_fourier_transform.hpp"

namespace py = pybind11;

using cdbl = std::complex<double>;
using cflt = std::complex<float>;

// ---------------------------------------------------------------------------
// Helpers: clear errors for a non-C-contiguous or read-only array, a shape
// that does not match, and a map entry outside its output axis (the kernels
// index the outputs with the maps unchecked).
// ---------------------------------------------------------------------------
static void require_c_contiguous(const py::buffer_info& info, const char* name) {
    py::ssize_t stride = info.itemsize;
    for (int i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != stride)
            throw std::runtime_error(
                std::string(name) + " must be C-contiguous (pass np.ascontiguousarray() if needed)");
        stride *= info.shape[i];
    }
}

static void require_writable(const py::buffer_info& info, const char* name) {
    if (info.readonly)
        throw std::runtime_error(
            std::string(name) + " must be a writable array (modified in place)");
}

static void require_map(const py::array_t<int64_t, py::array::c_style>& map, py::ssize_t length,
                        int64_t limit, const char* name) {
    if (map.ndim() != 1 || map.shape(0) != length)
        throw std::runtime_error(std::string(name) + " must have one entry per visibility " +
                                 "index along its axis (" + std::to_string(length) + ")");
    const int64_t* values = map.data();
    for (py::ssize_t i = 0; i < length; ++i) {
        if (values[i] < 0 || values[i] >= limit)
            throw std::runtime_error(std::string(name) + " entry " + std::to_string(values[i]) +
                                     " is outside the image axis of length " +
                                     std::to_string(limit));
    }
}

// The image pixels must lie in front of the horizon (n > 0): the imaging
// multiplies by n and the prediction divides by it.
static void require_direction_cosines(const py::array_t<double, py::array::c_style>& l,
                                      const py::array_t<double, py::array::c_style>& m,
                                      py::ssize_t n_l, py::ssize_t n_m) {
    if (l.ndim() != 1 || l.shape(0) != n_l)
        throw std::runtime_error("l must have one entry per image row (" + std::to_string(n_l) + ")");
    if (m.ndim() != 1 || m.shape(0) != n_m)
        throw std::runtime_error("m must have one entry per image column (" + std::to_string(n_m) +
                                 ")");
    double l2_max = 0.0, m2_max = 0.0;
    for (py::ssize_t i = 0; i < n_l; ++i) {
        const double value = l.data()[i];
        if (!std::isfinite(value)) throw std::runtime_error("l must be finite");
        l2_max = std::max(l2_max, value * value);
    }
    for (py::ssize_t j = 0; j < n_m; ++j) {
        const double value = m.data()[j];
        if (!std::isfinite(value)) throw std::runtime_error("m must be finite");
        m2_max = std::max(m2_max, value * value);
    }
    if (!(l2_max + m2_max < 1.0))
        throw std::runtime_error(
            "image pixels must lie within the celestial hemisphere of the phase centre "
            "(l^2 + m^2 < 1)");
}

struct VisibilityShape {
    int64_t n_time, n_baseline, n_vis_chan, n_pol;
};

static VisibilityShape visibility_shape(const py::buffer_info& uvw_info,
                                        const py::buffer_info& data_info, const char* name) {
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");
    if (data_info.ndim != 4)
        throw std::runtime_error(std::string(name) +
                                 " must be 4-D (n_time, n_baseline, n_vis_chan, n_pol)");
    if (data_info.shape[0] != uvw_info.shape[0] || data_info.shape[1] != uvw_info.shape[1])
        throw std::runtime_error(std::string(name) +
                                 " and uvw must have the same time and baseline axes");
    return {static_cast<int64_t>(data_info.shape[0]), static_cast<int64_t>(data_info.shape[1]),
            static_cast<int64_t>(data_info.shape[2]), static_cast<int64_t>(data_info.shape[3])};
}

// ---------------------------------------------------------------------------
// dft_image / dft_image_uv_sampling
//
// The image is an image-domain accumulator of float32, float64, complex64 or
// complex128 (float32/float64 only for the uv sampling): it is taken as an
// untyped py::array and the binding dispatches on its dtype, so pybind11
// cannot silently cast it into a temporary copy (which would drop the
// writes). The normalization accumulator is float64, checked the same way.
// The large read-only inputs (visibilities, weights, uvw) are taken with
// .noconvert(): an array of another dtype or layout is refused instead of
// copied. Python owns every array; the GIL is released during the compute.
// ---------------------------------------------------------------------------
static void check_image(const py::buffer_info& image_info, const py::buffer_info& norm_info,
                        const py::array& normalization) {
    if (image_info.ndim != 5)
        throw std::runtime_error("image must be 5-D (m_time, m_chan, m_pol, n_l, n_m)");
    if (norm_info.ndim != 3 || norm_info.shape[0] != image_info.shape[0] ||
        norm_info.shape[1] != image_info.shape[1] || norm_info.shape[2] != image_info.shape[2])
        throw std::runtime_error("normalization must have shape (m_time, m_chan, m_pol) of image");
    require_c_contiguous(image_info, "image");
    require_c_contiguous(norm_info, "normalization");
    require_writable(image_info, "image");
    require_writable(norm_info, "normalization");
    if (!normalization.dtype().is(py::dtype::of<double>()))
        throw std::runtime_error(
            "normalization must be float64 (no implicit conversion is performed; the array is "
            "modified in place)");
}

void dft_image_bind(
    py::array                                                        image,           // (m_time, m_chan, m_pol, n_l, n_m) float32/float64/complex64/complex128, writable
    py::array                                                        normalization,   // (m_time, m_chan, m_pol) float64, writable
    py::array_t<cdbl,    py::array::c_style>                         vis_data,        // (n_time, n_baseline, n_vis_chan, n_pol) complex128
    py::array_t<double,  py::array::c_style>                         uvw,             // (n_time, n_baseline, 3) float64, metres
    py::array_t<double,  py::array::c_style | py::array::forcecast>  frequency_coord, // (n_vis_chan,) Hz
    py::array_t<int64_t, py::array::c_style>                         frequency_map,   // (n_vis_chan,)
    py::array_t<int64_t, py::array::c_style>                         time_map,        // (n_time,)
    py::array_t<int64_t, py::array::c_style>                         pol_map,         // (n_pol,)
    py::array_t<double,  py::array::c_style>                         weight,          // (n_time, n_baseline, n_vis_chan, n_pol) float64
    py::array_t<double,  py::array::c_style | py::array::forcecast>  l,               // (n_l,) direction cosines
    py::array_t<double,  py::array::c_style | py::array::forcecast>  m,               // (n_m,) direction cosines
    int processing_function_threads
) {
    auto image_info = image.request();
    auto norm_info = normalization.request();
    auto vis_info = vis_data.request();
    auto uvw_info = uvw.request();
    auto wt_info = weight.request();

    check_image(image_info, norm_info, normalization);
    const VisibilityShape s = visibility_shape(uvw_info, vis_info, "vis_data");
    if (wt_info.ndim != 4 || wt_info.shape != vis_info.shape)
        throw std::runtime_error("weight must have the shape of vis_data");
    require_c_contiguous(vis_info, "vis_data");
    require_c_contiguous(uvw_info, "uvw");
    require_c_contiguous(wt_info, "weight");

    const int64_t m_time = image_info.shape[0], m_chan = image_info.shape[1],
                  m_pol = image_info.shape[2], n_l = image_info.shape[3],
                  n_m = image_info.shape[4];
    if (frequency_coord.ndim() != 1 || frequency_coord.shape(0) != s.n_vis_chan)
        throw std::runtime_error("frequency_coord must have one entry per visibility channel");
    require_map(frequency_map, s.n_vis_chan, m_chan, "frequency_map");
    require_map(time_map, s.n_time, m_time, "time_map");
    require_map(pol_map, s.n_pol, m_pol, "pol_map");
    require_direction_cosines(l, m, n_l, n_m);

    const py::dtype dtype = image.dtype();
    double* norm_ptr = static_cast<double*>(norm_info.ptr);
    const cdbl* vis_ptr = static_cast<const cdbl*>(vis_info.ptr);
    const double* uvw_ptr = static_cast<const double*>(uvw_info.ptr);
    const double* wt_ptr = static_cast<const double*>(wt_info.ptr);
    const double* freq_ptr = frequency_coord.data();
    const int64_t* freq_map_ptr = frequency_map.data();
    const int64_t* time_map_ptr = time_map.data();
    const int64_t* pol_map_ptr = pol_map.data();
    const double* l_ptr = l.data();
    const double* m_ptr = m.data();

    auto run = [&](auto image_tag) {
        using ImageT = decltype(image_tag);
        ImageT* image_ptr = static_cast<ImageT*>(image_info.ptr);
        py::gil_scoped_release release;
        direct_fourier_transform::dft_image<ImageT>(
            image_ptr, norm_ptr, vis_ptr, uvw_ptr, freq_ptr, freq_map_ptr, time_map_ptr,
            pol_map_ptr, wt_ptr, l_ptr, m_ptr, m_time, m_chan, m_pol, n_l, n_m, s.n_time,
            s.n_baseline, s.n_vis_chan, s.n_pol, processing_function_threads);
    };
    if (dtype.is(py::dtype::of<double>())) {
        run(double{});
    } else if (dtype.is(py::dtype::of<float>())) {
        run(float{});
    } else if (dtype.is(py::dtype::of<cdbl>())) {
        run(cdbl{});
    } else if (dtype.is(py::dtype::of<cflt>())) {
        run(cflt{});
    } else {
        throw std::runtime_error(
            "image must be float32, float64, complex64 or complex128 (no implicit conversion is "
            "performed; the array is modified in place)");
    }
}

void dft_image_uv_sampling_bind(
    py::array                                                        image,           // (m_time, m_chan, m_pol, n_l, n_m) float32/float64, writable
    py::array                                                        normalization,   // (m_time, m_chan, m_pol) float64, writable
    py::array_t<double,  py::array::c_style>                         uvw,             // (n_time, n_baseline, 3) float64, metres
    py::array_t<double,  py::array::c_style | py::array::forcecast>  frequency_coord, // (n_vis_chan,) Hz
    py::array_t<int64_t, py::array::c_style>                         frequency_map,   // (n_vis_chan,)
    py::array_t<int64_t, py::array::c_style>                         time_map,        // (n_time,)
    py::array_t<int64_t, py::array::c_style>                         pol_map,         // (n_pol,)
    py::array_t<double,  py::array::c_style>                         weight,          // (n_time, n_baseline, n_vis_chan, n_pol) float64
    py::array_t<double,  py::array::c_style | py::array::forcecast>  l,               // (n_l,)
    py::array_t<double,  py::array::c_style | py::array::forcecast>  m,               // (n_m,)
    int processing_function_threads
) {
    auto image_info = image.request();
    auto norm_info = normalization.request();
    auto uvw_info = uvw.request();
    auto wt_info = weight.request();

    check_image(image_info, norm_info, normalization);
    const VisibilityShape s = visibility_shape(uvw_info, wt_info, "weight");
    require_c_contiguous(uvw_info, "uvw");
    require_c_contiguous(wt_info, "weight");

    const int64_t m_time = image_info.shape[0], m_chan = image_info.shape[1],
                  m_pol = image_info.shape[2], n_l = image_info.shape[3],
                  n_m = image_info.shape[4];
    if (frequency_coord.ndim() != 1 || frequency_coord.shape(0) != s.n_vis_chan)
        throw std::runtime_error("frequency_coord must have one entry per visibility channel");
    require_map(frequency_map, s.n_vis_chan, m_chan, "frequency_map");
    require_map(time_map, s.n_time, m_time, "time_map");
    require_map(pol_map, s.n_pol, m_pol, "pol_map");
    require_direction_cosines(l, m, n_l, n_m);

    const py::dtype dtype = image.dtype();
    double* norm_ptr = static_cast<double*>(norm_info.ptr);
    const double* uvw_ptr = static_cast<const double*>(uvw_info.ptr);
    const double* wt_ptr = static_cast<const double*>(wt_info.ptr);
    const double* freq_ptr = frequency_coord.data();
    const int64_t* freq_map_ptr = frequency_map.data();
    const int64_t* time_map_ptr = time_map.data();
    const int64_t* pol_map_ptr = pol_map.data();
    const double* l_ptr = l.data();
    const double* m_ptr = m.data();

    auto run = [&](auto image_tag) {
        using ImageT = decltype(image_tag);
        ImageT* image_ptr = static_cast<ImageT*>(image_info.ptr);
        py::gil_scoped_release release;
        direct_fourier_transform::dft_image_uv_sampling<ImageT>(
            image_ptr, norm_ptr, uvw_ptr, freq_ptr, freq_map_ptr, time_map_ptr, pol_map_ptr,
            wt_ptr, l_ptr, m_ptr, m_time, m_chan, m_pol, n_l, n_m, s.n_time, s.n_baseline,
            s.n_vis_chan, s.n_pol, processing_function_threads);
    };
    if (dtype.is(py::dtype::of<double>())) {
        run(double{});
    } else if (dtype.is(py::dtype::of<float>())) {
        run(float{});
    } else {
        throw std::runtime_error(
            "image must be float32 or float64 (no implicit conversion is performed; the array "
            "is modified in place)");
    }
}

// ---------------------------------------------------------------------------
// dft_predict
//
// The model (an image-domain array, real or complex, single or double
// precision) and the visibilities (complex64 or complex128, written in place)
// are untyped py::array, dispatched on their dtypes.
// ---------------------------------------------------------------------------
void dft_predict_bind(
    py::array                                                        model,           // (m_time, m_chan, m_pol, n_l, n_m) float32/float64/complex64/complex128
    py::array                                                        vis_data,        // (n_time, n_baseline, n_vis_chan, n_pol) complex64/complex128, writable
    py::array_t<double,  py::array::c_style>                         uvw,             // (n_time, n_baseline, 3) float64, metres
    py::array_t<double,  py::array::c_style | py::array::forcecast>  frequency_coord, // (n_vis_chan,) Hz
    py::array_t<int64_t, py::array::c_style>                         frequency_map,   // (n_vis_chan,)
    py::array_t<int64_t, py::array::c_style>                         time_map,        // (n_time,)
    py::array_t<int64_t, py::array::c_style>                         pol_map,         // (n_pol,)
    py::array_t<double,  py::array::c_style | py::array::forcecast>  l,               // (n_l,)
    py::array_t<double,  py::array::c_style | py::array::forcecast>  m,               // (n_m,)
    int processing_function_threads
) {
    auto model_info = model.request();
    auto vis_info = vis_data.request();
    auto uvw_info = uvw.request();

    if (model_info.ndim != 5)
        throw std::runtime_error("model must be 5-D (m_time, m_chan, m_pol, n_l, n_m)");
    require_c_contiguous(model_info, "model");
    const VisibilityShape s = visibility_shape(uvw_info, vis_info, "vis_data");
    require_c_contiguous(vis_info, "vis_data");
    require_c_contiguous(uvw_info, "uvw");
    require_writable(vis_info, "vis_data");

    const int64_t m_time = model_info.shape[0], m_chan = model_info.shape[1],
                  m_pol = model_info.shape[2], n_l = model_info.shape[3],
                  n_m = model_info.shape[4];
    if (frequency_coord.ndim() != 1 || frequency_coord.shape(0) != s.n_vis_chan)
        throw std::runtime_error("frequency_coord must have one entry per visibility channel");
    require_map(frequency_map, s.n_vis_chan, m_chan, "frequency_map");
    require_map(time_map, s.n_time, m_time, "time_map");
    require_map(pol_map, s.n_pol, m_pol, "pol_map");
    require_direction_cosines(l, m, n_l, n_m);

    const py::dtype model_dtype = model.dtype();
    const py::dtype vis_dtype = vis_data.dtype();
    const bool vis_is_c128 = vis_dtype.is(py::dtype::of<cdbl>());
    const bool vis_is_c64 = vis_dtype.is(py::dtype::of<cflt>());
    if (!vis_is_c128 && !vis_is_c64)
        throw std::runtime_error(
            "vis_data must be complex64 or complex128 (no implicit conversion is performed; the "
            "array is modified in place)");

    const void* model_ptr = model_info.ptr;
    void* vis_ptr = vis_info.ptr;
    const double* uvw_ptr = static_cast<const double*>(uvw_info.ptr);
    const double* freq_ptr = frequency_coord.data();
    const int64_t* freq_map_ptr = frequency_map.data();
    const int64_t* time_map_ptr = time_map.data();
    const int64_t* pol_map_ptr = pol_map.data();
    const double* l_ptr = l.data();
    const double* m_ptr = m.data();

    auto run = [&](auto model_tag, auto vis_tag) {
        using ModelT = decltype(model_tag);
        using VisT = decltype(vis_tag);
        const ModelT* mp = static_cast<const ModelT*>(model_ptr);
        VisT* vp = static_cast<VisT*>(vis_ptr);
        py::gil_scoped_release release;
        direct_fourier_transform::dft_predict<ModelT, VisT>(
            mp, vp, uvw_ptr, freq_ptr, freq_map_ptr, time_map_ptr, pol_map_ptr, l_ptr, m_ptr,
            m_time, m_chan, m_pol, n_l, n_m, s.n_time, s.n_baseline, s.n_vis_chan, s.n_pol,
            processing_function_threads);
    };
    auto dispatch_vis = [&](auto model_tag) {
        if (vis_is_c128) {
            run(model_tag, cdbl{});
        } else {
            run(model_tag, cflt{});
        }
    };
    if (model_dtype.is(py::dtype::of<double>())) {
        dispatch_vis(double{});
    } else if (model_dtype.is(py::dtype::of<float>())) {
        dispatch_vis(float{});
    } else if (model_dtype.is(py::dtype::of<cdbl>())) {
        dispatch_vis(cdbl{});
    } else if (model_dtype.is(py::dtype::of<cflt>())) {
        dispatch_vis(cflt{});
    } else {
        throw std::runtime_error(
            "model must be float32, float64, complex64 or complex128 (no implicit conversion is "
            "performed)");
    }
}

py::tuple cos_sin_turns_bind(py::array_t<double, py::array::c_style | py::array::forcecast> turns) {
    py::array_t<double> cos_values(turns.size());
    py::array_t<double> sin_values(turns.size());
    const double* in = turns.data();
    double* c = cos_values.mutable_data();
    double* s = sin_values.mutable_data();
    const int64_t count = static_cast<int64_t>(turns.size());
    {
        py::gil_scoped_release release;
        direct_fourier_transform::cos_sin_turns(in, c, s, count);
    }
    return py::make_tuple(cos_values, sin_values);
}

PYBIND11_MODULE(_direct_fourier_transform_ext, module) {
    module.doc() =
        "C++ direct Fourier transform between visibilities and an image, with the exact w term";

    module.def("dft_image",
               &dft_image_bind,
               "Accumulate n * sum(weight * vis * exp(-2 pi i (u l + v m + w (n - 1)))) into the "
               "image and the sum of the weights into normalization, in place (the real part for "
               "a float32/float64 image). processing_function_threads selects the number of "
               "std::thread workers; <= 0 falls back to std::thread::hardware_concurrency().",
               py::arg("image"),
               py::arg("normalization"),
               py::arg("vis_data").noconvert(),
               py::arg("uvw").noconvert(),
               py::arg("frequency_coord"),
               py::arg("frequency_map"),
               py::arg("time_map"),
               py::arg("pol_map"),
               py::arg("weight").noconvert(),
               py::arg("l"),
               py::arg("m"),
               py::arg("processing_function_threads") = 1);

    module.def("dft_image_uv_sampling",
               &dft_image_uv_sampling_bind,
               "Accumulate n * sum(weight * cos(2 pi (u l + v m + w (n - 1)))) into the image "
               "and the sum of the weights into normalization, in place (the point spread "
               "function before its normalization). processing_function_threads selects the "
               "number of std::thread workers; <= 0 falls back to "
               "std::thread::hardware_concurrency().",
               py::arg("image"),
               py::arg("normalization"),
               py::arg("uvw").noconvert(),
               py::arg("frequency_coord"),
               py::arg("frequency_map"),
               py::arg("time_map"),
               py::arg("pol_map"),
               py::arg("weight").noconvert(),
               py::arg("l"),
               py::arg("m"),
               py::arg("processing_function_threads") = 1);

    module.def("dft_predict",
               &dft_predict_bind,
               "Write sum(model * exp(+2 pi i (u l + v m + w (n - 1))) / n) over the non-zero "
               "model pixels into vis_data in place (samples whose uvw is not finite or whose "
               "current value is NaN are left unchanged). processing_function_threads selects the "
               "number of std::thread workers; <= 0 falls back to "
               "std::thread::hardware_concurrency().",
               py::arg("model"),
               py::arg("vis_data"),
               py::arg("uvw").noconvert(),
               py::arg("frequency_coord"),
               py::arg("frequency_map"),
               py::arg("time_map"),
               py::arg("pol_map"),
               py::arg("l"),
               py::arg("m"),
               py::arg("processing_function_threads") = 1);

    module.def("cos_sin_turns",
               &cos_sin_turns_bind,
               "cos(2 pi t) and sin(2 pi t) of angles t in turns, as the transforms evaluate "
               "them.",
               py::arg("turns"));
}
