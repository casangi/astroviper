#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include "../include/prolate_spheroidal_grid.hpp"
#include "../include/prolate_spheroidal_degrid.hpp"

namespace py = pybind11;

using cdbl = std::complex<double>;
using cflt = std::complex<float>;

// ---------------------------------------------------------------------------
// Helper: raise a clear error if an array is not C-contiguous.
// pybind11 raises its own error for dtype mismatches when forcecast is absent,
// but a contiguity message helps the caller fix the issue quickly.
// ---------------------------------------------------------------------------
static void require_c_contiguous(const py::buffer_info& info, const char* name) {
    // Verify strides match a C-contiguous layout
    py::ssize_t stride = info.itemsize;
    for (int i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != stride)
            throw std::runtime_error(
                std::string(name) + " must be C-contiguous (pass np.ascontiguousarray() if needed)");
        stride *= info.shape[i];
    }
}

// ---------------------------------------------------------------------------
// prolate_spheroidal_grid
//
// The complex UV grid is an image-domain array and may be single (complex64) or
// double (complex128) precision. It is taken as an untyped py::array and the
// binding dispatches on its dtype, so pybind11 cannot silently safe-cast a
// complex64 buffer into a complex128 temporary copy. The visibilities, weights,
// uvw and kernel stay double precision (the visibilities are never down-cast),
// and the normalization accumulator stays float64.
//
// Memory is owned by Python throughout; no large array is copied. Arrays that
// are genuinely tiny (n_uv, delta_lm — 2 elements each) keep forcecast so
// callers aren't forced to match an exact int width.
// ---------------------------------------------------------------------------
void prolate_spheroidal_grid_bind(
    py::array                                                        grid,          // (m_time, m_chan, m_pol, m_u, m_v)  complex64 or complex128, writable
    py::array_t<double,  py::array::c_style>                         normalization, // (m_time, m_chan, m_pol)             float64     writable
    py::array_t<cdbl,    py::array::c_style>                         vis_data,      // (n_time, n_baseline, n_chan, n_pol) complex128
    py::array_t<double,  py::array::c_style>                         uvw,           // (n_time, n_baseline, 3)            float64
    py::array_t<double,  py::array::c_style>                         frequency_coord, // (n_vis_chan,)                    float64
    py::array_t<int64_t, py::array::c_style>                         frequency_map, // (n_vis_chan,)                      int64
    py::array_t<int64_t, py::array::c_style>                         time_map,      // (n_time,)                          int64
    py::array_t<int64_t, py::array::c_style>                         pol_map,       // (n_pol,)                           int64
    py::array_t<double,  py::array::c_style>                         weight,        // (n_time, n_baseline, n_chan, n_pol) float64
    py::array_t<double,  py::array::c_style>                         cgk_1D,        // (oversampling*(support//2+1),)      float64
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>  n_uv,          // (2,) — tiny, forcecast OK
    py::array_t<double,  py::array::c_style | py::array::forcecast>  delta_lm,      // (2,) — tiny, forcecast OK
    int support,
    int oversampling,
    int processing_function_threads
) {
    auto grid_info  = grid.request();
    auto norm_info  = normalization.request();
    auto vis_info   = vis_data.request();
    auto uvw_info   = uvw.request();
    auto wt_info    = weight.request();

    if (grid_info.ndim != 5)
        throw std::runtime_error("grid must be 5-D (m_time, m_chan, m_pol, m_u, m_v)");
    if (norm_info.ndim != 3)
        throw std::runtime_error("normalization must be 3-D (m_time, m_chan, m_pol)");
    if (vis_info.ndim != 4)
        throw std::runtime_error("vis_data must be 4-D (n_time, n_baseline, n_vis_chan, n_pol)");
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");

    require_c_contiguous(grid_info,  "grid");
    require_c_contiguous(norm_info,  "normalization");
    require_c_contiguous(vis_info,   "vis_data");
    require_c_contiguous(uvw_info,   "uvw");
    require_c_contiguous(wt_info,    "weight");

    if (grid_info.readonly)
        throw std::runtime_error("grid must be a writable array (modified in place)");

    const py::dtype dt_c64  = py::dtype::of<cflt>();
    const py::dtype dt_c128 = py::dtype::of<cdbl>();
    const py::dtype grid_dtype = grid.dtype();
    const bool grid_is_c64  = grid_dtype.is(dt_c64);
    const bool grid_is_c128 = grid_dtype.is(dt_c128);
    if (!grid_is_c64 && !grid_is_c128)
        throw std::runtime_error(
            "grid must be complex64 or complex128 (no implicit conversion is "
            "performed; the array is modified in place)");

    const int m_time_g  = static_cast<int>(grid_info.shape[0]);
    const int m_chan_g  = static_cast<int>(grid_info.shape[1]);
    const int m_pol_g   = static_cast<int>(grid_info.shape[2]);
    const int m_u       = static_cast<int>(grid_info.shape[3]);
    const int m_v       = static_cast<int>(grid_info.shape[4]);

    const int n_time     = static_cast<int>(uvw_info.shape[0]);
    const int n_baseline = static_cast<int>(uvw_info.shape[1]);
    const int n_vis_chan = static_cast<int>(frequency_map.shape(0));
    const int n_pol      = static_cast<int>(pol_map.shape(0));

    auto dlm = delta_lm.unchecked<1>();

    // Capture raw pointers before releasing the GIL; py::array_t accessors
    // require it, but the underlying numpy buffers remain valid for the
    // duration of the call because Python owns the arrays.
    double*       norm_ptr          = normalization.mutable_data();
    const cdbl*   vis_ptr           = static_cast<const cdbl*>(vis_info.ptr);
    const double* uvw_ptr           = static_cast<const double*>(uvw_info.ptr);
    const double* freq_ptr          = frequency_coord.data();
    const int64_t* freq_map_ptr     = frequency_map.data();
    const int64_t* time_map_ptr     = time_map.data();
    const int64_t* pol_map_ptr      = pol_map.data();
    const double* wt_ptr            = static_cast<const double*>(wt_info.ptr);
    const double* cgk_ptr           = cgk_1D.data();
    const double  dl                = dlm(0);
    const double  dm                = dlm(1);

    if (grid_is_c128) {
        cdbl* grid_ptr = static_cast<cdbl*>(grid_info.ptr);
        py::gil_scoped_release release;
        prolate_spheroidal::prolate_spheroidal_grid<double>(
            grid_ptr, norm_ptr, vis_ptr, uvw_ptr, freq_ptr,
            freq_map_ptr, time_map_ptr, pol_map_ptr, wt_ptr, cgk_ptr,
            m_time_g, m_chan_g, m_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol,
            dl, dm, support, oversampling, processing_function_threads);
    } else {
        cflt* grid_ptr = static_cast<cflt*>(grid_info.ptr);
        py::gil_scoped_release release;
        prolate_spheroidal::prolate_spheroidal_grid<float>(
            grid_ptr, norm_ptr, vis_ptr, uvw_ptr, freq_ptr,
            freq_map_ptr, time_map_ptr, pol_map_ptr, wt_ptr, cgk_ptr,
            m_time_g, m_chan_g, m_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol,
            dl, dm, support, oversampling, processing_function_threads);
    }
}

// ---------------------------------------------------------------------------
// prolate_spheroidal_grid_uv_sampling
// ---------------------------------------------------------------------------
void prolate_spheroidal_grid_uv_sampling_bind(
    py::array                                                        grid,
    py::array_t<double,  py::array::c_style>                         normalization,
    py::array_t<double,  py::array::c_style>                         uvw,
    py::array_t<double,  py::array::c_style>                         frequency_coord,
    py::array_t<int64_t, py::array::c_style>                         frequency_map,
    py::array_t<int64_t, py::array::c_style>                         time_map,
    py::array_t<int64_t, py::array::c_style>                         pol_map,
    py::array_t<double,  py::array::c_style>                         weight,
    py::array_t<double,  py::array::c_style>                         cgk_1D,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>  n_uv,
    py::array_t<double,  py::array::c_style | py::array::forcecast>  delta_lm,
    int support,
    int oversampling,
    int processing_function_threads
) {
    auto grid_info = grid.request();
    auto norm_info = normalization.request();
    auto uvw_info  = uvw.request();
    auto wt_info   = weight.request();

    if (grid_info.ndim != 5)
        throw std::runtime_error("grid must be 5-D (m_time, m_chan, m_pol, m_u, m_v)");
    if (norm_info.ndim != 3)
        throw std::runtime_error("normalization must be 3-D (m_time, m_chan, m_pol)");
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");

    require_c_contiguous(grid_info, "grid");
    require_c_contiguous(norm_info, "normalization");
    require_c_contiguous(uvw_info,  "uvw");
    require_c_contiguous(wt_info,   "weight");

    if (grid_info.readonly)
        throw std::runtime_error("grid must be a writable array (modified in place)");

    const py::dtype dt_c64  = py::dtype::of<cflt>();
    const py::dtype dt_c128 = py::dtype::of<cdbl>();
    const py::dtype grid_dtype = grid.dtype();
    const bool grid_is_c64  = grid_dtype.is(dt_c64);
    const bool grid_is_c128 = grid_dtype.is(dt_c128);
    if (!grid_is_c64 && !grid_is_c128)
        throw std::runtime_error(
            "grid must be complex64 or complex128 (no implicit conversion is "
            "performed; the array is modified in place)");

    const int m_time_g  = static_cast<int>(grid_info.shape[0]);
    const int m_chan_g  = static_cast<int>(grid_info.shape[1]);
    const int m_pol_g   = static_cast<int>(grid_info.shape[2]);
    const int m_u       = static_cast<int>(grid_info.shape[3]);
    const int m_v       = static_cast<int>(grid_info.shape[4]);

    const int n_time     = static_cast<int>(uvw_info.shape[0]);
    const int n_baseline = static_cast<int>(uvw_info.shape[1]);
    const int n_vis_chan = static_cast<int>(frequency_map.shape(0));
    const int n_pol      = static_cast<int>(pol_map.shape(0));

    auto dlm = delta_lm.unchecked<1>();

    double*       norm_ptr          = normalization.mutable_data();
    const double* uvw_ptr           = static_cast<const double*>(uvw_info.ptr);
    const double* freq_ptr          = frequency_coord.data();
    const int64_t* freq_map_ptr     = frequency_map.data();
    const int64_t* time_map_ptr     = time_map.data();
    const int64_t* pol_map_ptr      = pol_map.data();
    const double* wt_ptr            = static_cast<const double*>(wt_info.ptr);
    const double* cgk_ptr           = cgk_1D.data();
    const double  dl                = dlm(0);
    const double  dm                = dlm(1);

    if (grid_is_c128) {
        cdbl* grid_ptr = static_cast<cdbl*>(grid_info.ptr);
        py::gil_scoped_release release;
        prolate_spheroidal::prolate_spheroidal_grid_uv_sampling<double>(
            grid_ptr, norm_ptr, uvw_ptr, freq_ptr,
            freq_map_ptr, time_map_ptr, pol_map_ptr, wt_ptr, cgk_ptr,
            m_time_g, m_chan_g, m_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol,
            dl, dm, support, oversampling, processing_function_threads);
    } else {
        cflt* grid_ptr = static_cast<cflt*>(grid_info.ptr);
        py::gil_scoped_release release;
        prolate_spheroidal::prolate_spheroidal_grid_uv_sampling<float>(
            grid_ptr, norm_ptr, uvw_ptr, freq_ptr,
            freq_map_ptr, time_map_ptr, pol_map_ptr, wt_ptr, cgk_ptr,
            m_time_g, m_chan_g, m_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol,
            dl, dm, support, oversampling, processing_function_threads);
    }
}

// ---------------------------------------------------------------------------
// prolate_spheroidal_degrid
//
// Inverse of prolate_spheroidal_grid: samples the model `grid` at each
// visibility coordinate using the same prolate spheroidal kernel, and writes
// the convolution-interpolated value into `vis_data` in place.
//
// Both `grid` (a single- or double-precision image-domain array) and
// `vis_data` are taken as untyped py::array so pybind11 cannot silently
// safe-cast either buffer into a temporary copy. Layout, dtype, and
// writeability are checked explicitly; the four (grid dtype) x (vis dtype)
// combinations are dispatched to the templated kernel and written in place.
// ---------------------------------------------------------------------------
void prolate_spheroidal_degrid_bind(
    py::array                                                        grid,          // (m_time, m_chan, m_pol, m_u, m_v)   complex64 or complex128, read
    py::array                                                        vis_data,      // (n_time, n_baseline, n_chan, n_pol) complex64 or complex128, writable
    py::array_t<double,  py::array::c_style>                         uvw,           // (n_time, n_baseline, 3)             float64
    py::array_t<double,  py::array::c_style>                         frequency_coord, // (n_vis_chan,)                     float64
    py::array_t<int64_t, py::array::c_style>                         frequency_map, // (n_vis_chan,)                       int64
    py::array_t<int64_t, py::array::c_style>                         time_map,      // (n_time,)                           int64
    py::array_t<int64_t, py::array::c_style>                         pol_map,       // (n_pol,)                            int64
    py::array_t<double,  py::array::c_style>                         cgk_1D,        // (oversampling*(support//2+1),)      float64
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>  n_uv,          // (2,) — tiny, forcecast OK
    py::array_t<double,  py::array::c_style | py::array::forcecast>  delta_lm,      // (2,) — tiny, forcecast OK
    int support,
    int oversampling,
    int processing_function_threads
) {
    auto grid_info = grid.request();
    auto vis_info  = vis_data.request();
    auto uvw_info  = uvw.request();

    if (grid_info.ndim != 5)
        throw std::runtime_error("grid must be 5-D (m_time, m_chan, m_pol, m_u, m_v)");
    if (vis_info.ndim != 4)
        throw std::runtime_error("vis_data must be 4-D (n_time, n_baseline, n_vis_chan, n_pol)");
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");

    require_c_contiguous(grid_info, "grid");
    require_c_contiguous(vis_info,  "vis_data");
    require_c_contiguous(uvw_info,  "uvw");

    if (vis_info.readonly)
        throw std::runtime_error("vis_data must be a writable array");

    const py::dtype dt_c64  = py::dtype::of<cflt>();
    const py::dtype dt_c128 = py::dtype::of<cdbl>();

    const py::dtype grid_dtype = grid.dtype();
    const bool grid_is_c64  = grid_dtype.is(dt_c64);
    const bool grid_is_c128 = grid_dtype.is(dt_c128);
    if (!grid_is_c64 && !grid_is_c128)
        throw std::runtime_error(
            "grid must be complex64 or complex128 (no implicit conversion is "
            "performed)");

    const py::dtype vis_dtype = vis_data.dtype();
    const bool vis_is_c64  = vis_dtype.is(dt_c64);
    const bool vis_is_c128 = vis_dtype.is(dt_c128);
    if (!vis_is_c64 && !vis_is_c128)
        throw std::runtime_error(
            "vis_data must be complex64 or complex128 (no implicit conversion "
            "is performed; the array is modified in place)");

    const int m_time_g = static_cast<int>(grid_info.shape[0]);
    const int m_chan_g = static_cast<int>(grid_info.shape[1]);
    const int m_pol_g  = static_cast<int>(grid_info.shape[2]);
    const int m_u      = static_cast<int>(grid_info.shape[3]);
    const int m_v      = static_cast<int>(grid_info.shape[4]);

    const int n_time     = static_cast<int>(uvw_info.shape[0]);
    const int n_baseline = static_cast<int>(uvw_info.shape[1]);
    const int n_vis_chan = static_cast<int>(frequency_map.shape(0));
    const int n_pol      = static_cast<int>(pol_map.shape(0));

    auto dlm = delta_lm.unchecked<1>();

    const void*    grid_ptr_raw  = grid_info.ptr;
    void*          vis_ptr_raw   = vis_info.ptr;
    const double*  uvw_ptr       = static_cast<const double*>(uvw_info.ptr);
    const double*  freq_ptr      = frequency_coord.data();
    const int64_t* freq_map_ptr  = frequency_map.data();
    const int64_t* time_map_ptr  = time_map.data();
    const int64_t* pol_map_ptr   = pol_map.data();
    const double*  cgk_ptr       = cgk_1D.data();
    const double   dl            = dlm(0);
    const double   dm            = dlm(1);

    // Dispatch on (grid dtype) x (vis dtype). The grid/vis_info.ptr alias the
    // Python-owned buffers directly; no copy occurs.
    auto run = [&](auto grid_tag, auto vis_tag) {
        using GridT = decltype(grid_tag);
        using VisElem = decltype(vis_tag);
        using VisT = std::complex<VisElem>;
        const std::complex<GridT>* gp =
            static_cast<const std::complex<GridT>*>(grid_ptr_raw);
        VisT* vp = static_cast<VisT*>(vis_ptr_raw);
        py::gil_scoped_release release;
        prolate_spheroidal::prolate_spheroidal_degrid<GridT, VisT>(
            gp, vp, uvw_ptr, freq_ptr,
            freq_map_ptr, time_map_ptr, pol_map_ptr, cgk_ptr,
            m_time_g, m_chan_g, m_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol,
            dl, dm, support, oversampling, processing_function_threads);
    };

    if (grid_is_c128 && vis_is_c128) {
        run(double{}, double{});
    } else if (grid_is_c128 && vis_is_c64) {
        run(double{}, float{});
    } else if (grid_is_c64 && vis_is_c128) {
        run(float{}, double{});
    } else {
        run(float{}, float{});
    }
}

PYBIND11_MODULE(_prolate_spheroidal_grid_ext, m) {
    m.doc() = "C++ prolate spheroidal convolution gridder (UV gridding)";

    m.def("prolate_spheroidal_grid",
          &prolate_spheroidal_grid_bind,
          "Grid weighted visibilities onto a UV plane using a prolate spheroidal "
          "kernel (in-place). The grid may be complex64 or complex128; the "
          "visibilities stay complex128. processing_function_threads selects the number of "
          "std::thread workers; <= 0 falls back to "
          "std::thread::hardware_concurrency().",
          py::arg("grid"),
          py::arg("normalization"),
          py::arg("vis_data"),
          py::arg("uvw"),
          py::arg("frequency_coord"),
          py::arg("frequency_map"),
          py::arg("time_map"),
          py::arg("pol_map"),
          py::arg("weight"),
          py::arg("cgk_1D"),
          py::arg("n_uv"),
          py::arg("delta_lm"),
          py::arg("support"),
          py::arg("oversampling"),
          py::arg("processing_function_threads") = 1);

    m.def("prolate_spheroidal_grid_uv_sampling",
          &prolate_spheroidal_grid_uv_sampling_bind,
          "Grid imaging weights onto a UV plane (PSF / UV-sampling function, "
          "in-place). The grid may be complex64 or complex128. processing_function_threads "
          "selects the number of std::thread workers; <= 0 falls back to "
          "std::thread::hardware_concurrency().",
          py::arg("grid"),
          py::arg("normalization"),
          py::arg("uvw"),
          py::arg("frequency_coord"),
          py::arg("frequency_map"),
          py::arg("time_map"),
          py::arg("pol_map"),
          py::arg("weight"),
          py::arg("cgk_1D"),
          py::arg("n_uv"),
          py::arg("delta_lm"),
          py::arg("support"),
          py::arg("oversampling"),
          py::arg("processing_function_threads") = 1);

    m.def("prolate_spheroidal_degrid",
          &prolate_spheroidal_degrid_bind,
          "Sample (degrid) a model UV grid at visibility coordinates using a "
          "prolate spheroidal kernel. The model grid may be complex64 or "
          "complex128 and the visibilities complex64 or complex128; writes "
          "interpolated values into vis_data in place. processing_function_threads selects the "
          "number of std::thread workers; <= 0 falls back to "
          "std::thread::hardware_concurrency().",
          py::arg("grid"),
          py::arg("vis_data"),
          py::arg("uvw"),
          py::arg("frequency_coord"),
          py::arg("frequency_map"),
          py::arg("time_map"),
          py::arg("pol_map"),
          py::arg("cgk_1D"),
          py::arg("n_uv"),
          py::arg("delta_lm"),
          py::arg("support"),
          py::arg("oversampling"),
          py::arg("processing_function_threads") = 1);
}
