#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include "../include/grid_imaging_weights.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helper: raise a clear error if an array is not C-contiguous.
// pybind11 raises its own error for dtype mismatches when forcecast is absent,
// but a contiguity message helps the caller fix the issue quickly.
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

// ---------------------------------------------------------------------------
// grid_imaging_weights
//
// The weight-density grid is an image-domain array and may be single (float32)
// or double (float64) precision. It is taken as an untyped py::array and the
// binding dispatches on its dtype, so pybind11 cannot silently safe-cast a
// float32 buffer into a float64 temporary copy. Every other large array stays
// double precision. Memory is owned by Python throughout; no large array is
// copied. The kernel itself is serial so weight sums stay bit-reproducible;
// processing_function_threads is accepted for API consistency across the
// stack and ignored.
// ---------------------------------------------------------------------------
void grid_imaging_weights_bind(
    py::array                                                        grid,        // (n_chan_g, n_pol_g, m_u, m_v)  float32 or float64, writable
    py::array_t<double,  py::array::c_style>                         sum_weight,  // (n_chan_g, n_pol_g)             float64, writable
    py::array_t<double,  py::array::c_style>                         uvw,         // (n_time, n_baseline, 3)        float64
    py::array_t<double,  py::array::c_style>                         frequency_coord, // (n_vis_chan,)              float64
    py::array_t<int64_t, py::array::c_style>                         chan_map,    // (n_vis_chan,)                  int64
    py::array_t<double,  py::array::c_style>                         data_weights, // (n_time, n_baseline, n_vis_chan, n_pol) float64
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>  n_uv,        // (2,) — tiny, forcecast OK
    py::array_t<double,  py::array::c_style | py::array::forcecast>  delta_lm,    // (2,) — tiny, forcecast OK
    int processing_function_threads
) {
    (void)processing_function_threads;  // serial kernel: bit-reproducible sums

    auto grid_info = grid.request();
    auto sum_info  = sum_weight.request();
    auto uvw_info  = uvw.request();
    auto wt_info   = data_weights.request();

    if (grid_info.ndim != 4)
        throw std::runtime_error("grid must be 4-D (n_chan, n_pol, m_u, m_v)");
    if (sum_info.ndim != 2)
        throw std::runtime_error("sum_weight must be 2-D (n_chan, n_pol)");
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");
    if (wt_info.ndim != 4)
        throw std::runtime_error(
            "data_weights must be 4-D (n_time, n_baseline, n_vis_chan, n_pol)");

    require_c_contiguous(grid_info, "grid");
    require_c_contiguous(sum_info,  "sum_weight");
    require_c_contiguous(uvw_info,  "uvw");
    require_c_contiguous(wt_info,   "data_weights");

    if (grid_info.readonly)
        throw std::runtime_error("grid must be a writable array (modified in place)");

    const py::dtype dt_f32 = py::dtype::of<float>();
    const py::dtype dt_f64 = py::dtype::of<double>();
    const py::dtype grid_dtype = grid.dtype();
    const bool grid_is_f32 = grid_dtype.is(dt_f32);
    const bool grid_is_f64 = grid_dtype.is(dt_f64);
    if (!grid_is_f32 && !grid_is_f64)
        throw std::runtime_error(
            "grid must be float32 or float64 (no implicit conversion is "
            "performed; the array is modified in place)");

    const int n_chan_g = static_cast<int>(grid_info.shape[0]);
    const int n_pol_g  = static_cast<int>(grid_info.shape[1]);
    const int m_u      = static_cast<int>(grid_info.shape[2]);
    const int m_v      = static_cast<int>(grid_info.shape[3]);

    const int n_time     = static_cast<int>(uvw_info.shape[0]);
    const int n_baseline = static_cast<int>(uvw_info.shape[1]);
    const int n_vis_chan = static_cast<int>(chan_map.shape(0));
    const int n_pol      = static_cast<int>(wt_info.shape[3]);

    (void)n_uv;  // grid dimensions are taken from the grid array itself

    auto dlm = delta_lm.unchecked<1>();

    // Capture raw pointers before releasing the GIL; the underlying numpy
    // buffers remain valid for the duration of the call because Python owns
    // the arrays.
    double*        sum_ptr      = sum_weight.mutable_data();
    const double*  uvw_ptr      = static_cast<const double*>(uvw_info.ptr);
    const double*  freq_ptr     = frequency_coord.data();
    const int64_t* chan_map_ptr = chan_map.data();
    const double*  wt_ptr       = static_cast<const double*>(wt_info.ptr);
    const double   dl           = dlm(0);
    const double   dm           = dlm(1);

    if (grid_is_f64) {
        double* grid_ptr = static_cast<double*>(grid_info.ptr);
        py::gil_scoped_release release;
        imaging_weighting::grid_imaging_weights<double>(
            grid_ptr, sum_ptr, uvw_ptr, freq_ptr, chan_map_ptr, wt_ptr,
            n_chan_g, n_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol, dl, dm);
    } else {
        float* grid_ptr = static_cast<float*>(grid_info.ptr);
        py::gil_scoped_release release;
        imaging_weighting::grid_imaging_weights<float>(
            grid_ptr, sum_ptr, uvw_ptr, freq_ptr, chan_map_ptr, wt_ptr,
            n_chan_g, n_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol, dl, dm);
    }
}

// ---------------------------------------------------------------------------
// degrid_imaging_weights
// ---------------------------------------------------------------------------
void degrid_imaging_weights_bind(
    py::array_t<double,  py::array::c_style>                         imaging_weight, // (n_time, n_baseline, n_vis_chan, n_pol) float64, writable
    py::array                                                        grid_imaging_weight, // (n_chan_g, n_pol_g, m_u, m_v) float32 or float64
    py::array_t<double,  py::array::c_style>                         briggs_factors, // (2, n_chan_g, n_pol_g)          float64
    py::array_t<double,  py::array::c_style>                         uvw,            // (n_time, n_baseline, 3)         float64
    py::array_t<double,  py::array::c_style>                         frequency_coord, // (n_vis_chan,)                  float64
    py::array_t<int64_t, py::array::c_style>                         chan_map,       // (n_vis_chan,)                   int64
    py::array_t<int64_t, py::array::c_style>                         pol_map,        // (n_pol_out,)                    int64
    py::array_t<double,  py::array::c_style>                         data_weight,    // (n_time, n_baseline, n_vis_chan, n_pol) float64
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>  n_uv,           // (2,) — tiny, forcecast OK
    py::array_t<double,  py::array::c_style | py::array::forcecast>  delta_lm,       // (2,) — tiny, forcecast OK
    int processing_function_threads
) {
    (void)processing_function_threads;  // serial kernel: bit-reproducible sums

    auto out_info    = imaging_weight.request();
    auto grid_info   = grid_imaging_weight.request();
    auto briggs_info = briggs_factors.request();
    auto uvw_info    = uvw.request();
    auto wt_info     = data_weight.request();

    if (out_info.ndim != 4)
        throw std::runtime_error(
            "imaging_weight must be 4-D (n_time, n_baseline, n_vis_chan, n_pol)");
    if (grid_info.ndim != 4)
        throw std::runtime_error(
            "grid_imaging_weight must be 4-D (n_chan, n_pol, m_u, m_v)");
    if (briggs_info.ndim != 3 || briggs_info.shape[0] != 2)
        throw std::runtime_error("briggs_factors must be 3-D (2, n_chan, n_pol)");
    if (uvw_info.ndim != 3 || uvw_info.shape[2] != 3)
        throw std::runtime_error("uvw must be 3-D (n_time, n_baseline, 3)");
    if (wt_info.ndim != 4)
        throw std::runtime_error(
            "data_weight must be 4-D (n_time, n_baseline, n_vis_chan, n_pol)");

    require_c_contiguous(out_info,    "imaging_weight");
    require_c_contiguous(grid_info,   "grid_imaging_weight");
    require_c_contiguous(briggs_info, "briggs_factors");
    require_c_contiguous(uvw_info,    "uvw");
    require_c_contiguous(wt_info,     "data_weight");

    if (out_info.readonly)
        throw std::runtime_error(
            "imaging_weight must be a writable array (modified in place)");

    const py::dtype dt_f32 = py::dtype::of<float>();
    const py::dtype dt_f64 = py::dtype::of<double>();
    const py::dtype grid_dtype = grid_imaging_weight.dtype();
    const bool grid_is_f32 = grid_dtype.is(dt_f32);
    const bool grid_is_f64 = grid_dtype.is(dt_f64);
    if (!grid_is_f32 && !grid_is_f64)
        throw std::runtime_error(
            "grid_imaging_weight must be float32 or float64 (no implicit "
            "conversion is performed)");

    const int n_chan_g = static_cast<int>(grid_info.shape[0]);
    const int n_pol_g  = static_cast<int>(grid_info.shape[1]);
    const int m_u      = static_cast<int>(grid_info.shape[2]);
    const int m_v      = static_cast<int>(grid_info.shape[3]);

    const int n_time     = static_cast<int>(uvw_info.shape[0]);
    const int n_baseline = static_cast<int>(uvw_info.shape[1]);
    const int n_vis_chan = static_cast<int>(chan_map.shape(0));
    const int n_pol      = static_cast<int>(wt_info.shape[3]);
    const int n_pol_out  = static_cast<int>(pol_map.shape(0));

    (void)n_uv;  // grid dimensions are taken from the grid array itself

    auto dlm = delta_lm.unchecked<1>();

    double*        out_ptr      = imaging_weight.mutable_data();
    const double*  briggs_ptr   = static_cast<const double*>(briggs_info.ptr);
    const double*  uvw_ptr      = static_cast<const double*>(uvw_info.ptr);
    const double*  freq_ptr     = frequency_coord.data();
    const int64_t* chan_map_ptr = chan_map.data();
    const int64_t* pol_map_ptr  = pol_map.data();
    const double*  wt_ptr       = static_cast<const double*>(wt_info.ptr);
    const double   dl           = dlm(0);
    const double   dm           = dlm(1);

    if (grid_is_f64) {
        const double* grid_ptr = static_cast<const double*>(grid_info.ptr);
        py::gil_scoped_release release;
        imaging_weighting::degrid_imaging_weights<double>(
            out_ptr, grid_ptr, briggs_ptr, uvw_ptr, freq_ptr,
            chan_map_ptr, pol_map_ptr, wt_ptr,
            n_chan_g, n_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol, n_pol_out, dl, dm);
    } else {
        const float* grid_ptr = static_cast<const float*>(grid_info.ptr);
        py::gil_scoped_release release;
        imaging_weighting::degrid_imaging_weights<float>(
            out_ptr, grid_ptr, briggs_ptr, uvw_ptr, freq_ptr,
            chan_map_ptr, pol_map_ptr, wt_ptr,
            n_chan_g, n_pol_g, m_u, m_v,
            n_time, n_baseline, n_vis_chan, n_pol, n_pol_out, dl, dm);
    }
}

PYBIND11_MODULE(_grid_imaging_weights_ext, m) {
    m.doc() = "C++ imaging-weight UV gridding / degridding (Briggs weighting)";

    m.def("grid_imaging_weights",
          &grid_imaging_weights_bind,
          "Grid per-visibility data weights onto a UV plane at the nearest "
          "pixel, with a Hermitian-conjugate update (in-place). The grid may "
          "be float32 or float64. The kernel is serial so weight sums are "
          "bit-reproducible; processing_function_threads is accepted for API "
          "consistency and ignored.",
          py::arg("grid"),
          py::arg("sum_weight"),
          py::arg("uvw"),
          py::arg("frequency_coord"),
          py::arg("chan_map"),
          py::arg("data_weights"),
          py::arg("n_uv"),
          py::arg("delta_lm"),
          py::arg("processing_function_threads") = 1);

    m.def("degrid_imaging_weights",
          &degrid_imaging_weights_bind,
          "Sample a UV imaging-weight grid at each visibility's (u, v) and "
          "apply the Briggs/robust denominator, writing per-visibility "
          "imaging weights into imaging_weight in place. The grid may be "
          "float32 or float64. The kernel is serial so results are "
          "bit-reproducible; processing_function_threads is accepted for API "
          "consistency and ignored.",
          py::arg("imaging_weight"),
          py::arg("grid_imaging_weight"),
          py::arg("briggs_factors"),
          py::arg("uvw"),
          py::arg("frequency_coord"),
          py::arg("chan_map"),
          py::arg("pol_map"),
          py::arg("data_weight"),
          py::arg("n_uv"),
          py::arg("delta_lm"),
          py::arg("processing_function_threads") = 1);
}
