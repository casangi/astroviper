#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "../include/hclean.hpp"

namespace py = pybind11;

/**
 * Validate that a py::array matches the expected dtype T, is 2D with the
 * given spatial shape, and is C-contiguous. If require_writable is true,
 * also require the array to be writeable. Returns the buffer_info so the
 * caller can extract the raw pointer without any copy.
 *
 * All checks raise std::runtime_error on failure; no implicit conversion
 * or copy is ever performed.
 */
template<typename T>
static py::buffer_info validate_inplace_array(py::array& arr,
                                              const char* name,
                                              int ny, int nx,
                                              bool require_writable) {
    if (!arr.dtype().is(py::dtype::of<T>())) {
        throw std::runtime_error(
            std::string(name) + " has wrong dtype; expected " +
            py::cast<std::string>(py::dtype::of<T>().attr("name")) +
            " but got " + py::cast<std::string>(arr.dtype().attr("name")));
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error(
            std::string(name) + " must be C-contiguous (no copy will be made)");
    }
    if (require_writable && !arr.writeable()) {
        throw std::runtime_error(
            std::string(name) + " must be writeable (array is modified in place)");
    }

    py::buffer_info info = arr.request(require_writable);
    if (info.ndim != 2) {
        throw std::runtime_error(std::string(name) + " must be a 2D array");
    }
    if (info.shape[0] != ny || info.shape[1] != nx) {
        throw std::runtime_error(
            std::string(name) + " has wrong shape; expected (" +
            std::to_string(ny) + ", " + std::to_string(nx) + ")");
    }
    return info;
}

/**
 * Validate that a py::array matches dtype T, is 5D with the given
 * (nt, nf, np_dim, ny, nx) shape, and is C-contiguous. If
 * require_writable is true, also require the array to be writeable.
 */
template<typename T>
static py::buffer_info validate_inplace_cube(py::array& arr,
                                             const char* name,
                                             int nt, int nf, int np_dim,
                                             int ny, int nx,
                                             bool require_writable) {
    if (!arr.dtype().is(py::dtype::of<T>())) {
        throw std::runtime_error(
            std::string(name) + " has wrong dtype; expected " +
            py::cast<std::string>(py::dtype::of<T>().attr("name")) +
            " but got " + py::cast<std::string>(arr.dtype().attr("name")));
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error(
            std::string(name) + " must be C-contiguous (no copy will be made)");
    }
    if (require_writable && !arr.writeable()) {
        throw std::runtime_error(
            std::string(name) + " must be writeable (array is modified in place)");
    }

    py::buffer_info info = arr.request(require_writable);
    if (info.ndim != 5) {
        throw std::runtime_error(
            std::string(name) +
            " must be a 5D array (time, frequency, polarization, y, x)");
    }
    if (info.shape[0] != nt || info.shape[1] != nf ||
        info.shape[2] != np_dim ||
        info.shape[3] != ny || info.shape[4] != nx) {
        throw std::runtime_error(
            std::string(name) + " has wrong shape; expected (" +
            std::to_string(nt) + ", " + std::to_string(nf) + ", " +
            std::to_string(np_dim) + ", " +
            std::to_string(ny) + ", " + std::to_string(nx) + ")");
    }
    return info;
}

/**
 * Validate a per-plane control array: dtype U, 3D with shape
 * (nt, nf, np_dim), and C-contiguous. Returns the buffer_info so the caller
 * can extract the raw pointer (length nt*nf*np_dim in (t, f, p) C-order)
 * without any copy. Used for the per-plane niter and threshold arrays so
 * that iteration control is independent for each (time, frequency,
 * polarization) plane.
 */
template<typename U>
static py::buffer_info validate_per_plane_array(py::array& arr,
                                                const char* name,
                                                int nt, int nf, int np_dim) {
    if (!arr.dtype().is(py::dtype::of<U>())) {
        throw std::runtime_error(
            std::string(name) + " has wrong dtype; expected " +
            py::cast<std::string>(py::dtype::of<U>().attr("name")) +
            " but got " + py::cast<std::string>(arr.dtype().attr("name")));
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error(
            std::string(name) + " must be C-contiguous");
    }
    py::buffer_info info = arr.request(false);
    if (info.ndim != 3) {
        throw std::runtime_error(
            std::string(name) +
            " must be a 3D array (time, frequency, polarization)");
    }
    if (info.shape[0] != nt || info.shape[1] != nf || info.shape[2] != np_dim) {
        throw std::runtime_error(
            std::string(name) + " has wrong shape; expected (" +
            std::to_string(nt) + ", " + std::to_string(nf) + ", " +
            std::to_string(np_dim) + ")");
    }
    return info;
}

/**
 * Templated maximg wrapper. The image is templated on T (float/double); the
 * mask is always bool. Both are validated as zero-copy: they must be
 * C-contiguous arrays of the exact dtype with matching spatial shapes — no
 * implicit conversion or copy is ever made (see maximg_dispatch).
 */
template<typename T>
py::tuple maximg_impl(py::array image_array, py::array mask_array) {
    if (image_array.ndim() != 2) {
        throw std::runtime_error("Image must be 2D array");
    }

    const int ny = static_cast<int>(image_array.shape(0));
    const int nx = static_cast<int>(image_array.shape(1));

    py::buffer_info image_info = validate_inplace_array<T>(
        image_array, "image", ny, nx, /*require_writable=*/false);

    int domask = 0;
    const bool* mask_ptr = nullptr;
    if (mask_array.size() > 0) {
        py::buffer_info mask_info = validate_inplace_array<bool>(
            mask_array, "mask", ny, nx, /*require_writable=*/false);
        domask = 1;
        mask_ptr = static_cast<const bool*>(mask_info.ptr);
    }

    T fmin, fmax;
    {
        // Full-image scan touches no Python objects; let other worker
        // threads (Dask threads_per_worker > 1) run during it.
        py::gil_scoped_release release;
        hclean::maximg<T>(
            static_cast<const T*>(image_info.ptr),
            domask, mask_ptr,
            nx, ny,
            fmin, fmax
        );
    }

    return py::make_tuple(fmin, fmax);
}

/**
 * Runtime dtype dispatcher for maximg_impl. Selects the float32 or float64
 * code path from the image dtype; anything else raises (no forcecast, so a
 * wrong dtype is never silently copy-converted).
 */
static py::tuple maximg_dispatch(py::array image, py::array mask) {
    auto dt = image.dtype();
    if (dt.is(py::dtype::of<float>())) {
        return maximg_impl<float>(image, mask);
    } else if (dt.is(py::dtype::of<double>())) {
        return maximg_impl<double>(image, mask);
    } else {
        throw std::runtime_error("image must be float32 or float64");
    }
}

/**
 * Templated Hogbom CLEAN wrapper. The dirty_image and model arrays are
 * mutated in place via pointers into the Python-owned buffers: no copies
 * of the (ny, nx) arrays are made on the C++ side. The caller must supply
 * C-contiguous writeable arrays of matching dtype; otherwise the call
 * raises std::runtime_error.
 */
template<typename T>
py::dict hclean_impl(
    py::array dirty_image,
    py::array psf_array,
    py::array model_image,
    py::array mask_array,
    py::tuple clean_box,
    int max_iter,
    int start_iter,
    T gain,
    T threshold,
    T speedup,
    py::object progress_callback,
    py::object stop_callback
) {
    // First, read dirty_image shape so we can cross-validate the others.
    if (!dirty_image.dtype().is(py::dtype::of<T>())) {
        throw std::runtime_error("dirty_image has wrong dtype");
    }
    if (dirty_image.ndim() != 2) {
        throw std::runtime_error("dirty_image must be 2D [ny, nx]");
    }
    int ny = static_cast<int>(dirty_image.shape(0));
    int nx = static_cast<int>(dirty_image.shape(1));

    // Strict no-copy validation of all mutable/readable arrays.
    py::buffer_info dirty_info = validate_inplace_array<T>(
        dirty_image, "dirty_image", ny, nx, /*require_writable=*/true);
    py::buffer_info psf_info = validate_inplace_array<T>(
        psf_array, "psf", ny, nx, /*require_writable=*/false);
    py::buffer_info model_info = validate_inplace_array<T>(
        model_image, "model", ny, nx, /*require_writable=*/true);

    int domask = 0;
    const bool* mask_ptr = nullptr;
    if (mask_array.size() > 0) {
        py::buffer_info mask_info = validate_inplace_array<bool>(
            mask_array, "mask", ny, nx, /*require_writable=*/false);
        domask = 1;
        // buffer_info is just a descriptor; the underlying memory is
        // owned by mask_array (a function argument kept alive for the
        // duration of this call), so the raw pointer stays valid.
        mask_ptr = static_cast<const bool*>(mask_info.ptr);
    }

    // Parse clean box with -1 meaning "full extent".
    int xbeg = 0, xend = nx, ybeg = 0, yend = ny;
    if (clean_box.size() == 4) {
        xbeg = py::cast<int>(clean_box[0]);
        xend = py::cast<int>(clean_box[1]);
        ybeg = py::cast<int>(clean_box[2]);
        yend = py::cast<int>(clean_box[3]);

        if (xbeg == -1) xbeg = 0;
        if (xend == -1) xend = nx;
        if (ybeg == -1) ybeg = 0;
        if (yend == -1) yend = ny;

        xbeg = std::max(0, std::min(xbeg, nx - 1));
        xend = std::max(xbeg + 1, std::min(xend, nx));
        ybeg = std::max(0, std::min(ybeg, ny - 1));
        yend = std::max(ybeg + 1, std::min(yend, ny));
    }

    // The CLEAN loop below runs with the GIL released; the callbacks are
    // invoked from inside that loop, so they re-acquire the GIL before
    // touching any Python object. is_none() is a raw pointer comparison and
    // is safe without the GIL, keeping the no-callback case GIL-free.
    std::function<void(int, int, int, T)> msgput_func =
        [progress_callback](int iter, int px, int py_coord, T peak) {
            if (!progress_callback.is_none()) {
                py::gil_scoped_acquire acquire;
                try {
                    progress_callback(iter, px, py_coord, peak);
                } catch (const std::runtime_error& e) {
                    py::print("Warning: Progress callback failed:", e.what());
                }
            }
        };

    std::function<void(int&)> stopnow_func =
        [stop_callback](int& should_stop) {
            if (!stop_callback.is_none()) {
                py::gil_scoped_acquire acquire;
                try {
                    py::object result = stop_callback();
                    should_stop = py::cast<bool>(result) ? 1 : 0;
                } catch (const std::runtime_error& e) {
                    py::print("Warning: Stop callback failed:", e.what());
                    should_stop = 0;
                }
            }
        };

    // Run CLEAN directly on the Python-owned buffers, GIL released (the
    // buffers are owned by the caller's arrays, which outlive this call).
    int final_iter = start_iter;
    T total_flux = static_cast<T>(0);
    T final_min, final_max;
    {
        py::gil_scoped_release release;
        hclean::clean<T>(
            static_cast<T*>(model_info.ptr),
            static_cast<T*>(dirty_info.ptr),
            static_cast<const T*>(psf_info.ptr),
            domask, mask_ptr,
            nx, ny,
            xbeg, xend, ybeg, yend,
            max_iter, start_iter, final_iter,
            gain, threshold, speedup,
            msgput_func, stopnow_func
        );

        // Post-CLEAN diagnostics; arrays remain in place.
        const T* model_data = static_cast<const T*>(model_info.ptr);
        for (int i = 0; i < ny * nx; ++i) {
            total_flux += std::abs(model_data[i]);
        }

        hclean::maximg<T>(
            static_cast<const T*>(dirty_info.ptr),
            domask, mask_ptr,
            nx, ny,
            final_min, final_max
        );
    }
    T final_peak = std::max(std::abs(final_min), std::abs(final_max));

    py::dict results;
    results["iterations_performed"] = final_iter;
    results["final_peak"] = final_peak;
    results["total_flux_cleaned"] = total_flux;
    results["converged"] = (final_peak <= threshold);
    return results;
}

/**
 * Templated Hogbom CLEAN cube wrapper. The residual_cube and model_cube
 * arrays are modified in place via pointers into Python-owned buffers;
 * no copies are made on the C++ side. The outer (t, f, p) loop runs in
 * processing_function_threads worker threads via std::thread.
 *
 * The PSF cube may have np_psf == np_img (one PSF per image pol) or
 * np_psf == 1 (broadcast across all image pols). Per-plane progress and
 * stop callbacks are not supported in cube mode; bulk logging should be
 * done Python-side from the returned per-plane iteration counts.
 */
template<typename T>
py::dict hclean_cube_impl(
    py::array residual_cube,
    py::array psf_cube,
    py::array model_cube,
    py::array mask_cube,
    py::tuple clean_box,
    py::array max_iter,
    T gain,
    py::array threshold,
    T speedup,
    int processing_function_threads,
    bool many_threads
) {
    if (!residual_cube.dtype().is(py::dtype::of<T>())) {
        throw std::runtime_error("residual_cube has wrong dtype");
    }
    if (residual_cube.ndim() != 5) {
        throw std::runtime_error(
            "residual_cube must be a 5D array [time, frequency, polarization, y, x]");
    }
    const int nt = static_cast<int>(residual_cube.shape(0));
    const int nf = static_cast<int>(residual_cube.shape(1));
    const int np_img = static_cast<int>(residual_cube.shape(2));
    const int ny = static_cast<int>(residual_cube.shape(3));
    const int nx = static_cast<int>(residual_cube.shape(4));

    py::buffer_info residual_info = validate_inplace_cube<T>(
        residual_cube, "residual_cube", nt, nf, np_img, ny, nx,
        /*require_writable=*/true);
    py::buffer_info model_info = validate_inplace_cube<T>(
        model_cube, "model_cube", nt, nf, np_img, ny, nx,
        /*require_writable=*/true);

    // PSF may broadcast on polarization axis.
    if (!psf_cube.dtype().is(py::dtype::of<T>())) {
        throw std::runtime_error("psf_cube has wrong dtype");
    }
    if (psf_cube.ndim() != 5) {
        throw std::runtime_error(
            "psf_cube must be a 5D array [time, frequency, polarization, y, x]");
    }
    if (psf_cube.shape(0) != nt || psf_cube.shape(1) != nf ||
        psf_cube.shape(3) != ny || psf_cube.shape(4) != nx) {
        throw std::runtime_error(
            "psf_cube must match residual_cube on (time, frequency, y, x)");
    }
    const int np_psf = static_cast<int>(psf_cube.shape(2));
    if (np_psf != np_img && np_psf != 1) {
        throw std::runtime_error(
            "psf_cube polarization axis must equal residual_cube polarization "
            "axis or be 1 (Stokes I broadcast)");
    }
    py::buffer_info psf_info = validate_inplace_cube<T>(
        psf_cube, "psf_cube", nt, nf, np_psf, ny, nx,
        /*require_writable=*/false);

    int domask = 0;
    const bool* mask_ptr = nullptr;
    if (mask_cube.size() > 0) {
        py::buffer_info mask_info = validate_inplace_cube<bool>(
            mask_cube, "mask_cube", nt, nf, np_img, ny, nx,
            /*require_writable=*/false);
        domask = 1;
        mask_ptr = static_cast<const bool*>(mask_info.ptr);
    }

    int xbeg = 0, xend = nx, ybeg = 0, yend = ny;
    if (clean_box.size() == 4) {
        xbeg = py::cast<int>(clean_box[0]);
        xend = py::cast<int>(clean_box[1]);
        ybeg = py::cast<int>(clean_box[2]);
        yend = py::cast<int>(clean_box[3]);

        if (xbeg == -1) xbeg = 0;
        if (xend == -1) xend = nx;
        if (ybeg == -1) ybeg = 0;
        if (yend == -1) yend = ny;

        xbeg = std::max(0, std::min(xbeg, nx - 1));
        xend = std::max(xbeg + 1, std::min(xend, nx));
        ybeg = std::max(0, std::min(ybeg, ny - 1));
        yend = std::max(ybeg + 1, std::min(yend, ny));
    }

    // Per-plane iteration control: niter (int32) and threshold (T) are
    // (nt, nf, np_img) arrays so every (time, frequency, polarization) plane
    // is cleaned with its own iteration limit and threshold.
    py::buffer_info niter_info = validate_per_plane_array<int>(
        max_iter, "max_iter", nt, nf, np_img);
    py::buffer_info thres_info = validate_per_plane_array<T>(
        threshold, "threshold", nt, nf, np_img);
    const int* niter_ptr = static_cast<const int*>(niter_info.ptr);
    const T* thres_ptr = static_cast<const T*>(thres_info.ptr);

    const int nplanes = nt * nf * np_img;
    std::vector<int> iter_out(nplanes, 0);

    // Drop the GIL: workers do not touch Python objects.
    {
        py::gil_scoped_release release;
        if (many_threads) {
            hclean::clean_cube_many_threads<T>(
                static_cast<T*>(residual_info.ptr),
                static_cast<T*>(model_info.ptr),
                static_cast<const T*>(psf_info.ptr),
                domask, mask_ptr,
                nt, nf, np_img, np_psf,
                ny, nx,
                xbeg, xend, ybeg, yend,
                niter_ptr, gain, thres_ptr, speedup,
                processing_function_threads, iter_out.data());
        } else {
            hclean::clean_cube<T>(
                static_cast<T*>(residual_info.ptr),
                static_cast<T*>(model_info.ptr),
                static_cast<const T*>(psf_info.ptr),
                domask, mask_ptr,
                nt, nf, np_img, np_psf,
                ny, nx,
                xbeg, xend, ybeg, yend,
                niter_ptr, gain, thres_ptr, speedup,
                processing_function_threads, iter_out.data());
        }
    }

    // Assemble per-plane summary arrays. Plane ordering is C-contiguous
    // over (time, frequency, polarization). The arrays are allocated (and
    // their raw pointers taken) while holding the GIL; the summary scan
    // itself traverses the full residual + model cubes, so it runs with the
    // GIL released like the CLEAN loop above.
    py::array_t<int> iters_arr({nt, nf, np_img});
    std::memcpy(iters_arr.mutable_data(), iter_out.data(),
                sizeof(int) * nplanes);

    py::array_t<T> final_peaks_arr({nt, nf, np_img});
    py::array_t<T> total_flux_arr({nt, nf, np_img});
    py::array_t<bool> converged_arr({nt, nf, np_img});
    T* fp = final_peaks_arr.mutable_data();
    T* tf = total_flux_arr.mutable_data();
    bool* cv = converged_arr.mutable_data();

    const T* residual_data = static_cast<const T*>(residual_info.ptr);
    const T* model_data = static_cast<const T*>(model_info.ptr);
    const std::size_t plane_size =
        static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);

    {
        py::gil_scoped_release release;
        for (int tt = 0; tt < nt; ++tt) {
            for (int nn = 0; nn < nf; ++nn) {
                for (int pp = 0; pp < np_img; ++pp) {
                    const std::size_t off =
                        ((static_cast<std::size_t>(tt) * nf + nn) * np_img + pp) * plane_size;

                    T total = static_cast<T>(0);
                    for (std::size_t i = 0; i < plane_size; ++i) {
                        total += std::abs(model_data[off + i]);
                    }
                    T fmin_p, fmax_p;
                    const bool* mptr = (domask && mask_ptr != nullptr) ? mask_ptr + off : nullptr;
                    hclean::maximg<T>(residual_data + off, domask, mptr,
                                      nx, ny, fmin_p, fmax_p);
                    T fp_val = std::max(std::abs(fmin_p), std::abs(fmax_p));

                    const int lin = (tt * nf + nn) * np_img + pp;
                    fp[lin] = fp_val;
                    tf[lin] = total;
                    cv[lin] = (fp_val <= thres_ptr[lin]);
                }
            }
        }
    }

    py::dict results;
    results["iterations_performed"] = iters_arr;
    results["final_peak"] = final_peaks_arr;
    results["total_flux_cleaned"] = total_flux_arr;
    results["converged"] = converged_arr;
    return results;
}

/**
 * Runtime dtype dispatcher for hclean_impl. Selects float32 or float64
 * code path based on the dtype of `dirty_image`; all other arrays must
 * share that dtype.
 */
static py::dict clean_dispatch(
    py::array dirty_image,
    py::array psf_array,
    py::array model_image,
    py::array mask_array,
    py::tuple clean_box,
    int max_iter,
    int start_iter,
    double gain,
    double threshold,
    double speedup,
    py::object progress_callback,
    py::object stop_callback
) {
    auto dt = dirty_image.dtype();
    if (dt.is(py::dtype::of<float>())) {
        return hclean_impl<float>(
            dirty_image, psf_array, model_image, mask_array,
            clean_box, max_iter, start_iter,
            static_cast<float>(gain),
            static_cast<float>(threshold),
            static_cast<float>(speedup),
            progress_callback, stop_callback);
    } else if (dt.is(py::dtype::of<double>())) {
        return hclean_impl<double>(
            dirty_image, psf_array, model_image, mask_array,
            clean_box, max_iter, start_iter,
            gain, threshold, speedup,
            progress_callback, stop_callback);
    } else {
        throw std::runtime_error(
            "dirty_image must be float32 or float64");
    }
}

/**
 * Runtime dtype dispatcher for hclean_cube_impl.
 */
static py::dict clean_cube_dispatch(
    py::array residual_cube,
    py::array psf_cube,
    py::array model_cube,
    py::array mask_cube,
    py::tuple clean_box,
    py::array max_iter,
    double gain,
    py::array threshold,
    double speedup,
    int processing_function_threads
) {
    auto dt = residual_cube.dtype();
    if (dt.is(py::dtype::of<float>())) {
        return hclean_cube_impl<float>(
            residual_cube, psf_cube, model_cube, mask_cube,
            clean_box, max_iter,
            static_cast<float>(gain),
            threshold,
            static_cast<float>(speedup),
            processing_function_threads, /*many_threads=*/false);
    } else if (dt.is(py::dtype::of<double>())) {
        return hclean_cube_impl<double>(
            residual_cube, psf_cube, model_cube, mask_cube,
            clean_box, max_iter,
            gain, threshold, speedup,
            processing_function_threads, /*many_threads=*/false);
    } else {
        throw std::runtime_error(
            "residual_cube must be float32 or float64");
    }
}

/**
 * Runtime dtype dispatcher for the many-threads (across + within plane) cube
 * cleaner. Same arguments as clean_cube_dispatch; routes to the
 * clean_cube_many_threads kernel.
 */
static py::dict clean_cube_many_threads_dispatch(
    py::array residual_cube,
    py::array psf_cube,
    py::array model_cube,
    py::array mask_cube,
    py::tuple clean_box,
    py::array max_iter,
    double gain,
    py::array threshold,
    double speedup,
    int processing_function_threads
) {
    auto dt = residual_cube.dtype();
    if (dt.is(py::dtype::of<float>())) {
        return hclean_cube_impl<float>(
            residual_cube, psf_cube, model_cube, mask_cube,
            clean_box, max_iter,
            static_cast<float>(gain),
            threshold,
            static_cast<float>(speedup),
            processing_function_threads, /*many_threads=*/true);
    } else if (dt.is(py::dtype::of<double>())) {
        return hclean_cube_impl<double>(
            residual_cube, psf_cube, model_cube, mask_cube,
            clean_box, max_iter,
            gain, threshold, speedup,
            processing_function_threads, /*many_threads=*/true);
    } else {
        throw std::runtime_error(
            "residual_cube must be float32 or float64");
    }
}

PYBIND11_MODULE(_hogbom_ext, m) {
    m.doc() = "Templated Hogbom CLEAN algorithm - in-place, zero-copy arrays";

    // maximg: single entry point; runtime-dispatched on dtype (float32 or
    // float64, C-contiguous, no copies). Mask, if provided, must be a bool
    // array of the same shape.
    m.def("maximg", &maximg_dispatch,
          "Find minimum and maximum values in a 2D image (float32 or "
          "float64; C-contiguous, zero-copy). Mask, if provided, must be a "
          "C-contiguous bool array of the same shape.",
          py::arg("image"), py::arg("mask") = py::array());

    // clean: single entry point; runtime-dispatched on dtype. All input
    // arrays must be C-contiguous, of matching dtype (float32 or float64),
    // and the mutable ones (dirty_image, model) must be writeable. No
    // copies of the (ny, nx) buffers are made on the C++ side: the
    // dirty_image is updated in place with the residual and the model
    // accumulates CLEAN components in place.
    m.def("clean", &clean_dispatch,
          "Hogbom CLEAN (in-place). dirty_image and model are modified in "
          "place; the residual replaces dirty_image and CLEAN components "
          "are added into model. Both must be C-contiguous, writeable, and "
          "share a dtype (float32 or float64) with psf and mask.",
          py::arg("dirty_image"),
          py::arg("psf"),
          py::arg("model"),
          py::arg("mask") = py::array(),
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1),
          py::arg("max_iter") = 100,
          py::arg("start_iter") = 0,
          py::arg("gain") = 0.1,
          py::arg("threshold") = 0.0,
          py::arg("speedup") = 0.0,
          py::arg("progress_callback") = py::none(),
          py::arg("stop_callback") = py::none());

    // clean_cube: operate on a full (time, frequency, polarization, y, x)
    // cube in place. All (t, f, p) planes are dispatched to processing_function_threads
    // std::thread workers. The PSF cube may have np_psf == np_img or
    // np_psf == 1 (broadcast across polarization).
    m.def("clean_cube", &clean_cube_dispatch,
          "Hogbom CLEAN over a 5D (time, frequency, polarization, y, x) "
          "cube, parallelized across planes with std::thread. The "
          "residual and model cubes are modified in place; the PSF cube "
          "may be single-polarization to broadcast across image pols. "
          "Iteration control is independent per plane: max_iter and "
          "threshold are (nt, nf, np) arrays (int32 and matching the image "
          "dtype respectively) giving each (time, frequency, polarization) "
          "plane its own iteration limit and threshold. Returns per-plane "
          "arrays of iterations_performed, final_peak, total_flux_cleaned, "
          "and converged with shape (nt, nf, np).",
          py::arg("residual_cube"),
          py::arg("psf_cube"),
          py::arg("model_cube"),
          py::arg("mask_cube") = py::array(),
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1),
          py::arg("max_iter") = py::array(),
          py::arg("gain") = 0.1,
          py::arg("threshold") = py::array(),
          py::arg("speedup") = 0.0,
          py::arg("processing_function_threads") = 1);

    // clean_cube_many_threads: same as clean_cube but the parallelism spans
    // both the (time, frequency, polarization) planes AND the rows within each
    // plane, so all processing_function_threads workers stay busy even for a single-channel
    // cube (a few planes). Identical algorithm, tie-breaking and outputs to
    // clean_cube. processing_function_threads is the worker count (set from
    // processing_function_threads).
    m.def("clean_cube_many_threads", &clean_cube_many_threads_dispatch,
          "Hogbom CLEAN over a 5D (time, frequency, polarization, y, x) cube, "
          "parallelized across planes AND within each plane (rows) with "
          "std::thread, so all processing_function_threads workers are used regardless of the "
          "plane count. Residual and model are modified in place; same "
          "per-plane max_iter / threshold arrays and outputs as clean_cube.",
          py::arg("residual_cube"),
          py::arg("psf_cube"),
          py::arg("model_cube"),
          py::arg("mask_cube") = py::array(),
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1),
          py::arg("max_iter") = py::array(),
          py::arg("gain") = 0.1,
          py::arg("threshold") = py::array(),
          py::arg("speedup") = 0.0,
          py::arg("processing_function_threads") = 1);

    m.def("get_dtype_name", [](py::array arr) {
        return arr.dtype().str();
    }, "Get string representation of array dtype");

    m.def("is_float32", [](py::array arr) {
        return arr.dtype().is(py::dtype::of<float>());
    }, "Check if array is float32");

    m.def("is_float64", [](py::array arr) {
        return arr.dtype().is(py::dtype::of<double>());
    }, "Check if array is float64");
}
