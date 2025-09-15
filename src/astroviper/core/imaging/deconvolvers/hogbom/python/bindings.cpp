#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include "../include/hclean.hpp"

namespace py = pybind11;

/**
 * Templated wrapper for maximg function - supports both float and double
 */
template<typename T>
py::tuple maximg_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> image_array,
    py::array_t<T, py::array::c_style | py::array::forcecast> mask_array = py::array_t<T>()
) {
    py::buffer_info image_info = image_array.request();
    
    // Validate image dimensions - expect [pol, ny, nx]
    if (image_info.ndim != 3) {
        throw std::runtime_error("Image must be 3D array");
    }
    
    // Extract dimensions
    int npol = image_info.shape[0];
    int ny = image_info.shape[1]; 
    int nx = image_info.shape[2];
    
    // Handle optional mask
    int domask = 0;
    T* mask_ptr = nullptr;
    if (mask_array.size() > 0) {
        py::buffer_info mask_info = mask_array.request();
        if (mask_info.ndim != 2) {
            throw std::runtime_error("Mask must be 2D array");
        }
        if (mask_info.shape[0] != ny || mask_info.shape[1] != nx) {
            throw std::runtime_error("Mask dimensions must match image spatial dimensions");
        }
        domask = 1;
        mask_ptr = static_cast<T*>(mask_info.ptr);
    }
    
    // Call templated maximg directly with native type
    T fmin, fmax;
    hclean::maximg<T>(
        static_cast<const T*>(image_info.ptr),
        domask, static_cast<const T*>(mask_ptr),
        nx, ny, npol,
        fmin, fmax
    );
    
    // Return results in original type T
    return py::make_tuple(fmin, fmax);
}

/**
 * Templated Hogbom CLEAN wrapper - supports both float and double
 */
template<typename T>
py::dict hclean_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> dirty_image,
    py::array_t<T, py::array::c_style | py::array::forcecast> psf_array,
    py::array_t<T, py::array::c_style | py::array::forcecast> mask_array = py::array_t<T>(),
    py::tuple clean_box = py::make_tuple(-1, -1, -1, -1),
    int max_iter = 100,
    int start_iter = 0,
    T gain = static_cast<T>(0.1),
    T threshold = static_cast<T>(0.0),
    T speedup = static_cast<T>(0.0),
    py::object progress_callback = py::none(),
    py::object stop_callback = py::none()
) {
    // Validate input arrays
    py::buffer_info dirty_info = dirty_image.request();
    py::buffer_info psf_info = psf_array.request();
    
    if (dirty_info.ndim != 3) {
        throw std::runtime_error("Dirty image must be 3D [pol, ny, nx]");
    }
    if (psf_info.ndim != 2) {
        throw std::runtime_error("PSF must be 2D [ny, nx]");
    }
    
    // Extract dimensions
    int npol = dirty_info.shape[0];
    int ny = dirty_info.shape[1];
    int nx = dirty_info.shape[2];
    
    // Validate PSF dimensions match image
    if (psf_info.shape[0] != ny || psf_info.shape[1] != nx) {
        throw std::runtime_error("PSF dimensions must match image spatial dimensions");
    }
    
    // Handle optional mask
    int domask = 0;
    T* mask_ptr = nullptr;
    if (mask_array.size() > 0) {
        py::buffer_info mask_info = mask_array.request();
        if (mask_info.ndim != 2 || mask_info.shape[0] != ny || mask_info.shape[1] != nx) {
            throw std::runtime_error("Mask must be 2D with same spatial dimensions as image");
        }
        domask = 1;
        mask_ptr = static_cast<T*>(mask_info.ptr);
    }
    
    // Parse clean box
    int xbeg = 0, xend = nx, ybeg = 0, yend = ny;
    if (clean_box.size() == 4) {
        xbeg = py::cast<int>(clean_box[0]);
        xend = py::cast<int>(clean_box[1]);  
        ybeg = py::cast<int>(clean_box[2]);
        yend = py::cast<int>(clean_box[3]);
        
        // Handle -1 values (use full dimension)
        if (xbeg == -1) xbeg = 0;
        if (xend == -1) xend = nx;
        if (ybeg == -1) ybeg = 0; 
        if (yend == -1) yend = ny;
        
        // Validate bounds
        xbeg = std::max(0, std::min(xbeg, nx-1));
        xend = std::max(xbeg+1, std::min(xend, nx));
        ybeg = std::max(0, std::min(ybeg, ny-1));
        yend = std::max(ybeg+1, std::min(yend, ny));
    }
    
    std::vector<py::ssize_t> shape_3d = {npol, ny, nx};
    
    // Create working arrays in native type T - no conversion needed
    py::array_t<T> dirty_copy = py::array_t<T>(shape_3d);
    py::array_t<T> model_image = py::array_t<T>(shape_3d);
    
    // Zero-initialize and copy using safe buffer access
    py::buffer_info model_info = model_image.request();
    py::buffer_info dirty_copy_info = dirty_copy.request();
    std::memset(model_info.ptr, 0, model_info.size * model_info.itemsize);
    std::memcpy(dirty_copy_info.ptr, dirty_info.ptr, dirty_info.size * dirty_info.itemsize);
    
    // Set up progress callback wrapper
    std::function<void(int, int, int, int, int, T)> msgput_func = 
        [progress_callback](int npol, int pol, int iter, int px, int py, T peak) {
            if (!progress_callback.is_none()) {
                try {
                    progress_callback(npol, pol, iter, px, py, peak);
                } catch (const std::runtime_error& e) {
                    py::print("Warning: Progress callback failed:", e.what());
                }
            }
        };
    
    // Set up stop callback wrapper  
    std::function<void(int&)> stopnow_func = 
        [stop_callback](int& should_stop) {
            if (!stop_callback.is_none()) {
                try {
                    py::object result = stop_callback();
                    should_stop = py::cast<bool>(result) ? 1 : 0;
                } catch (const std::runtime_error& e) {
                    py::print("Warning: Stop callback failed:", e.what());
                    should_stop = 0;
                }
            }
        };
    
    int final_iter = start_iter;
    hclean::clean<T>(
        static_cast<T*>(model_info.ptr),           // model image output
        static_cast<T*>(dirty_copy_info.ptr),      // dirty/residual (copy)
        static_cast<const T*>(psf_info.ptr),       // PSF
        domask, static_cast<const T*>(mask_ptr),   // mask
        nx, ny, npol,                              // dimensions
        xbeg, xend, ybeg, yend,                   // clean box
        max_iter, start_iter, final_iter,         // iteration control
        gain, threshold, speedup,                 // parameters
        msgput_func, stopnow_func                 // callbacks
    );
    
    // Calculate total flux cleaned
    py::array_t<T> model_flat = model_image.reshape({npol * ny * nx});
    T total_flux = static_cast<T>(0);
    T* model_data = static_cast<T*>(model_flat.request().ptr);
    for (int i = 0; i < npol * ny * nx; ++i) {
        total_flux += std::abs(model_data[i]);
    }
    
    // Find final peak in residual
    T final_min, final_max;
    hclean::maximg<T>(
        static_cast<const T*>(dirty_copy_info.ptr),
        domask, static_cast<const T*>(mask_ptr),
        nx, ny, npol,
        final_min, final_max
    );
    T final_peak = std::max(std::abs(final_min), std::abs(final_max));
    
    // Return comprehensive results in original type T
    py::dict results;
    results["model_image"] = model_image;
    results["residual_image"] = dirty_copy;
    results["iterations_performed"] = final_iter;
    results["final_peak"] = final_peak;
    results["total_flux_cleaned"] = total_flux;
    results["converged"] = (final_peak <= threshold);
    
    return results;
}

/**
 * Simplified templated wrapper for basic CLEAN without callbacks
 */
template<typename T>
py::dict hclean_simple_impl(
    py::array_t<T> dirty_image,
    py::array_t<T> psf_array,
    T gain = static_cast<T>(0.1),
    T threshold = static_cast<T>(0.0),
    int max_iter = 100,
    py::tuple clean_box = py::make_tuple(-1, -1, -1, -1)
) {
    return hclean_impl<T>(
        dirty_image, psf_array, py::array_t<T>(), // no mask
        clean_box, max_iter, 0,  // start from iteration 0
        gain, threshold, static_cast<T>(0.0),   // no speedup
        py::none(), py::none()   // no callbacks
    );
}

PYBIND11_MODULE(_hogbom_ext, m) {
    m.doc() = "Templated Hogbom CLEAN algorithm - supports both float32 and float64";
    
    // Float32 versions
    m.def("maximg", &maximg_impl<float>,
          "Find minimum and maximum values in 3D image (float32)",
          py::arg("image"), py::arg("mask") = py::array_t<float>());
    
    m.def("clean", &hclean_impl<float>,
          "Hogbom CLEAN algorithm (float32)",
          py::arg("dirty_image"),
          py::arg("psf"),
          py::arg("mask") = py::array_t<float>(),
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1),
          py::arg("max_iter") = 100,
          py::arg("start_iter") = 0,
          py::arg("gain") = 0.1f,
          py::arg("threshold") = 0.0f,
          py::arg("speedup") = 0.0f,
          py::arg("progress_callback") = py::none(),
          py::arg("stop_callback") = py::none());
    
    m.def("clean_nc", &hclean_simple_impl<float>,
          "Simple Hogbom CLEAN interface (float32)",
          py::arg("dirty_image"),
          py::arg("psf"),
          py::arg("gain") = 0.1f,
          py::arg("threshold") = 0.0f,
          py::arg("max_iter") = 100,
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1));
    
    // Float64 versions  
    m.def("maximg", &maximg_impl<double>,
          "Find minimum and maximum values in 3D image (float64)",
          py::arg("image"), py::arg("mask") = py::array_t<double>());
    
    m.def("clean", &hclean_impl<double>,
          "Hogbom CLEAN algorithm (float64)",
          py::arg("dirty_image"),
          py::arg("psf"),
          py::arg("mask") = py::array_t<double>(),
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1),
          py::arg("max_iter") = 100,
          py::arg("start_iter") = 0,
          py::arg("gain") = 0.1,
          py::arg("threshold") = 0.0,
          py::arg("speedup") = 0.0,
          py::arg("progress_callback") = py::none(),
          py::arg("stop_callback") = py::none());
    
    m.def("clean_nc", &hclean_simple_impl<double>,
          "Simple Hogbom CLEAN interface (float64)",
          py::arg("dirty_image"),
          py::arg("psf"),
          py::arg("gain") = 0.1,
          py::arg("threshold") = 0.0,
          py::arg("max_iter") = 100,
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1));
    
    // Utility functions to check array dtypes
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
