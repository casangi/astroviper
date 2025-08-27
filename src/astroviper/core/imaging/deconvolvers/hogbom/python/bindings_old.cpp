#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "../src/hclean.cpp"

namespace py = pybind11;

/**
 * Wrapper for maximg function - find min/max values in 3D image
 */
py::tuple maximg(
    py::array_t<float> image_array,
    py::array_t<float> mask_array = py::array_t<float>()
) {
    py::buffer_info image_info = image_array.request();
    
    // Validate image dimensions - expect nx, ny, npol
    if (image_info.ndim != 3) {
        throw std::runtime_error("Image must be 3D array");
    }
    
    // Extract dimensions - assume [pol, ny, nx]
    int npol = image_info.shape[0];
    int ny = image_info.shape[1]; 
    int nx = image_info.shape[2];
    
    // Handle optional mask
    int domask = 0;
    float* mask_ptr = nullptr;
    if (mask_array.size() > 0) {
        py::buffer_info mask_info = mask_array.request();
        if (mask_info.ndim != 2) {
            throw std::runtime_error("Mask must be 2D array");
        }
        if (mask_info.shape[0] != ny || mask_info.shape[1] != nx) {
            throw std::runtime_error("Mask dimensions must match image spatial dimensions");
        }
        domask = 1;
        mask_ptr = static_cast<float*>(mask_info.ptr);
    }
    
    // Call C++ function
    float fmin, fmax;
    hclean::maximg(
        static_cast<const float*>(image_info.ptr),
        domask, mask_ptr,
        nx, ny, npol,
        fmin, fmax);
    
    return py::make_tuple(fmin, fmax);
}

/**
 * Hogbom CLEAN wrapper
 */
py::dict hogbom_clean(
    py::array_t<float> dirty_image,
    py::array_t<float> psf_array,
    py::array_t<float> mask_array = py::array_t<float>(),
    py::tuple clean_box = py::make_tuple(-1, -1, -1, -1),
    int max_iter = 100,
    int start_iter = 0,
    float gain = 0.1f,
    float threshold = 0.0f,
    float speedup = 0.0f,
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
    float* mask_ptr = nullptr;
    if (mask_array.size() > 0) {
        py::buffer_info mask_info = mask_array.request();
        if (mask_info.ndim != 2 || mask_info.shape[0] != ny || mask_info.shape[1] != nx) {
            throw std::runtime_error("Mask must be 2D with same spatial dimensions as image");
        }
        domask = 1;
        mask_ptr = static_cast<float*>(mask_info.ptr);
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
    
    // Create working copies of arrays
    py::array_t<float> dirty_copy = py::array_t<float>(dirty_info.shape);
    py::array_t<float> model_image = py::array_t<float>(dirty_info.shape);
    
    // Zero-initialize model image using template unchecked for new array
    std::memset(model_image.template mutable_unchecked<3>().mutable_data(), 0, 
                dirty_info.size * sizeof(float));
    
    std::memcpy(dirty_copy.template mutable_unchecked<3>().mutable_data(),
                dirty_info.ptr, dirty_info.size * sizeof(float));
    
    py::buffer_info dirty_copy_info = dirty_copy.request();
    py::buffer_info model_info = model_image.request();
    
    // Set up progress callback wrapper
    std::function<void(int, int, int, int, int, float)> msgput_func = 
        [progress_callback](int npol, int pol, int iter, int px, int py, float peak) {
            if (!progress_callback.is_none()) {
                try {
                    progress_callback(npol, pol, iter, px, py, peak);
                } catch (const std::runtime_error& e) {
                    // Swallow callback exceptions to prevent algorithm termination
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
                    should_stop = 0;  // Continue on callback error
                }
            }
        };
    
    // Run CLEAN algorithm
    int final_iter = start_iter;
    hclean::hclean(
        static_cast<float*>(model_info.ptr),           // model image output
        static_cast<float*>(dirty_copy_info.ptr),      // dirty/residual (copy)
        static_cast<const float*>(psf_info.ptr),       // PSF
        domask, mask_ptr,                              // mask
        nx, ny, npol,                                  // dimensions
        xbeg, xend, ybeg, yend,                       // clean box
        max_iter, start_iter, final_iter,             // iteration control
        gain, threshold, speedup,                     // algorithm parameters
        msgput_func, stopnow_func                     // callbacks
    );
    
    // Calculate total flux cleaned
    py::array_t<float> model_flat = model_image.reshape({npol * ny * nx});
    float total_flux = 0.0f;
    float* model_data = static_cast<float*>(model_flat.request().ptr);
    for (int i = 0; i < npol * ny * nx; ++i) {
        total_flux += std::abs(model_data[i]);
    }
    
    // Find final peak in residual
    float final_min, final_max;
    hclean::maximg(
        static_cast<const float*>(dirty_copy_info.ptr),
        domask, mask_ptr,
        nx, ny, npol,
        final_min, final_max
    );
    float final_peak = std::max(std::abs(final_min), std::abs(final_max));
    
    // Return comprehensive results
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
 * Simplified wrapper for basic CLEAN without callbacks
 */
py::dict hclean_noprogress(
    py::array_t<float> dirty_image,
    py::array_t<float> psf_array,
    float gain = 0.1f,
    float threshold = 0.0f,
    int max_iter = 100,
    py::tuple clean_box = py::make_tuple(-1, -1, -1, -1)
) {
    return hogbom_clean(
        dirty_image, psf_array, py::array_t<float>(), // no mask
        clean_box, max_iter, 0,  // start from iteration 0
        gain, threshold, 0.0f,   // no speedup
        py::none(), py::none()   // no callbacks
    );
}

PYBIND11_MODULE(hclean, m) {
    m.doc() = "Hogbom CLEAN algorithm - Adapted from casacore hclean.f";
    
    // Min/max finding utility
    m.def("maximg", &maximg,
          "Find minimum and maximum values in 3D image with optional masking",
          py::arg("image"), py::arg("mask") = py::array_t<float>());
    
    // Full-featured traditional CLEAN
    m.def("hclean", &hogbom_clean,
           "Hogbom CLEAN algorithm - Adapted from casacore hclean.f",
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
    
    // Simplified interface
    m.def("clean", &hclean_noprogress,
          "Hogbom CLEAN interface without any progress callbacks",
          py::arg("dirty_image"),
          py::arg("psf"),
          py::arg("gain") = 0.1f,
          py::arg("threshold") = 0.0f,
          py::arg("max_iter") = 100,
          py::arg("clean_box") = py::make_tuple(-1, -1, -1, -1));
    
    // Utility to convert between array layouts
    // m.def("convert_to_pol_first", [](py::array_t<float> array) {
    //     py::buffer_info info = array.request();
    //     if (info.ndim != 3) {
    //         throw std::runtime_error("Array must be 3D");
    //     }
    //
    //     // If input is [ny, nx, npol], transpose to [npol, ny, nx]
    //     if (info.shape[2] <= 4) {  // Assume last dim is polarizations if ≤ 4
    //         return array.attr("transpose")(py::make_tuple(2, 0, 1));
    //     } else {
    //         return array;  // Already in correct format
    //     }
    // }, "Convert 3D array to polarization-first layout [npol, ny, nx]");
    //
    // m.def("convert_from_pol_first", [](py::array_t<float> array) {
    //     py::buffer_info info = array.request();
    //     if (info.ndim != 3) {
    //         throw std::runtime_error("Array must be 3D");
    //     }
    //
    //     // If input is [npol, ny, nx], transpose to [ny, nx, npol]
    //     if (info.shape[0] <= 4) {  // Assume first dim is polarizations if ≤ 4
    //         return array.attr("transpose")(py::make_tuple(1, 2, 0));
    //     } else {
    //         return array;  // Already in correct format
    //     }
    // }, "Convert 3D array from polarization-first to spatial-first layout [ny, nx, npol]");
}
