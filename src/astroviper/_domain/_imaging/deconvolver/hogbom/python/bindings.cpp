#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <cstring>
#include "hogbom_clean.hpp"

namespace py = pybind11;

py::array_t<double> numpy_hogbom_clean(
    py::array_t<double> dirty_map,
    py::array_t<double> beam,
    double gain = 0.1,
    double threshold = 0.0,
    int max_iter = 100,
    py::tuple clean_window = py::make_tuple(-1, -1, -1, -1)
) {
    // Get buffer info
    py::buffer_info dirty_info = dirty_map.request();
    py::buffer_info beam_info = beam.request();
    
    // Validate dimensions
    if (dirty_info.ndim != 2) {
        throw std::runtime_error("Dirty map must be 2D");
    }
    if (beam_info.ndim != 2) {
        throw std::runtime_error("Beam must be 2D");
    }
    
    // Extract dimensions
    int ny = dirty_info.shape[0];
    int nx = dirty_info.shape[1];
    int beam_ny = beam_info.shape[0];
    int beam_nx = beam_info.shape[1];
    
    // Create a copy of the dirty map to modify
    py::array_t<double> dirty_copy = py::array_t<double>(dirty_info.shape);
    std::memcpy(dirty_copy.mutable_unchecked().mutable_data(), dirty_info.ptr, dirty_info.size * sizeof(double));
    py::buffer_info dirty_copy_info = dirty_copy.request();
    
    // Set up parameters
    hogbom::CleanParams params;
    params.gain = gain;
    params.threshold = threshold;
    params.max_iter = max_iter;
    
    // Parse clean window
    if (clean_window.size() == 4) {
        params.x_begin = py::cast<int>(clean_window[0]);
        params.x_end = py::cast<int>(clean_window[1]);
        params.y_begin = py::cast<int>(clean_window[2]);
        params.y_end = py::cast<int>(clean_window[3]);
        
        // Handle -1 values (use full dimension)
        if (params.x_begin == -1) params.x_begin = 0;
        if (params.x_end == -1) params.x_end = nx;
        if (params.y_begin == -1) params.y_begin = 0;
        if (params.y_end == -1) params.y_end = ny;
    }
    
    // Run CLEAN algorithm
    auto results = hogbom::hogbom_clean(
        static_cast<double*>(dirty_copy_info.ptr),
        nx, ny,
        static_cast<const double*>(beam_info.ptr),
        beam_nx, beam_ny,
        params
    );
    
    return dirty_copy;
}

py::dict numpy_hogbom_clean_with_components(
    py::array_t<double> dirty_map,
    py::array_t<double> beam,
    double gain = 0.1,
    double threshold = 0.0,
    int max_iter = 100,
    py::tuple clean_window = py::make_tuple(-1, -1, -1, -1)
) {
    // Get buffer info
    py::buffer_info dirty_info = dirty_map.request();
    py::buffer_info beam_info = beam.request();
    
    // Validate dimensions
    if (dirty_info.ndim != 2) {
        throw std::runtime_error("Dirty map must be 2D");
    }
    if (beam_info.ndim != 2) {
        throw std::runtime_error("Beam must be 2D");
    }
    
    // Extract dimensions
    int ny = dirty_info.shape[0];
    int nx = dirty_info.shape[1];
    int beam_ny = beam_info.shape[0];
    int beam_nx = beam_info.shape[1];
    
    // Create a copy of the dirty map to modify
    py::array_t<double> dirty_copy = py::array_t<double>(dirty_info.shape);
    std::memcpy(dirty_copy.mutable_unchecked().mutable_data(), dirty_info.ptr, dirty_info.size * sizeof(double));
    py::buffer_info dirty_copy_info = dirty_copy.request();
    
    // Set up parameters
    hogbom::CleanParams params;
    params.gain = gain;
    params.threshold = threshold;
    params.max_iter = max_iter;
    
    // Parse clean window
    if (clean_window.size() == 4) {
        params.x_begin = py::cast<int>(clean_window[0]);
        params.x_end = py::cast<int>(clean_window[1]);
        params.y_begin = py::cast<int>(clean_window[2]);
        params.y_end = py::cast<int>(clean_window[3]);
        
        // Handle -1 values (use full dimension)
        if (params.x_begin == -1) params.x_begin = 0;
        if (params.x_end == -1) params.x_end = nx;
        if (params.y_begin == -1) params.y_begin = 0;
        if (params.y_end == -1) params.y_end = ny;
    }
    
    // Run CLEAN algorithm
    auto results = hogbom::hogbom_clean(
        static_cast<double*>(dirty_copy_info.ptr),
        nx, ny,
        static_cast<const double*>(beam_info.ptr),
        beam_nx, beam_ny,
        params
    );
    
    // Convert components to numpy arrays
    py::array_t<double> component_flux = py::cast(results.component_flux);
    py::array_t<int> component_x = py::cast(results.component_x);
    py::array_t<int> component_y = py::cast(results.component_y);
    
    // Create return dictionary
    py::dict result_dict;
    result_dict["residual_map"] = dirty_copy;
    result_dict["component_flux"] = component_flux;
    result_dict["component_x"] = component_x;
    result_dict["component_y"] = component_y;
    result_dict["iterations"] = results.iterations_performed;
    result_dict["final_peak"] = results.final_peak;
    result_dict["total_flux_cleaned"] = results.total_flux_cleaned;
    
    return result_dict;
}

py::dict numpy_hogbom_clean_multipol(
    py::array_t<double> dirty_maps,
    py::array_t<double> beam,
    double gain = 0.1,
    double threshold = 0.0,
    int max_iter = 100,
    py::tuple clean_window = py::make_tuple(-1, -1, -1, -1)
) {
    // Get buffer info
    py::buffer_info dirty_info = dirty_maps.request();
    py::buffer_info beam_info = beam.request();
    
    // Validate dimensions
    if (dirty_info.ndim != 3) {
        throw std::runtime_error("Dirty maps must be 3D (npol, ny, nx)");
    }
    if (beam_info.ndim != 2) {
        throw std::runtime_error("Beam must be 2D");
    }
    
    // Extract dimensions
    int npol = dirty_info.shape[0];
    int ny = dirty_info.shape[1];
    int nx = dirty_info.shape[2];
    int beam_ny = beam_info.shape[0];
    int beam_nx = beam_info.shape[1];
    
    // Create a copy of the dirty maps to modify
    py::array_t<double> dirty_copy = py::array_t<double>(dirty_info.shape);
    std::memcpy(dirty_copy.mutable_unchecked().mutable_data(), dirty_info.ptr, dirty_info.size * sizeof(double));
    py::buffer_info dirty_copy_info = dirty_copy.request();
    
    // Set up parameters
    hogbom::CleanParams params;
    params.gain = gain;
    params.threshold = threshold;
    params.max_iter = max_iter;
    
    // Parse clean window
    if (clean_window.size() == 4) {
        params.x_begin = py::cast<int>(clean_window[0]);
        params.x_end = py::cast<int>(clean_window[1]);
        params.y_begin = py::cast<int>(clean_window[2]);
        params.y_end = py::cast<int>(clean_window[3]);
        
        // Handle -1 values (use full dimension)
        if (params.x_begin == -1) params.x_begin = 0;
        if (params.x_end == -1) params.x_end = nx;
        if (params.y_begin == -1) params.y_begin = 0;
        if (params.y_end == -1) params.y_end = ny;
    }
    
    // Run multi-pol CLEAN algorithm
    auto results = hogbom::hogbom_clean_multipol(
        static_cast<double*>(dirty_copy_info.ptr),
        nx, ny, npol,
        static_cast<const double*>(beam_info.ptr),
        beam_nx, beam_ny,
        params
    );
    
    // Convert components to numpy arrays
    py::array_t<double> component_flux = py::cast(results.component_flux);
    py::array_t<int> component_x = py::cast(results.component_x);
    py::array_t<int> component_y = py::cast(results.component_y);
    
    // Create return dictionary
    py::dict result_dict;
    result_dict["residual_maps"] = dirty_copy;
    result_dict["component_flux"] = component_flux;
    result_dict["component_x"] = component_x;
    result_dict["component_y"] = component_y;
    result_dict["iterations"] = results.iterations_performed;
    result_dict["final_peak"] = results.final_peak;
    result_dict["total_flux_cleaned"] = results.total_flux_cleaned;
    
    return result_dict;
}

PYBIND11_MODULE(hogbom_clean, m) {
    m.doc() = "Hogbom CLEAN algorithm implementation";
    
    // Bind CleanParams structure
    py::class_<hogbom::CleanParams>(m, "CleanParams")
        .def(py::init<>())
        .def_readwrite("gain", &hogbom::CleanParams::gain)
        .def_readwrite("threshold", &hogbom::CleanParams::threshold)
        .def_readwrite("max_iter", &hogbom::CleanParams::max_iter)
        .def_readwrite("x_begin", &hogbom::CleanParams::x_begin)
        .def_readwrite("x_end", &hogbom::CleanParams::x_end)
        .def_readwrite("y_begin", &hogbom::CleanParams::y_begin)
        .def_readwrite("y_end", &hogbom::CleanParams::y_end);
    
    // Bind CleanResults structure
    py::class_<hogbom::CleanResults>(m, "CleanResults")
        .def(py::init<>())
        .def_readwrite("component_flux", &hogbom::CleanResults::component_flux)
        .def_readwrite("component_x", &hogbom::CleanResults::component_x)
        .def_readwrite("component_y", &hogbom::CleanResults::component_y)
        .def_readwrite("iterations_performed", &hogbom::CleanResults::iterations_performed)
        .def_readwrite("final_peak", &hogbom::CleanResults::final_peak)
        .def_readwrite("total_flux_cleaned", &hogbom::CleanResults::total_flux_cleaned);
    
    // High-level NumPy-friendly functions
    m.def("clean", &numpy_hogbom_clean,
          "Hogbom CLEAN algorithm (returns residual map only)",
          py::arg("dirty_map"), py::arg("beam"),
          py::arg("gain") = 0.1, py::arg("threshold") = 0.0,
          py::arg("max_iter") = 100, py::arg("clean_window") = py::make_tuple(-1, -1, -1, -1));
    
    m.def("clean_with_components", &numpy_hogbom_clean_with_components,
          "Hogbom CLEAN algorithm (returns full results)",
          py::arg("dirty_map"), py::arg("beam"),
          py::arg("gain") = 0.1, py::arg("threshold") = 0.0,
          py::arg("max_iter") = 100, py::arg("clean_window") = py::make_tuple(-1, -1, -1, -1));
    
    m.def("clean_multipol", &numpy_hogbom_clean_multipol,
          "Multi-polarization Hogbom CLEAN algorithm",
          py::arg("dirty_maps"), py::arg("beam"),
          py::arg("gain") = 0.1, py::arg("threshold") = 0.0,
          py::arg("max_iter") = 100, py::arg("clean_window") = py::make_tuple(-1, -1, -1, -1));
    
    // Lower-level C++ functions (for advanced users)
    m.def("find_peak", [](py::array_t<double> data, py::tuple window) {
        py::buffer_info info = data.request();
        if (info.ndim != 2) {
            throw std::runtime_error("Data must be 2D");
        }
        
        int ny = info.shape[0];
        int nx = info.shape[1];
        
        int x_begin = 0, x_end = nx, y_begin = 0, y_end = ny;
        if (window.size() == 4) {
            x_begin = py::cast<int>(window[0]);
            x_end = py::cast<int>(window[1]);
            y_begin = py::cast<int>(window[2]);
            y_end = py::cast<int>(window[3]);
        }
        
        auto [peak_val, peak_x, peak_y] = hogbom::find_peak(
            static_cast<const double*>(info.ptr),
            nx, ny, x_begin, x_end, y_begin, y_end
        );
        
        return py::make_tuple(peak_val, peak_x, peak_y);
    }, "Find peak in 2D array within window",
       py::arg("data"), py::arg("window") = py::make_tuple(0, -1, 0, -1));
}