#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <string>

#define FORCE_IMPORT_ARRAY
namespace py = pybind11;

#include "single_cf_gridder.h"

class single_cf_gridder_pybind : private single_cf_gridder {
public:
    single_cf_gridder_pybind() : single_cf_gridder() {}

    void create_grid(long n_imag_chan, long n_imag_pol, long image_size) {
        py::gil_scoped_release gil_release;

        single_cf_gridder::create_grid(n_imag_chan, n_imag_pol, image_size);
    }

    using complex = std::complex<double>;
    using ComplexArr = py::array_t<complex>;
    using DoubleArr = py::array_t<double>;
    using LongArr = py::array_t<long>;
    
    void set_grid(ComplexArr& grid){
        auto grid_ptr = reinterpret_cast<complex*>(grid.mutable_unchecked<4>().mutable_data(0, 0, 0, 0));
        single_cf_gridder::set_grid(grid_ptr);
    }

    void add_to_grid(LongArr& grid_shape, DoubleArr& sum_weight, ComplexArr& vis_data, LongArr& vis_shape, DoubleArr& uvw,
                     DoubleArr& freq_chan, LongArr& chan_map, LongArr& pol_map, DoubleArr& weight, DoubleArr& cgk_1D,
                     DoubleArr& delta_lm, int support, int oversampling) {
        py::gil_scoped_release gil_release;

        auto start = std::chrono::high_resolution_clock::now();

        auto grid_shape_ptr = reinterpret_cast<long*>(grid_shape.mutable_unchecked<1>().mutable_data(0));
        auto sum_weight_ptr = reinterpret_cast<double*>(sum_weight.mutable_unchecked<2>().mutable_data(0, 0));
        auto vis_data_ptr = reinterpret_cast<complex*>(vis_data.mutable_unchecked<4>().mutable_data(0, 0, 0, 0));
        auto vis_shape_ptr = reinterpret_cast<long*>(vis_shape.mutable_unchecked<1>().mutable_data(0));
        auto uvw_ptr = reinterpret_cast<double*>(uvw.mutable_unchecked<3>().mutable_data(0, 0, 0));
        auto freq_chan_ptr = reinterpret_cast<double*>(freq_chan.mutable_unchecked<1>().mutable_data(0));
        auto chan_map_ptr = reinterpret_cast<long*>(chan_map.mutable_unchecked<1>().mutable_data(0));
        auto pol_map_ptr = reinterpret_cast<long*>(pol_map.mutable_unchecked<1>().mutable_data(0));
        auto weight_ptr = reinterpret_cast<double*>(weight.mutable_unchecked<4>().mutable_data(0, 0, 0, 0));
        auto cgk_1D_ptr = reinterpret_cast<double*>(cgk_1D.mutable_unchecked<1>().mutable_data(0));
        auto delta_lm_ptr = reinterpret_cast<double*>(delta_lm.mutable_unchecked<1>().mutable_data(0));

        auto end = std::chrono::high_resolution_clock::now();
        auto convert_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "**convert_time " << convert_time << std::endl;

        start = std::chrono::high_resolution_clock::now();
        single_cf_gridder::add_to_grid(grid_shape_ptr, sum_weight_ptr, vis_data_ptr, vis_shape_ptr, uvw_ptr,
                                       freq_chan_ptr, chan_map_ptr, pol_map_ptr, weight_ptr, cgk_1D_ptr, delta_lm_ptr,
                                       support, oversampling);
        end = std::chrono::high_resolution_clock::now();
        auto grid_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "**grid_time " << grid_time << std::endl;
    }

    std::pair<int, int> grid(std::string vis_data_folder, int image_size, int n_time_chunks, int n_chan_chunks) {
        py::gil_scoped_release gil_release;

        return single_cf_gridder::grid(vis_data_folder,image_size, n_time_chunks, n_chan_chunks);
    }
};

PYBIND11_MODULE(pybind11_wrapper, m) {
    py::class_<single_cf_gridder_pybind>(m, "single_cf_gridder_pybind")
        .def(py::init<>())
        .def("create_grid", &single_cf_gridder_pybind::create_grid,py::arg().noconvert(), py::arg().noconvert(),
             py::arg().noconvert())
        .def("set_grid", &single_cf_gridder_pybind::set_grid,py::arg().noconvert())
        .def("add_to_grid", &single_cf_gridder_pybind::add_to_grid, py::arg().noconvert(), py::arg().noconvert(),
             py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
             py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
             py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert())
        .def("grid", &single_cf_gridder_pybind::grid, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
             py::arg().noconvert());
}
