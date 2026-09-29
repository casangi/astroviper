// pybind11 bindings for the CASA-compatible PSF Gaussian fit.
//
// Follows the AstroVIPER pybind11 memory contract: typed py::array_t without
// forcecast (dtype/contiguity errors instead of silent copies), output written
// in place into a caller-owned array, GIL released around the C++ work.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>
#include <vector>

#include "psf_gaussian_fit.hpp"

namespace py = pybind11;

namespace {

template <typename T>
int fit_psf_beam_impl(
    py::array_t<T, py::array::c_style> psf,
    double delta_x,
    double delta_y,
    double psfcutoff,
    py::array_t<double, py::array::c_style> beams,
    int n_threads) {
    if (psf.ndim() != 3)
        throw std::invalid_argument("psf must be [n_plane, nx, ny].");
    const std::size_t n_plane = static_cast<std::size_t>(psf.shape(0));
    const std::size_t nx = static_cast<std::size_t>(psf.shape(1));
    const std::size_t ny = static_cast<std::size_t>(psf.shape(2));
    if (beams.ndim() != 2 ||
        static_cast<std::size_t>(beams.shape(0)) != n_plane ||
        beams.shape(1) != 3)
        throw std::invalid_argument("beams must be [n_plane, 3] float64.");

    const T* psf_ptr = psf.data();
    double* beams_ptr = beams.mutable_data();

    int n_fallback = 0;
    {
        py::gil_scoped_release release;
        if constexpr (std::is_same_v<T, double>) {
            n_fallback = astroviper::psf_gaussian_fit::fit_planes(
                psf_ptr, n_plane, nx, ny, delta_x, delta_y, psfcutoff,
                beams_ptr, n_threads);
        } else {
            // float32 planes: promote once (the fit works in double, as CASA
            // promotes its Float lattice to Double for the fitter).
            std::vector<double> promoted(n_plane * nx * ny);
            for (std::size_t k = 0; k < promoted.size(); ++k)
                promoted[k] = static_cast<double>(psf_ptr[k]);
            n_fallback = astroviper::psf_gaussian_fit::fit_planes(
                promoted.data(), n_plane, nx, ny, delta_x, delta_y, psfcutoff,
                beams_ptr, n_threads);
        }
    }
    return n_fallback;
}

}  // namespace

PYBIND11_MODULE(_psf_gaussian_fit_ext, m) {
    m.doc() =
        "CASA-compatible PSF Gaussian beam fit (StokesImageUtil::FitGaussianPSF "
        "port, CAS-13022).";
    m.def("fit_psf_beam", &fit_psf_beam_impl<double>, py::arg("psf"),
          py::arg("delta_x"), py::arg("delta_y"), py::arg("psfcutoff"),
          py::arg("beams"), py::arg("n_threads") = 1,
          "Fit [n_plane, nx, ny] float64 PSF planes into the in-place "
          "[n_plane, 3] float64 beams array ([major, minor, pa] radians); "
          "returns the number of planes that used the fallback beam.");
    m.def("fit_psf_beam", &fit_psf_beam_impl<float>, py::arg("psf"),
          py::arg("delta_x"), py::arg("delta_y"), py::arg("psfcutoff"),
          py::arg("beams"), py::arg("n_threads") = 1,
          "float32 overload of fit_psf_beam.");
}
