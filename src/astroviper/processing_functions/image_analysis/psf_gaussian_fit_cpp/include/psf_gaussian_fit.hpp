// CASA-compatible PSF Gaussian beam fit (port of CAS-13022).
//
// Port of casa6 StokesImageUtil::FitGaussianPSF(psf, beam, psfcutoff)
// (synthesis/TransformMachines/StokesImageUtil.cc) with its helpers
// FindNpoints and ResamplePSF (Numerical-Recipes bicubic, as in casacore
// Interpolate2D::CUBIC) and a Levenberg-Marquardt fit of the casacore
// Gaussian2D functional (height and centre fixed, as CASA fixes them).
//
// Beams are returned in radians as [major FWHM, minor FWHM, position angle],
// the same pixel-frame position-angle convention as the casacore Gaussian2D
// (major axis along (-sin pa, cos pa) in (axis0, axis1) pixel space, i.e. the
// convention of astroviper's BEAM_FIT_PARAMS, mod pi).
#pragma once

#include <cstddef>

namespace astroviper::psf_gaussian_fit {

// Fit one PSF plane (values normalised internally). Returns 0 on a converged
// fit, 1 when the CASA fallback beam (2.5 pixels round) was used. Throws
// std::runtime_error for a zero peak or a peak outside the inner quarter.
int fit_plane(const double* psf, std::size_t nx, std::size_t ny, double delta_x,
              double delta_y, double psfcutoff, double beam[3]);

// Fit n_plane PSF planes [n_plane, nx, ny] into beams [n_plane, 3], with
// planes partitioned over n_threads std::threads. Returns the number of
// planes that used the fallback beam.
int fit_planes(const double* psf, std::size_t n_plane, std::size_t nx,
               std::size_t ny, double delta_x, double delta_y, double psfcutoff,
               double* beams, int n_threads);

}  // namespace astroviper::psf_gaussian_fit
