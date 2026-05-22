#pragma once

// Adaptive Scale Pixel (Asp / "AAspClean") deconvolution, ported from CASA's
// synthesis/MeasurementEquations/AspMatrixCleaner.{h,cc} and its driver
// synthesis/ImagerObjects/SDAlgorithmAAspClean.{h,cc}. All casacore
// dependencies (Matrix, FFTServer, IPosition, LogIO, ...) and the ALGLIB
// L-BFGS dependency have been removed and replaced with the C++ standard
// library plus the small self-contained headers asp_fft.hpp and
// asp_lbfgs.hpp.
//
// The algorithm finds, at each minor-cycle iteration, the best "Aspen" (a
// Gaussian component whose amplitude, scale size and position are optimized by
// L-BFGS) from the residual, then subtracts its PSF response and adds it to
// the model. It falls back to Hogbom-style point cleaning when the optimal
// scale collapses to zero. See the .cpp for the per-step correspondence with
// the CASA source.

#include <cstddef>

namespace aspclean {

struct AspResult {
    int iterations = 0;            // minor-cycle iterations performed
    int retval = 0;               //  1 converged
                                  //  0 not converged, behaving normally
                                  // -1 stopped on consecutive smallest scale
                                  // -2 large-scale negative / diverging
                                  // -3 diverging
    double peak_residual = 0.0;   // |peak| of the (masked) residual at exit
    double model_flux = 0.0;      // sum of the model image at exit
    bool switched_to_hogbom = false;
    double psf_width = 0.0;       // PSF Gaussian width (pixels) used for scales
};

// Estimate the PSF Gaussian width (mean FWHM in pixels) that seeds the Asp
// initial scale sizes (0, w, 2w, 4w, 8w). `psf` is row-major [ny][nx].
double psf_gaussian_width(const double* psf, int nx, int ny);

// FFT-based circular convolution of two real images (row-major [ny][nx]),
// recentred so that convolving two centre-origin images yields a centre-origin
// result. This is the convolution primitive the deconvolver uses internally
// (casacore fft0 + flip). Exposed for testing.
void convolve_centered(const double* a, const double* b, int nx, int ny,
                       double* out);

// Run the Asp minor cycle on a single 2-D plane, in place.
//   residual : [ny][nx] dirty image -> residual          (in/out, writable)
//   model    : [ny][nx] model, components accumulated     (in/out, writable)
//   psf      : [ny][nx] point spread function             (read only)
//   mask     : [ny][nx] clean mask, or nullptr for "clean everywhere"
// `psf_width <= 0` requests an internal PSF Gaussian-width estimate.
// `largestscale < 0`, `stoppointmode < 0`, `norm_method == 1` reproduce the
// CASA defaults. No copies of the caller's buffers are made.
template <typename T>
AspResult aspclean_plane(T* residual, T* model, const T* psf, const T* mask,
                         int nx, int ny, double gain, double threshold,
                         int niter, double fusedthreshold, double psf_width,
                         int largestscale, int stoppointmode, int norm_method,
                         bool verbose);

// Run aspclean_plane independently on every (time, frequency, polarization)
// plane of a 5-D cube [nt, nf, np, ny, nx], in place, across `num_threads`
// worker threads. The PSF cube may have np_psf == np_img or np_psf == 1
// (Stokes-I broadcast). `results` must have length nt*nf*np_img.
template <typename T>
void aspclean_cube(T* residual, T* model, const T* psf, const T* mask,
                   int nt, int nf, int np_img, int np_psf, int nx, int ny,
                   double gain, double threshold, int niter,
                   double fusedthreshold, double psf_width, int largestscale,
                   int stoppointmode, int norm_method, int num_threads,
                   AspResult* results);

}  // namespace aspclean
