#pragma once

// Adaptive Scale Pixel (Asp / "AAspClean") deconvolution, ported from CASA's
// synthesis/MeasurementEquations/AspMatrixCleaner.{h,cc} and its driver
// synthesis/ImagerObjects/SDAlgorithmAAspClean.{h,cc}. All casacore
// dependencies (Matrix, FFTServer, IPosition, LogIO, ...) and the ALGLIB
// L-BFGS dependency have been removed and replaced with the C++ standard
// library plus the small self-contained headers asp_fft.hpp (FFTs via the
// vendored pocketfft header) and asp_lbfgs.hpp.
//
// Deliberate departures from CASA, documented at the code:
//  * the FFT bookkeeping (separable Aspen spectra, paired inverse transforms)
//    computes the same quantities with roughly an eighth of the transforms;
//  * the residual update covers the whole image, not a PSF-sized box;
//  * there is no "diverging" guard (AspMatrixCleaner "5. Diverging" in CASA
//    6.7.5.18 and LibRA fc54ee25): the model update stops on the divergence
//    test shared with
//    the other deconvolvers (../../include/plane_stop_test.hpp), whose two
//    limits are off unless the caller enables them; an infinite peak always
//    stops the model update. astroviper's Python layer gives Asp the RMS
//    limit with the factor (1 + gain) and no peak limit.
// A residual pixel outside the search region (mask not above 0.99) that is
// not finite is zero during the model update and gets its value back
// afterwards.
//
// The algorithm finds, at each model update iteration, the best "Aspen" (a
// Gaussian component whose amplitude, scale size and position are optimized by
// L-BFGS) from the residual, then subtracts its PSF response and adds it to
// the model. It falls back to Hogbom-style point cleaning when the optimal
// scale collapses to zero. See the .cpp for the per-step correspondence with
// the CASA source.

#include <cstddef>

#include "../../include/plane_stop_test.hpp"

namespace aspclean {

struct AspResult {
    int iterations = 0;            // model update iterations performed
    int retval = 0;               //  1 converged
                                  //  0 not converged, behaving normally
                                  // -1 stopped on consecutive smallest scale
                                  // -2 large-scale negative / diverging
                                  // -3 diverging (the divergence test shared
                                  //    with the other deconvolvers)
    int stop_code = deconvolvers::STOP_NONE;  // why the model update ended, as
                                  // the other deconvolvers report it; Asp's own
                                  // tests (optimum strength, end of a Hogbom
                                  // run, stop point mode) are STOP_ALGORITHM
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

// Run the Asp model update on a single 2-D plane, in place.
//   residual : [ny][nx] dirty image -> residual          (in/out, writable)
//   model    : [ny][nx] model, components accumulated     (in/out, writable)
//   psf      : [ny][nx] point spread function             (read only)
//   mask     : [ny][nx] clean mask, or nullptr for "clean everywhere"
// `psf_width <= 0` requests an internal PSF Gaussian-width estimate.
// `largestscale < 0`, `stoppointmode < 0`, `norm_method == 1` reproduce the
// CASA defaults. `divergence` is the divergence test shared with the other
// deconvolvers (plane_stop_test.hpp). No copies of the caller's buffers are made.
template <typename T>
AspResult aspclean_plane(T* residual, T* model, const T* psf, const T* mask,
                         int nx, int ny, double gain, double threshold_per_cycle,
                         int max_iter_per_cycle_updated, double fusedthreshold, double psf_width,
                         int largestscale, int stoppointmode, int norm_method,
                         bool verbose,
                         const deconvolvers::DivergenceControl<double>& divergence);

// Run aspclean_plane independently on every (time, frequency, polarization)
// plane of a 5-D cube [nt, nf, np, ny, nx], in place, across `processing_function_threads`
// worker threads. The PSF cube may have np_psf == np_img or np_psf == 1
// (Stokes-I broadcast). `results` must have length nt*nf*np_img.
//
// Iteration control is independent per plane: `threshold_per_cycle` and
// `max_iter_per_cycle_updated` are per-plane arrays of length nt*nf*np_img in
// (t, f, p) C-order, so every (time, frequency, polarization) plane is cleaned
// with its own threshold and iteration limit.
template <typename T>
void aspclean_cube(T* residual, T* model, const T* psf, const T* mask,
                   int nt, int nf, int np_img, int np_psf, int nx, int ny,
                   double gain, const double* threshold_per_cycle,
                   const int* max_iter_per_cycle_updated,
                   double fusedthreshold, double psf_width, int largestscale,
                   int stoppointmode, int norm_method, int processing_function_threads,
                   const deconvolvers::DivergenceControl<double>& divergence,
                   AspResult* results);

}  // namespace aspclean
