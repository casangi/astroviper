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
//  * the residual update and the Aspen fit use exact linear convolutions with
//    the image-sized PSF (no wrap-around at the image edges, so residual ==
//    dirty - model (*) psf holds on the whole image), computed with separable
//    spectra and real transforms, and point components need no transform at
//    all;
//  * the per-Aspen L-BFGS (asp_lbfgs.hpp) is an independent implementation of
//    the ALGLIB minlbfgs behaviour CASA and LibRA rely on, More-Thuente line
//    search included;
//  * a model update does not end after a Hogbom run (unless
//    stop_after_hogbom_run): CASA and LibRA end it there once the RMS has
//    fallen by a factor 1.5 or more since its start, or risen, because their
//    residual drifts from the true one; here it is exact, so the model update
//    goes on to its threshold, which takes fewer imaging cycles;
//  * there is no "diverging" guard (AspMatrixCleaner "5. Diverging" in CASA
//    6.7.5.18 and LibRA fc54ee25): the model update stops on the divergence
//    test shared with
//    the other deconvolvers (../../include/plane_stop_test.hpp), whose two
//    limits are off unless the caller enables them; an infinite peak always
//    stops the model update. astroviper's Python layer gives Asp the RMS
//    limit with the factor (1 + gain) and no peak limit;
//  * a largestscale below the PSF width gives the initial scales 0,
//    largestscale and the PSF width, as in LibRA, not the point scale only.
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
#include <vector>

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
    // Wall time (seconds) of the parts of the model update, and their counts.
    double t_setup = 0.0;         // PSF spectra, initial scales and masks, first search
    double t_search = 0.0;        // residual (*) initial scales and peak search
    double t_fit = 0.0;           // per-Aspen L-BFGS fits
    double t_update_scale = 0.0;  // model and residual updates of scale components
    double t_update_point = 0.0;  // model and residual updates of point components
    double t_bookkeeping = 0.0;   // RMS, peak, scale image per iteration
    int n_scale_components = 0;
    int n_point_components = 0;
    int n_fits = 0;
    int n_fit_evaluations = 0;
    // Every Asp component the model update subtracted,
    // as (i, j, scale, flux) per component, when AspOptions::record_components.
    std::vector<double> components;
    // Noise of the residual smoothed with each initial scale
    // (AspOptions::scale_significance).
    std::vector<double> scale_noise;
    // Residual-shaped components (AspOptions::residual_components) and their time.
    int n_residual_components = 0;
    int n_residual_fit_evaluations = 0;
    double t_residual_components = 0.0;
    // Iterations below the threshold on a significant smoothed residual
    // (AspOptions::scale_significance) and their time, search and fit included.
    int n_significance_components = 0;
    double t_significance = 0.0;
    // Hogbom runs: switches from the Asp iterations to point components, and
    // the switches of each test: two iterations without change of the peak
    // residual, 5 of the last 10 Aspens of scale 0, the peak residual or the
    // optimum strength below fusedthreshold.
    int n_hogbom_runs = 0;
    int n_switch_no_change = 0;
    int n_switch_point_scales = 0;
    int n_switch_fused = 0;
    // The record of every component (AspOptions::record_history): the
    // signed peak of the masked residual before it, the RMS of the masked
    // residual after it, its pixel (row, column), its scale in pixels (0 a
    // point, -1 residual shaped) and its flux.
    std::vector<double> history_peak, history_rms, history_l, history_m, history_scale,
        history_flux;
    // Per residual-shaped component fit (peak, frac, eps_frac, 0, 0, the
    // objective at the end of the fit), when record_components.
    std::vector<double> residual_fits;
};

// Options of Asp beyond CASA's and LibRA's AspMatrixCleaner, all off by default
// in the kernel; deconvolution.asp_clean() turns scale_significance,
// residual_components and rc_positive_only on: the first two together clean
// the ring residuals Asp leaves on resolved emission, with far fewer
// iterations.
//   scale_significance: measure the noise of the residual smoothed with each
//     initial scale (robust, outside the mask) and, once the peak is below
//     the threshold, go on cleaning the most significant smoothed residual
//     while it is as significant as the threshold is at scale 0 (or as
//     `significance` times its noise when that is > 0), subtracting
//     significance_gain (when > 0, else the gain) of each such component.
//     Each noise is sampled outside the mask and a border of 3 * scale size
//     + 1 pixels (the outermost row and column at scale 0), one pixel in every
//     max(1, nx * ny / 400000); with fewer than 100 samples it is +inf in
//     AspResult::scale_noise. A noise is used only when finite and positive:
//     a scale without one is never significant, and without one at scale 0
//     (no mask, a mask that leaves too few pixels outside, or a residual
//     mostly 0 outside it) no smoothed residual is, with `significance` > 0
//     as well.
//   record_components: return the components in AspResult::components.
//   residual_components: before the Asp iterations of a model update, clean
//     with components shaped by the residual (Ma, Zhang & Zhang 2026, A&A
//     713, A164): C = R tanh((sR - T) / eps) on the mask pixels with sR >= T,
//     T = max(frac |R_peak|, rc_noise_floor * noise[, the threshold when
//     rc_floor_at_threshold, the default]), eps = eps_frac |R_peak|,
//     s = sign R_peak; (frac, eps_frac) fitted within their bounds by least
//     squares of the masked residual, down to rc_switch_factor times the
//     threshold; the Asp iterations take it on to the threshold. With
//     rc_positive_only the components take only positive residual: R_peak is
//     the largest positive residual and s = +1, so the Asp iterations clean
//     the negative residual.
//   record_history: record every component (AspResult::history_*).
//   no_change_tolerance: an Asp iteration that changes the peak residual by
//     less than this fraction of the previous peak counts as no change (two
//     switch to a Hogbom run); 0 keeps CASA's and LibRA's absolute 1e-4, in
//     the units of the image, which counts every iteration once the peak
//     residual is below about a mJy.
struct AspOptions {
    bool scale_significance = false;
    double significance = 0.0;
    bool record_components = false;
    bool residual_components = false;
    double rc_frac_min = 0.3, rc_frac_max = 0.99;
    double rc_eps_min = 1e-3, rc_eps_max = 10.0;
    double rc_noise_floor = 2.0;
    double rc_switch_factor = 1.0;
    bool rc_floor_at_threshold = true;
    bool rc_positive_only = false;
    double significance_gain = 0.0;
    int rc_fit_iterations = 10;
    double no_change_tolerance = 0.0;
    bool record_history = false;
};

// Estimate the PSF Gaussian width (mean FWHM in pixels) that seeds the Asp
// initial scale sizes (0, w, 2w, 4w, 8w). `psf` is row-major [ny][nx].
double psf_gaussian_width(const double* psf, int nx, int ny);

// FFT-based circular convolution of two real images (row-major [ny][nx]),
// recentred so that convolving two centre-origin images yields a centre-origin
// result (casacore fft0 + flip). Exposed for testing; the residual update uses
// the linear convolution instead.
void convolve_centered(const double* a, const double* b, int nx, int ny,
                       double* out);

// Run the Asp model update on a single 2-D plane, in place.
//   residual : [ny][nx] dirty image -> residual          (in/out, writable)
//   model    : [ny][nx] model, components accumulated     (in/out, writable)
//   psf      : [ny][nx] point spread function             (read only)
//   mask     : [ny][nx] clean mask, or nullptr for "clean everywhere"
// `psf_width <= 0` requests an internal PSF Gaussian-width estimate.
// `largestscale < 0`, `stoppointmode < 0`, `norm_method == 1` reproduce the
// CASA defaults, and `stop_after_hogbom_run` CASA's end of the model update
// after a Hogbom run. `divergence` is the divergence test shared with the other
// deconvolvers (plane_stop_test.hpp). `fft_threads` is the number of threads
// of every 2-D transform. No copies of the caller's buffers are made.
template <typename T>
AspResult aspclean_plane(T* residual, T* model, const T* psf, const T* mask,
                         int nx, int ny, double gain, double threshold_per_cycle,
                         int max_iter_per_cycle_updated, double fusedthreshold, double psf_width,
                         int largestscale, int stoppointmode, int norm_method,
                         bool stop_after_hogbom_run, bool verbose,
                         const deconvolvers::DivergenceControl<double>& divergence,
                         int fft_threads = 1, const AspOptions& options = AspOptions{});

// Run aspclean_plane independently on every (time, frequency, polarization)
// plane of a 5-D cube [nt, nf, np, ny, nx], in place, across `processing_function_threads`
// worker threads (one plane per thread; when there are more threads than
// planes, each plane's transforms share the rest). The PSF cube may have np_psf == np_img or np_psf == 1
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
                   int stoppointmode, int norm_method, bool stop_after_hogbom_run,
                   int processing_function_threads,
                   const deconvolvers::DivergenceControl<double>& divergence,
                   AspResult* results, const AspOptions& options = AspOptions{});

}  // namespace aspclean
