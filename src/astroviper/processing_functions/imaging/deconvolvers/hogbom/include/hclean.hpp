#pragma once

#include <cmath>
#include <functional>

namespace hclean {

/**
 * Why a plane's Hogbom model update ended. The values follow the iteration
 * controller's model update stop codes (imaging/utils/iteration_control.py).
 */
enum StopReason : int {
    STOP_NONE = 0,       ///< did not run (no iteration budget) or stopped on request
    STOP_MAX_ITER = 1,   ///< spent its iteration budget (max_iter_remaining)
    STOP_THRESHOLD = 2,  ///< peak residual at or below its threshold
    STOP_DIVERGED = 4    ///< residual diverging (RMS or peak) or peak not finite
};

/**
 * Divergence test of a model update.
 *
 * A plane is stopped as diverged when
 *
 *   - the RMS of its residual has been above rms_factor x (lowest RMS so far)
 *     for max_iter_divergence consecutive iterations (soft limit). The RMS is
 *     taken over the pixels the peak search covers, the clean box inside the
 *     mask. It is an average over many pixels and so it jitters far less than
 *     the peak: measured in healthy cleans, it rises by at most 0.09 x gain
 *     above its lowest value, where the peak rises by up to 0.44 x gain;
 *   - its peak exceeds peak_factor x (peak at the start of the model update):
 *     a runaway, stopped at once (hard limit);
 *   - its peak is not finite.
 *
 * max_iter_divergence < 0 disables the first two tests. The caller derives
 * the factors from the loop gain: rms_factor = 1 + gain / 10 and
 * peak_factor = 1 + gain.
 *
 * The kernels hand the stop test the sum of the squares and not the RMS. The
 * number of pixels of a plane does not change during a model update, so
 * RMS > rms_factor x lowest RMS is the same as
 * sum > rms_factor^2 x lowest sum, without a division or a square root.
 */
template<typename T>
struct DivergenceControl {
    int max_iter_divergence = -1;
    double rms_factor = 1.0;
    T peak_factor = static_cast<T>(1);
};

/**
 * Per-plane state of the stop tests, shared by all Hogbom kernels so that
 * they decide identically. check() is called once per iteration with the
 * plane's current peak |residual| and the sum of the squares of its residual
 * (both over the clean box inside the mask), before the component is
 * subtracted.
 *
 * A sum of squares that is not finite while the peak is finite (pixels that
 * are not a number, which the peak search skips) switches the soft limit off
 * for the plane and never stops it; the hard limit still applies.
 */
template<typename T>
class PlaneStopTest {
public:
    /// STOP_NONE to carry on, else the reason the plane stops now.
    int check(T peak, double sum_squares, T threshold,
              const DivergenceControl<T>& control) {
        if (!std::isfinite(peak)) {
            return STOP_DIVERGED;
        }
        // A peak AT the threshold stops the plane too, so an all-zero residual
        // with a zero threshold does no empty iterations.
        if (peak <= threshold) {
            return STOP_THRESHOLD;
        }
        if (!started_) {
            started_ = true;
            start_peak_ = peak;
            min_sum_squares_ = sum_squares;
        }
        if (control.max_iter_divergence >= 0) {
            if (peak > control.peak_factor * start_peak_) {
                return STOP_DIVERGED;
            }
            if (sum_squares
                > control.rms_factor * control.rms_factor * min_sum_squares_) {
                if (++n_above_ >= control.max_iter_divergence) {
                    return STOP_DIVERGED;
                }
            } else {
                n_above_ = 0;
            }
        }
        if (sum_squares < min_sum_squares_) {
            min_sum_squares_ = sum_squares;
        }
        return STOP_NONE;
    }

private:
    bool started_ = false;
    T start_peak_ = static_cast<T>(0);
    double min_sum_squares_ = 0.0;
    int n_above_ = 0;
};

/**
 * Sum of the squares of one image row over [xbeg, xend) inside the mask,
 * accumulated in double precision together with the search for the largest
 * |value| of the row. Every kernel scans its rows with this function and adds
 * the row sums in row order, so that all kernels, with any number of threads,
 * hand the stop test the same number to the last bit.
 *
 * @param row pointer to the first pixel of the row
 * @param mask_row pointer to the mask of the row, or nullptr for no mask
 * @param xbeg,xend column range of the clean box
 * @param row_max output: largest |value| of the row, 0 if no pixel is searched
 * @param row_ix output: column of that value (xbeg if no pixel is searched)
 * @return sum of the squares of the searched pixels of the row
 */
template<typename T>
inline double scan_row(const T* row, const bool* mask_row, int xbeg, int xend,
                       T& row_max, int& row_ix) {
    T m = static_cast<T>(0);
    int mi = xbeg;
    double sum_squares = 0.0;
    for (int ix = xbeg; ix < xend; ++ix) {
        if (mask_row == nullptr || mask_row[ix]) {
            const T value = row[ix];
            sum_squares += static_cast<double>(value) * static_cast<double>(value);
            const T v = std::abs(value);
            if (v > m) { m = v; mi = ix; }
        }
    }
    row_max = m;
    row_ix = mi;
    return sum_squares;
}

/**
 * Templated function to find minimum and maximum values in a 2D image array.
 * The mask is a bool array; pixels with mask == true are considered.
 */
template<typename T>
void maximg(const T* limagestep, int domask, const bool* lmask,
           int nx, int ny, T& fmin, T& fmax);

/**
 * Templated Hogbom CLEAN algorithm implementation on a single 2D plane.
 * The mask is a bool array; pixels with mask == true are considered in the
 * peak search and in the RMS of the divergence test. The plane stops when it
 * has spent max_iter_remaining, when its peak is at or below thres, or when
 * the divergence test fires (see DivergenceControl); stop_reason reports which
 * (a StopReason).
 */
template<typename T>
void clean(T* limage, T* limagestep, const T* lpsf,
           int domask, const bool* lmask, int nx, int ny,
           int xbeg, int xend, int ybeg, int yend,
           int max_iter_remaining, int siter, int& iter, T gain, T thres,
           T cspeedup,
           std::function<void(int, int, int, T)> msgput,
           std::function<void(int&)> stopnow,
           const DivergenceControl<T>& divergence,
           int& stop_reason);

/**
 * Templated Hogbom CLEAN over a full (nt, nf, np_img, ny, nx) image cube,
 * parallelized over (time, frequency, polarization) planes with std::thread.
 *
 * The residual_cube and model_cube buffers are modified in place: the
 * residual is updated as CLEAN components are subtracted, and the model
 * accumulates the found components. No intermediate copies of the plane
 * buffers are made; threads operate on disjoint (ny*nx) regions of the
 * supplied C-contiguous memory, so no inter-plane synchronization is
 * required beyond work-dispatch.
 *
 * The PSF cube may be provided with np_psf == np_img (one PSF per image
 * polarization) or np_psf == 1 (Stokes-I-only PSF broadcast across all
 * image polarizations).
 *
 * No Python-callable progress/stop callbacks are accepted: with multiple
 * worker threads these would require GIL-aware synchronization that
 * degrades throughput. Per-plane logging, if desired, should be done on
 * the Python side after the call returns.
 *
 * @param residual_cube 5D in-place residual buffer [nt, nf, np_img, ny, nx]
 * @param model_cube 5D in-place model buffer [nt, nf, np_img, ny, nx]
 * @param psf_cube 5D PSF buffer [nt, nf, np_psf, ny, nx], read-only
 * @param domask 0 if mask_cube is absent, 1 if present
 * @param mask_cube 5D mask buffer [nt, nf, np_img, ny, nx] or nullptr
 * @param nt number of time planes
 * @param nf number of frequency planes
 * @param np_img number of image polarization planes
 * @param np_psf number of PSF polarization planes (must be np_img or 1)
 * @param ny image height
 * @param nx image width
 * @param xbeg,xend,ybeg,yend per-plane clean box (0-based, exclusive upper)
 * @param max_iter_remaining per-plane maximum iterations; flat array of length
 *              nt*nf*np_img in (t, f, p) C-order (iteration control is
 *              independent for every time/frequency/polarization plane)
 * @param gain clean loop gain
 * @param thres per-plane flux cleaning threshold; flat array of length
 *              nt*nf*np_img in (t, f, p) C-order
 * @param cspeedup adaptive threshold speedup (0 disables)
 * @param processing_function_threads number of worker threads; clamped to [1, nplanes]
 * @param iter_out output iterations performed per plane, flat
 *                 array of length nt*nf*np_img in (t, f, p) C-order
 * @param divergence divergence test applied to every plane (DivergenceControl)
 * @param stop_out output StopReason per plane, flat array of length
 *                 nt*nf*np_img in (t, f, p) C-order
 */
template<typename T>
void clean_cube(T* residual_cube, T* model_cube, const T* psf_cube,
                int domask, const bool* mask_cube,
                int nt, int nf, int np_img, int np_psf,
                int ny, int nx,
                int xbeg, int xend, int ybeg, int yend,
                const int* max_iter_remaining, T gain, const T* thres, T cspeedup,
                int processing_function_threads, int* iter_out,
                const DivergenceControl<T>& divergence, int* stop_out);

/**
 * Many-threads Hogbom CLEAN over a (nt, nf, np_img, ny, nx) image cube whose
 * parallelism spans BOTH the planes and the rows within each plane.
 *
 * Unlike clean_cube (which runs one full per-plane clean per std::thread, so a
 * single-channel cube uses only n_pol threads), this synchronizes the
 * per-iteration peak search and PSF subtraction across the whole (plane, row)
 * work domain and distributes it over processing_function_threads std::thread workers, keeping
 * every thread busy even with only one or two planes. The per-plane algorithm,
 * tie-breaking (lowest row then column) and outputs are identical to clean_cube.
 *
 * Same parameters and semantics as clean_cube; processing_function_threads is the worker count
 * (clamped to [1, nplanes*ny]).
 */
template<typename T>
void clean_cube_many_threads(T* residual_cube, T* model_cube, const T* psf_cube,
                int domask, const bool* mask_cube,
                int nt, int nf, int np_img, int np_psf,
                int ny, int nx,
                int xbeg, int xend, int ybeg, int yend,
                const int* max_iter_remaining, T gain, const T* thres, T cspeedup,
                int processing_function_threads, int* iter_out,
                const DivergenceControl<T>& divergence, int* stop_out);

// Explicit template instantiation declarations
extern template void maximg<float>(const float* limagestep, int domask, const bool* lmask,
                                  int nx, int ny, float& fmin, float& fmax);

extern template void maximg<double>(const double* limagestep, int domask, const bool* lmask,
                                   int nx, int ny, double& fmin, double& fmax);

extern template void clean<float>(float* limage, float* limagestep, const float* lpsf,
                                 int domask, const bool* lmask, int nx, int ny,
                                 int xbeg, int xend, int ybeg, int yend,
                                 int max_iter_remaining, int siter, int& iter, float gain, float thres,
                                 float cspeedup,
                                 std::function<void(int, int, int, float)> msgput,
                                 std::function<void(int&)> stopnow,
                                 const DivergenceControl<float>& divergence,
                                 int& stop_reason);

extern template void clean<double>(double* limage, double* limagestep, const double* lpsf,
                                  int domask, const bool* lmask, int nx, int ny,
                                  int xbeg, int xend, int ybeg, int yend,
                                  int max_iter_remaining, int siter, int& iter, double gain, double thres,
                                  double cspeedup,
                                  std::function<void(int, int, int, double)> msgput,
                                  std::function<void(int&)> stopnow,
                                 const DivergenceControl<double>& divergence,
                                 int& stop_reason);

extern template void clean_cube<float>(float* residual_cube, float* model_cube,
                                       const float* psf_cube, int domask,
                                       const bool* mask_cube,
                                       int nt, int nf, int np_img, int np_psf,
                                       int ny, int nx,
                                       int xbeg, int xend, int ybeg, int yend,
                                       const int* max_iter_remaining, float gain, const float* thres, float cspeedup,
                                       int processing_function_threads, int* iter_out,
                const DivergenceControl<float>& divergence, int* stop_out);

extern template void clean_cube<double>(double* residual_cube, double* model_cube,
                                        const double* psf_cube, int domask,
                                        const bool* mask_cube,
                                        int nt, int nf, int np_img, int np_psf,
                                        int ny, int nx,
                                        int xbeg, int xend, int ybeg, int yend,
                                        const int* max_iter_remaining, double gain, const double* thres, double cspeedup,
                                        int processing_function_threads, int* iter_out,
                const DivergenceControl<double>& divergence, int* stop_out);

extern template void clean_cube_many_threads<float>(float* residual_cube, float* model_cube,
                                       const float* psf_cube, int domask,
                                       const bool* mask_cube,
                                       int nt, int nf, int np_img, int np_psf,
                                       int ny, int nx,
                                       int xbeg, int xend, int ybeg, int yend,
                                       const int* max_iter_remaining, float gain, const float* thres, float cspeedup,
                                       int processing_function_threads, int* iter_out,
                const DivergenceControl<float>& divergence, int* stop_out);

extern template void clean_cube_many_threads<double>(double* residual_cube, double* model_cube,
                                        const double* psf_cube, int domask,
                                        const bool* mask_cube,
                                        int nt, int nf, int np_img, int np_psf,
                                        int ny, int nx,
                                        int xbeg, int xend, int ybeg, int yend,
                                        const int* max_iter_remaining, double gain, const double* thres, double cspeedup,
                                        int processing_function_threads, int* iter_out,
                const DivergenceControl<double>& divergence, int* stop_out);

} // namespace hclean
