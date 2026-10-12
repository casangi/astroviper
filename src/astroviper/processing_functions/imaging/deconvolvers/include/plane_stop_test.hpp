#pragma once

// Stop tests of a model update, shared by the deconvolver kernels (Hogbom,
// Asp, MTMFS) so that they decide divergence identically.

#include <cmath>

namespace deconvolvers {

/**
 * Why a plane's model update ended. The values follow the cycle
 * controller's model update stop codes (imaging/utils/cycle_control.py).
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
 *     taken over the pixels whose peak the kernel tests against its threshold
 *     (for Hogbom the clean box inside the mask). It is an average over many
 *     pixels and so it jitters far less than the peak: measured in healthy
 *     Hogbom cleans, it rises by at most 0.09 x gain above its lowest value,
 *     where the peak rises by up to 0.44 x gain;
 *   - its peak exceeds peak_factor x (peak at the start of the model update):
 *     a runaway, stopped at once (hard limit);
 *   - its peak is not finite.
 *
 * max_iter_divergence < 0 disables the first two tests. The caller derives
 * the factors from the loop gain: rms_factor = 1 + gain / 10 and
 * peak_factor = 1 + gain. For Asp, whose least-squares components can raise
 * the peak of a healthy residual and its RMS by more than gain / 10, the
 * caller passes rms_factor = 1 + gain and an infinite peak_factor.
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
 * Per-plane state of the stop tests, shared by all deconvolver kernels so
 * that they decide identically. check(), or check_divergence() in a kernel
 * with a threshold test of its own, is called once per iteration with the
 * plane's current peak |residual| and the sum of the squares of its residual
 * (both over the pixels whose peak is tested against the threshold), before
 * the component is subtracted.
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
        return check_divergence(peak, sum_squares, control);
    }

    /// The divergence test alone, for a kernel that has let the plane carry
    /// on after its own threshold test: STOP_NONE or STOP_DIVERGED.
    int check_divergence(T peak, double sum_squares,
                         const DivergenceControl<T>& control) {
        if (!std::isfinite(peak)) {
            return STOP_DIVERGED;
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

}  // namespace deconvolvers
