#pragma once

#include <complex>
#include <cstdint>

namespace direct_fourier_transform {

/**
 * Exact direct Fourier transforms between visibilities and an image.
 *
 * Convention (the measurement equation of AstroVIPER's simulator,
 * processing_functions/simulation/visibility_kernel_cpp): a sample k of
 * baseline coordinates (u_k, v_k, w_k) in wavelengths (uvw in metres times
 * frequency / c) and a pixel of direction cosines (l, m), with
 * n = sqrt(1 - l^2 - m^2), have the phase, in turns,
 *
 *     phase_k(l, m) = u_k l + v_k m + w_k (n - 1)
 *
 * and the transforms are
 *
 *     prediction:  V_k = sum_pixels M(l, m) exp(+2 pi i phase_k(l, m)) / n
 *     imaging:     I(l, m) = n sum_k W_k V_k exp(-2 pi i phase_k(l, m))
 *
 * so that the prediction of a model holding a point source's apparent flux
 * in its pixel is the simulated visibility, and the image (divided by the
 * sum of the weights) of a simulated point source is its apparent flux at
 * its pixel. The w term is exact: no small field approximation is made.
 *
 * A phase is reduced to [-1/2, 1/2] turns exactly before its cosine and sine
 * are evaluated, so the transforms are accurate to the rounding of the phase
 * itself. Imaging evaluates exp(2 pi i phase) at a tile of pixels as the
 * product of the factor of the pixel's row (u l), of its column (v m) and of
 * the w term, which is exactly 1 for w = 0 and the Taylor polynomials of the
 * cosine and sine while |2 pi w (n - 1)| <= 2^-7 over the tile (truncation
 * errors below 4e-19); a larger w term is evaluated with the phase per pixel.
 *
 * Determinism: imaging gives every image plane's pixels, in tiles, to the
 * threads (each tile sums all samples in their stored order), prediction
 * gives every (time, baseline) row to one thread (each sample sums the
 * non-zero model pixels in their stored order). No output is shared between
 * threads, no atomic operation is used, and the result does not depend on
 * the number of threads.
 *
 * Memory beyond the inputs and outputs: the valid samples of imaging
 * (O(samples)), the non-zero model pixels of prediction (O(pixels)) and
 * per thread buffers of one pixel tile or block. No samples x pixels array is
 * made.
 */

/**
 * Accumulate the direct Fourier transform of weighted visibilities into an
 * image (the imaging transform).
 *
 * For every sample k = (i_time, i_baseline, i_chan, i_pol) whose u, v and w
 * are finite and whose weighted visibility weight * vis is neither NaN nor
 * zero (the samples the gridder grids, except that the gridder, which checks
 * u and v only, also grids a sample whose w alone is not finite):
 *
 *     image[a_time, a_chan, a_pol, i, j] +=
 *         n_ij weight_k vis_k exp(-2 pi i phase_k(l_i, m_j))
 *     normalization[a_time, a_chan, a_pol] += weight_k
 *
 * with a_time = time_map[i_time], a_chan = frequency_map[i_chan] and
 * a_pol = pol_map[i_pol]. Both outputs are accumulated in place (repeated
 * calls sum over measurement sets). A real ``ImageT`` (float or double)
 * receives the real part, a complex one the complex sum; the sums are made
 * in double precision and narrowed to ``ImageT`` once per pixel and call.
 *
 * @tparam ImageT  float, double, std::complex<float> or std::complex<double>.
 *
 * @param image          Output, shape (m_time, m_chan, m_pol, n_l, n_m).
 * @param normalization  Output, shape (m_time, m_chan, m_pol).
 * @param vis_data       Visibilities, shape (n_time, n_baseline, n_vis_chan, n_pol).
 * @param uvw            Baseline coordinates in metres, shape (n_time, n_baseline, 3).
 * @param frequency_coord Channel frequencies in Hz, shape (n_vis_chan,).
 * @param frequency_map  Image channel of every visibility channel, shape (n_vis_chan,).
 * @param time_map       Image time of every visibility time, shape (n_time,).
 * @param pol_map        Image polarization of every visibility polarization, shape (n_pol,).
 * @param weight         Imaging weights, shape (n_time, n_baseline, n_vis_chan, n_pol).
 * @param l              Direction cosine l of the image rows, shape (n_l,).
 * @param m              Direction cosine m of the image columns, shape (n_m,).
 * @param processing_function_threads  Worker threads; <= 0 is
 *                       std::thread::hardware_concurrency(), 1 runs serial.
 */
template <typename ImageT>
void dft_image(
    ImageT* image,
    double* normalization,
    const std::complex<double>* vis_data,
    const double* uvw,
    const double* frequency_coord,
    const std::int64_t* frequency_map,
    const std::int64_t* time_map,
    const std::int64_t* pol_map,
    const double* weight,
    const double* l,
    const double* m,
    std::int64_t m_time, std::int64_t m_chan, std::int64_t m_pol,
    std::int64_t n_l, std::int64_t n_m,
    std::int64_t n_time, std::int64_t n_baseline, std::int64_t n_vis_chan,
    std::int64_t n_pol,
    int processing_function_threads
);

/**
 * Accumulate the direct Fourier transform of the imaging weights into an
 * image (the point spread function before its normalization).
 *
 * As ::dft_image with vis_k = 1, for the samples whose u, v and w are finite
 * and whose weight is neither NaN nor zero (the samples the gridder grids
 * into the uv sampling grid, except one whose w alone is not finite); the
 * real part is accumulated:
 *
 *     image[a_time, a_chan, a_pol, i, j] +=
 *         n_ij weight_k cos(2 pi phase_k(l_i, m_j))
 *     normalization[a_time, a_chan, a_pol] += weight_k
 *
 * @tparam ImageT  float or double.
 */
template <typename ImageT>
void dft_image_uv_sampling(
    ImageT* image,
    double* normalization,
    const double* uvw,
    const double* frequency_coord,
    const std::int64_t* frequency_map,
    const std::int64_t* time_map,
    const std::int64_t* pol_map,
    const double* weight,
    const double* l,
    const double* m,
    std::int64_t m_time, std::int64_t m_chan, std::int64_t m_pol,
    std::int64_t n_l, std::int64_t n_m,
    std::int64_t n_time, std::int64_t n_baseline, std::int64_t n_vis_chan,
    std::int64_t n_pol,
    int processing_function_threads
);

/**
 * Predict visibilities from a model image by the direct Fourier transform.
 *
 * For every sample k = (i_time, i_baseline, i_chan, i_pol) whose u, v and w
 * are finite and whose current value in ``vis_data`` is not NaN (the
 * degridder's contract, except that the degridder, which checks u and v
 * only, also predicts a sample whose w alone is not finite), writes in place
 *
 *     vis_data[k] = sum_{i, j} model[a_time, a_chan, a_pol, i, j]
 *                   exp(+2 pi i phase_k(l_i, m_j)) / n_ij
 *
 * over the pixels where any polarization plane of the model is non-zero
 * (zero for a model plane without one). Other samples are left unchanged.
 * The sums are made in double precision and narrowed to ``VisT`` on write.
 *
 * @tparam ModelT  float, double, std::complex<float> or std::complex<double>.
 * @tparam VisT    std::complex<float> or std::complex<double>.
 */
template <typename ModelT, typename VisT>
void dft_predict(
    const ModelT* model,
    VisT* vis_data,
    const double* uvw,
    const double* frequency_coord,
    const std::int64_t* frequency_map,
    const std::int64_t* time_map,
    const std::int64_t* pol_map,
    const double* l,
    const double* m,
    std::int64_t m_time, std::int64_t m_chan, std::int64_t m_pol,
    std::int64_t n_l, std::int64_t n_m,
    std::int64_t n_time, std::int64_t n_baseline, std::int64_t n_vis_chan,
    std::int64_t n_pol,
    int processing_function_threads
);

/**
 * cos(2 pi t) and sin(2 pi t) of an angle t in turns, as the transforms
 * evaluate them (exposed for the tests).
 *
 * @param turns  Angles in turns, ``count`` values.
 * @param cos_values, sin_values  Outputs, ``count`` values each.
 */
void cos_sin_turns(const double* turns, double* cos_values, double* sin_values,
                   std::int64_t count);

}  // namespace direct_fourier_transform
