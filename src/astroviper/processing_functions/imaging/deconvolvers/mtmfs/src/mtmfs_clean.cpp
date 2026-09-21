// MTMFS CLEAN engine: port of CASA stdcleaner/StdMultiTermCleaner.cc (itself a
// faithful casacore-free port of synthesis::MultiTermMatrixCleaner setup +
// mtclean, verified against casacore) with the following changes:
//   * the residual and model stacks are caller-owned and updated in place
//     through raw pointers (no internal copies);
//   * the "initial model" snapshot casacore/StdMultiTermCleaner kept for the
//     end-of-cycle residual update is replaced by a per-cycle delta-model
//     accumulator, so the caller's model is never copied;
//   * the mask is read through a const pointer while the scale masks are
//     built, never stored;
//   * MultiTermMatrixCleaner::computeprincipalsolution() is ported;
//   * templated on the image value type (float / double).

#include "../include/mtmfs_clean.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../include/mt_fft.hpp"
#include "../include/mt_image_math.hpp"
#include "../include/mt_scales.hpp"

namespace mtmfs {

namespace {

int find_beam_patch(float max_scale_size, int nx, int ny, float psf_beam, float nbeams) {
    int psupport = static_cast<int>(
        std::sqrt(psf_beam * psf_beam + max_scale_size * max_scale_size) * nbeams);
    if (psupport < psf_beam * nbeams) psupport = static_cast<int>(psf_beam * nbeams);
    if (psupport > nx || psupport > ny) psupport = std::min(nx, ny);
    if (psupport % 2 != 0) psupport -= 1;
    return psupport;
}

inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

void verify_box(std::array<int, 2>& blc, std::array<int, 2>& trc, const std::array<int, 2>& shape) {
    for (int i = 0; i < 2; ++i) {
        blc[i] = clampi(blc[i], 0, shape[i] - 1);
        trc[i] = clampi(trc[i], 0, shape[i] - 1);
        if (trc[i] < blc[i]) trc[i] = blc[i];
    }
}

void make_boxes_same_size(std::array<int, 2>& blc1, std::array<int, 2>& trc1,
                          std::array<int, 2>& blc2, std::array<int, 2>& trc2) {
    for (int i = 0; i < 2; ++i) {
        const int shape1 = trc1[i] - blc1[i];
        const int shape2 = trc2[i] - blc2[i];
        if (shape1 == shape2) continue;
        const int min_length = std::min(shape1, shape2);
        const int inc1 = shape1 - min_length;
        const int inc2 = shape2 - min_length;
        blc1[i] += inc1 / 2;
        trc1[i] -= inc1 / 2 + (inc1 % 2 != 0 ? 1 : 0);
        blc2[i] += inc2 / 2;
        trc2[i] -= inc2 / 2 + (inc2 % 2 != 0 ? 1 : 0);
    }
}

// Cholesky inverse of a symmetric positive-definite n x n matrix.
bool invert_spd(const std::vector<double>& A, int n, std::vector<double>& Ainv) {
    std::vector<double> L(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double sum = A[static_cast<std::size_t>(i) * n + j];
            for (int k = 0; k < j; ++k)
                sum -= L[static_cast<std::size_t>(i) * n + k] * L[static_cast<std::size_t>(j) * n + k];
            if (i == j) {
                if (sum <= 0.0) return false;
                L[static_cast<std::size_t>(i) * n + i] = std::sqrt(sum);
            } else {
                L[static_cast<std::size_t>(i) * n + j] = sum / L[static_cast<std::size_t>(j) * n + j];
            }
        }
    Ainv.assign(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> y(n), x(n);
    for (int col = 0; col < n; ++col) {
        for (int i = 0; i < n; ++i) {
            double s = (i == col) ? 1.0 : 0.0;
            for (int k = 0; k < i; ++k) s -= L[static_cast<std::size_t>(i) * n + k] * y[k];
            y[i] = s / L[static_cast<std::size_t>(i) * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = y[i];
            for (int k = i + 1; k < n; ++k) s -= L[static_cast<std::size_t>(k) * n + i] * x[k];
            x[i] = s / L[static_cast<std::size_t>(i) * n + i];
        }
        for (int i = 0; i < n; ++i) Ainv[static_cast<std::size_t>(i) * n + col] = x[i];
    }
    return true;
}

// Extract a sup x sup patch of `full` (nx wide) centred at (cx, cy).
template <typename T>
void extract_patch(const std::vector<T>& full, int nx, int cx, int cy, int sup, std::vector<T>& out) {
    out.assign(static_cast<std::size_t>(sup) * sup, static_cast<T>(0));
    for (int j = 0; j < sup; ++j)
        for (int i = 0; i < sup; ++i)
            out[static_cast<std::size_t>(j) * sup + i] =
                full[static_cast<std::size_t>(cy - sup / 2 + j) * nx + (cx - sup / 2 + i)];
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction (SDAlgorithmMSMFS constructor + initializeDeconvolver semantics)
// ---------------------------------------------------------------------------

template <typename T>
MultiTermCleaner<T>::MultiTermCleaner(int nterms, const std::vector<float>& scales, int nx, int ny,
                                      float small_scale_bias)
    : nx_(nx), ny_(ny), ncx_(rfft_width(nx)), ntaylor_(nterms), psf_ntaylor_(2 * nterms - 1) {
    if (nterms < 1) throw std::invalid_argument("nterms must be >= 1");
    if (nx < 2 || ny < 2) throw std::invalid_argument("image dimensions must be >= 2 pixels");

    // SDAlgorithmMSMFS: an empty scale list means a single point-source scale.
    std::vector<float> requested = scales;
    if (requested.empty()) requested.push_back(0.0f);

    // MultiTermMatrixCleaner::verifyScaleSizes(): scales too big for the image
    // are ignored (CASA logs a warning); the effective list is what scales()
    // reports.
    for (float s : requested) {
        if (!(s >= 0.0f)) throw std::invalid_argument("scale sizes must be non-negative");
        if (s > nx / 2 || s > ny / 2) continue;
        scale_sizes_.push_back(s);
    }
    if (scale_sizes_.empty())
        throw std::invalid_argument("all scale sizes are larger than half the image size");

    // SDAlgorithmMSMFS::initializeDeconvolver(): acceptable smallscalebias
    // values are [-1, 1]; out-of-range values are clamped (CASA warns).
    small_scale_bias_ = std::max(-1.0f, std::min(1.0f, small_scale_bias));
}

// ---------------------------------------------------------------------------
// Setup: PSF transforms, scale functions, Hessian
// ---------------------------------------------------------------------------

template <typename T>
int MultiTermCleaner<T>::set_psf(const T* psf) {
    has_psf_ = false;
    psf_ft_.assign(psf_ntaylor_, std::vector<complex_t>(nspec(), complex_t(0, 0)));
    for (int order = 0; order < psf_ntaylor_; ++order) {
        const T* p = psf + static_cast<std::size_t>(order) * nimg();
        forward_r2c<T>(p, psf_ft_[static_cast<std::size_t>(order)].data(), nx_, ny_);
    }

    // PSF peak position from Taylor term 0, searched in a central patch
    // (MultiTermMatrixCleaner::setpsf).
    {
        const T* p0 = psf;
        const int supp = find_beam_patch(0.0f, nx_, ny_, 4.0f, 20.0f);
        const int blc0 = (nx_ > supp) ? nx_ / 2 - supp / 2 : 0;
        const int blc1 = (ny_ > supp) ? ny_ / 2 - supp / 2 : 0;
        const int trc0 = (nx_ > supp) ? nx_ / 2 + supp / 2 : nx_ - 1;
        const int trc1 = (ny_ > supp) ? ny_ / 2 + supp / 2 : ny_ - 1;
        T max_val = static_cast<T>(0);
        psf_peak_x_ = 0;
        psf_peak_y_ = 0;
        for (int j = blc1; j < trc1; ++j)
            for (int i = blc0; i < trc0; ++i) {
                const T a = std::abs(p0[static_cast<std::size_t>(j) * nx_ + i]);
                if (a > max_val) {
                    max_val = a;
                    psf_peak_x_ = i;
                    psf_peak_y_ = j;
                }
            }
    }

    setup_scale_functions();

    // MultiTermMatrixCleaner::initialise(): the adaptive-gain history persists
    // across minor cycles and is reset only when the cleaner is (re)initialised.
    total_iters_ = 0;
    prev_max_ = static_cast<T>(1e10);
    min_max_ = static_cast<T>(1e10);

    const int rc = compute_hessian();
    has_psf_ = (rc == 0);
    return rc;
}

template <typename T>
void MultiTermCleaner<T>::setup_scale_functions() {
    const int ns = nscales();

    psf_support_ = find_beam_patch(scale_sizes_[static_cast<std::size_t>(ns - 1)], nx_, ny_, 4.0f, 20.0f);

    scale_bias_.assign(ns, static_cast<T>(1));
    if (ns > 1)
        for (int s = 0; s < ns; ++s)
            scale_bias_[s] = static_cast<T>(1 - small_scale_bias_ * scale_sizes_[s] /
                                                    scale_sizes_[static_cast<std::size_t>(ns - 1)]);

    scale_ft_.assign(ns, std::vector<complex_t>(nspec()));
    vec_scales_.assign(ns, std::vector<T>());
    std::vector<T> scale_img(nimg());
    for (int s = 0; s < ns; ++s) {
        make_scale<T>(scale_img.data(), nx_, ny_, scale_sizes_[static_cast<std::size_t>(s)]);
        forward_r2c<T>(scale_img.data(), scale_ft_[static_cast<std::size_t>(s)].data(), nx_, ny_);
        extract_patch(scale_img, nx_, nx_ / 2, ny_ / 2, psf_support_, vec_scales_[static_cast<std::size_t>(s)]);
    }
}

template <typename T>
int MultiTermCleaner<T>::cross_index4(int t1, int t2, int s1, int s2) const {
    const int ns = nscales();
    const int S1 = std::max(s1, s2), S2 = std::min(s1, s2);
    const int T1 = std::max(t1, t2), T2 = std::min(t1, t2);
    const int totscale = ns * (ns + 1) / 2;
    return ((T1 * (T1 + 1) / 2) + T2) * totscale + ((S1 * (S1 + 1) / 2) + S2);
}

template <typename T>
int MultiTermCleaner<T>::compute_hessian() {
    const int ns = nscales();
    const int nt = ntaylor_;
    const int sup = psf_support_;
    const std::size_t peak_local = static_cast<std::size_t>(sup / 2) * sup + (sup / 2);

    const int ntotal4d = (ns * (ns + 1) / 2) * (nt * (nt + 1) / 2);
    cube_a_.assign(static_cast<std::size_t>(ntotal4d), std::vector<T>());
    mat_a_.assign(ns, std::vector<double>(static_cast<std::size_t>(nt) * nt, 0.0));
    inv_mat_a_.assign(ns, std::vector<double>(static_cast<std::size_t>(nt) * nt, 0.0));

    std::vector<complex_t> work(nspec());
    std::vector<T> full(nimg());
    for (int t1 = 0; t1 < nt; ++t1)
        for (int t2 = 0; t2 <= t1; ++t2)
            for (int s1 = 0; s1 < ns; ++s1)
                for (int s2 = 0; s2 <= s1; ++s2) {
                    const std::vector<complex_t>& pf = psf_ft_[static_cast<std::size_t>(t1 + t2)];
                    const std::vector<complex_t>& x1 = scale_ft_[static_cast<std::size_t>(s1)];
                    const std::vector<complex_t>& x2 = scale_ft_[static_cast<std::size_t>(s2)];
                    for (std::size_t k = 0; k < nspec(); ++k) work[k] = pf[k] * x1[k] * x2[k];
                    backward_c2r<T>(work.data(), full.data(), nx_, ny_);  // no flip
                    extract_patch(full, nx_, psf_peak_x_, psf_peak_y_, sup,
                                  cube_a_[static_cast<std::size_t>(cross_index4(t1, t2, s1, s2))]);
                }

    for (int scale = 0; scale < ns; ++scale) {
        for (int t1 = 0; t1 < nt; ++t1)
            for (int t2 = 0; t2 < nt; ++t2)
                mat_a_[static_cast<std::size_t>(scale)][static_cast<std::size_t>(t1) * nt + t2] =
                    static_cast<double>(
                        cube_a_[static_cast<std::size_t>(cross_index4(t1, t2, scale, scale))][peak_local]);
        if (!invert_spd(mat_a_[static_cast<std::size_t>(scale)], nt, inv_mat_a_[static_cast<std::size_t>(scale)]))
            return -2;
    }
    return 0;
}

template <typename T>
const std::vector<double>& MultiTermCleaner<T>::hessian(int scale) const {
    if (!has_psf_) throw std::logic_error("set_psf() must be called before hessian()");
    if (scale < 0 || scale >= nscales()) throw std::out_of_range("scale index out of range");
    return mat_a_[static_cast<std::size_t>(scale)];
}

template <typename T>
const std::vector<double>& MultiTermCleaner<T>::inverse_hessian(int scale) const {
    if (!has_psf_) throw std::logic_error("set_psf() must be called before inverse_hessian()");
    if (scale < 0 || scale >= nscales()) throw std::out_of_range("scale index out of range");
    return inv_mat_a_[static_cast<std::size_t>(scale)];
}

// ---------------------------------------------------------------------------
// Per-cycle setup: masks and RHS
// ---------------------------------------------------------------------------

template <typename T>
void MultiTermCleaner<T>::setup_user_mask() {
    const int ns = nscales();
    vec_scale_masks_.assign(ns, std::vector<T>(nimg(), static_cast<T>(1)));

    if (mask_ != nullptr) {
        std::vector<complex_t> mask_ft(nspec()), work(nspec());
        forward_r2c<T>(mask_, mask_ft.data(), nx_, ny_);
        for (int s = 0; s < ns; ++s) {
            std::vector<T>& sm = vec_scale_masks_[static_cast<std::size_t>(s)];
            for (std::size_t k = 0; k < nspec(); ++k)
                work[k] = mask_ft[k] * scale_ft_[static_cast<std::size_t>(s)][k];
            backward_c2r<T>(work.data(), sm.data(), nx_, ny_);
            flip_quadrants<T>(sm.data(), nx_, ny_);
            // MultiTermMatrixCleaner: a positive threshold binarises the
            // scale-convolved mask at 0.1 (the threshold value itself is not
            // used as the cut), reproduced as-is.
            if (mask_threshold_ > static_cast<T>(0))
                for (std::size_t k = 0; k < nimg(); ++k)
                    sm[k] = (sm[k] > static_cast<T>(0.1)) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }

    // Scale-dependent border (scale >= 1 only).
    for (int s = 1; s < ns; ++s) {
        std::vector<T>& sm = vec_scale_masks_[static_cast<std::size_t>(s)];
        const int border = static_cast<int>(scale_sizes_[static_cast<std::size_t>(s)] * 1.5f);
        auto zero_region = [&](int bx, int by, int tx, int ty) {
            bx = clampi(bx, 0, nx_ - 1);
            by = clampi(by, 0, ny_ - 1);
            tx = clampi(tx, 0, nx_ - 1);
            ty = clampi(ty, 0, ny_ - 1);
            for (int j = by; j <= ty; ++j)
                for (int i = bx; i <= tx; ++i) sm[static_cast<std::size_t>(j) * nx_ + i] = static_cast<T>(0);
        };
        zero_region(0, 0, nx_ - 1, border);
        zero_region(0, ny_ - border - 1, nx_ - 1, ny_ - 1);
        zero_region(0, border, border, ny_ - border - 1);
        zero_region(nx_ - border - 1, border, nx_ - 1, ny_ - border - 1);
    }
}

template <typename T>
void MultiTermCleaner<T>::compute_rhs() {
    const int ns = nscales();
    const int nt = ntaylor_;
    mat_r_.assign(static_cast<std::size_t>(nt) * ns, std::vector<T>(nimg(), static_cast<T>(0)));
    mat_coeffs_.assign(static_cast<std::size_t>(nt) * ns, std::vector<T>(nimg(), static_cast<T>(0)));
    vec_work_.assign(ns, std::vector<T>(nimg(), static_cast<T>(0)));

    std::vector<complex_t> dirty_ft(nspec()), work(nspec());
    for (int t = 0; t < nt; ++t) {
        forward_r2c<T>(residual_term(t), dirty_ft.data(), nx_, ny_);
        for (int s = 0; s < ns; ++s) {
            for (std::size_t k = 0; k < nspec(); ++k)
                work[k] = dirty_ft[k] * scale_ft_[static_cast<std::size_t>(s)][k];
            std::vector<T>& r = mat_r_[static_cast<std::size_t>(ind2(t, s))];
            backward_c2r<T>(work.data(), r.data(), nx_, ny_);
            flip_quadrants<T>(r.data(), nx_, ny_);
        }
    }
}

// ---------------------------------------------------------------------------
// Minor-cycle steps
// ---------------------------------------------------------------------------

template <typename T>
void MultiTermCleaner<T>::solve_matrix_eqn(int scale, const std::array<int, 2>& blc,
                                           const std::array<int, 2>& trc) {
    const int nt = ntaylor_;
    for (int t1 = 0; t1 < nt; ++t1) {
        std::vector<T>& coeffs = mat_coeffs_[static_cast<std::size_t>(ind2(t1, scale))];
        for (int iy = blc[1]; iy <= trc[1]; ++iy)
            for (int ix = blc[0]; ix <= trc[0]; ++ix)
                coeffs[static_cast<std::size_t>(iy) * nx_ + ix] = static_cast<T>(0);
        for (int t2 = 0; t2 < nt; ++t2) {
            const T inv = static_cast<T>(
                inv_mat_a_[static_cast<std::size_t>(scale)][static_cast<std::size_t>(t1) * nt + t2]);
            const std::vector<T>& rhs = mat_r_[static_cast<std::size_t>(ind2(t2, scale))];
            for (int iy = blc[1]; iy <= trc[1]; ++iy)
                for (int ix = blc[0]; ix <= trc[0]; ++ix) {
                    const std::size_t k = static_cast<std::size_t>(iy) * nx_ + ix;
                    coeffs[k] += inv * rhs[k];
                }
        }
    }
}

template <typename T>
void MultiTermCleaner<T>::choose_component(int scale, const std::array<int, 2>& blc,
                                           const std::array<int, 2>& trc) {
    const int nt = ntaylor_;
    std::vector<T>& work = vec_work_[static_cast<std::size_t>(scale)];
    for (int iy = blc[1]; iy <= trc[1]; ++iy)
        for (int ix = blc[0]; ix <= trc[0]; ++ix)
            work[static_cast<std::size_t>(iy) * nx_ + ix] = static_cast<T>(0);
    for (int t1 = 0; t1 < nt; ++t1) {
        const std::vector<T>& coeffs = mat_coeffs_[static_cast<std::size_t>(ind2(t1, scale))];
        const std::vector<T>& resid = mat_r_[static_cast<std::size_t>(ind2(t1, scale))];
        for (int iy = blc[1]; iy <= trc[1]; ++iy)
            for (int ix = blc[0]; ix <= trc[0]; ++ix) {
                const std::size_t k = static_cast<std::size_t>(iy) * nx_ + ix;
                work[k] += coeffs[k] * resid[k];
            }
    }
    const PeakResult<T> pr =
        find_max_abs_mask<T>(work.data(), vec_scale_masks_[static_cast<std::size_t>(scale)].data(), nx_, ny_);
    max_scale_val_[static_cast<std::size_t>(scale)] = pr.value;
    max_scale_pos_[static_cast<std::size_t>(scale)] = {pr.ix, pr.iy};
}

template <typename T>
void MultiTermCleaner<T>::build_image_patches() {
    const std::array<int, 2> shape{nx_, ny_};
    const std::array<int, 2> psf_shape{psf_support_, psf_support_};
    const std::array<int, 2> psf_peak_local{psf_support_ / 2, psf_support_ / 2};
    blc_ = {global_max_pos_[0] - psf_support_ / 2, global_max_pos_[1] - psf_support_ / 2};
    trc_ = {global_max_pos_[0] + psf_support_ / 2 - 1, global_max_pos_[1] + psf_support_ / 2 - 1};
    verify_box(blc_, trc_, shape);
    blc_psf_ = {blc_[0] + psf_peak_local[0] - global_max_pos_[0],
                blc_[1] + psf_peak_local[1] - global_max_pos_[1]};
    trc_psf_ = {trc_[0] + psf_peak_local[0] - global_max_pos_[0],
                trc_[1] + psf_peak_local[1] - global_max_pos_[1]};
    verify_box(blc_psf_, trc_psf_, psf_shape);
    make_boxes_same_size(blc_, trc_, blc_psf_, trc_psf_);
}

template <typename T>
void MultiTermCleaner<T>::update_model_and_rhs(T loopgain) {
    const int ns = nscales();
    const int nt = ntaylor_;
    const int sup = psf_support_;
    const std::size_t gpk = static_cast<std::size_t>(global_max_pos_[1]) * nx_ + global_max_pos_[0];
    const std::vector<T>& scale_sub = vec_scales_[static_cast<std::size_t>(max_scale_index_)];
    const int bw = trc_[0] - blc_[0];
    const int bh = trc_[1] - blc_[1];

    // Update the caller's model images in place, and the per-cycle delta.
    for (int t = 0; t < nt; ++t) {
        const T coeff = mat_coeffs_[static_cast<std::size_t>(ind2(t, max_scale_index_))][gpk];
        T* mdl = model_term(t);
        std::vector<T>& dlt = delta_model_[static_cast<std::size_t>(t)];
        for (int dy = 0; dy <= bh; ++dy)
            for (int dx = 0; dx <= bw; ++dx) {
                const std::size_t km = static_cast<std::size_t>(blc_[1] + dy) * nx_ + (blc_[0] + dx);
                const std::size_t kp = static_cast<std::size_t>(blc_psf_[1] + dy) * sup + (blc_psf_[0] + dx);
                const T add = scale_sub[kp] * loopgain * coeff;
                mdl[km] += add;
                dlt[km] += add;
            }
    }

    std::vector<T> coeffs(nt);
    for (int t = 0; t < nt; ++t)
        coeffs[t] = mat_coeffs_[static_cast<std::size_t>(ind2(t, max_scale_index_))][gpk];

    // Update convolved residuals (RHS) for every scale.
    for (int scale = 0; scale < ns; ++scale)
        for (int t1 = 0; t1 < nt; ++t1) {
            std::vector<T>& resid = mat_r_[static_cast<std::size_t>(ind2(t1, scale))];
            for (int t2 = 0; t2 < nt; ++t2) {
                const std::vector<T>& smooth =
                    cube_a_[static_cast<std::size_t>(cross_index4(t1, t2, scale, max_scale_index_))];
                const T g = loopgain * coeffs[t2];
                for (int dy = 0; dy <= bh; ++dy)
                    for (int dx = 0; dx <= bw; ++dx) {
                        const std::size_t kr = static_cast<std::size_t>(blc_[1] + dy) * nx_ + (blc_[0] + dx);
                        const std::size_t kp = static_cast<std::size_t>(blc_psf_[1] + dy) * sup + (blc_psf_[0] + dx);
                        resid[kr] -= smooth[kp] * g;
                    }
            }
        }
}

template <typename T>
int MultiTermCleaner<T>::check_convergence(T& fluxlimit, T& loopgain) {
    const PeakResult<T> pr = find_max_abs_mask<T>(mat_r_[static_cast<std::size_t>(ind2(0, 0))].data(),
                                                  vec_scale_masks_[0].data(), nx_, ny_);
    const T norma = static_cast<T>(1.0 / mat_a_[0][0]);
    const T rmaxval = std::abs(pr.value * norma);

    int convergedflag = 0;
    if (std::abs(rmaxval) < std::max(user_threshold_, fluxlimit)) convergedflag = 1;

    if (itercount_ > 1 && input_gain_ <= static_cast<T>(0)) {
        loopgain = (global_max_val_ < prev_max_) ? loopgain * static_cast<T>(1.5)
                                                 : loopgain / static_cast<T>(1.5);
        loopgain = std::min(static_cast<T>(1) - stop_fraction_, loopgain);
        loopgain = std::min(static_cast<T>(0.6), loopgain);
        if (loopgain < static_cast<T>(0.01)) convergedflag = -1;
        if (std::abs((min_max_ - global_max_val_) / min_max_) > static_cast<T>(2)) convergedflag = -1;
    }
    prev_max_ = global_max_val_;
    min_max_ = std::min(min_max_, std::abs(global_max_val_));

    if (convergedflag == 0 && fluxlimit == static_cast<T>(-1)) fluxlimit = rmaxval * stop_fraction_;
    return convergedflag;
}

// ---------------------------------------------------------------------------
// The minor cycle (MultiTermMatrixCleaner::mtclean)
// ---------------------------------------------------------------------------

template <typename T>
int MultiTermCleaner<T>::clean(T* residual, T* model, const T* mask, T mask_threshold, int max_iter,
                               T stop_fraction, T gain, T threshold) {
    if (!has_psf_) throw std::logic_error("set_psf() must be called before clean()");

    residual_ = residual;
    model_ = model;
    mask_ = mask;
    mask_threshold_ = mask_threshold;

    max_iter_ = max_iter;
    stop_fraction_ = stop_fraction;
    input_gain_ = gain;
    user_threshold_ = threshold;
    const int ns = nscales();
    const int nt = ntaylor_;

    max_scale_val_.assign(ns, static_cast<T>(0));
    max_scale_pos_.assign(ns, {0, 0});

    T loopgain = (gain > static_cast<T>(0)) ? gain : static_cast<T>(0.5);
    T fluxlimit = static_cast<T>(-1);
    itercount_ = 0;
    int iterdone = 0;

    // Per-cycle delta model (replaces the casacore initial-model snapshot).
    delta_model_.assign(nt, std::vector<T>(nimg(), static_cast<T>(0)));

    setup_user_mask();
    compute_rhs();

    int convergedflag = check_convergence(fluxlimit, loopgain);
    if (convergedflag == 1) {
        residual_ = nullptr;
        model_ = nullptr;
        mask_ = nullptr;
        return 0;
    }

    for (itercount_ = 0; itercount_ < max_iter_; ++itercount_) {
        global_max_val_ = static_cast<T>(-1e10);
        if (itercount_ == 0) {
            blc_ = {0, 0};
            trc_ = {nx_ - 1, ny_ - 1};
        }
        for (int scale = 0; scale < ns; ++scale) {
            solve_matrix_eqn(scale, blc_, trc_);
            choose_component(scale, blc_, trc_);
        }
        for (int scale = 0; scale < ns; ++scale)
            if (max_scale_val_[static_cast<std::size_t>(scale)] * scale_bias_[static_cast<std::size_t>(scale)] >
                global_max_val_) {
                global_max_val_ = max_scale_val_[static_cast<std::size_t>(scale)] *
                                  scale_bias_[static_cast<std::size_t>(scale)];
                global_max_pos_ = max_scale_pos_[static_cast<std::size_t>(scale)];
                max_scale_index_ = scale;
            }
        build_image_patches();
        update_model_and_rhs(loopgain);
        total_iters_++;
        iterdone++;
        convergedflag = check_convergence(fluxlimit, loopgain);
        if (convergedflag) break;
    }

    // Update the caller's image-domain residuals in place to account for the
    // new model components: residual_t1 -= sum_t2 psf_{t1+t2} (*) delta_t2.
    std::vector<T> smooth_mod(nimg());
    std::vector<complex_t> model_ft(nspec()), work(nspec());
    for (int t2 = 0; t2 < nt; ++t2) {
        forward_r2c<T>(delta_model_[static_cast<std::size_t>(t2)].data(), model_ft.data(), nx_, ny_);
        for (int t1 = 0; t1 < nt; ++t1) {
            const std::vector<complex_t>& pf = psf_ft_[static_cast<std::size_t>(t1 + t2)];
            for (std::size_t k = 0; k < nspec(); ++k) work[k] = pf[k] * model_ft[k];
            backward_c2r<T>(work.data(), smooth_mod.data(), nx_, ny_);
            flip_quadrants<T>(smooth_mod.data(), nx_, ny_);
            T* d = residual_term(t1);
            for (std::size_t k = 0; k < nimg(); ++k) d[k] -= smooth_mod[k];
        }
    }

    residual_ = nullptr;
    model_ = nullptr;
    mask_ = nullptr;
    return iterdone;
}

// ---------------------------------------------------------------------------
// Principal solution (MultiTermMatrixCleaner::computeprincipalsolution)
// ---------------------------------------------------------------------------

template <typename T>
void MultiTermCleaner<T>::compute_principal_solution(T* residual) const {
    if (!has_psf_) throw std::logic_error("set_psf() must be called before compute_principal_solution()");
    const int nt = ntaylor_;
    const std::vector<double>& inv0 = inv_mat_a_[0];
    std::vector<T> in(nt);
    const std::size_t n = nimg();
    for (std::size_t k = 0; k < n; ++k) {
        for (int t = 0; t < nt; ++t) in[t] = residual[static_cast<std::size_t>(t) * n + k];
        for (int t1 = 0; t1 < nt; ++t1) {
            T acc = static_cast<T>(0);
            for (int t2 = 0; t2 < nt; ++t2)
                acc += static_cast<T>(inv0[static_cast<std::size_t>(t1) * nt + t2]) * in[t2];
            residual[static_cast<std::size_t>(t1) * n + k] = acc;
        }
    }
}

template class MultiTermCleaner<float>;
template class MultiTermCleaner<double>;

}  // namespace mtmfs
