// Stateless MTMFS CLEAN: one-shot free functions. Working state lives on the
// stack for the duration of the call and is discarded on return. Port of
// CASA stdcleaner/StdMultiTermCleaner.cc (casacore-free MultiTermMatrixCleaner)
// with caller-owned residual/model (no copies) and a per-cycle delta model
// instead of the casacore initial-model snapshot.

#include "../include/mtmfs_clean.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "../include/mt_fft.hpp"
#include "../include/mt_image_math.hpp"
#include "../include/mt_scales.hpp"

namespace mtmfs {

namespace {

int find_beam_patch(float max_scale_size, int nx, int ny, float psf_beam, float nbeams) {
    int psupport = static_cast<int>(std::sqrt(psf_beam * psf_beam + max_scale_size * max_scale_size) * nbeams);
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

void make_boxes_same_size(std::array<int, 2>& blc1, std::array<int, 2>& trc1, std::array<int, 2>& blc2,
                          std::array<int, 2>& trc2) {
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

template <typename T>
void extract_patch(const T* full, int nx, int ny, int cx, int cy, int sup, std::vector<T>& out) {
    out.assign(static_cast<std::size_t>(sup) * sup, static_cast<T>(0));
    for (int j = 0; j < sup; ++j)
        for (int i = 0; i < sup; ++i) {
            const int fy = cy - sup / 2 + j;
            const int fx = cx - sup / 2 + i;
            if (fy < 0 || fy >= ny || fx < 0 || fx >= nx) continue;
            out[static_cast<std::size_t>(j) * sup + i] = full[static_cast<std::size_t>(fy) * nx + fx];
        }
}

void check_geometry(int nterms, int nx, int ny) {
    if (nterms < 1 || nterms > kMaxTaylorTerms)
        throw std::invalid_argument("nterms must be in [1, " + std::to_string(kMaxTaylorTerms) + "]");
    if (nx < 2 || ny < 2) throw std::invalid_argument("image dimensions must be >= 2 pixels");
}

// Per-call working state. Constructed, used, destroyed inside one free function.
template <typename T>
struct Work {
    using complex_t = std::complex<T>;

    int nx = 0, ny = 0, ncx = 0, nterms = 0, npsf = 0, nscales = 0;
    int psf_peak_x = 0, psf_peak_y = 0, psf_support = 0;
    float small_scale_bias = 0.0f;

    std::vector<float> scales;
    std::vector<T> scale_bias;
    std::vector<std::vector<complex_t>> psf_ft;
    std::vector<std::vector<complex_t>> scale_ft;
    std::vector<std::vector<T>> vec_scales;
    std::vector<std::vector<double>> mat_a;
    std::vector<std::vector<double>> inv_mat_a;
    std::vector<std::vector<T>> cube_a;

    std::size_t nimg() const { return static_cast<std::size_t>(nx) * ny; }
    std::size_t nspec() const { return static_cast<std::size_t>(ny) * ncx; }
    int ind2(int taylor, int scale) const { return taylor * nscales + scale; }
    int cross_index4(int t1, int t2, int s1, int s2) const {
        const int S1 = std::max(s1, s2), S2 = std::min(s1, s2);
        const int T1 = std::max(t1, t2), T2 = std::min(t1, t2);
        const int totscale = nscales * (nscales + 1) / 2;
        return ((T1 * (T1 + 1) / 2) + T2) * totscale + ((S1 * (S1 + 1) / 2) + S2);
    }

    void init(int nt, int inx, int iny, const std::vector<float>& user_scales, float bias) {
        check_geometry(nt, inx, iny);
        nterms = nt;
        npsf = 2 * nt - 1;
        nx = inx;
        ny = iny;
        ncx = rfft_width(nx);
        scales = effective_scales(user_scales, nx, ny);
        nscales = static_cast<int>(scales.size());
        small_scale_bias = clamp_small_scale_bias(bias);
    }

    int setup_from_psf(const T* psf) {
        psf_ft.assign(npsf, std::vector<complex_t>(nspec(), complex_t(0, 0)));
        for (int order = 0; order < npsf; ++order)
            forward_r2c<T>(psf + static_cast<std::size_t>(order) * nimg(), psf_ft[static_cast<std::size_t>(order)].data(),
                           nx, ny);

        {
            const T* p0 = psf;
            const int supp = find_beam_patch(0.0f, nx, ny, 4.0f, 20.0f);
            const int blc0 = (nx > supp) ? nx / 2 - supp / 2 : 0;
            const int blc1 = (ny > supp) ? ny / 2 - supp / 2 : 0;
            const int trc0 = (nx > supp) ? nx / 2 + supp / 2 : nx - 1;
            const int trc1 = (ny > supp) ? ny / 2 + supp / 2 : ny - 1;
            T max_val = static_cast<T>(0);
            psf_peak_x = nx / 2;
            psf_peak_y = ny / 2;
            for (int j = blc1; j < trc1; ++j)
                for (int i = blc0; i < trc0; ++i) {
                    const T a = std::abs(p0[static_cast<std::size_t>(j) * nx + i]);
                    if (a > max_val) {
                        max_val = a;
                        psf_peak_x = i;
                        psf_peak_y = j;
                    }
                }
        }

        psf_support = find_beam_patch(scales[static_cast<std::size_t>(nscales - 1)], nx, ny, 4.0f, 20.0f);
        scale_bias.assign(nscales, static_cast<T>(1));
        if (nscales > 1)
            for (int s = 0; s < nscales; ++s)
                scale_bias[s] = static_cast<T>(1 - small_scale_bias * scales[s] / scales[static_cast<std::size_t>(nscales - 1)]);

        scale_ft.assign(nscales, std::vector<complex_t>(nspec()));
        vec_scales.assign(nscales, std::vector<T>());
        std::vector<T> scale_img(nimg());
        for (int s = 0; s < nscales; ++s) {
            make_scale<T>(scale_img.data(), nx, ny, scales[static_cast<std::size_t>(s)]);
            forward_r2c<T>(scale_img.data(), scale_ft[static_cast<std::size_t>(s)].data(), nx, ny);
            extract_patch(scale_img.data(), nx, ny, nx / 2, ny / 2, psf_support, vec_scales[static_cast<std::size_t>(s)]);
        }

        const int nt = nterms;
        const int ns = nscales;
        const int sup = psf_support;
        const std::size_t peak_local = static_cast<std::size_t>(sup / 2) * sup + (sup / 2);
        const int ntotal4d = (ns * (ns + 1) / 2) * (nt * (nt + 1) / 2);
        cube_a.assign(static_cast<std::size_t>(ntotal4d), std::vector<T>());
        mat_a.assign(ns, std::vector<double>(static_cast<std::size_t>(nt) * nt, 0.0));
        inv_mat_a.assign(ns, std::vector<double>(static_cast<std::size_t>(nt) * nt, 0.0));

        std::vector<complex_t> work(nspec());
        std::vector<T> full(nimg());
        for (int t1 = 0; t1 < nt; ++t1)
            for (int t2 = 0; t2 <= t1; ++t2)
                for (int s1 = 0; s1 < ns; ++s1)
                    for (int s2 = 0; s2 <= s1; ++s2) {
                        const std::vector<complex_t>& pf = psf_ft[static_cast<std::size_t>(t1 + t2)];
                        const std::vector<complex_t>& x1 = scale_ft[static_cast<std::size_t>(s1)];
                        const std::vector<complex_t>& x2 = scale_ft[static_cast<std::size_t>(s2)];
                        for (std::size_t k = 0; k < nspec(); ++k) work[k] = pf[k] * x1[k] * x2[k];
                        backward_c2r<T>(work.data(), full.data(), nx, ny);
                        extract_patch(full.data(), nx, ny, psf_peak_x, psf_peak_y, sup,
                                      cube_a[static_cast<std::size_t>(cross_index4(t1, t2, s1, s2))]);
                    }

        for (int scale = 0; scale < ns; ++scale) {
            for (int t1 = 0; t1 < nt; ++t1)
                for (int t2 = 0; t2 < nt; ++t2)
                    mat_a[static_cast<std::size_t>(scale)][static_cast<std::size_t>(t1) * nt + t2] = static_cast<double>(
                        cube_a[static_cast<std::size_t>(cross_index4(t1, t2, scale, scale))][peak_local]);
            if (!invert_spd(mat_a[static_cast<std::size_t>(scale)], nt, inv_mat_a[static_cast<std::size_t>(scale)]))
                return kSingularHessian;
        }
        return 0;
    }

    void pack_hessians(std::vector<double>& hessian, std::vector<double>& inverse) const {
        const std::size_t block = static_cast<std::size_t>(nterms) * nterms;
        hessian.resize(static_cast<std::size_t>(nscales) * block);
        inverse.resize(static_cast<std::size_t>(nscales) * block);
        for (int s = 0; s < nscales; ++s) {
            std::copy(mat_a[static_cast<std::size_t>(s)].begin(), mat_a[static_cast<std::size_t>(s)].end(),
                      hessian.begin() + static_cast<std::size_t>(s) * block);
            std::copy(inv_mat_a[static_cast<std::size_t>(s)].begin(), inv_mat_a[static_cast<std::size_t>(s)].end(),
                      inverse.begin() + static_cast<std::size_t>(s) * block);
        }
    }
};

}  // namespace

std::vector<float> effective_scales(const std::vector<float>& scales, int nx, int ny) {
    std::vector<float> requested = scales;
    if (requested.empty()) requested.push_back(0.0f);
    std::vector<float> kept;
    kept.reserve(requested.size());
    for (float s : requested) {
        if (!std::isfinite(s) || s < 0.0f) throw std::invalid_argument("scale sizes must be finite and non-negative");
        if (s > nx / 2 || s > ny / 2) continue;
        kept.push_back(s);
    }
    if (kept.empty()) throw std::invalid_argument("all scale sizes are larger than half the image size");
    std::sort(kept.begin(), kept.end());
    kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
    return kept;
}

float clamp_small_scale_bias(float bias) {
    if (!std::isfinite(bias)) throw std::invalid_argument("small_scale_bias must be finite");
    return std::max(-1.0f, std::min(1.0f, bias));
}

template <typename T>
int taylor_hessian(const T* psf, int nterms, int nx, int ny, std::vector<float>& scales, float small_scale_bias,
                   std::vector<double>& hessian, std::vector<double>& inverse_hessian, int& psf_support) {
    Work<T> w;
    w.init(nterms, nx, ny, scales, small_scale_bias);
    const int rc = w.setup_from_psf(psf);
    scales = w.scales;
    psf_support = w.psf_support;
    if (rc == 0) w.pack_hessians(hessian, inverse_hessian);
    return rc;
}

template <typename T>
CleanResult<T> clean(T* residual, T* model, const T* psf, const T* mask, int nterms, int nx, int ny,
                     const std::vector<float>& scales, float small_scale_bias, int niter, T gain, T threshold,
                     T stop_fraction, T mask_threshold) {
    if (niter < 0) throw std::invalid_argument("niter must be >= 0");
    if (!std::isfinite(static_cast<double>(gain)) || !std::isfinite(static_cast<double>(threshold)) ||
        !std::isfinite(static_cast<double>(stop_fraction)))
        throw std::invalid_argument("gain, threshold and stop_fraction must be finite");

    Work<T> w;
    w.init(nterms, nx, ny, scales, small_scale_bias);
    if (w.setup_from_psf(psf) == kSingularHessian)
        throw std::runtime_error(
            "MT-Cleaner error : Non-invertible Hessian. Please check if the multi-frequency data "
            "selection is appropriate for a polynomial fit of the desired order.");

    CleanResult<T> out;
    out.scales = w.scales;
    out.small_scale_bias = w.small_scale_bias;
    out.psf_support = w.psf_support;
    w.pack_hessians(out.hessian, out.inverse_hessian);

    const int ns = w.nscales;
    const int nt = w.nterms;
    const std::size_t nimg = w.nimg();
    const std::size_t nspec = w.nspec();

    std::vector<std::vector<T>> vec_scale_masks(ns, std::vector<T>(nimg, static_cast<T>(1)));
    if (mask != nullptr) {
        std::vector<typename Work<T>::complex_t> mask_ft(nspec), work(nspec);
        std::vector<T> scratch(nimg);
        forward_r2c<T>(mask, mask_ft.data(), nx, ny);
        for (int s = 0; s < ns; ++s) {
            std::vector<T>& sm = vec_scale_masks[static_cast<std::size_t>(s)];
            for (std::size_t k = 0; k < nspec; ++k) work[k] = mask_ft[k] * w.scale_ft[static_cast<std::size_t>(s)][k];
            backward_c2r<T>(work.data(), sm.data(), nx, ny);
            flip_quadrants<T>(sm.data(), nx, ny, scratch.data());
            if (mask_threshold > static_cast<T>(0))
                for (std::size_t k = 0; k < nimg; ++k)
                    sm[k] = (sm[k] > static_cast<T>(0.1)) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
    for (int s = 1; s < ns; ++s) {
        std::vector<T>& sm = vec_scale_masks[static_cast<std::size_t>(s)];
        const int border = static_cast<int>(w.scales[static_cast<std::size_t>(s)] * 1.5f);
        auto zero_region = [&](int bx, int by, int tx, int ty) {
            bx = clampi(bx, 0, nx - 1);
            by = clampi(by, 0, ny - 1);
            tx = clampi(tx, 0, nx - 1);
            ty = clampi(ty, 0, ny - 1);
            for (int j = by; j <= ty; ++j)
                for (int i = bx; i <= tx; ++i) sm[static_cast<std::size_t>(j) * nx + i] = static_cast<T>(0);
        };
        zero_region(0, 0, nx - 1, border);
        zero_region(0, ny - border - 1, nx - 1, ny - 1);
        zero_region(0, border, border, ny - border - 1);
        zero_region(nx - border - 1, border, nx - 1, ny - border - 1);
    }

    std::vector<std::vector<T>> mat_r(static_cast<std::size_t>(nt) * ns, std::vector<T>(nimg, static_cast<T>(0)));
    std::vector<std::vector<T>> mat_coeffs(static_cast<std::size_t>(nt) * ns, std::vector<T>(nimg, static_cast<T>(0)));
    std::vector<std::vector<T>> vec_work(ns, std::vector<T>(nimg, static_cast<T>(0)));
    {
        std::vector<typename Work<T>::complex_t> dirty_ft(nspec), work(nspec);
        std::vector<T> scratch(nimg);
        for (int t = 0; t < nt; ++t) {
            forward_r2c<T>(residual + static_cast<std::size_t>(t) * nimg, dirty_ft.data(), nx, ny);
            for (int s = 0; s < ns; ++s) {
                for (std::size_t k = 0; k < nspec; ++k)
                    work[k] = dirty_ft[k] * w.scale_ft[static_cast<std::size_t>(s)][k];
                std::vector<T>& r = mat_r[static_cast<std::size_t>(w.ind2(t, s))];
                backward_c2r<T>(work.data(), r.data(), nx, ny);
                flip_quadrants<T>(r.data(), nx, ny, scratch.data());
            }
        }
    }

    T loopgain = (gain > static_cast<T>(0)) ? gain : static_cast<T>(0.5);
    T fluxlimit = static_cast<T>(-1);
    T prev_max = static_cast<T>(1e10);
    T min_max = static_cast<T>(1e10);
    T global_max_val = static_cast<T>(0);
    std::array<int, 2> global_max_pos{0, 0};
    int max_scale_index = 0;
    std::array<int, 2> blc{0, 0}, trc{nx - 1, ny - 1}, blc_psf{0, 0}, trc_psf{0, 0};
    std::vector<T> max_scale_val(ns, static_cast<T>(0));
    std::vector<std::array<int, 2>> max_scale_pos(ns, {0, 0});
    std::vector<std::vector<T>> delta_model(nt, std::vector<T>(nimg, static_cast<T>(0)));

    auto check_convergence = [&](int itercount) -> int {
        const PeakResult<T> pr =
            find_max_abs_mask<T>(mat_r[static_cast<std::size_t>(w.ind2(0, 0))].data(), vec_scale_masks[0].data(), nx, ny);
        const T norma = static_cast<T>(1.0 / w.mat_a[0][0]);
        const T rmaxval = std::abs(pr.value * norma);
        int flag = 0;
        if (std::abs(rmaxval) < std::max(threshold, fluxlimit)) flag = kStopThreshold;
        if (itercount > 1 && gain <= static_cast<T>(0)) {
            loopgain = (global_max_val < prev_max) ? loopgain * static_cast<T>(1.5) : loopgain / static_cast<T>(1.5);
            loopgain = std::min(static_cast<T>(1) - stop_fraction, loopgain);
            loopgain = std::min(static_cast<T>(0.6), loopgain);
            if (loopgain < static_cast<T>(0.01)) flag = kStopDiverged;
            if (std::abs((min_max - global_max_val) / min_max) > static_cast<T>(2)) flag = kStopDiverged;
        }
        prev_max = global_max_val;
        min_max = std::min(min_max, std::abs(global_max_val));
        if (flag == 0 && fluxlimit == static_cast<T>(-1)) fluxlimit = rmaxval * stop_fraction;
        return flag;
    };

    int stop = check_convergence(0);
    int iterdone = 0;
    if (stop == kStopThreshold) {
        out.iterations = 0;
        out.stop_code = kStopThreshold;
    } else {
        for (int itercount = 0; itercount < niter; ++itercount) {
            global_max_val = static_cast<T>(-1e10);
            if (itercount == 0) {
                blc = {0, 0};
                trc = {nx - 1, ny - 1};
            }
            for (int scale = 0; scale < ns; ++scale) {
                for (int t1 = 0; t1 < nt; ++t1) {
                    std::vector<T>& coeffs = mat_coeffs[static_cast<std::size_t>(w.ind2(t1, scale))];
                    for (int iy = blc[1]; iy <= trc[1]; ++iy)
                        for (int ix = blc[0]; ix <= trc[0]; ++ix)
                            coeffs[static_cast<std::size_t>(iy) * nx + ix] = static_cast<T>(0);
                    for (int t2 = 0; t2 < nt; ++t2) {
                        const T inv = static_cast<T>(
                            w.inv_mat_a[static_cast<std::size_t>(scale)][static_cast<std::size_t>(t1) * nt + t2]);
                        const std::vector<T>& rhs = mat_r[static_cast<std::size_t>(w.ind2(t2, scale))];
                        for (int iy = blc[1]; iy <= trc[1]; ++iy)
                            for (int ix = blc[0]; ix <= trc[0]; ++ix) {
                                const std::size_t k = static_cast<std::size_t>(iy) * nx + ix;
                                coeffs[k] += inv * rhs[k];
                            }
                    }
                }
                std::vector<T>& work = vec_work[static_cast<std::size_t>(scale)];
                for (int iy = blc[1]; iy <= trc[1]; ++iy)
                    for (int ix = blc[0]; ix <= trc[0]; ++ix)
                        work[static_cast<std::size_t>(iy) * nx + ix] = static_cast<T>(0);
                for (int t1 = 0; t1 < nt; ++t1) {
                    const std::vector<T>& coeffs = mat_coeffs[static_cast<std::size_t>(w.ind2(t1, scale))];
                    const std::vector<T>& resid = mat_r[static_cast<std::size_t>(w.ind2(t1, scale))];
                    for (int iy = blc[1]; iy <= trc[1]; ++iy)
                        for (int ix = blc[0]; ix <= trc[0]; ++ix) {
                            const std::size_t k = static_cast<std::size_t>(iy) * nx + ix;
                            work[k] += coeffs[k] * resid[k];
                        }
                }
                const PeakResult<T> pr = find_max_abs_mask<T>(
                    work.data(), vec_scale_masks[static_cast<std::size_t>(scale)].data(), nx, ny);
                max_scale_val[static_cast<std::size_t>(scale)] = pr.value;
                max_scale_pos[static_cast<std::size_t>(scale)] = {pr.ix, pr.iy};
            }
            for (int scale = 0; scale < ns; ++scale)
                if (max_scale_val[static_cast<std::size_t>(scale)] * w.scale_bias[static_cast<std::size_t>(scale)] >
                    global_max_val) {
                    global_max_val = max_scale_val[static_cast<std::size_t>(scale)] *
                                     w.scale_bias[static_cast<std::size_t>(scale)];
                    global_max_pos = max_scale_pos[static_cast<std::size_t>(scale)];
                    max_scale_index = scale;
                }

            const std::size_t gpk =
                static_cast<std::size_t>(global_max_pos[1]) * nx + static_cast<std::size_t>(global_max_pos[0]);
            const T mask_at_peak = vec_scale_masks[static_cast<std::size_t>(max_scale_index)][gpk];
            if (mask_at_peak == static_cast<T>(0) || global_max_val <= static_cast<T>(0)) {
                stop = kStopNothingToClean;
                break;
            }

            const std::array<int, 2> shape{nx, ny};
            const std::array<int, 2> psf_shape{w.psf_support, w.psf_support};
            const std::array<int, 2> psf_peak_local{w.psf_support / 2, w.psf_support / 2};
            blc = {global_max_pos[0] - w.psf_support / 2, global_max_pos[1] - w.psf_support / 2};
            trc = {global_max_pos[0] + w.psf_support / 2 - 1, global_max_pos[1] + w.psf_support / 2 - 1};
            verify_box(blc, trc, shape);
            blc_psf = {blc[0] + psf_peak_local[0] - global_max_pos[0], blc[1] + psf_peak_local[1] - global_max_pos[1]};
            trc_psf = {trc[0] + psf_peak_local[0] - global_max_pos[0], trc[1] + psf_peak_local[1] - global_max_pos[1]};
            verify_box(blc_psf, trc_psf, psf_shape);
            make_boxes_same_size(blc, trc, blc_psf, trc_psf);

            const int sup = w.psf_support;
            const std::vector<T>& scale_sub = w.vec_scales[static_cast<std::size_t>(max_scale_index)];
            const int bw = trc[0] - blc[0];
            const int bh = trc[1] - blc[1];
            for (int t = 0; t < nt; ++t) {
                const T coeff = mat_coeffs[static_cast<std::size_t>(w.ind2(t, max_scale_index))][gpk];
                T* mdl = model + static_cast<std::size_t>(t) * nimg;
                std::vector<T>& dlt = delta_model[static_cast<std::size_t>(t)];
                for (int dy = 0; dy <= bh; ++dy)
                    for (int dx = 0; dx <= bw; ++dx) {
                        const std::size_t km = static_cast<std::size_t>(blc[1] + dy) * nx + (blc[0] + dx);
                        const std::size_t kp = static_cast<std::size_t>(blc_psf[1] + dy) * sup + (blc_psf[0] + dx);
                        const T add = scale_sub[kp] * loopgain * coeff;
                        mdl[km] += add;
                        dlt[km] += add;
                    }
            }
            std::vector<T> coeffs(nt);
            for (int t = 0; t < nt; ++t)
                coeffs[t] = mat_coeffs[static_cast<std::size_t>(w.ind2(t, max_scale_index))][gpk];
            for (int scale = 0; scale < ns; ++scale)
                for (int t1 = 0; t1 < nt; ++t1) {
                    std::vector<T>& resid = mat_r[static_cast<std::size_t>(w.ind2(t1, scale))];
                    for (int t2 = 0; t2 < nt; ++t2) {
                        const std::vector<T>& smooth =
                            w.cube_a[static_cast<std::size_t>(w.cross_index4(t1, t2, scale, max_scale_index))];
                        const T g = loopgain * coeffs[t2];
                        for (int dy = 0; dy <= bh; ++dy)
                            for (int dx = 0; dx <= bw; ++dx) {
                                const std::size_t kr = static_cast<std::size_t>(blc[1] + dy) * nx + (blc[0] + dx);
                                const std::size_t kp =
                                    static_cast<std::size_t>(blc_psf[1] + dy) * sup + (blc_psf[0] + dx);
                                resid[kr] -= smooth[kp] * g;
                            }
                    }
                }

            ++iterdone;
            stop = check_convergence(itercount);
            if (stop) break;
        }
        if (stop == 0) stop = kStopMaxIter;
        out.iterations = iterdone;
        out.stop_code = stop;

        std::vector<T> smooth_mod(nimg), scratch(nimg);
        std::vector<typename Work<T>::complex_t> model_ft(nspec), work(nspec);
        for (int t2 = 0; t2 < nt; ++t2) {
            forward_r2c<T>(delta_model[static_cast<std::size_t>(t2)].data(), model_ft.data(), nx, ny);
            for (int t1 = 0; t1 < nt; ++t1) {
                const std::vector<typename Work<T>::complex_t>& pf = w.psf_ft[static_cast<std::size_t>(t1 + t2)];
                for (std::size_t k = 0; k < nspec; ++k) work[k] = pf[k] * model_ft[k];
                backward_c2r<T>(work.data(), smooth_mod.data(), nx, ny);
                flip_quadrants<T>(smooth_mod.data(), nx, ny, scratch.data());
                T* d = residual + static_cast<std::size_t>(t1) * nimg;
                for (std::size_t k = 0; k < nimg; ++k) d[k] -= smooth_mod[k];
            }
        }
    }

    if (mask != nullptr)
        out.peak_residual = peak_abs_masked<T>(residual, mask, nimg);
    else
        out.peak_residual = peak_abs<T>(residual, nimg);
    out.model_flux = sum_array<T>(model, nimg);
    return out;
}

template <typename T>
void principal_solution(T* residual, const double* inverse_hessian, int nterms, int nx, int ny) {
    check_geometry(nterms, nx, ny);
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    std::vector<T> in(nterms);
    for (std::size_t k = 0; k < n; ++k) {
        for (int t = 0; t < nterms; ++t) in[t] = residual[static_cast<std::size_t>(t) * n + k];
        for (int t1 = 0; t1 < nterms; ++t1) {
            T acc = static_cast<T>(0);
            for (int t2 = 0; t2 < nterms; ++t2)
                acc += static_cast<T>(inverse_hessian[static_cast<std::size_t>(t1) * nterms + t2]) * in[t2];
            residual[static_cast<std::size_t>(t1) * n + k] = acc;
        }
    }
}

template int taylor_hessian<float>(const float*, int, int, int, std::vector<float>&, float, std::vector<double>&,
                                   std::vector<double>&, int&);
template int taylor_hessian<double>(const double*, int, int, int, std::vector<float>&, float, std::vector<double>&,
                                    std::vector<double>&, int&);
template CleanResult<float> clean<float>(float*, float*, const float*, const float*, int, int, int,
                                         const std::vector<float>&, float, int, float, float, float, float);
template CleanResult<double> clean<double>(double*, double*, const double*, const double*, int, int, int,
                                           const std::vector<float>&, float, int, double, double, double, double);
template void principal_solution<float>(float*, const double*, int, int, int);
template void principal_solution<double>(double*, const double*, int, int, int);

}  // namespace mtmfs
