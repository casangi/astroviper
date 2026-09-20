// Implementation of the Adaptive Scale Pixel (Asp) deconvolver. Ported from
// CASA synthesis/MeasurementEquations/AspMatrixCleaner.cc (the active "gold"
// objective path) and synthesis/ImagerObjects/SDAlgorithmAAspClean.cc.
//
// casacore types are replaced with the small `Mat` double-precision matrix
// below and raw pointers into the caller's (Python-owned) buffers; FFTServer
// is replaced by asp_fft.hpp (pocketfft) and ALGLIB by asp_lbfgs.hpp.
//
// The algorithm is CASA's, but the FFT bookkeeping is not: CASA transforms
// every Aspen image (and its scale derivative) with full-size forward FFTs
// inside the L-BFGS objective and inverts each convolution separately. Here
// the Aspen is separable, so its spectrum is the outer product of two 1-D
// transforms, and pairs of real convolutions share one complex inverse
// transform (ifft2(A + iB) = a + ib). Per objective evaluation that is one
// 2-D FFT instead of four, and per minor-cycle iteration about eleven instead
// of ~80, with results identical to rounding.

#include "../include/asp_clean.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <deque>
#include <numeric>
#include <thread>
#include <vector>

#include "../include/asp_fft.hpp"
#include "../include/asp_lbfgs.hpp"

namespace aspclean {

namespace {

using aspfft::cd;

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kSqrtTwoPi = 2.5066282746310002;

// ---------------------------------------------------------------------------
// Lightweight column-compatible 2-D matrix of doubles.
//
// Element (i, j) lives at d[j*nx + i] with i the fast ("x") axis. This matches
// both a C-contiguous numpy array of shape (ny, nx) and a casacore
// Matrix<Float> of shape (nx, ny), so the ported index expressions are
// unchanged from the original source.
// ---------------------------------------------------------------------------
struct Mat {
    int nx = 0, ny = 0;
    std::vector<double> d;

    Mat() = default;
    Mat(int nx_, int ny_) : nx(nx_), ny(ny_), d(static_cast<std::size_t>(nx_) * ny_, 0.0) {}

    double& operator()(int i, int j) { return d[static_cast<std::size_t>(j) * nx + i]; }
    double operator()(int i, int j) const { return d[static_cast<std::size_t>(j) * nx + i]; }
    void zero() { std::fill(d.begin(), d.end(), 0.0); }
    std::size_t size() const { return d.size(); }
};

// Recentre (fftshift) a real image so the corner-origin convolution result is
// moved back to the image centre: out(i,j) = in((i+cx)%nx, (j+cy)%ny).
template <typename V>
void flip_inplace(std::vector<V>& a, int nx, int ny) {
    const int cx = nx / 2, cy = ny / 2;
    std::vector<V> out(a.size());
    for (int j = 0; j < ny; ++j) {
        const int sj = (j + cy) % ny;
        for (int i = 0; i < nx; ++i) {
            const int si = (i + cx) % nx;
            out[static_cast<std::size_t>(j) * nx + i] =
                a[static_cast<std::size_t>(sj) * nx + si];
        }
    }
    a.swap(out);
}

// out = flip( ifft2( xfrA .* xfrB ) ): the centred circular convolution of the
// two real images whose forward spectra are xfrA, xfrB.
void conv_from_spectra(const std::vector<cd>& xfrA, const std::vector<cd>& xfrB,
                       int nx, int ny, std::vector<double>& out) {
    std::vector<cd> prod(xfrA.size());
    for (std::size_t k = 0; k < prod.size(); ++k) prod[k] = xfrA[k] * xfrB[k];
    out.assign(static_cast<std::size_t>(nx) * ny, 0.0);
    aspfft::irfft2_inverse(std::move(prod), nx, ny, out.data());
    flip_inplace(out, nx, ny);
}

// min/max of (array * weight), as in casacore minMaxMasked.
void min_max_masked(const Mat& a, const Mat& w, double& minVal, double& maxVal,
                    int& minI, int& minJ, int& maxI, int& maxJ) {
    minVal = a(0, 0) * w(0, 0);
    maxVal = minVal;
    minI = minJ = maxI = maxJ = 0;
    for (int j = 0; j < a.ny; ++j) {
        for (int i = 0; i < a.nx; ++i) {
            const double v = a(i, j) * w(i, j);
            if (v > maxVal) { maxVal = v; maxI = i; maxJ = j; }
            if (v < minVal) { minVal = v; minI = i; minJ = j; }
        }
    }
}

// findMaxAbsMask: peak by absolute value over (array * weight); signed value
// returned. The cleaner always runs with a mask present (CASA's driver always
// supplies a mask image, all-ones when "clean everywhere"), so only the masked
// peak finder is needed.
void find_max_abs_mask(const Mat& a, const Mat& w, double& maxAbs, int& posI,
                       int& posJ) {
    double mn, mx; int mnI, mnJ, mxI, mxJ;
    min_max_masked(a, w, mn, mx, mnI, mnJ, mxI, mxJ);
    maxAbs = mx; posI = mxI; posJ = mxJ;
    if (std::abs(mn) > std::abs(mx)) { maxAbs = mn; posI = mnI; posJ = mnJ; }
}

// Clamp a box [blc, trc] (inclusive) into the image, keeping blc <= trc
// (casacore LCBox::verify equivalent for our 2-D use).
void verify_box(int& blc0, int& blc1, int& trc0, int& trc1, int nx, int ny) {
    blc0 = std::max(0, std::min(blc0, nx - 1));
    blc1 = std::max(0, std::min(blc1, ny - 1));
    trc0 = std::max(blc0, std::min(trc0, nx - 1));
    trc1 = std::max(blc1, std::min(trc1, ny - 1));
}

// ---------------------------------------------------------------------------
// Separable Gaussian helpers. A Gaussian (or delta) centred on a pixel is the
// outer product of a row profile and a column profile, so its 2-D DFT is the
// outer product of the two 1-D transforms: no 2-D forward FFT is needed for
// any scale or Aspen image.
// ---------------------------------------------------------------------------

// Row profile exp(-(i - c)^2 / (2 sigma^2)) for i in [lo, hi] (inclusive,
// clamped to [0, n)), zero elsewhere; sigma == 0 gives a delta at c.
std::vector<double> gaussian_profile(int n, int c, double sigma, int lo, int hi) {
    std::vector<double> row(static_cast<std::size_t>(n), 0.0);
    lo = std::max(lo, 0);
    hi = std::min(hi, n - 1);
    if (sigma == 0.0) {
        if (c >= 0 && c < n) row[static_cast<std::size_t>(c)] = 1.0;
        return row;
    }
    const double inv2s2 = 0.5 / (sigma * sigma);
    for (int i = lo; i <= hi; ++i) {
        const double d = static_cast<double>(i - c);
        row[static_cast<std::size_t>(i)] = std::exp(-d * d * inv2s2);
    }
    return row;
}

// spec(kx, ky) += a * X(kx) * Y(ky), image layout (kx fast).
void add_outer_product(std::vector<cd>& spec, const std::vector<cd>& X,
                       const std::vector<cd>& Y, double a, int nx, int ny) {
    for (int ky = 0; ky < ny; ++ky) {
        const cd ay = a * Y[static_cast<std::size_t>(ky)];
        cd* row = spec.data() + static_cast<std::size_t>(ky) * nx;
        for (int kx = 0; kx < nx; ++kx) row[kx] += ay * X[static_cast<std::size_t>(kx)];
    }
}

// Forward spectrum of the full-image scale image that makeScaleImage produces:
// a delta (scaleSize == 0) or the Gaussian norm * exp(-r^2 / (2 s^2)) centred
// on (ci, cj) with norm = 1 / (sqrt(2 pi) s), CASA's normalisation.
std::vector<cd> scale_image_spectrum(int nx, int ny, double scaleSize, int ci, int cj) {
    const std::vector<cd> X = aspfft::fft1d_forward(gaussian_profile(nx, ci, scaleSize, 0, nx - 1));
    const std::vector<cd> Y = aspfft::fft1d_forward(gaussian_profile(ny, cj, scaleSize, 0, ny - 1));
    const double norm = (scaleSize == 0.0) ? 1.0 : 1.0 / (kSqrtTwoPi * scaleSize);
    std::vector<cd> spec(static_cast<std::size_t>(nx) * ny, cd(0.0, 0.0));
    add_outer_product(spec, X, Y, norm, nx, ny);
    return spec;
}

// ---------------------------------------------------------------------------
// Objective for the per-Aspen (amplitude, scale) optimization. This is the
// "gold" objfunc_alglib for a single active Aspen (AspLen == 1): minimize
// sum_box (dirty / rescale - amp * (Aspen (*) psf))^2, where `rescale` is
// CASA's itsRescale (|strength| / psfWidth): the optimizer works on the
// rescaled amplitude amp = strength / rescale (about psfWidth in size), which
// balances the two variables for the scaled L-BFGS.
//
// The Aspen is CASA's box-truncated Gaussian (zero beyond 2.5 sigma) and the
// residual sum and gradients run over that box with exclusive upper bounds,
// exactly as in the original. What differs is only how (Aspen (*) psf) and
// (dAspen/dscale (*) psf) are obtained: the truncated Gaussian is separable
// and its scale derivative is a sum of three separable terms, so both spectra
// come from four 1-D transforms, and the two real convolutions are recovered
// from a single complex inverse FFT of (AspFT + i dAspFT) * psfFT.
// ---------------------------------------------------------------------------
struct AspObjective {
    const Mat& dirty;              // current residual
    const std::vector<cd>& psfFT;  // forward spectrum of the PSF
    int nx, ny;
    int cx, cy;                    // Aspen centre (positionOptimum)
    double inv_rescale;            // 1 / itsRescale

    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        const double amp = x[0];
        double asc = std::fabs(x[1]);
        if (!std::isfinite(amp) || asc < 1e-3) asc = 1e-3;  // keep finite

        const double sigma5 = 5.0 * asc / 2.0;
        const int minI = std::max(0, static_cast<int>(cx - sigma5));
        const int maxI = std::min(nx - 1, static_cast<int>(cx + sigma5));
        const int minJ = std::max(0, static_cast<int>(cy - sigma5));
        const int maxJ = std::min(ny - 1, static_cast<int>(cy + sigma5));

        // Separable factors on the truncation box: g(d) = exp(-d^2 / 2 s^2)
        // and h(d) = (d^2 / s^2) g(d), so that
        //   Asp  = norm * g(di) g(dj)
        //   dAsp = Asp * ((di^2 + dj^2) / s^2 - 1) / s
        //        = (norm / s) * [h(di) g(dj) + g(di) h(dj) - g(di) g(dj)].
        const double inv_s2 = 1.0 / (asc * asc);
        std::vector<double> gx = gaussian_profile(nx, cx, asc, minI, maxI);
        std::vector<double> gy = gaussian_profile(ny, cy, asc, minJ, maxJ);
        std::vector<double> hx(gx.size(), 0.0), hy(gy.size(), 0.0);
        for (int i = minI; i <= maxI; ++i) {
            const double d = static_cast<double>(i - cx);
            hx[static_cast<std::size_t>(i)] = d * d * inv_s2 * gx[static_cast<std::size_t>(i)];
        }
        for (int j = minJ; j <= maxJ; ++j) {
            const double d = static_cast<double>(j - cy);
            hy[static_cast<std::size_t>(j)] = d * d * inv_s2 * gy[static_cast<std::size_t>(j)];
        }
        const std::vector<cd> Gx = aspfft::fft1d_forward(gx);
        const std::vector<cd> Gy = aspfft::fft1d_forward(gy);
        const std::vector<cd> Hx = aspfft::fft1d_forward(hx);
        const std::vector<cd> Hy = aspfft::fft1d_forward(hy);

        const double norm = 1.0 / (kSqrtTwoPi * asc);
        const double dnorm = norm / asc;
        std::vector<cd> spec(static_cast<std::size_t>(nx) * ny);
        for (int ky = 0; ky < ny; ++ky) {
            const cd gyk = Gy[static_cast<std::size_t>(ky)];
            const cd hyk = Hy[static_cast<std::size_t>(ky)];
            const std::size_t base = static_cast<std::size_t>(ky) * nx;
            for (int kx = 0; kx < nx; ++kx) {
                const cd gxk = Gx[static_cast<std::size_t>(kx)];
                const cd hxk = Hx[static_cast<std::size_t>(kx)];
                const cd aspFT = norm * gxk * gyk;
                const cd dAspFT = dnorm * (hxk * gyk + gxk * hyk - gxk * gyk);
                // (Asp + i dAsp) (*) psf in one transform: both results are real
                spec[base + kx] = cd(aspFT.real() - dAspFT.imag(), aspFT.imag() + dAspFT.real()) *
                                  psfFT[base + kx];
            }
        }
        std::vector<cd> conv = aspfft::cfft2_inverse(std::move(spec), nx, ny);
        flip_inplace(conv, nx, ny);

        double func = 0.0, dA = 0.0, dS = 0.0;
        for (int j = minJ; j < maxJ; ++j) {
            for (int i = minI; i < maxI; ++i) {
                const std::size_t off = static_cast<std::size_t>(j) * nx + i;
                const double aspConvPsf = conv[off].real();
                const double dAspConvPsf = conv[off].imag();
                const double res = dirty(i, j) * inv_rescale - amp * aspConvPsf;
                func += res * res;
                dA += -2.0 * res * aspConvPsf;
                dS += -2.0 * amp * res * dAspConvPsf;
            }
        }
        grad[0] = dA;
        grad[1] = dS;
        return func;
    }
};

// ---------------------------------------------------------------------------
// The Asp cleaner. Owns its internal work buffers; the residual and model are
// referenced through raw pointers into the caller's arrays (never copied).
// ---------------------------------------------------------------------------
template <typename T>
class Cleaner {
public:
    Cleaner(T* residual, T* model, const T* psf, const T* mask, int nx, int ny,
            double gain, double threshold, int niter, double fusedthreshold,
            double psf_width, int largestscale, int stoppointmode,
            int norm_method, bool verbose)
        : residual_(residual), model_(model), psf_(psf), mask_(mask),
          nx_(nx), ny_(ny), gain_(gain), threshold_(threshold),
          maxNiter_(niter), fusedThreshold_(fusedthreshold),
          userLargestScale_(static_cast<double>(largestscale)),
          stopPointMode_(stoppointmode),
          normMethod_(norm_method <= 0 ? 1 : norm_method), verbose_(verbose),
          psfWidthInput_(psf_width) {}

    AspResult run() {
        setup();
        AspResult res;
        if (noClean_) {  // mask forbids cleaning everywhere
            res.psf_width = psfWidth_;
            res.peak_residual = 0.0;
            res.model_flux = modelSum();
            return res;
        }
        res.retval = aspclean();
        res.iterations = iteration_;
        res.peak_residual = peakResidual_;
        res.model_flux = modelSum();
        res.switched_to_hogbom = switchedToHogbom_;
        res.psf_width = psfWidth_;
        return res;
    }

private:
    // --- caller-owned buffers ---
    T* residual_;
    T* model_;
    const T* psf_;
    const T* mask_;
    int nx_, ny_;

    // --- control parameters ---
    double gain_, threshold_;
    int maxNiter_;
    double fusedThreshold_;
    double userLargestScale_;
    int stopPointMode_;
    int normMethod_;
    bool verbose_;
    double psfWidthInput_;

    // --- derived / state ---
    double psfWidth_ = 0.0;
    std::vector<cd> psfXfr_;                 // itsXfr
    std::vector<double> initScaleSizes_;     // itsInitScaleSizes
    int nInitScales_ = 5;
    std::vector<Mat> initScales_;            // itsInitScales
    std::vector<std::vector<cd>> initScaleXfrs_;  // itsInitScaleXfrs
    std::vector<Mat> initScaleMasks_;        // itsInitScaleMasks
    std::vector<Mat> dirtyConvInitScales_;   // itsDirtyConvInitScales
    Mat maskMat_;                            // itsMask (always present)
    double maskThreshold_ = 0.99;
    bool noClean_ = false;
    int blc0_ = 0, blc1_ = 0, trc0_ = 0, trc1_ = 0;  // blcDirty/trcDirty bbox

    // optimum found by getActiveSetAspen()
    double strengthOptimum_ = 0.0;
    int optimumScale_ = 0;
    double optimumScaleSize_ = 0.0;
    int posOptI_ = 0, posOptJ_ = 0;

    // running state
    double peakResidual_ = 1000.0, prevPeakResidual_ = 0.0;
    bool switchedToHogbom_ = false;
    unsigned int numHogbomIter_ = 0, nthHogbom_ = 0;
    std::deque<int> numIterNoGoodAspen_;
    unsigned int numNoChange_ = 0;
    bool stopAtLargeScaleNegative_ = false;
    bool didStopPointMode_ = false;
    int iteration_ = 0, startingIter_ = 0;
    double maximumResidual_ = 0.0, totalFlux_ = 0.0;
    std::vector<double> goodAspActiveSet_, goodAspAmplitude_;

    // ---- accessors into caller buffers ----
    double dirtyAt(int i, int j) const {
        return static_cast<double>(residual_[static_cast<std::size_t>(j) * nx_ + i]);
    }
    void dirtySet(int i, int j, double v) {
        residual_[static_cast<std::size_t>(j) * nx_ + i] = static_cast<T>(v);
    }
    double modelAt(int i, int j) const {
        return static_cast<double>(model_[static_cast<std::size_t>(j) * nx_ + i]);
    }
    void modelAdd(int i, int j, double v) {
        model_[static_cast<std::size_t>(j) * nx_ + i] += static_cast<T>(v);
    }
    double modelSum() const {
        double s = 0.0;
        for (std::size_t k = 0; k < static_cast<std::size_t>(nx_) * ny_; ++k)
            s += static_cast<double>(model_[k]);
        return s;
    }

    double computeThreshold() const { return std::max(0.0 * maximumResidual_, threshold_); }

    // -----------------------------------------------------------------------
    // One-time setup, equivalent to SDAlgorithmAAspClean::initializeDeconvolver.
    // -----------------------------------------------------------------------
    void setup() {
        // setPsf: forward spectrum of the PSF.
        {
            std::vector<double> psfd(static_cast<std::size_t>(nx_) * ny_);
            for (std::size_t k = 0; k < psfd.size(); ++k)
                psfd[k] = static_cast<double>(psf_[k]);
            psfXfr_ = aspfft::rfft2_forward(psfd.data(), nx_, ny_);
        }

        // getPsfGaussianWidth (or use the supplied width).
        if (psfWidthInput_ > 0.0) {
            psfWidth_ = psfWidthInput_;
        } else {
            std::vector<double> psfd(static_cast<std::size_t>(nx_) * ny_);
            for (std::size_t k = 0; k < psfd.size(); ++k)
                psfd[k] = static_cast<double>(psf_[k]);
            psfWidth_ = psf_gaussian_width(psfd.data(), nx_, ny_);
        }
        if (psfWidth_ < 1.0) psfWidth_ = 1.0;

        setInitScaleXfrs();
        setInitScaleMasks();

        if (noClean_) return;

        // getActiveSetAspen() once to seed the first optimum, then defineAspScales.
        std::vector<double> scaleSizes = getActiveSetAspen();
        scaleSizes.push_back(0.0);
        defineAspScales(scaleSizes);
    }

    // setInitScales: choose 0, w, 2w, 4w, 8w (restricted by largestscale).
    void setInitScales() {
        if (userLargestScale_ < 0.0) {
            initScaleSizes_ = {0.0, psfWidth_, 2.0 * psfWidth_, 4.0 * psfWidth_,
                               8.0 * psfWidth_};
            return;
        }
        double uls = userLargestScale_;
        if (uls > std::min(nx_ / 10, ny_ / 10))
            uls = std::ceil(std::min(nx_ / 10, ny_ / 10));
        const int numscale = static_cast<int>(std::floor(uls / psfWidth_));
        if (numscale == 0) {
            nInitScales_ = 1;
            initScaleSizes_ = {0.0};
        } else {
            initScaleSizes_ = {0.0};
            int scale = 1;
            while (((psfWidth_ * std::pow(2.0, scale - 1)) < uls) && (scale < 5)) {
                initScaleSizes_.push_back(psfWidth_ * std::pow(2.0, scale - 1));
                ++scale;
            }
            if (scale <= 4) initScaleSizes_.push_back(uls);
            nInitScales_ = static_cast<int>(initScaleSizes_.size());
        }
        userLargestScale_ = uls;
    }

    // makeInitScaleImage: centred Gaussian (or delta for size 0).
    void makeInitScaleImage(Mat& iscale, double scaleSize) const {
        iscale.zero();
        const double refi = nx_ / 2;
        const double refj = ny_ / 2;
        if (scaleSize == 0.0) {
            iscale(static_cast<int>(refi), static_cast<int>(refj)) = 1.0;
            return;
        }
        const double inv2s2 = 0.5 / (scaleSize * scaleSize);
        const double norm = 1.0 / (kSqrtTwoPi * scaleSize);
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                const double r2 = (i - refi) * (i - refi) + (j - refj) * (j - refj);
                iscale(i, j) = norm * std::exp(-r2 * inv2s2);
            }
    }

    // makeScaleImage: Gaussian (or delta) centred at `center`, evaluated as the
    // outer product of its row and column profiles.
    void makeScaleImage(Mat& iscale, double scaleSize, int ci, int cj) const {
        iscale.zero();
        if (scaleSize == 0.0) {
            iscale(ci, cj) = 1.0;
            return;
        }
        const double norm = 1.0 / (kSqrtTwoPi * scaleSize);
        const std::vector<double> gx = gaussian_profile(nx_, ci, scaleSize, 0, nx_ - 1);
        const std::vector<double> gy = gaussian_profile(ny_, cj, scaleSize, 0, ny_ - 1);
        for (int j = 0; j < ny_; ++j) {
            const double gyj = norm * gy[static_cast<std::size_t>(j)];
            for (int i = 0; i < nx_; ++i) iscale(i, j) = gyj * gx[static_cast<std::size_t>(i)];
        }
    }

    void setInitScaleXfrs() {
        nInitScales_ = 5;
        setInitScales();  // may reset nInitScales_
        initScales_.assign(nInitScales_, Mat(nx_, ny_));
        initScaleXfrs_.assign(nInitScales_, {});
        for (int s = 0; s < nInitScales_; ++s) {
            initScales_[s] = Mat(nx_, ny_);
            makeInitScaleImage(initScales_[s], initScaleSizes_[s]);
            initScaleXfrs_[s] = aspfft::rfft2_forward(initScales_[s].d.data(), nx_, ny_);
        }
    }

    // setInitScaleMasks: per-scale masks = (mask (*) scale) thresholded, with a
    // scale-dependent border zeroed. An absent mask is treated as all ones.
    void setInitScaleMasks() {
        maskMat_ = Mat(nx_, ny_);
        double maxMask = -1e30;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                const double v = mask_
                    ? static_cast<double>(mask_[static_cast<std::size_t>(j) * nx_ + i])
                    : 1.0;
                maskMat_(i, j) = v;
                if (v > maxMask) maxMask = v;
            }
        noClean_ = (maxMask < maskThreshold_);
        if (noClean_) return;

        std::vector<cd> maskFT = aspfft::rfft2_forward(maskMat_.d.data(), nx_, ny_);
        initScaleMasks_.assign(nInitScales_, Mat(nx_, ny_));
        for (int s = 0; s < nInitScales_; ++s) {
            initScaleMasks_[s] = Mat(nx_, ny_);
            std::vector<double> conv;
            conv_from_spectra(maskFT, initScaleXfrs_[s], nx_, ny_, conv);
            for (int j = 0; j < ny_; ++j)
                for (int i = 0; i < nx_; ++i) {
                    double v = conv[static_cast<std::size_t>(j) * nx_ + i];
                    if (maskThreshold_ > 0) v = (v > maskThreshold_) ? 1.0 : 0.0;
                    initScaleMasks_[s](i, j) = v;
                }
        }

        // Zero a border of width 1.5*scaleSize around each scale mask.
        for (int s = 0; s < nInitScales_; ++s) {
            const int border = static_cast<int>(initScaleSizes_[s] * 1.5);
            Mat& m = initScaleMasks_[s];
            auto zeroBox = [&](int b0, int b1, int t0, int t1) {
                verify_box(b0, b1, t0, t1, nx_, ny_);
                for (int j = b1; j <= t1; ++j)
                    for (int i = b0; i <= t0; ++i) m(i, j) = 0.0;
            };
            if (border > 0) {
                zeroBox(0, 0, nx_ - 1, border);                       // bottom
                zeroBox(0, ny_ - border - 1, nx_ - 1, ny_ - 1);       // top
                zeroBox(0, border, border, ny_ - border - 1);         // left
                zeroBox(nx_ - border - 1, border, nx_ - 1, ny_ - border - 1);  // right
            }
        }

        // blcDirty/trcDirty: bounding box of the (raw) mask.
        blc0_ = 0; blc1_ = 0; trc0_ = nx_ - 1; trc1_ = ny_ - 1;
        int xbeg = nx_ - 1, ybeg = ny_ - 1, xend = 0, yend = 0;
        bool any = false;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i)
                if (maskMat_(i, j) > 0.000001) {
                    any = true;
                    xbeg = std::min(xbeg, i); ybeg = std::min(ybeg, j);
                    xend = std::max(xend, i); yend = std::max(yend, j);
                }
        if (any) { blc0_ = xbeg; blc1_ = ybeg; trc0_ = xend; trc1_ = yend; }
    }

    // Sort the active scale set (CASA defineAspScales). itsUseZhang is false in
    // the gold path, so no scale rewriting is needed and the optimum used by the
    // next iteration is carried in the member variables set by getActiveSetAspen.
    void defineAspScales(std::vector<double>& scaleSizes) {
        std::sort(scaleSizes.begin(), scaleSizes.end());
    }

    void switchedToHogbom(bool runlong) {
        switchedToHogbom_ = true;
        if (fusedThreshold_ < 0) switchedToHogbom_ = false;  // user opted out
        nthHogbom_ += 1;
        numIterNoGoodAspen_.clear();
        numHogbomIter_ = 51;
        if (runlong) numHogbomIter_ = 510;
    }

    // maxDirtyConvInitScales: pick the strongest smoothed-residual peak across
    // the initial scales; renormalize for nonzero scale.
    void maxDirtyConvInitScales(double& strengthOptimum, int& optimumScale,
                                int& posI, int& posJ) {
        std::vector<double> maxima(nInitScales_, 0.0);
        std::vector<int> posMaxI(nInitScales_, 0), posMaxJ(nInitScales_, 0);

        Mat work(nx_, ny_);
        for (int s = 0; s < nInitScales_; ++s) {
            work.zero();
            for (int j = blc1_; j <= trc1_; ++j)
                for (int i = blc0_; i <= trc0_; ++i)
                    work(i, j) = dirtyConvInitScales_[s](i, j);

            double mx; int pi, pj;
            find_max_abs_mask(work, initScaleMasks_[s], mx, pi, pj);
            maxima[s] = mx; posMaxI[s] = pi; posMaxJ[s] = pj;

            if (normMethod_ == 2 && s > 0) {
                const double normalization = std::sqrt(kTwoPi * initScaleSizes_[s]);
                maxima[s] /= normalization;
            }
        }

        for (int s = 0; s < nInitScales_; ++s) {
            if (std::abs(maxima[s]) > std::abs(strengthOptimum)) {
                optimumScale = s;
                strengthOptimum = maxima[s];
                posI = posMaxI[s];
                posJ = posMaxJ[s];
            }
        }

        if (optimumScale > 0) {
            const double normalization = std::sqrt(
                kTwoPi / (std::pow(1.0 / psfWidth_, 2) +
                          std::pow(1.0 / initScaleSizes_[optimumScale], 2)));
            if (normMethod_ == 2)
                strengthOptimum *= std::sqrt(kTwoPi * initScaleSizes_[optimumScale]);
            strengthOptimum /= normalization;
        }
    }

    // getActiveSetAspen: find the next Aspen and optimize its (amp, scale).
    std::vector<double> getActiveSetAspen() {
        if (!switchedToHogbom_ &&
            std::accumulate(numIterNoGoodAspen_.begin(), numIterNoGoodAspen_.end(), 0) >= 5)
            switchedToHogbom(false);

        nInitScales_ = switchedToHogbom_ ? 1 : static_cast<int>(initScaleSizes_.size());

        // Dirty (*) initial scales.
        std::vector<double> dirtyd(static_cast<std::size_t>(nx_) * ny_);
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i)
                dirtyd[static_cast<std::size_t>(j) * nx_ + i] = dirtyAt(i, j);
        std::vector<cd> dirtyFT = aspfft::rfft2_forward(dirtyd.data(), nx_, ny_);

        // Two real convolutions per complex inverse transform:
        // ifft2(dirtyFT * X_s + i dirtyFT * X_{s+1}) = dirty (*) scale_s + i dirty (*) scale_{s+1}.
        dirtyConvInitScales_.assign(nInitScales_, Mat(nx_, ny_));
        for (int s = 0; s < nInitScales_; s += 2) {
            const bool paired = (s + 1 < nInitScales_);
            std::vector<cd> spec(dirtyFT.size());
            const std::vector<cd>& xa = initScaleXfrs_[s];
            for (std::size_t k = 0; k < spec.size(); ++k) {
                spec[k] = dirtyFT[k] * xa[k];
                if (paired) {
                    const cd b = dirtyFT[k] * initScaleXfrs_[s + 1][k];
                    spec[k] += cd(-b.imag(), b.real());
                }
            }
            std::vector<cd> conv = aspfft::cfft2_inverse(std::move(spec), nx_, ny_);
            flip_inplace(conv, nx_, ny_);
            dirtyConvInitScales_[s] = Mat(nx_, ny_);
            if (paired) dirtyConvInitScales_[s + 1] = Mat(nx_, ny_);
            for (std::size_t k = 0; k < conv.size(); ++k) {
                dirtyConvInitScales_[s].d[k] = conv[k].real();
                if (paired) dirtyConvInitScales_[s + 1].d[k] = conv[k].imag();
            }
        }

        double strengthOptimum = 0.0;
        int optimumScale = 0, posI = 0, posJ = 0;
        goodAspActiveSet_.clear();
        goodAspAmplitude_.clear();

        maxDirtyConvInitScales(strengthOptimum, optimumScale, posI, posJ);

        strengthOptimum_ = strengthOptimum;
        posOptI_ = posI;
        posOptJ_ = posJ;
        optimumScale_ = optimumScale;
        optimumScaleSize_ = initScaleSizes_[optimumScale];

        if (optimumScale_ == 0) return {};  // 0 scale needs no optimization

        Mat dirtyMat(nx_, ny_);
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) dirtyMat(i, j) = dirtyAt(i, j);

        // CASA: itsRescale = |strengthOptimum / itsPsfWidth|; the optimizer
        // sees the rescaled amplitude x[0] = strength / rescale with variable
        // scales s = (|strength| / rescale, scaleSize) = (psfWidth, scaleSize).
        double rescale = std::fabs(strengthOptimum / psfWidth_);
        if (!(rescale > 0.0) || !std::isfinite(rescale)) rescale = 1.0;
        AspObjective obj{dirtyMat, psfXfr_, nx_, ny_, posI, posJ, 1.0 / rescale};
        std::vector<double> x = {strengthOptimum / rescale, initScaleSizes_[optimumScale]};
        std::vector<double> sc = {std::fabs(strengthOptimum) / rescale, initScaleSizes_[optimumScale]};
        asplbfgs::Options opt;  // m=1, maxits=5, eps*=1e-3 -> matches ALGLIB call
        asplbfgs::minimize(x, sc, [&obj](const std::vector<double>& xx,
                                         std::vector<double>& gg) { return obj(xx, gg); },
                           opt);

        double amp = x[0] * rescale;
        double scale = x[1];
        if (std::fabs(scale) < 0.4) {
            scale = 0.0;
            amp = dirtyAt(posOptI_, posOptJ_);  // avoid divergence from huge amp
        } else {
            scale = std::fabs(scale);
        }

        goodAspAmplitude_.push_back(amp);
        goodAspActiveSet_.push_back(scale);
        strengthOptimum_ = amp;
        optimumScaleSize_ = scale;

        return goodAspActiveSet_;
    }

    // Peak |residual| over the scale-0 mask (itsInitScaleMasks[0]).
    double peakResidualMasked() {
        double minVal = dirtyAt(0, 0) * initScaleMasks_[0](0, 0);
        double maxVal = minVal;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                const double v = dirtyAt(i, j) * initScaleMasks_[0](i, j);
                if (v > maxVal) maxVal = v;
                if (v < minVal) minVal = v;
            }
        return std::max(std::fabs(maxVal), std::fabs(minVal));
    }

    // -----------------------------------------------------------------------
    // The minor cycle (AspMatrixCleaner::aspclean).
    // -----------------------------------------------------------------------
    int aspclean() {
        double totalFlux = 0.0;
        int converged = 0;
        int stopPointModeCounter = 0;
        double tmpMaximumResidual = 0.0;
        double minMaximumResidual = 1000.0;
        double initRMSResidual = 1000.0;

        iteration_ = startingIter_;

        peakResidual_ = peakResidualMasked();

        const int num = std::max(1, (trc0_ - blc0_) * (trc1_ - blc1_));
        auto rmsResidual = [&]() {
            double rms = 0.0;
            for (int j = blc1_; j <= trc1_; ++j)
                for (int i = blc0_; i <= trc0_; ++i) rms += dirtyAt(i, j) * dirtyAt(i, j);
            return rms / num;
        };
        initRMSResidual = rmsResidual();

        Mat itsScale(nx_, ny_);

        for (int ii = startingIter_; ii < maxNiter_; ++ii) {
            ++iteration_;
            const double rms = rmsResidual();

            // make the optimized scale image for the current optimum (a delta
            // while in Hogbom mode). The same scale is used for the residual
            // update below, even if the Hogbom switch flips in between, so the
            // model and residual updates always describe the same component.
            const double usedScale = switchedToHogbom_ ? 0.0 : optimumScaleSize_;
            makeScaleImage(itsScale, usedScale, posOptI_, posOptJ_);

            // hogbom-switch heuristics (norm method 1 only)
            if (normMethod_ == 1) {
                if (!switchedToHogbom_ &&
                    (std::abs(peakResidual_) < fusedThreshold_ ||
                     ((std::abs(strengthOptimum_) < (5e-4 * fusedThreshold_)) &&
                      (numNoChange_ >= 2)))) {
                    bool runlong = (initRMSResidual > rms && initRMSResidual / rms < 1.5);
                    switchedToHogbom(runlong);
                    if (numNoChange_ >= 2) numNoChange_ = 0;
                }
                if (!switchedToHogbom_ && numNoChange_ >= 2) {
                    numNoChange_ = 0;
                    bool runlong = (initRMSResidual > rms && initRMSResidual / rms < 1.5);
                    switchedToHogbom(runlong);
                }
            }

            if (!switchedToHogbom_) {
                if (numIterNoGoodAspen_.size() >= 10) numIterNoGoodAspen_.pop_front();
                numIterNoGoodAspen_.push_back(optimumScaleSize_ == 0 ? 1 : 0);
            }

            totalFlux += (strengthOptimum_ * gain_);
            totalFlux_ = totalFlux;

            if (ii == startingIter_) {
                maximumResidual_ = std::abs(peakResidual_);
                tmpMaximumResidual = maximumResidual_;
            }
            if (std::abs(minMaximumResidual) > std::abs(peakResidual_))
                minMaximumResidual = std::abs(peakResidual_);

            // 0. below cycle threshold
            if (std::abs(peakResidual_) < computeThreshold()) {
                converged = 1;
                switchedToHogbom_ = false;
                break;
            }
            // 1. optimum strength below threshold
            if (std::abs(strengthOptimum_) < (1e-6 * fusedThreshold_)) {
                converged = 1;
                switchedToHogbom_ = false;
                break;
            }
            // 2. negative on largest scale
            if ((nInitScales_ > 1) && stopAtLargeScaleNegative_ &&
                optimumScale_ == (nInitScales_ - 1) && strengthOptimum_ < 0.0) {
                converged = -2;
                break;
            }
            // 3. stop-point mode
            if (stopPointMode_ > 0) {
                stopPointModeCounter = (optimumScale_ == 0) ? stopPointModeCounter + 1 : 0;
                if (stopPointModeCounter >= stopPointMode_) {
                    didStopPointMode_ = true;
                    converged = -1;
                    break;
                }
            }
            // PSF (*) scale, centred at the optimum position (the scale image is
            // separable, so its spectrum needs no 2-D forward transform). Needed
            // both for the divergence test and for the residual update.
            std::vector<cd> scaleXfr = scale_image_spectrum(nx_, ny_, usedScale, posOptI_, posOptJ_);
            std::vector<double> psfConvScale;
            conv_from_spectra(psfXfr_, scaleXfr, nx_, ny_, psfConvScale);

            // 5. diverging. CASA compares |strengthOptimum| itself with the
            // initial peak residual, but the strength multiplies a scale image
            // normalised to 1 / (sqrt(2 pi) scale), so it is not in Jy/beam:
            // for an Aspen much wider than the PSF the strength legitimately
            // exceeds the peak by the ratio scale / (2.5 psf_sigma^2) and the
            // CASA test aborts the minor cycle on perfectly good components
            // (any Aspen wider than ~4 beams on a 5-pixel beam). The peak of
            // the fitted component's dirty-image response, |strength| *
            // max|psf (*) scale|, is the unit-consistent quantity the test is
            // meant to bound; the residual-based parts are unchanged.
            double componentPeak = 0.0;
            for (double v : psfConvScale) componentPeak = std::max(componentPeak, std::fabs(v));
            componentPeak *= std::abs(strengthOptimum_);
            if ((componentPeak - std::abs(tmpMaximumResidual)) >
                    (std::abs(tmpMaximumResidual) / 2.0) ||
                (std::abs(peakResidual_) - std::abs(tmpMaximumResidual)) >
                    (std::abs(tmpMaximumResidual) / 2.0) ||
                (std::abs(peakResidual_) - std::abs(minMaximumResidual)) >
                    (std::abs(minMaximumResidual) / 2.0)) {
                converged = -3;
                switchedToHogbom_ = false;
                break;
            }

            // --- update model and residual with the optimum scale ---
            // CASA restricts both updates to a PSF-sized box around the
            // component, which with a full-image PSF only ever *cuts off*
            // part of an off-centre component's response (psf (*) scale is
            // computed on the whole image anyway) and breaks the invariant
            // residual == dirty - model (*) psf near the edges. The update is
            // applied to the whole image instead, at no extra cost.
            const double scaleFactor = gain_ * strengthOptimum_;
            const int blc0 = 0, blc1 = 0, trc0 = nx_ - 1, trc1 = ny_ - 1;

            for (int j = blc1; j <= trc1; ++j)
                for (int i = blc0; i <= trc0; ++i) {
                    modelAdd(i, j, scaleFactor * itsScale(i, j));
                    const std::size_t off = static_cast<std::size_t>(j) * nx_ + i;
                    dirtySet(i, j, dirtyAt(i, j) - scaleFactor * psfConvScale[off]);
                }

            // update peak residual
            prevPeakResidual_ = peakResidual_;
            peakResidual_ = peakResidualMasked();
            if (!switchedToHogbom_ && std::fabs(peakResidual_ - prevPeakResidual_) < 1e-4)
                numNoChange_ += 1;

            // Hogbom bookkeeping: count down, then return to Asp.
            if (switchedToHogbom_) {
                if (numHogbomIter_ == 0) {
                    switchedToHogbom_ = false;
                    if (!(initRMSResidual > rms && initRMSResidual / rms < 1.5)) {
                        converged = 1;
                        break;
                    }
                } else {
                    numHogbomIter_ -= 1;
                }
            }

            // find the next Aspen
            std::vector<double> tempScaleSizes = getActiveSetAspen();
            tempScaleSizes.push_back(0.0);
            defineAspScales(tempScaleSizes);
        }

        if (!converged && verbose_)
            std::fprintf(stderr, "[aspclean] failed to reach stopping threshold\n");
        return converged;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

void convolve_centered(const double* a, const double* b, int nx, int ny,
                       double* out) {
    std::vector<cd> fa = aspfft::rfft2_forward(a, nx, ny);
    std::vector<cd> fb = aspfft::rfft2_forward(b, nx, ny);
    std::vector<double> res;
    conv_from_spectra(fa, fb, nx, ny, res);
    std::copy(res.begin(), res.end(), out);
}

double psf_gaussian_width(const double* psf, int nx, int ny) {
    // Locate the PSF peak (expected near the centre, value ~ 1).
    int px = nx / 2, py = ny / 2;
    double peak = psf[static_cast<std::size_t>(py) * nx + px];
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double v = psf[static_cast<std::size_t>(j) * nx + i];
            if (v > peak) { peak = v; px = i; py = j; }
        }
    if (peak <= 0.0) return std::max(1.0, 2.5 * std::abs(psf[0]) + 1.0);

    // Quadratic-in-log least-squares fit of the main lobe:
    //   ln(psf/peak) ~ a0 + a1 dx + a2 dy + a3 dx^2 + a4 dx dy + a5 dy^2
    const double cutoff = 0.35;  // CASA psfcutoff
    const int win = std::max(4, std::min(nx, ny) / 4);
    double A[6][7] = {{0}};  // augmented normal equations
    int npts = 0;
    for (int dj = -win; dj <= win; ++dj) {
        const int j = py + dj;
        if (j < 0 || j >= ny) continue;
        for (int di = -win; di <= win; ++di) {
            const int i = px + di;
            if (i < 0 || i >= nx) continue;
            const double val = psf[static_cast<std::size_t>(j) * nx + i] / peak;
            if (val <= cutoff) continue;
            const double dx = di, dy = dj;
            const double basis[6] = {1.0, dx, dy, dx * dx, dx * dy, dy * dy};
            const double lz = std::log(val);
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 6; ++c) A[r][c] += basis[r] * basis[c];
                A[r][6] += basis[r] * lz;
            }
            ++npts;
        }
    }

    auto fwhm_fallback = [&]() {
        // Count main-lobe pixels above half maximum along the peak row/column.
        int wx = 1, wy = 1;
        for (int i = px + 1; i < nx && psf[static_cast<std::size_t>(py) * nx + i] / peak > 0.5; ++i) ++wx;
        for (int i = px - 1; i >= 0 && psf[static_cast<std::size_t>(py) * nx + i] / peak > 0.5; --i) ++wx;
        for (int j = py + 1; j < ny && psf[static_cast<std::size_t>(j) * nx + px] / peak > 0.5; ++j) ++wy;
        for (int j = py - 1; j >= 0 && psf[static_cast<std::size_t>(j) * nx + px] / peak > 0.5; --j) ++wy;
        return std::ceil((wx + wy) / 2.0);
    };

    if (npts < 6) return std::max(1.0, fwhm_fallback());

    // Gaussian elimination with partial pivoting on the 6x6 system.
    for (int col = 0; col < 6; ++col) {
        int piv = col;
        for (int r = col + 1; r < 6; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-12) return std::max(1.0, fwhm_fallback());
        for (int c = 0; c < 7; ++c) std::swap(A[col][c], A[piv][c]);
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / A[col][col];
            for (int c = col; c < 7; ++c) A[r][c] -= f * A[col][c];
        }
    }
    const double a3 = A[3][6] / A[3][3];
    const double a4 = A[4][6] / A[4][4];
    const double a5 = A[5][6] / A[5][5];

    // psf ~ exp(a3 dx^2 + a4 dx dy + a5 dy^2) = exp(-(M)) with
    // M = [[-a3, -a4/2], [-a4/2, -a5]] positive definite for a peak.
    const double Mxx = -a3, Myy = -a5, Mxy = -a4 / 2.0;
    const double tr = Mxx + Myy;
    const double det = Mxx * Myy - Mxy * Mxy;
    const double disc = std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
    const double mu1 = tr / 2.0 + disc;
    const double mu2 = tr / 2.0 - disc;
    if (!(mu1 > 0.0 && mu2 > 0.0)) return std::max(1.0, fwhm_fallback());

    // sigma_i = 1/sqrt(2 mu_i); FWHM_i = 2 sqrt(2 ln2) sigma_i.
    const double k = 2.0 * std::sqrt(2.0 * std::log(2.0));
    const double fwhm1 = k / std::sqrt(2.0 * mu1);
    const double fwhm2 = k / std::sqrt(2.0 * mu2);
    return std::max(1.0, std::ceil((fwhm1 + fwhm2) / 2.0));
}

// ---------------------------------------------------------------------------
// Plane / cube drivers
// ---------------------------------------------------------------------------

template <typename T>
AspResult aspclean_plane(T* residual, T* model, const T* psf, const T* mask,
                         int nx, int ny, double gain, double threshold,
                         int niter, double fusedthreshold, double psf_width,
                         int largestscale, int stoppointmode, int norm_method,
                         bool verbose) {
    Cleaner<T> cleaner(residual, model, psf, mask, nx, ny, gain, threshold,
                       niter, fusedthreshold, psf_width, largestscale,
                       stoppointmode, norm_method, verbose);
    return cleaner.run();
}

template <typename T>
void aspclean_cube(T* residual, T* model, const T* psf, const T* mask, int nt,
                   int nf, int np_img, int np_psf, int nx, int ny, double gain,
                   const double* threshold, const int* niter, double fusedthreshold,
                   double psf_width, int largestscale, int stoppointmode,
                   int norm_method, int processing_function_threads, AspResult* results) {
    const int nplanes = nt * nf * np_img;
    if (nplanes <= 0) return;
    const std::size_t plane = static_cast<std::size_t>(ny) * nx;

    std::atomic<int> next(0);
    auto worker = [&]() {
        int p;
        while ((p = next.fetch_add(1)) < nplanes) {
            const int tt = p / (nf * np_img);
            const int nn = (p / np_img) % nf;
            const int pp = p % np_img;
            const int pidx = (np_psf == 1) ? 0 : pp;
            const std::size_t img_off =
                ((static_cast<std::size_t>(tt) * nf + nn) * np_img + pp) * plane;
            const std::size_t psf_off =
                ((static_cast<std::size_t>(tt) * nf + nn) * np_psf + pidx) * plane;
            const T* maskp = mask ? mask + img_off : nullptr;
            // Iteration control is independent per plane: each (t, f, p)
            // plane uses its own threshold and iteration limit.
            results[p] = aspclean_plane<T>(residual + img_off, model + img_off,
                                           psf + psf_off, maskp, nx, ny, gain,
                                           threshold[p], niter[p], fusedthreshold,
                                           psf_width, largestscale, stoppointmode,
                                           norm_method, false);
        }
    };

    int nthreads = std::max(1, std::min(processing_function_threads, nplanes));
    if (nthreads == 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int i = 0; i < nthreads; ++i) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
    }
}

// Explicit instantiations for float and double image buffers.
template AspResult aspclean_plane<float>(float*, float*, const float*, const float*,
                                         int, int, double, double, int, double,
                                         double, int, int, int, bool);
template AspResult aspclean_plane<double>(double*, double*, const double*, const double*,
                                          int, int, double, double, int, double,
                                          double, int, int, int, bool);
template void aspclean_cube<float>(float*, float*, const float*, const float*, int,
                                   int, int, int, int, int, double, const double*,
                                   const int*, double, double, int, int, int, int,
                                   AspResult*);
template void aspclean_cube<double>(double*, double*, const double*, const double*,
                                    int, int, int, int, int, int, double,
                                    const double*, const int*, double, double, int,
                                    int, int, int, AspResult*);

}  // namespace aspclean
