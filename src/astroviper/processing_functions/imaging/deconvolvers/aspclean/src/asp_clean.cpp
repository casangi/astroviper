// Implementation of the Adaptive Scale Pixel (Asp) deconvolver. Ported from
// CASA synthesis/MeasurementEquations/AspMatrixCleaner.cc (the active "gold"
// objective path) and synthesis/ImagerObjects/SDAlgorithmAAspClean.cc.
//
// casacore types are replaced with the small `Mat` double-precision matrix
// below and raw pointers into the caller's (Python-owned) buffers; FFTServer
// is replaced by asp_fft.hpp (pocketfft) and ALGLIB by asp_lbfgs.hpp.
//
// The algorithm is CASA's, but the convolutions are not computed the same way:
//  * CASA convolves on the image grid with circular FFTs, so the PSF response
//    of a component near an edge wraps around to the opposite edge. Here the
//    residual update subtracts the exact linear convolution with the
//    image-sized PSF (on a grid of 1.5 times the image), and the L-BFGS
//    objective convolves the Aspen's box with the PSF exactly on a grid of
//    two box sizes (or on the same padded grid, whichever is smaller).
//  * The Aspen and the scale images are separable, so their spectra are outer
//    products of 1-D transforms; every image is real, so every 2-D transform
//    is a real one on half spectra, and the work buffers are reused instead of
//    allocated per transform.
//  * Point components (scale 0, and every Hogbom iteration) need no transform:
//    the residual loses the shifted PSF directly.
//  * The peak searches scan only the box where their mask can be nonzero.
// None of this changes what is computed beyond rounding.

#include "../include/asp_clean.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "../include/asp_fft.hpp"
#include "../include/asp_lbfgs.hpp"

namespace aspclean {

namespace {

using aspfft::cd;

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kSqrtTwoPi = 2.5066282746310002;
// A pixel belongs to the search region where the mask is above this value.
constexpr double kMaskThreshold = 0.99;

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

// Inclusive pixel box [i0, i1] x [j0, j1]; empty when i1 < i0.
struct Box {
    int i0 = 0, j0 = 0, i1 = -1, j1 = -1;
};

// findMaxAbsMask (casacore minMaxMasked, then the larger in absolute value,
// signed value returned) of (array * weight), for an [ny][nx] array `a`
// (of the residual's type, or double) whose product with the weight is zero
// outside `box`: the same value and position as the scan of the whole image,
// from the box alone. The whole-image scan starts from pixel (0, 0) and keeps
// the first pixel (in scan order) that reaches the extreme value; a nonzero
// extreme lies in the box, and a zero one is reported at (0, 0) as the
// whole-image scan does whenever it matters (both extremes zero). The cleaner
// always runs with a mask present (CASA's driver always supplies a mask
// image, all-ones when "clean everywhere"), so only the masked finder is
// needed.
template <typename V>
void find_max_abs_mask_box(const V* a, const Mat& w, const Box& box, double& maxAbs,
                           int& posI, int& posJ) {
    const int nx = w.nx, ny = w.ny;
    double mn = 0.0, mx = 0.0;
    int mnI = 0, mnJ = 0, mxI = 0, mxJ = 0;
    if (box.i0 == 0 && box.j0 == 0 && box.i1 == nx - 1 && box.j1 == ny - 1)
        mn = mx = static_cast<double>(a[0]) * w(0, 0);
    for (int j = box.j0; j <= box.j1; ++j) {
        const V* arow = a + static_cast<std::size_t>(j) * nx;
        const double* wrow = w.d.data() + static_cast<std::size_t>(j) * nx;
        for (int i = box.i0; i <= box.i1; ++i) {
            const double v = static_cast<double>(arow[i]) * wrow[i];
            if (v > mx) { mx = v; mxI = i; mxJ = j; }
            if (v < mn) { mn = v; mnI = i; mnJ = j; }
        }
    }
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

// Seconds since `start`; `start` is reset to now.
double lap(std::chrono::steady_clock::time_point& start) {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - start).count();
    start = now;
    return dt;
}

// Smallest FFT length >= n that pocketfft transforms fast (2, 3, 5, 7, 11).
int good_fft_size(int n) {
    return static_cast<int>(pocketfft::detail::util::good_size_cmplx(static_cast<std::size_t>(std::max(n, 1))));
}

// Threads of the 2-D transforms of a plane: `total` shared evenly by the
// planes still being cleaned (`active`; none for a plane cleaned alone), so
// the planes that run longest get the threads of those that have finished.
struct FftThreads {
    int total = 1;
    const std::atomic<int>* active = nullptr;

    std::size_t get() const {
        const int n = active ? std::max(1, active->load(std::memory_order_relaxed)) : 1;
        return static_cast<std::size_t>(std::max(1, total / n));
    }
};

// Half spectrum of the PSF cutout the objective's local convolution needs, on
// an lx x ly grid: the PSF at offsets |d| <= (lx - 1) / 2 along x (and
// (ly - 1) / 2 along y) from its centre (nx/2, ny/2), zero beyond the PSF
// image, offset d at grid index d mod lx. An Aspen box of bx <= (lx + 1) / 2
// pixels at grid indices [0, bx) then convolves with the cutout without
// wrap-around onto [0, bx): every product lands at an index in
// [-(lx - 1) / 2, bx - 1 + (lx - 1) / 2], which is never [0, bx) shifted by a
// multiple of lx. The cutout depends on the grid only, so one spectrum serves
// every box size (and Aspen scale) that maps to the same FFT length.
struct PsfCutoutSpectra {
    const double* psf;
    int nx, ny;
    const FftThreads* threads;
    std::vector<std::pair<std::pair<int, int>, std::vector<cd>>> cache;

    const std::vector<cd>& get(int lx, int ly) {
        for (auto& entry : cache)
            if (entry.first.first == lx && entry.first.second == ly) return entry.second;
        std::vector<double> grid(static_cast<std::size_t>(lx) * ly, 0.0);
        const int cx = nx / 2, cy = ny / 2;
        const int hx = (lx - 1) / 2, hy = (ly - 1) / 2;
        for (int dj = -hy; dj <= hy; ++dj) {
            const int pj = cy + dj;
            if (pj < 0 || pj >= ny) continue;
            const std::size_t row = static_cast<std::size_t>((dj + ly) % ly) * lx;
            for (int di = -hx; di <= hx; ++di) {
                const int pi = cx + di;
                if (pi < 0 || pi >= nx) continue;
                grid[row + static_cast<std::size_t>((di + lx) % lx)] =
                    psf[static_cast<std::size_t>(pj) * nx + pi];
            }
        }
        if (cache.size() >= 8) cache.erase(cache.begin());
        cache.emplace_back(std::make_pair(lx, ly), aspfft::rfft2_half(grid.data(), lx, ly, threads->get()));
        return cache.back().second;
    }
};

// Work buffers reused across iterations and objective evaluations: a large
// std::vector gets fresh pages from the operating system on every
// allocation, and faulting them in costs about as much as the transform.
struct Workspace {
    std::vector<cd> specA, specD;
    std::vector<double> convA, convD;
};

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
// come from four 1-D transforms, and each convolution is one inverse real
// transform of a half spectrum.
// ---------------------------------------------------------------------------
template <typename T>
struct AspObjective {
    const T* dirty;                // current residual, [ny][nx]
    PsfCutoutSpectra& psfCutouts;  // PSF cutout spectra, cached per grid
    const std::vector<cd>& psfXfrPad;  // PSF half spectrum on the padded image grid, centre at 0
    int padNx, padNy;
    int nx, ny;
    int cx, cy;                    // Aspen centre (positionOptimum)
    double inv_rescale;            // 1 / itsRescale
    const FftThreads& threads;     // FFT threads
    Workspace& ws;

    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        const double amp = x[0];
        double asc = std::fabs(x[1]);
        if (!std::isfinite(amp) || asc < 1e-3) asc = 1e-3;  // keep finite

        const double sigma5 = 5.0 * asc / 2.0;
        const int minI = std::max(0, static_cast<int>(cx - sigma5));
        const int maxI = std::min(nx - 1, static_cast<int>(cx + sigma5));
        const int minJ = std::max(0, static_cast<int>(cy - sigma5));
        const int maxJ = std::min(ny - 1, static_cast<int>(cy + sigma5));
        const int bx = maxI - minI + 1, by = maxJ - minJ + 1;

        // The objective needs (Aspen (*) psf) only on the box, and the Aspen
        // is zero outside it, so the convolution is the exact linear one of
        // the box with the PSF cutout, computed on a grid of at least two box
        // sizes - 1 (see PsfCutoutSpectra). CASA convolves on the whole image
        // with a circular FFT, which for an Aspen near an edge folds in the
        // PSF response from the opposite side.
        int lx = good_fft_size(2 * bx - 1), ly = good_fft_size(2 * by - 1);
        // For a wide Aspen the padded image grid of the residual update (no
        // wrap-around inside the image either) is the smaller transform.
        const bool padded = static_cast<double>(lx) * ly >= static_cast<double>(padNx) * padNy;
        // Grid index of image pixel (i, j): (i + ox, j + oy); box pixel k of
        // the profiles sits at k on the local grid and at minI + k (minJ + k)
        // on the padded one.
        int ox = -minI, oy = -minJ, px = 0, py = 0;
        if (padded) {
            lx = padNx;
            ly = padNy;
            ox = 0;
            oy = 0;
            px = minI;
            py = minJ;
        }
        const std::vector<cd>& psfFT = padded ? psfXfrPad : psfCutouts.get(lx, ly);
        const int hx = lx / 2 + 1;

        // Separable factors on the truncation box: g(d) = exp(-d^2 / 2 s^2)
        // and h(d) = (d^2 / s^2) g(d), so that
        //   Asp  = norm * g(di) g(dj)
        //   dAsp = Asp * ((di^2 + dj^2) / s^2 - 1) / s
        //        = (norm / s) * [h(di) g(dj) + g(di) h(dj) - g(di) g(dj)].
        const double inv_s2 = 1.0 / (asc * asc);
        const double inv2s2 = 0.5 * inv_s2;
        std::vector<double> gx(lx, 0.0), gy(ly, 0.0), hxv(lx, 0.0), hy(ly, 0.0);
        for (int i = minI; i <= maxI; ++i) {
            const double d = static_cast<double>(i - cx);
            const int k = px + i - minI;
            gx[k] = std::exp(-d * d * inv2s2);
            hxv[k] = d * d * inv_s2 * gx[k];
        }
        for (int j = minJ; j <= maxJ; ++j) {
            const double d = static_cast<double>(j - cy);
            const int k = py + j - minJ;
            gy[k] = std::exp(-d * d * inv2s2);
            hy[k] = d * d * inv_s2 * gy[k];
        }
        const std::vector<cd> Gx = aspfft::rfft1d_half(gx);
        const std::vector<cd> Hx = aspfft::rfft1d_half(hxv);
        const std::vector<cd> Gy = aspfft::fft1d_forward(gy);
        const std::vector<cd> Hy = aspfft::fft1d_forward(hy);

        const double norm = 1.0 / (kSqrtTwoPi * asc);
        const double dnorm = norm / asc;
        std::vector<cd>& specA = ws.specA;
        std::vector<cd>& specD = ws.specD;
        specA.resize(static_cast<std::size_t>(hx) * ly);
        specD.resize(specA.size());
        for (int ky = 0; ky < ly; ++ky) {
            const cd gyk = Gy[static_cast<std::size_t>(ky)];
            const cd hyk = Hy[static_cast<std::size_t>(ky)];
            const std::size_t base = static_cast<std::size_t>(ky) * hx;
            for (int kx = 0; kx < hx; ++kx) {
                const cd gxk = Gx[static_cast<std::size_t>(kx)];
                const cd hxk = Hx[static_cast<std::size_t>(kx)];
                const cd p = psfFT[base + kx];
                const cd gg = aspfft::mul(gxk, gyk);
                specA[base + kx] = aspfft::mul(gg, p) * norm;
                specD[base + kx] =
                    aspfft::mul(aspfft::mul(hxk, gyk) + aspfft::mul(gxk, hyk) - gg, p) * dnorm;
            }
        }
        std::vector<double>& convA = ws.convA;
        std::vector<double>& convD = ws.convD;
        convA.resize(static_cast<std::size_t>(lx) * ly);
        convD.resize(convA.size());
        const std::size_t nthreads = threads.get();
        aspfft::irfft2_half(specA.data(), lx, ly, convA.data(), nthreads);
        aspfft::irfft2_half(specD.data(), lx, ly, convD.data(), nthreads);

        double func = 0.0, dA = 0.0, dS = 0.0;
        for (int j = minJ; j < maxJ; ++j) {
            const std::size_t row = static_cast<std::size_t>(j + oy) * lx;
            const T* drow = dirty + static_cast<std::size_t>(j) * nx;
            for (int i = minI; i < maxI; ++i) {
                const std::size_t k = row + static_cast<std::size_t>(i + ox);
                const double aspConvPsf = convA[k];
                const double dAspConvPsf = convD[k];
                const double res = static_cast<double>(drow[i]) * inv_rescale - amp * aspConvPsf;
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
            double gain, double threshold, int max_iter_per_cycle_updated, double fusedthreshold,
            double psf_width, int largestscale, int stoppointmode,
            int norm_method, bool stop_after_hogbom_run, bool verbose,
            const deconvolvers::DivergenceControl<double>& divergence, const FftThreads& fft_threads)
        : residual_(residual), model_(model), psf_(psf), mask_(mask),
          nx_(nx), ny_(ny), gain_(gain), threshold_(threshold),
          maxNiter_(max_iter_per_cycle_updated), fusedThreshold_(fusedthreshold),
          userLargestScale_(static_cast<double>(largestscale)),
          stopPointMode_(stoppointmode),
          normMethod_(norm_method <= 0 ? 1 : norm_method),
          stopAfterHogbomRun_(stop_after_hogbom_run), verbose_(verbose),
          psfWidthInput_(psf_width), divergence_(divergence),
          fftThreads_(fft_threads) {}

    AspResult run() {
        auto t0 = std::chrono::steady_clock::now();
        setup();
        // The first search and fit run inside setup(); they are counted there.
        timing_.t_setup = lap(t0) - timing_.t_search - timing_.t_fit;
        AspResult res = timing_;
        if (noClean_) {  // mask forbids cleaning everywhere
            res.psf_width = psfWidth_;
            res.peak_residual = 0.0;
            res.model_flux = modelSum();
            return res;
        }
        const int retval = aspclean();
        const double t_setup = res.t_setup;
        res = timing_;
        res.t_setup = t_setup;
        res.retval = retval;
        res.stop_code = stopCode_;
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
    bool stopAfterHogbomRun_;  // CASA's end of the model update after a Hogbom run
    bool verbose_;
    double psfWidthInput_;
    deconvolvers::DivergenceControl<double> divergence_;
    FftThreads fftThreads_;  // threads of every 2-D transform

    // --- derived / state ---
    double psfWidth_ = 0.0;
    std::vector<double> psfd_;               // the PSF in double precision
    int padNx_ = 0, padNy_ = 0;              // grid of the exact residual update
    std::vector<cd> psfXfrPad_;              // PSF half spectrum on that grid, centre at 0
    PsfCutoutSpectra psfCutouts_{nullptr, 0, 0, nullptr, {}};
    std::vector<double> initScaleSizes_;     // itsInitScaleSizes
    int nInitScales_ = 5;
    // itsInitScaleXfrs, kept separable: the spectrum of initial scale s is
    // initScaleXfrX_[s](kx) * initScaleXfrY_[s](ky) (half spectrum along x).
    std::vector<std::vector<cd>> initScaleXfrX_, initScaleXfrY_;
    std::vector<Mat> initScaleMasks_;        // itsInitScaleMasks
    std::vector<Box> initScaleMaskBoxes_;    // where each scale mask can be nonzero
    std::vector<Mat> dirtyConvInitScales_;   // itsDirtyConvInitScales (scale 0: the residual)
    Workspace work_;                         // reused transform buffers
    std::vector<cd> dirtyFT_;                // residual half spectrum of the search
    Mat maskMat_;                            // itsMask (always present)
    double maskThreshold_ = kMaskThreshold;
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
    int stopCode_ = deconvolvers::STOP_NONE;  // AspResult::stop_code
    double maximumResidual_ = 0.0, totalFlux_ = 0.0;
    std::vector<double> goodAspActiveSet_, goodAspAmplitude_;
    deconvolvers::PlaneStopTest<double> stopTest_;
    double sumSquares_ = 0.0;  // of the masked residual, set with peakResidual_
    double bboxSumSquares_ = 0.0;  // of the residual over [blc, trc], set with peakResidual_
    AspResult timing_;         // section timings and counts

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
        // setPsf. The residual update subtracts the exact linear convolution
        // of a component with the image-sized PSF (the response the Hogbom
        // kernels subtract): on a grid of at least 1.5 times the image along
        // each axis the circular FFT convolution of the image-sized arrays has
        // no wrap-around inside the image. CASA convolves on the image grid,
        // where the response of an off-centre component wraps around to the
        // opposite edge, and cuts most of it off with a box around the
        // component.
        psfd_.resize(static_cast<std::size_t>(nx_) * ny_);
        for (std::size_t k = 0; k < psfd_.size(); ++k) psfd_[k] = static_cast<double>(psf_[k]);
        padNx_ = good_fft_size((3 * nx_ + 1) / 2);
        padNy_ = good_fft_size((3 * ny_ + 1) / 2);
        {
            // PSF centre (nx/2, ny/2) at the origin of the padded grid.
            std::vector<double> grid(static_cast<std::size_t>(padNx_) * padNy_, 0.0);
            for (int j = 0; j < ny_; ++j) {
                const int pj = ((j - ny_ / 2) % padNy_ + padNy_) % padNy_;
                for (int i = 0; i < nx_; ++i) {
                    const int pi = ((i - nx_ / 2) % padNx_ + padNx_) % padNx_;
                    grid[static_cast<std::size_t>(pj) * padNx_ + pi] =
                        psfd_[static_cast<std::size_t>(j) * nx_ + i];
                }
            }
            psfXfrPad_ = aspfft::rfft2_half(grid.data(), padNx_, padNy_, fftThreads_.get());
        }
        psfCutouts_ = PsfCutoutSpectra{psfd_.data(), nx_, ny_, &fftThreads_, {}};

        // getPsfGaussianWidth (or use the supplied width).
        if (psfWidthInput_ > 0.0) {
            psfWidth_ = psfWidthInput_;
        } else {
            psfWidth_ = psf_gaussian_width(psfd_.data(), nx_, ny_);
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

    // itsInitScaleXfrs. CASA's initial scale image is the Gaussian centred on
    // (nx/2, ny/2) over the whole image (a delta for size 0), convolved
    // circularly with the residual and recentred. Here the same Gaussian is
    // centred on the origin with wrapped distances, so the circular
    // convolution needs no recentring, and as it is separable its spectrum is
    // the outer product of two 1-D transforms. The delta (scale 0) is never
    // transformed: convolving with it is the identity.
    void setInitScaleXfrs() {
        nInitScales_ = 5;
        setInitScales();  // may reset nInitScales_
        initScaleXfrX_.assign(nInitScales_, {});
        initScaleXfrY_.assign(nInitScales_, {});
        auto wrapped_profile = [](int n, double sigma) {
            // distance of pixel q from the origin: q - n * (q >= n - n / 2)
            std::vector<double> row(static_cast<std::size_t>(n));
            const double inv2s2 = 0.5 / (sigma * sigma);
            for (int q = 0; q < n; ++q) {
                const double d = static_cast<double>(q < n - n / 2 ? q : q - n);
                row[static_cast<std::size_t>(q)] = std::exp(-d * d * inv2s2);
            }
            return row;
        };
        for (int s = 1; s < nInitScales_; ++s) {
            const double size = initScaleSizes_[s];
            initScaleXfrX_[s] = aspfft::rfft1d_half(wrapped_profile(nx_, size));
            initScaleXfrY_[s] = aspfft::fft1d_forward(wrapped_profile(ny_, size));
            const double norm = 1.0 / (kSqrtTwoPi * size);
            for (cd& v : initScaleXfrY_[s]) v *= norm;
        }
    }

    // ifft2(spec * scale_s spectrum) into `out` (the circular convolution of
    // the image whose half spectrum is `spec` with initial scale s).
    void convolveInitScale(const std::vector<cd>& spec, int s, std::vector<double>& out) {
        const int hx = nx_ / 2 + 1;
        std::vector<cd>& prod = work_.specA;
        prod.resize(spec.size());
        const std::vector<cd>& X = initScaleXfrX_[s];
        const std::vector<cd>& Y = initScaleXfrY_[s];
        for (int ky = 0; ky < ny_; ++ky) {
            const cd yk = Y[static_cast<std::size_t>(ky)];
            const std::size_t base = static_cast<std::size_t>(ky) * hx;
            for (int kx = 0; kx < hx; ++kx)
                prod[base + kx] = aspfft::mul(spec[base + kx], aspfft::mul(X[static_cast<std::size_t>(kx)], yk));
        }
        out.resize(static_cast<std::size_t>(nx_) * ny_);
        aspfft::irfft2_half(prod.data(), nx_, ny_, out.data(), fftThreads_.get());
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

        std::vector<cd> maskFT;
        if (nInitScales_ > 1) maskFT = aspfft::rfft2_half(maskMat_.d.data(), nx_, ny_, fftThreads_.get());
        initScaleMasks_.assign(nInitScales_, Mat(nx_, ny_));
        for (int s = 0; s < nInitScales_; ++s) {
            initScaleMasks_[s] = Mat(nx_, ny_);
            std::vector<double> conv;
            if (s == 0)
                conv = maskMat_.d;  // mask (*) delta
            else
                convolveInitScale(maskFT, s, conv);
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

        // The peak search of scale s looks at the residual (*) scale s inside
        // [blc, trc] weighted by the scale mask, and zero elsewhere: only the
        // part of the box where the scale mask is nonzero can hold a nonzero
        // value.
        initScaleMaskBoxes_.assign(nInitScales_, Box{});
        for (int s = 0; s < nInitScales_; ++s) {
            Box box{nx_, ny_, -1, -1};
            for (int j = blc1_; j <= trc1_; ++j)
                for (int i = blc0_; i <= trc0_; ++i)
                    if (initScaleMasks_[s](i, j) != 0.0) {
                        box.i0 = std::min(box.i0, i); box.j0 = std::min(box.j0, j);
                        box.i1 = std::max(box.i1, i); box.j1 = std::max(box.j1, j);
                    }
            initScaleMaskBoxes_[s] = box;
        }
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

        for (int s = 0; s < nInitScales_; ++s) {
            // CASA searches a copy of dirty (*) scale s that is zero outside
            // [blc, trc]; the scale mask box lies inside [blc, trc].
            // Scale 0 is the residual itself.
            double mx; int pi, pj;
            if (s == 0)
                find_max_abs_mask_box(residual_, initScaleMasks_[0], initScaleMaskBoxes_[0], mx, pi, pj);
            else
                find_max_abs_mask_box(dirtyConvInitScales_[s].d.data(), initScaleMasks_[s],
                                      initScaleMaskBoxes_[s], mx, pi, pj);
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
        auto clock = std::chrono::steady_clock::now();

        // Dirty (*) initial scales. Scale 0 is a delta: dirty (*) scale_0 is
        // the residual itself and needs no transform, nor a copy (in Hogbom
        // mode it is the only scale, so a Hogbom iteration does no FFT at
        // all). The others are convolved from the residual's spectrum.
        if (nInitScales_ > 1) {
            if (static_cast<int>(dirtyConvInitScales_.size()) < nInitScales_)
                dirtyConvInitScales_.resize(nInitScales_);
            std::vector<double>& dirtyd = work_.convD;
            dirtyd.resize(static_cast<std::size_t>(nx_) * ny_);
            for (std::size_t k = 0; k < dirtyd.size(); ++k) dirtyd[k] = static_cast<double>(residual_[k]);
            aspfft::rfft2_half(dirtyd.data(), nx_, ny_, dirtyFT_, fftThreads_.get());
            for (int s = 1; s < nInitScales_; ++s) {
                dirtyConvInitScales_[s].nx = nx_;
                dirtyConvInitScales_[s].ny = ny_;
                convolveInitScale(dirtyFT_, s, dirtyConvInitScales_[s].d);
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

        timing_.t_search += lap(clock);
        if (optimumScale_ == 0) return {};  // 0 scale needs no optimization

        // CASA: itsRescale = |strengthOptimum / itsPsfWidth|; the optimizer
        // sees the rescaled amplitude x[0] = strength / rescale with variable
        // scales s = (|strength| / rescale, scaleSize) = (psfWidth, scaleSize).
        double rescale = std::fabs(strengthOptimum / psfWidth_);
        if (!(rescale > 0.0) || !std::isfinite(rescale)) rescale = 1.0;
        AspObjective<T> obj{residual_, psfCutouts_, psfXfrPad_, padNx_, padNy_, nx_, ny_,
                            posI, posJ, 1.0 / rescale, fftThreads_, work_};
        std::vector<double> x = {strengthOptimum / rescale, initScaleSizes_[optimumScale]};
        std::vector<double> sc = {std::fabs(strengthOptimum) / rescale, initScaleSizes_[optimumScale]};
        asplbfgs::Options opt;  // m=1, maxits=5, eps*=1e-3 -> matches ALGLIB call
        const asplbfgs::Report fitReport = asplbfgs::minimize(
            x, sc, [&obj](const std::vector<double>& xx, std::vector<double>& gg) { return obj(xx, gg); },
            opt);
        timing_.t_fit += lap(clock);
        timing_.n_fits += 1;
        timing_.n_fit_evaluations += fitReport.evaluations;

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

    // Peak |residual| over the scale-0 mask (itsInitScaleMasks[0]); the sum
    // of the squares of the same masked residual goes to sumSquares_. The
    // search starts from zero, so a pixel that is not a number is skipped.
    // The same pass also sums the squares of the residual over [blc, trc]
    // (bboxSumSquares_, for the RMS of the next iteration); the scale-0 mask
    // box lies inside [blc, trc].
    double peakResidualMasked() {
        double minVal = 0.0;
        double maxVal = 0.0;
        sumSquares_ = 0.0;
        bboxSumSquares_ = 0.0;
        // Outside its box the scale-0 mask is zero (and the residual finite).
        const Box& box = initScaleMaskBoxes_[0];
        for (int j = blc1_; j <= trc1_; ++j) {
            const bool inRows = (j >= box.j0 && j <= box.j1);
            for (int i = blc0_; i <= trc0_; ++i) {
                const double d = dirtyAt(i, j);
                bboxSumSquares_ += d * d;
                if (inRows && i >= box.i0 && i <= box.i1) {
                    const double v = d * initScaleMasks_[0](i, j);
                    sumSquares_ += v * v;
                    if (v > maxVal) maxVal = v;
                    if (v < minVal) minVal = v;
                }
            }
        }
        return std::max(std::fabs(maxVal), std::fabs(minVal));
    }

    // -----------------------------------------------------------------------
    // The model update (AspMatrixCleaner::aspclean).
    // -----------------------------------------------------------------------
    int aspclean() {
        double totalFlux = 0.0;
        int converged = 0;
        int stopPointModeCounter = 0;
        double initRMSResidual = 1000.0;

        iteration_ = startingIter_;

        peakResidual_ = peakResidualMasked();

        const int num = std::max(1, (trc0_ - blc0_) * (trc1_ - blc1_));
        // The squares of the residual over [blc, trc] are summed by the last
        // peakResidualMasked(), on the residual the iteration starts from.
        auto rmsResidual = [&]() { return bboxSumSquares_ / num; };
        initRMSResidual = rmsResidual();

        auto clock = std::chrono::steady_clock::now();
        for (int ii = startingIter_; ii < maxNiter_; ++ii) {
            ++iteration_;
            lap(clock);
            const double rms = rmsResidual();

            // The scale of the component of this iteration (a delta while in
            // Hogbom mode). The same scale is used for the residual update
            // below, even if the Hogbom switch flips in between, so the model
            // and residual updates always describe the same component.
            const double usedScale = switchedToHogbom_ ? 0.0 : optimumScaleSize_;
            timing_.t_bookkeeping += lap(clock);

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

            if (ii == startingIter_) maximumResidual_ = std::abs(peakResidual_);

            // 0. below cycle threshold
            if (std::abs(peakResidual_) < computeThreshold()) {
                converged = 1;
                stopCode_ = deconvolvers::STOP_THRESHOLD;
                switchedToHogbom_ = false;
                break;
            }
            // Divergence test shared with the other deconvolvers, on the
            // masked residual of the threshold test. It replaces the
            // "5. Diverging" guard of AspMatrixCleaner in CASA 6.7.5.18 and
            // LibRA (main at fc54ee25, AspMatrixCleaner.cc L515-518). Its fixed
            // factor 1.5 stops healthy Asp model updates, and its running
            // minimum starts at 1000, so a plane with a peak above 1500
            // stopped at once. casa master has dropped the running minimum
            // and keeps the two factor 1.5 tests against the starting peak.
            if (stopTest_.check_divergence(std::abs(peakResidual_), sumSquares_,
                                           divergence_) != deconvolvers::STOP_NONE) {
                converged = -3;
                stopCode_ = deconvolvers::STOP_DIVERGED;
                switchedToHogbom_ = false;
                break;
            }
            // 1. optimum strength below threshold
            if (std::abs(strengthOptimum_) < (1e-6 * fusedThreshold_)) {
                converged = 1;
                stopCode_ = deconvolvers::STOP_ALGORITHM;
                switchedToHogbom_ = false;
                break;
            }
            // 2. negative on largest scale
            if ((nInitScales_ > 1) && stopAtLargeScaleNegative_ &&
                optimumScale_ == (nInitScales_ - 1) && strengthOptimum_ < 0.0) {
                converged = -2;
                stopCode_ = deconvolvers::STOP_ALGORITHM;
                break;
            }
            // 3. stop-point mode
            if (stopPointMode_ > 0) {
                stopPointModeCounter = (optimumScale_ == 0) ? stopPointModeCounter + 1 : 0;
                if (stopPointModeCounter >= stopPointMode_) {
                    didStopPointMode_ = true;
                    converged = -1;
                    stopCode_ = deconvolvers::STOP_ALGORITHM;
                    break;
                }
            }
            // --- update model and residual with the optimum scale ---
            // The residual loses the exact linear convolution of the component
            // with the image-sized PSF (see setup()), so residual == dirty -
            // model (*) psf holds on the whole image, as for the Hogbom kernels.
            const double scaleFactor = gain_ * strengthOptimum_;
            lap(clock);
            if (usedScale == 0.0) {
                // A point component: the shifted PSF, no transform needed.
                modelAdd(posOptI_, posOptJ_, scaleFactor);
                const int di = nx_ / 2 - posOptI_, dj = ny_ / 2 - posOptJ_;
                const int i0 = std::max(0, -di), i1 = std::min(nx_, nx_ - di);
                const int j0 = std::max(0, -dj), j1 = std::min(ny_, ny_ - dj);
                for (int j = j0; j < j1; ++j) {
                    const double* prow = psfd_.data() + static_cast<std::size_t>(j + dj) * nx_;
                    for (int i = i0; i < i1; ++i)
                        dirtySet(i, j, dirtyAt(i, j) - scaleFactor * prow[i + di]);
                }
                timing_.t_update_point += lap(clock);
                timing_.n_point_components += 1;
            } else {
                // The scale image (CASA makeScaleImage: the Gaussian centred on
                // the component over the whole image) is separable: its
                // spectrum on the padded grid is the outer product of two 1-D
                // transforms of its (image-sized) row and column profiles.
                const std::vector<double> rx = gaussian_profile(nx_, posOptI_, usedScale, 0, nx_ - 1);
                const std::vector<double> ry = gaussian_profile(ny_, posOptJ_, usedScale, 0, ny_ - 1);
                std::vector<double> gx(padNx_, 0.0), gy(padNy_, 0.0);
                std::copy(rx.begin(), rx.end(), gx.begin());
                std::copy(ry.begin(), ry.end(), gy.begin());
                const std::vector<cd> X = aspfft::rfft1d_half(gx);
                const std::vector<cd> Y = aspfft::fft1d_forward(gy);
                const double norm = 1.0 / (kSqrtTwoPi * usedScale);
                const int hx = padNx_ / 2 + 1;
                std::vector<cd>& spec = work_.specA;
                spec.resize(static_cast<std::size_t>(hx) * padNy_);
                for (int ky = 0; ky < padNy_; ++ky) {
                    const cd ay = norm * Y[static_cast<std::size_t>(ky)];
                    const std::size_t base = static_cast<std::size_t>(ky) * hx;
                    for (int kx = 0; kx < hx; ++kx)
                        spec[base + kx] = aspfft::mul(aspfft::mul(ay, X[static_cast<std::size_t>(kx)]),
                                                      psfXfrPad_[base + kx]);
                }
                std::vector<double>& conv = work_.convA;
                conv.resize(static_cast<std::size_t>(padNx_) * padNy_);
                aspfft::irfft2_half(spec.data(), padNx_, padNy_, conv.data(), fftThreads_.get());
                for (int j = 0; j < ny_; ++j) {
                    const double gyj = norm * ry[static_cast<std::size_t>(j)];
                    const double* crow = conv.data() + static_cast<std::size_t>(j) * padNx_;
                    for (int i = 0; i < nx_; ++i) {
                        modelAdd(i, j, scaleFactor * (gyj * rx[static_cast<std::size_t>(i)]));
                        dirtySet(i, j, dirtyAt(i, j) - scaleFactor * crow[i]);
                    }
                }
                timing_.t_update_scale += lap(clock);
                timing_.n_scale_components += 1;
            }

            // update peak residual
            prevPeakResidual_ = peakResidual_;
            peakResidual_ = peakResidualMasked();
            timing_.t_bookkeeping += lap(clock);
            if (!switchedToHogbom_ && std::fabs(peakResidual_ - prevPeakResidual_) < 1e-4)
                numNoChange_ += 1;

            // Hogbom bookkeeping: count down, then return to Asp.
            if (switchedToHogbom_) {
                if (numHogbomIter_ == 0) {
                    switchedToHogbom_ = false;
                    // CASA and LibRA end the model update here unless the
                    // RMS has fallen, by less than a factor 1.5, since its
                    // start: the residual their model update works on drifts
                    // from the true one. The residual here is the exact one
                    // (see setup()), so by default the model update goes on
                    // to its threshold, its iteration limit or the divergence
                    // test, which takes fewer imaging cycles.
                    if (stopAfterHogbomRun_ &&
                        !(initRMSResidual > rms && initRMSResidual / rms < 1.5)) {
                        converged = 1;
                        stopCode_ = deconvolvers::STOP_ALGORITHM;
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

        // No stop test fired: the iteration limit is spent (or was 0).
        if (!converged && maxNiter_ > startingIter_) stopCode_ = deconvolvers::STOP_MAX_ITER;
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

namespace {

template <typename T>
AspResult clean_plane(T* residual, T* model, const T* psf, const T* mask,
                         int nx, int ny, double gain, double threshold_per_cycle,
                         int max_iter_per_cycle_updated, double fusedthreshold, double psf_width,
                         int largestscale, int stoppointmode, int norm_method,
                         bool stop_after_hogbom_run, bool verbose,
                         const deconvolvers::DivergenceControl<double>& divergence,
                         const FftThreads& fft_threads) {
    // A pixel outside the search region that is not finite would spread over
    // the plane through the scale convolutions and the fit: it is zero during
    // the model update and gets its value back afterwards.
    std::vector<std::pair<std::size_t, T>> nonFinite;
    if (mask != nullptr) {
        const std::size_t n = static_cast<std::size_t>(nx) * ny;
        for (std::size_t k = 0; k < n; ++k)
            if (!(static_cast<double>(mask[k]) > kMaskThreshold) &&
                !std::isfinite(residual[k])) {
                nonFinite.emplace_back(k, residual[k]);
                residual[k] = T(0);
            }
    }
    Cleaner<T> cleaner(residual, model, psf, mask, nx, ny, gain, threshold_per_cycle,
                       max_iter_per_cycle_updated, fusedthreshold, psf_width, largestscale,
                       stoppointmode, norm_method, stop_after_hogbom_run, verbose, divergence,
                       fft_threads);
    const AspResult result = cleaner.run();
    for (const auto& [k, value] : nonFinite) residual[k] = value;
    return result;
}

}  // namespace

template <typename T>
AspResult aspclean_plane(T* residual, T* model, const T* psf, const T* mask,
                         int nx, int ny, double gain, double threshold_per_cycle,
                         int max_iter_per_cycle_updated, double fusedthreshold, double psf_width,
                         int largestscale, int stoppointmode, int norm_method,
                         bool stop_after_hogbom_run, bool verbose,
                         const deconvolvers::DivergenceControl<double>& divergence,
                         int fft_threads) {
    return clean_plane<T>(residual, model, psf, mask, nx, ny, gain, threshold_per_cycle,
                          max_iter_per_cycle_updated, fusedthreshold, psf_width, largestscale,
                          stoppointmode, norm_method, stop_after_hogbom_run, verbose, divergence,
                          FftThreads{std::max(1, fft_threads), nullptr});
}

template <typename T>
void aspclean_cube(T* residual, T* model, const T* psf, const T* mask, int nt,
                   int nf, int np_img, int np_psf, int nx, int ny, double gain,
                   const double* threshold_per_cycle,
                   const int* max_iter_per_cycle_updated, double fusedthreshold,
                   double psf_width, int largestscale, int stoppointmode,
                   int norm_method, bool stop_after_hogbom_run, int processing_function_threads,
                   const deconvolvers::DivergenceControl<double>& divergence,
                   AspResult* results) {
    const int nplanes = nt * nf * np_img;
    if (nplanes <= 0) return;
    const std::size_t plane = static_cast<std::size_t>(ny) * nx;

    // Planes run concurrently, one per worker thread; the transforms of each
    // plane share all processing_function_threads with the planes still
    // running, so threads beyond the plane count, and those of workers that
    // have run out of planes, go to the planes left.
    const int nthreads = std::max(1, std::min(processing_function_threads, nplanes));
    std::atomic<int> active(nthreads);
    const FftThreads fft_threads{std::max(1, processing_function_threads), &active};

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
            results[p] = clean_plane<T>(residual + img_off, model + img_off,
                                           psf + psf_off, maskp, nx, ny, gain,
                                           threshold_per_cycle[p], max_iter_per_cycle_updated[p], fusedthreshold,
                                           psf_width, largestscale, stoppointmode,
                                           norm_method, stop_after_hogbom_run, false, divergence,
                                           fft_threads);
        }
        active.fetch_sub(1, std::memory_order_relaxed);
    };

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
                                         double, int, int, int, bool, bool,
                                         const deconvolvers::DivergenceControl<double>&, int);
template AspResult aspclean_plane<double>(double*, double*, const double*, const double*,
                                          int, int, double, double, int, double,
                                          double, int, int, int, bool, bool,
                                          const deconvolvers::DivergenceControl<double>&, int);
template void aspclean_cube<float>(float*, float*, const float*, const float*, int,
                                   int, int, int, int, int, double, const double*,
                                   const int*, double, double, int, int, int, bool, int,
                                   const deconvolvers::DivergenceControl<double>&,
                                   AspResult*);
template void aspclean_cube<double>(double*, double*, const double*, const double*,
                                    int, int, int, int, int, int, double,
                                    const double*, const int*, double, double, int,
                                    int, int, bool, int,
                                    const deconvolvers::DivergenceControl<double>&,
                                    AspResult*);

}  // namespace aspclean
