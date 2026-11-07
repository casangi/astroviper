// Fast CPU implementation of the Asp (Coral) model update.
//
// The algorithm, its control flow and every floating-point expression that
// feeds a decision or an output are those of the reference, asp_clean.cpp of
// deconvolvers/aspclean at e0c0d126 (casangi/astroviper#410,
// asp_min_mask_coverage), so the model, the residual, the components and the
// history are the same bits. A call with a value that is not finite (or of
// magnitude 2^64 or more) where the transforms would read it is the
// reference's to run (needs_reference): only on such values can a NaN's sign
// bit or payload differ. What changes is how the work is done:
//  * the 2-D transforms are those of asp_fft2.hpp: pocketfft's own 1-D plans
//    (bit-identical lines), pruned to the rows that are non-zero on input or
//    read on output, fused with the products of the spectra and with the
//    passes that consume the inverse rows (peak searches, residual updates,
//    noise samples), and threaded over row and column blocks;
//  * the search convolves the residual with the initial scales only on the
//    rows of their mask boxes and searches each row as it is made: the
//    smoothed residuals are never stored;
//  * the elementwise passes (point and scale updates, residual-shaped
//    components and their Jacobian, model updates) run on all threads; the
//    peak searches run on row blocks and merge the per-row extrema in scan
//    order (the first pixel that reaches an extreme, as the serial scan);
//  * the share of a Gaussian on the mask (maskCoverage) makes its row sums in
//    parallel and adds them serially, in the reference's order;
//  * every other sum stays serial, in the reference's order: their rounding
//    feeds the stop tests and the fits;
//  * the Levenberg-Marquardt fit of the residual-shaped components reuses the
//    misfit of a point it has evaluated (the same inputs give the same bits);
//  * work buffers persist across iterations.

#include "asp_clean_fast.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "asp_fft.hpp"
#include "asp_fft2.hpp"
#include "asp_lbfgs.hpp"

namespace aspclean_fast {

namespace {

using aspfft::cd;
using aspfft2::Buf;
using aspfft2::cv;
using cpx = aspfft2::cx;
using aspfft2::VL;
using aspfft2::parallel_for;
using aspfft2::threads_for;
namespace pd = pocketfft::detail;

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kSqrtTwoPi = 2.5066282746310002;
constexpr double kMaskThreshold = 0.99;
constexpr double kElemGrain = 32768.0;  // elementwise work per thread

struct Mat {
    int nx = 0, ny = 0;
    std::vector<double> d;
    Mat() = default;
    Mat(int nx_, int ny_) : nx(nx_), ny(ny_), d(static_cast<std::size_t>(nx_) * ny_, 0.0) {}
    double& operator()(int i, int j) { return d[static_cast<std::size_t>(j) * nx + i]; }
    double operator()(int i, int j) const { return d[static_cast<std::size_t>(j) * nx + i]; }
    std::size_t size() const { return d.size(); }
};

struct Box {
    int i0 = 0, j0 = 0, i1 = -1, j1 = -1;
};

void verify_box(int& blc0, int& blc1, int& trc0, int& trc1, int nx, int ny) {
    blc0 = std::max(0, std::min(blc0, nx - 1));
    blc1 = std::max(0, std::min(blc1, ny - 1));
    trc0 = std::max(blc0, std::min(trc0, nx - 1));
    trc1 = std::max(blc1, std::min(trc1, ny - 1));
}

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

double lap(std::chrono::steady_clock::time_point& start) {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - start).count();
    start = now;
    return dt;
}

int good_fft_size(int n) {
    return static_cast<int>(pocketfft::detail::util::good_size_cmplx(static_cast<std::size_t>(std::max(n, 1))));
}

struct FftThreads {
    int total = 1;
    const std::atomic<int>* active = nullptr;
    std::size_t get() const {
        const int n = active ? std::max(1, active->load(std::memory_order_relaxed)) : 1;
        return static_cast<std::size_t>(std::max(1, total / n));
    }
};

// PSF Gaussian width (mean FWHM in pixels) that seeds the initial scales: the
// reference's psf_gaussian_width, verbatim.
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
// Masked extrema of find_max_abs_mask_box (reference), by rows. The serial
// scan keeps the first pixel in row-major order that beats the running
// extreme (strict comparisons, NaN never), from 0 at (0, 0) (or the value at
// (0, 0) when the box is the whole image). Each row's first strict extreme,
// merged in row order with the same strict test, is the same pixel.
// ---------------------------------------------------------------------------
struct RowExtrema {
    double mx = -std::numeric_limits<double>::infinity();
    double mn = std::numeric_limits<double>::infinity();
    int mxI = -1, mnI = -1;
};

template <typename V>
inline RowExtrema row_extrema(const V* arow, const double* wrow, int i0, int i1) {
    RowExtrema e;
    for (int i = i0; i <= i1; ++i) {
        const double v = static_cast<double>(arow[i]) * wrow[i];
        if (v > e.mx) { e.mx = v; e.mxI = i; }
        if (v < e.mn) { e.mn = v; e.mnI = i; }
    }
    return e;
}

// Merge of the rows j0.. of a box into find_max_abs_mask_box's result.
inline void merge_extrema(const RowExtrema* rows, int j0, int nrows, double start, double& maxAbs, int& posI,
                          int& posJ) {
    double mn = start, mx = start;
    int mnI = 0, mnJ = 0, mxI = 0, mxJ = 0;
    for (int r = 0; r < nrows; ++r) {
        const RowExtrema& e = rows[r];
        if (e.mxI >= 0 && e.mx > mx) { mx = e.mx; mxI = e.mxI; mxJ = j0 + r; }
        if (e.mnI >= 0 && e.mn < mn) { mn = e.mn; mnI = e.mnI; mnJ = j0 + r; }
    }
    maxAbs = mx; posI = mxI; posJ = mxJ;
    if (std::abs(mn) > std::abs(mx)) { maxAbs = mn; posI = mnI; posJ = mnJ; }
}

template <typename V>
void find_max_abs_mask_box(const V* a, const Mat& w, const Box& box, double& maxAbs, int& posI, int& posJ,
                           std::size_t nthreads, std::vector<RowExtrema>& scratch) {
    const int nx = w.nx, ny = w.ny;
    double start = 0.0;
    if (box.i0 == 0 && box.j0 == 0 && box.i1 == nx - 1 && box.j1 == ny - 1)
        start = static_cast<double>(a[0]) * w(0, 0);
    const int nrows = std::max(0, box.j1 - box.j0 + 1);
    scratch.resize(static_cast<std::size_t>(nrows));
    const double work = double(nrows) * std::max(0, box.i1 - box.i0 + 1);
    parallel_for(threads_for(nthreads, work, kElemGrain), static_cast<std::size_t>(nrows),
                 [&](std::size_t b, std::size_t e) {
                     for (std::size_t r = b; r < e; ++r) {
                         const int j = box.j0 + static_cast<int>(r);
                         scratch[r] = row_extrema(a + static_cast<std::size_t>(j) * nx,
                                                  w.d.data() + static_cast<std::size_t>(j) * nx, box.i0, box.i1);
                     }
                 });
    merge_extrema(scratch.data(), box.j0, nrows, start, maxAbs, posI, posJ);
}

// ---------------------------------------------------------------------------
// 2-D forward transform of a real image (aspfft::rfft2_half, bit for bit) into
// `out` ([ny][nx/2 + 1]); rows for which data(j) is false are +0 rows and get
// the zero-row spectrum, the others are written by load(j, dst).
// ---------------------------------------------------------------------------
template <typename Data, typename Load>
void forward2d(int nx, int ny, std::size_t nthreads, Data&& data, Load&& load, cpx* out) {
    const std::size_t hx = static_cast<std::size_t>(nx) / 2 + 1;
    std::vector<int> rows;
    rows.reserve(static_cast<std::size_t>(ny));
    for (int j = 0; j < ny; ++j)
        if (data(j)) rows.push_back(j);
    const std::vector<cpx>& z = aspfft2::zero_row(static_cast<std::size_t>(nx));
    for (int j = 0; j < ny; ++j)
        if (!data(j)) std::copy(z.begin(), z.end(), out + static_cast<std::size_t>(j) * hx);
    aspfft2::r2c_rows(static_cast<std::size_t>(nx), rows.size(), nthreads,
                      [&](std::size_t r, double* dst) { load(rows[r], dst); },
                      [&](std::size_t r) { return out + static_cast<std::size_t>(rows[r]) * hx; });
    auto plan = pd::get_plan<pd::pocketfft_c<double>>(static_cast<std::size_t>(ny));
    aspfft2::column_blocks(hx, static_cast<std::size_t>(ny), 1, nthreads,
                           [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                               cv* F = bufs[0].data();
                               aspfft2::gather(out, hx, k0, nl, 0, static_cast<std::size_t>(ny), F);
                               plan->exec(F, 1.0, true);
                               aspfft2::scatter(F, 0, static_cast<std::size_t>(ny), out, hx, k0, nl);
                           });
}

// PSF cutout spectra of the objective (PsfCutoutSpectra of the reference),
// cached per grid with the same eviction (8 entries, oldest first).
struct CutoutSpectra {
    const double* psf = nullptr;
    int nx = 0, ny = 0;
    const FftThreads* threads = nullptr;
    std::deque<std::pair<std::pair<int, int>, Buf<cpx>>> cache;

    const cpx* get(int lx, int ly) {
        for (auto& entry : cache)
            if (entry.first.first == lx && entry.first.second == ly) return entry.second.data();
        const int cxp = nx / 2, cyp = ny / 2;
        const int hx = (lx - 1) / 2, hy = (ly - 1) / 2;
        Buf<cpx> spec(static_cast<std::size_t>(lx / 2 + 1) * ly);
        // grid row (dj + ly) % ly holds PSF row cy + dj for |dj| <= hy inside the PSF image
        std::vector<int> srcRow(static_cast<std::size_t>(ly), -1);
        for (int dj = -hy; dj <= hy; ++dj) {
            const int pj = cyp + dj;
            if (pj < 0 || pj >= ny) continue;
            srcRow[static_cast<std::size_t>((dj + ly) % ly)] = pj;
        }
        forward2d(lx, ly, threads->get(), [&](int r) { return srcRow[static_cast<std::size_t>(r)] >= 0; },
                  [&](int r, double* dst) {
                      std::fill_n(dst, lx, 0.0);
                      const double* prow = psf + static_cast<std::size_t>(srcRow[static_cast<std::size_t>(r)]) * nx;
                      for (int di = -hx; di <= hx; ++di) {
                          const int pi = cxp + di;
                          if (pi < 0 || pi >= nx) continue;
                          dst[(di + lx) % lx] = prow[pi];
                      }
                  },
                  spec.data());
        if (cache.size() >= 8) cache.pop_front();
        cache.emplace_back(std::make_pair(lx, ly), std::move(spec));
        return cache.back().second.data();
    }
};

// Work buffers of the objective (persist across evaluations).
struct ObjWork {
    Buf<cpx> UA, UD;
    Buf<double> CA, CD;
};

// ---------------------------------------------------------------------------
// The per-Aspen objective (AspObjective of the reference): the same values;
// (Aspen (*) psf) and its scale derivative are computed only on the grid rows
// and columns the sum reads, with both spectra built and inverted column
// block by column block.
// ---------------------------------------------------------------------------
template <typename T>
struct FastObjective {
    const T* dirty;
    CutoutSpectra& psfCutouts;
    const Buf<cpx>& psfXfrPad;
    int padNx, padNy;
    int nx, ny;
    int cx, cy;
    double inv_rescale;
    const FftThreads& threads;
    ObjWork& ws;

    double operator()(const std::vector<double>& x, std::vector<double>& grad) const {
        const double amp = x[0];
        double asc = std::fabs(x[1]);
        if (!std::isfinite(amp) || asc < 1e-3) asc = 1e-3;

        const double sigma5 = 5.0 * asc / 2.0;
        const int minI = std::max(0, static_cast<int>(cx - sigma5));
        const int maxI = std::min(nx - 1, static_cast<int>(cx + sigma5));
        const int minJ = std::max(0, static_cast<int>(cy - sigma5));
        const int maxJ = std::min(ny - 1, static_cast<int>(cy + sigma5));
        const int bx = maxI - minI + 1, by = maxJ - minJ + 1;
        int lx = good_fft_size(2 * bx - 1), ly = good_fft_size(2 * by - 1);
        const bool padded = static_cast<double>(lx) * ly >= static_cast<double>(padNx) * padNy;
        int ox = -minI, oy = -minJ, px = 0, py = 0;
        if (padded) {
            lx = padNx; ly = padNy; ox = 0; oy = 0; px = minI; py = minJ;
        }
        const cpx* psfFT = padded ? psfXfrPad.data() : psfCutouts.get(lx, ly);
        const int hx = lx / 2 + 1;

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
        // The sum reads grid rows j + oy, j in [minJ, maxJ), and columns
        // i + ox, i in [minI, maxI).
        const int r0 = minJ + oy, nr = maxJ - minJ;
        const int c0 = minI + ox, nc = maxI - minI;
        const std::size_t nthreads = threads.get();
        if (nr > 0 && nc > 0) {
            ws.UA.resize(static_cast<std::size_t>(nr) * hx);
            ws.UD.resize(static_cast<std::size_t>(nr) * hx);
            auto plan = pd::get_plan<pd::pocketfft_c<double>>(static_cast<std::size_t>(ly));
            aspfft2::column_blocks(
                static_cast<std::size_t>(hx), static_cast<std::size_t>(ly), 2, nthreads,
                [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                    cv* SA = bufs[0].data();
                    cv* SD = bufs[1].data();
                    const cv gxk = aspfft2::vlanes(Gx.data(), k0, nl);
                    const cv hxk = aspfft2::vlanes(Hx.data(), k0, nl);
                    for (int ky = 0; ky < ly; ++ky) {
                        const cv gyk = aspfft2::vbroadcast(Gy[static_cast<std::size_t>(ky)]);
                        const cv hyk = aspfft2::vbroadcast(Hy[static_cast<std::size_t>(ky)]);
                        const cv p = aspfft2::vlanes(psfFT + static_cast<std::size_t>(ky) * hx, k0, nl);
                        const cv gg = aspfft2::vmul(gxk, gyk);
                        SA[ky] = aspfft2::vscale(aspfft2::vmul(gg, p), norm);
                        SD[ky] = aspfft2::vscale(
                            aspfft2::vmul(aspfft2::vsub(aspfft2::vadd(aspfft2::vmul(hxk, gyk), aspfft2::vmul(gxk, hyk)), gg), p),
                            dnorm);
                    }
                    plan->exec(SA, 1.0, false);
                    plan->exec(SD, 1.0, false);
                    aspfft2::scatter(SA, static_cast<std::size_t>(r0), static_cast<std::size_t>(r0 + nr), ws.UA.data(),
                                     static_cast<std::size_t>(hx), k0, nl);
                    aspfft2::scatter(SD, static_cast<std::size_t>(r0), static_cast<std::size_t>(r0 + nr), ws.UD.data(),
                                     static_cast<std::size_t>(hx), k0, nl);
                });
            ws.CA.resize(static_cast<std::size_t>(nr) * nc);
            ws.CD.resize(static_cast<std::size_t>(nr) * nc);
            const double fct = 1.0 / (static_cast<double>(lx) * static_cast<double>(ly));
            aspfft2::c2r_rows(
                static_cast<std::size_t>(lx), static_cast<std::size_t>(2 * nr), fct, nthreads,
                [&](std::size_t r) {
                    return r < static_cast<std::size_t>(nr) ? ws.UA.data() + r * hx
                                                            : ws.UD.data() + (r - nr) * hx;
                },
                [&](std::size_t r, const double* row) {
                    double* dst = r < static_cast<std::size_t>(nr) ? ws.CA.data() + r * nc
                                                                   : ws.CD.data() + (r - nr) * nc;
                    std::copy(row + c0, row + c0 + nc, dst);
                });
        }

        double func = 0.0, dA = 0.0, dS = 0.0;
        for (int j = minJ; j < maxJ; ++j) {
            const double* arow = ws.CA.data() + static_cast<std::size_t>(j - minJ) * nc;
            const double* drow2 = ws.CD.data() + static_cast<std::size_t>(j - minJ) * nc;
            const T* drow = dirty + static_cast<std::size_t>(j) * nx;
            for (int i = minI; i < maxI; ++i) {
                const double aspConvPsf = arow[i - minI];
                const double dAspConvPsf = drow2[i - minI];
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
// The fast cleaner.
// ---------------------------------------------------------------------------
template <typename T>
class Cleaner {
public:
    Cleaner(T* residual, T* model, Unaligned<T> psf, Unaligned<T> mask, int nx, int ny, double gain, double threshold,
            int max_iter_per_cycle_updated, double asp_fused_threshold, double asp_psf_width, int asp_largest_scale,
            int asp_stop_point_mode, int asp_scale_normalization_method, bool asp_stop_after_hogbom_run,
            bool verbose, const deconvolvers::DivergenceControl<double>& divergence, const FftThreads& fft_threads,
            const AspOptions& options = AspOptions{})
        : residual_(residual), model_(model), psf_(psf), mask_(mask), nx_(nx), ny_(ny), gain_(gain),
          threshold_(threshold), maxNiter_(max_iter_per_cycle_updated), fusedThreshold_(asp_fused_threshold),
          userLargestScale_(static_cast<double>(asp_largest_scale)), stopPointMode_(asp_stop_point_mode),
          normMethod_(asp_scale_normalization_method), stopAfterHogbomRun_(asp_stop_after_hogbom_run),
          verbose_(verbose), psfWidthInput_(asp_psf_width), divergence_(divergence), fftThreads_(fft_threads),
          options_(options) {}

    AspResult run() {
        auto t0 = std::chrono::steady_clock::now();
        setup();
        timing_.t_setup = lap(t0) - timing_.t_search - timing_.t_fit;
        AspResult res = timing_;
        if (noClean_) {
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
        res.n_hogbom_runs = fusedThreshold_ < 0 ? 0 : static_cast<int>(nthHogbom_);
        res.psf_width = psfWidth_;
        res.components = components_;
        res.scale_noise = noiseScale_;
        return res;
    }

private:
    T* residual_;
    T* model_;
    Unaligned<T> psf_;   // read once, at setup
    Unaligned<T> mask_;  // read once, at setup; null: no mask
    int nx_, ny_;

    double gain_, threshold_;
    int maxNiter_;
    double fusedThreshold_;
    double userLargestScale_;
    int stopPointMode_;
    int normMethod_;
    bool stopAfterHogbomRun_;
    bool verbose_;
    double psfWidthInput_;
    deconvolvers::DivergenceControl<double> divergence_;
    FftThreads fftThreads_;
    AspOptions options_;
    std::vector<double> noiseScale_;
    double maxSignificance_ = 0.0;
    bool lastSearchAllScales_ = false;

    double psfWidth_ = 0.0;
    std::vector<double> psfd_;
    int padNx_ = 0, padNy_ = 0;
    Buf<cpx> psfXfrPad_;  // [padNy][padNx / 2 + 1]
    CutoutSpectra psfCutouts_;
    std::vector<double> initScaleSizes_;
    int nInitScales_ = 5;
    std::vector<std::vector<cd>> initScaleXfrX_, initScaleXfrY_;
    std::vector<Mat> initScaleMasks_;
    std::vector<Box> initScaleMaskBoxes_;
    Mat maskMat_;
    double maskThreshold_ = kMaskThreshold;
    bool noClean_ = false;
    int blc0_ = 0, blc1_ = 0, trc0_ = 0, trc1_ = 0;

    // work buffers
    Buf<cpx> spec_;                       // [ny][nx/2 + 1] forward spectrum of a search / noise / mask image
    std::vector<Buf<cpx>> rowSpec_;       // per scale: inverse-transformed columns, the rows a pass reads
    Buf<cpx> padSpec_;                    // [rows][padNx/2 + 1] of a padded inverse
    std::vector<std::vector<RowExtrema>> rowExtrema_;
    ObjWork objWork_;
    // noise samples (estimateScaleNoise): per scale, the sampled pixels in scan order
    std::vector<std::vector<std::size_t>> noiseIdx_;
    std::vector<std::vector<std::size_t>> noiseRowStart_;  // per scale: offsets into noiseIdx_ by row
    bool noiseIdxReady_ = false;
    // residual-shaped components: box transforms
    Buf<cpx> boxSpec_, boxRows_;

    double strengthOptimum_ = 0.0;
    int optimumScale_ = 0;
    double optimumScaleSize_ = 0.0;
    int posOptI_ = 0, posOptJ_ = 0;

    double peakResidual_ = 1000.0, prevPeakResidual_ = 0.0;
    double peakSigned_ = 0.0;
    double nMasked_ = 0.0;
    bool switchedToHogbom_ = false;
    unsigned int numHogbomIter_ = 0, nthHogbom_ = 0;
    std::deque<int> numIterNoGoodAspen_;
    unsigned int numNoChange_ = 0;
    bool stopAtLargeScaleNegative_ = false;
    bool didStopPointMode_ = false;
    int iteration_ = 0, startingIter_ = 0;
    int stopCode_ = deconvolvers::STOP_NONE;
    double maximumResidual_ = 0.0, totalFlux_ = 0.0;
    std::vector<double> goodAspActiveSet_, goodAspAmplitude_;
    deconvolvers::PlaneStopTest<double> stopTest_;
    double sumSquares_ = 0.0;
    double bboxSumSquares_ = 0.0;
    AspResult timing_;
    std::vector<double> components_;

    std::size_t nthreads() const { return fftThreads_.get(); }

    double dirtyAt(int i, int j) const { return static_cast<double>(residual_[static_cast<std::size_t>(j) * nx_ + i]); }
    void dirtySet(int i, int j, double v) { residual_[static_cast<std::size_t>(j) * nx_ + i] = static_cast<T>(v); }
    void modelAdd(int i, int j, double v) { model_[static_cast<std::size_t>(j) * nx_ + i] += static_cast<T>(v); }
    double modelSum() const {
        double s = 0.0;
        for (std::size_t k = 0; k < static_cast<std::size_t>(nx_) * ny_; ++k) s += static_cast<double>(model_[k]);
        return s;
    }

    double computeThreshold() const { return std::max(0.0 * maximumResidual_, threshold_); }

    // -----------------------------------------------------------------------
    // Convolution of an image with the initial scales s in [1, nInit), on the
    // rows [rb[s], re[s]) of each, as aspfft::rfft2_half of the image followed
    // by convolveInitScale (reference), bit for bit; use(s, j, row) gets each
    // row. data(j) false marks a +0 row of the image, load(j, dst) writes the
    // others. Scales are fused into one column pass when their rows fit the
    // budget, else the spectrum is kept and each scale makes its own pass.
    // -----------------------------------------------------------------------
    template <typename Data, typename Load, typename Use>
    void convolveScalesRows(int nInit, Data&& data, Load&& load, const std::vector<int>& rb,
                            const std::vector<int>& re, Use&& use) {
        const std::size_t hx = static_cast<std::size_t>(nx_) / 2 + 1;
        const std::size_t ny = static_cast<std::size_t>(ny_);
        const std::size_t nt = nthreads();
        if (static_cast<int>(rowSpec_.size()) < nInit) rowSpec_.resize(static_cast<std::size_t>(nInit));
        std::size_t totalRows = 0;
        for (int s = 1; s < nInit; ++s) totalRows += static_cast<std::size_t>(std::max(0, re[s] - rb[s]));
        if (totalRows == 0) return;
        spec_.resize(hx * ny);
        // x pass of the image rows
        std::vector<int> rows;
        rows.reserve(ny);
        for (int j = 0; j < ny_; ++j)
            if (data(j)) rows.push_back(j);
        {
            const std::vector<cpx>& z = aspfft2::zero_row(static_cast<std::size_t>(nx_));
            for (int j = 0; j < ny_; ++j)
                if (!data(j)) std::copy(z.begin(), z.end(), spec_.data() + static_cast<std::size_t>(j) * hx);
        }
        aspfft2::r2c_rows(static_cast<std::size_t>(nx_), rows.size(), nt,
                          [&](std::size_t r, double* dst) { load(rows[r], dst); },
                          [&](std::size_t r) { return spec_.data() + static_cast<std::size_t>(rows[r]) * hx; });
        auto plan = pd::get_plan<pd::pocketfft_c<double>>(ny);
        const bool fused = totalRows * hx * sizeof(cpx) <= (std::size_t(96) << 20);
        auto productInverse = [&](int s, std::size_t k0, std::size_t nl, const cv* F, cv* P) {
            const cv xs = aspfft2::vlanes(initScaleXfrX_[static_cast<std::size_t>(s)].data(), k0, nl);
            const std::vector<cd>& Y = initScaleXfrY_[static_cast<std::size_t>(s)];
            for (std::size_t ky = 0; ky < ny; ++ky)
                P[ky] = aspfft2::vmul(F[ky], aspfft2::vmul(xs, aspfft2::vbroadcast(Y[ky])));
            plan->exec(P, 1.0, false);
            aspfft2::scatter(P, static_cast<std::size_t>(rb[s]), static_cast<std::size_t>(re[s]),
                             rowSpec_[static_cast<std::size_t>(s)].data(), hx, k0, nl);
        };
        std::vector<int> scales;
        for (int s = 1; s < nInit; ++s)
            if (re[s] > rb[s]) {
                scales.push_back(s);
                rowSpec_[static_cast<std::size_t>(s)].resize(static_cast<std::size_t>(re[s] - rb[s]) * hx);
            }
        const double fct = 1.0 / (static_cast<double>(nx_) * static_cast<double>(ny_));
        auto rowPass = [&](const std::vector<int>& which) {
            std::vector<std::size_t> start(which.size() + 1, 0);
            for (std::size_t q = 0; q < which.size(); ++q)
                start[q + 1] = start[q] + static_cast<std::size_t>(re[which[q]] - rb[which[q]]);
            auto locate = [&](std::size_t r, int& s, int& j) {
                std::size_t q = 0;
                while (r >= start[q + 1]) ++q;
                s = which[q];
                j = rb[s] + static_cast<int>(r - start[q]);
            };
            aspfft2::c2r_rows(
                static_cast<std::size_t>(nx_), start.back(), fct, nt,
                [&](std::size_t r) {
                    int s, j;
                    locate(r, s, j);
                    return rowSpec_[static_cast<std::size_t>(s)].data() + static_cast<std::size_t>(j - rb[s]) * hx;
                },
                [&](std::size_t r, const double* row) {
                    int s, j;
                    locate(r, s, j);
                    use(s, j, row);
                });
        };
        if (fused) {
            aspfft2::column_blocks(hx, ny, 2, nt, [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                cv* F = bufs[0].data();
                cv* P = bufs[1].data();
                aspfft2::gather(spec_.data(), hx, k0, nl, 0, ny, F);
                plan->exec(F, 1.0, true);
                for (int s : scales) productInverse(s, k0, nl, F, P);
            });
            rowPass(scales);
        } else {
            aspfft2::column_blocks(hx, ny, 1, nt, [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                cv* F = bufs[0].data();
                aspfft2::gather(spec_.data(), hx, k0, nl, 0, ny, F);
                plan->exec(F, 1.0, true);
                aspfft2::scatter(F, 0, ny, spec_.data(), hx, k0, nl);
            });
            for (int s : scales) {
                aspfft2::column_blocks(hx, ny, 2, nt, [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                    cv* F = bufs[0].data();
                    cv* P = bufs[1].data();
                    aspfft2::gather(spec_.data(), hx, k0, nl, 0, ny, F);
                    productInverse(s, k0, nl, F, P);
                });
                rowPass(std::vector<int>{s});
            }
        }
    }

    // -----------------------------------------------------------------------
    // Exact linear convolution of `count` box images (bx x by, row-major) with
    // the PSF cutout spectrum psfFT on the lx x ly grid, evaluated on the box:
    // boxConvolve (reference) with adjoint false, for several inputs at once.
    // -----------------------------------------------------------------------
    void boxConvolve(const cpx* psfFT, int bx, int by, int lx, int ly, const double* const* in, double* const* out,
                     int count) {
        const std::size_t hx = static_cast<std::size_t>(lx) / 2 + 1;
        const std::size_t nrows = static_cast<std::size_t>(count) * by;
        const std::size_t nt = nthreads();
        boxSpec_.resize(nrows * hx);
        boxRows_.resize(nrows * hx);
        aspfft2::r2c_rows(
            static_cast<std::size_t>(lx), nrows, nt,
            [&](std::size_t r, double* dst) {
                const std::size_t m = r / by, j = r % by;
                const double* src = in[m] + j * bx;
                std::copy(src, src + bx, dst);
                std::fill(dst + bx, dst + lx, 0.0);
            },
            [&](std::size_t r) { return boxSpec_.data() + r * hx; });
        const std::vector<cpx>& z = aspfft2::zero_row(static_cast<std::size_t>(lx));
        auto plan = pd::get_plan<pd::pocketfft_c<double>>(static_cast<std::size_t>(ly));
        aspfft2::column_blocks(hx, static_cast<std::size_t>(ly), 2, nt,
                               [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                                   cv* F = bufs[0].data();
                                   cv* Pp = bufs[1].data();
                                   aspfft2::gather(psfFT, hx, k0, nl, 0, static_cast<std::size_t>(ly), Pp);
                                   for (int m = 0; m < count; ++m) {
                                       aspfft2::gather(boxSpec_.data() + static_cast<std::size_t>(m) * by * hx, hx, k0,
                                                       nl, 0, static_cast<std::size_t>(by), F);
                                       aspfft2::gather_zero(z, k0, nl, static_cast<std::size_t>(by),
                                                            static_cast<std::size_t>(ly), F);
                                       plan->exec(F, 1.0, true);
                                       for (int ky = 0; ky < ly; ++ky) F[ky] = aspfft2::vmul(F[ky], Pp[ky]);
                                       plan->exec(F, 1.0, false);
                                       aspfft2::scatter(F, 0, static_cast<std::size_t>(by),
                                                        boxRows_.data() + static_cast<std::size_t>(m) * by * hx, hx,
                                                        k0, nl);
                                   }
                               });
        const double fct = 1.0 / (static_cast<double>(lx) * static_cast<double>(ly));
        aspfft2::c2r_rows(
            static_cast<std::size_t>(lx), nrows, fct, nt, [&](std::size_t r) { return boxRows_.data() + r * hx; },
            [&](std::size_t r, const double* row) {
                const std::size_t m = r / by, j = r % by;
                std::copy(row, row + bx, out[m] + j * bx);
            });
    }

    // -----------------------------------------------------------------------
    // Convolution of an image held on rows [j0, j0 + nrows) of the padded grid
    // (data at columns [i0, i0 + ncols), +0 elsewhere) with the PSF on the
    // padded grid, as rfft2_half + product with psfXfrPad_ + irfft2_half
    // (reference); use(j, row) gets the rows j < ny of the result.
    // -----------------------------------------------------------------------
    template <typename Load, typename Use>
    void paddedConvolve(int j0, int nrows, Load&& load, Use&& use) {
        const std::size_t hx = static_cast<std::size_t>(padNx_) / 2 + 1;
        const std::size_t nt = nthreads();
        Buf<cpx>& S = spec_;
        S.resize(static_cast<std::size_t>(nrows) * hx);
        aspfft2::r2c_rows(static_cast<std::size_t>(padNx_), static_cast<std::size_t>(nrows), nt,
                          [&](std::size_t r, double* dst) { load(j0 + static_cast<int>(r), dst); },
                          [&](std::size_t r) { return S.data() + r * hx; });
        const std::vector<cpx>& z = aspfft2::zero_row(static_cast<std::size_t>(padNx_));
        padSpec_.resize(static_cast<std::size_t>(ny_) * hx);
        auto plan = pd::get_plan<pd::pocketfft_c<double>>(static_cast<std::size_t>(padNy_));
        aspfft2::column_blocks(hx, static_cast<std::size_t>(padNy_), 1, nt,
                               [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                                   cv* F = bufs[0].data();
                                   aspfft2::gather_zero(z, k0, nl, 0, static_cast<std::size_t>(j0), F);
                                   aspfft2::gather_from(S.data(), hx, 0, k0, nl, static_cast<std::size_t>(j0),
                                                        static_cast<std::size_t>(j0 + nrows), F);
                                   aspfft2::gather_zero(z, k0, nl, static_cast<std::size_t>(j0 + nrows),
                                                        static_cast<std::size_t>(padNy_), F);
                                   plan->exec(F, 1.0, true);
                                   for (int ky = 0; ky < padNy_; ++ky)
                                       F[ky] = aspfft2::vmul(F[ky], aspfft2::vlanes(psfXfrPad_.data() +
                                                                                        static_cast<std::size_t>(ky) * hx,
                                                                                    k0, nl));
                                   plan->exec(F, 1.0, false);
                                   aspfft2::scatter(F, 0, static_cast<std::size_t>(ny_), padSpec_.data(), hx, k0, nl);
                               });
        const double fct = 1.0 / (static_cast<double>(padNx_) * static_cast<double>(padNy_));
        aspfft2::c2r_rows(static_cast<std::size_t>(padNx_), static_cast<std::size_t>(ny_), fct, nt,
                          [&](std::size_t r) { return padSpec_.data() + r * hx; },
                          [&](std::size_t r, const double* row) { use(static_cast<int>(r), row); });
    }

    // -----------------------------------------------------------------------
    // One-time setup (reference setup()).
    // -----------------------------------------------------------------------
    void setup() {
        psfd_.resize(static_cast<std::size_t>(nx_) * ny_);
        parallel_for(threads_for(nthreads(), double(psfd_.size()), 4 * kElemGrain), static_cast<std::size_t>(ny_),
                     [&](std::size_t b, std::size_t e) {
                         for (std::size_t k = b * nx_; k < e * nx_; ++k) psfd_[k] = static_cast<double>(psf_[k]);
                     });
        padNx_ = good_fft_size((3 * nx_ + 1) / 2);
        padNy_ = good_fft_size((3 * ny_ + 1) / 2);
        {
            // PSF centre (nx/2, ny/2) at the origin of the padded grid: grid row
            // pj = (j - ny/2) mod padNy holds PSF row j.
            std::vector<int> srcRow(static_cast<std::size_t>(padNy_), -1);
            for (int j = 0; j < ny_; ++j) srcRow[static_cast<std::size_t>(((j - ny_ / 2) % padNy_ + padNy_) % padNy_)] = j;
            const std::size_t hxp = static_cast<std::size_t>(padNx_) / 2 + 1;
            psfXfrPad_.resize(hxp * padNy_);
            forward2d(padNx_, padNy_, nthreads(), [&](int r) { return srcRow[static_cast<std::size_t>(r)] >= 0; },
                      [&](int r, double* dst) {
                          std::fill_n(dst, padNx_, 0.0);
                          const double* prow = psfd_.data() + static_cast<std::size_t>(srcRow[static_cast<std::size_t>(r)]) * nx_;
                          for (int i = 0; i < nx_; ++i) dst[((i - nx_ / 2) % padNx_ + padNx_) % padNx_] = prow[i];
                      },
                      psfXfrPad_.data());
        }
        psfCutouts_.psf = psfd_.data();
        psfCutouts_.nx = nx_;
        psfCutouts_.ny = ny_;
        psfCutouts_.threads = &fftThreads_;
        psfCutouts_.cache.clear();

        // An empty plane has no PSF to measure (the reference reads past the
        // end of it, and crashes): it takes the floor width.
        if (psfWidthInput_ > 0.0) psfWidth_ = psfWidthInput_;
        else psfWidth_ = psfd_.empty() ? 1.0 : psf_gaussian_width(psfd_.data(), nx_, ny_);
        if (psfWidth_ < 1.0) psfWidth_ = 1.0;

        setInitScaleXfrs();
        setInitScaleMasks();
        if (noClean_) return;
        if (options_.asp_scale_significance || options_.asp_residual_components) estimateScaleNoise();
        std::vector<double> scaleSizes = getActiveSetAspen();
        scaleSizes.push_back(0.0);
        defineAspScales(scaleSizes);
    }

    // Robust noise of the residual and of its convolution with each initial
    // scale (reference estimateScaleNoise): the same samples in the same order.
    void estimateScaleNoise() {
        const int ns = static_cast<int>(initScaleSizes_.size());
        noiseScale_.assign(ns, std::numeric_limits<double>::infinity());
        const std::size_t n_all = static_cast<std::size_t>(nx_) * ny_;
        const std::size_t stride = std::max<std::size_t>(1, n_all / 400000);
        if (!noiseIdxReady_) {
            // The pixels sampled at each scale (they depend on the mask only).
            noiseIdx_.assign(ns, {});
            noiseRowStart_.assign(ns, std::vector<std::size_t>(static_cast<std::size_t>(ny_) + 1, 0));
            for (int s = 0; s < ns; ++s) {
                const int border = static_cast<int>(3.0 * initScaleSizes_[s]) + 1;
                std::vector<std::size_t>& idx = noiseIdx_[s];
                std::vector<std::size_t>& rs = noiseRowStart_[s];
                for (std::size_t k = 0; k < n_all; k += stride) {
                    const int i = static_cast<int>(k % nx_), j = static_cast<int>(k / nx_);
                    if (i < border || j < border || i >= nx_ - border || j >= ny_ - border) continue;
                    if (maskMat_(i, j) > 1e-6) continue;
                    idx.push_back(k);
                    rs[static_cast<std::size_t>(j) + 1] += 1;
                }
                for (int j = 0; j < ny_; ++j) rs[static_cast<std::size_t>(j) + 1] += rs[static_cast<std::size_t>(j)];
            }
            noiseIdxReady_ = true;
        }
        std::vector<std::vector<double>> vals(static_cast<std::size_t>(ns));
        for (int s = 0; s < ns; ++s) vals[s].resize(noiseIdx_[s].size());
        // scale 0: the residual itself
        for (std::size_t m = 0; m < noiseIdx_[0].size(); ++m) vals[0][m] = static_cast<double>(residual_[noiseIdx_[0][m]]);
        if (ns > 1) {
            std::vector<int> rb(ns, 0), re(ns, 0);
            for (int s = 1; s < ns; ++s) {
                if (noiseIdx_[s].size() < 100) continue;  // no noise from this scale: skip its transform
                rb[s] = static_cast<int>(noiseIdx_[s].front() / nx_);
                re[s] = static_cast<int>(noiseIdx_[s].back() / nx_) + 1;
            }
            convolveScalesRows(
                ns, [](int) { return true; },
                [&](int j, double* dst) {
                    const T* src = residual_ + static_cast<std::size_t>(j) * nx_;
                    for (int i = 0; i < nx_; ++i) dst[i] = static_cast<double>(src[i]);
                },
                rb, re,
                [&](int s, int j, const double* row) {
                    const std::vector<std::size_t>& idx = noiseIdx_[s];
                    const std::vector<std::size_t>& rs = noiseRowStart_[s];
                    const std::size_t base = static_cast<std::size_t>(j) * nx_;
                    for (std::size_t m = rs[static_cast<std::size_t>(j)]; m < rs[static_cast<std::size_t>(j) + 1]; ++m)
                        vals[s][m] = row[idx[m] - base];
                });
        }
        parallel_for(std::min<std::size_t>(nthreads(), static_cast<std::size_t>(ns)), static_cast<std::size_t>(ns),
                     [&](std::size_t b, std::size_t e) {
                         for (std::size_t s = b; s < e; ++s) {
                             std::vector<double>& v = vals[s];
                             if (v.size() < 100) continue;
                             auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
                             std::nth_element(v.begin(), mid, v.end());
                             const double med = *mid;
                             for (double& x : v) x = std::fabs(x - med);
                             std::nth_element(v.begin(), mid, v.end());
                             noiseScale_[s] = 1.4826 * (*mid);
                         }
                     });
    }

    // The residual-shaped component phase (reference residualComponentPhase).
    int residualComponentPhase() {
        auto clock = std::chrono::steady_clock::now();
        const Box& box = initScaleMaskBoxes_[0];
        if (box.i1 < box.i0) return deconvolvers::STOP_NONE;
        const int bx = box.i1 - box.i0 + 1, by = box.j1 - box.j0 + 1;
        const int lx = good_fft_size(2 * bx - 1), ly = good_fft_size(2 * by - 1);
        Buf<cpx> psfFTcopy(static_cast<std::size_t>(lx / 2 + 1) * ly);  // a copy: the cache may evict it
        {
            const cpx* p = psfCutouts_.get(lx, ly);
            std::copy(p, p + psfFTcopy.size(), psfFTcopy.data());
        }
        const cpx* psfFT = psfFTcopy.data();
        const std::size_t nb = static_cast<std::size_t>(bx) * by;
        const std::size_t nt = nthreads();
        const std::size_t ntElem = threads_for(nt, double(nb), kElemGrain);
        std::vector<double> R(nb), inMask(nb), acc(nb, 0.0), C(nb), dCdT(nb), dCde(nb), CP(nb), r(nb);
        for (int j = 0; j < by; ++j)
            for (int i = 0; i < bx; ++i) {
                const std::size_t k = static_cast<std::size_t>(j) * bx + i;
                R[k] = dirtyAt(box.i0 + i, box.j0 + j);
                inMask[k] = initScaleMasks_[0](box.i0 + i, box.j0 + j) != 0.0 ? 1.0 : 0.0;
            }
        double floor = (!noiseScale_.empty() && std::isfinite(noiseScale_[0]))
                           ? options_.asp_residual_component_noise_floor * noiseScale_[0] : 0.0;
        if (options_.asp_residual_component_floor_at_threshold) floor = std::max(floor, threshold_);
        const double flo = options_.asp_residual_component_min_fraction;
        const double fhi = options_.asp_residual_component_max_fraction;
        const double elo = options_.asp_residual_component_min_epsilon_fraction;
        const double ehi = options_.asp_residual_component_max_epsilon_fraction;
        const double llo = std::log(elo), lhi = std::log(ehi);
        std::vector<double> J1(nb), J2(nb);
        double frac = std::min(std::max(0.7, flo), fhi);
        double logEps = std::min(std::max(std::log(1.0), llo), lhi);
        int status = deconvolvers::STOP_NONE;
        // The point (frac, log eps_frac) whose C, CP, r and f the buffers hold. The reference evaluates the
        // misfit again at the point of a trial it has just accepted (to get the Jacobian there) and once more
        // for the final component: the same inputs, so the same bits. Here those repeats take C (*) psf, r
        // and f from the buffers; only the Jacobian columns are transformed.
        bool memoValid = false;
        double memoFr = 0.0, memoLe = 0.0, memoF = 0.0;
        auto samePoint = [&](double fr, double le) {
            return memoValid && std::memcmp(&fr, &memoFr, sizeof fr) == 0 && std::memcmp(&le, &memoLe, sizeof le) == 0;
        };
        while (iteration_ < maxNiter_) {
            memoValid = false;  // R changes with every component
            double A = 0.0, peak = 0.0, sumSquares = 0.0;
            std::size_t kPeak = 0;
            for (std::size_t k = 0; k < nb; ++k) {
                if (inMask[k] == 0.0) continue;
                sumSquares += R[k] * R[k];
                const double v = options_.asp_residual_component_positive_only ? R[k] : std::fabs(R[k]);
                if (v > A) { A = v; peak = R[k]; kPeak = k; }
            }
            if (A <= options_.asp_residual_component_switch_factor * threshold_) break;
            if (stopTest_.check_divergence(A, sumSquares, divergence_) != deconvolvers::STOP_NONE) {
                status = deconvolvers::STOP_DIVERGED;
                break;
            }
            const double s = peak > 0 ? 1.0 : -1.0;
            auto component = [&](double fr, double epsf, bool derivatives, bool& floorActive) {
                const double T0 = fr * A;
                floorActive = floor > T0;
                const double level = std::max(T0, floor), eps = epsf * A;
                parallel_for(ntElem, nb, [&](std::size_t b, std::size_t e) {
                    for (std::size_t k = b; k < e; ++k) {
                        const double sR = s * R[k];
                        if (inMask[k] == 0.0 || sR < level) {
                            C[k] = 0.0;
                            if (derivatives) { dCdT[k] = 0.0; dCde[k] = 0.0; }
                            continue;
                        }
                        const double u = (sR - level) / eps;
                        const double th = std::tanh(u);
                        C[k] = R[k] * th;
                        if (derivatives) {
                            const double sech2 = 1.0 - th * th;
                            dCdT[k] = -R[k] * sech2 / eps;
                            dCde[k] = -R[k] * sech2 * u / eps;
                        }
                    }
                });
            };
            auto misfit = [&](double fr, double le, bool jacobian) {
                bool floorActive = false;
                const double epsf = std::exp(le);
                double f = 0.0;
                if (jacobian && samePoint(fr, le)) {
                    // C (made again, the same bits), CP, r and f are those of the evaluation at this point
                    component(fr, epsf, true, floorActive);
                    const double* in[2] = {dCdT.data(), dCde.data()};
                    double* out[2] = {J1.data(), J2.data()};
                    boxConvolve(psfFT, bx, by, lx, ly, in, out, 2);
                    f = memoF;
                } else {
                    component(fr, epsf, jacobian, floorActive);
                    if (jacobian) {
                        const double* in[3] = {C.data(), dCdT.data(), dCde.data()};
                        double* out[3] = {CP.data(), J1.data(), J2.data()};
                        boxConvolve(psfFT, bx, by, lx, ly, in, out, 3);
                    } else {
                        const double* in[1] = {C.data()};
                        double* out[1] = {CP.data()};
                        boxConvolve(psfFT, bx, by, lx, ly, in, out, 1);
                    }
                    for (std::size_t k = 0; k < nb; ++k) {
                        r[k] = inMask[k] * (R[k] - CP[k]);
                        f += r[k] * r[k];
                    }
                    memoValid = true;
                    memoFr = fr;
                    memoLe = le;
                    memoF = f;
                }
                if (jacobian) {
                    const double c1 = floorActive ? 0.0 : -A, c2 = -epsf * A;
                    parallel_for(ntElem, nb, [&](std::size_t b, std::size_t e) {
                        for (std::size_t k = b; k < e; ++k) {
                            J1[k] = inMask[k] * c1 * J1[k];
                            J2[k] = inMask[k] * c2 * J2[k];
                        }
                    });
                }
                ++timing_.n_residual_fit_evaluations;
                return f;
            };
            double f = misfit(frac, logEps, true);
            double lambda = 1e-3;
            const int fitIterations = options_.asp_residual_component_fit_iterations;
            for (int it = 0; it < fitIterations; ++it) {
                double a11 = 0, a12 = 0, a22 = 0, g1 = 0, g2 = 0;
                for (std::size_t k = 0; k < nb; ++k) {
                    a11 += J1[k] * J1[k]; a12 += J1[k] * J2[k]; a22 += J2[k] * J2[k];
                    g1 += J1[k] * r[k]; g2 += J2[k] * r[k];
                }
                bool accepted = false;
                for (int tries = 0; tries < 8 && !accepted; ++tries) {
                    const double m11 = a11 * (1 + lambda) + 1e-30, m22 = a22 * (1 + lambda) + 1e-30;
                    const double det = m11 * m22 - a12 * a12;
                    if (!(std::fabs(det) > 0)) { lambda *= 10; continue; }
                    const double d1 = -(m22 * g1 - a12 * g2) / det, d2 = -(m11 * g2 - a12 * g1) / det;
                    const double nf = std::min(std::max(frac + d1, flo), fhi);
                    const double nl = std::min(std::max(logEps + d2, llo), lhi);
                    const double fTrial = misfit(nf, nl, false);
                    if (fTrial < f) {
                        const double rel = (f - fTrial) / std::max(f, 1e-300);
                        frac = nf; logEps = nl; f = fTrial; accepted = true;
                        lambda = std::max(lambda / 3.0, 1e-9);
                        if (rel < 1e-7) it = fitIterations;
                    } else {
                        lambda *= 5.0;
                    }
                }
                if (!accepted) break;
                if (it + 1 < fitIterations) f = misfit(frac, logEps, true);
            }
            if (options_.record_components) {
                const double rec[6] = {peak, frac, std::exp(logEps), 0.0, 0.0, f};
                timing_.residual_fits.insert(timing_.residual_fits.end(), rec, rec + 6);
            }
            if (!samePoint(frac, logEps)) {  // else C and CP already hold the final component
                bool floorActive = false;
                component(frac, std::exp(logEps), false, floorActive);
                const double* in[1] = {C.data()};
                double* out[1] = {CP.data()};
                boxConvolve(psfFT, bx, by, lx, ly, in, out, 1);
            }
            double fNone = 0.0, fComp = 0.0;
            for (std::size_t k = 0; k < nb; ++k) {
                if (inMask[k] == 0.0) continue;
                fNone += R[k] * R[k];
                const double d = R[k] - gain_ * CP[k];
                fComp += d * d;
            }
            if (!(fComp < fNone * (1.0 - 1e-6))) break;
            double flux = 0.0, sumSquaresAfter = 0.0;
            for (std::size_t k = 0; k < nb; ++k) {
                acc[k] += gain_ * C[k];
                flux += gain_ * C[k];
                R[k] -= gain_ * CP[k];
                sumSquaresAfter += inMask[k] * R[k] * R[k];
            }
            recordComponent(peak, nMasked_ > 0 ? std::sqrt(sumSquaresAfter / nMasked_) : 0.0,
                            box.i0 + static_cast<int>(kPeak % bx), box.j0 + static_cast<int>(kPeak / bx), -1.0,
                            flux);
            ++iteration_;
            ++timing_.n_residual_components;
        }
        bool any = false;
        for (double v : acc) any = any || v != 0.0;
        if (any) {
            parallel_for(threads_for(nt, double(nb), kElemGrain), static_cast<std::size_t>(by),
                         [&](std::size_t b, std::size_t e) {
                             for (std::size_t j = b; j < e; ++j)
                                 for (int i = 0; i < bx; ++i)
                                     modelAdd(box.i0 + i, box.j0 + static_cast<int>(j), acc[j * bx + i]);
                         });
            paddedConvolve(
                box.j0, by,
                [&](int pj, double* dst) {
                    std::fill_n(dst, padNx_, 0.0);
                    std::copy(acc.data() + static_cast<std::size_t>(pj - box.j0) * bx,
                              acc.data() + static_cast<std::size_t>(pj - box.j0 + 1) * bx, dst + box.i0);
                },
                [&](int j, const double* row) {
                    for (int i = 0; i < nx_; ++i) dirtySet(i, j, dirtyAt(i, j) - row[i]);
                });
        }
        timing_.t_residual_components += lap(clock);
        return status;
    }

    static bool measuredNoise(double noise) { return std::isfinite(noise) && noise > 0.0; }

    double significanceLimit() const {
        if (options_.asp_significance > 0.0) return options_.asp_significance;
        return threshold_ / noiseScale_[0];
    }

    bool significanceReached() const {
        if (!measuredNoise(noiseScale_[0])) return false;
        return maxSignificance_ >= significanceLimit();
    }

    void setInitScales() {
        if (userLargestScale_ < 0.0) {
            initScaleSizes_ = {0.0, psfWidth_, 2.0 * psfWidth_, 4.0 * psfWidth_, 8.0 * psfWidth_};
            return;
        }
        double uls = userLargestScale_;
        if (uls > std::min(nx_ / 10, ny_ / 10)) uls = std::ceil(std::min(nx_ / 10, ny_ / 10));
        const int numscale = static_cast<int>(std::floor(uls / psfWidth_));
        if (numscale == 0) {
            if (uls > 0.0) initScaleSizes_ = {0.0, uls, psfWidth_};
            else initScaleSizes_ = {0.0};
            nInitScales_ = static_cast<int>(initScaleSizes_.size());
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

    void setInitScaleXfrs() {
        nInitScales_ = 5;
        setInitScales();
        initScaleXfrX_.assign(nInitScales_, {});
        initScaleXfrY_.assign(nInitScales_, {});
        auto wrapped_profile = [](int n, double sigma) {
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

    // Share of the flux of the Gaussian at (ci, cj) on the mask (reference
    // maskCoverage, PR #410). The same bits as the reference's serial loops:
    // the reference makes each row sum of the mask box serially, row = +0 then
    // row += maskMat_(i, j) * rx[i] for i = blc0_..trc0_ in order, and adds
    // the rows serially, onMask = +0 then onMask += ry[j] * row for j =
    // blc1_..trc1_ in order. Here the row sums are made by that same loop
    // (the same operands in the same order; -ffp-contract=off, so no fused
    // multiply-add, and doubles are not widened on x86-64) on blocks of rows
    // in parallel, and kept; then the sum over the rows is the reference's
    // serial loop over the kept row sums. Which thread makes a row sum, and
    // when, does not change its value, so the share has the reference's bits
    // for every thread count. Rows are independent and only the order of the
    // operations inside a row and across the rows sets the rounding, so this
    // is the only change: no row is skipped (a zero profile value times a
    // mask value that is not finite is NaN, which the reference propagates),
    // and the two normalising sums stay std::accumulate over the whole
    // profiles, as in the reference.
    double maskCoverage(int ci, int cj, double sigma) const {
        const std::vector<double> rx = gaussian_profile(nx_, ci, sigma, 0, nx_ - 1);
        const std::vector<double> ry = gaussian_profile(ny_, cj, sigma, 0, ny_ - 1);
        const std::size_t nrows = static_cast<std::size_t>(std::max(0, trc1_ - blc1_ + 1));
        std::vector<double> rowSum(nrows);
        const double work = double(nrows) * std::max(0, trc0_ - blc0_ + 1);
        parallel_for(threads_for(nthreads(), work, kElemGrain), nrows, [&](std::size_t b, std::size_t e) {
            for (std::size_t r = b; r < e; ++r) {
                const int j = blc1_ + static_cast<int>(r);
                double row = 0.0;
                for (int i = blc0_; i <= trc0_; ++i)
                    row += maskMat_(i, j) * rx[static_cast<std::size_t>(i)];
                rowSum[r] = row;
            }
        });
        double onMask = 0.0;
        for (int j = blc1_; j <= trc1_; ++j)
            onMask += ry[static_cast<std::size_t>(j)] * rowSum[static_cast<std::size_t>(j - blc1_)];
        return onMask / (std::accumulate(rx.begin(), rx.end(), 0.0) *
                         std::accumulate(ry.begin(), ry.end(), 0.0));
    }

    // setInitScaleMasks (reference, with PR #410: a scale s > 0 keeps the mask
    // pixels where (mask (*) scale) / (the sum of the scale) is above
    // asp_min_mask_coverage and the mask is above the search threshold). The
    // scale masks are read only inside the bounding box [blc, trc] of the mask
    // (the search boxes, the peak and the residual-shaped components all lie
    // in it), so their convolutions are made on its rows only; elsewhere they
    // stay 0. Since #410 the reference's are 0 there too: outside the rows of
    // the box the mask is at most 1e-6 (or not a number), so it is not above
    // the search threshold, and the scale masks are subsets of the scale-0
    // mask (before their borders are zeroed).
    void setInitScaleMasks() {
        maskMat_ = Mat(nx_, ny_);
        double maxMask = -1e30;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                const double v = mask_ ? static_cast<double>(mask_[static_cast<std::size_t>(j) * nx_ + i]) : 1.0;
                maskMat_(i, j) = v;
                if (v > maxMask) maxMask = v;
            }
        noClean_ = (maxMask < maskThreshold_);
        if (noClean_) return;

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

        // The sum of each scale and the least share it must keep on the mask
        // (reference, PR #410; scale 0: the delta, the search threshold).
        std::vector<double> scaleSums(static_cast<std::size_t>(nInitScales_));
        std::vector<double> leasts(static_cast<std::size_t>(nInitScales_));
        for (int s = 0; s < nInitScales_; ++s) {
            const double scaleSum =
                s == 0 ? 1.0 : initScaleXfrX_[s][0].real() * initScaleXfrY_[s][0].real();
            const double least = s == 0 ? maskThreshold_ : options_.asp_min_mask_coverage;
            scaleSums[static_cast<std::size_t>(s)] = scaleSum;
            leasts[static_cast<std::size_t>(s)] = least;
        }

        initScaleMasks_.assign(nInitScales_, Mat(nx_, ny_));
        {
            // scale 0 on the whole image: conv = maskMat_.d (mask (*) delta)
            const double scaleSum = scaleSums[0], least = leasts[0];
            for (int j = 0; j < ny_; ++j)
                for (int i = 0; i < nx_; ++i) {
                    double v = maskMat_(i, j);
                    if (maskThreshold_ > 0)
                        v = (v / scaleSum > least && maskMat_(i, j) > maskThreshold_) ? 1.0 : 0.0;
                    initScaleMasks_[0](i, j) = v;
                }
        }
        if (nInitScales_ > 1) {
            std::vector<char> rowData(static_cast<std::size_t>(ny_), 0);
            for (int j = 0; j < ny_; ++j)
                rowData[static_cast<std::size_t>(j)] =
                    !aspfft2::all_positive_zero(maskMat_.d.data() + static_cast<std::size_t>(j) * nx_,
                                                static_cast<std::size_t>(nx_));
            std::vector<int> rb(nInitScales_, blc1_), re(nInitScales_, trc1_ + 1);
            convolveScalesRows(
                nInitScales_, [&](int j) { return rowData[static_cast<std::size_t>(j)] != 0; },
                [&](int j, double* dst) {
                    std::copy(maskMat_.d.data() + static_cast<std::size_t>(j) * nx_,
                              maskMat_.d.data() + static_cast<std::size_t>(j + 1) * nx_, dst);
                },
                rb, re,
                [&](int s, int j, const double* row) {
                    // row = (mask (*) scale s) on image row j, the reference's conv bits
                    const double scaleSum = scaleSums[static_cast<std::size_t>(s)];
                    const double least = leasts[static_cast<std::size_t>(s)];
                    double* m = initScaleMasks_[s].d.data() + static_cast<std::size_t>(j) * nx_;
                    for (int i = 0; i < nx_; ++i) {
                        double v = row[i];
                        if (maskThreshold_ > 0)
                            v = (v / scaleSum > least && maskMat_(i, j) > maskThreshold_) ? 1.0 : 0.0;
                        m[i] = v;
                    }
                });
        }

        for (int s = 0; s < nInitScales_; ++s) {
            const int border = static_cast<int>(initScaleSizes_[s] * 1.5);
            Mat& m = initScaleMasks_[s];
            auto zeroBox = [&](int b0, int b1, int t0, int t1) {
                verify_box(b0, b1, t0, t1, nx_, ny_);
                for (int j = b1; j <= t1; ++j)
                    for (int i = b0; i <= t0; ++i) m(i, j) = 0.0;
            };
            if (border > 0) {
                zeroBox(0, 0, nx_ - 1, border);
                zeroBox(0, ny_ - border - 1, nx_ - 1, ny_ - 1);
                zeroBox(0, border, border, ny_ - border - 1);
                zeroBox(nx_ - border - 1, border, nx_ - 1, ny_ - border - 1);
            }
        }

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

    void defineAspScales(std::vector<double>& scaleSizes) { std::sort(scaleSizes.begin(), scaleSizes.end()); }

    void switchedToHogbom(bool runlong) {
        switchedToHogbom_ = true;
        if (fusedThreshold_ < 0) switchedToHogbom_ = false;
        nthHogbom_ += 1;
        numIterNoGoodAspen_.clear();
        numHogbomIter_ = 51;
        if (runlong) numHogbomIter_ = 510;
    }

    // The masked extrema of the residual (scale 0) and of its convolutions
    // with the other initial scales, on their mask boxes (the search of
    // maxDirtyConvInitScales).
    void searchScales(std::vector<double>& maxima, std::vector<int>& posMaxI, std::vector<int>& posMaxJ) {
        if (static_cast<int>(rowExtrema_.size()) < nInitScales_) rowExtrema_.resize(static_cast<std::size_t>(nInitScales_));
        find_max_abs_mask_box(residual_, initScaleMasks_[0], initScaleMaskBoxes_[0], maxima[0], posMaxI[0],
                              posMaxJ[0], nthreads(), rowExtrema_[0]);
        if (nInitScales_ <= 1) return;
        std::vector<int> rb(nInitScales_, 0), re(nInitScales_, 0);
        for (int s = 1; s < nInitScales_; ++s) {
            const Box& b = initScaleMaskBoxes_[s];
            if (b.i1 < b.i0) continue;
            rb[s] = b.j0;
            re[s] = b.j1 + 1;
            rowExtrema_[s].assign(static_cast<std::size_t>(re[s] - rb[s]), RowExtrema{});
        }
        std::vector<double> start(nInitScales_, 0.0);
        convolveScalesRows(
            nInitScales_, [](int) { return true; },
            [&](int j, double* dst) {
                const T* src = residual_ + static_cast<std::size_t>(j) * nx_;
                for (int i = 0; i < nx_; ++i) dst[i] = static_cast<double>(src[i]);
            },
            rb, re,
            [&](int s, int j, const double* row) {
                const Box& b = initScaleMaskBoxes_[s];
                rowExtrema_[s][static_cast<std::size_t>(j - rb[s])] =
                    row_extrema(row, initScaleMasks_[s].d.data() + static_cast<std::size_t>(j) * nx_, b.i0, b.i1);
                if (j == 0 && b.i0 == 0 && b.j0 == 0 && b.i1 == nx_ - 1 && b.j1 == ny_ - 1)
                    start[s] = row[0] * initScaleMasks_[s](0, 0);
            });
        for (int s = 1; s < nInitScales_; ++s) {
            const Box& b = initScaleMaskBoxes_[s];
            if (b.i1 < b.i0) { maxima[s] = 0.0; posMaxI[s] = 0; posMaxJ[s] = 0; continue; }
            merge_extrema(rowExtrema_[s].data(), b.j0, re[s] - rb[s], start[s], maxima[s], posMaxI[s], posMaxJ[s]);
        }
    }

    void maxDirtyConvInitScales(double& strengthOptimum, int& optimumScale, int& posI, int& posJ) {
        std::vector<double> maxima(nInitScales_, 0.0);
        std::vector<int> posMaxI(nInitScales_, 0), posMaxJ(nInitScales_, 0);
        searchScales(maxima, posMaxI, posMaxJ);
        for (int s = 0; s < nInitScales_; ++s) {
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
        lastSearchAllScales_ = nInitScales_ > 1;
        if (options_.asp_scale_significance && nInitScales_ > 1 && !noiseScale_.empty()) {
            maxSignificance_ = 0.0;
            int sigScale = -1;
            for (int s = 1; s < nInitScales_; ++s) {
                if (!measuredNoise(noiseScale_[s])) continue;
                const double sig = std::abs(maxima[s]) / noiseScale_[s];
                if (sig > maxSignificance_) { maxSignificance_ = sig; sigScale = s; }
            }
            if (std::abs(maxima[0]) < threshold_ && sigScale > 0 && significanceReached()) {
                optimumScale = sigScale;
                strengthOptimum = maxima[sigScale];
                posI = posMaxI[sigScale];
                posJ = posMaxJ[sigScale];
            }
        }
        if (optimumScale > 0) {
            const double normalization = std::sqrt(
                kTwoPi / (std::pow(1.0 / psfWidth_, 2) + std::pow(1.0 / initScaleSizes_[optimumScale], 2)));
            if (normMethod_ == 2) strengthOptimum *= std::sqrt(kTwoPi * initScaleSizes_[optimumScale]);
            strengthOptimum /= normalization;
        }
    }

    std::vector<double> getActiveSetAspen() {
        if (!switchedToHogbom_ &&
            std::accumulate(numIterNoGoodAspen_.begin(), numIterNoGoodAspen_.end(), 0) >= 5) {
            switchedToHogbom(false);
            timing_.n_switch_point_scales += switchedToHogbom_;
        }
        nInitScales_ = switchedToHogbom_ ? 1 : static_cast<int>(initScaleSizes_.size());
        auto clock = std::chrono::steady_clock::now();

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
        if (optimumScale_ == 0) return {};

        double rescale = std::fabs(strengthOptimum / psfWidth_);
        if (!(rescale > 0.0) || !std::isfinite(rescale)) rescale = 1.0;
        FastObjective<T> obj{residual_, psfCutouts_, psfXfrPad_, padNx_, padNy_, nx_, ny_,
                             posI, posJ, 1.0 / rescale, fftThreads_, objWork_};
        std::vector<double> x = {strengthOptimum / rescale, initScaleSizes_[optimumScale]};
        std::vector<double> sc = {std::fabs(strengthOptimum) / rescale, initScaleSizes_[optimumScale]};
        asplbfgs::Options opt;  // m=1, maxits=5, eps*=1e-3 -> matches ALGLIB call
        const asplbfgs::Report fitReport = asplbfgs::minimize(
            x, sc, [&obj](const std::vector<double>& xx, std::vector<double>& gg) { return obj(xx, gg); }, opt);
        timing_.t_fit += lap(clock);
        timing_.n_fits += 1;
        timing_.n_fit_evaluations += fitReport.evaluations;

        double amp = x[0] * rescale;
        double scale = std::fabs(x[1]);
        // The fit can widen a Gaussian past the scale the search tested: cut
        // it back; the amplitude that fits is where the gradient, linear in
        // the amplitude, vanishes. (Reference, PR #410, verbatim. obj is this
        // file's objective: a pure function of its argument and of the
        // residual, which nothing changes between the fit and here; it keeps
        // no memo of earlier evaluations, and its cutout cache, plans and
        // per-thread buffers are keyed by the grid or rewritten before they
        // are read. Like the reference, the two evaluations are not counted
        // in n_fit_evaluations, and their time goes to t_fit.)
        const double least = options_.asp_min_mask_coverage;
        if (maskCoverage(posI, posJ, scale) < least) {
            const double tested = initScaleSizes_[optimumScale];
            double lo = tested < scale && maskCoverage(posI, posJ, tested) >= least ? tested : 0.0;
            double hi = scale;
            for (int step = 0; step < 12; ++step) {
                const double mid = 0.5 * (lo + hi);
                if (maskCoverage(posI, posJ, mid) < least)
                    hi = mid;
                else
                    lo = mid;
            }
            scale = lo;
            std::vector<double> gradAt0(2), gradAt1(2);
            obj({0.0, scale}, gradAt0);
            obj({1.0, scale}, gradAt1);
            amp = gradAt0[0] / (gradAt0[0] - gradAt1[0]) * rescale;
            timing_.t_fit += lap(clock);
        }
        if (scale < 0.4) {
            scale = 0.0;
            amp = dirtyAt(posOptI_, posOptJ_);  // avoid divergence from huge amp
        }
        goodAspAmplitude_.push_back(amp);
        goodAspActiveSet_.push_back(scale);
        strengthOptimum_ = amp;
        optimumScaleSize_ = scale;
        return goodAspActiveSet_;
    }

    double peakResidualMasked() {
        double minVal = 0.0;
        double maxVal = 0.0;
        sumSquares_ = 0.0;
        bboxSumSquares_ = 0.0;
        const Box& box = initScaleMaskBoxes_[0];
        for (int j = blc1_; j <= trc1_; ++j) {
            const bool inRows = (j >= box.j0 && j <= box.j1);
            const T* drow = residual_ + static_cast<std::size_t>(j) * nx_;
            const double* mrow = initScaleMasks_[0].d.data() + static_cast<std::size_t>(j) * nx_;
            for (int i = blc0_; i <= trc0_; ++i) {
                const double d = static_cast<double>(drow[i]);
                bboxSumSquares_ += d * d;
                if (inRows && i >= box.i0 && i <= box.i1) {
                    const double v = d * mrow[i];
                    sumSquares_ += v * v;
                    if (v > maxVal) maxVal = v;
                    if (v < minVal) minVal = v;
                }
            }
        }
        peakSigned_ = std::fabs(maxVal) >= std::fabs(minVal) ? maxVal : minVal;
        return std::max(std::fabs(maxVal), std::fabs(minVal));
    }

    void recordComponent(double peak, double rms, int i, int j, double scale, double flux) {
        if (!options_.record_iteration_history) return;
        timing_.history_peak.push_back(peak);
        timing_.history_rms.push_back(rms);
        timing_.history_l.push_back(j);
        timing_.history_m.push_back(i);
        timing_.history_scale.push_back(scale);
        timing_.history_flux.push_back(flux);
    }
    double maskedRms() const { return nMasked_ > 0 ? std::sqrt(sumSquares_ / nMasked_) : 0.0; }

    // The model update (reference aspclean()).
    int aspclean() {
        double totalFlux = 0.0;
        int converged = 0;
        int stopPointModeCounter = 0;
        double initRMSResidual = 1000.0;

        iteration_ = startingIter_;
        if (options_.record_iteration_history) {
            const Box& box = initScaleMaskBoxes_[0];
            for (int j = box.j0; j <= box.j1; ++j)
                for (int i = box.i0; i <= box.i1; ++i) nMasked_ += initScaleMasks_[0](i, j) != 0.0;
        }
        if (options_.asp_residual_components) {
            if (residualComponentPhase() == deconvolvers::STOP_DIVERGED) {
                stopCode_ = deconvolvers::STOP_DIVERGED;
                peakResidual_ = peakResidualMasked();
                return -3;
            }
            startingIter_ = iteration_;
            if (options_.asp_scale_significance) estimateScaleNoise();
            std::vector<double> scaleSizes = getActiveSetAspen();
            scaleSizes.push_back(0.0);
            defineAspScales(scaleSizes);
        }

        peakResidual_ = peakResidualMasked();
        const int num = std::max(1, (trc0_ - blc0_) * (trc1_ - blc1_));
        auto rmsResidual = [&]() { return bboxSumSquares_ / num; };
        initRMSResidual = rmsResidual();

        auto clock = std::chrono::steady_clock::now();
        for (int ii = startingIter_; ii < maxNiter_; ++ii) {
            ++iteration_;
            lap(clock);
            const auto iterationStart = clock;
            const double rms = rmsResidual();
            double usedScale = switchedToHogbom_ ? 0.0 : optimumScaleSize_;
            timing_.t_bookkeeping += lap(clock);

            if (normMethod_ == 1) {
                if (!switchedToHogbom_ &&
                    (std::abs(peakResidual_) < fusedThreshold_ ||
                     ((std::abs(strengthOptimum_) < (5e-4 * fusedThreshold_)) && (numNoChange_ >= 2)))) {
                    bool runlong = (initRMSResidual > rms && initRMSResidual / rms < 1.5);
                    switchedToHogbom(runlong);
                    timing_.n_switch_fused += switchedToHogbom_;
                    if (numNoChange_ >= 2) numNoChange_ = 0;
                }
                if (!switchedToHogbom_ && numNoChange_ >= 2) {
                    numNoChange_ = 0;
                    bool runlong = (initRMSResidual > rms && initRMSResidual / rms < 1.5);
                    switchedToHogbom(runlong);
                    timing_.n_switch_no_change += switchedToHogbom_;
                }
            }

            if (!switchedToHogbom_) {
                if (numIterNoGoodAspen_.size() >= 10) numIterNoGoodAspen_.pop_front();
                numIterNoGoodAspen_.push_back(optimumScaleSize_ == 0 ? 1 : 0);
            }

            totalFlux += (strengthOptimum_ * gain_);
            totalFlux_ = totalFlux;
            if (ii == startingIter_) maximumResidual_ = std::abs(peakResidual_);

            bool significantScale = false;
            if (options_.asp_scale_significance && std::abs(peakResidual_) < computeThreshold() &&
                !noiseScale_.empty()) {
                if (!lastSearchAllScales_ || switchedToHogbom_) {
                    switchedToHogbom_ = false;
                    numHogbomIter_ = 0;
                    getActiveSetAspen();
                }
                significantScale = optimumScale_ > 0 && significanceReached();
                if (significantScale) usedScale = optimumScaleSize_;
            }
            if (std::abs(peakResidual_) < computeThreshold() && !significantScale) {
                converged = 1;
                stopCode_ = deconvolvers::STOP_THRESHOLD;
                switchedToHogbom_ = false;
                break;
            }
            if (stopTest_.check_divergence(std::abs(peakResidual_), sumSquares_, divergence_) !=
                deconvolvers::STOP_NONE) {
                converged = -3;
                stopCode_ = deconvolvers::STOP_DIVERGED;
                switchedToHogbom_ = false;
                break;
            }
            if (std::abs(strengthOptimum_) < (1e-6 * fusedThreshold_)) {
                converged = 1;
                stopCode_ = deconvolvers::STOP_ALGORITHM;
                switchedToHogbom_ = false;
                break;
            }
            if ((nInitScales_ > 1) && stopAtLargeScaleNegative_ && optimumScale_ == (nInitScales_ - 1) &&
                strengthOptimum_ < 0.0) {
                converged = -2;
                stopCode_ = deconvolvers::STOP_ALGORITHM;
                break;
            }
            if (stopPointMode_ > 0) {
                stopPointModeCounter = (optimumScale_ == 0) ? stopPointModeCounter + 1 : 0;
                if (stopPointModeCounter >= stopPointMode_) {
                    didStopPointMode_ = true;
                    converged = -1;
                    stopCode_ = deconvolvers::STOP_ALGORITHM;
                    break;
                }
            }
            const double iterationGain =
                (significantScale && options_.asp_significance_gain > 0.0) ? options_.asp_significance_gain : gain_;
            const double scaleFactor = iterationGain * strengthOptimum_;
            const double peakBefore = peakSigned_;
            double componentFlux = scaleFactor;
            if (options_.record_components) {
                components_.push_back(static_cast<double>(posOptI_));
                components_.push_back(static_cast<double>(posOptJ_));
                components_.push_back(usedScale);
                components_.push_back(scaleFactor);
            }
            lap(clock);
            if (usedScale == 0.0) {
                modelAdd(posOptI_, posOptJ_, scaleFactor);
                const int di = nx_ / 2 - posOptI_, dj = ny_ / 2 - posOptJ_;
                const int i0 = std::max(0, -di), i1 = std::min(nx_, nx_ - di);
                const int j0 = std::max(0, -dj), j1 = std::min(ny_, ny_ - dj);
                if (j1 > j0) {
                    parallel_for(threads_for(nthreads(), double(j1 - j0) * std::max(0, i1 - i0), kElemGrain),
                                 static_cast<std::size_t>(j1 - j0), [&](std::size_t b, std::size_t e) {
                                     for (std::size_t r = b; r < e; ++r) {
                                         const int j = j0 + static_cast<int>(r);
                                         const double* prow = psfd_.data() + static_cast<std::size_t>(j + dj) * nx_;
                                         T* drow = residual_ + static_cast<std::size_t>(j) * nx_;
                                         for (int i = i0; i < i1; ++i)
                                             drow[i] = static_cast<T>(static_cast<double>(drow[i]) -
                                                                      scaleFactor * prow[i + di]);
                                     }
                                 });
                }
                timing_.t_update_point += lap(clock);
                timing_.n_point_components += 1;
            } else {
                const std::vector<double> rx = gaussian_profile(nx_, posOptI_, usedScale, 0, nx_ - 1);
                const std::vector<double> ry = gaussian_profile(ny_, posOptJ_, usedScale, 0, ny_ - 1);
                std::vector<double> gx(padNx_, 0.0), gy(padNy_, 0.0);
                std::copy(rx.begin(), rx.end(), gx.begin());
                std::copy(ry.begin(), ry.end(), gy.begin());
                const std::vector<cd> X = aspfft::rfft1d_half(gx);
                const std::vector<cd> Y = aspfft::fft1d_forward(gy);
                const double norm = 1.0 / (kSqrtTwoPi * usedScale);
                componentFlux = scaleFactor * norm * std::accumulate(rx.begin(), rx.end(), 0.0) *
                                std::accumulate(ry.begin(), ry.end(), 0.0);
                scaleUpdate(X, Y, rx, ry, norm, scaleFactor);
                timing_.t_update_scale += lap(clock);
                timing_.n_scale_components += 1;
            }

            prevPeakResidual_ = peakResidual_;
            peakResidual_ = peakResidualMasked();
            recordComponent(peakBefore, maskedRms(), posOptI_, posOptJ_, usedScale, componentFlux);
            timing_.t_bookkeeping += lap(clock);
            const double noChange = options_.asp_no_change_tolerance > 0.0
                                        ? options_.asp_no_change_tolerance * std::fabs(prevPeakResidual_)
                                        : 1e-4;
            if (!switchedToHogbom_ && std::fabs(peakResidual_ - prevPeakResidual_) < noChange) numNoChange_ += 1;

            if (switchedToHogbom_) {
                if (numHogbomIter_ == 0) {
                    switchedToHogbom_ = false;
                    if (stopAfterHogbomRun_ && !(initRMSResidual > rms && initRMSResidual / rms < 1.5)) {
                        converged = 1;
                        stopCode_ = deconvolvers::STOP_ALGORITHM;
                        break;
                    }
                } else {
                    numHogbomIter_ -= 1;
                }
            }

            std::vector<double> tempScaleSizes = getActiveSetAspen();
            tempScaleSizes.push_back(0.0);
            defineAspScales(tempScaleSizes);
            if (significantScale) {
                timing_.n_significance_components += 1;
                timing_.t_significance +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - iterationStart).count();
            }
        }
        if (!converged && maxNiter_ > startingIter_) stopCode_ = deconvolvers::STOP_MAX_ITER;
        if (!converged && verbose_) std::fprintf(stderr, "[aspclean] failed to reach stopping threshold\n");
        return converged;
    }

    // The residual update of a scale component (reference: the separable
    // spectrum times the PSF spectrum on the padded grid, inverted; model and
    // residual updated on the image), with only the image rows inverted and
    // each row applied as it is made.
    void scaleUpdate(const std::vector<cd>& X, const std::vector<cd>& Y, const std::vector<double>& rx,
                     const std::vector<double>& ry, double norm, double scaleFactor) {
        const std::size_t hx = static_cast<std::size_t>(padNx_) / 2 + 1;
        const std::size_t nt = nthreads();
        padSpec_.resize(static_cast<std::size_t>(ny_) * hx);
        auto plan = pd::get_plan<pd::pocketfft_c<double>>(static_cast<std::size_t>(padNy_));
        aspfft2::column_blocks(hx, static_cast<std::size_t>(padNy_), 1, nt,
                               [&](std::size_t k0, std::size_t nl, std::vector<Buf<cv>>& bufs) {
                                   cv* P = bufs[0].data();
                                   const cv xk = aspfft2::vlanes(X.data(), k0, nl);
                                   for (int ky = 0; ky < padNy_; ++ky) {
                                       const cd ay = norm * Y[static_cast<std::size_t>(ky)];
                                       const cv p =
                                           aspfft2::vlanes(psfXfrPad_.data() + static_cast<std::size_t>(ky) * hx, k0, nl);
                                       P[ky] = aspfft2::vmul(aspfft2::vmul(aspfft2::vbroadcast(ay), xk), p);
                                   }
                                   plan->exec(P, 1.0, false);
                                   aspfft2::scatter(P, 0, static_cast<std::size_t>(ny_), padSpec_.data(), hx, k0, nl);
                               });
        const double fct = 1.0 / (static_cast<double>(padNx_) * static_cast<double>(padNy_));
        aspfft2::c2r_rows(static_cast<std::size_t>(padNx_), static_cast<std::size_t>(ny_), fct, nt,
                          [&](std::size_t r) { return padSpec_.data() + r * hx; },
                          [&](std::size_t r, const double* row) {
                              const int j = static_cast<int>(r);
                              const double gyj = norm * ry[static_cast<std::size_t>(j)];
                              T* mrow = model_ + static_cast<std::size_t>(j) * nx_;
                              T* drow = residual_ + static_cast<std::size_t>(j) * nx_;
                              for (int i = 0; i < nx_; ++i) {
                                  mrow[i] += static_cast<T>(scaleFactor * (gyj * rx[static_cast<std::size_t>(i)]));
                                  drow[i] = static_cast<T>(static_cast<double>(drow[i]) - scaleFactor * row[i]);
                              }
                          });
    }
};

template <typename T>
AspResult clean_plane(T* residual, T* model, Unaligned<T> psf, Unaligned<T> mask, int nx, int ny, double gain,
                      double threshold_per_cycle, int max_iter_per_cycle_updated, double asp_fused_threshold,
                      double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                      int asp_scale_normalization_method, bool asp_stop_after_hogbom_run, bool verbose,
                      const deconvolvers::DivergenceControl<double>& divergence, const FftThreads& fft_threads,
                      const AspOptions& options = AspOptions{}) {
    std::vector<std::pair<std::size_t, T>> nonFinite;
    if (mask) {
        const std::size_t n = static_cast<std::size_t>(nx) * ny;
        for (std::size_t k = 0; k < n; ++k)
            if (!(static_cast<double>(mask[k]) > kMaskThreshold) && !std::isfinite(residual[k])) {
                nonFinite.emplace_back(k, residual[k]);
                residual[k] = T(0);
            }
    }
    Cleaner<T> cleaner(residual, model, psf, mask, nx, ny, gain, threshold_per_cycle, max_iter_per_cycle_updated,
                       asp_fused_threshold, asp_psf_width, asp_largest_scale, asp_stop_point_mode,
                       asp_scale_normalization_method, asp_stop_after_hogbom_run, verbose, divergence, fft_threads,
                       options);
    const AspResult result = cleaner.run();
    for (const auto& [k, value] : nonFinite) residual[k] = value;
    return result;
}

// Whether one of the n items of `a` is not finite or of magnitude 2^64 or
// more, that is, has a biased exponent e of at least bias + 64: exactly then
// does e + (2^bits - bias - 64), added in the exponent field, carry into the
// top bit of the word. An and, an add and an or per item, which vectorise;
// `threads` share the items.
template <typename T>
bool any_wild(Unaligned<T> a, std::size_t n, std::size_t threads) {
    static_assert(std::numeric_limits<T>::is_iec559 && (sizeof(T) == 4 || sizeof(T) == 8));
    using U = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
    constexpr int kExponentBits = sizeof(T) == 8 ? 11 : 8;
    constexpr int kFractionBits = std::numeric_limits<T>::digits - 1;
    constexpr U kBias = (U(1) << (kExponentBits - 1)) - 1;
    constexpr U kExponent = ((U(1) << kExponentBits) - 1) << kFractionBits;
    constexpr U kCarry = ((U(1) << kExponentBits) - kBias - 64) << kFractionBits;
    std::atomic<bool> wild{false};
    parallel_for(threads_for(threads, double(n), 8 * kElemGrain), n, [&](std::size_t b, std::size_t e) {
        U any = 0;
        for (std::size_t k = b; k < e; ++k) {
            U bits;
            std::memcpy(&bits, a.bytes + k * sizeof(T), sizeof(T));
            any |= (bits & kExponent) + kCarry;
        }
        if (any >> (8 * sizeof(U) - 1)) wild.store(true, std::memory_order_relaxed);
    });
    return wild.load(std::memory_order_relaxed);
}

}  // namespace

void check_controls(double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                    int asp_scale_normalization_method, double asp_min_mask_coverage) {
    if (asp_psf_width != -1.0 && !(std::isfinite(asp_psf_width) && asp_psf_width > 0.0))
        throw std::invalid_argument("asp_psf_width must be positive, or -1 to estimate it");
    if (asp_largest_scale < -1)
        throw std::invalid_argument("asp_largest_scale must be at least 0, or -1 for the default scales");
    if (asp_stop_point_mode == 0 || asp_stop_point_mode < -1)
        throw std::invalid_argument("asp_stop_point_mode must be positive, or -1 to disable it");
    if (asp_scale_normalization_method != 1 && asp_scale_normalization_method != 2)
        throw std::invalid_argument("asp_scale_normalization_method must be 1 or 2");
    if (!(asp_min_mask_coverage >= 0.0 && asp_min_mask_coverage < 1.0))
        throw std::invalid_argument("asp_min_mask_coverage must be at least 0 and below 1");
}

template <typename T>
bool needs_reference(Unaligned<T> residual, Unaligned<T> model, Unaligned<T> psf, Unaligned<T> mask,
                     std::size_t n, std::size_t npsf, int threads) {
    const std::size_t nt = static_cast<std::size_t>(std::max(1, threads));
    if (any_wild(psf, npsf, nt) || any_wild(model, n, nt)) return true;
    if (!any_wild(residual, n, nt)) return false;
    // The residual has a wild value: is it one the model update reads? A value
    // that is not finite outside the search region is zero during the model
    // update (clean_plane), as in the reference.
    for (std::size_t k = 0; k < n; ++k) {
        const double r = static_cast<double>(residual[k]);
        if (std::isfinite(r) ? !(std::fabs(r) < 0x1p64)
                             : (!mask || static_cast<double>(mask[k]) > kMaskThreshold))
            return true;
    }
    return false;
}

template <typename T>
AspResult aspclean_plane(T* residual, T* model, Unaligned<T> psf, Unaligned<T> mask, int nx, int ny, double gain,
                         double threshold_per_cycle, int max_iter_per_cycle_updated, double asp_fused_threshold,
                         double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                         int asp_scale_normalization_method, bool asp_stop_after_hogbom_run, bool verbose,
                         const deconvolvers::DivergenceControl<double>& divergence, int fft_threads,
                         const AspOptions& options) {
    check_controls(asp_psf_width, asp_largest_scale, asp_stop_point_mode, asp_scale_normalization_method,
                   options.asp_min_mask_coverage);
    return clean_plane<T>(residual, model, psf, mask, nx, ny, gain, threshold_per_cycle, max_iter_per_cycle_updated,
                          asp_fused_threshold, asp_psf_width, asp_largest_scale, asp_stop_point_mode,
                          asp_scale_normalization_method, asp_stop_after_hogbom_run, verbose, divergence,
                          FftThreads{std::max(1, fft_threads), nullptr}, options);
}

template <typename T>
void aspclean_cube(T* residual, T* model, Unaligned<T> psf, Unaligned<T> mask, int nt, int nf, int np_img,
                   int np_psf, int nx, int ny, double gain, const double* threshold_per_cycle,
                   const int* max_iter_per_cycle_updated, double asp_fused_threshold, double asp_psf_width,
                   int asp_largest_scale, int asp_stop_point_mode, int asp_scale_normalization_method,
                   bool asp_stop_after_hogbom_run, int processing_function_threads,
                   const deconvolvers::DivergenceControl<double>& divergence, AspResult* results,
                   const AspOptions& options) {
    check_controls(asp_psf_width, asp_largest_scale, asp_stop_point_mode, asp_scale_normalization_method,
                   options.asp_min_mask_coverage);
    const int nplanes = nt * nf * np_img;
    if (nplanes <= 0) return;
    const std::size_t plane = static_cast<std::size_t>(ny) * nx;
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
            const std::size_t img_off = ((static_cast<std::size_t>(tt) * nf + nn) * np_img + pp) * plane;
            const std::size_t psf_off = ((static_cast<std::size_t>(tt) * nf + nn) * np_psf + pidx) * plane;
            results[p] = clean_plane<T>(residual + img_off, model + img_off, psf.at(psf_off), mask.at(img_off), nx,
                                        ny, gain, threshold_per_cycle[p], max_iter_per_cycle_updated[p],
                                        asp_fused_threshold,
                                        asp_psf_width, asp_largest_scale, asp_stop_point_mode,
                                        asp_scale_normalization_method, asp_stop_after_hogbom_run, false, divergence,
                                        fft_threads, options);
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

template bool needs_reference<float>(Unaligned<float>, Unaligned<float>, Unaligned<float>, Unaligned<float>,
                                      std::size_t, std::size_t, int);
template bool needs_reference<double>(Unaligned<double>, Unaligned<double>, Unaligned<double>, Unaligned<double>,
                                       std::size_t, std::size_t, int);
template AspResult aspclean_plane<float>(float*, float*, Unaligned<float>, Unaligned<float>, int, int, double, double,
                                         int, double, double, int, int, int, bool, bool,
                                         const deconvolvers::DivergenceControl<double>&, int, const AspOptions&);
template AspResult aspclean_plane<double>(double*, double*, Unaligned<double>, Unaligned<double>, int, int, double,
                                          double, int, double, double, int, int, int, bool, bool,
                                          const deconvolvers::DivergenceControl<double>&, int, const AspOptions&);
template void aspclean_cube<float>(float*, float*, Unaligned<float>, Unaligned<float>, int, int, int, int, int, int,
                                   double, const double*, const int*, double, double, int, int, int, bool, int,
                                   const deconvolvers::DivergenceControl<double>&, AspResult*, const AspOptions&);
template void aspclean_cube<double>(double*, double*, Unaligned<double>, Unaligned<double>, int, int, int, int, int,
                                    int, double, const double*, const int*, double, double, int, int, int, bool, int,
                                    const deconvolvers::DivergenceControl<double>&, AspResult*, const AspOptions&);

}  // namespace aspclean_fast
