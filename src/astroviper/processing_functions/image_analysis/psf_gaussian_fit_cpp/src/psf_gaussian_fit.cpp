// CASA-compatible PSF Gaussian beam fit -- see psf_gaussian_fit.hpp.
//
// The structure mirrors casa6 StokesImageUtil::FitGaussianPSF (CAS-13022):
//   1. locate + normalise the PSF peak,
//   2. FindNpoints: walk outward from the peak collecting main-lobe pixels
//      above the cutoff (never crossing below it, so sidelobes are excluded),
//   3. window the main lobe (+5 pixel border), oversample it with the
//      Numerical-Recipes bicubic used by casacore Interpolate2D::CUBIC to
//      ~3001 points, and re-collect the points on the oversampled grid,
//   4. Levenberg-Marquardt fit of the casacore Gaussian2D functional with
//      height and centre fixed (free: major-axis FWHM, axial ratio, position
//      angle), retrying with rotated initial position angles when the normal
//      equations are singular (CAS-13515),
//   5. on non-convergence divide the cutoff by 1.5 and repeat (>= 0.009,
//      at most 50 rounds); the final fallback is a round 2.5-pixel beam.

#include "psf_gaussian_fit.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace astroviper::psf_gaussian_fit {

namespace {

constexpr double kLn16 = 2.772588722239781;  // ln(16)

struct Points {
    std::vector<double> x0;  // offset along axis 0 (radians)
    std::vector<double> x1;  // offset along axis 1 (radians)
    std::vector<double> y;   // psf value
};

// Port of StokesImageUtil::FindNpoints: collect main-lobe pixels above amin by
// walking rows outward from (px, py), alternating direction, and walking each
// row outward from the centre column, stopping when leaving the lobe. Also
// returns the (squared) bounding box of the collected pixels.
void find_n_points(int nrow, double amin, int px, int py, double dx, double dy,
                   const std::vector<double>& psf, int nx, int ny, Points& pts,
                   int blc[2], int trc[2]) {
    const int maxnpoints = (2 * nrow + 1) * (2 * nrow + 1);
    pts.x0.clear();
    pts.x1.clear();
    pts.y.clear();
    pts.x0.reserve(maxnpoints);
    pts.x1.reserve(maxnpoints);
    pts.y.reserve(maxnpoints);

    blc[0] = nx - 1;
    blc[1] = ny - 1;
    trc[0] = 0;
    trc[1] = 0;

    auto value = [&](int i, int j) { return psf[static_cast<std::size_t>(i) * ny + j]; };

    int npoints = 0;
    int iflip = 1;
    int jflip = 1;
    bool done = false;
    for (int jlo = 0; jlo < 2 && !done; ++jlo) {
        jflip *= -1;
        for (int j = jlo; j <= nrow && !done; ++j) {
            const int jrow = py + j * jflip;
            for (int ilo = 0; ilo < 2 && !done; ++ilo) {
                iflip *= -1;
                if (jrow > ny - 1 || jrow < 0) break;
                bool inlobe = value(px, jrow) > amin;
                for (int i = ilo; i <= nrow; ++i) {
                    if (npoints >= maxnpoints) break;
                    const int irow = px + i * iflip;
                    if (irow > nx - 1 || irow < 0) break;
                    if (inlobe && value(irow, jrow) < amin) break;
                    if (value(irow, jrow) > amin) {
                        inlobe = true;
                        pts.x0.push_back((irow - px) * std::abs(dx));
                        pts.x1.push_back((jrow - py) * std::abs(dy));
                        pts.y.push_back(value(irow, jrow));
                        blc[0] = std::min(blc[0], irow);
                        blc[1] = std::min(blc[1], jrow);
                        trc[0] = std::max(trc[0], irow);
                        trc[1] = std::max(trc[1], jrow);
                        ++npoints;
                        if (npoints > maxnpoints - 1) {
                            done = true;  // CASA: goto endSearch
                            break;
                        }
                    }
                }
            }
        }
    }

    // CASA squares the bounding box.
    blc[0] = blc[1] = std::min(blc[0], blc[1]);
    trc[0] = trc[1] = std::max(trc[0], trc[1]);
}

// Numerical-Recipes bcucof (casacore Interpolate2D::bcucof), d1 = d2 = 1.
void bcucof(double c[4][4], const double y[4], const double y1[4],
            const double y2[4], const double y12[4]) {
    static const double wt[16][16] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0},
        {2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1},
        {0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1},
        {-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0},
        {9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2},
        {-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2},
        {2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0},
        {-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1},
        {4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1}};
    double cl[16];
    double xx[16];
    for (int i = 0; i < 4; ++i) {
        xx[i] = y[i];
        xx[i + 4] = y1[i];
        xx[i + 8] = y2[i];
        xx[i + 12] = y12[i];
    }
    for (int i = 0; i < 16; ++i) {
        double v = 0.0;
        for (int k = 0; k < 16; ++k) v += wt[i][k] * xx[k];
        cl[i] = v;
    }
    int l = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) c[i][j] = cl[l++];
}

// casacore Interpolate2D::interpLinear (bilinear; the cubic's edge fallback).
double interp_linear(double wx, double wy, const std::vector<double>& data,
                     int nx, int ny) {
    int i = static_cast<int>(std::floor(wx));
    int j = static_cast<int>(std::floor(wy));
    i = std::max(0, std::min(i, nx - 2));
    j = std::max(0, std::min(j, ny - 2));
    const double t = wx - i;
    const double u = wy - j;
    auto v = [&](int a, int b) { return data[static_cast<std::size_t>(a) * ny + b]; };
    return (1 - t) * (1 - u) * v(i, j) + t * (1 - u) * v(i + 1, j) +
           t * u * v(i + 1, j + 1) + (1 - t) * u * v(i, j + 1);
}

// casacore Interpolate2D::interpCubic (Numerical Recipes 3.6 bicubic with
// centred finite-difference derivatives; linear at the edges).
double interp_cubic(double wx, double wy, const std::vector<double>& data,
                    int nx, int ny) {
    const int i = static_cast<int>(wx);
    const int j = static_cast<int>(wy);
    if (i <= 0 || i >= nx - 2 || j <= 0 || j >= ny - 2)
        return interp_linear(wx, wy, data, nx, ny);

    auto v = [&](int a, int b) { return data[static_cast<std::size_t>(a) * ny + b]; };
    const double tt = wx - i;
    const double uu = wy - j;

    double y[4] = {v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)};
    double y1[4] = {v(i + 1, j) - v(i - 1, j), v(i + 2, j) - v(i, j),
                    v(i + 2, j + 1) - v(i, j + 1), v(i + 1, j + 1) - v(i - 1, j + 1)};
    double y2[4] = {v(i, j + 1) - v(i, j - 1), v(i + 1, j + 1) - v(i + 1, j - 1),
                    v(i + 1, j + 2) - v(i + 1, j), v(i, j + 2) - v(i, j)};
    double y12[4] = {
        v(i + 1, j + 1) + v(i - 1, j - 1) - v(i - 1, j + 1) - v(i + 1, j - 1),
        v(i + 2, j + 1) + v(i, j - 1) - v(i, j + 1) - v(i + 2, j - 1),
        v(i + 2, j + 2) + v(i, j) - v(i, j + 2) - v(i + 2, j),
        v(i + 1, j + 2) + v(i - 1, j) - v(i - 1, j + 2) - v(i + 1, j)};
    for (int k = 0; k < 4; ++k) {
        y1[k] /= 2.0;
        y2[k] /= 2.0;
        y12[k] /= 4.0;
    }
    double c[4][4];
    bcucof(c, y, y1, y2, y12);
    double result = 0.0;
    for (int k = 3; k >= 0; --k)
        result = tt * result + ((c[k][3] * uu + c[k][2]) * uu + c[k][1]) * uu + c[k][0];
    return result;
}

// StokesImageUtil::ResamplePSF with the CUBIC method.
void resample_psf(const std::vector<double>& psf, int nx, int ny,
                  int oversampling, std::vector<double>& out, int& onx,
                  int& ony) {
    onx = nx * oversampling - oversampling + 1;
    ony = ny * oversampling - oversampling + 1;
    out.assign(static_cast<std::size_t>(onx) * ony, 0.0);
    for (int i = 0; i < onx; ++i) {
        for (int j = 0; j < ony; ++j) {
            const double wx = static_cast<double>(i) / oversampling;
            const double wy = static_cast<double>(j) / oversampling;
            out[static_cast<std::size_t>(i) * ony + j] =
                interp_cubic(wx, wy, psf, nx, ny);
        }
    }
}

// casacore Gaussian2D with height = 1 and centre = 0 fixed:
//   A = cos(pa) x + sin(pa) y ; B = -sin(pa) x + cos(pa) y
//   f = exp(-ln16 (A^2 / (w r)^2 + B^2 / w^2))
// with w the major-axis FWHM (parameter YWIDTH) and r the axial ratio.
struct Model {
    double w, r, pa;
};

void model_and_jacobian(const Model& m, double x, double y, double& f,
                        double j[3]) {
    const double c = std::cos(m.pa);
    const double s = std::sin(m.pa);
    const double a = c * x + s * y;
    const double b = -s * x + c * y;
    const double wr = m.w * m.r;
    const double qa = a * a / (wr * wr);
    const double qb = b * b / (m.w * m.w);
    const double q = kLn16 * (qa + qb);
    f = std::exp(-q);
    j[0] = f * 2.0 * q / m.w;                          // d f / d w
    j[1] = f * 2.0 * kLn16 * qa / m.r;                 // d f / d r
    j[2] = -f * 2.0 * kLn16 * a * b * (1.0 / (m.r * m.r) - 1.0) / (m.w * m.w);
}

double chi_squared(const Model& m, const Points& pts) {
    double chi2 = 0.0;
    double f, j[3];
    for (std::size_t k = 0; k < pts.y.size(); ++k) {
        model_and_jacobian(m, pts.x0[k], pts.x1[k], f, j);
        const double r = pts.y[k] - f;
        chi2 += r * r;
    }
    return chi2;
}

bool solve3(const double a[3][3], const double b[3], double x[3]) {
    // Cramer's rule with a singularity guard (the CAS-13515 failure mode).
    const double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
                       a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
                       a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    double scale = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) scale = std::max(scale, std::abs(a[i][j]));
    if (!(std::abs(det) > 1e-30 * scale * scale * scale)) return false;
    double m[3][3];
    for (int col = 0; col < 3; ++col) {
        std::memcpy(m, a, sizeof(m));
        for (int i = 0; i < 3; ++i) m[i][col] = b[i];
        x[col] = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                  m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                  m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) /
                 det;
    }
    return true;
}

// Levenberg-Marquardt for the 3 free Gaussian parameters. Mirrors the
// casacore NonLinearFitLM configuration used by CASA: at most 1000
// iterations, convergence when the chi-squared change falls below
// 0.0001 relative. Returns false ("error in loop solution") when the
// normal equations are singular, with `converged` reporting convergence.
bool levenberg_marquardt(Model& m, const Points& pts, bool& converged) {
    converged = false;
    double lambda = 1e-3;
    double chi2 = chi_squared(m, pts);
    for (int iter = 0; iter < 1000; ++iter) {
        double jtj[3][3] = {{0}};
        double jtr[3] = {0, 0, 0};
        double f, j[3];
        for (std::size_t k = 0; k < pts.y.size(); ++k) {
            model_and_jacobian(m, pts.x0[k], pts.x1[k], f, j);
            const double r = pts.y[k] - f;
            for (int a = 0; a < 3; ++a) {
                jtr[a] += j[a] * r;
                for (int b = 0; b < 3; ++b) jtj[a][b] += j[a] * j[b];
            }
        }
        bool stepped = false;
        for (int tries = 0; tries < 40; ++tries) {
            double damped[3][3];
            std::memcpy(damped, jtj, sizeof(damped));
            for (int a = 0; a < 3; ++a) damped[a][a] *= (1.0 + lambda);
            double delta[3];
            if (!solve3(damped, jtr, delta)) {
                // Degenerate position angle (circular beam: the pa column of
                // the Jacobian vanishes at ratio 1). Solve the 2x2 (width,
                // ratio) subsystem with the angle held fixed; only a failure
                // of that too is the CAS-13515 "loop solution" error.
                const double det2 =
                    damped[0][0] * damped[1][1] - damped[0][1] * damped[1][0];
                const double scale2 = std::max(
                    {std::abs(damped[0][0]), std::abs(damped[0][1]),
                     std::abs(damped[1][0]), std::abs(damped[1][1])});
                if (!(std::abs(det2) > 1e-30 * scale2 * scale2)) return false;
                delta[0] = (jtr[0] * damped[1][1] - jtr[1] * damped[0][1]) / det2;
                delta[1] = (damped[0][0] * jtr[1] - damped[1][0] * jtr[0]) / det2;
                delta[2] = 0.0;
            }
            const Model trial{m.w + delta[0], m.r + delta[1], m.pa + delta[2]};
            const double trial_chi2 = chi_squared(trial, pts);
            if (trial_chi2 <= chi2) {
                const double change = chi2 - trial_chi2;
                m = trial;
                lambda = std::max(lambda * 0.1, 1e-12);
                stepped = true;
                if (change < 1e-4 * std::max(trial_chi2, 1e-30)) {
                    converged = true;
                    chi2 = trial_chi2;
                    return true;
                }
                chi2 = trial_chi2;
                break;
            }
            lambda *= 10.0;
            if (lambda > 1e12) break;
        }
        if (!stepped) {
            converged = true;  // cannot improve further: treat as converged
            return true;
        }
    }
    return true;  // solvable but not converged within the iteration budget
}

}  // namespace

int fit_plane(const double* psf_in, std::size_t nx_in, std::size_t ny_in,
              double delta_x, double delta_y, double psfcutoff,
              double beam[3]) {
    const int nx = static_cast<int>(nx_in);
    const int ny = static_cast<int>(ny_in);
    const double dx = std::abs(delta_x);
    const double dy = std::abs(delta_y);

    // Locate + validate the peak (StokesImageUtil::locatePeakPSF + checks).
    int px = 0, py = 0;
    double bamp = 0.0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const double v = psf_in[static_cast<std::size_t>(i) * ny + j];
            if (v > bamp) {
                bamp = v;
                px = i;
                py = j;
            }
        }
    }
    if (bamp < 1e-7) throw std::runtime_error("Psf peak is zero");
    if (px < nx / 4.0 || px > 3.0 * nx / 4.0 || py < ny / 4.0 ||
        py > 3.0 * ny / 4.0)
        throw std::runtime_error(
            "Peak of psf is outside the inner quarter of defined image");

    std::vector<double> lpsf(psf_in, psf_in + static_cast<std::size_t>(nx) * ny);
    for (auto& v : lpsf) v /= bamp;

    constexpr int kNrow = 20;          // CASA npix
    constexpr int kExpandPixel = 5;    // CASA expand_pixel
    constexpr int kTargetNpoints = 3001;

    double amin = psfcutoff;
    bool converged = false;
    Model solution{0.0, 0.0, 0.0};
    int kounter = 0;

    while (amin > 0.009 && !converged && kounter < 50) {
        ++kounter;
        Points pts;
        int blc[2], trc[2];
        find_n_points(kNrow, amin, px, py, dx, dy, lpsf, nx, ny, pts, blc, trc);

        blc[0] = std::max(blc[0] - kExpandPixel, 0);
        blc[1] = std::max(blc[1] - kExpandPixel, 0);
        trc[0] = std::min(trc[0] + kExpandPixel, nx - 1);
        trc[1] = std::min(trc[1] + kExpandPixel, ny - 1);

        const int wnx = trc[0] - blc[0] + 1;
        const int wny = trc[1] - blc[1] + 1;
        std::vector<double> windowed(static_cast<std::size_t>(wnx) * wny);
        for (int i = 0; i < wnx; ++i)
            for (int j = 0; j < wny; ++j)
                windowed[static_cast<std::size_t>(i) * wny + j] =
                    lpsf[static_cast<std::size_t>(i + blc[0]) * ny + (j + blc[1])];

        int oversampling = static_cast<int>(
            std::sqrt(static_cast<double>(kTargetNpoints) /
                      (static_cast<double>(wnx) * wny)));
        if (oversampling == 0) oversampling = 1;

        std::vector<double> resampled;
        int onx, ony;
        resample_psf(windowed, wnx, wny, oversampling, resampled, onx, ony);

        double maxval = 0.0;
        int mx = 0, my = 0;
        for (int i = 0; i < onx; ++i)
            for (int j = 0; j < ony; ++j) {
                const double v = resampled[static_cast<std::size_t>(i) * ony + j];
                if (v > maxval) {
                    maxval = v;
                    mx = i;
                    my = j;
                }
            }
        for (auto& v : resampled) v /= maxval;

        const double rdx = dx / oversampling;
        const double rdy = dy / oversampling;
        const int min_len = std::min(trc[0] - blc[0], trc[1] - blc[1]) + 1;
        const int nrow_re = (oversampling * min_len - 1) / 2;

        int blc2[2], trc2[2];
        find_n_points(nrow_re, amin, mx, my, rdx, rdy, resampled, onx, ony, pts,
                      blc2, trc2);

        // CAS-13515: retry with rotated initial position angles when the
        // normal equations are singular.
        bool loop_solution_found = false;
        for (int retry = 0; retry < 10 && !loop_solution_found; ++retry) {
            Model trial{2.5 * dx, 0.5, 1.0 + retry * M_PI / 10.0};
            if (retry == 0) trial.pa = 1.0;
            loop_solution_found = levenberg_marquardt(trial, pts, converged);
            if (loop_solution_found) solution = trial;
        }
        if (!loop_solution_found)
            throw std::runtime_error(
                "Error in psf_gaussian_fit: error in loop solution.");
        if (!converged) amin /= 1.5;
    }

    if (!converged) {
        beam[0] = 2.5 * dx;
        beam[1] = 2.5 * dx;
        beam[2] = 0.0;
        return 1;
    }

    double major, minor, pa;
    if (std::abs(solution.r) > 1.0) {
        major = std::abs(solution.w * solution.r);
        minor = std::abs(solution.w);
        pa = solution.pa - M_PI / 2.0;
    } else {
        major = std::abs(solution.w);
        minor = std::abs(solution.w * solution.r);
        pa = solution.pa;
    }
    // CAS-8627: normalise the position angle into (-pi/2, pi/2].
    pa = std::fmod(pa, 2.0 * M_PI);
    while (std::abs(pa) > M_PI / 2.0) {
        if (pa > 1.5 * M_PI)
            pa -= 2.0 * M_PI;
        else if (pa > M_PI / 2.0)
            pa -= M_PI;
        else if (pa < -1.5 * M_PI)
            pa += 2.0 * M_PI;
        else
            pa += M_PI;
    }
    beam[0] = major;
    beam[1] = minor;
    beam[2] = pa;
    return 0;
}

int fit_planes(const double* psf, std::size_t n_plane, std::size_t nx,
               std::size_t ny, double delta_x, double delta_y, double psfcutoff,
               double* beams, int n_threads) {
    const std::size_t plane_size = nx * ny;
    std::vector<int> fallback(n_plane, 0);
    std::vector<std::string> errors(n_plane);

    auto work = [&](std::size_t begin, std::size_t end) {
        for (std::size_t p = begin; p < end; ++p) {
            try {
                fallback[p] = fit_plane(psf + p * plane_size, nx, ny, delta_x,
                                        delta_y, psfcutoff, beams + p * 3);
            } catch (const std::exception& err) {
                errors[p] = err.what();
            }
        }
    };

    const int usable = std::max(
        1, std::min<int>(n_threads, static_cast<int>(n_plane)));
    if (usable <= 1) {
        work(0, n_plane);
    } else {
        std::vector<std::thread> pool;
        const std::size_t chunk = (n_plane + usable - 1) / usable;
        for (int t = 0; t < usable; ++t) {
            const std::size_t begin = static_cast<std::size_t>(t) * chunk;
            const std::size_t end = std::min(n_plane, begin + chunk);
            if (begin >= end) break;
            pool.emplace_back(work, begin, end);
        }
        for (auto& th : pool) th.join();
    }

    for (const auto& message : errors)
        if (!message.empty()) throw std::runtime_error(message);
    int n_fallback = 0;
    for (int f : fallback) n_fallback += f;
    return n_fallback;
}

}  // namespace astroviper::psf_gaussian_fit
