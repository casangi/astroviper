#pragma once

// A small, self-contained limited-memory BFGS (L-BFGS) minimizer used by the
// Asp deconvolver to optimize each Aspen's (amplitude, scale) pair. It
// replaces the ALGLIB `minlbfgs*` routines used by the original CASA code
// (synthesis/MeasurementEquations/AspMatrixCleaner.cc, the "gold" objective)
// with an implementation built only on the C++ standard library.
//
// It reproduces the relevant behaviour of the original call:
//   minlbfgscreate(1, x, state);              // m = 1 correction pair
//   minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//   minlbfgssetscale(state, s);               // per-variable scale
// Like ALGLIB, the algorithm runs in the *scaled* variables z = x / s: the
// gradient, the step lengths and every stopping test are measured in z, and
// the very first step is the scaled steepest-descent direction with unit
// length (ALGLIB's `stp = 1 / |d|` on iteration one), so the first trial point
// moves each variable by at most its own scale. That bound is what keeps the
// optimization from jumping to absurd (amplitude, scale) pairs: an earlier
// version of this file used H0 = diag(s^2) instead, which for the Asp objective
// (gradients of order the image sum of squares) produced first trial steps of
// hundreds of scale lengths, tens of wasted line-search evaluations per Aspen
// (each costing FFTs) and occasional acceptance of garbage minima that then
// tripped the cleaner's divergence guard. Later steps use the standard
// two-loop recursion with H0 = (s'y / y'y) I and a backtracking Armijo line
// search with quadratic interpolation; the function value never increases.
// Exact bit-for-bit agreement with
// ALGLIB is neither expected nor required; functional equivalence (a few
// well-behaved descent steps from a good initial guess) is.

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace asplbfgs {

struct Options {
    int m = 1;                 // number of stored correction pairs
    int max_iters = 5;         // ALGLIB maxits
    double epsg = 1e-3;        // scaled-gradient stopping tolerance
    double epsf = 1e-3;        // relative function-change tolerance
    double epsx = 1e-3;        // scaled-step tolerance
    int max_line_search = 20;  // step halvings before the line search gives up
};

struct Report {
    int iterations = 0;   // accepted steps
    int evaluations = 0;  // objective (function + gradient) evaluations
    double f = 0.0;       // objective at the returned point
    bool line_search_failed = false;
};

// Objective signature: fills `grad` (same length as x) and returns f(x).
using ObjFunc = std::function<double(const std::vector<double>& x,
                                     std::vector<double>& grad)>;

namespace detail {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

}  // namespace detail

// Minimize `fg` starting from `x` (modified in place to hold the result).
// `scale` must be non-zero and the same length as `x` (its sign is ignored,
// as ALGLIB does).
inline Report minimize(std::vector<double>& x,
                       const std::vector<double>& scale,
                       const ObjFunc& fg,
                       const Options& opt = Options{}) {
    const std::size_t n = x.size();
    Report rep;

    std::vector<double> s(n);
    for (std::size_t k = 0; k < n; ++k) {
        s[k] = std::fabs(scale[k]);
        if (!(s[k] > 0.0) || !std::isfinite(s[k])) s[k] = 1.0;
    }

    // Scaled variables z = x / s; f~(z) = f(s z), grad f~ = s * grad f.
    std::vector<double> z(n), g(n), x_trial(n), g_trial(n), z_trial(n), dir(n), grad_x(n);
    auto evaluate = [&](const std::vector<double>& zz, std::vector<double>& gg) {
        for (std::size_t k = 0; k < n; ++k) x_trial[k] = zz[k] * s[k];
        const double f = fg(x_trial, grad_x);
        ++rep.evaluations;
        for (std::size_t k = 0; k < n; ++k) gg[k] = grad_x[k] * s[k];
        return f;
    };

    for (std::size_t k = 0; k < n; ++k) z[k] = x[k] / s[k];
    double f = evaluate(z, g);
    rep.f = f;
    if (!std::isfinite(f)) return rep;

    const int m = std::max(1, opt.m);
    std::vector<std::vector<double>> s_hist, y_hist;  // s = dz, y = dg
    std::vector<double> rho_hist;

    for (int iter = 0; iter < opt.max_iters; ++iter) {
        const double g_norm = detail::norm(g);
        if (g_norm <= opt.epsg) break;

        const int hist = static_cast<int>(s_hist.size());
        if (hist == 0) {
            // First step (or no curvature information yet): unit-length
            // steepest descent in the scaled variables.
            for (std::size_t k = 0; k < n; ++k) dir[k] = -g[k] / g_norm;
        } else {
            // Two-loop recursion with H0 = gamma I.
            std::vector<double> q = g;
            std::vector<double> alpha(hist, 0.0);
            for (int i = hist - 1; i >= 0; --i) {
                alpha[i] = rho_hist[i] * detail::dot(s_hist[i], q);
                for (std::size_t k = 0; k < n; ++k) q[k] -= alpha[i] * y_hist[i][k];
            }
            const double sy = detail::dot(s_hist.back(), y_hist.back());
            const double yy = detail::dot(y_hist.back(), y_hist.back());
            const double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
            for (std::size_t k = 0; k < n; ++k) dir[k] = gamma * q[k];
            for (int i = 0; i < hist; ++i) {
                const double beta = rho_hist[i] * detail::dot(y_hist[i], dir);
                for (std::size_t k = 0; k < n; ++k) dir[k] += s_hist[i][k] * (alpha[i] - beta);
            }
            for (std::size_t k = 0; k < n; ++k) dir[k] = -dir[k];
            if (detail::dot(dir, g) >= 0.0) {
                // Not a descent direction: fall back to unit steepest descent.
                for (std::size_t k = 0; k < n; ++k) dir[k] = -g[k] / g_norm;
            }
        }
        const double dg = detail::dot(dir, g);
        if (!(dg < 0.0)) break;

        // Backtracking Armijo line search from the unit step. A rejected step
        // is shortened by quadratic interpolation of phi(a) = f(z + a dir)
        // through phi(0), phi'(0) and the rejected phi(step) (Nocedal & Wright
        // 3.5), safeguarded to [0.1, 0.5] of the rejected step; plain halving
        // needed three to four times as many objective evaluations here.
        const double c1 = 1e-4;
        double step = 1.0;
        double f_trial = f;
        bool accepted = false;
        for (int ls = 0; ls < opt.max_line_search; ++ls) {
            for (std::size_t k = 0; k < n; ++k) z_trial[k] = z[k] + step * dir[k];
            f_trial = evaluate(z_trial, g_trial);
            if (std::isfinite(f_trial) && f_trial <= f + c1 * step * dg) {
                accepted = true;
                break;
            }
            double next = 0.5 * step;
            if (std::isfinite(f_trial)) {
                const double denom = 2.0 * (f_trial - f - dg * step);
                if (denom > 0.0) next = -dg * step * step / denom;
                next = std::min(0.5 * step, std::max(0.1 * step, next));
            } else {
                next = 0.1 * step;
            }
            step = next;
        }
        if (!accepted) {
            rep.line_search_failed = true;
            break;
        }

        std::vector<double> s_vec(n), y_vec(n);
        for (std::size_t k = 0; k < n; ++k) {
            s_vec[k] = z_trial[k] - z[k];
            y_vec[k] = g_trial[k] - g[k];
        }
        const double f_old = f;
        z = z_trial;
        g = g_trial;
        f = f_trial;
        rep.f = f;
        rep.iterations = iter + 1;

        if (std::fabs(f_old - f) <= opt.epsf * std::max({std::fabs(f_old), std::fabs(f), 1.0}))
            break;
        if (detail::norm(s_vec) <= opt.epsx) break;

        const double sy = detail::dot(s_vec, y_vec);
        if (sy > 1e-12) {
            s_hist.push_back(std::move(s_vec));
            y_hist.push_back(std::move(y_vec));
            rho_hist.push_back(1.0 / sy);
            if (static_cast<int>(s_hist.size()) > m) {
                s_hist.erase(s_hist.begin());
                y_hist.erase(y_hist.begin());
                rho_hist.erase(rho_hist.begin());
            }
        }
    }

    for (std::size_t k = 0; k < n; ++k) x[k] = z[k] * s[k];
    return rep;
}

}  // namespace asplbfgs
