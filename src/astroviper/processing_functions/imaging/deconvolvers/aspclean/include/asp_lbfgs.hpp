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
// i.e. a low-memory (m=1) L-BFGS with variable scaling and the same set of
// stopping conditions. The scale vector `s` is used both to precondition the
// initial step (H0 = diag(s_i^2)) and to measure the scaled gradient/step in
// the stopping tests, matching ALGLIB's scaled conventions. Exact bit-for-bit
// agreement with ALGLIB is neither expected nor required; functional
// equivalence (a few well-behaved descent steps from a good initial guess) is.

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace asplbfgs {

struct Options {
    int m = 1;             // number of stored correction pairs
    int max_iters = 5;     // ALGLIB maxits
    double epsg = 1e-3;    // scaled-gradient stopping tolerance
    double epsf = 1e-3;    // relative function-change tolerance
    double epsx = 1e-3;    // scaled-step tolerance
};

struct Report {
    int iterations = 0;
    double f = 0.0;
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

// ||g||_scaled = sqrt(sum (g_i * s_i)^2)  (ALGLIB scaled-gradient norm).
inline double scaled_grad_norm(const std::vector<double>& g,
                               const std::vector<double>& s) {
    double t = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double v = g[i] * s[i];
        t += v * v;
    }
    return std::sqrt(t);
}

// ||dx||_scaled = sqrt(sum (dx_i / s_i)^2).
inline double scaled_step_norm(const std::vector<double>& dx,
                               const std::vector<double>& s) {
    double t = 0.0;
    for (std::size_t i = 0; i < dx.size(); ++i) {
        const double v = dx[i] / s[i];
        t += v * v;
    }
    return std::sqrt(t);
}

}  // namespace detail

// Minimize `fg` starting from `x` (modified in place to hold the result).
// `scale` must be positive and the same length as `x`.
inline Report minimize(std::vector<double>& x,
                       const std::vector<double>& scale,
                       const ObjFunc& fg,
                       const Options& opt = Options{}) {
    const std::size_t n = x.size();
    Report rep;

    std::vector<double> g(n), x_new(n), g_new(n), dir(n);
    double f = fg(x, g);
    rep.f = f;

    const int m = std::max(1, opt.m);
    std::vector<std::vector<double>> s_hist, y_hist;  // s = dx, y = dg
    std::vector<double> rho_hist;

    for (int iter = 0; iter < opt.max_iters; ++iter) {
        // Stopping test on the (scaled) gradient.
        if (detail::scaled_grad_norm(g, scale) <= opt.epsg) break;

        // ----- Two-loop recursion to get the search direction -----
        std::vector<double> q = g;
        const int hist = static_cast<int>(s_hist.size());
        std::vector<double> alpha(hist, 0.0);
        for (int i = hist - 1; i >= 0; --i) {
            alpha[i] = rho_hist[i] * detail::dot(s_hist[i], q);
            for (std::size_t k = 0; k < n; ++k) q[k] -= alpha[i] * y_hist[i][k];
        }

        if (hist == 0) {
            // No curvature yet: precondition with the variable scales,
            // H0 = diag(scale_i^2), so the first step folds in the scaling.
            for (std::size_t k = 0; k < n; ++k) dir[k] = scale[k] * scale[k] * q[k];
        } else {
            const double sy = detail::dot(s_hist.back(), y_hist.back());
            const double yy = detail::dot(y_hist.back(), y_hist.back());
            const double gamma = (yy > 0.0) ? (sy / yy) : 1.0;
            for (std::size_t k = 0; k < n; ++k) dir[k] = gamma * q[k];
        }

        for (int i = 0; i < hist; ++i) {
            const double beta = rho_hist[i] * detail::dot(y_hist[i], dir);
            for (std::size_t k = 0; k < n; ++k)
                dir[k] += s_hist[i][k] * (alpha[i] - beta);
        }
        // Descent direction is -H*g.
        for (std::size_t k = 0; k < n; ++k) dir[k] = -dir[k];

        double dg = detail::dot(dir, g);
        if (dg >= 0.0) {
            // Not a descent direction (can happen with a poor H0); fall back to
            // scaled steepest descent.
            for (std::size_t k = 0; k < n; ++k) dir[k] = -scale[k] * scale[k] * g[k];
            dg = detail::dot(dir, g);
            if (dg >= 0.0) break;  // gradient is essentially zero
        }

        // ----- Backtracking line search (Armijo + weak curvature) -----
        const double c1 = 1e-4, c2 = 0.9, shrink = 0.5;
        double step = 1.0;
        bool ok = false;
        double f_new = f;
        for (int ls = 0; ls < 30; ++ls) {
            for (std::size_t k = 0; k < n; ++k) x_new[k] = x[k] + step * dir[k];
            f_new = fg(x_new, g_new);
            if (!std::isfinite(f_new)) {
                step *= shrink;
                continue;
            }
            const bool armijo = f_new <= f + c1 * step * dg;
            const bool curv = detail::dot(dir, g_new) >= c2 * dg;
            if (armijo && curv) { ok = true; break; }
            if (armijo) { ok = true; break; }  // accept Armijo-only if curvature fails
            step *= shrink;
        }
        if (!ok) break;

        // ----- Accept step and update history -----
        std::vector<double> s_vec(n), y_vec(n);
        for (std::size_t k = 0; k < n; ++k) {
            s_vec[k] = x_new[k] - x[k];
            y_vec[k] = g_new[k] - g[k];
        }
        const double sy = detail::dot(s_vec, y_vec);

        const double f_old = f;
        x = x_new;
        g = g_new;
        f = f_new;
        rep.f = f;
        rep.iterations = iter + 1;

        // Convergence on function value and step length (scaled).
        if (std::abs(f_old - f) <=
            opt.epsf * std::max({std::abs(f_old), std::abs(f), 1.0}))
            break;
        if (detail::scaled_step_norm(s_vec, scale) <= opt.epsx) break;

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

    return rep;
}

}  // namespace asplbfgs
