#pragma once

// A small, self-contained limited-memory BFGS (L-BFGS) minimizer used by the
// Asp deconvolver to optimize each Aspen's (amplitude, scale) pair. It
// replaces the ALGLIB `minlbfgs*` routines used by CASA and LibRA
// (synthesis/MeasurementEquations/AspMatrixCleaner.cc) with an implementation
// built only on the C++ standard library, and it follows the behaviour of the
// call they make,
//   minlbfgscreate(1, x, state);              // m = 1 correction pair
//   minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//   minlbfgssetscale(state, s);               // per-variable scale
// step for step:
//
//  * The search runs in the variables themselves. The scale s enters only the
//    stopping tests (the gradient as g * s, the step as dx / s): without
//    minlbfgssetprecscale the preconditioner is the identity.
//  * The first direction is the steepest descent -g with the initial step
//    min(1 / |g|, 1), so the first trial point moves by at most one unit; later
//    directions come from the standard two-loop recursion with
//    H0 = (s'y / y'y) I and the initial step 1. Every direction is normalised
//    to unit length and the step rescaled to keep the same trial point.
//  * The line search is the More-Thuente search (J. J. More and D. J. Thuente,
//    "Line search algorithms with guaranteed sufficient decrease", ACM TOMS 20,
//    286-307, 1994) as in MINPACK's mcsrch/mcstep, with ALGLIB's settings:
//    sufficient decrease ftol = 1e-3, curvature gtol = 0.4, xtol = 100 machine
//    epsilons, at most 20 evaluations, extrapolation by up to 4 times the step.
//    It can lengthen a step, not only shorten it. When it fails (no step met
//    both conditions) the next direction is the steepest descent again.
//  * After each line search: stop after maxits iterations, when |g * s| <=
//    epsg, when f_old - f <= epsf * max(|f_old|, |f|, 1), or when
//    |dx / s| <= epsx.
//
// An earlier version took unit steps in the scaled variables x / s and
// shortened them by backtracking only. On an Aspen whose start was far from
// the minimum along the scale axis it moved the scale by tens of pixels where
// ALGLIB moves it by one or two, and the Asp model updates diverged.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace asplbfgs {

struct Options {
    int m = 1;                 // number of stored correction pairs
    int max_iters = 5;         // ALGLIB maxits
    double epsg = 1e-3;        // scaled-gradient stopping tolerance
    double epsf = 1e-3;        // relative function-change tolerance
    double epsx = 1e-3;        // scaled-step tolerance
    double ftol = 1e-3;        // line search: sufficient decrease
    double gtol = 0.4;         // line search: curvature
    int max_line_search = 20;  // line search: evaluations per search
};

struct Report {
    int iterations = 0;   // line searches completed
    int evaluations = 0;  // objective (function + gradient) evaluations
    double f = 0.0;       // objective at the returned point
    bool line_search_failed = false;  // a line search ended without its conditions
    int termination = 0;  // ALGLIB's codes: 1 epsf, 2 epsx, 4 epsg, 5 maxits, <0 failures
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

// One step of the More-Thuente safeguarded cubic/quadratic interpolation: the
// interval [stx, sty] of uncertainty and the trial step stp are updated from
// the function values f and derivatives d at stx, sty and stp. `bracketed`
// becomes true once the interval brackets a minimiser; `ok` is false when the
// inputs are inconsistent.
inline void step_interval(double& stx, double& fx, double& dx, double& sty, double& fy,
                          double& dy, double& stp, double fp, double dp, bool& bracketed,
                          double stpmin, double stpmax, bool& ok) {
    ok = false;
    if ((bracketed && (stp <= std::min(stx, sty) || stp >= std::max(stx, sty))) ||
        dx * (stp - stx) >= 0.0 || stpmax < stpmin)
        return;
    ok = true;
    const double sgnd = dp * (dx / std::fabs(dx));
    bool bound = false;
    double stpf = stp;
    if (fp > fx) {
        // Higher function value: the minimum is bracketed. Take the cubic step
        // if it is closer to stx than the quadratic step, else their average.
        bound = true;
        const double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        const double s = std::max({std::fabs(theta), std::fabs(dx), std::fabs(dp)});
        double gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp < stx) gamma = -gamma;
        const double p = (gamma - dx) + theta;
        const double q = ((gamma - dx) + gamma) + dp;
        const double r = p / q;
        const double stpc = stx + r * (stp - stx);
        const double stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
        stpf = (std::fabs(stpc - stx) < std::fabs(stpq - stx)) ? stpc : stpc + (stpq - stpc) / 2.0;
        bracketed = true;
    } else if (sgnd < 0.0) {
        // Lower value, derivatives of opposite sign: bracketed. Take the step
        // (cubic or secant) farther from stp.
        const double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        const double s = std::max({std::fabs(theta), std::fabs(dx), std::fabs(dp)});
        double gamma = s * std::sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp > stx) gamma = -gamma;
        const double p = (gamma - dp) + theta;
        const double q = ((gamma - dp) + gamma) + dx;
        const double r = p / q;
        const double stpc = stp + r * (stx - stp);
        const double stpq = stp + (dp / (dp - dx)) * (stx - stp);
        stpf = (std::fabs(stpc - stp) > std::fabs(stpq - stp)) ? stpc : stpq;
        bracketed = true;
    } else if (std::fabs(dp) < std::fabs(dx)) {
        // Lower value, same sign, smaller derivative: the cubic step is used
        // only if it goes the right way and the cubic tends to infinity in
        // that direction.
        bound = true;
        const double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        const double s = std::max({std::fabs(theta), std::fabs(dx), std::fabs(dp)});
        double gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
        if (stp > stx) gamma = -gamma;
        const double p = (gamma - dp) + theta;
        const double q = (gamma + (dx - dp)) + gamma;
        const double r = p / q;
        double stpc;
        if (r < 0.0 && gamma != 0.0)
            stpc = stp + r * (stx - stp);
        else
            stpc = (stp > stx) ? stpmax : stpmin;
        const double stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (bracketed)
            stpf = (std::fabs(stp - stpc) < std::fabs(stp - stpq)) ? stpc : stpq;
        else
            stpf = (std::fabs(stp - stpc) > std::fabs(stp - stpq)) ? stpc : stpq;
    } else {
        // Lower value, same sign, derivative not smaller: if bracketed take the
        // cubic step through (sty, fy, dy), else move to the step limit.
        if (bracketed) {
            const double theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
            const double s = std::max({std::fabs(theta), std::fabs(dy), std::fabs(dp)});
            double gamma = s * std::sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
            if (stp > sty) gamma = -gamma;
            const double p = (gamma - dp) + theta;
            const double q = ((gamma - dp) + gamma) + dy;
            const double r = p / q;
            stpf = stp + r * (sty - stp);
        } else {
            stpf = (stp > stx) ? stpmax : stpmin;
        }
    }
    // Update the interval of uncertainty.
    if (fp > fx) {
        sty = stp; fy = fp; dy = dp;
    } else {
        if (sgnd < 0.0) { sty = stx; fy = fx; dy = dx; }
        stx = stp; fx = fp; dx = dp;
    }
    stpf = std::max(stpmin, std::min(stpmax, stpf));
    stp = stpf;
    if (bracketed && bound) {
        if (sty > stx)
            stp = std::min(stx + 0.66 * (sty - stx), stp);
        else
            stp = std::max(stx + 0.66 * (sty - stx), stp);
    }
}

// More-Thuente line search along `dir` from (x, f, g) with initial step stp.
// On return x, f, g hold the last point evaluated (the best one when the
// search ends on a safeguard) and the result is 1 when the sufficient decrease
// and curvature conditions hold, another positive code when a safeguard ended
// the search, and 0 for invalid input (no descent direction).
template <typename Eval>
int more_thuente(std::vector<double>& x, double& f, std::vector<double>& g,
                 const std::vector<double>& dir, double& stp, const Options& opt,
                 int& nfev, Eval&& evaluate) {
    constexpr double xtrapf = 4.0;
    const double xtol = 100.0 * std::numeric_limits<double>::epsilon();
    const double stpmin = 1e-50, stpmax = 1e50;
    nfev = 0;
    const double dginit = dot(g, dir);
    if (!(stp > 0.0) || !(dginit < 0.0)) return 0;
    bool bracketed = false, stage1 = true, interval_ok = true;
    const double finit = f;
    const double dgtest = opt.ftol * dginit;
    double width = stpmax - stpmin;
    double width1 = width / 0.5;
    const std::vector<double> x0 = x;
    double stx = 0.0, fx = finit, dgx = dginit;
    double sty = 0.0, fy = finit, dgy = dginit;
    double stmin = 0.0, stmax = 0.0;
    while (true) {
        if (bracketed) {
            stmin = std::min(stx, sty);
            stmax = std::max(stx, sty);
        } else {
            stmin = stx;
            stmax = stp + xtrapf * (stp - stx);
        }
        stp = std::max(stp, stpmin);
        stp = std::min(stp, stpmax);
        // No progress possible: go back to the best step found.
        if ((bracketed && (stp <= stmin || stp >= stmax)) || nfev >= opt.max_line_search - 1 ||
            !interval_ok || (bracketed && stmax - stmin <= xtol * stmax))
            stp = stx;
        for (std::size_t k = 0; k < x.size(); ++k) x[k] = x0[k] + stp * dir[k];
        f = evaluate(x, g);
        ++nfev;
        const double dg = dot(g, dir);
        const double ftest1 = finit + stp * dgtest;
        int info = 0;
        if ((bracketed && (stp <= stmin || stp >= stmax)) || !interval_ok) info = 6;
        if (stp == stpmax && f <= ftest1 && dg <= dgtest) info = 5;
        if (stp == stpmin && (f > ftest1 || dg >= dgtest)) info = 4;
        if (nfev >= opt.max_line_search) info = 3;
        if (bracketed && stmax - stmin <= xtol * stmax) info = 2;
        if (f <= ftest1 && std::fabs(dg) <= -opt.gtol * dginit) info = 1;
        if (!std::isfinite(f)) info = 7;
        if (info != 0) return info;
        if (stage1 && f <= ftest1 && dg >= std::min(opt.ftol, opt.gtol) * dginit) stage1 = false;
        if (stage1 && f <= fx && f > ftest1) {
            // Modified function psi(stp) = f(stp) - f(0) - ftol stp f'(0).
            double fm = f - stp * dgtest;
            double fxm = fx - stx * dgtest, fym = fy - sty * dgtest;
            double dgm = dg - dgtest, dgxm = dgx - dgtest, dgym = dgy - dgtest;
            step_interval(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, bracketed, stmin, stmax,
                          interval_ok);
            fx = fxm + stx * dgtest;
            fy = fym + sty * dgtest;
            dgx = dgxm + dgtest;
            dgy = dgym + dgtest;
        } else {
            step_interval(stx, fx, dgx, sty, fy, dgy, stp, f, dg, bracketed, stmin, stmax,
                          interval_ok);
        }
        if (bracketed) {
            if (std::fabs(sty - stx) >= 0.66 * width1) stp = stx + 0.5 * (sty - stx);
            width1 = width;
            width = std::fabs(sty - stx);
        }
    }
}

}  // namespace detail

// Minimize `fg` starting from `x` (modified in place to hold the result).
// `scale` must be the same length as `x`; it scales the stopping tests (its
// sign is ignored, and a zero or non-finite entry counts as 1, as ALGLIB does).
inline Report minimize(std::vector<double>& x,
                       const std::vector<double>& scale,
                       const ObjFunc& fg,
                       const Options& opt = Options{}) {
    using detail::dot;
    using detail::norm;
    const std::size_t n = x.size();
    Report rep;

    std::vector<double> s(n);
    for (std::size_t k = 0; k < n; ++k) {
        s[k] = std::fabs(scale[k]);
        if (!(s[k] > 0.0) || !std::isfinite(s[k])) s[k] = 1.0;
    }
    auto scaled_norm = [&](const std::vector<double>& v, bool multiply) {
        double sum = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double t = multiply ? v[k] * s[k] : v[k] / s[k];
            sum += t * t;
        }
        return std::sqrt(sum);
    };
    auto evaluate = [&](const std::vector<double>& xx, std::vector<double>& gg) {
        ++rep.evaluations;
        return fg(xx, gg);
    };

    std::vector<double> g(n);
    double f = evaluate(x, g);
    rep.f = f;
    if (!std::isfinite(f)) {
        rep.termination = -8;
        return rep;
    }
    if (scaled_norm(g, true) <= opt.epsg) {
        rep.termination = 4;
        return rep;
    }

    const int m = std::max(1, opt.m);
    std::vector<std::vector<double>> sk(m, std::vector<double>(n)), yk(m, std::vector<double>(n));
    std::vector<double> rho(m, 0.0), theta(m, 0.0);
    std::vector<double> dir(n), work(n);
    for (std::size_t k = 0; k < n; ++k) dir[k] = -g[k];
    double stp = std::min(1.0 / norm(g), 1.0);
    double fold = f;
    int k = 0;

    while (true) {
        const int p = k % m;
        const int q = std::min(k, m - 1);
        for (std::size_t i = 0; i < n; ++i) {
            sk[p][i] = -x[i];
            yk[p][i] = -g[i];
        }
        if (k != 0) stp = 1.0;
        // Unit-length direction, the trial point unchanged.
        double mx = 0.0;
        for (double v : dir) mx = std::max(mx, std::fabs(v));
        if (mx > 0.0) {
            for (double& v : dir) v /= mx;
            stp *= mx;
            const double len = norm(dir);
            for (double& v : dir) v /= len;
            stp *= len;
        }
        int nfev = 0;
        const int info = detail::more_thuente(x, f, g, dir, stp, opt, nfev, evaluate);
        rep.f = f;
        if (info == 0) {
            rep.termination = -2;
            break;
        }
        ++rep.iterations;
        if (info != 1) rep.line_search_failed = true;
        for (std::size_t i = 0; i < n; ++i) {
            sk[p][i] += x[i];
            yk[p][i] += g[i];
        }
        const double gnorm = scaled_norm(g, true);
        if (!std::isfinite(gnorm) || !std::isfinite(f)) {
            rep.termination = -8;
            break;
        }
        if (rep.iterations >= opt.max_iters && opt.max_iters > 0) {
            rep.termination = 5;
            break;
        }
        if (gnorm <= opt.epsg) {
            rep.termination = 4;
            break;
        }
        if (fold - f <= opt.epsf * std::max({std::fabs(fold), std::fabs(f), 1.0})) {
            rep.termination = 1;
            break;
        }
        if (scaled_norm(sk[p], false) <= opt.epsx) {
            rep.termination = 2;
            break;
        }
        if (info != 1) {
            // The line search failed: restart from the steepest descent.
            fold = f;
            for (std::size_t i = 0; i < n; ++i) dir[i] = -g[i];
            continue;
        }
        const double sy = dot(yk[p], sk[p]);
        const double yy = dot(yk[p], yk[p]);
        if (sy == 0.0 || yy == 0.0) {
            rep.termination = -2;
            break;
        }
        rho[p] = 1.0 / sy;
        const double gamma = sy / yy;
        work = g;
        for (int i = k; i >= k - q; --i) {
            const int ic = i % m;
            theta[ic] = dot(sk[ic], work);
            const double a = theta[ic] * rho[ic];
            for (std::size_t j = 0; j < n; ++j) work[j] -= a * yk[ic][j];
        }
        for (double& v : work) v *= gamma;
        for (int i = k - q; i <= k; ++i) {
            const int ic = i % m;
            const double b = rho[ic] * (theta[ic] - dot(yk[ic], work));
            for (std::size_t j = 0; j < n; ++j) work[j] += b * sk[ic][j];
        }
        for (std::size_t i = 0; i < n; ++i) dir[i] = -work[i];
        fold = f;
        ++k;
    }
    return rep;
}

}  // namespace asplbfgs
