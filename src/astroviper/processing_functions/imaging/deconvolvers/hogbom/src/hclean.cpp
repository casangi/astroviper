/*
 * Copyright (C) 1999,2000,2003
 * Associated Universities, Inc. Washington DC, USA.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 675 Massachusetts Ave, Cambridge,
 * MA 02139, USA.
 *
 * Correspondence concerning AIPS++ should be addressed as follows:
 *        Internet email: casa-feedback@nrao.edu.
 *        Postal address: AIPS++ Project Office
 *                        National Radio Astronomy Observatory
 *                        520 Edgemont Road
 *                        Charlottesville, VA 22903-2475 USA
 *
 * Converted from Fortran to C++ for modern usage
 * Templated version to support both float and double precision
 */

#include "../include/hclean.hpp"
#include <cmath>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace hclean {

/**
 * Templated function to find minimum and maximum values in a 2D image array.
 * The mask is a bool array; pixels with mask == true are considered.
 */
template<typename T>
void maximg(const T* limagestep, int domask, const bool* lmask,
           int nx, int ny, T& fmin, T& fmax) {

    // Use appropriate large values for each type
    if constexpr (std::is_same_v<T, float>) {
        fmin = 1e20f;
        fmax = -1e20f;
    } else {
        fmin = 1e20;
        fmax = -1e20;
    }

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            if ((domask == 0) || lmask[iy * nx + ix]) {
                T wpeak = limagestep[iy * nx + ix];

                if (wpeak > fmax) {
                    fmax = wpeak;
                }
                if (wpeak < fmin) {
                    fmin = wpeak;
                }
            }
        }
    }
}

/**
 * Templated Hogbom CLEAN algorithm implementation
 *
 * @param limage 2D output model image (clean components) [ny][nx]
 * @param limagestep 2D input dirty image / output residual [ny][nx]
 * @param lpsf 2D point spread function [ny][nx]
 * @param domask flag indicating if mask is present (0 = no mask)
 * @param lmask 2D input mask image [ny][nx]
 * @param nx width of images
 * @param ny height of images
 * @param xbeg starting x coordinate for clean box (0-based)
 * @param xend ending x coordinate for clean box (0-based, exclusive)
 * @param ybeg starting y coordinate for clean box (0-based)
 * @param yend ending y coordinate for clean box (0-based, exclusive)
 * @param niter maximum allowed iterations
 * @param siter starting iteration number
 * @param iter output: last iteration number reached
 * @param gain clean loop gain
 * @param thres flux cleaning threshold
 * @param cspeedup if > 0, adaptive threshold: thres * 2^(iter/cspeedup)
 * @param msgput callback function for status messages
 * @param stopnow callback function to check if stopping is requested
 */
template<typename T>
void clean(T* limage, T* limagestep, const T* lpsf,
           int domask, const bool* lmask, int nx, int ny,
           int xbeg, int xend, int ybeg, int yend,
           int niter, int siter, int& iter, T gain, T thres,
           T cspeedup,
           std::function<void(int, int, int, T)> msgput,
           std::function<void(int&)> stopnow) {

    int yes = 0;

    // Find peak in image within clean box
    T maxval = static_cast<T>(0);
    T absval = static_cast<T>(0);
    int px = 0;  // Convert from Fortran 1-based to 0-based
    int py = 0;

    // Main iteration loop
    for (iter = siter; iter < niter; ++iter) {
        absval = static_cast<T>(0);
        for (int iy = ybeg; iy < yend; ++iy) {
            for (int ix = xbeg; ix < xend; ++ix) {
                if ((domask == 0) || lmask[iy * nx + ix]) {
                    T val = std::abs(limagestep[iy * nx + ix]);
                    if (val > absval) {
                        px = ix;
                        py = iy;
                        absval = val;
                    }
                }
            }
        }

        // Get actual value (with sign) at peak location
        maxval = limagestep[py * nx + px];

        // Calculate adaptive threshold if requested
        T cthres;
        T zero_val = static_cast<T>(0);
        T two_val = static_cast<T>(2);
        if (cspeedup > zero_val) {
            cthres = thres * std::pow(two_val, static_cast<T>(iter - siter) / cspeedup);
        } else {
            cthres = thres;
        }

        // Check convergence criteria
        if ((yes == 1) || (absval < cthres)) {
            break;  // goto 200 equivalent
        }

        // Output progress information
        int cycle = std::max(1, (niter - siter) / 10);
        if ((iter == siter) || ((iter % cycle) == 1)) {
            msgput(iter, px, py, maxval);
            stopnow(yes);
        }

        if (yes == 1) {
            break;
        }

        // Calculate bounds for PSF subtraction
        // Note: In Fortran code, PSF appears to be centered, so we use nx/2, ny/2
        int x1 = std::max(0, px - nx / 2);
        int y1 = std::max(0, py - ny / 2);
        int x2 = std::min(nx - 1, px + nx / 2 - 1);
        int y2 = std::min(ny - 1, py + ny / 2 - 1);

        // Add component to model and subtract from residual
        T pv = gain * maxval;
        limage[py * nx + px] += pv;

        // Subtract scaled PSF from residual
        for (int iy = y1; iy <= y2; ++iy) {
            for (int ix = x1; ix <= x2; ++ix) {
                // PSF indexing: convert from Fortran centered indexing
                // Fortran: lpsf(nx/2+ix-px+1, ny/2+iy-py+1)
                // C++: lpsf[(ny/2 + iy - py) * nx + (nx/2 + ix - px)]
                int psf_x = nx / 2 + ix - px;
                int psf_y = ny / 2 + iy - py;

                // Check if psf_x and psf_y are within bounds
                if (psf_x < 0 || psf_x >= nx || psf_y < 0 || psf_y >= ny) {
                    continue; // Skip out-of-bounds
                }

                limagestep[iy * nx + ix] -=
                    pv * lpsf[psf_y * nx + psf_x];
            }
        }
    }

    // Output final status
    if (iter > siter) {
        msgput(iter, px, py, maxval);
    }

    // Ensure iter doesn't exceed niter
    if (iter > niter) {
        iter = niter;
    }
}

/**
 * Templated cube driver. Runs clean<T> on each (time, frequency,
 * polarization) plane of a 5D input cube and dispatches planes across
 * std::thread workers using an atomic index. The per-plane inputs are
 * non-overlapping slices of the same contiguous cube buffer, so no
 * synchronization is needed beyond the atomic work counter.
 */
template<typename T>
void clean_cube(T* residual_cube, T* model_cube, const T* psf_cube,
                int domask, const bool* mask_cube,
                int nt, int nf, int np_img, int np_psf,
                int ny, int nx,
                int xbeg, int xend, int ybeg, int yend,
                const int* niter, T gain, const T* thres, T cspeedup,
                int num_threads, int* iter_out) {

    const int nplanes = nt * nf * np_img;
    if (nplanes <= 0) {
        return;
    }

    // std::thread callbacks are never invoked from cube mode; supply
    // no-ops so the inner clean<T> signature is still satisfied.
    auto noop_msgput = [](int, int, int, T) {};
    auto noop_stopnow = [](int& x) { x = 0; };

    const std::size_t plane_size = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);

    std::atomic<int> next_plane(0);

    auto worker = [&]() {
        int plane;
        while ((plane = next_plane.fetch_add(1)) < nplanes) {
            const int tt = plane / (nf * np_img);
            const int nn = (plane / np_img) % nf;
            const int pp = plane % np_img;
            const int pidx = (np_psf == 1) ? 0 : pp;

            const std::size_t img_offset =
                ((static_cast<std::size_t>(tt) * nf + nn) * np_img + pp) * plane_size;
            const std::size_t psf_offset =
                ((static_cast<std::size_t>(tt) * nf + nn) * np_psf + pidx) * plane_size;

            T* residual = residual_cube + img_offset;
            T* model = model_cube + img_offset;
            const T* psf = psf_cube + psf_offset;
            const bool* mask = (domask && mask_cube != nullptr)
                ? mask_cube + img_offset
                : nullptr;

            int iter_val = 0;
            // Iteration control is independent per plane: each (t, f, p)
            // plane uses its own maximum iteration count and threshold.
            clean<T>(model, residual, psf,
                     domask, mask,
                     nx, ny, xbeg, xend, ybeg, yend,
                     niter[plane], 0, iter_val, gain, thres[plane], cspeedup,
                     noop_msgput, noop_stopnow);
            iter_out[plane] = iter_val;
        }
    };

    int nthreads = num_threads;
    if (nthreads < 1) nthreads = 1;
    if (nthreads > nplanes) nthreads = nplanes;

    if (nthreads == 1) {
        worker();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(nthreads);
        for (int i = 0; i < nthreads; ++i) {
            threads.emplace_back(worker);
        }
        for (auto& t : threads) {
            t.join();
        }
    }
}

/**
 * Persistent barrier thread pool over a fixed [0, n_rows) row partition.
 *
 * Workers are created once and reused for every iteration's peak search and
 * subtraction, so clean_cube_many_threads pays the std::thread creation cost
 * only once instead of twice per minor cycle. Each run(body) call dispatches
 * the same contiguous row chunk to each worker and blocks until all finish
 * (a generation counter + two condition variables form the barrier).
 */
namespace {

class RowThreadPool {
public:
    RowThreadPool(int nthreads, long n_rows) : nthreads_(nthreads) {
        const long step = (n_rows + nthreads - 1) / nthreads;
        chunks_.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            const long a = std::min(static_cast<long>(t) * step, n_rows);
            const long b = std::min(a + step, n_rows);
            chunks_.emplace_back(a, b);
        }
        workers_.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            workers_.emplace_back([this, t]() { worker(t); });
        }
    }

    ~RowThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
            ++generation_;
        }
        cv_go_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    // Run body over every worker's chunk; blocks until all workers finish.
    void run(const std::function<void(long, long)>& body) {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            body_ = &body;
            done_ = 0;
            ++generation_;
        }
        cv_go_.notify_all();
        std::unique_lock<std::mutex> lk(mtx_);
        cv_done_.wait(lk, [this]() { return done_ == nthreads_; });
    }

private:
    void worker(int id) {
        long last_gen = 0;
        while (true) {
            const std::function<void(long, long)>* fn = nullptr;
            long r0 = 0, r1 = 0;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_go_.wait(lk, [this, &last_gen]() {
                    return stop_ || generation_ != last_gen;
                });
                if (stop_) return;
                last_gen = generation_;
                fn = body_;
                r0 = chunks_[id].first;
                r1 = chunks_[id].second;
            }
            if (fn && r0 < r1) (*fn)(r0, r1);
            {
                std::unique_lock<std::mutex> lk(mtx_);
                if (++done_ == nthreads_) {
                    lk.unlock();
                    cv_done_.notify_one();
                }
            }
        }
    }

    int nthreads_;
    std::vector<std::pair<long, long>> chunks_;
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_go_;
    std::condition_variable cv_done_;
    const std::function<void(long, long)>* body_ = nullptr;
    long generation_ = 0;
    int done_ = 0;
    bool stop_ = false;
};

} // namespace

/**
 * Many-threads cube driver. Same algorithm as clean<T> applied per plane, but
 * the parallelism spans the whole (plane, row) work domain: each iteration's
 * peak search and PSF subtraction are split into num_threads contiguous row
 * ranges and run on a persistent thread pool, with a cheap serial per-plane
 * reduction in between. This keeps every thread busy even with one or two
 * planes (single-channel imaging), unlike clean_cube which is one thread per
 * plane. Tie-breaking (lowest row, then lowest column) and the per-plane
 * iter_out match clean_cube exactly.
 */
template<typename T>
void clean_cube_many_threads(T* residual_cube, T* model_cube, const T* psf_cube,
                int domask, const bool* mask_cube,
                int nt, int nf, int np_img, int np_psf,
                int ny, int nx,
                int xbeg, int xend, int ybeg, int yend,
                const int* niter, T gain, const T* thres, T cspeedup,
                int num_threads, int* iter_out) {

    const int nplanes = nt * nf * np_img;
    if (nplanes <= 0) {
        return;
    }
    const std::size_t plane_size =
        static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);

    std::vector<char> active(nplanes, 1);
    std::vector<T> row_max(static_cast<std::size_t>(nplanes) * ny);
    std::vector<int> row_ix(static_cast<std::size_t>(nplanes) * ny);
    std::vector<int> peak_y(nplanes, 0);
    std::vector<int> peak_x(nplanes, 0);
    std::vector<T> peak_pv(nplanes, static_cast<T>(0));

    int max_niter = 0;
    for (int pl = 0; pl < nplanes; ++pl) {
        if (niter[pl] > max_niter) max_niter = niter[pl];
    }

    int nthreads = num_threads;
    if (nthreads < 1) nthreads = 1;

    const long n_rows = static_cast<long>(nplanes) * ny;
    // Don't create more workers than rows of work.
    if (static_cast<long>(nthreads) > n_rows) {
        nthreads = static_cast<int>(std::max(1L, n_rows));
    }

    // Persistent thread pool: workers are created once here and reused for
    // every iteration's peak search and subtraction (no per-iteration spawn).
    std::unique_ptr<RowThreadPool> pool;
    if (nthreads > 1 && n_rows > 1) {
        pool = std::make_unique<RowThreadPool>(nthreads, n_rows);
    }
    auto parallel_for = [&](const std::function<void(long, long)>& body) {
        if (pool) {
            pool->run(body);
        } else {
            body(0L, n_rows);
        }
    };

    const T zero_val = static_cast<T>(0);
    const T two_val = static_cast<T>(2);

    for (int it = 0; it < max_niter; ++it) {
        // ---- peak search: per (plane, row) max |residual| within the box ----
        parallel_for([&](long r0, long r1) {
            for (long r = r0; r < r1; ++r) {
                const int pl = static_cast<int>(r / ny);
                const int iy = static_cast<int>(r % ny);
                if (!active[pl] || it >= niter[pl] || iy < ybeg || iy >= yend) {
                    row_max[r] = static_cast<T>(-1);
                    continue;
                }
                const T* res = residual_cube
                    + static_cast<std::size_t>(pl) * plane_size
                    + static_cast<std::size_t>(iy) * nx;
                const bool* msk = (domask && mask_cube)
                    ? mask_cube + static_cast<std::size_t>(pl) * plane_size
                                + static_cast<std::size_t>(iy) * nx
                    : nullptr;
                T m = static_cast<T>(0);
                int mi = xbeg;
                for (int ix = xbeg; ix < xend; ++ix) {
                    if (!domask || msk[ix]) {
                        T v = std::abs(res[ix]);
                        if (v > m) { m = v; mi = ix; }
                    }
                }
                row_max[r] = m;
                row_ix[r] = mi;
            }
        });

        // ---- serial per-plane combine + convergence + model update ----
        bool any_active = false;
        for (int pl = 0; pl < nplanes; ++pl) {
            if (!active[pl] || it >= niter[pl]) continue;
            T best = static_cast<T>(-1);
            int py = ybeg, px = xbeg;
            const long base = static_cast<long>(pl) * ny;
            for (int iy = ybeg; iy < yend; ++iy) {
                T rm = row_max[base + iy];
                if (rm > best) { best = rm; py = iy; px = row_ix[base + iy]; }
            }
            T cthres = thres[pl];
            if (cspeedup > zero_val) {
                cthres = thres[pl] * std::pow(two_val, static_cast<T>(it) / cspeedup);
            }
            if (best < cthres) { active[pl] = 0; continue; }
            T* res = residual_cube + static_cast<std::size_t>(pl) * plane_size;
            T* mod = model_cube + static_cast<std::size_t>(pl) * plane_size;
            T maxval = res[static_cast<std::size_t>(py) * nx + px];
            T pv = gain * maxval;
            mod[static_cast<std::size_t>(py) * nx + px] += pv;
            peak_y[pl] = py;
            peak_x[pl] = px;
            peak_pv[pl] = pv;
            iter_out[pl] += 1;
            any_active = true;
        }
        if (!any_active) break;

        // ---- PSF subtraction over the same (plane, row) domain ----
        parallel_for([&](long r0, long r1) {
            for (long r = r0; r < r1; ++r) {
                const int pl = static_cast<int>(r / ny);
                const int iy = static_cast<int>(r % ny);
                if (!active[pl] || it >= niter[pl]) continue;
                const int py = peak_y[pl];
                const int px = peak_x[pl];
                const T pv = peak_pv[pl];
                const int tt = pl / (nf * np_img);
                const int rem = pl % (nf * np_img);
                const int nn = rem / np_img;
                const int pp = rem % np_img;
                const int pidx = (np_psf == 1) ? 0 : pp;
                const T* psf = psf_cube
                    + (static_cast<std::size_t>(tt * nf + nn) * np_psf + pidx)
                          * plane_size;
                const int psf_y = ny / 2 + iy - py;
                if (psf_y < 0 || psf_y >= ny) continue;
                T* res = residual_cube
                    + static_cast<std::size_t>(pl) * plane_size
                    + static_cast<std::size_t>(iy) * nx;
                const T* psf_row = psf + static_cast<std::size_t>(psf_y) * nx;
                const int x1 = std::max(0, px - nx / 2);
                const int x2 = std::min(nx - 1, px + nx / 2 - 1);
                for (int ix = x1; ix <= x2; ++ix) {
                    const int psf_x = nx / 2 + ix - px;
                    if (psf_x < 0 || psf_x >= nx) continue;
                    res[ix] -= pv * psf_row[psf_x];
                }
            }
        });
    }
}

// Explicit template instantiations for float and double
template void maximg<float>(const float* limagestep, int domask, const bool* lmask,
                           int nx, int ny, float& fmin, float& fmax);

template void maximg<double>(const double* limagestep, int domask, const bool* lmask,
                            int nx, int ny, double& fmin, double& fmax);

template void clean<float>(float* limage, float* limagestep, const float* lpsf,
                          int domask, const bool* lmask, int nx, int ny,
                          int xbeg, int xend, int ybeg, int yend,
                          int niter, int siter, int& iter, float gain, float thres,
                          float cspeedup,
                          std::function<void(int, int, int, float)> msgput,
                          std::function<void(int&)> stopnow);

template void clean<double>(double* limage, double* limagestep, const double* lpsf,
                           int domask, const bool* lmask, int nx, int ny,
                           int xbeg, int xend, int ybeg, int yend,
                           int niter, int siter, int& iter, double gain, double thres,
                           double cspeedup,
                           std::function<void(int, int, int, double)> msgput,
                           std::function<void(int&)> stopnow);

template void clean_cube<float>(float* residual_cube, float* model_cube,
                                const float* psf_cube, int domask,
                                const bool* mask_cube,
                                int nt, int nf, int np_img, int np_psf,
                                int ny, int nx,
                                int xbeg, int xend, int ybeg, int yend,
                                const int* niter, float gain, const float* thres, float cspeedup,
                                int num_threads, int* iter_out);

template void clean_cube<double>(double* residual_cube, double* model_cube,
                                 const double* psf_cube, int domask,
                                 const bool* mask_cube,
                                 int nt, int nf, int np_img, int np_psf,
                                 int ny, int nx,
                                 int xbeg, int xend, int ybeg, int yend,
                                 const int* niter, double gain, const double* thres, double cspeedup,
                                 int num_threads, int* iter_out);

template void clean_cube_many_threads<float>(float* residual_cube, float* model_cube,
                                const float* psf_cube, int domask,
                                const bool* mask_cube,
                                int nt, int nf, int np_img, int np_psf,
                                int ny, int nx,
                                int xbeg, int xend, int ybeg, int yend,
                                const int* niter, float gain, const float* thres, float cspeedup,
                                int num_threads, int* iter_out);

template void clean_cube_many_threads<double>(double* residual_cube, double* model_cube,
                                 const double* psf_cube, int domask,
                                 const bool* mask_cube,
                                 int nt, int nf, int np_img, int np_psf,
                                 int ny, int nx,
                                 int xbeg, int xend, int ybeg, int yend,
                                 const int* niter, double gain, const double* thres, double cspeedup,
                                 int num_threads, int* iter_out);

} // namespace hclean
