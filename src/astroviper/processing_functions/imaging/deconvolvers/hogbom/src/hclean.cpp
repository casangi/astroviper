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

namespace {

// Inclusive pixel box [i0, i1] x [j0, j1], as in Asp and MT-MFS.
struct Box {
    int i0 = 0, j0 = 0, i1 = -1, j1 = -1;
};

Box mask_bounding_box(const bool* mask, int nx,
                      int xbeg, int xend, int ybeg, int yend) {
    const Box clean_box{xbeg, ybeg, xend - 1, yend - 1};
    if (mask == nullptr) return clean_box;
    Box box{xend, yend, -1, -1};
    for (int j = ybeg; j < yend; ++j)
        for (int i = xbeg; i < xend; ++i)
            if (mask[static_cast<std::size_t>(j) * nx + i]) {
                box.i0 = std::min(box.i0, i); box.j0 = std::min(box.j0, j);
                box.i1 = std::max(box.i1, i); box.j1 = std::max(box.j1, j);
            }
    // Preserve the existing empty-mask stop and peak conventions, including
    // native callers using a negative threshold to disable threshold stopping.
    return (box.i1 < box.i0) ? clean_box : box;
}

} // namespace

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
 * @param lpsf 2D point spread function [psf_ny][psf_nx], peak at
 *             (psf_ny/2, psf_nx/2)
 * @param domask flag indicating if mask is present (0 = no mask)
 * @param lmask 2D input mask image [ny][nx]
 * @param nx width of images
 * @param ny height of images
 * @param psf_nx width of the PSF: nx, or 2 nx to subtract over the whole image
 * @param psf_ny height of the PSF: ny, or 2 ny
 * @param xbeg starting x coordinate for clean box (0-based)
 * @param xend ending x coordinate for clean box (0-based, exclusive)
 * @param ybeg starting y coordinate for clean box (0-based)
 * @param yend ending y coordinate for clean box (0-based, exclusive)
 * @param max_iter_per_cycle_updated maximum allowed iterations
 * @param siter starting iteration number
 * @param iter output: last iteration number reached
 * @param gain clean loop gain
 * @param thres flux cleaning threshold
 * @param cspeedup if > 0, adaptive threshold: thres * 2^(iter/cspeedup)
 * @param msgput callback function for status messages
 * @param stopnow callback function to check if stopping is requested
 * @param divergence divergence test (see DivergenceControl)
 * @param stop_reason output: why the plane stopped (a StopReason)
 * @param history output: the record of every iteration (IterationHistory),
 *                or nullptr to record nothing
 */
template<typename T>
void clean(T* limage, T* limagestep, const T* lpsf,
           int domask, const bool* lmask, int nx, int ny, int psf_nx, int psf_ny,
           int xbeg, int xend, int ybeg, int yend,
           int max_iter_per_cycle_updated, int siter, int& iter, T gain, T thres,
           T cspeedup,
           std::function<void(int, int, int, T)> msgput,
           std::function<void(int&)> stopnow,
           const DivergenceControl<T>& divergence,
           int& stop_reason, IterationHistory* history) {

    int yes = 0;
    PlaneStopTest<T> stop_test;
    stop_reason = STOP_NONE;
    const bool* search_mask = (domask == 0) ? nullptr : lmask;
    // Build once per plane, intersecting mask support with the supplied box.
    // Only the search shrinks; PSF subtraction still covers its full footprint.
    const Box box = mask_bounding_box(search_mask, nx, xbeg, xend, ybeg, yend);
    xbeg = box.i0; xend = box.i1 + 1;
    ybeg = box.j0; yend = box.j1 + 1;
    // Pixels the peak search covers: the denominator of the recorded RMS.
    const double n_searched = (history == nullptr)
        ? 0.0
        : static_cast<double>(
              searched_pixels(search_mask, nx, xbeg, xend, ybeg, yend));

    // Find peak in image within clean box
    T maxval = static_cast<T>(0);
    T absval = static_cast<T>(0);
    int px = 0;  // Convert from Fortran 1-based to 0-based
    int py = 0;

    // Main iteration loop
    for (iter = siter; iter < max_iter_per_cycle_updated; ++iter) {
        // Peak of |residual| and sum of its squares over the clean box inside
        // the mask, row by row (see scan_row): the first of equal peaks wins,
        // lowest row, then lowest column.
        absval = static_cast<T>(0);
        double sum_squares = 0.0;
        for (int iy = ybeg; iy < yend; ++iy) {
            const std::size_t row_offset = static_cast<std::size_t>(iy) * nx;
            T row_max;
            int row_ix;
            sum_squares += scan_row<T>(
                limagestep + row_offset,
                (domask == 0) ? nullptr : lmask + row_offset,
                xbeg, xend, row_max, row_ix);
            if (row_max > absval) {
                px = row_ix;
                py = iy;
                absval = row_max;
            }
        }
        // The search summed the squares of the residual the previous
        // iteration left: its RMS.
        if (history != nullptr && iter > siter) {
            history->rms.push_back(std::sqrt(sum_squares / n_searched));
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

        // Stop tests: a stop request, then the threshold and the divergence
        // test (shared with the many-threads kernel through PlaneStopTest).
        if (yes == 1) {
            break;  // goto 200 equivalent
        }
        const int reason = stop_test.check(absval, sum_squares, cthres, divergence);
        if (reason != STOP_NONE) {
            stop_reason = reason;
            break;
        }

        // Output progress information
        int cycle = std::max(1, (max_iter_per_cycle_updated - siter) / 10);
        if ((iter == siter) || ((iter % cycle) == 1)) {
            msgput(iter, px, py, maxval);
            stopnow(yes);
        }

        if (yes == 1) {
            break;
        }

        // Offsets -psf_n/2 .. (psf_n - 1)/2 from the PSF centre, every sample of
        // the PSF, clipped to the image: the whole image for a PSF of twice its
        // size.
        int x1 = std::max(0, px - psf_nx / 2);
        int y1 = std::max(0, py - psf_ny / 2);
        int x2 = std::min(nx - 1, px + (psf_nx - 1) / 2);
        int y2 = std::min(ny - 1, py + (psf_ny - 1) / 2);

        // Add component to model and subtract from residual
        T pv = gain * maxval;
        limage[py * nx + px] += pv;
        if (history != nullptr) {
            history->add_component(maxval, py, px, pv);
        }

        // Subtract scaled PSF from residual
        for (int iy = y1; iy <= y2; ++iy) {
            for (int ix = x1; ix <= x2; ++ix) {
                // PSF indexing: convert from Fortran centered indexing
                // C++: lpsf[(psf_ny/2 + iy - py) * psf_nx + (psf_nx/2 + ix - px)]
                int psf_x = psf_nx / 2 + ix - px;
                int psf_y = psf_ny / 2 + iy - py;

                // Check if psf_x and psf_y are within bounds
                if (psf_x < 0 || psf_x >= psf_nx || psf_y < 0 || psf_y >= psf_ny) {
                    continue; // Skip out-of-bounds
                }

                limagestep[iy * nx + ix] -=
                    pv * lpsf[static_cast<std::size_t>(psf_y) * psf_nx + psf_x];
            }
        }
    }

    // Ran the whole budget without meeting another stop test.
    if (stop_reason == STOP_NONE && yes == 0 && max_iter_per_cycle_updated > siter
        && iter >= max_iter_per_cycle_updated) {
        stop_reason = STOP_MAX_ITER;
    }

    // The RMS after the last iteration of a plane that spent its budget: no
    // peak search followed it.
    if (history != nullptr && history->rms.size() < history->peak.size()) {
        history->rms.push_back(std::sqrt(
            searched_sum_squares<T>(limagestep, search_mask, nx,
                                    xbeg, xend, ybeg, yend)
            / n_searched));
    }

    // Output final status
    if (iter > siter) {
        msgput(iter, px, py, maxval);
    }

    // Ensure iter doesn't exceed max_iter_per_cycle_updated
    if (iter > max_iter_per_cycle_updated) {
        iter = max_iter_per_cycle_updated;
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
                int ny, int nx, int psf_ny, int psf_nx,
                int xbeg, int xend, int ybeg, int yend,
                const int* max_iter_per_cycle_updated, T gain, const T* thres, T cspeedup,
                int processing_function_threads, int* iter_out,
                const DivergenceControl<T>& divergence, int* stop_out,
                IterationHistory* history) {

    const int nplanes = nt * nf * np_img;
    if (nplanes <= 0) {
        return;
    }

    // std::thread callbacks are never invoked from cube mode; supply
    // no-ops so the inner clean<T> signature is still satisfied.
    auto noop_msgput = [](int, int, int, T) {};
    auto noop_stopnow = [](int& x) { x = 0; };

    const std::size_t plane_size = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
    const std::size_t psf_plane_size =
        static_cast<std::size_t>(psf_ny) * static_cast<std::size_t>(psf_nx);

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
                ((static_cast<std::size_t>(tt) * nf + nn) * np_psf + pidx) * psf_plane_size;

            T* residual = residual_cube + img_offset;
            T* model = model_cube + img_offset;
            const T* psf = psf_cube + psf_offset;
            const bool* mask = (domask && mask_cube != nullptr)
                ? mask_cube + img_offset
                : nullptr;

            int iter_val = 0;
            int reason = STOP_NONE;
            // Iteration control is independent per plane: each (t, f, p)
            // plane uses its own maximum iteration count and threshold, and
            // its own divergence state.
            clean<T>(model, residual, psf,
                     domask, mask,
                     nx, ny, psf_nx, psf_ny, xbeg, xend, ybeg, yend,
                     max_iter_per_cycle_updated[plane], 0, iter_val, gain, thres[plane], cspeedup,
                     noop_msgput, noop_stopnow, divergence, reason,
                     (history == nullptr) ? nullptr : history + plane);
            iter_out[plane] = iter_val;
            stop_out[plane] = reason;
        }
    };

    int nthreads = processing_function_threads;
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
 * only once instead of twice per model update. Each run(body) call dispatches
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
 * peak search and PSF subtraction are split into processing_function_threads contiguous row
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
                int ny, int nx, int psf_ny, int psf_nx,
                int xbeg, int xend, int ybeg, int yend,
                const int* max_iter_per_cycle_updated, T gain, const T* thres, T cspeedup,
                int processing_function_threads, int* iter_out,
                const DivergenceControl<T>& divergence, int* stop_out,
                IterationHistory* history) {

    const int nplanes = nt * nf * np_img;
    if (nplanes <= 0) {
        return;
    }
    const std::size_t plane_size =
        static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
    const std::size_t psf_plane_size =
        static_cast<std::size_t>(psf_ny) * static_cast<std::size_t>(psf_nx);

    std::vector<char> active(nplanes, 1);
    std::vector<PlaneStopTest<T>> stop_test(nplanes);
    for (int pl = 0; pl < nplanes; ++pl) {
        stop_out[pl] = STOP_NONE;
    }
    // Pixels the peak search of every plane covers: the denominators of the
    // recorded RMS (only with an iteration history).
    auto plane_mask = [&](int pl) -> const bool* {
        return (domask && mask_cube)
            ? mask_cube + static_cast<std::size_t>(pl) * plane_size
            : nullptr;
    };
    std::vector<Box> mask_boxes(nplanes, Box{});
    for (int pl = 0; pl < nplanes; ++pl)
        mask_boxes[pl] = mask_bounding_box(plane_mask(pl), nx, xbeg, xend, ybeg, yend);
    std::vector<double> n_searched;
    if (history != nullptr) {
        n_searched.resize(nplanes);
        for (int pl = 0; pl < nplanes; ++pl) {
            const Box& box = mask_boxes[pl];
            n_searched[pl] = static_cast<double>(searched_pixels(
                plane_mask(pl), nx, box.i0, box.i1 + 1, box.j0, box.j1 + 1));
        }
    }
    std::vector<T> row_max(static_cast<std::size_t>(nplanes) * ny);
    std::vector<int> row_ix(static_cast<std::size_t>(nplanes) * ny);
    std::vector<double> row_sum_squares(static_cast<std::size_t>(nplanes) * ny, 0.0);
    std::vector<int> peak_y(nplanes, 0);
    std::vector<int> peak_x(nplanes, 0);
    std::vector<T> peak_pv(nplanes, static_cast<T>(0));

    // Outer loop bound: the largest remaining budget over all planes.
    int max_iter_over_planes = 0;
    for (int pl = 0; pl < nplanes; ++pl) {
        if (max_iter_per_cycle_updated[pl] > max_iter_over_planes) max_iter_over_planes = max_iter_per_cycle_updated[pl];
    }

    int nthreads = processing_function_threads;
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

    for (int it = 0; it < max_iter_over_planes; ++it) {
        // ---- peak search: per (plane, row) max |residual| within the box ----
        parallel_for([&](long r0, long r1) {
            for (long r = r0; r < r1; ++r) {
                const int pl = static_cast<int>(r / ny);
                const int iy = static_cast<int>(r % ny);
                const Box& box = mask_boxes[pl];
                if (!active[pl] || it >= max_iter_per_cycle_updated[pl] || iy < box.j0 || iy > box.j1) {
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
                // Same row scan as clean<T>: row maximum and row sum of squares.
                row_sum_squares[r] = scan_row<T>(
                    res, msk, box.i0, box.i1 + 1, row_max[r], row_ix[r]);
            }
        });

        // ---- serial per-plane combine + convergence + model update ----
        bool any_active = false;
        for (int pl = 0; pl < nplanes; ++pl) {
            if (!active[pl] || it >= max_iter_per_cycle_updated[pl]) continue;
            const Box& box = mask_boxes[pl];
            T best = static_cast<T>(-1);
            int py = box.j0, px = box.i0;
            const long base = static_cast<long>(pl) * ny;
            // Row sums are added in row order, as in clean<T>, so the sum does
            // not depend on how the rows were split over the threads.
            double sum_squares = 0.0;
            for (int iy = box.j0; iy <= box.j1; ++iy) {
                T rm = row_max[base + iy];
                if (rm > best) { best = rm; py = iy; px = row_ix[base + iy]; }
                sum_squares += row_sum_squares[base + iy];
            }
            // A zero peak used to retain the supplied box's first pixel.
            // This matters only when a negative threshold keeps the plane running.
            if (best == static_cast<T>(0)) { py = ybeg; px = xbeg; }
            // The search summed the squares of the residual the plane's
            // previous iteration left: its RMS, as in clean<T>.
            if (history != nullptr && iter_out[pl] > 0) {
                history[pl].rms.push_back(std::sqrt(sum_squares / n_searched[pl]));
            }
            T cthres = thres[pl];
            if (cspeedup > zero_val) {
                cthres = thres[pl] * std::pow(two_val, static_cast<T>(it) / cspeedup);
            }
            // Threshold and divergence tests, identical to clean<T>.
            const int reason = stop_test[pl].check(best, sum_squares, cthres, divergence);
            if (reason != STOP_NONE) {
                active[pl] = 0;
                stop_out[pl] = reason;
                continue;
            }
            T* res = residual_cube + static_cast<std::size_t>(pl) * plane_size;
            T* mod = model_cube + static_cast<std::size_t>(pl) * plane_size;
            T maxval = res[static_cast<std::size_t>(py) * nx + px];
            T pv = gain * maxval;
            mod[static_cast<std::size_t>(py) * nx + px] += pv;
            if (history != nullptr) {
                history[pl].add_component(maxval, py, px, pv);
            }
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
                if (!active[pl] || it >= max_iter_per_cycle_updated[pl]) continue;
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
                          * psf_plane_size;
                const int psf_y = psf_ny / 2 + iy - py;
                if (psf_y < 0 || psf_y >= psf_ny) continue;
                T* res = residual_cube
                    + static_cast<std::size_t>(pl) * plane_size
                    + static_cast<std::size_t>(iy) * nx;
                const T* psf_row = psf + static_cast<std::size_t>(psf_y) * psf_nx;
                const int x1 = std::max(0, px - psf_nx / 2);
                const int x2 = std::min(nx - 1, px + (psf_nx - 1) / 2);
                for (int ix = x1; ix <= x2; ++ix) {
                    const int psf_x = psf_nx / 2 + ix - px;
                    if (psf_x < 0 || psf_x >= psf_nx) continue;
                    res[ix] -= pv * psf_row[psf_x];
                }
            }
        });
    }

    // Planes that no stop test ended ran their whole budget.
    for (int pl = 0; pl < nplanes; ++pl) {
        if (stop_out[pl] == STOP_NONE && max_iter_per_cycle_updated[pl] > 0
            && iter_out[pl] >= max_iter_per_cycle_updated[pl]) {
            stop_out[pl] = STOP_MAX_ITER;
        }
    }

    // The RMS after the last iteration of the planes that spent their budget:
    // no peak search followed it. One serial scan per such plane and model
    // update, summed row by row as in clean<T>.
    if (history != nullptr) {
        for (int pl = 0; pl < nplanes; ++pl) {
            if (history[pl].rms.size() < history[pl].peak.size()) {
                const Box& box = mask_boxes[pl];
                history[pl].rms.push_back(std::sqrt(
                    searched_sum_squares<T>(
                        residual_cube + static_cast<std::size_t>(pl) * plane_size,
                        plane_mask(pl), nx, box.i0, box.i1 + 1, box.j0, box.j1 + 1)
                    / n_searched[pl]));
            }
        }
    }
}

// Explicit template instantiations for float and double
template void maximg<float>(const float* limagestep, int domask, const bool* lmask,
                           int nx, int ny, float& fmin, float& fmax);

template void maximg<double>(const double* limagestep, int domask, const bool* lmask,
                            int nx, int ny, double& fmin, double& fmax);

template void clean<float>(float* limage, float* limagestep, const float* lpsf,
                          int domask, const bool* lmask, int nx, int ny, int psf_nx, int psf_ny,
                          int xbeg, int xend, int ybeg, int yend,
                          int max_iter_per_cycle_updated, int siter, int& iter, float gain, float thres,
                          float cspeedup,
                          std::function<void(int, int, int, float)> msgput,
                          std::function<void(int&)> stopnow,
                          const DivergenceControl<float>& divergence,
                          int& stop_reason, IterationHistory* history);

template void clean<double>(double* limage, double* limagestep, const double* lpsf,
                           int domask, const bool* lmask, int nx, int ny, int psf_nx, int psf_ny,
                           int xbeg, int xend, int ybeg, int yend,
                           int max_iter_per_cycle_updated, int siter, int& iter, double gain, double thres,
                           double cspeedup,
                           std::function<void(int, int, int, double)> msgput,
                           std::function<void(int&)> stopnow,
                          const DivergenceControl<double>& divergence,
                          int& stop_reason, IterationHistory* history);

template void clean_cube<float>(float* residual_cube, float* model_cube,
                                const float* psf_cube, int domask,
                                const bool* mask_cube,
                                int nt, int nf, int np_img, int np_psf,
                                int ny, int nx, int psf_ny, int psf_nx,
                                int xbeg, int xend, int ybeg, int yend,
                                const int* max_iter_per_cycle_updated, float gain, const float* thres, float cspeedup,
                                int processing_function_threads, int* iter_out,
                                const DivergenceControl<float>& divergence, int* stop_out,
                                IterationHistory* history);

template void clean_cube<double>(double* residual_cube, double* model_cube,
                                 const double* psf_cube, int domask,
                                 const bool* mask_cube,
                                 int nt, int nf, int np_img, int np_psf,
                                 int ny, int nx, int psf_ny, int psf_nx,
                                 int xbeg, int xend, int ybeg, int yend,
                                 const int* max_iter_per_cycle_updated, double gain, const double* thres, double cspeedup,
                                 int processing_function_threads, int* iter_out,
                                const DivergenceControl<double>& divergence, int* stop_out,
                                IterationHistory* history);

template void clean_cube_many_threads<float>(float* residual_cube, float* model_cube,
                                const float* psf_cube, int domask,
                                const bool* mask_cube,
                                int nt, int nf, int np_img, int np_psf,
                                int ny, int nx, int psf_ny, int psf_nx,
                                int xbeg, int xend, int ybeg, int yend,
                                const int* max_iter_per_cycle_updated, float gain, const float* thres, float cspeedup,
                                int processing_function_threads, int* iter_out,
                                const DivergenceControl<float>& divergence, int* stop_out,
                                IterationHistory* history);

template void clean_cube_many_threads<double>(double* residual_cube, double* model_cube,
                                 const double* psf_cube, int domask,
                                 const bool* mask_cube,
                                 int nt, int nf, int np_img, int np_psf,
                                 int ny, int nx, int psf_ny, int psf_nx,
                                 int xbeg, int xend, int ybeg, int yend,
                                 const int* max_iter_per_cycle_updated, double gain, const double* thres, double cspeedup,
                                 int processing_function_threads, int* iter_out,
                                const DivergenceControl<double>& divergence, int* stop_out,
                                IterationHistory* history);

} // namespace hclean
