#pragma once

// The fast dft_predict: internal interfaces of the module.
//
// It computes what the reference dft_predict computes (direct_fourier_transform.cpp
// of direct_fourier_transform_cpp), the exact direct sum over the model pixels
//
//     V_k = sum_q (M_q / n_q) exp(+2 pi i (u_k l_q + v_k m_q + w_k (n_q - 1))),
//
// by exact algebraic refactorings only, whose results differ from the exact
// value by rounding (no approximation of any term). A call takes one path,
// decided from its inputs alone (all of its model planes, rows and w) before
// anything is written (dft_separable.cpp):
//
//   separable (path S)  every used row has w = 0: the sum separates into the
//                       GEMM of dft_separable.cpp. A sample's bits depend only
//                       on its u, v and model plane, not on the threads, the
//                       blocking or the ISA, so they are the same however the
//                       rows and channels are split into calls that take S;
//   declined            anything else, including a call that cannot allocate
//                       its buffers or start its threads; nothing is written
//                       and the caller runs the reference binding (its bits,
//                       its exceptions).
//
// Path B (the exact w term, WP6) becomes a third path for calls with w != 0.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "dft_s_kernels.hpp"

namespace dft_fast {

enum class Path { separable, declined };

class Team;  // the workers of a call's parallel loops, below

enum class Element { float32, float64, complex64, complex128 };

// One call, as the reference binding sees it after its checks: C-contiguous
// arrays with the reference's shapes, maps within their image axes and
// direction cosines in front of the horizon (python/bindings.cpp). The inputs
// are aligned; vis may be at any address (path S reads and writes its values
// as bytes), and none of them may change during the call.
struct Call {
    const void* model;  // (m_time, m_chan, m_pol, n_l, n_m) of model_type
    void* vis;          // (n_time, n_baseline, n_vis_chan, n_pol) of vis_type
    const double* uvw;  // (n_time, n_baseline, 3)
    const double* frequency;
    const i64* frequency_map;
    const i64* time_map;
    const i64* pol_map;
    const double* l;
    const double* m;
    i64 m_time, m_chan, m_pol, n_l, n_m;
    i64 n_time, n_baseline, n_vis_chan, n_pol;
    Element model_type, vis_type;
    int threads;  // processing_function_threads
};

// Decides the path of a call and, for path S, writes its visibilities: the
// samples that the reference writes (those of the used rows whose value is
// not NaN), and no other byte of vis. Called without the GIL. Raises nothing
// for a resource failure: a call that cannot allocate or start its threads
// declines with vis untouched.
Path predict(const Call& call);

// The routines of the instruction set chosen for this process
// (astroviper::fast::select_isa of fast/common/isa.hpp), read once, at the
// first call (the module's import). Throws std::invalid_argument for an
// invalid ASTROVIPER_KERNELS_ISA.
const Kernels& kernels();

// Path S's view of one model plane: one time and channel of the model, with
// all of its referenced polarizations (so a polarization's bits may depend on
// which others the call references, at rounding level): the real planes it
// sums and where they are non-zero (dft_pack.cpp).
struct PlaneLayout {
    // A real component plane: the real (part 0) or imaginary (part 1) part of
    // the model polarization pol.
    struct Component {
        i64 pol;
        int part;
    };
    // Visibility polarization p is re_sign X[re] + i im_sign X[im], with X[c]
    // the sum of component c and index -1 for a part that is zero everywhere.
    struct Correlation {
        int re = -1, im = -1;
        double re_sign = 1.0, im_sign = 1.0;
    };

    // The distinct non-zero components of the referenced polarizations; one
    // bitwise equal to another, or to its bitwise negation, is summed once.
    std::vector<Component> components;
    std::vector<Correlation> correlations;  // per visibility polarization
    // Kept image rows: those non-zero in some component.
    std::vector<i64> rows;
    // Folded columns: column_a is a kept image column, column_b the kept column
    // whose m is bitwise -m[column_a] (paired, m > 0 in column_a) or -1; m_a is
    // m[column_a]. Pairs first, then the unpaired kept columns.
    std::vector<i64> column_a, column_b;
    std::vector<double> m_a;
    // Pixels where a referenced polarization is non-zero: the reference's
    // terms per sample, the measure of the cost rule.
    i64 nonzero_pixels = 0;
};

// The layout of plane (the plane's m_pol polarizations of n_l x n_m pixels),
// or false if a referenced value is not finite or so large that a sum of path
// S could overflow. One pass over the plane on up to `threads` workers; the
// result does not depend on their number.
template <typename ModelT>
bool analyse_plane(const ModelT* plane, i64 n_l, i64 n_m, const i64* pol_map, i64 n_pol,
                   const double* m, int threads, PlaneLayout& layout);

// 64-byte aligned doubles, uninitialised, kept between uses: get(n) returns
// room for n values (reallocated only when it grows).
class Buffer {
public:
    double* get(i64 size) {
        if (size > capacity_) {
            data_.reset(static_cast<double*>(
                ::operator new[](static_cast<std::size_t>(size) * sizeof(double), kAlignment)));
            capacity_ = size;
        }
        return data_.get();
    }

private:
    static constexpr std::align_val_t kAlignment{64};
    struct Free {
        void operator()(double* p) const noexcept { ::operator delete[](p, kAlignment); }
    };
    std::unique_ptr<double[], Free> data_;
    i64 capacity_ = 0;
};

// Per-thread buffers of path S (the thread's index in parallel_for).
struct Scratch {
    Buffer n_a, n_b;                // packing: n of one row's folded columns
    Buffer u, v;                    // a block's samples
    Buffer column_cos, column_sin;  // their column phasors
    Buffer t_re, t_im;              // the GEMM's band of rows
    Buffer row_cos, row_sin;        // the phasors of one kept row
    Buffer x_re, x_im;              // the sums of every component
};

// The model plane's P = W_a + W_b and Q = W_a - W_b (W = M / n formed as the
// reference forms it; W_a alone for an unpaired column) for every kept row and
// component, row r = kept row * components + component, as micro-panels of
// kTileRows rows: element (r, f) at ((r / kTileRows) * columns + f) * kTileRows
// + r % kTileRows. The rows past the last (up to a multiple of kTileRows) are 0.
// One item of the team's loop per kept row, with the scratch of its worker.
template <typename ModelT>
void pack_plane(const ModelT* plane, i64 n_l, i64 n_m, const PlaneLayout& layout,
                const double* l, const double* m, Team& team, std::vector<Scratch>& scratch,
                double* p, double* q);

// Workers of a call: processing_function_threads as the reference resolves it
// (<= 0: the hardware concurrency), at most one per item of work.
inline int resolve_threads(int processing_function_threads, i64 n_work) {
    int threads = processing_function_threads;
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 1;
    }
    if (static_cast<i64>(threads) > n_work) threads = static_cast<int>(std::max<i64>(n_work, 1));
    return threads;
}

// The workers of a call's parallel loops: `workers` threads, the calling
// thread as worker 0, started together when the team is made and kept for all
// of its loops, so that a failure to start one (std::system_error or
// std::bad_alloc, thrown by the constructor once the started ones have
// stopped) comes before the loops. run(n_items, body) calls body(worker, item)
// for every item < n_items, each worker taking the next item from a shared
// counter, and returns when every worker is done; the first exception of a
// worker stops the others at their next item and is rethrown. Which worker
// computes an item depends on timing, but an item's result does not: items
// write disjoint outputs with the same operations whoever runs them.
class Team {
public:
    explicit Team(int workers) {
        try {
            threads_.reserve(static_cast<std::size_t>(std::max(workers - 1, 0)));
            for (int worker = 1; worker < workers; ++worker)
                threads_.emplace_back(&Team::serve, this, worker);
        } catch (...) {
            stop();
            throw;
        }
    }
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;
    ~Team() { stop(); }

    template <typename Body>
    void run(i64 n_items, Body&& body) {
        using Function = std::remove_reference_t<Body>;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            loop_ = {n_items, std::addressof(body), [](void* function, int worker, i64 item) {
                         (*static_cast<Function*>(function))(worker, item);
                     }};
            next_.store(0, std::memory_order_relaxed);
            failed_.store(false, std::memory_order_relaxed);
            busy_ = threads_.size();
            ++generation_;
        }
        wake_.notify_all();
        work(0);
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return busy_ == 0; });
        if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
    }

private:
    struct Loop {
        i64 n_items = 0;
        void* body = nullptr;
        void (*call)(void*, int, i64) = nullptr;
    };

    // The loop's items, from the counter; loop_ is read without the lock,
    // after the lock that published it (the workers wait for a new
    // generation under it).
    void work(int worker) {
        try {
            for (i64 item = next_.fetch_add(1, std::memory_order_relaxed);
                 item < loop_.n_items && !failed_.load(std::memory_order_relaxed);
                 item = next_.fetch_add(1, std::memory_order_relaxed)) {
                loop_.call(loop_.body, worker, item);
            }
        } catch (...) {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
            failed_.store(true, std::memory_order_relaxed);
        }
    }

    void serve(int worker) {
        std::uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            lock.unlock();
            work(worker);
            lock.lock();
            if (--busy_ == 0) done_.notify_one();
        }
    }

    void stop() noexcept {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (std::thread& thread : threads_) thread.join();
    }

    std::mutex mutex_;
    std::condition_variable wake_, done_;
    std::vector<std::thread> threads_;
    Loop loop_;
    std::atomic<i64> next_{0};
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    std::size_t busy_ = 0;  // workers other than 0 that have not finished the loop
    std::uint64_t generation_ = 0;
    bool stopping_ = false;
};

// One loop on a team of its own (Team): body(worker, item) for every item <
// n_items on `threads` workers.
template <typename Body>
void parallel_for(int threads, i64 n_items, Body&& body) {
    Team team(threads);
    team.run(n_items, body);
}

}  // namespace dft_fast
