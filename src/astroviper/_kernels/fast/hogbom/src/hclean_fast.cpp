// The fast Hogbom minor cycle (hclean_fast.hpp): the plans of the planes, the
// threads of a call and the iterations of each plane. The passes themselves
// are in hclean_fast_kernels.inc, built once per instruction set.

#include "hclean_fast.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include "isa.hpp"

namespace hclean_fast {

namespace {

using deconvolvers::PlaneStopTest;
using deconvolvers::STOP_MAX_ITER;
using deconvolvers::STOP_NONE;
using std::size_t;

// The pass that justifies a thread of a crew (E): one thread per
// kPixelsPerThread pixels of a pass over the crew's largest plane, so that a
// pass gives each thread about 10 us of work against about a microsecond to
// synchronise. Half as many made crews of two threads on 128 x 128 planes,
// which were slower than one thread per plane at 12 and 24 threads (M).
constexpr long kPixelsPerThread = 16384;

// The same for the crew of a lone iterating plane, whose other threads would
// only wait: 8 threads on a 256 x 256 plane were 1.5x as fast as 4 (M, the
// e2e review's variant).
constexpr long kLonePixelsPerThread = 4096;

// A call starts a thread for at most every kUpdatesPerThread pixel updates of
// its work (E): starting, running and joining a thread costs about 25-35 us
// (M, a call of 24 planes of 8 x 8 for one iteration on 1 to 24 threads),
// about 30,000 updates of a pass, so that a thread gets 4 times its cost.
constexpr double kUpdatesPerThread = 131072.0;

// The work of an iteration of the direct pass, in such updates (E): about one
// per kDirectPixelsPerUpdate pixels of the box it scans and of the footprint
// it subtracts (at most the plane), and kIterationUpdates for the stop tests,
// the component and the setup of the pass, which outweigh the pixels of a
// tiny plane.
constexpr double kDirectPixelsPerUpdate = 4.0;
constexpr double kIterationUpdates = 16.0;

// Planes cleaned side by side get about sqrt(work / kThreadUpdates) threads
// (E), which minimises work / n + n x the cost of a thread: about 22 us each
// in the reference clean_cube (M: 24 planes of 8 x 8 for 200 iterations took
// 280 us at 12 threads and 571 us at 24, for about 340 us of work).
constexpr double kThreadUpdates = 20000.0;

// The planes that the direct pass cleans (E, measured on this machine): planes
// of at most kDirectPixels pixels, and planes of at most kDirectWidth columns
// and kDirectNarrowPixels pixels (512 KiB of float64, the second-level cache).
// On those the split into S and E saves little, and its plan, the per-row
// results of its passes and the deferred subtractions over short runs cost
// more than an immediate subtraction. A larger plane is better split: the
// direct pass reads its pixels twice an iteration (to subtract and to scan),
// the split's fused pass once (M: 12 planes of 1400 x 48 with a PSF of twice
// their size took 26 ms in the reference and 43 ms direct at 12 threads).
constexpr long kDirectPixels = 8192;
constexpr int kDirectWidth = 64;
constexpr long kDirectNarrowPixels = 65536;

// A small clean box on a plane of at least kSharedFinalPixels pixels leaves
// final passes (the deferred subtractions from the rest of the plane) that
// the call's threads share in a team (E): on smaller planes they are too
// short for a team's start to pay.
constexpr size_t kSharedFinalPixels = 65536;

// The direct pass holds a plane column by column (ColumnPass) when the box it
// scans has at most kColumnWidth columns and either at least kColumnRows rows
// or a plane of at most kColumnWidth columns and as many rows, from its pass
// kRowPasses + 2 on, if its budget exceeds 2 kRowPasses (E): along so few
// columns, lanes along the rows find too little to do, while a lane per row
// fills them, and on so narrow a plane the footprint's columns are as long as
// its rows or longer. The transposes of the plane, its mask and its PSF cost
// about two of its passes, so it makes its first passes row by row: a plane
// that stops early pays none of them (M: one that stopped after 5 lost 3 %).
constexpr int kColumnWidth = 64;
constexpr int kColumnRows = 8;
constexpr int kRowPasses = 8;

// The values and mask bytes after a plane held column by column that the
// column pass reads (and ignores): the most lanes of a routine.
constexpr size_t kColumnPad = 8;

// The pixels of a tile of the final pass over a plane, which applies the
// deferred subtractions and takes the binding's summary (whole rows).
constexpr int kTilePixels = 8192;

// The subtractions from E that a plane defers at most: when that many are
// pending they are applied, in order, so that the memory of a plane does not
// grow with its iterations (32 bytes each, 512 KiB in all).
constexpr size_t kMaxPending = 16384;

// ------------------------------------------------------------------ threads

inline void relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// How long a thread polls before it sleeps, for one waiter: doubled after a
// wait that ended while polling, halved after one that had to sleep (the
// event came late, or the machine is oversubscribed), within [kMinPoll,
// kMaxPoll], starting from kMaxPoll. A crew that keeps pace polls through its
// short gaps; one that waits long sleeps and burns no CPU. A wait that sleeps
// lets its CPU go idle, and waking an idle CPU can cost hundreds of
// microseconds: starting from kMinPoll made 64 x 64 planes at 12 and 24
// threads take 1.3-1.4x as long (M). The first wait of a team's thread, for
// the rest of the team to be started, may poll for up to kStartPoll.
class Patience {
public:
    // Starting and joining a thread costs about 25-35 us (M), so a team of 24
    // starts in under a millisecond (E).
    static constexpr std::chrono::nanoseconds kStartPoll{1'000'000};

    // `first` is the budget of the first wait.
    explicit Patience(std::chrono::nanoseconds first = kMaxPoll) : budget_(first) {}

    // Polls ready() until it holds (true) or the budget is spent (false).
    template <typename Ready>
    bool poll(Ready ready) {
        if (ready()) return true;
        const auto deadline = Clock::now() + budget_;
        for (;;) {
            for (int i = 0; i < kRelaxes; ++i) {
                relax();
                if (ready()) {
                    if (budget_ != kMaxPoll) budget_ = std::min(2 * budget_, kMaxPoll);
                    return true;
                }
            }
            if (Clock::now() >= deadline) {
                if (budget_ != kMinPoll) budget_ = std::clamp(budget_ / 2, kMinPoll, kMaxPoll);
                return false;
            }
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr std::chrono::nanoseconds kMinPoll{1'000};
    static constexpr std::chrono::nanoseconds kMaxPoll{50'000};
    static constexpr int kRelaxes = 32;  // between two readings of the clock
    std::chrono::nanoseconds budget_;
};

// A crew: size() threads that run their leader's steps together. The leader
// (rank 0) calls run(step), which calls step(rank) on every rank and returns
// when all have returned; the other ranks call serve(rank), which runs the
// steps until the leader calls dismiss() or runs a step with `last`, after
// which they return without waiting to be dismissed. A step starts only once
// the previous one has ended on every rank. A step must not throw (it is
// noexcept, which run() checks): a rank that left a step early would never be
// awaited.
class Crew {
public:
    int size() const { return size_; }

    // Before any step (the ranks do not read it).
    void resize(int size) { size_ = size; }

    template <typename Step>
    void run(Step& step, bool last = false) {
        static_assert(std::is_nothrow_invocable_v<Step&, int>, "a crew's steps must be noexcept");
        if (size_ == 1) {
            step(0);
            return;
        }
        call_ = [](void* function, int rank) noexcept { (*static_cast<Step*>(function))(rank); };
        step_ = &step;
        last_ = last;
        pending_.store(size_ - 1, std::memory_order_relaxed);
        advance();
        step(0);
        if (patience_.poll([this] { return pending_.load(std::memory_order_acquire) == 0; })) return;
        leader_asleep_.store(true, std::memory_order_seq_cst);
        for (int left; (left = pending_.load(std::memory_order_seq_cst)) != 0;) {
            pending_.wait(left, std::memory_order_seq_cst);
        }
        leader_asleep_.store(false, std::memory_order_relaxed);
    }

    void dismiss() {
        dismissed_ = true;
        advance();
    }

    void serve(int rank, Patience patience = Patience()) {
        for (std::uint32_t seen = 0;;) {
            seen = await(seen, patience);
            if (dismissed_) return;
            const bool last = last_;  // the leader writes it again only after this step
            call_(step_, rank);
            if (pending_.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
                leader_asleep_.load(std::memory_order_seq_cst)) {
                pending_.notify_one();
            }
            if (last) return;
        }
    }

private:
    // Publishes the step (or the dismissal) written before it.
    void advance() {
        generation_.fetch_add(1, std::memory_order_seq_cst);
        if (asleep_.load(std::memory_order_seq_cst) > 0) generation_.notify_all();
    }

    // The generation after `seen`, polled for, then slept for.
    std::uint32_t await(std::uint32_t seen, Patience& patience) {
        std::uint32_t now = seen;
        const auto changed = [&] {
            now = generation_.load(std::memory_order_acquire);
            return now != seen;
        };
        if (patience.poll(changed)) return now;
        asleep_.fetch_add(1, std::memory_order_seq_cst);
        while ((now = generation_.load(std::memory_order_seq_cst)) == seen) {
            generation_.wait(seen, std::memory_order_seq_cst);
        }
        asleep_.fetch_sub(1, std::memory_order_seq_cst);
        return now;
    }

    int size_ = 1;
    void (*call_)(void*, int) noexcept = nullptr;
    void* step_ = nullptr;
    bool last_ = false;
    bool dismissed_ = false;
    alignas(64) Patience patience_;  // the leader's, written only when it adapts
    alignas(64) std::atomic<std::uint32_t> generation_{0};
    alignas(64) std::atomic<int> pending_{0};
    std::atomic<bool> leader_asleep_{false};
    alignas(64) std::atomic<int> asleep_{0};
};

// The leader's side of a crew for a scope: dismisses it at the end, also
// when an exception leaves the scope.
class Dismissal {
public:
    explicit Dismissal(Crew& crew) : crew_(crew) {}
    Dismissal(const Dismissal&) = delete;
    Dismissal& operator=(const Dismissal&) = delete;
    ~Dismissal() { crew_.dismiss(); }

private:
    Crew& crew_;
};

// The first exception that the threads of a group throw, kept for the thread
// that waits for them all; the others read failed() to stop early.
class FirstError {
public:
    void record(std::exception_ptr error) noexcept {
        if (!failed_.exchange(true, std::memory_order_acq_rel)) error_ = std::move(error);
    }

    bool failed() const { return failed_.load(std::memory_order_relaxed); }

    // By the waiting thread, once every thread of the group has finished.
    void rethrow() {
        if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
    }

private:
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
};

// The threads of a call: the calling thread (member 0) and up to size - 1
// more, which run every phase with it (phase(member) on each member) until
// its last phase (run_last) or until the team is destroyed; when it is, they
// have all ended. A thread that cannot be started leaves a smaller team: the
// results do not depend on the number of threads. The first exception that a
// phase throws on any member is rethrown by run() or run_last() once every
// member has finished the phase.
class Team {
public:
    explicit Team(int size) {
        threads_.reserve(static_cast<size_t>(size - 1));
        for (int member = 1; member < size; ++member) {
            try {
                threads_.emplace_back(
                    [this, member] { crew_.serve(member, Patience(Patience::kStartPoll)); });
            } catch (const std::system_error&) {
                break;
            } catch (const std::bad_alloc&) {
                break;
            }
        }
        crew_.resize(static_cast<int>(threads_.size()) + 1);
    }
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;
    ~Team() {
        crew_.dismiss();
        for (std::thread& thread : threads_) thread.join();
    }

    int size() const { return crew_.size(); }

    bool failed() const { return error_.failed(); }

    template <typename Phase>
    void run(Phase&& phase) {
        run_phase(phase, false);
    }

    // The team's last phase: its threads end with it, so that none has to be
    // woken to end (the team runs nothing after it).
    template <typename Phase>
    void run_last(Phase&& phase) {
        run_phase(phase, true);
    }

private:
    template <typename Phase>
    void run_phase(Phase& phase, bool last) {
        auto guarded = [&](int member) noexcept {
            try {
                phase(member);
            } catch (...) {
                error_.record(std::current_exception());
            }
        };
        crew_.run(guarded, last);
        error_.rethrow();
    }

    Crew crew_;  // constructed before the threads start and destroyed after they end
    std::vector<std::thread> threads_;
    FirstError error_;
};

// The CPUs this process may run on.
int available_cpus() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) return count;
    }
#endif
    const unsigned count = std::thread::hardware_concurrency();
    return count > 0 ? static_cast<int>(count) : 1;
}

// ------------------------------------------------------------------ buffers

// The cache line of the buffers that different threads write.
constexpr size_t kLine = 64;

// An allocator of whole cache lines, so that the buffers of different threads
// (the row results of every pass, the deferred subtractions, the plans) share
// none: a line that two threads write makes them wait on each other at every
// pass. 64 planes of 8 x 8 at 12 threads took 2.3 times the CPU time per
// plane and iteration of one thread on one plane, with tiny row buffers
// allocated one after another (M).
template <typename T>
struct LineAllocator {
    using value_type = T;

    LineAllocator() = default;
    template <typename U>
    LineAllocator(const LineAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        if (n > (static_cast<size_t>(-1) - kLine) / sizeof(T)) throw std::bad_alloc();
        const size_t bytes = (n * sizeof(T) + kLine - 1) / kLine * kLine;
        return static_cast<T*>(::operator new(bytes, std::align_val_t{kLine}));
    }
    void deallocate(T* p, size_t) noexcept { ::operator delete(p, std::align_val_t{kLine}); }
};

template <typename T, typename U>
bool operator==(const LineAllocator<T>&, const LineAllocator<U>&) noexcept {
    return true;
}

// A buffer of a call, in whole cache lines of its own.
template <typename T>
using Buffer = std::vector<T, LineAllocator<T>>;

// ------------------------------------------------------------------ plans

// The split of a plane into the pixels the passes keep current, S (the
// segments of the searched rows), and the others, E.
struct Plan {
    int scan_begin = 0;  // the first column of clean<T>'s box (mask_bounding_box)
    long searched = 0;   // the searched pixels: the clean box inside the mask
    long work = 0;       // the pixels of the segments, a pass's work
    Buffer<SearchRow> rows;
    Buffer<Run> runs;            // the segments of S
    Buffer<Run> exterior;        // the runs of E: those of row y are
    Buffer<int> exterior_start;  // exterior[exterior_start[y], exterior_start[y + 1])
};

// Eight bool bytes all false, all true.
constexpr std::uint64_t kNoneSet = 0;
constexpr std::uint64_t kAllSet = 0x0101010101010101ULL;

// Appends the runs of the mask bytes in [x0, x1) that are not 0 (true, as the
// reference tests them) to runs.
void mask_runs(const std::uint8_t* mask, int x0, int x1, Buffer<Run>& runs) {
    int x = x0;
    while (x < x1) {
        for (std::uint64_t bytes; x + 8 <= x1; x += 8) {
            std::memcpy(&bytes, mask + x, 8);
            if (bytes != kNoneSet) break;
        }
        while (x < x1 && mask[x] == 0) ++x;
        if (x == x1) break;
        const int begin = x;
        for (std::uint64_t bytes; x + 8 <= x1; x += 8) {
            std::memcpy(&bytes, mask + x, 8);
            if (bytes != kAllSet) break;
        }
        while (x < x1 && mask[x] != 0) ++x;
        runs.push_back({begin, x});
    }
}

// Runs of the mask at most this many pixels apart are merged into one
// segment of a searched row (E): updating the pixels between them at every
// iteration costs less than a segment of their own, in the passes and in the
// final subtractions.
constexpr int kMergeGap = 32;

// Reserves all that build_plan can need for a plane of ny rows with a clean
// box of w x h pixels, so that building its plan allocates nothing. The
// segments of a row are more than kMergeGap pixels apart, so a row of w
// pixels has at most (w + kMergeGap + 1) / (kMergeGap + 2) of them, and E one
// run more. (The mask runs of a row, build_plan's `runs`, are at most
// w / 2 + 1.)
void reserve_plan(Plan& plan, int w, int h, int ny) {
    const size_t rows = static_cast<size_t>(std::max(h, 0));
    const size_t segments = rows * static_cast<size_t>((std::max(w, 0) + kMergeGap + 1) / (kMergeGap + 2));
    plan.rows.reserve(rows);
    plan.runs.reserve(segments);
    plan.exterior.reserve(static_cast<size_t>(ny) + segments);
    plan.exterior_start.reserve(static_cast<size_t>(ny) + 1);
}

// The plan of a plane with the binding's clean box [xbeg, xend) x [ybeg, yend),
// replacing what `plan` held. The searched pixels are the box inside the mask;
// clean<T>'s box (mask_bounding_box of hclean.cpp) is their bounding box, or
// the clean box itself when there are none. `runs` holds the mask runs of one
// row.
void build_plan(Plan& plan, Buffer<Run>& runs, const std::uint8_t* mask, int nx, int ny,
                int xbeg, int xend, int ybeg, int yend) {
    plan.searched = 0;
    plan.work = 0;
    plan.rows.clear();
    plan.runs.clear();
    plan.exterior.clear();
    int scan_begin = xend;
    for (int y = ybeg; y < yend; ++y) {
        runs.clear();
        if (mask == nullptr) {
            runs.push_back({xbeg, xend});
        } else {
            mask_runs(mask + static_cast<size_t>(y) * nx, xbeg, xend, runs);
        }
        if (runs.empty()) continue;
        const int first = static_cast<int>(plan.runs.size());
        Run segment = runs.front();
        long searched = segment.end - segment.begin;
        bool masked = false;
        for (size_t r = 1; r < runs.size(); ++r) {
            searched += runs[r].end - runs[r].begin;
            if (runs[r].begin - segment.end <= kMergeGap) {
                segment.end = runs[r].end;
                masked = true;
            } else {
                plan.runs.push_back(segment);
                segment = runs[r];
            }
        }
        plan.runs.push_back(segment);
        const int last = static_cast<int>(plan.runs.size());
        int pixels = 0;
        for (int r = first; r < last; ++r) pixels += plan.runs[r].end - plan.runs[r].begin;
        plan.rows.push_back({y, first, last, pixels, masked});
        plan.searched += searched;
        plan.work += pixels;
        scan_begin = std::min(scan_begin, runs.front().begin);
    }
    plan.scan_begin = plan.rows.empty() ? xbeg : scan_begin;
    plan.exterior_start.resize(static_cast<size_t>(ny) + 1);
    size_t next = 0;
    for (int y = 0; y < ny; ++y) {
        plan.exterior_start[y] = static_cast<int>(plan.exterior.size());
        int x = 0;
        if (next < plan.rows.size() && plan.rows[next].y == y) {
            const SearchRow& row = plan.rows[next++];
            for (int r = row.run_begin; r < row.run_end; ++r) {
                if (plan.runs[r].begin > x) plan.exterior.push_back({x, plan.runs[r].begin});
                x = plan.runs[r].end;
            }
        }
        if (x < nx) plan.exterior.push_back({x, nx});
    }
    plan.exterior_start[ny] = static_cast<int>(plan.exterior.size());
}

// Whether pixel (y, x) is current: in S, which the passes update.
bool in_search(const Plan& plan, int y, int x) {
    const auto row = std::lower_bound(plan.rows.begin(), plan.rows.end(), y,
                                      [](const SearchRow& r, int value) { return r.y < value; });
    if (row == plan.rows.end() || row->y != y) return false;
    for (int r = row->run_begin; r < row->run_end; ++r) {
        if (x >= plan.runs[r].begin && x < plan.runs[r].end) return true;
    }
    return false;
}

// The searched rows split into `parts` ranges of consecutive rows with about
// as much work each: part t is rows [split[t], split[t + 1]).
void split_rows(const Plan& plan, int parts, Buffer<int>& split) {
    const int n = static_cast<int>(plan.rows.size());
    split.assign(static_cast<size_t>(parts) + 1, n);
    split[0] = 0;
    long sum = 0;
    int t = 1;
    for (int i = 0; i < n && t < parts; ++i) {
        sum += plan.rows[i].pixels;
        while (t < parts && sum * parts >= plan.work * t) split[t++] = i + 1;
    }
}

// ------------------------------------------------------------------ planes

// maximg's fmin and fmax (hclean.cpp), over a part of a plane.
template <typename T>
struct Extrema {
    T low, high;
};

// maximg's starting values.
template <typename T>
Extrema<T> no_extrema() {
    if constexpr (std::is_same_v<T, float>) {
        return {1e20f, -1e20f};
    } else {
        return {1e20, -1e20};
    }
}

// A plane of a call (in cache lines of its own: threads write their planes).
template <typename T>
struct alignas(kLine) Plane {
    T* residual = nullptr;
    T* model = nullptr;
    const T* psf = nullptr;
    const std::uint8_t* mask = nullptr;
    int max_iter = 0;
    T threshold = 0;
    IterationHistory* history = nullptr;
    bool iterates = false;  // the team's schedule: its first stop test lets it iterate (a hint)
    Plan plan;              // while the plane is cleaned (a budget only)
    Buffer<Component<T>> components;  // the subtractions not yet applied to E
    Buffer<Extrema<T>> tiles;         // of a final pass that several threads share
    T total = 0;
};

// The buffers of a crew's passes.
template <typename T>
struct Rows {
    Buffer<double> sum;
    Buffer<T> max;
    Buffer<int> ix;
    Buffer<int> split;
};

// A thread's buffers, which each plane it cleans alone uses in turn
// (reserved before the first write, so that cleaning allocates nothing): a
// split plane's plan, components and rows, or a plane held column by column,
// its mask and its PSF (and which PSF that is).
template <typename T>
struct alignas(kLine) Scratch {
    Plan plan;
    Buffer<Component<T>> components;
    Buffer<Run> runs;  // the mask runs of one row
    Rows<T> rows;
    Buffer<T> columns;
    Buffer<std::uint8_t> mask_columns;
    Buffer<T> psf_columns;
    const T* psf = nullptr;
};

inline void search(const Kernels& kernels, const SearchPass<double>& pass, int begin, int end) {
    kernels.search_f64(pass, begin, end);
}

inline void search(const Kernels& kernels, const SearchPass<float>& pass, int begin, int end) {
    kernels.search_f32(pass, begin, end);
}

inline void flush(const Kernels& kernels, const ExteriorFlush<double>& exterior, int begin, int end) {
    kernels.flush_f64(exterior, begin, end);
}

inline void flush(const Kernels& kernels, const ExteriorFlush<float>& exterior, int begin, int end) {
    kernels.flush_f32(exterior, begin, end);
}

inline void direct(const Kernels& kernels, const DirectPass<double>& pass, DirectScan<double>& scan) {
    kernels.direct_f64(pass, scan);
}

inline void direct(const Kernels& kernels, const DirectPass<float>& pass, DirectScan<float>& scan) {
    kernels.direct_f32(pass, scan);
}

inline void columns(const Kernels& kernels, const ColumnPass<double>& pass, DirectScan<double>& scan) {
    kernels.columns_f64(pass, scan);
}

inline void columns(const Kernels& kernels, const ColumnPass<float>& pass, DirectScan<float>& scan) {
    kernels.columns_f32(pass, scan);
}

template <typename T>
size_t plane_pixels(const Cube<T>& c) {
    return static_cast<size_t>(c.ny) * static_cast<size_t>(c.nx);
}

// Whether the direct pass cleans the planes of a call (they share their shape).
template <typename T>
bool unsplit_planes(const Cube<T>& c) {
    const size_t pixels = plane_pixels(c);
    return pixels <= static_cast<size_t>(kDirectPixels) ||
           (c.nx <= kDirectWidth && pixels <= static_cast<size_t>(kDirectNarrowPixels));
}

template <typename T>
int tile_rows(const Cube<T>& c) {
    return std::max(1, kTilePixels / c.nx);
}

template <typename T>
int tiles(const Cube<T>& c) {
    const int rows = tile_rows(c);
    return (c.ny + rows - 1) / rows;
}

// The subtractions of the plane not yet applied to E, as an exterior flush.
template <typename T>
ExteriorFlush<T> pending(const Cube<T>& c, const Plane<T>& p) {
    return {p.residual, c.nx, p.psf, c.psf_nx, p.plan.exterior.data(), p.plan.exterior_start.data(),
            p.components.data(), static_cast<int>(p.components.size())};
}

// The component that subtracts pv x PSF at pixel (py, px): the footprint and
// PSF indices of clean<T> (whose bounds check never fails there).
template <typename T>
Component<T> component(const Cube<T>& c, T pv, int py, int px) {
    Component<T> k;
    k.pv = pv;
    k.x1 = std::max(0, px - c.psf_nx / 2);
    k.y1 = std::max(0, py - c.psf_ny / 2);
    k.x2 = std::min(c.nx - 1, px + (c.psf_nx - 1) / 2);
    k.y2 = std::min(c.ny - 1, py + (c.psf_ny - 1) / 2);
    k.offset = static_cast<std::ptrdiff_t>(c.psf_ny / 2 - py) * c.psf_nx + (c.psf_nx / 2 - px);
    return k;
}

// Applies the plane's pending subtractions to E, with its crew, and forgets
// them.
template <typename T>
void flush_exterior(const Cube<T>& c, Plane<T>& p, Crew& crew, const Kernels& kernels) {
    const ExteriorFlush<T> exterior = pending(c, p);
    const int rows = tile_rows(c), n_tiles = tiles(c);
    std::atomic<int> next{0};
    auto step = [&](int) noexcept {
        for (int t; (t = next.fetch_add(1, std::memory_order_relaxed)) < n_tiles;) {
            flush(kernels, exterior, t * rows, std::min(c.ny, (t + 1) * rows));
        }
    };
    crew.run(step);
    p.components.clear();
}

// The binding's total of a plane: |model[i]| summed in T, one sequential sum
// in index order. A model is mostly zeros, and skipping its +0 terms is exact
// (the sum is >= +0 or NaN): eight values at a time are tested for +-0 at
// once, by their bits without the sign.
template <typename T>
T sum_of_magnitudes(const T* model, size_t n) noexcept {
    using Bits = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
    constexpr Bits kMagnitude = static_cast<Bits>(~Bits{0}) >> 1;
    constexpr size_t kGroup = 8;
    T total = static_cast<T>(0);
    size_t i = 0;
    for (; i + kGroup <= n; i += kGroup) {
        Bits bits[kGroup];
        std::memcpy(bits, model + i, sizeof(bits));
        Bits any = 0;
        for (size_t j = 0; j < kGroup; ++j) any |= bits[j] & kMagnitude;
        if (any == 0) continue;
        for (size_t j = 0; j < kGroup; ++j) {
            const T magnitude = std::abs(model[i + j]);
            if (magnitude != static_cast<T>(0)) total += magnitude;
        }
    }
    for (; i < n; ++i) {
        const T magnitude = std::abs(model[i]);
        if (magnitude != static_cast<T>(0)) total += magnitude;
    }
    return total;
}

// maximg's extrema of rows [y0, y1) of a plane folded into e, in index order.
template <typename T>
void fold_extrema(const Cube<T>& c, const Plane<T>& p, int y0, int y1, Extrema<T>& e) noexcept {
    const size_t begin = static_cast<size_t>(y0) * c.nx, end = static_cast<size_t>(y1) * c.nx;
    for (size_t i = begin; i < end; ++i) {
        if (p.mask == nullptr || p.mask[i] != 0) {
            const T w = p.residual[i];
            if (w > e.high) e.high = w;
            if (w < e.low) e.low = w;
        }
    }
}

// Item i of a final pass that several threads share: i = 0 sums |model| (the
// binding's total), and i = 1 + t applies the pending subtractions to the
// rows of tile t and takes maximg's extrema over them.
template <typename T>
void finish_item(const Cube<T>& c, Plane<T>& p, int item, const Kernels& kernels) noexcept {
    if (item == 0) {
        p.total = sum_of_magnitudes(p.model, plane_pixels(c));
        return;
    }
    const int tile = item - 1, rows = tile_rows(c);
    const int y0 = tile * rows, y1 = std::min(c.ny, y0 + rows);
    if (!p.components.empty()) flush(kernels, pending(c, p), y0, y1);
    Extrema<T> e = no_extrema<T>();
    fold_extrema(c, p, y0, y1, e);
    p.tiles[tile] = e;
}

// The extrema of the tiles merged in order, as one scan finds them.
template <typename T>
Extrema<T> merged(const Buffer<Extrema<T>>& tiles) {
    Extrema<T> e = no_extrema<T>();
    for (const Extrema<T>& tile : tiles) {
        if (tile.high > e.high) e.high = tile.high;
        if (tile.low < e.low) e.low = tile.low;
    }
    return e;
}

// The binding's summary of plane `index` from maximg's extrema of its final
// residual: final_peak, converged and total_flux.
template <typename T>
void summarize(const Cube<T>& c, int index, const Plane<T>& p, const Extrema<T>& e) {
    const T peak = std::max(std::abs(e.low), std::abs(e.high));
    c.final_peak[index] = peak;
    c.total_flux[index] = p.total;
    c.converged[index] = peak <= p.threshold;
}

// The final pass over one plane, with its crew. A thread alone takes the
// tiles in order and folds their extrema as it goes (no tiles are kept).
template <typename T>
void finish(const Cube<T>& c, int index, Plane<T>& p, Crew& crew, const Kernels& kernels) {
    if (crew.size() == 1) {
        const int rows = tile_rows(c), n_tiles = tiles(c);
        Extrema<T> e = no_extrema<T>();
        for (int t = 0; t < n_tiles; ++t) {
            const int y0 = t * rows, y1 = std::min(c.ny, y0 + rows);
            if (!p.components.empty()) flush(kernels, pending(c, p), y0, y1);
            fold_extrema(c, p, y0, y1, e);
        }
        p.total = sum_of_magnitudes(p.model, plane_pixels(c));
        summarize(c, index, p, e);
        return;
    }
    const int items = 1 + tiles(c);
    std::atomic<int> next{0};
    auto step = [&](int) noexcept {
        for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < items;) {
            finish_item(c, p, i, kernels);
        }
    };
    crew.run(step);
    summarize(c, index, p, merged(p.tiles));
}

// Frees what plane p used while it was cleaned.
template <typename T>
void release(Plane<T>& p) {
    p.plan = Plan{};
    Buffer<Component<T>>().swap(p.components);
}

// The plane's sum of squares and peak from a pass over its searched rows, the
// rows taken in row order as in both reference kernels (the first row with
// the largest |value| wins), and the peak's pixel (px, py): clean_cube keeps
// the previous one when no pixel is above 0, clean_cube_many_threads takes the
// first pixel of the clean box.
template <typename T>
T plane_peak(const Cube<T>& c, const Plan& plan, const Rows<T>& rows, double& sum_squares, int& px,
             int& py) {
    const int n_rows = static_cast<int>(plan.rows.size());
    const T zero = static_cast<T>(0);
    sum_squares = 0.0;
    T peak = zero;
    if (c.entry == Entry::clean_cube) {
        for (int i = 0; i < n_rows; ++i) {
            sum_squares += rows.sum[i];
            if (rows.max[i] > peak) {
                peak = rows.max[i];
                px = rows.ix[i];
                py = plan.rows[i].y;
            }
        }
    } else {
        T best = static_cast<T>(-1);
        for (int i = 0; i < n_rows; ++i) {
            sum_squares += rows.sum[i];
            if (rows.max[i] > best) {
                best = rows.max[i];
                px = rows.ix[i];
                py = plan.rows[i].y;
            }
        }
        if (best > zero) {
            peak = best;
        } else {  // a zero peak (or no searched pixel): the first pixel of the clean box
            px = c.xbeg;
            py = c.ybeg;
        }
    }
    return peak;
}

// The threshold of iteration `iter`: threshold x 2^(iter / speedup) when
// speedup > 0, as in both reference kernels.
template <typename T>
T iteration_threshold(const Cube<T>& c, T threshold, int iter) {
    const T zero = static_cast<T>(0), two = static_cast<T>(2);
    return c.speedup > zero ? threshold * std::pow(two, static_cast<T>(iter) / c.speedup) : threshold;
}

// Whether plane p iterates at all: its first search and stop test as
// iterate() makes them, on one thread (a hint for the schedule: iterate()
// makes them again and decides).
template <typename T>
bool iterates(const Cube<T>& c, const Plane<T>& p, Rows<T>& rows, const Kernels& kernels) {
    const Plan& plan = p.plan;
    const SearchPass<T> pass{p.residual,      c.nx,           p.psf,          c.psf_nx,
                             p.mask,          plan.rows.data(), plan.runs.data(), nullptr,
                             rows.sum.data(), rows.max.data(), rows.ix.data(), plan.scan_begin};
    search(kernels, pass, 0, static_cast<int>(plan.rows.size()));
    double sum_squares = 0.0;
    int px = 0, py = 0;
    const T peak = plane_peak(c, plan, rows, sum_squares, px, py);
    PlaneStopTest<T> stop_test;
    return stop_test.check(peak, sum_squares, iteration_threshold(c, p.threshold, 0), c.divergence) ==
           STOP_NONE;
}

// The model update of plane `index` (a budget > 0), with its crew: clean<T>
// for Entry::clean_cube, the per-plane part of clean_cube_many_threads for
// Entry::clean_cube_many_threads, with the searched pixels kept current by
// the fused pass and the subtractions from E deferred.
template <typename T>
void iterate(const Cube<T>& c, int index, Plane<T>& p, Crew& crew, Rows<T>& rows,
             const Kernels& kernels) {
    const Plan& plan = p.plan;
    split_rows(plan, crew.size(), rows.split);
    SearchPass<T> pass{p.residual,        c.nx,           p.psf,           c.psf_nx,
                       p.mask,            plan.rows.data(), plan.runs.data(), nullptr,
                       rows.sum.data(),   rows.max.data(), rows.ix.data(),  plan.scan_begin};
    auto step = [&](int rank) noexcept {
        search(kernels, pass, rows.split[rank], rows.split[rank + 1]);
    };
    const auto scan = [&](const Component<T>* subtracted) {
        pass.component = subtracted;
        crew.run(step);
    };

    IterationHistory* const history = p.history;
    const double n_searched = static_cast<double>(plan.searched);
    const T zero = static_cast<T>(0);
    PlaneStopTest<T> stop_test;
    int stop = STOP_NONE;
    int px = 0, py = 0;  // clean<T> keeps the peak's pixel between iterations: (0, 0) at first
    scan(nullptr);
    int iter = 0;
    for (; iter < p.max_iter; ++iter) {
        double sum_squares = 0.0;
        const T peak = plane_peak(c, plan, rows, sum_squares, px, py);
        // The pass summed the squares of the residual the previous iteration
        // left: its RMS.
        if (history != nullptr && iter > 0) history->rms.push_back(std::sqrt(sum_squares / n_searched));
        const int reason =
            stop_test.check(peak, sum_squares, iteration_threshold(c, p.threshold, iter), c.divergence);
        if (reason != STOP_NONE) {
            stop = reason;
            break;
        }
        // The peak's pixel is searched, so current, unless the peak is zero:
        // then it may lie in E, whose pending subtractions are applied first.
        // They are applied as well when kMaxPending of them are pending.
        if ((!(peak > zero) && !in_search(plan, py, px)) || p.components.size() >= kMaxPending) {
            flush_exterior(c, p, crew, kernels);
        }
        const size_t at = static_cast<size_t>(py) * c.nx + px;
        const T value = p.residual[at];
        const T pv = c.gain * value;
        p.model[at] += pv;
        if (history != nullptr) history->add_component(value, py, px, pv);
        p.components.push_back(component(c, pv, py, px));
        scan(&p.components.back());
    }
    if (stop == STOP_NONE && iter >= p.max_iter) stop = STOP_MAX_ITER;
    // The RMS after the last iteration of a plane that spent its budget: the
    // last pass summed it.
    if (history != nullptr && history->rms.size() < history->peak.size()) {
        double sum_squares = 0.0;
        for (size_t i = 0; i < plan.rows.size(); ++i) sum_squares += rows.sum[i];
        history->rms.push_back(std::sqrt(sum_squares / n_searched));
    }
    c.iterations[index] = iter;
    c.stop_code[index] = stop;
}

// clean<T>'s box of a plane, [x0, x1) x [y0, y1) (mask_bounding_box of
// hclean.cpp): the bounding box of the searched pixels, which are the clean box
// inside the mask, or the clean box itself when there are none; and how many
// pixels are searched.
struct SearchBox {
    int x0, x1, y0, y1;
    long searched;
};

template <typename T>
SearchBox search_box(const Cube<T>& c, const std::uint8_t* mask) {
    const SearchBox clean{c.xbeg, c.xend, c.ybeg, c.yend,
                          static_cast<long>(c.xend - c.xbeg) * static_cast<long>(c.yend - c.ybeg)};
    if (mask == nullptr) return clean;
    SearchBox box{c.xend, c.xbeg, c.yend, c.ybeg, 0};
    for (int y = c.ybeg; y < c.yend; ++y) {
        const std::uint8_t* const row = mask + static_cast<size_t>(y) * c.nx;
        int count = 0;
        for (int x = c.xbeg; x < c.xend; ++x) count += row[x] != 0;
        if (count == 0) continue;
        int first = c.xbeg, last = c.xend - 1;
        while (row[first] == 0) ++first;
        while (row[last] == 0) --last;
        box.x0 = std::min(box.x0, first);
        box.x1 = std::max(box.x1, last + 1);
        box.y0 = std::min(box.y0, y);
        box.y1 = y + 1;
        box.searched += count;
    }
    return box.searched > 0 ? box : SearchBox{clean.x0, clean.x1, clean.y0, clean.y1, 0};
}

// out[s * rows + r] = in[r * cols + s]: a plane of rows x cols held column by
// column (bits copied as they are).
template <typename T>
void transpose(const T* in, int rows, int cols, T* out) noexcept {
    for (int r = 0; r < rows; ++r) {
        for (int s = 0; s < cols; ++s) {
            out[static_cast<size_t>(s) * rows + r] = in[static_cast<size_t>(r) * cols + s];
        }
    }
}

// The direct passes over a plane: row by row (DirectPass), and with `own`
// (a thread's buffers, reserved before the first write) column by column
// (ColumnPass) from its pass kRowPasses + 2 on, until release() writes the
// plane back.
template <typename T>
struct DirectPasses {
    const Cube<T>& c;
    const Kernels& kernels;
    Scratch<T>* own;
    DirectPass<T> rows;
    ColumnPass<T> held{};
    Component<T> subtracted{};
    int passes = 0;

    // A pass with no component, and with the component that subtracts
    // pv x PSF at pixel (py, px).
    void run(DirectScan<T>& scan) {
        rows.component = nullptr;
        direct(kernels, rows, scan);
    }
    void run(T pv, int py, int px, DirectScan<T>& scan) {
        subtracted = component(c, pv, py, px);
        if (held.columns == nullptr && own != nullptr && ++passes > kRowPasses) hold();
        if (held.columns == nullptr) {
            rows.component = &subtracted;
            direct(kernels, rows, scan);
            return;
        }
        // component()'s footprint, with the offset of the PSF held column by column.
        subtracted.offset = static_cast<std::ptrdiff_t>(c.psf_nx / 2 - px) * c.psf_ny + (c.psf_ny / 2 - py);
        held.component = &subtracted;
        columns(kernels, held, scan);
    }
    T value(int py, int px) const {
        return held.columns == nullptr ? rows.residual[static_cast<size_t>(py) * c.nx + px]
                                       : held.columns[static_cast<size_t>(px) * c.ny + py];
    }

    // The plane, its mask and its PSF (unless the thread holds it already)
    // column by column. The buffers grow within the room reserved before the
    // first write, so nothing is allocated (and the zeros after the plane,
    // which the column pass reads, are written once per thread).
    void hold() noexcept {
        const size_t pixels = static_cast<size_t>(c.ny) * static_cast<size_t>(c.nx);
        own->columns.resize(pixels + kColumnPad);
        if (rows.mask != nullptr) own->mask_columns.resize(pixels + kColumnPad);
        own->psf_columns.resize(static_cast<size_t>(c.psf_ny) * static_cast<size_t>(c.psf_nx));
        transpose(rows.residual, c.ny, c.nx, own->columns.data());
        if (rows.mask != nullptr) transpose(rows.mask, c.ny, c.nx, own->mask_columns.data());
        if (own->psf != rows.psf) {
            transpose(rows.psf, c.psf_ny, c.psf_nx, own->psf_columns.data());
            own->psf = rows.psf;
        }
        held = {own->columns.data(), c.ny, own->psf_columns.data(), c.psf_ny,
                rows.mask != nullptr ? own->mask_columns.data() : nullptr,
                rows.xbeg, rows.xend, rows.ybeg, rows.yend, nullptr};
    }
    void release() noexcept {
        if (held.columns != nullptr) transpose(held.columns, c.nx, c.ny, rows.residual);
        held.columns = nullptr;
    }
};

// The model update of plane `index` (a budget > 0) by direct passes (`pass`),
// on this thread: clean<T> for Entry::clean_cube, the per-plane part of
// clean_cube_many_threads for Entry::clean_cube_many_threads. Each pass
// subtracts the previous component from its whole footprint, then scans the
// box for the next peak, so every pixel is current and nothing is deferred.
template <typename T, typename Pass>
void iterate_direct(const Cube<T>& c, int index, Plane<T>& p, double n_searched, Pass& pass) {
    DirectScan<T> scan;
    pass.run(scan);
    IterationHistory* const history = p.history;
    const T zero = static_cast<T>(0);
    PlaneStopTest<T> stop_test;
    int stop = STOP_NONE;
    int px = 0, py = 0;  // clean<T> keeps the peak's pixel between iterations: (0, 0) at first
    int iter = 0;
    for (; iter < p.max_iter; ++iter) {
        // The pass summed the squares of the residual the previous iteration
        // left: its RMS.
        if (history != nullptr && iter > 0) history->rms.push_back(std::sqrt(scan.sum_squares / n_searched));
        // The peak and its pixel (plane_peak's conventions for a zero peak).
        T peak = zero;
        if (scan.peak > zero) {
            peak = scan.peak;
            px = scan.x;
            py = scan.y;
        } else if (c.entry == Entry::clean_cube_many_threads) {
            px = c.xbeg;
            py = c.ybeg;
        }
        const int reason =
            stop_test.check(peak, scan.sum_squares, iteration_threshold(c, p.threshold, iter), c.divergence);
        if (reason != STOP_NONE) {
            stop = reason;
            break;
        }
        const T value = pass.value(py, px);
        const T pv = c.gain * value;
        p.model[static_cast<size_t>(py) * c.nx + px] += pv;
        if (history != nullptr) history->add_component(value, py, px, pv);
        pass.run(pv, py, px, scan);
    }
    if (stop == STOP_NONE && iter >= p.max_iter) stop = STOP_MAX_ITER;
    // The RMS after the last iteration of a plane that spent its budget: the
    // last pass summed it.
    if (history != nullptr && history->rms.size() < history->peak.size()) {
        history->rms.push_back(std::sqrt(scan.sum_squares / n_searched));
    }
    c.iterations[index] = iter;
    c.stop_code[index] = stop;
}

// Whether the planes of a call may be held column by column: those of the
// direct pass whose clean box is narrow enough (or whose mask may make it so)
// and tall enough or on a narrow plane, if a budget is at least twice the
// passes made row by row.
template <typename T>
bool column_planes(const Cube<T>& c, bool unsplit, int most_iterations) {
    return unsplit && most_iterations > 2 * kRowPasses &&
           (c.yend - c.ybeg >= kColumnRows || (c.nx <= kColumnWidth && c.ny >= c.nx)) &&
           (c.xend - c.xbeg <= kColumnWidth || c.mask != nullptr);
}

// The model update of plane `index` (a budget > 0) by the direct pass, on this
// thread, held column by column in `own`'s buffers (their room reserved
// before the first write when column_planes() holds, or null) when its box
// suits that.
template <typename T>
void clean_direct(const Cube<T>& c, int index, Plane<T>& p, Scratch<T>* own, const Kernels& kernels) {
    const SearchBox box = search_box(c, p.mask);
    // A mask that keeps every pixel of the box is read as none.
    const bool every = box.searched == static_cast<long>(box.x1 - box.x0) * (box.y1 - box.y0);
    const bool narrow = box.x1 - box.x0 <= kColumnWidth && p.max_iter > 2 * kRowPasses &&
                        (box.y1 - box.y0 >= kColumnRows || (c.nx <= kColumnWidth && c.ny >= c.nx));
    const DirectPass<T> rows{p.residual, c.nx,   p.psf,  c.psf_nx, every ? nullptr : p.mask,
                             box.x0,     box.x1, box.y0, box.y1,   nullptr};
    DirectPasses<T> passes{c, kernels, narrow ? own : nullptr, rows};
    try {
        iterate_direct(c, index, p, static_cast<double>(box.searched), passes);
    } catch (...) {  // the history outgrew the memory: the plane is left as far as it got
        passes.release();
        throw;
    }
    passes.release();
}

// Whether plane p stops at its first search, as its model update would make
// it (a hint for the schedule: the update makes it again and decides).
template <typename T>
bool stops_at_once(const Cube<T>& c, const Plane<T>& p, const Kernels& kernels) {
    const SearchBox box = search_box(c, p.mask);
    const DirectPass<T> pass{p.residual, c.nx,   p.psf,  c.psf_nx, p.mask,
                             box.x0,     box.x1, box.y0, box.y1,   nullptr};
    DirectScan<T> scan;
    direct(kernels, pass, scan);
    PlaneStopTest<T> stop_test;
    const T threshold = iteration_threshold(c, p.threshold, 0);
    return stop_test.check(scan.peak, scan.sum_squares, threshold, c.divergence) != STOP_NONE;
}

// How the planes share a team of `team` threads, given the planes with a
// budget (`active`), those of them that iterate at all (`iterating`: the
// others stop at their first search, by the hint of iterates()) and the
// largest work of a pass among those (pixels).
struct Schedule {
    int crews = 1;      // crews that iterate planes side by side
    int crew_size = 1;  // threads per crew
    bool deferred = false;  // the planes' final passes wait for the whole team
};

Schedule schedule(int team, int active, int iterating, long largest_work) {
    Schedule s;
    if (iterating == 0) {
        // Every plane stops at its first search: one pass each, on threads of
        // their own, then the final passes on the whole team.
        s.crews = std::clamp(team, 1, std::max(active, 1));
        s.deferred = true;
        return s;
    }
    if (iterating >= 2 * team) {
        // At least two iterating planes per thread: whole planes per thread,
        // as the reference clean_cube runs them (a crew of one synchronises
        // nothing).
        s.crews = team;
        return s;
    }
    // A crew of more threads than its plane's pass justifies would spend more
    // time synchronising than it saves; side by side, crews share no step.
    // Planes that stop at once count for nothing, so that no crew sits idle
    // while the others iterate (Stokes V beside I).
    const long justified = std::clamp(largest_work / kPixelsPerThread, 1L, static_cast<long>(team));
    const int useful = static_cast<int>(justified);
    s.crews = std::clamp(team / useful, 1, iterating);
    // One crew runs the planes one after another: every thread for planes
    // that justify more than half of the team, and for a lone iterating
    // plane as many as its pass keeps busy (kLonePixelsPerThread each), as
    // the other threads would only wait.
    s.crew_size = s.crews > 1 ? std::min(useful, team / s.crews)
                              : static_cast<int>(std::clamp(largest_work / kLonePixelsPerThread,
                                                            static_cast<long>(useful),
                                                            static_cast<long>(team)));
    // When every iterating plane has a crew of its own but the crews leave
    // threads idle, the whole team does the final passes, after the
    // iterations (the memory of no more than `team` planes' components is
    // kept until then).
    s.deferred = s.crews == iterating && s.crews * s.crew_size < team;
    return s;
}

// The threads of a call: processing_function_threads (< 1 counts as 1, as in
// both reference entries), at most the threads its work can use (`usable`),
// those it justifies (one per kUpdatesPerThread of `updates`, the pixel
// updates of the call at most) and the CPUs the process may run on.
int team_size(int requested, long usable, double updates) {
    long threads = std::min<long>(std::max(requested, 1), usable);
    const double justified = updates / kUpdatesPerThread;
    if (justified < static_cast<double>(threads)) threads = static_cast<long>(justified);
    threads = std::max(threads, 1L);
    if (threads > 1) threads = std::min<long>(threads, available_cpus());  // a system call
    return static_cast<int>(threads);
}

// The threads of planes cleaned side by side: processing_function_threads
// (< 1 counts as 1), at most one per plane with a budget (`active`) and the
// CPUs, and about sqrt(updates / kThreadUpdates) for the call's `updates`:
// started one after another, they finish after about updates / n + n thread
// costs.
int side_by_side_threads(int requested, int active, double updates) {
    long threads = std::min<long>(std::max(requested, 1), std::max(active, 1));
    const double balanced = std::sqrt(updates / kThreadUpdates);
    if (balanced < static_cast<double>(threads)) threads = std::max(1L, std::lround(balanced));
    if (threads > 1) threads = std::min<long>(threads, available_cpus());  // a system call
    return static_cast<int>(threads);
}

// Runs work(member) on up to `threads` threads side by side, the calling
// thread being member 0 and `workers` (with room reserved) holding the others,
// as the reference clean_cube runs its threads: each starts on its work at
// once, and none waits for another. The calling thread starts the others one
// after another, and only while left() says that work is left for one (planes
// that no thread has taken): when the planes take less time than a thread's
// start, the threads already running take them all. A thread that cannot be
// started leaves fewer members. The first exception that a member throws is
// rethrown once all have ended; the others read error.failed() to stop early.
template <typename Work, typename Left>
void run_side_by_side(std::vector<std::thread>& workers, int threads, FirstError& error, Work& work,
                      const Left& left) {
    const auto guarded = [&](int member) noexcept {
        try {
            work(member);
        } catch (...) {
            error.record(std::current_exception());
        }
    };
    for (int member = 1; member < threads && left(); ++member) {
        try {
            workers.emplace_back(guarded, member);
        } catch (const std::system_error&) {
            break;
        } catch (const std::bad_alloc&) {
            break;
        }
    }
    guarded(0);
    for (std::thread& worker : workers) worker.join();
    error.rethrow();
}

}  // namespace

template <typename T>
bool clean_cube(const Cube<T>& c) {
    const int nplanes = c.nt * c.nf * c.np_img;
    if (nplanes <= 0) return true;
    const size_t plane_size = plane_pixels(c);
    const size_t psf_plane_size = static_cast<size_t>(c.psf_ny) * static_cast<size_t>(c.psf_nx);
    const Kernels& kernels = hclean_fast::kernels();
    const bool unsplit = unsplit_planes(c);  // the direct pass cleans the planes

    // Everything that can fail to allocate, before the first write.
    std::vector<Plane<T>> planes;
    int threads = 1;
    bool side_by_side = false;
    std::vector<std::thread> workers;  // side by side: the threads after the caller
    std::vector<Scratch<T>> scratch;   // side by side: each thread's buffers
    std::unique_ptr<Team> team;
    Schedule s;
    std::unique_ptr<Crew[]> crews;
    std::vector<Rows<T>> rows;  // a crew's
    try {
        planes.resize(static_cast<size_t>(nplanes));
        int active = 0, most_iterations = 0;
        // The work the call can use and justifies, at most: a pass searches
        // at most the clean box (a direct pass also subtracts from at most
        // the plane, kDirectPixelsPerUpdate pixels an update, and costs
        // kIterationUpdates besides), and the final pass reads every pixel
        // and, after a split, subtracts each component at most once from each
        // pixel of E, which is the plane outside the clean box (no mask) or at
        // most the plane.
        const int w = c.xend - c.xbeg, h = std::max(c.yend - c.ybeg, 0);
        const long box = static_cast<long>(w) * h;
        const double pixels = static_cast<double>(plane_size);
        const double iteration = unsplit ? (static_cast<double>(box) + pixels) / kDirectPixelsPerUpdate +
                                               kIterationUpdates
                                         : static_cast<double>(box);
        const double outside = unsplit ? 0.0 : static_cast<double>(plane_size - box);  // in E in any case
        const double exterior = c.mask != nullptr && !unsplit ? pixels : outside;
        double pass_updates = 0.0, final_updates = 0.0, final_least = 0.0;
        for (int pl = 0; pl < nplanes; ++pl) {
            const int t = pl / (c.nf * c.np_img);
            const int f = (pl / c.np_img) % c.nf;
            const int pol = pl % c.np_img;
            const int psf_pol = c.np_psf == 1 ? 0 : pol;
            Plane<T>& p = planes[pl];
            p.residual = c.residual + static_cast<size_t>(pl) * plane_size;
            p.model = c.model + static_cast<size_t>(pl) * plane_size;
            p.psf = c.psf + ((static_cast<size_t>(t) * c.nf + f) * c.np_psf + psf_pol) * psf_plane_size;
            p.mask = c.mask != nullptr ? c.mask + static_cast<size_t>(pl) * plane_size : nullptr;
            p.max_iter = c.max_iter[pl];
            p.threshold = c.threshold[pl];
            p.history = c.history != nullptr ? c.history + pl : nullptr;
            final_updates += pixels;
            final_least += pixels;
            if (p.max_iter > 0) {
                ++active;
                most_iterations = std::max(most_iterations, p.max_iter);
                pass_updates += iteration * p.max_iter;
                final_updates += exterior * p.max_iter;
                final_least += outside * p.max_iter;
            }
        }
        // A crew per plane with a budget, of up to one thread per
        // kPixelsPerThread pixels of its pass, or one thread per tile of the
        // final passes, if their work justifies that many (both bounded by
        // the request, which bounds the team anyway).
        const long requested = std::max(c.threads, 1);
        const auto final_threads = [&](double updates) {  // the threads of the final passes' work
            const double justified = std::min(updates / kUpdatesPerThread, static_cast<double>(requested));
            return std::min(static_cast<long>(nplanes) * tiles(c), static_cast<long>(justified));
        };
        const long crew_threads = std::clamp(box / kPixelsPerThread, 1L, requested);
        // The planes of the direct pass, and planes too small for a crew of
        // two (a pass of fewer than 2 x kPixelsPerThread pixels), get at most
        // a thread per plane with a budget (one without any) and run side by
        // side, as the reference clean_cube runs them: each thread cleans
        // whole planes (after a split, with plans in its own buffers, reserved
        // here) and waits for no other. More threads could only wait for the
        // final passes, and starting one costs about 25-35 us (M). Unless the
        // final passes are long, of a small box on a large plane: they then go
        // to the team, whose idle threads share them, if they justify more
        // threads than planes with a budget, counting only the pixels outside
        // the clean box (which E always holds) of the planes that iterate (by
        // the hint of their first search). A thread per plane made one thread
        // subtract all the deferred components from the rest of a 2048 x 2048
        // plane (8.4 times as long as a team, as the adversarial review
        // measured).
        bool unshared = plane_size < kSharedFinalPixels || tiles(c) == 1 ||
                        final_threads(final_least) <= std::max(active, 1);
        if (!unsplit && box < 2 * kPixelsPerThread && !unshared) {
            double least = 0.0;
            for (const Plane<T>& p : planes) {
                least += pixels;
                if (p.max_iter > 0 && !stops_at_once(c, p, kernels)) least += outside * p.max_iter;
            }
            unshared = final_threads(least) <= std::max(active, 1);
        }
        side_by_side = unsplit || (box < 2 * kPixelsPerThread && unshared);
        if (unsplit) {
            threads = side_by_side_threads(c.threads, active, pass_updates + final_updates);
        } else {
            const long usable = std::max(active * crew_threads, final_threads(final_updates));
            threads = team_size(c.threads, usable, pass_updates + final_updates);
            if (side_by_side) threads = std::min(threads, std::max(active, 1));
        }
        if (side_by_side) {
            const bool by_columns = column_planes(c, unsplit, most_iterations);
            scratch.resize(unsplit && !by_columns ? 0 : static_cast<size_t>(threads));
            for (Scratch<T>& own : scratch) {
                if (active == 0) break;
                if (unsplit) {  // room for a plane held column by column (and more), its mask, its PSF
                    own.columns.reserve(plane_size + kColumnPad);
                    if (c.mask != nullptr) own.mask_columns.reserve(plane_size + kColumnPad);
                    own.psf_columns.reserve(psf_plane_size);
                    continue;
                }
                reserve_plan(own.plan, w, h, c.ny);
                own.components.reserve(std::min(static_cast<size_t>(most_iterations), kMaxPending));
                own.runs.reserve(static_cast<size_t>(std::max(w, 0)) / 2 + 1);
                own.rows.sum.resize(static_cast<size_t>(h));
                own.rows.max.resize(static_cast<size_t>(h));
                own.rows.ix.resize(static_cast<size_t>(h));
                own.rows.split.reserve(2);
            }
            workers.reserve(static_cast<size_t>(threads - 1));
        } else {
            team = std::make_unique<Team>(threads);
            // The plans of the planes with a budget.
            std::atomic<int> next{0};
            team->run([&](int) {
                Buffer<Run> runs;  // of the mask, in one row
                for (int pl; (pl = next.fetch_add(1, std::memory_order_relaxed)) < nplanes;) {
                    Plane<T>& p = planes[pl];
                    if (p.max_iter <= 0) continue;
                    p.components.reserve(static_cast<size_t>(std::min(p.max_iter, 256)));
                    build_plan(p.plan, runs, p.mask, c.nx, c.ny, c.xbeg, c.xend, c.ybeg, c.yend);
                    p.iterates = true;
                }
            });
            const auto schedule_iterating = [&] {
                int iterating = 0;
                long largest = 0;
                for (const Plane<T>& p : planes) {
                    if (p.max_iter <= 0 || !p.iterates) continue;
                    ++iterating;
                    largest = std::max(largest, p.plan.work);
                }
                return schedule(team->size(), active, iterating, largest);
            };
            s = schedule_iterating();
            if (s.crews > 1) {
                // Crews side by side: only the planes that iterate at all
                // count (the hint of their first search), so that no crew
                // waits idle beside a plane that stops at once.
                next.store(0, std::memory_order_relaxed);
                team->run([&](int) {
                    Rows<T> first;  // of a plane's first search
                    for (int pl; (pl = next.fetch_add(1, std::memory_order_relaxed)) < nplanes;) {
                        Plane<T>& p = planes[pl];
                        if (p.max_iter <= 0) continue;
                        first.sum.resize(p.plan.rows.size());
                        first.max.resize(p.plan.rows.size());
                        first.ix.resize(p.plan.rows.size());
                        p.iterates = iterates(c, p, first, kernels);
                    }
                });
                s = schedule_iterating();
            }
            size_t most_rows = 0;
            for (const Plane<T>& p : planes) most_rows = std::max(most_rows, p.plan.rows.size());
            crews.reset(new Crew[static_cast<size_t>(s.crews)]);
            rows.resize(static_cast<size_t>(s.crews));
            for (int g = 0; g < s.crews; ++g) {
                crews[g].resize(s.crew_size);
                rows[g].sum.resize(most_rows);
                rows[g].max.resize(most_rows);
                rows[g].ix.resize(most_rows);
                rows[g].split.reserve(static_cast<size_t>(s.crew_size) + 1);
            }
            if (s.deferred || s.crew_size > 1) {  // final passes that several threads share
                for (Plane<T>& p : planes) p.tiles.resize(static_cast<size_t>(tiles(c)));
            }
        }
    } catch (const std::bad_alloc&) {
        return false;
    }

    for (int pl = 0; pl < nplanes; ++pl) {
        if (planes[pl].max_iter > 0) continue;
        // No iteration: clean<T> reports a negative budget as its iterations,
        // clean_cube_many_threads none.
        const int budget = planes[pl].max_iter;
        c.iterations[pl] = (c.entry == Entry::clean_cube && budget < 0) ? budget : 0;
        c.stop_code[pl] = STOP_NONE;
    }

    if (side_by_side) {
        std::atomic<int> next{0};
        FirstError error;
        auto clean_planes = [&](int member) {
            Crew alone;  // of one thread: its steps run on this one
            for (int pl; !error.failed() && (pl = next.fetch_add(1, std::memory_order_relaxed)) < nplanes;) {
                Plane<T>& p = planes[pl];
                if (p.max_iter <= 0 || unsplit) {
                    Scratch<T>* const own = scratch.empty() ? nullptr : &scratch[member];
                    if (p.max_iter > 0) clean_direct(c, pl, p, own, kernels);
                    finish(c, pl, p, alone, kernels);
                    continue;
                }
                // The plane uses this thread's buffers, and gives them back.
                Scratch<T>& own = scratch[member];
                std::swap(p.plan, own.plan);
                std::swap(p.components, own.components);
                build_plan(p.plan, own.runs, p.mask, c.nx, c.ny, c.xbeg, c.xend, c.ybeg, c.yend);
                iterate(c, pl, p, alone, own.rows, kernels);
                finish(c, pl, p, alone, kernels);
                p.components.clear();
                std::swap(p.plan, own.plan);
                std::swap(p.components, own.components);
            }
        };
        const auto left = [&] { return next.load(std::memory_order_relaxed) < nplanes && !error.failed(); };
        run_side_by_side(workers, threads, error, clean_planes, left);
        return true;
    }

    // The iterations (and final passes) of the planes, the team's last phase
    // unless the final passes are deferred to the whole team.
    std::atomic<int> next{0};
    const auto clean_planes = [&](int member) {
        const int g = member / s.crew_size, rank = member % s.crew_size;
        if (g >= s.crews) return;
        Crew& crew = crews[g];
        if (rank != 0) {
            crew.serve(rank);
            return;
        }
        const Dismissal dismissal(crew);
        for (int pl; !team->failed() && (pl = next.fetch_add(1, std::memory_order_relaxed)) < nplanes;) {
            Plane<T>& p = planes[pl];
            if (p.max_iter > 0) iterate(c, pl, p, crew, rows[g], kernels);
            if (!s.deferred) {
                finish(c, pl, p, crew, kernels);
                release(p);
            }
        }
    };
    if (!s.deferred) {
        team->run_last(clean_planes);
    } else {
        team->run(clean_planes);
        // The final passes of every plane on the whole team: first the sums of
        // |model| (the longest items), then the tiles.
        const int n_tiles = tiles(c);
        const long items = static_cast<long>(nplanes) * (1 + n_tiles);
        std::atomic<long> next_item{0};
        team->run_last([&](int) {
            for (long i; (i = next_item.fetch_add(1, std::memory_order_relaxed)) < items;) {
                if (i < nplanes) {
                    finish_item(c, planes[i], 0, kernels);
                } else {
                    const long tile = i - nplanes;
                    finish_item(c, planes[tile / n_tiles], 1 + static_cast<int>(tile % n_tiles), kernels);
                }
            }
        });
        for (int pl = 0; pl < nplanes; ++pl) summarize(c, pl, planes[pl], merged(planes[pl].tiles));
    }
    return true;
}

const Kernels& kernels() {
    static const Kernels selected = [] {
        const bool avx2 = astroviper::fast::select_isa() == astroviper::fast::Isa::avx2;
#if defined(HCLEAN_FAST_AVX2)
        if (avx2) return avx2_kernels();
#endif
        static_cast<void>(avx2);  // read in a build with the AVX2 routines only
        // The baseline routines: a CPU without AVX2, the generic override, or
        // a build without the AVX2 routines.
        return generic_kernels();
    }();
    return selected;
}

template bool clean_cube<float>(const Cube<float>&);
template bool clean_cube<double>(const Cube<double>&);

}  // namespace hclean_fast
