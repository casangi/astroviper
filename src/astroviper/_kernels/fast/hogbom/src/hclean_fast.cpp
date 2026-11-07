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
#include <mutex>
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

// A call starts a thread for at most every kUpdatesPerThread pixel updates of
// its work (E): starting, running and joining a thread costs about 25-35 us
// (M, a call of 24 planes of 8 x 8 for one iteration on 1 to 24 threads),
// about 30,000 updates of a pass, so that a thread gets 4 times its cost.
constexpr double kUpdatesPerThread = 131072.0;

// The pixels of a tile of the final pass over a plane, which applies the
// deferred subtractions and takes the binding's summary (whole rows).
constexpr int kTilePixels = 8192;

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
// the previous one has ended on every rank.
class Crew {
public:
    int size() const { return size_; }

    // Before any step (the ranks do not read it).
    void resize(int size) { size_ = size; }

    template <typename Step>
    void run(Step& step, bool last = false) {
        if (size_ == 1) {
            step(0);
            return;
        }
        call_ = [](void* function, int rank) { (*static_cast<Step*>(function))(rank); };
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
    void (*call_)(void*, int) = nullptr;
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

    bool failed() const { return failed_.load(std::memory_order_relaxed); }

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
        auto guarded = [&](int member) {
            try {
                phase(member);
            } catch (...) {
                fail(std::current_exception());
            }
        };
        crew_.run(guarded, last);
        if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
    }

    void fail(std::exception_ptr error) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) error_ = std::move(error);
        failed_.store(true, std::memory_order_relaxed);
    }

    Crew crew_;  // constructed before the threads start and destroyed after they end
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::exception_ptr error_;
    std::atomic<bool> failed_{false};
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

// ------------------------------------------------------------------ plans

// A pixel box with inclusive bounds, as in hclean.cpp.
struct Box {
    int i0 = 0, j0 = 0, i1 = -1, j1 = -1;
};

// The split of a plane into the pixels the passes keep current, S (the
// segments of the searched rows), and the others, E.
struct Plan {
    Box box;            // the clean box shrunk to the mask's bounding box (clean<T>'s)
    long searched = 0;  // the searched pixels: the clean box inside the mask
    long work = 0;      // the pixels of the segments, a pass's work
    std::vector<SearchRow> rows;
    std::vector<Run> runs;      // the segments of S
    std::vector<Run> exterior;  // the runs of E: those of row y are
    std::vector<int> exterior_start;  // exterior[exterior_start[y], exterior_start[y + 1])
};

// Eight bool bytes all false, all true.
constexpr std::uint64_t kNoneSet = 0;
constexpr std::uint64_t kAllSet = 0x0101010101010101ULL;

// Appends the runs of the mask bytes in [x0, x1) that are not 0 (true, as the
// reference tests them) to runs.
void mask_runs(const std::uint8_t* mask, int x0, int x1, std::vector<Run>& runs) {
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

// The plan of a plane with the binding's clean box [xbeg, xend) x [ybeg, yend).
// The searched pixels are the box inside the mask; clean<T>'s box
// (mask_bounding_box of hclean.cpp) is their bounding box, or the clean box
// itself when there are none. `runs` holds the mask runs of one row.
void build_plan(Plan& plan, std::vector<Run>& runs, const std::uint8_t* mask, int nx, int ny,
                int xbeg, int xend, int ybeg, int yend) {
    Box box{xend, yend, -1, -1};
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
        box.i0 = std::min(box.i0, runs.front().begin);
        box.i1 = std::max(box.i1, runs.back().end - 1);
        box.j0 = std::min(box.j0, y);
        box.j1 = y;
    }
    plan.box = plan.rows.empty() ? Box{xbeg, ybeg, xend - 1, yend - 1} : box;
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
void split_rows(const Plan& plan, int parts, std::vector<int>& split) {
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

template <typename T>
struct Plane {
    T* residual = nullptr;
    T* model = nullptr;
    const T* psf = nullptr;
    const std::uint8_t* mask = nullptr;
    int max_iter = 0;
    T threshold = 0;
    IterationHistory* history = nullptr;
    Plan plan;  // planes with a budget only
    std::vector<Component<T>> components;
    size_t flushed = 0;  // components already subtracted from E
    std::vector<Extrema<T>> tiles;  // of the final pass
    T total = 0;
};

// The buffers of a crew's passes.
template <typename T>
struct Rows {
    std::vector<double> sum;
    std::vector<T> max;
    std::vector<int> ix;
    std::vector<int> split;
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

template <typename T>
size_t plane_pixels(const Cube<T>& c) {
    return static_cast<size_t>(c.ny) * static_cast<size_t>(c.nx);
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
ExteriorFlush<T> pending(const Cube<T>& c, Plane<T>& p) {
    return {p.residual, c.nx, p.psf, c.psf_nx, p.plan.exterior.data(),
            p.plan.exterior_start.data(), p.components.data() + p.flushed,
            static_cast<int>(p.components.size() - p.flushed)};
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

// Applies the plane's pending subtractions to E, with its crew.
template <typename T>
void flush_exterior(const Cube<T>& c, Plane<T>& p, Crew& crew, const Kernels& kernels) {
    const ExteriorFlush<T> exterior = pending(c, p);
    const int rows = tile_rows(c), n_tiles = tiles(c);
    std::atomic<int> next{0};
    auto step = [&](int) {
        for (int t; (t = next.fetch_add(1, std::memory_order_relaxed)) < n_tiles;) {
            flush(kernels, exterior, t * rows, std::min(c.ny, (t + 1) * rows));
        }
    };
    crew.run(step);
    p.flushed = p.components.size();
}

// The binding's total of a plane: |model[i]| summed in T, one sequential sum
// in index order. A model is mostly zeros, and skipping its +0 terms is exact
// (the sum is >= +0 or NaN): eight values at a time are tested for +-0 at
// once, by their bits without the sign.
template <typename T>
T sum_of_magnitudes(const T* model, size_t n) {
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

// Item i of the final pass over a plane: i = 0 sums |model| (the binding's
// total), and i = 1 + t applies the pending subtractions to the rows of tile
// t and takes maximg's extrema over them.
template <typename T>
void finish_item(const Cube<T>& c, Plane<T>& p, int item, const Kernels& kernels) {
    if (item == 0) {
        p.total = sum_of_magnitudes(p.model, plane_pixels(c));
        return;
    }
    const int tile = item - 1, rows = tile_rows(c);
    const int y0 = tile * rows, y1 = std::min(c.ny, y0 + rows);
    if (p.flushed < p.components.size()) flush(kernels, pending(c, p), y0, y1);
    Extrema<T> e = no_extrema<T>();
    const size_t begin = static_cast<size_t>(y0) * c.nx, end = static_cast<size_t>(y1) * c.nx;
    for (size_t i = begin; i < end; ++i) {
        if (p.mask == nullptr || p.mask[i] != 0) {
            const T w = p.residual[i];
            if (w > e.high) e.high = w;
            if (w < e.low) e.low = w;
        }
    }
    p.tiles[tile] = e;
}

// The binding's summary of plane `index` once its final pass is done: maximg's
// extrema (the tiles merged in order, as one scan finds them), final_peak,
// converged and total_flux.
template <typename T>
void summarize(const Cube<T>& c, int index, Plane<T>& p) {
    p.flushed = p.components.size();
    Extrema<T> e = no_extrema<T>();
    for (const Extrema<T>& tile : p.tiles) {
        if (tile.high > e.high) e.high = tile.high;
        if (tile.low < e.low) e.low = tile.low;
    }
    const T peak = std::max(std::abs(e.low), std::abs(e.high));
    c.final_peak[index] = peak;
    c.total_flux[index] = p.total;
    c.converged[index] = peak <= p.threshold;
}

// The final pass over one plane, with its crew.
template <typename T>
void finish(const Cube<T>& c, int index, Plane<T>& p, Crew& crew, const Kernels& kernels) {
    const int items = 1 + tiles(c);
    std::atomic<int> next{0};
    auto step = [&](int) {
        for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < items;) {
            finish_item(c, p, i, kernels);
        }
    };
    crew.run(step);
    summarize(c, index, p);
}

// The model update of plane `index` (a budget > 0), with its crew: clean<T>
// for Entry::clean_cube, the per-plane part of clean_cube_many_threads for
// Entry::clean_cube_many_threads, with the searched pixels kept current by
// the fused pass and the subtractions from E deferred.
template <typename T>
void iterate(const Cube<T>& c, int index, Plane<T>& p, Crew& crew, Rows<T>& rows,
             const Kernels& kernels) {
    const Plan& plan = p.plan;
    const int n_rows = static_cast<int>(plan.rows.size());
    split_rows(plan, crew.size(), rows.split);
    SearchPass<T> pass{p.residual,        c.nx,           p.psf,           c.psf_nx,
                       p.mask,            plan.rows.data(), plan.runs.data(), nullptr,
                       rows.sum.data(),   rows.max.data(), rows.ix.data(),  plan.box.i0};
    auto step = [&](int rank) { search(kernels, pass, rows.split[rank], rows.split[rank + 1]); };
    const auto scan = [&](const Component<T>* subtracted) {
        pass.component = subtracted;
        crew.run(step);
    };

    IterationHistory* const history = p.history;
    const double n_searched = static_cast<double>(plan.searched);
    const T zero = static_cast<T>(0), two = static_cast<T>(2);
    PlaneStopTest<T> stop_test;
    int stop = STOP_NONE;
    int px = 0, py = 0;  // clean<T> keeps the peak's pixel between iterations: (0, 0) at first
    scan(nullptr);
    int iter = 0;
    for (; iter < p.max_iter; ++iter) {
        // The plane's sum of squares and peak, the rows taken in row order as
        // in both reference kernels: the first row with the largest |value|.
        double sum_squares = 0.0;
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
        // The pass summed the squares of the residual the previous iteration
        // left: its RMS.
        if (history != nullptr && iter > 0) history->rms.push_back(std::sqrt(sum_squares / n_searched));
        const T threshold =
            c.speedup > zero ? p.threshold * std::pow(two, static_cast<T>(iter) / c.speedup)
                             : p.threshold;
        const int reason = stop_test.check(peak, sum_squares, threshold, c.divergence);
        if (reason != STOP_NONE) {
            stop = reason;
            break;
        }
        // The peak's pixel is searched, so current, unless the peak is zero:
        // then it may lie in E, whose pending subtractions are applied first.
        if (!(peak > zero) && !in_search(plan, py, px)) flush_exterior(c, p, crew, kernels);
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
        for (int i = 0; i < n_rows; ++i) sum_squares += rows.sum[i];
        history->rms.push_back(std::sqrt(sum_squares / n_searched));
    }
    c.iterations[index] = iter;
    c.stop_code[index] = stop;
}

// How the planes share a team of `team` threads, given the planes with a
// budget (`active`) and the largest work of a pass among them (pixels).
struct Schedule {
    int crews = 1;      // crews that iterate planes side by side
    int crew_size = 1;  // threads per crew
    bool deferred = false;  // the planes' final passes wait for the whole team
};

Schedule schedule(int team, int active, long largest_work) {
    Schedule s;
    if (active == 0) {
        s.deferred = true;
        return s;
    }
    // A crew of more threads than its plane's pass justifies would
    // spend more time synchronising than it saves; side by side, crews share
    // no step.
    const long justified = std::clamp(largest_work / kPixelsPerThread, 1L, static_cast<long>(team));
    const int useful = static_cast<int>(justified);
    s.crews = std::clamp(team / useful, 1, active);
    // One crew for several planes (it runs them one after another): every
    // thread, as the planes justify more than half of them.
    s.crew_size = (s.crews == 1 && active > 1) ? team : std::min(useful, team / s.crews);
    // When every plane has a crew of its own but the crews leave threads idle,
    // the whole team does the final passes, after the iterations (the memory
    // of no more than `team` planes' components is kept until then).
    s.deferred = s.crews == active && s.crews * s.crew_size < team;
    return s;
}

// The threads of a call: processing_function_threads (< 1 counts as 1, as in
// both reference entries), at most the CPUs the process may run on, the
// threads its work can use (`usable`) and those it justifies: one per
// kUpdatesPerThread of `updates`, the pixel updates of the call at most.
int team_size(int requested, long usable, double updates) {
    long threads = std::min<long>(std::max(requested, 1), available_cpus());
    threads = std::min(threads, usable);
    const double justified = updates / kUpdatesPerThread;
    if (justified < static_cast<double>(threads)) threads = static_cast<long>(justified);
    return static_cast<int>(std::max(threads, 1L));
}

// Runs work(member) on `threads` threads side by side, the calling thread
// being member 0 and `workers` (with room reserved) holding the others, as
// the reference clean_cube runs its threads: each starts on its work at once,
// and none waits for another. A thread that cannot be started leaves fewer
// members. The first exception that a member throws is rethrown once all
// have ended, and sets `failed`, which the others read to stop early.
template <typename Work>
void run_side_by_side(std::vector<std::thread>& workers, int threads, std::atomic<bool>& failed,
                      Work& work) {
    std::mutex mutex;
    std::exception_ptr error;
    const auto guarded = [&](int member) {
        try {
            work(member);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(mutex);
            if (!error) error = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    for (int member = 1; member < threads; ++member) {
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
    if (error) std::rethrow_exception(error);
}

}  // namespace

template <typename T>
bool clean_cube(const Cube<T>& c) {
    const int nplanes = c.nt * c.nf * c.np_img;
    if (nplanes <= 0) return true;
    const size_t plane_size = plane_pixels(c);
    const size_t psf_plane_size = static_cast<size_t>(c.psf_ny) * static_cast<size_t>(c.psf_nx);
    const Kernels& kernels = hclean_fast::kernels();

    // Everything that can fail to allocate, before the first write.
    std::vector<Plane<T>> planes;
    int threads = 1;
    bool side_by_side = false;
    std::vector<std::thread> workers;          // side by side: the threads after the caller
    std::vector<std::vector<Run>> mask_rows;   // side by side: each thread's mask runs of a row
    std::unique_ptr<Team> team;
    Schedule s;
    std::unique_ptr<Crew[]> crews;
    std::vector<Rows<T>> rows;  // a crew's, or (side by side) a thread's
    try {
        planes.resize(static_cast<size_t>(nplanes));
        int active = 0;
        // The work the call can use and justifies, at most: a pass searches
        // at most the clean box, and the final pass reads every pixel and
        // subtracts each component at most once from each pixel of E, which
        // is the plane outside the clean box (no mask) or at most the plane.
        const long box = static_cast<long>(c.xend - c.xbeg) * (c.yend - c.ybeg);
        const double exterior = c.mask != nullptr ? static_cast<double>(plane_size)
                                                  : static_cast<double>(plane_size - box);
        double pass_updates = 0.0, final_updates = 0.0;
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
            p.tiles.resize(static_cast<size_t>(tiles(c)));
            final_updates += static_cast<double>(plane_size);
            if (p.max_iter > 0) {
                ++active;
                p.components.reserve(static_cast<size_t>(std::min(p.max_iter, 256)));
                pass_updates += static_cast<double>(box) * p.max_iter;
                final_updates += exterior * p.max_iter;
            }
        }
        // A crew per plane with a budget, of up to one thread per
        // kPixelsPerThread pixels of its pass, or one thread per tile of the
        // final passes, if their work justifies that many (both bounded by
        // the request, which bounds the team anyway).
        const long requested = std::max(c.threads, 1);
        const long crew_threads = std::clamp(box / kPixelsPerThread, 1L, requested);
        const long final_threads = std::min(
            static_cast<long>(nplanes) * tiles(c),
            static_cast<long>(std::min(final_updates / kUpdatesPerThread, static_cast<double>(requested))));
        const long usable = std::max(active * crew_threads, final_threads);
        threads = team_size(c.threads, usable, pass_updates + final_updates);
        // Planes too small for a crew of two (a pass of fewer than 2 x
        // kPixelsPerThread pixels) and no more threads than planes with a
        // budget: schedule() would give every thread planes of its own and
        // defer nothing, so the threads run side by side from their start,
        // each building the plans of the planes it cleans (in memory reserved
        // here). Starting a thread costs about 25-35 us (M), and the phases
        // of a team wait for all of its threads.
        side_by_side = box < 2 * kPixelsPerThread && threads <= active;
        if (side_by_side) {
            const int w = c.xend - c.xbeg, h = std::max(c.yend - c.ybeg, 0);
            for (Plane<T>& p : planes) {
                if (p.max_iter > 0) reserve_plan(p.plan, w, h, c.ny);
            }
            mask_rows.resize(static_cast<size_t>(threads));
            rows.resize(static_cast<size_t>(threads));
            for (int member = 0; member < threads; ++member) {
                mask_rows[member].reserve(static_cast<size_t>(std::max(w, 0)) / 2 + 1);
                rows[member].sum.resize(static_cast<size_t>(h));
                rows[member].max.resize(static_cast<size_t>(h));
                rows[member].ix.resize(static_cast<size_t>(h));
                rows[member].split.reserve(2);
            }
            workers.reserve(static_cast<size_t>(threads - 1));
        } else {
            team = std::make_unique<Team>(threads);
            std::atomic<int> next{0};
            team->run([&](int) {
                std::vector<Run> runs;  // of the mask, in one row
                for (int pl; (pl = next.fetch_add(1, std::memory_order_relaxed)) < nplanes;) {
                    Plane<T>& p = planes[pl];
                    if (p.max_iter > 0) {
                        build_plan(p.plan, runs, p.mask, c.nx, c.ny, c.xbeg, c.xend, c.ybeg, c.yend);
                    }
                }
            });
            long largest = 0;
            size_t most_rows = 0;
            for (const Plane<T>& p : planes) {
                largest = std::max(largest, p.plan.work);
                most_rows = std::max(most_rows, p.plan.rows.size());
            }
            s = schedule(team->size(), active, largest);
            crews.reset(new Crew[static_cast<size_t>(s.crews)]);
            rows.resize(static_cast<size_t>(s.crews));
            for (int g = 0; g < s.crews; ++g) {
                crews[g].resize(s.crew_size);
                rows[g].sum.resize(most_rows);
                rows[g].max.resize(most_rows);
                rows[g].ix.resize(most_rows);
                rows[g].split.reserve(static_cast<size_t>(s.crew_size) + 1);
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
        std::atomic<bool> failed{false};
        auto clean_planes = [&](int member) {
            Crew alone;  // of one thread: its steps run on this one
            for (int pl; !failed.load(std::memory_order_relaxed) &&
                         (pl = next.fetch_add(1, std::memory_order_relaxed)) < nplanes;) {
                Plane<T>& p = planes[pl];
                if (p.max_iter > 0) {
                    build_plan(p.plan, mask_rows[member], p.mask, c.nx, c.ny, c.xbeg, c.xend, c.ybeg,
                               c.yend);
                    iterate(c, pl, p, alone, rows[member], kernels);
                }
                finish(c, pl, p, alone, kernels);
            }
        };
        run_side_by_side(workers, threads, failed, clean_planes);
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
            if (!s.deferred) finish(c, pl, p, crew, kernels);
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
        for (int pl = 0; pl < nplanes; ++pl) summarize(c, pl, planes[pl]);
    }
    return true;
}

const Kernels& kernels() {
    static const Kernels selected = [] {
        const bool avx2 = astroviper::fast::select_isa() == astroviper::fast::Isa::avx2;
#if defined(AV_HAVE_AVX2)
        if (avx2) return avx2_kernels();
#endif
        static_cast<void>(avx2);  // no AVX2 routines in this build
        return generic_kernels();
    }();
    return selected;
}

template bool clean_cube<float>(const Cube<float>&);
template bool clean_cube<double>(const Cube<double>&);

}  // namespace hclean_fast
