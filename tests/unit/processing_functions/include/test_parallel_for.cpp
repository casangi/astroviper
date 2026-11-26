// parallel_for.hpp, the parallel loops of the C++ kernels: thread_count's
// bounds, run_threads' workers (worker 0 on the calling thread, each other
// worker on a thread of its own) and parallel_for's chunks (every item once,
// the documented bounds, chunk t on worker t, nothing for no items).

#include "parallel_for.hpp"

#include <atomic>
#include <climits>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_thread_count() {
    using astroviper::thread_count;
    const std::tuple<int, std::int64_t, int> cases[] = {
        {1, 10, 1},  {4, 10, 4},       {4, 4, 4},         {4, 3, 3},  {0, 10, 1},
        {-5, 10, 1}, {4, 0, 1},        {4, -2, 1},        {0, 0, 1},  {INT_MAX, 7, 7},
        {8, std::int64_t{1} << 40, 8}, {INT_MAX, std::int64_t{1} << 40, INT_MAX},
    };
    for (const auto& [threads, n_items, expected] : cases)
        require(thread_count(threads, n_items) == expected,
                "thread_count(" + std::to_string(threads) + ", " + std::to_string(n_items) +
                    ") != " + std::to_string(expected));
}

void check_run_threads(int n_threads) {
    const std::thread::id caller = std::this_thread::get_id();
    std::mutex mutex;
    std::vector<int> calls(static_cast<std::size_t>(std::max(n_threads, 1)), 0);
    std::set<std::thread::id> others;
    astroviper::run_threads(n_threads, [&](int t) {
        std::lock_guard<std::mutex> lock(mutex);
        ++calls.at(static_cast<std::size_t>(t));
        const bool on_caller = std::this_thread::get_id() == caller;
        require(on_caller == (t == 0), "worker " + std::to_string(t) + " on the wrong thread");
        if (!on_caller) others.insert(std::this_thread::get_id());
    });
    for (int t = 0; t < static_cast<int>(calls.size()); ++t)
        require(calls[static_cast<std::size_t>(t)] == 1,
                "worker " + std::to_string(t) + " not called once");
    require(static_cast<int>(others.size()) == static_cast<int>(calls.size()) - 1,
            "workers 1 .. n_threads - 1 must run on threads of their own");
}

void check_parallel_for(int threads, std::int64_t n_items) {
    const std::thread::id caller = std::this_thread::get_id();
    std::mutex mutex;
    std::vector<std::tuple<std::int64_t, std::int64_t, std::thread::id>> chunks;
    std::vector<std::atomic<int>> visits(static_cast<std::size_t>(std::max<std::int64_t>(n_items, 0)));
    astroviper::parallel_for(threads, n_items, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) ++visits.at(static_cast<std::size_t>(i));
        std::lock_guard<std::mutex> lock(mutex);
        chunks.emplace_back(begin, end, std::this_thread::get_id());
    });
    const std::string name = "parallel_for(" + std::to_string(threads) + ", " + std::to_string(n_items) + ")";
    if (n_items <= 0) {
        require(chunks.empty(), name + " must not run its body");
        return;
    }
    for (const auto& visit : visits) require(visit == 1, name + ": an item not visited once");
    const int n_threads = astroviper::thread_count(threads, n_items);
    const std::int64_t chunk = (n_items + n_threads - 1) / n_threads;
    std::set<std::tuple<std::int64_t, std::int64_t>> expected, found;
    for (std::int64_t begin = 0; begin < n_items; begin += chunk)
        expected.emplace(begin, std::min(n_items, begin + chunk));
    std::set<std::thread::id> thread_ids;
    for (const auto& [begin, end, id] : chunks) {
        found.emplace(begin, end);
        thread_ids.insert(id);
        require((id == caller) == (begin == 0), name + ": only chunk 0 runs on the calling thread");
    }
    require(found == expected && chunks.size() == expected.size(),
            name + ": chunks are not [t c, min(n, (t + 1) c)) with c = ceil(n / threads)");
    require(thread_ids.size() == chunks.size(), name + ": two chunks on one thread");
    require(static_cast<int>(chunks.size()) <= n_threads, name + ": more chunks than threads");
}

}  // namespace

int main() {
    try {
        check_thread_count();
        for (int n_threads : {-1, 0, 1, 2, 3, 8}) check_run_threads(n_threads);
        for (int threads : {-3, 0, 1, 2, 3, 4, 7, 64})
            for (std::int64_t n_items : {-1, 0, 1, 2, 5, 7, 8, 9, 100, 1001})
                check_parallel_for(threads, n_items);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
