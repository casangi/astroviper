#pragma once

// The parallel loops of the C++ kernels. A kernel runs at most
// processing_function_threads threads, never more threads than work items,
// and one thread for a count below 1.

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

namespace astroviper {

// processing_function_threads, at least 1 and at most max(n_items, 1).
inline int thread_count(int processing_function_threads, std::int64_t n_items) {
    return static_cast<int>(std::min<std::int64_t>(std::max(1, processing_function_threads),
                                                    std::max<std::int64_t>(n_items, 1)));
}

// worker(t) for t in [0, max(n_threads, 1)): t = 0 on the calling thread,
// the others on new threads, all joined before it returns.
template <typename Worker>
void run_threads(int n_threads, Worker worker) {
    if (n_threads <= 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    for (int t = 1; t < n_threads; ++t) threads.emplace_back(worker, t);
    worker(0);
    for (auto& th : threads) th.join();
}

// body(begin, end) on the chunks [t c, min(n_items, (t + 1) c)) of
// c = ceil(n_items / thread_count(processing_function_threads, n_items))
// items that are not empty, chunk t on thread t of run_threads. Nothing runs
// for n_items <= 0.
template <typename Body>
void parallel_for(int processing_function_threads, std::int64_t n_items, Body body) {
    if (n_items <= 0) return;
    const int n_threads = thread_count(processing_function_threads, n_items);
    const std::int64_t chunk = (n_items + n_threads - 1) / n_threads;
    const int n_chunks = static_cast<int>((n_items + chunk - 1) / chunk);
    run_threads(n_chunks, [&](int t) {
        const std::int64_t begin = t * chunk;
        body(begin, std::min(n_items, begin + chunk));
    });
}

}  // namespace astroviper
