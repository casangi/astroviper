// Compile probe of the fast Hogbom module (cmake/AstroviperKernels.cmake): the
// language and library features its sources use, among them the reference's
// header and std::atomic wait and notify (C++20; macOS 11 or later).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "hclean.hpp"

int main() {
    std::atomic<std::uint32_t> generation{0};
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            generation.fetch_add(1, std::memory_order_seq_cst);
            generation.notify_all();
        } catch (...) {
            error = std::current_exception();
        }
    });
    generation.wait(0, std::memory_order_seq_cst);
    worker.join();
    hclean::IterationHistory history;
    history.add_component(1.0, 2, 3, 0.1);
    hclean::PlaneStopTest<float> stop_test;
    const int reason = stop_test.check(1.0f, 1.0, 0.5f, hclean::DivergenceControl<float>{});
    const auto start = std::chrono::steady_clock::now();
    std::vector<int> values{3, 1, 2};
    std::sort(values.begin(), values.end());
    return generation.load() == 1 && !error && reason == hclean::STOP_NONE &&
                   history.peak.size() == 1 && std::chrono::steady_clock::now() >= start &&
                   values.front() == 1
               ? 0
               : 1;
}
