// Compile probe of the fast Hogbom module (cmake/AstroviperKernels.cmake): the
// language and library features its sources use, among them std::atomic wait
// and notify (C++20; macOS 11 or later) and noexcept function types, and the
// floating-point semantics they need (strict_fp.hpp, which they include). The
// only project header it includes is strict_fp.hpp, whose text is part of its
// cached verdict's key, so that the verdict depends on the two texts and the
// toolchain only; the reference header the module includes is compiled by
// the reference module of the same build.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "strict_fp.hpp"

int main() {
    std::atomic<std::uint32_t> generation{0};
    std::exception_ptr error;
    auto step = [&]() noexcept {
        generation.fetch_add(1, std::memory_order_seq_cst);
        generation.notify_all();
    };
    static_assert(std::is_nothrow_invocable_v<decltype(step)&>);
    void (*call)(void*) noexcept = [](void* f) noexcept { (*static_cast<decltype(step)*>(f))(); };
    std::thread worker([&] {
        try {
            call(&step);
        } catch (...) {
            error = std::current_exception();
        }
    });
    generation.wait(0, std::memory_order_seq_cst);
    worker.join();
    const auto start = std::chrono::steady_clock::now();
    std::vector<int> values{3, 1, 2};
    std::sort(values.begin(), values.end());
    const auto copy = std::make_unique<double[]>(2);
    copy[0] = __builtin_fabs(-1.0);
    return generation.load() == 1 && !error && std::chrono::steady_clock::now() >= start &&
                   values.front() == 1 && copy[0] == 1.0
               ? 0
               : 1;
}
