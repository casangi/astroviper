// Compile probe of the fast dft_predict module (cmake/AstroviperKernels.cmake):
// the language and library features its sources use, and the floating-point
// semantics they need (strict_fp.hpp, which they include).
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <new>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "strict_fp.hpp"

int main() {
    std::atomic<std::int64_t> next{0};
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            next.fetch_add(1, std::memory_order_relaxed);
        } catch (...) {
            error = std::current_exception();
        }
    });
    worker.join();
    auto* aligned = static_cast<double*>(::operator new[](64, std::align_val_t{64}));
    ::operator delete[](aligned, std::align_val_t{64});
    std::unordered_map<std::uint64_t, std::int64_t> bits{{std::bit_cast<std::uint64_t>(-0.5), 1}};
    const std::optional<std::complex<float>> value = std::complex<float>(1.0f, 2.0f);
    return next.load() == 1 && !error && bits.size() == 1 && std::isfinite(value->imag()) ? 0 : 1;
}
