#include "scalar_buffer.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

template <typename T, bool Aligned, typename Rhs>
void check_add(T initial, Rhs rhs, std::size_t offset, bool early_narrowing_differs) {
    alignas(T) std::array<unsigned char, 8 * sizeof(T)> storage;
    storage.fill(0xA5);
    auto* data = storage.data() + sizeof(T) + offset;
    // Exercise a nonzero buffer index, with guard bytes on both sides.
    auto* target = data + sizeof(T);
    std::memcpy(target, &initial, sizeof(T));
    const auto before = storage;
    astroviper::ScalarBuffer<T, Aligned> buffer(data);

    T expected = initial;
    expected += rhs;  // Independent oracle: the built-in compound assignment.
    if (early_narrowing_differs) {
        T narrowed = initial;
        narrowed += static_cast<T>(rhs);
        if (narrowed == expected)
            throw std::runtime_error("Regression values do not distinguish early narrowing");
    }
    auto reference = buffer[1];
    auto& result = (reference += rhs);
    T actual;
    std::memcpy(&actual, target, sizeof(T));
    if (actual != expected || buffer.load(1) != expected)
        throw std::runtime_error("Proxy addition differs from built-in +=");
    if (&result != &reference)
        throw std::runtime_error("operator+= did not return its own reference");

    // Returned references must also support subsequent compound assignment.
    (result += rhs) += rhs;
    expected += rhs;
    expected += rhs;
    std::memcpy(&actual, target, sizeof(T));
    if (actual != expected)
        throw std::runtime_error("Chained proxy addition differs from built-in +=");
    const auto target_offset = static_cast<std::size_t>(target - storage.data());
    for (std::size_t i = 0; i < storage.size(); ++i)
        if ((i < target_offset || i >= target_offset + sizeof(T)) && storage[i] != before[i])
            throw std::runtime_error("Proxy addition changed a guard byte");
}

template <typename T, typename Rhs>
void check_both_paths(T initial, Rhs rhs, bool early_narrowing_differs = false) {
    check_add<T, true>(initial, rhs, 0, early_narrowing_differs);
    // Include aligned addresses on the byte-safe path, and every unaligned offset.
    for (std::size_t offset = 0; offset < alignof(T); ++offset)
        check_add<T, false>(initial, rhs, offset, early_narrowing_differs);
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Expected one scenario name");
        const std::string scenario = argv[1];
        if (scenario == "cancellation") {
            check_both_paths<float>(16777216.0f, -16777215.5, true);
        } else if (scenario == "rounding") {
            check_both_paths<float>(1.0f, std::ldexp(1.0, -24) + std::ldexp(1.0, -50), true);
        } else if (scenario == "fractional_integer") {
            check_both_paths<std::int32_t>(1, -0.75, true);
        } else if (scenario == "same_type") {
            check_both_paths<float>(1.25f, 0.5f);
            check_both_paths<double>(1.25, 0.5);
            check_both_paths<std::int32_t>(1, -2);
        } else {
            throw std::runtime_error("Unknown scenario");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
