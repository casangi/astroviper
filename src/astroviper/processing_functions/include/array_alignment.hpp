#pragma once

#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace astroviper {
// No allocation or conversion. Check both the base and all used strides;
// empty arrays have no accessed elements. This also covers strided residuals.
inline void require_aligned(const pybind11::array& array, const char* name) {
    if (array.size() == 0) return;
    const auto alignment = pybind11::cast<std::size_t>(array.dtype().attr("alignment"));
    if (alignment <= 1) return;
    bool aligned = reinterpret_cast<std::uintptr_t>(array.data()) % alignment == 0;
    for (pybind11::ssize_t axis = 0; axis < array.ndim(); ++axis)
        if (array.shape(axis) > 1 && array.strides(axis) % static_cast<pybind11::ssize_t>(alignment))
            aligned = false;
    if (!aligned)
        throw std::invalid_argument(std::string(name) +
            " must be naturally aligned for its dtype (no copy is made)");
}
}  // namespace astroviper
