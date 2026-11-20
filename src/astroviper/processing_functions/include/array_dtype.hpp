#pragma once

#include <pybind11/numpy.h>
#include <bit>
#include <stdexcept>
#include <string>

namespace astroviper {
// NumPy's dtype.name omits byte order. Include dtype.str and the effective
// order so swapped float64 is distinguishable from native float64. Call only
// with the GIL held; this reads metadata and never converts array storage.
inline std::string dtype_error(const pybind11::array& array, const char* name,
                               const char* expected) {
    const auto dtype = array.dtype();
    const bool native = pybind11::cast<bool>(dtype.attr("isnative"));
    const auto order = pybind11::cast<std::string>(dtype.attr("byteorder"));
    const char* native_order = std::endian::native == std::endian::little
        ? "little-endian" : "big-endian";
    const char* actual_order = order == ">" ? "big-endian" : order == "<" ? "little-endian"
        : order == "|" ? "byte-order independent" : native_order;
    return std::string(name) + (native ? " has wrong dtype; " : " has non-native byte order; ") +
        "expected native " + expected + " (" + native_order + "), got " +
        pybind11::cast<std::string>(dtype.attr("name")) + " (" +
        pybind11::cast<std::string>(dtype.attr("str")) + ", " + actual_order +
        "); no implicit conversion is performed";
}

template <typename T>
inline void require_dtype(const pybind11::array& array, const char* name) {
    const auto expected = pybind11::dtype::of<T>();
    if (!array.dtype().equal(expected)) {
        const auto expected_name = pybind11::cast<std::string>(expected.attr("name"));
        throw std::runtime_error(dtype_error(array, name, expected_name.c_str()));
    }
}
}  // namespace astroviper
