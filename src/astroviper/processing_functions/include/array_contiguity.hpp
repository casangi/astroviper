#pragma once

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

namespace astroviper {
// Raise a clear error if an array is not C-contiguous: its strides must be
// those of a C-contiguous layout. pybind11 raises its own error for dtype
// mismatches when forcecast is absent, but a contiguity message helps the
// caller fix the issue quickly.
inline void require_c_contiguous(const pybind11::buffer_info& info, const char* name) {
    pybind11::ssize_t stride = info.itemsize;
    for (int i = info.ndim - 1; i >= 0; --i) {
        if (info.strides[i] != stride)
            throw std::runtime_error(
                std::string(name) + " must be C-contiguous (pass np.ascontiguousarray() if needed)");
        stride *= info.shape[i];
    }
}
}  // namespace astroviper
