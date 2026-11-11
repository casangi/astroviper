#pragma once

// The instruction set of a fast module's routines, shared by the fast modules
// (cmake/AstroviperKernels.cmake puts this directory on their include path).
// For generic units only: an *_avx2.cpp unit must not include it, as its
// inline function would be a weak symbol of that unit.

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace astroviper::fast {

enum class Isa { generic, avx2 };

// AVX2 when the module has AVX2 routines (it is built with AV_HAVE_AVX2, on
// x86-64 only) and the CPU supports AVX2, unless ASTROVIPER_KERNELS_ISA is
// "generic" (the baseline routines, which give the same bits; for tests).
// Throws std::invalid_argument for any other value of ASTROVIPER_KERNELS_ISA.
// A module calls it once, at its import, and keeps the result.
inline Isa select_isa() {
    const char* value = std::getenv("ASTROVIPER_KERNELS_ISA");
    const std::string_view isa = value != nullptr ? value : "";
    if (!isa.empty() && isa != "generic") {
        throw std::invalid_argument("ASTROVIPER_KERNELS_ISA=" + std::string(isa) +
                                    ": expected 'generic' (the baseline instruction set) "
                                    "or unset (the best one of the CPU)");
    }
#if defined(AV_HAVE_AVX2)
    if (isa.empty() && __builtin_cpu_supports("avx2")) return Isa::avx2;
#endif
    return Isa::generic;
}

}  // namespace astroviper::fast
