// The fast Hogbom routines for the baseline instruction set
// (hclean_fast_kernels.hpp): portable C++ with one lane, which the compiler
// may vectorise where that keeps the operations of each pixel. It is the only
// build on platforms other than x86-64 and the ASTROVIPER_KERNELS_ISA=generic
// override, and gives the AVX2 routines' bits (NaN payloads aside).

#include <cstddef>
#include <cstdint>

#include "hclean_fast_kernels.hpp"

namespace {

template <typename T>
struct Simd {
    using V = T;
    static constexpr int W = 1;
    static V load(const T* p) { return *p; }
    static void store(T* p, V v) { *p = v; }
    static V splat(T x) { return x; }
    static V mul(V a, V b) { return a * b; }
    static V sub(V a, V b) { return a - b; }
};

#include "hclean_fast_kernels.inc"

}  // namespace

hclean_fast::Kernels hclean_fast::generic_kernels() {
    return {"generic", search_f64, search_f32, flush_f64, flush_f32};
}
