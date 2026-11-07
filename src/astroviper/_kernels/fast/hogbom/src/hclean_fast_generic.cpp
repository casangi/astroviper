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
    static V zero() { return static_cast<T>(0); }
    static V magnitude(V a) {
        if constexpr (sizeof(T) == sizeof(double)) {
            return __builtin_fabs(a);
        } else {
            return __builtin_fabsf(a);
        }
    }
    static bool greater(V a, V b) { return a > b; }  // false for NaN
    static V select(V a, V b, bool take) { return take ? b : a; }
    // Whether the mask byte is not 0, and a value cleared to +0.0 where not.
    static bool keep(const std::uint8_t* mask) { return *mask != 0; }
    static V clear(V v, bool keep) { return keep ? v : static_cast<T>(0); }
    // A sum of squares in double (of a row of the column pass).
    using Sums = double;
    static Sums no_sums() { return 0.0; }
    static void add_squares(Sums& sums, V v) { sums += static_cast<double>(v) * static_cast<double>(v); }
    static void add_squares(Sums& sums, V v, bool keep) {
        const double square = static_cast<double>(v) * static_cast<double>(v);
        sums += keep ? square : 0.0;
    }
    static void store_sums(double* out, Sums sums) { *out = sums; }
};

#include "hclean_fast_kernels.inc"

}  // namespace

hclean_fast::Kernels hclean_fast::generic_kernels() {
    return {"generic",  search_f64, search_f32,  flush_f64,  flush_f32,
            direct_f64, direct_f32, columns_f64, columns_f32};
}
