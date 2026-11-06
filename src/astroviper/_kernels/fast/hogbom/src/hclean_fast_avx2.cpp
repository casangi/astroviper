// The fast Hogbom routines for AVX2 (hclean_fast_kernels.hpp), compiled with
// -mavx2 and without LTO, and used only when the CPU has AVX2. -mavx2 does
// not enable FMA, so every subtraction stays a rounded product and a rounded
// difference, as in the generic build. Everything is internal to this unit
// except the table it exports.

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hclean_fast_kernels.hpp"

namespace {

template <typename T>
struct Simd;

template <>
struct Simd<double> {
    using V = __m256d;
    static constexpr int W = 4;
    static V load(const double* p) { return _mm256_loadu_pd(p); }
    static void store(double* p, V v) { _mm256_storeu_pd(p, v); }
    static V splat(double x) { return _mm256_set1_pd(x); }
    static V zero() { return _mm256_setzero_pd(); }
    static V iota(int start) {
        const double s = start;
        return _mm256_setr_pd(s, s + 1, s + 2, s + 3);
    }
    static V mul(V a, V b) { return _mm256_mul_pd(a, b); }
    static V sub(V a, V b) { return _mm256_sub_pd(a, b); }
    static V add(V a, V b) { return _mm256_add_pd(a, b); }
    static V magnitude(V a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a); }
    static V greater(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }  // false for NaN
    static V select(V a, V b, V mask) { return _mm256_blendv_pd(a, b, mask); }
    static void store_squares(double* out, V v) { _mm256_storeu_pd(out, _mm256_mul_pd(v, v)); }
    // The lanes whose mask byte is not 0 (all ones), and the values or
    // squares of the other lanes cleared to +0.0.
    static V keep(const std::uint8_t* mask) {
        std::int32_t bytes;
        std::memcpy(&bytes, mask, sizeof(bytes));
        const __m256i lanes = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(bytes));
        return _mm256_castsi256_pd(_mm256_cmpgt_epi64(lanes, _mm256_setzero_si256()));
    }
    static V clear(V v, V keep) { return _mm256_and_pd(v, keep); }
    static void store_squares(double* out, V v, V keep) {
        _mm256_storeu_pd(out, _mm256_and_pd(_mm256_mul_pd(v, v), keep));
    }
};

template <>
struct Simd<float> {
    using V = __m256;
    static constexpr int W = 8;
    static V load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, V v) { _mm256_storeu_ps(p, v); }
    static V splat(float x) { return _mm256_set1_ps(x); }
    static V zero() { return _mm256_setzero_ps(); }
    static V iota(int start) {
        const float s = static_cast<float>(start);
        return _mm256_setr_ps(s, s + 1, s + 2, s + 3, s + 4, s + 5, s + 6, s + 7);
    }
    static V mul(V a, V b) { return _mm256_mul_ps(a, b); }
    static V sub(V a, V b) { return _mm256_sub_ps(a, b); }
    static V add(V a, V b) { return _mm256_add_ps(a, b); }
    static V magnitude(V a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
    static V greater(V a, V b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }  // false for NaN
    static V select(V a, V b, V mask) { return _mm256_blendv_ps(a, b, mask); }
    // The squares in double, as scan_row: exact products of the widened values.
    static void store_squares(double* out, V v) {
        const __m256d low = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
        const __m256d high = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
        _mm256_storeu_pd(out, _mm256_mul_pd(low, low));
        _mm256_storeu_pd(out + 4, _mm256_mul_pd(high, high));
    }
    // The lanes whose mask byte is not 0 (all ones), for the values and for
    // their squares (in double); the other lanes are cleared to +0.0.
    struct Keep {
        __m256 values;
        __m256d low, high;
    };
    static Keep keep(const std::uint8_t* mask) {
        std::int64_t bytes;
        std::memcpy(&bytes, mask, sizeof(bytes));
        const __m128i b = _mm_cvtsi64_si128(bytes);
        const __m256i zero = _mm256_setzero_si256();
        return {_mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_cvtepu8_epi32(b), zero)),
                _mm256_castsi256_pd(_mm256_cmpgt_epi64(_mm256_cvtepu8_epi64(b), zero)),
                _mm256_castsi256_pd(
                    _mm256_cmpgt_epi64(_mm256_cvtepu8_epi64(_mm_srli_si128(b, 4)), zero))};
    }
    static V clear(V v, const Keep& keep) { return _mm256_and_ps(v, keep.values); }
    static void store_squares(double* out, V v, const Keep& keep) {
        const __m256d low = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
        const __m256d high = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
        _mm256_storeu_pd(out, _mm256_and_pd(_mm256_mul_pd(low, low), keep.low));
        _mm256_storeu_pd(out + 4, _mm256_and_pd(_mm256_mul_pd(high, high), keep.high));
    }
};

#include "hclean_fast_kernels.inc"

}  // namespace

hclean_fast::Kernels hclean_fast::avx2_kernels() {
    return {"avx2", search_f64, search_f32, flush_f64, flush_f32};
}

#endif  // defined(__x86_64__) || defined(_M_X64)
