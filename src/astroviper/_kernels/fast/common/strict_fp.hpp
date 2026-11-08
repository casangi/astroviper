#pragma once

// The floating-point semantics that the fast modules need: every float and
// double operation rounded once, as written. The fast Hogbom and Asp give the
// reference's bits, and the fast DFT's exact phases rest on error-free
// transformations (Dekker's split and two-product); none of that holds under
// reassociation, reciprocals, ignored signed zeros, assumed finite values,
// excess precision or constants rounded to single precision. This header
// refuses (#error, static_assert) what the compiler announces of them. Every
// translation unit of a fast module and every fast module's compile probe
// includes it, and cmake/AstroviperKernels.cmake compiles it alone first, with
// the flags of each configuration that is built, so that such a toolchain
// skips the fast modules at configure time instead of building inexact code.
//
// Contraction (a * b + c fused into one rounding) has no macro: the fast
// modules are built only by the compilers that astroviper_cpp_flags gives
// -ffp-contract=off (GNU, Clang and AppleClang; AstroviperPybind.cmake).
// GCC defines each macro refused here. Clang defines only __FAST_MATH__ (when
// every part of -ffast-math is on) and __FINITE_MATH_ONLY__ of them, so with
// Clang the fast modules are compiled with -fno-fast-math after the flags they
// inherit, which turns -fassociative-math, -freciprocal-math,
// -fno-signed-zeros, -fapprox-func, -fno-honor-nans and the like off again.
// Neither covers options given straight to Clang's front end (-Xclang) or link
// options: LDFLAGS=-mdaz-ftz links crtfastmath.o into every module, which then
// sets flush-to-zero and denormals-are-zero in the thread that loads it.
// Preprocessor checks and a static_assert only: an *_avx2.cpp unit may
// include it.

#include <cfloat>  // FLT_EVAL_METHOD

#if defined(__FAST_MATH__)
#error "AstroVIPER fast kernels: __FAST_MATH__ (-ffast-math, -Ofast); they need floating-point operations rounded as written"
#endif
#if defined(__ASSOCIATIVE_MATH__)
#error "AstroVIPER fast kernels: __ASSOCIATIVE_MATH__ (-fassociative-math, -funsafe-math-optimizations); they need floating-point operations rounded as written"
#endif
#if defined(__RECIPROCAL_MATH__)
#error "AstroVIPER fast kernels: __RECIPROCAL_MATH__ (-freciprocal-math, -funsafe-math-optimizations); they need floating-point operations rounded as written"
#endif
#if defined(__NO_SIGNED_ZEROS__)
#error "AstroVIPER fast kernels: __NO_SIGNED_ZEROS__ (-fno-signed-zeros, -funsafe-math-optimizations); they need floating-point operations rounded as written"
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "AstroVIPER fast kernels: __FINITE_MATH_ONLY__ (-ffinite-math-only); they need floating-point operations rounded as written"
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "AstroVIPER fast kernels: FLT_EVAL_METHOD is not 0 (excess precision, as with -mfpmath=387); they need floating-point operations rounded as written"
#endif

// GCC announces -fsingle-precision-constant by no macro. Under it 2^27 + 1,
// the splitter of the fast DFT's Dekker split, becomes 2^27.
static_assert(134217729.0 > 134217728.0,
              "AstroVIPER fast kernels: floating constants are rounded to single precision "
              "(-fsingle-precision-constant); they need floating-point operations rounded as "
              "written");
