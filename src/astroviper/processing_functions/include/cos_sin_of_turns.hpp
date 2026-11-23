#pragma once

// cos(2 pi t) and sin(2 pi t) of angles t in turns, as the direct Fourier
// transform evaluates its phases.

#include <cmath>

namespace astroviper {

inline constexpr double kTwoPi = 6.28318530717958647692528676655900577;

// 1.5 * 2^52: (x + kRoundMagic) - kRoundMagic is x rounded to the nearest
// integer (ties to even) for |x| < 2^51, in plain arithmetic that the
// compiler vectorises (no library call). -ffp-contract=off and the absence
// of -ffast-math keep the two additions as written.
inline constexpr double kRoundMagic = 6755399441055744.0;

// 2^51 and 2^52: from 2^51 to 2^52 a double is a multiple of 1/2, from 2^52
// on an integer.
inline constexpr double kTwoTo51 = 2251799813685248.0;
inline constexpr double kTwoTo52 = 4503599627370496.0;

// Cephes sin.c coefficients of sin(x) and cos(x) for |x| <= pi / 4.
inline constexpr double kSin0 = 1.58962301576546568060E-10;
inline constexpr double kSin1 = -2.50507477628578072866E-8;
inline constexpr double kSin2 = 2.75573136213857245213E-6;
inline constexpr double kSin3 = -1.98412698295895385996E-4;
inline constexpr double kSin4 = 8.33333333332211858878E-3;
inline constexpr double kSin5 = -1.66666666666666307295E-1;
inline constexpr double kCos0 = -1.13585365213876817300E-11;
inline constexpr double kCos1 = 2.08757008419747316778E-9;
inline constexpr double kCos2 = -2.75573141792967388112E-7;
inline constexpr double kCos3 = 2.48015872888517045348E-5;
inline constexpr double kCos4 = -1.38888888888730564116E-3;
inline constexpr double kCos5 = 4.16666666666665929218E-2;

inline double round_to_integer(double x) {
    return (x + kRoundMagic) - kRoundMagic;
}

// turns minus the integer nearest to it (ties to even), exactly, for every
// finite turns: (turns + shift) - shift is that integer with shift =
// kRoundMagic below 2^51 (round_to_integer), 2^52 with the sign of turns from
// 2^51 to 2^52 (where kRoundMagic would round to an even integer) and 0 from
// 2^52 on (turns is an integer). An infinite or NaN turns gives NaN. Only the
// shift is selected, so that the loops around stay vectorised.
inline double reduce_turns(double turns) {
    const double magnitude = std::abs(turns);
    const double shift = magnitude < kTwoTo51   ? kRoundMagic
                         : magnitude < kTwoTo52 ? std::copysign(kTwoTo52, turns)
                                                : 0.0;
    return turns - ((turns + shift) - shift);
}

// cos(2 pi t) and sin(2 pi t). The reduction to [-1/2, 1/2] turns and then
// to [-1/8, 1/8] turns around a multiple q of a quarter turn is exact, so the
// only errors are the rounding of 2 pi times the reduced angle and that of
// the polynomials (about one unit in the last place each). The quarter turn
// rotation, with cos(q pi / 2) = 1 - |q| and sin(q pi / 2) = q (2 - |q|) for
// q in -2 .. 2, multiplies by -1, 0 or 1 only, which is exact. The first
// reduction is round_to_integer's, exact for |t| < 2^51, or with AnyTurns
// reduce_turns', exact for every finite t and bitwise the same below 2^51.
template <bool AnyTurns = false>
inline void cos_sin_of_turns(double turns, double& cos_value, double& sin_value) {
    const double reduced = AnyTurns ? reduce_turns(turns) : turns - round_to_integer(turns);
    const double quarter = round_to_integer(4.0 * reduced);
    const double x = kTwoPi * (reduced - 0.25 * quarter);
    const double z = x * x;
    const double sin_x =
        x + x * z * (((((kSin0 * z + kSin1) * z + kSin2) * z + kSin3) * z + kSin4) * z + kSin5);
    const double cos_x =
        1.0 - 0.5 * z +
        z * z * (((((kCos0 * z + kCos1) * z + kCos2) * z + kCos3) * z + kCos4) * z + kCos5);
    const double abs_quarter = std::abs(quarter);
    const double cos_quarter = 1.0 - abs_quarter;
    const double sin_quarter = quarter * (2.0 - abs_quarter);
    cos_value = cos_quarter * cos_x - sin_quarter * sin_x;
    sin_value = sin_quarter * cos_x + cos_quarter * sin_x;
}

}  // namespace astroviper
