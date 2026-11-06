// The code of path S shared by every ISA build (dft_s_kernels.hpp). Included
// inside an anonymous namespace by dft_s_generic.cpp and dft_s_avx2.cpp, after
// <cmath> and dft_s_kernels.hpp; plain functions only (no inline or template
// code from a header is instantiated, so the AVX2 unit defines no weak symbol).

using dft_fast::i64;
using dft_fast::kTileSamples;

// The phasors are those of the reference prediction: the constants,
// round_to_integer and cos_sin_of_turns below are verbatim copies of the
// reference's (processing_functions/imaging/gridders/direct_fourier_transform_cpp
// /src/direct_fourier_transform.cpp, where they are internal to that unit),
// built with the same -ffp-contract=off. The tests compare the text with the
// reference source and the values bit for bit with its cos_sin_turns.

constexpr double kTwoPi = 6.28318530717958647692528676655900577;

// 1.5 * 2^52: (x + kRoundMagic) - kRoundMagic is x rounded to the nearest
// integer (ties to even) for |x| < 2^51, in plain arithmetic that the
// compiler vectorises (no library call). -ffp-contract=off and the absence
// of -ffast-math keep the two additions as written.
constexpr double kRoundMagic = 6755399441055744.0;

// Cephes sin.c coefficients of sin(x) and cos(x) for |x| <= pi / 4.
constexpr double kSin0 = 1.58962301576546568060E-10;
constexpr double kSin1 = -2.50507477628578072866E-8;
constexpr double kSin2 = 2.75573136213857245213E-6;
constexpr double kSin3 = -1.98412698295895385996E-4;
constexpr double kSin4 = 8.33333333332211858878E-3;
constexpr double kSin5 = -1.66666666666666307295E-1;
constexpr double kCos0 = -1.13585365213876817300E-11;
constexpr double kCos1 = 2.08757008419747316778E-9;
constexpr double kCos2 = -2.75573141792967388112E-7;
constexpr double kCos3 = 2.48015872888517045348E-5;
constexpr double kCos4 = -1.38888888888730564116E-3;
constexpr double kCos5 = 4.16666666666665929218E-2;

inline double round_to_integer(double x) {
    return (x + kRoundMagic) - kRoundMagic;
}

// cos(2 pi t) and sin(2 pi t). The reduction to [-1/2, 1/2] turns and then
// to [-1/8, 1/8] turns around a multiple q of a quarter turn is exact, so the
// only errors are the rounding of 2 pi times the reduced angle and that of
// the polynomials (about one unit in the last place each). The quarter turn
// rotation, with cos(q pi / 2) = 1 - |q| and sin(q pi / 2) = q (2 - |q|) for
// q in -2 .. 2, multiplies by -1, 0 or 1 only, which is exact.
inline void cos_sin_of_turns(double turns, double& cos_value, double& sin_value) {
    const double reduced = turns - round_to_integer(turns);
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

void cos_sin_turns(const double* turns, i64 count, double* cos_values, double* sin_values) {
    for (i64 k = 0; k < count; ++k) {
        double c, s;
        cos_sin_of_turns(turns[k], c, s);
        cos_values[k] = c;
        sin_values[k] = s;
    }
}

// Lane s of a panel: cos_sin_of_turns(v m) is bitwise even in its argument for
// the cosine and odd for the sine (the reduction rounds half to even, which is
// symmetric), which the folding of the m axis relies on.
void column_phasors(const double* v, i64 count, const double* m, i64 columns,
                    double* cos_panels, double* sin_panels) {
    const i64 panels = (count + kTileSamples - 1) / kTileSamples;
    for (i64 p = 0; p < panels; ++p) {
        double lane_v[kTileSamples];
        for (int s = 0; s < kTileSamples; ++s) {
            const i64 k = p * kTileSamples + s;
            lane_v[s] = k < count ? v[k] : 0.0;
        }
        double* panel_cos = cos_panels + p * columns * kTileSamples;
        double* panel_sin = sin_panels + p * columns * kTileSamples;
        for (i64 f = 0; f < columns; ++f) {
            const double m_f = m[f];
            for (int s = 0; s < kTileSamples; ++s) {
                double c, sn;
                cos_sin_of_turns(lane_v[s] * m_f, c, sn);
                panel_cos[f * kTileSamples + s] = c;
                panel_sin[f * kTileSamples + s] = sn;
            }
        }
    }
}

void row_phasors(const double* u, i64 count, double l, double* cos_values,
                 double* sin_values) {
    for (i64 k = 0; k < count; ++k) {
        double c, s;
        cos_sin_of_turns(u[k] * l, c, s);
        cos_values[k] = c;
        sin_values[k] = s;
    }
}

void add_row(const double* __restrict cos_values, const double* __restrict sin_values,
             const double* __restrict t_re, const double* __restrict t_im, i64 count,
             double* __restrict x_re, double* __restrict x_im) {
    for (i64 k = 0; k < count; ++k) {
        x_re[k] += cos_values[k] * t_re[k] - sin_values[k] * t_im[k];
        x_im[k] += cos_values[k] * t_im[k] + sin_values[k] * t_re[k];
    }
}
