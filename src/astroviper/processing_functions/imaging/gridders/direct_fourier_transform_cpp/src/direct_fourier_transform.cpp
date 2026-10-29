#include "../include/direct_fourier_transform.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

namespace direct_fourier_transform {

namespace {

constexpr double kSpeedOfLight = 299792458.0;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

// Pixels per block of the prediction: one block's coordinates and phases stay
// in the L1 cache.
constexpr std::int64_t kBlock = 128;

// Imaging tiles of kTileRows image rows by at most kTileColumns image
// columns, the columns of an image shared out evenly between its tiles: the
// accumulators of a tile stay in the cache while all samples stream past, and
// an image has many equal tiles to share between the threads. The tiles
// depend on the image size only (never on the number of threads).
constexpr std::int64_t kTileRows = 4;
constexpr std::int64_t kTileColumns = 64;
constexpr std::int64_t kTilePixels = kTileRows * kTileColumns;

// Up to this |2 pi w (n - 1)| (radians) the cosine and sine of the w term
// are their Taylor polynomials to x^6 and x^5 (cos_sin_of_small_angle),
// whose truncation errors x^8 / 8! and x^7 / 7! stay below 4e-19.
constexpr double kSmallAngle = 0.0078125;  // 2^-7

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

// cos(x) and sin(x) for |x| <= kSmallAngle (radians).
inline void cos_sin_of_small_angle(double x, double& cos_value, double& sin_value) {
    const double z = x * x;
    sin_value = x + x * z * (-1.0 / 6.0 + z * (1.0 / 120.0));
    cos_value = 1.0 + z * (-0.5 + z * (1.0 / 24.0 - z * (1.0 / 720.0)));
}

// Phases of one sample at a block of pixels, then their cosines and sines.
// Plain loops over arrays with an inlined body: the compiler vectorises them.
inline void block_cos_sin(double u, double v, double w, const double* l_block,
                          const double* m_block, const double* n_minus_one_block,
                          std::int64_t count, double* cos_block, double* sin_block) {
    for (std::int64_t j = 0; j < count; ++j) {
        const double turns = u * l_block[j] + v * m_block[j] + w * n_minus_one_block[j];
        double c, s;
        cos_sin_of_turns(turns, c, s);
        cos_block[j] = c;
        sin_block[j] = s;
    }
}

// One sample's value * exp(-i phase) added to the accumulators of every
// polarization over a block of pixels, from the block's cosines and sines, in
// one pass (each cosine and sine read once). The number of polarizations is a
// template argument, so the loop over them is unrolled and the loop over the
// pixels vectorises. Each accumulator is updated with exactly the operations
// of the generic loop over the polarizations in image_samples.
template <int NPol, int NComponents, bool ComplexImage>
inline void accumulate_sample(const double* value, const double* cos_block,
                              const double* sin_block, std::int64_t count,
                              double* accumulator_re, double* accumulator_im) {
    double re[NPol], im[NPol];
    for (int p = 0; p < NPol; ++p) {
        re[p] = value[NComponents * p];
        im[p] = NComponents == 2 ? value[2 * p + 1] : 0.0;
    }
    for (std::int64_t j = 0; j < count; ++j) {
        const double c = cos_block[j];
        const double s = sin_block[j];
        for (int p = 0; p < NPol; ++p) {
            if constexpr (NComponents == 2) {
                accumulator_re[p * kTilePixels + j] += re[p] * c + im[p] * s;
                if constexpr (ComplexImage) {
                    accumulator_im[p * kTilePixels + j] += im[p] * c - re[p] * s;
                }
            } else {
                accumulator_re[p * kTilePixels + j] += re[p] * c;
            }
        }
    }
}

// The prediction's sums over a block of pixels for NPol polarizations:
// sum_re[p] + i sum_im[p] += model_over_n[q, pol_map[p]] * (cos + i sin),
// q in order.
template <int NPol>
inline void sum_pixels(const double* model_over_n, std::int64_t m_pol,
                       const std::int64_t* pol_map, const double* cos_block,
                       const double* sin_block, std::int64_t count, double* sum_re,
                       double* sum_im) {
    double re[NPol], im[NPol];
    std::int64_t offset[NPol];
    for (int p = 0; p < NPol; ++p) {
        re[p] = sum_re[p];
        im[p] = sum_im[p];
        offset[p] = 2 * pol_map[p];
    }
    for (std::int64_t j = 0; j < count; ++j) {
        const double* value = model_over_n + j * m_pol * 2;
        for (int p = 0; p < NPol; ++p) {
            const double model_re = value[offset[p]];
            const double model_im = value[offset[p] + 1];
            re[p] += model_re * cos_block[j] - model_im * sin_block[j];
            im[p] += model_re * sin_block[j] + model_im * cos_block[j];
        }
    }
    for (int p = 0; p < NPol; ++p) {
        sum_re[p] = re[p];
        sum_im[p] = im[p];
    }
}

// n = sqrt(1 - l^2 - m^2) and n - 1 = -(l^2 + m^2) / (1 + n), the latter
// without the cancellation of sqrt(...) - 1 for a small field.
inline void direction_n(double l, double m, double& n, double& n_minus_one) {
    const double r2 = l * l + m * m;
    n = std::sqrt(1.0 - r2);
    n_minus_one = -r2 / (1.0 + n);
}

template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename ImageT>
inline void add_to_pixel(ImageT* pixel, double re, double im) {
    if constexpr (is_complex<ImageT>::value) {
        using Element = typename ImageT::value_type;
        *pixel += ImageT(static_cast<Element>(re), static_cast<Element>(im));
    } else {
        (void)im;
        *pixel += static_cast<ImageT>(re);
    }
}

template <typename ModelT>
inline void read_model(const ModelT& value, double& re, double& im) {
    if constexpr (is_complex<ModelT>::value) {
        re = static_cast<double>(value.real());
        im = static_cast<double>(value.imag());
    } else {
        re = static_cast<double>(value);
        im = 0.0;
    }
}

int resolve_threads(int processing_function_threads, std::int64_t n_work) {
    int threads = processing_function_threads;
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 1;
    }
    if (static_cast<std::int64_t>(threads) > n_work) {
        threads = static_cast<int>(std::max<std::int64_t>(n_work, 1));
    }
    return threads;
}

template <typename Worker>
void run_threads(int n_threads, Worker worker) {
    if (n_threads <= 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    for (int t = 1; t < n_threads; ++t) threads.emplace_back(worker, t);
    worker(0);
    for (auto& th : threads) th.join();
}

// The valid samples of every image plane (a_time, a_chan), in their order in
// the visibility arrays: u, v, w in wavelengths and, per polarization, the
// weighted visibility (zero for a polarization whose sample is not valid).
// Samples without a valid polarization are not kept.
struct PlaneSamples {
    std::vector<std::int64_t> plane_start;  // n_planes + 1
    std::vector<double> u, v, w;            // n_samples
    std::vector<double> values;             // n_samples * n_pol * n_components
};

// n_components = 2: weight * vis (re, im); 1: weight (the uv sampling).
template <int NComponents>
PlaneSamples collect_samples(
    double* normalization, const std::complex<double>* vis_data, const double* uvw,
    const double* frequency_coord, const std::int64_t* frequency_map,
    const std::int64_t* time_map, const std::int64_t* pol_map, const double* weight,
    std::int64_t m_chan, std::int64_t m_pol, std::int64_t n_planes, std::int64_t n_time,
    std::int64_t n_baseline, std::int64_t n_vis_chan, std::int64_t n_pol) {
    std::vector<double> scale(static_cast<std::size_t>(n_vis_chan));
    for (std::int64_t c = 0; c < n_vis_chan; ++c) scale[c] = frequency_coord[c] / kSpeedOfLight;

    // weighted value of one polarization of a sample; false if the sample is
    // skipped (as the gridders skip it)
    auto sample_value = [&](std::int64_t index, double* out) -> bool {
        const double sample_weight = weight[index];
        if constexpr (NComponents == 2) {
            const std::complex<double> weighted = vis_data[index] * sample_weight;
            if (std::isnan(weighted.real()) || std::isnan(weighted.imag())) return false;
            if (weighted == std::complex<double>(0.0, 0.0)) return false;
            out[0] = weighted.real();
            out[1] = weighted.imag();
        } else {
            if (std::isnan(sample_weight) || sample_weight == 0.0) return false;
            out[0] = sample_weight;
        }
        return true;
    };

    PlaneSamples samples;
    samples.plane_start.assign(static_cast<std::size_t>(n_planes + 1), 0);
    double value[2];
    // first pass: the number of valid samples per plane
    for (std::int64_t t = 0; t < n_time; ++t) {
        for (std::int64_t b = 0; b < n_baseline; ++b) {
            const double* uvw_b = uvw + (t * n_baseline + b) * 3;
            if (!std::isfinite(uvw_b[0]) || !std::isfinite(uvw_b[1]) || !std::isfinite(uvw_b[2]))
                continue;
            for (std::int64_t c = 0; c < n_vis_chan; ++c) {
                const std::int64_t base = ((t * n_baseline + b) * n_vis_chan + c) * n_pol;
                bool any = false;
                for (std::int64_t p = 0; p < n_pol && !any; ++p) any = sample_value(base + p, value);
                if (any) ++samples.plane_start[time_map[t] * m_chan + frequency_map[c] + 1];
            }
        }
    }
    for (std::int64_t plane = 0; plane < n_planes; ++plane) {
        samples.plane_start[plane + 1] += samples.plane_start[plane];
    }
    const std::int64_t n_samples = samples.plane_start[n_planes];
    samples.u.resize(static_cast<std::size_t>(n_samples));
    samples.v.resize(static_cast<std::size_t>(n_samples));
    samples.w.resize(static_cast<std::size_t>(n_samples));
    samples.values.assign(static_cast<std::size_t>(n_samples * n_pol * NComponents), 0.0);

    // second pass: fill them in order and sum the weights
    std::vector<std::int64_t> cursor(samples.plane_start.begin(), samples.plane_start.end() - 1);
    for (std::int64_t t = 0; t < n_time; ++t) {
        for (std::int64_t b = 0; b < n_baseline; ++b) {
            const double* uvw_b = uvw + (t * n_baseline + b) * 3;
            if (!std::isfinite(uvw_b[0]) || !std::isfinite(uvw_b[1]) || !std::isfinite(uvw_b[2]))
                continue;
            for (std::int64_t c = 0; c < n_vis_chan; ++c) {
                const std::int64_t base = ((t * n_baseline + b) * n_vis_chan + c) * n_pol;
                const std::int64_t a_time = time_map[t];
                const std::int64_t a_chan = frequency_map[c];
                const std::int64_t plane = a_time * m_chan + a_chan;
                const std::int64_t k = cursor[plane];
                bool any = false;
                for (std::int64_t p = 0; p < n_pol; ++p) {
                    if (!sample_value(base + p, value)) continue;
                    any = true;
                    double* stored = samples.values.data() + (k * n_pol + p) * NComponents;
                    for (int i = 0; i < NComponents; ++i) stored[i] = value[i];
                    normalization[(a_time * m_chan + a_chan) * m_pol + pol_map[p]] +=
                        weight[base + p];
                }
                if (!any) continue;
                samples.u[k] = uvw_b[0] * scale[c];
                samples.v[k] = uvw_b[1] * scale[c];
                samples.w[k] = uvw_b[2] * scale[c];
                ++cursor[plane];
            }
        }
    }
    return samples;
}

// The imaging transform of collected samples, by tiles of kTileRows image
// rows and tile_columns (at most kTileColumns) image columns: every thread
// takes whole tiles of whole planes (unit, unit + n_threads, ...). For every
// sample the phase factor of a pixel is exp(2 pi i u l) of its row times
// exp(2 pi i v m) of its column times that of the w term, which is exactly 1
// for w = 0 and the small angle polynomials while |2 pi w (n - 1)| <=
// kSmallAngle over the tile; otherwise the phase is reduced and evaluated per
// pixel. The three are accurate to rounding.
template <typename ImageT, int NComponents>
void image_samples(
    ImageT* image, const PlaneSamples& samples, const std::int64_t* pol_map,
    const double* l, const double* m, std::int64_t m_pol,
    std::int64_t n_planes, std::int64_t n_l, std::int64_t n_m, std::int64_t n_pol,
    int processing_function_threads) {
    constexpr bool complex_image = is_complex<ImageT>::value;
    const std::int64_t n_pixels = n_l * n_m;
    const std::int64_t n_row_tiles = (n_l + kTileRows - 1) / kTileRows;
    const std::int64_t n_column_tiles = (n_m + kTileColumns - 1) / kTileColumns;
    const std::int64_t tile_columns = (n_m + n_column_tiles - 1) / n_column_tiles;
    const std::int64_t n_tiles = n_row_tiles * n_column_tiles;
    const std::int64_t n_units = n_planes * n_tiles;
    if (n_units <= 0 || samples.plane_start[n_planes] == 0) return;
    const int n_threads = resolve_threads(processing_function_threads, n_units);

    auto worker = [&](int thread_index) {
        std::vector<double> l_tile(kTileRows), m_tile(kTileColumns), n_tile(kTilePixels),
            n_minus_one_tile(kTilePixels), cos_u(kTileRows), sin_u(kTileRows),
            cos_v(kTileColumns), sin_v(kTileColumns), cos_block(kTileColumns),
            sin_block(kTileColumns),
            accumulator_re(static_cast<std::size_t>(n_pol * kTilePixels)),
            accumulator_im(static_cast<std::size_t>(n_pol * kTilePixels));
        for (std::int64_t unit = thread_index; unit < n_units; unit += n_threads) {
            const std::int64_t plane = unit / n_tiles;
            const std::int64_t k_begin = samples.plane_start[plane];
            const std::int64_t k_end = samples.plane_start[plane + 1];
            if (k_begin == k_end) continue;
            const std::int64_t row_begin = ((unit % n_tiles) / n_column_tiles) * kTileRows;
            const std::int64_t column_begin = ((unit % n_tiles) % n_column_tiles) * tile_columns;
            const std::int64_t rows = std::min(kTileRows, n_l - row_begin);
            const std::int64_t columns = std::min(tile_columns, n_m - column_begin);
            double max_abs_n_minus_one = 0.0;
            for (std::int64_t r = 0; r < rows; ++r) l_tile[r] = l[row_begin + r];
            for (std::int64_t j = 0; j < columns; ++j) m_tile[j] = m[column_begin + j];
            for (std::int64_t r = 0; r < rows; ++r) {
                for (std::int64_t j = 0; j < columns; ++j) {
                    const std::int64_t t = r * kTileColumns + j;
                    direction_n(l_tile[r], m_tile[j], n_tile[t], n_minus_one_tile[t]);
                    max_abs_n_minus_one =
                        std::max(max_abs_n_minus_one, std::abs(n_minus_one_tile[t]));
                }
            }
            std::fill(accumulator_re.begin(), accumulator_re.end(), 0.0);
            std::fill(accumulator_im.begin(), accumulator_im.end(), 0.0);
            for (std::int64_t k = k_begin; k < k_end; ++k) {
                const double* value = samples.values.data() + k * n_pol * NComponents;
                const double u = samples.u[k], v = samples.v[k], w = samples.w[k];
                const double two_pi_w = kTwoPi * w;
                const int w_term = w == 0.0 ? 0
                                   : std::abs(two_pi_w) * max_abs_n_minus_one <= kSmallAngle
                                       ? 1
                                       : 2;
                if (w_term != 2) {
                    for (std::int64_t r = 0; r < rows; ++r) {
                        cos_sin_of_turns(u * l_tile[r], cos_u[r], sin_u[r]);
                    }
                    for (std::int64_t j = 0; j < columns; ++j) {
                        double c, s;
                        cos_sin_of_turns(v * m_tile[j], c, s);
                        cos_v[j] = c;
                        sin_v[j] = s;
                    }
                }
                for (std::int64_t r = 0; r < rows; ++r) {
                    const double* n_minus_one_row = n_minus_one_tile.data() + r * kTileColumns;
                    if (w_term == 0) {
                        // exp(2 pi i (u l + v m))
                        const double ca = cos_u[r], sa = sin_u[r];
                        for (std::int64_t j = 0; j < columns; ++j) {
                            cos_block[j] = ca * cos_v[j] - sa * sin_v[j];
                            sin_block[j] = ca * sin_v[j] + sa * cos_v[j];
                        }
                    } else if (w_term == 1) {
                        // exp(2 pi i (u l + v m)) exp(2 pi i w (n - 1))
                        const double ca = cos_u[r], sa = sin_u[r];
                        for (std::int64_t j = 0; j < columns; ++j) {
                            double cw, sw;
                            cos_sin_of_small_angle(two_pi_w * n_minus_one_row[j], cw, sw);
                            const double c = ca * cos_v[j] - sa * sin_v[j];
                            const double s = ca * sin_v[j] + sa * cos_v[j];
                            cos_block[j] = c * cw - s * sw;
                            sin_block[j] = c * sw + s * cw;
                        }
                    } else {
                        const double l_row = l_tile[r];
                        for (std::int64_t j = 0; j < columns; ++j) {
                            const double turns =
                                u * l_row + v * m_tile[j] + w * n_minus_one_row[j];
                            double c, s;
                            cos_sin_of_turns(turns, c, s);
                            cos_block[j] = c;
                            sin_block[j] = s;
                        }
                    }
                    // value * exp(-i phase) = (re c + im s) + i (im c - re s)
                    double* acc_re = accumulator_re.data() + r * kTileColumns;
                    double* acc_im = accumulator_im.data() + r * kTileColumns;
                    if (n_pol == 4) {
                        accumulate_sample<4, NComponents, complex_image>(
                            value, cos_block.data(), sin_block.data(), columns, acc_re, acc_im);
                    } else if (n_pol == 2) {
                        accumulate_sample<2, NComponents, complex_image>(
                            value, cos_block.data(), sin_block.data(), columns, acc_re, acc_im);
                    } else {
                        for (std::int64_t p = 0; p < n_pol; ++p) {
                            accumulate_sample<1, NComponents, complex_image>(
                                value + p * NComponents, cos_block.data(), sin_block.data(),
                                columns, acc_re + p * kTilePixels, acc_im + p * kTilePixels);
                        }
                    }
                }
            }
            for (std::int64_t p = 0; p < n_pol; ++p) {
                ImageT* out = image + (plane * m_pol + pol_map[p]) * n_pixels;
                for (std::int64_t r = 0; r < rows; ++r) {
                    ImageT* out_row = out + (row_begin + r) * n_m + column_begin;
                    const std::int64_t t = p * kTilePixels + r * kTileColumns;
                    const double* n_row = n_tile.data() + r * kTileColumns;
                    for (std::int64_t j = 0; j < columns; ++j) {
                        add_to_pixel(out_row + j, n_row[j] * accumulator_re[t + j],
                                     n_row[j] * accumulator_im[t + j]);
                    }
                }
            }
        }
    };
    run_threads(n_threads, worker);
}

}  // namespace

void cos_sin_turns(const double* turns, double* cos_values, double* sin_values,
                   std::int64_t count) {
    for (std::int64_t j = 0; j < count; ++j) {
        double c, s;
        cos_sin_of_turns(turns[j], c, s);
        cos_values[j] = c;
        sin_values[j] = s;
    }
}

template <typename ImageT>
void dft_image(
    ImageT* image, double* normalization, const std::complex<double>* vis_data,
    const double* uvw, const double* frequency_coord, const std::int64_t* frequency_map,
    const std::int64_t* time_map, const std::int64_t* pol_map, const double* weight,
    const double* l, const double* m, std::int64_t m_time, std::int64_t m_chan,
    std::int64_t m_pol, std::int64_t n_l, std::int64_t n_m, std::int64_t n_time,
    std::int64_t n_baseline, std::int64_t n_vis_chan, std::int64_t n_pol,
    int processing_function_threads) {
    const std::int64_t n_planes = m_time * m_chan;
    const PlaneSamples samples = collect_samples<2>(
        normalization, vis_data, uvw, frequency_coord, frequency_map, time_map, pol_map, weight,
        m_chan, m_pol, n_planes, n_time, n_baseline, n_vis_chan, n_pol);
    image_samples<ImageT, 2>(image, samples, pol_map, l, m, m_pol, n_planes, n_l, n_m,
                             n_pol, processing_function_threads);
}

template <typename ImageT>
void dft_image_uv_sampling(
    ImageT* image, double* normalization, const double* uvw, const double* frequency_coord,
    const std::int64_t* frequency_map, const std::int64_t* time_map,
    const std::int64_t* pol_map, const double* weight, const double* l, const double* m,
    std::int64_t m_time, std::int64_t m_chan, std::int64_t m_pol, std::int64_t n_l,
    std::int64_t n_m, std::int64_t n_time, std::int64_t n_baseline, std::int64_t n_vis_chan,
    std::int64_t n_pol, int processing_function_threads) {
    const std::int64_t n_planes = m_time * m_chan;
    const PlaneSamples samples = collect_samples<1>(
        normalization, nullptr, uvw, frequency_coord, frequency_map, time_map, pol_map, weight,
        m_chan, m_pol, n_planes, n_time, n_baseline, n_vis_chan, n_pol);
    image_samples<ImageT, 1>(image, samples, pol_map, l, m, m_pol, n_planes, n_l, n_m,
                             n_pol, processing_function_threads);
}

template <typename ModelT, typename VisT>
void dft_predict(
    const ModelT* model, VisT* vis_data, const double* uvw, const double* frequency_coord,
    const std::int64_t* frequency_map, const std::int64_t* time_map,
    const std::int64_t* pol_map, const double* l, const double* m, std::int64_t m_time,
    std::int64_t m_chan, std::int64_t m_pol, std::int64_t n_l, std::int64_t n_m,
    std::int64_t n_time, std::int64_t n_baseline, std::int64_t n_vis_chan, std::int64_t n_pol,
    int processing_function_threads) {
    using VisElement = typename VisT::value_type;
    const std::int64_t n_planes = m_time * m_chan;
    const std::int64_t n_pixels = n_l * n_m;

    // The pixels where any polarization of a model plane is non-zero, per
    // plane in pixel order: l, m, n - 1 and the model over n per polarization.
    std::vector<std::int64_t> pixel_start(static_cast<std::size_t>(n_planes + 1), 0);
    for (std::int64_t plane = 0; plane < n_planes; ++plane) {
        std::int64_t count = 0;
        for (std::int64_t pixel = 0; pixel < n_pixels; ++pixel) {
            for (std::int64_t a_pol = 0; a_pol < m_pol; ++a_pol) {
                double re, im;
                read_model(model[(plane * m_pol + a_pol) * n_pixels + pixel], re, im);
                if (re != 0.0 || im != 0.0) {
                    ++count;
                    break;
                }
            }
        }
        pixel_start[plane + 1] = pixel_start[plane] + count;
    }
    const std::int64_t n_components = pixel_start[n_planes];
    std::vector<double> l_pixel(static_cast<std::size_t>(n_components)),
        m_pixel(static_cast<std::size_t>(n_components)),
        n_minus_one_pixel(static_cast<std::size_t>(n_components)),
        model_over_n(static_cast<std::size_t>(n_components * m_pol * 2));
    for (std::int64_t plane = 0; plane < n_planes; ++plane) {
        std::int64_t q = pixel_start[plane];
        for (std::int64_t pixel = 0; pixel < n_pixels; ++pixel) {
            bool any = false;
            for (std::int64_t a_pol = 0; a_pol < m_pol && !any; ++a_pol) {
                double re, im;
                read_model(model[(plane * m_pol + a_pol) * n_pixels + pixel], re, im);
                any = re != 0.0 || im != 0.0;
            }
            if (!any) continue;
            l_pixel[q] = l[pixel / n_m];
            m_pixel[q] = m[pixel % n_m];
            double n;
            direction_n(l_pixel[q], m_pixel[q], n, n_minus_one_pixel[q]);
            for (std::int64_t a_pol = 0; a_pol < m_pol; ++a_pol) {
                double re, im;
                read_model(model[(plane * m_pol + a_pol) * n_pixels + pixel], re, im);
                model_over_n[(q * m_pol + a_pol) * 2 + 0] = re / n;
                model_over_n[(q * m_pol + a_pol) * 2 + 1] = im / n;
            }
            ++q;
        }
    }

    std::vector<double> scale(static_cast<std::size_t>(n_vis_chan));
    for (std::int64_t c = 0; c < n_vis_chan; ++c) scale[c] = frequency_coord[c] / kSpeedOfLight;

    const std::int64_t n_rows = n_time * n_baseline;
    if (n_rows <= 0) return;
    const int n_threads = resolve_threads(processing_function_threads, n_rows);
    const std::int64_t chunk = (n_rows + n_threads - 1) / n_threads;

    auto worker = [&](int thread_index) {
        std::vector<double> cos_block(kBlock), sin_block(kBlock);
        std::vector<double> sum_re(static_cast<std::size_t>(n_pol)),
            sum_im(static_cast<std::size_t>(n_pol));
        const std::int64_t row_end = std::min(n_rows, (thread_index + 1) * chunk);
        for (std::int64_t row = thread_index * chunk; row < row_end; ++row) {
            const std::int64_t t = row / n_baseline;
            const double* uvw_b = uvw + row * 3;
            if (!std::isfinite(uvw_b[0]) || !std::isfinite(uvw_b[1]) || !std::isfinite(uvw_b[2]))
                continue;
            for (std::int64_t c = 0; c < n_vis_chan; ++c) {
                const std::int64_t plane = time_map[t] * m_chan + frequency_map[c];
                const double u = uvw_b[0] * scale[c];
                const double v = uvw_b[1] * scale[c];
                const double w = uvw_b[2] * scale[c];
                std::fill(sum_re.begin(), sum_re.end(), 0.0);
                std::fill(sum_im.begin(), sum_im.end(), 0.0);
                const std::int64_t q_end = pixel_start[plane + 1];
                for (std::int64_t q0 = pixel_start[plane]; q0 < q_end; q0 += kBlock) {
                    const std::int64_t count = std::min(kBlock, q_end - q0);
                    block_cos_sin(u, v, w, l_pixel.data() + q0, m_pixel.data() + q0,
                                  n_minus_one_pixel.data() + q0, count, cos_block.data(),
                                  sin_block.data());
                    // (model / n) * exp(+i phase), summed over the pixels in order;
                    // the polarizations' sums are independent chains
                    const double* value = model_over_n.data() + q0 * m_pol * 2;
                    if (n_pol == 4) {
                        sum_pixels<4>(value, m_pol, pol_map, cos_block.data(), sin_block.data(),
                                      count, sum_re.data(), sum_im.data());
                    } else if (n_pol == 2) {
                        sum_pixels<2>(value, m_pol, pol_map, cos_block.data(), sin_block.data(),
                                      count, sum_re.data(), sum_im.data());
                    } else if (n_pol == 1) {
                        sum_pixels<1>(value, m_pol, pol_map, cos_block.data(), sin_block.data(),
                                      count, sum_re.data(), sum_im.data());
                    } else {
                        for (std::int64_t p = 0; p < n_pol; ++p) {
                            sum_pixels<1>(value, m_pol, pol_map + p, cos_block.data(),
                                          sin_block.data(), count, sum_re.data() + p,
                                          sum_im.data() + p);
                        }
                    }
                }
                VisT* out = vis_data + (row * n_vis_chan + c) * n_pol;
                for (std::int64_t p = 0; p < n_pol; ++p) {
                    const VisT current = out[p];
                    if (std::isnan(static_cast<double>(current.real())) ||
                        std::isnan(static_cast<double>(current.imag())))
                        continue;
                    out[p] = VisT(static_cast<VisElement>(sum_re[p]),
                                  static_cast<VisElement>(sum_im[p]));
                }
            }
        }
    };
    run_threads(n_threads, worker);
}

// Explicit instantiations.
#define DFT_IMAGE_INSTANTIATE(IMAGET)                                                         \
    template void dft_image<IMAGET>(                                                           \
        IMAGET*, double*, const std::complex<double>*, const double*, const double*,          \
        const std::int64_t*, const std::int64_t*, const std::int64_t*, const double*,         \
        const double*, const double*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, int);

DFT_IMAGE_INSTANTIATE(float)
DFT_IMAGE_INSTANTIATE(double)
DFT_IMAGE_INSTANTIATE(std::complex<float>)
DFT_IMAGE_INSTANTIATE(std::complex<double>)
#undef DFT_IMAGE_INSTANTIATE

#define DFT_UV_SAMPLING_INSTANTIATE(IMAGET)                                                    \
    template void dft_image_uv_sampling<IMAGET>(                                              \
        IMAGET*, double*, const double*, const double*, const std::int64_t*,                  \
        const std::int64_t*, const std::int64_t*, const double*, const double*, const double*, \
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,   \
        std::int64_t, std::int64_t, std::int64_t, int);

DFT_UV_SAMPLING_INSTANTIATE(float)
DFT_UV_SAMPLING_INSTANTIATE(double)
#undef DFT_UV_SAMPLING_INSTANTIATE

#define DFT_PREDICT_INSTANTIATE(MODELT, VIST)                                                  \
    template void dft_predict<MODELT, VIST>(                                                   \
        const MODELT*, VIST*, const double*, const double*, const std::int64_t*,              \
        const std::int64_t*, const std::int64_t*, const double*, const double*, std::int64_t, \
        std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,  \
        std::int64_t, std::int64_t, int);

DFT_PREDICT_INSTANTIATE(float, std::complex<float>)
DFT_PREDICT_INSTANTIATE(float, std::complex<double>)
DFT_PREDICT_INSTANTIATE(double, std::complex<float>)
DFT_PREDICT_INSTANTIATE(double, std::complex<double>)
DFT_PREDICT_INSTANTIATE(std::complex<float>, std::complex<float>)
DFT_PREDICT_INSTANTIATE(std::complex<float>, std::complex<double>)
DFT_PREDICT_INSTANTIATE(std::complex<double>, std::complex<float>)
DFT_PREDICT_INSTANTIATE(std::complex<double>, std::complex<double>)
#undef DFT_PREDICT_INSTANTIATE

}  // namespace direct_fourier_transform
