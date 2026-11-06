// The decision between the paths of the fast dft_predict, and path S.
//
// Path S (every used row has w = 0). The phase factor of a sample k then
// separates over the pixel (i, j): exp(2 pi i (u_k l_i + v_k m_j)) = A_ik B_jk,
// so for one real component plane of W = M / n (a part of a model polarization)
//
//     X_k = sum_i A_ik T_ik,   T_ik = sum_j W_ij B_jk,
//
// a GEMM T = W B per model plane, then one sum over the image rows per sample.
// dft_pack.cpp drops the rows and columns that are zero in every component,
// folds the columns with m_b = -m_a (B_b = conj B_a), so that T_re = P cos and
// T_im = Q sin with P = W_a + W_b and Q = W_a - W_b over about n_m / 2
// columns, and sums each distinct component once (XX = YY of a Stokes I sky).
// Visibility polarization p is then V_p = re_sign X[re] + i im_sign X[im].
//
// Every T element is one sequential sum over the folded columns in their fixed
// order (the micro-kernel's depth steps continue it through memory), and every
// X element one over the kept rows in order; a sample's values depend only on
// its u, v and model plane, and a block of samples is computed by one thread.
// The bits therefore do not depend on the threads, the blocking, the order of
// the work or on how the rows are split between calls, and the generic and
// AVX2 routines perform the same rounded operations (dft_s_kernels.hpp). The
// planes are packed and computed one at a time, so the packed weights take
// 16 bytes x components x kept rows x folded columns of one plane at most.

#include <cmath>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include "dft_fast.hpp"

namespace dft_fast {

namespace {

constexpr double kSpeedOfLight = 299792458.0;

// Samples per block (one work item), packed rows per band of the GEMM and
// folded columns per step of its depth, as tuned in the prototype (HL Tau,
// Zen 3). Any values give the same bits.
constexpr i64 kBlockSamples = 256;
constexpr i64 kBandRows = 120;
constexpr i64 kDepth = 344;

// Path S is taken when its multiply-adds per sample (components x kept rows x
// folded columns) are at most kAlpha times the reference's terms per sample
// (the non-zero pixels): the prototype measured the crossover against the
// reference at about 70, with scattered sparse models at 2 and 6 threads.
constexpr double kAlpha = 64.0;

// |u| and |v| below 2^51 turns keep u l and v m (|l|, |m| < 1) in the range
// where cos_sin_of_turns reduces exactly.
constexpr double kMaxTurns = 0x1p51;

// A row of the visibilities that the reference predicts (the others are left
// unchanged by both).
bool used(const double* uvw) {
    return std::isfinite(uvw[0]) && std::isfinite(uvw[1]) && std::isfinite(uvw[2]);
}

i64 round_up(i64 x, i64 step) {
    return (x + step - 1) / step * step;
}

i64 greatest_common_divisor(i64 a, i64 b) {
    while (b != 0) {
        const i64 r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// The samples of every model plane (a_time, a_chan) are the used rows of model
// time a_time times the channels of model channel a_chan; sample s of a plane
// is row rows[row_start[a_time] + s / n_channels] and channel
// channels[channel_start[a_chan] + s % n_channels].
struct Samples {
    std::vector<i64> row_start, rows;
    std::vector<i64> channel_start, channels;

    explicit Samples(const Call& call)
        : row_start(static_cast<std::size_t>(call.m_time + 1), 0),
          channel_start(static_cast<std::size_t>(call.m_chan + 1), 0) {
        const i64 n_rows = call.n_time * call.n_baseline;
        for (i64 row = 0; row < n_rows; ++row) {
            if (used(call.uvw + 3 * row)) ++row_start[call.time_map[row / call.n_baseline] + 1];
        }
        for (i64 c = 0; c < call.n_vis_chan; ++c) ++channel_start[call.frequency_map[c] + 1];
        for (i64 t = 0; t < call.m_time; ++t) row_start[t + 1] += row_start[t];
        for (i64 c = 0; c < call.m_chan; ++c) channel_start[c + 1] += channel_start[c];
        rows.resize(static_cast<std::size_t>(row_start[call.m_time]));
        channels.resize(static_cast<std::size_t>(channel_start[call.m_chan]));
        std::vector<i64> next(row_start.begin(), row_start.end() - 1);
        for (i64 row = 0; row < n_rows; ++row) {
            if (used(call.uvw + 3 * row)) rows[next[call.time_map[row / call.n_baseline]]++] = row;
        }
        next.assign(channel_start.begin(), channel_start.end() - 1);
        for (i64 c = 0; c < call.n_vis_chan; ++c) channels[next[call.frequency_map[c]]++] = c;
    }

    i64 n_rows(i64 a_time) const { return row_start[a_time + 1] - row_start[a_time]; }
    i64 n_channels(i64 a_chan) const { return channel_start[a_chan + 1] - channel_start[a_chan]; }
};

// A model plane with samples, and its layout.
struct Plane {
    i64 a_time, a_chan, n_samples;
    PlaneLayout layout;
};

// One block of a plane's samples: its phasors, the GEMM bands with the sums
// over their rows, then its visibilities.
template <typename VisT>
void compute_block(const Call& call, const Samples& samples, const Plane& plane,
                   const double* scale, const double* p, const double* q, i64 block,
                   const Kernels& routines, Scratch& scratch) {
    const PlaneLayout& layout = plane.layout;
    const i64* rows = samples.rows.data() + samples.row_start[plane.a_time];
    const i64* channels = samples.channels.data() + samples.channel_start[plane.a_chan];
    const i64 n_channels = samples.n_channels(plane.a_chan);
    const i64 first = block * kBlockSamples;
    const i64 count = std::min(kBlockSamples, plane.n_samples - first);
    const i64 width = round_up(count, kTileSamples);  // the block in whole micro-panels
    const auto n_components = static_cast<i64>(layout.components.size());

    double* x_re = scratch.x_re.get(n_components * width);
    double* x_im = scratch.x_im.get(n_components * width);
    if (n_components > 0) {
        double* u = scratch.u.get(count);
        double* v = scratch.v.get(count);
        for (i64 k = 0; k < count; ++k) {
            const double* uvw = call.uvw + 3 * rows[(first + k) / n_channels];
            const double factor = scale[channels[(first + k) % n_channels]];
            u[k] = uvw[0] * factor;
            v[k] = uvw[1] * factor;
        }
        const auto n_columns = static_cast<i64>(layout.column_a.size());
        const i64 n_packed = static_cast<i64>(layout.rows.size()) * n_components;
        const i64 padded_rows = round_up(n_packed, kTileRows);
        // A band holds whole micro-panels and whole rows (all their components).
        const i64 step =
            kTileRows / greatest_common_divisor(kTileRows, n_components) * n_components;
        const i64 band_rows = std::max(step, kBandRows / step * step);
        double* column_cos = scratch.column_cos.get(n_columns * width);
        double* column_sin = scratch.column_sin.get(n_columns * width);
        double* t_re = scratch.t_re.get(band_rows * width);
        double* t_im = scratch.t_im.get(band_rows * width);
        double* row_cos = scratch.row_cos.get(count);
        double* row_sin = scratch.row_sin.get(count);
        std::fill(x_re, x_re + n_components * width, 0.0);
        std::fill(x_im, x_im + n_components * width, 0.0);

        routines.column_phasors(v, count, layout.m_a.data(), n_columns, column_cos, column_sin);
        const i64 panels = width / kTileSamples;
        for (i64 band = 0; band < padded_rows; band += band_rows) {
            const i64 band_panels = std::min(band_rows, padded_rows - band) / kTileRows;
            const i64 band_offset = band / kTileRows * n_columns * kTileRows;
            for (i64 f0 = 0; f0 < n_columns; f0 += kDepth) {
                const i64 depth = std::min(kDepth, n_columns - f0);
                for (int pass = 0; pass < 2; ++pass) {  // T_re = P cos, T_im = Q sin
                    const double* a = (pass == 0 ? p : q) + band_offset + f0 * kTileRows;
                    const double* b = (pass == 0 ? column_cos : column_sin) + f0 * kTileSamples;
                    double* t = pass == 0 ? t_re : t_im;
                    for (i64 jr = 0; jr < panels; ++jr) {
                        for (i64 ir = 0; ir < band_panels; ++ir) {
                            routines.multiply(depth, a + ir * n_columns * kTileRows,
                                              b + jr * n_columns * kTileSamples,
                                              t + ir * kTileRows * width + jr * kTileSamples,
                                              width, f0 > 0);
                        }
                    }
                }
            }
            // X_c += A_i T_ic over the band's kept rows, in row order.
            const i64 band_end = std::min(band + band_rows, n_packed);
            for (i64 r = band; r < band_end; r += n_components) {
                const double l = call.l[layout.rows[r / n_components]];
                routines.row_phasors(u, count, l, row_cos, row_sin);
                for (i64 c = 0; c < n_components; ++c) {
                    const i64 local = (r + c - band) * width;
                    routines.add_row(row_cos, row_sin, t_re + local, t_im + local, count,
                                     x_re + c * width, x_im + c * width);
                }
            }
        }
    }

    // The visibilities, overwritten as the reference writes them: NaN values
    // are kept, the others narrowed from the double sums.
    using VisElement = typename VisT::value_type;
    VisT* vis = static_cast<VisT*>(call.vis);
    for (i64 k = 0; k < count; ++k) {
        const i64 row = rows[(first + k) / n_channels];
        VisT* out = vis + (row * call.n_vis_chan + channels[(first + k) % n_channels]) * call.n_pol;
        for (i64 pol = 0; pol < call.n_pol; ++pol) {
            const VisT current = out[pol];
            if (std::isnan(static_cast<double>(current.real())) ||
                std::isnan(static_cast<double>(current.imag())))
                continue;
            double re = 0.0, im = 0.0;
            if (n_components > 0) {
                const PlaneLayout::Correlation& correlation = layout.correlations[pol];
                if (correlation.re >= 0) {
                    re = correlation.re_sign * x_re[correlation.re * width + k];
                    im = correlation.re_sign * x_im[correlation.re * width + k];
                }
                if (correlation.im >= 0) {  // i X = -X_im + i X_re
                    re = re - correlation.im_sign * x_im[correlation.im * width + k];
                    im = im + correlation.im_sign * x_re[correlation.im * width + k];
                }
            }
            out[pol] = VisT(static_cast<VisElement>(re), static_cast<VisElement>(im));
        }
    }
}

template <typename ModelT, typename VisT>
Path predict_typed(const Call& call) {
    const i64 n_rows = call.n_time * call.n_baseline;
    // The decision, from the inputs alone and cheapest test first; nothing is
    // written before it ends. (1) Every used row has w == 0 (-0.0 is 0, a
    // single 5e-324 is not). A call with w != 0 is path B's (WP6).
    for (i64 row = 0; row < n_rows; ++row) {
        const double* uvw = call.uvw + 3 * row;
        if (used(uvw) && uvw[2] != 0.0) return Path::declined;
    }
    // (2) u and v in turns, as the reference scales them, finite and in range.
    std::vector<double> scale(static_cast<std::size_t>(call.n_vis_chan));
    for (i64 c = 0; c < call.n_vis_chan; ++c) scale[c] = call.frequency[c] / kSpeedOfLight;
    for (i64 row = 0; row < n_rows; ++row) {
        const double* uvw = call.uvw + 3 * row;
        if (!used(uvw)) continue;
        for (i64 c = 0; c < call.n_vis_chan; ++c) {
            const double u = uvw[0] * scale[c];
            const double v = uvw[1] * scale[c];
            if (!(std::abs(u) < kMaxTurns && std::abs(v) < kMaxTurns)) return Path::declined;
        }
    }
    // (3) Every referenced value of the planes with samples finite and bounded,
    // and (4) the cost rule over the call's samples.
    const auto* model = static_cast<const ModelT*>(call.model);
    const i64 plane_size = call.m_pol * call.n_l * call.n_m;
    const Samples samples(call);
    std::vector<Plane> planes;
    double cost = 0.0, reference_cost = 0.0;
    for (i64 a_time = 0; a_time < call.m_time; ++a_time) {
        for (i64 a_chan = 0; a_chan < call.m_chan; ++a_chan) {
            const i64 n_samples = samples.n_rows(a_time) * samples.n_channels(a_chan);
            if (n_samples == 0) continue;
            Plane plane{a_time, a_chan, n_samples, {}};
            const ModelT* values = model + (a_time * call.m_chan + a_chan) * plane_size;
            if (!analyse_plane(values, call.n_l, call.n_m, call.pol_map, call.n_pol, call.m,
                               plane.layout))
                return Path::declined;
            const PlaneLayout& layout = plane.layout;
            const auto samples_of_plane = static_cast<double>(n_samples);
            cost += samples_of_plane * static_cast<double>(layout.components.size()) *
                    static_cast<double>(layout.rows.size() * layout.column_a.size());
            reference_cost += samples_of_plane * static_cast<double>(layout.nonzero_pixels);
            planes.push_back(std::move(plane));
        }
    }
    if (!(cost <= kAlpha * reference_cost)) return Path::declined;
    if (call.n_pol == 0) return Path::separable;  // nothing to write

    // Path S, one plane at a time: pack its weights (one item per kept row),
    // then compute its blocks of samples.
    const Kernels& routines = kernels();
    auto n_blocks = [](const Plane& plane) {
        return round_up(plane.n_samples, kBlockSamples) / kBlockSamples;
    };
    int workers = 1;
    for (const Plane& plane : planes) {
        const auto n_kept_rows = static_cast<i64>(plane.layout.rows.size());
        workers = std::max({workers, resolve_threads(call.threads, n_blocks(plane)),
                            resolve_threads(call.threads, n_kept_rows)});
    }
    std::vector<Scratch> scratch(static_cast<std::size_t>(workers));
    Buffer p_buffer, q_buffer;
    for (const Plane& plane : planes) {
        const PlaneLayout& layout = plane.layout;
        const auto n_packed = static_cast<i64>(layout.rows.size() * layout.components.size());
        const auto n_columns = static_cast<i64>(layout.column_a.size());
        const i64 packed_size = round_up(n_packed, kTileRows) * n_columns;
        double* p = p_buffer.get(packed_size);
        double* q = q_buffer.get(packed_size);
        if (n_packed > 0) {
            pack_plane(model + (plane.a_time * call.m_chan + plane.a_chan) * plane_size, call.n_l,
                       call.n_m, layout, call.l, call.m, call.threads, scratch, p, q);
        }
        parallel_for(resolve_threads(call.threads, n_blocks(plane)), n_blocks(plane),
                     [&](int worker, i64 block) {
                         compute_block<VisT>(call, samples, plane, scale.data(), p, q, block,
                                             routines, scratch[worker]);
                     });
    }
    return Path::separable;
}

template <typename ModelT>
Path predict_model(const Call& call) {
    return call.vis_type == Element::complex128 ? predict_typed<ModelT, std::complex<double>>(call)
                                                : predict_typed<ModelT, std::complex<float>>(call);
}

}  // namespace

Path predict(const Call& call) {
    switch (call.model_type) {
        case Element::float32:
            return predict_model<float>(call);
        case Element::float64:
            return predict_model<double>(call);
        case Element::complex64:
            return predict_model<std::complex<float>>(call);
        case Element::complex128:
            return predict_model<std::complex<double>>(call);
    }
    return Path::declined;
}

const Kernels& kernels() {
    static const Kernels selected = [] {
        const char* value = std::getenv("ASTROVIPER_KERNELS_ISA");
        const std::string_view isa = value != nullptr ? value : "";
        if (!isa.empty() && isa != "generic") {
            throw std::invalid_argument("ASTROVIPER_KERNELS_ISA=" + std::string(isa) +
                                        ": expected 'generic' (the baseline instruction set) "
                                        "or unset (the best one of the CPU)");
        }
#if defined(AV_HAVE_AVX2)
        if (isa.empty() && __builtin_cpu_supports("avx2")) return avx2_kernels();
#endif
        return generic_kernels();
    }();
    return selected;
}

}  // namespace dft_fast
