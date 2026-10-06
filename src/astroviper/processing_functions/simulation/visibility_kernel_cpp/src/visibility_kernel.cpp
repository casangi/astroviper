#include "../include/visibility_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace visibility_kernel {

namespace {

constexpr double kSpeedOfLight = 299792458.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kArcminToRad = kPi / (180.0 * 60.0);
constexpr double kCasaAiryTwiddle =
    (180.0 * 7.016 * kSpeedOfLight) / ((kPi * kPi) * 1e9 * 1.566 * 24.5);
constexpr int kCasaAiryNSample = 10000;

// row-wise 4x4 Mueller element -> (Jones 1 index, Jones 2 index)
constexpr int kMapMuellerToJones[16][2] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3},
    {2, 0}, {2, 1}, {3, 0}, {3, 1}, {2, 2}, {2, 3}, {3, 2}, {3, 3}};

// --- Cephes j1 (identical to scipy.special.j1) ------------------------------
const double RP[4] = {-8.99971225705559398224E8, 4.52228297998194034323E11,
                      -7.27494245221818276015E13, 3.68295732863852883286E15};
const double RQ[8] = {6.20836478118054335476E2,  2.56987256757748830383E5,
                      8.35146791431949253037E7,  2.21511595479792499675E10,
                      4.74914122079991414898E12, 7.84369607876235854894E14,
                      8.95222336184627338078E16, 5.32278620332680085395E18};
const double PP[7] = {7.62125616208173112003E-4, 7.31397056940917570436E-2,
                      1.12719608129684925192E0,  5.11207951146807644818E0,
                      8.42404590141772420927E0,  5.21451598682361504063E0,
                      1.00000000000000000254E0};
const double PQ[7] = {5.71323128072548699714E-4, 6.88455908754495404082E-2,
                      1.10514232634061696926E0,  5.07386386128601488557E0,
                      8.39985554327604159757E0,  5.20982848682361821619E0,
                      9.99999999999999997461E-1};
const double QP[8] = {5.10862594750176621635E-2, 4.98213872951233449420E0,
                      7.58238284132545283818E1,  3.66779609360150777800E2,
                      7.10856304998926107277E2,  5.97489612400613639965E2,
                      2.11688757100572135698E2,  2.52070205858023719784E1};
const double QQ[7] = {7.42373277035675149943E1, 1.05644886038262816351E3,
                      4.98641058337653607651E3, 9.56231892404756170795E3,
                      7.99704160447350683650E3, 2.82619278517639096600E3,
                      3.36093607810698293419E2};
constexpr double Z1 = 1.46819706421238932572E1;
constexpr double Z2 = 4.92184563216946036703E1;
constexpr double THPIO4 = 2.35619449019234492885;
constexpr double SQ2OPI = 7.9788456080286535587989E-1;

inline double polevl(double x, const double* c, int n) {
    double a = c[0];
    for (int i = 1; i <= n; ++i) a = a * x + c[i];
    return a;
}
inline double p1evl(double x, const double* c, int n) {
    double a = x + c[0];
    for (int i = 1; i < n; ++i) a = a * x + c[i];
    return a;
}

// --- analytic responses -----------------------------------------------------
inline double airy_pattern(double r, double dish, double blockage) {
    if (r == 0.0) return 1.0;
    if (blockage == 0.0) return 2.0 * bessel_j1(r) / r;
    const double e = blockage / dish;
    return (2.0 * bessel_j1(r) / r - 2.0 * e * bessel_j1(r * e) / r) / (1.0 - e * e);
}

inline double casa_airy_pattern(double r, double dish, double blockage) {
    if (r == 0.0) return 1.0;
    if (blockage == 0.0) return 2.0 * bessel_j1(r) / r;
    const double area_ratio = (dish / blockage) * (dish / blockage);
    const double length_ratio = dish / blockage;
    return (area_ratio * 2.0 * bessel_j1(r) / r -
            2.0 * bessel_j1(r * length_ratio) / (r * length_ratio)) /
           (area_ratio - 1.0);
}

inline double analytic_response(const BeamModel& bm, double l, double m, double freq) {
    const double rho = std::sqrt(l * l + m * m);
    if (bm.func == 0) return 1.0;  // "none"
    const double k = 2.0 * kPi * freq / kSpeedOfLight;
    const double aperture = bm.dish_diameter / 2.0;
    if (bm.func == 1) {  // casa_airy, 10000-sample lookup quantisation
        const double r_max = bm.max_rad_1GHz / (freq / 1e9);
        const double r_inc = r_max / (kCasaAiryNSample - 1);
        const double r = (std::trunc(rho / r_inc) * r_inc) * aperture * k * kCasaAiryTwiddle;
        return casa_airy_pattern(r, bm.dish_diameter, bm.blockage_diameter);
    }
    const double r = rho * k * aperture;  // airy
    return airy_pattern(r, bm.dish_diameter, bm.blockage_diameter);
}

inline std::int64_t nearest_index(const double* values, std::int64_t n, double x) {
    std::int64_t best = 0;
    double best_d = std::abs(x - values[0]);
    for (std::int64_t i = 1; i < n; ++i) {
        const double d = std::abs(x - values[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

inline double wrapped_angle_difference(double a, double b) {
    return std::abs(std::fmod(a - b + kPi, 2.0 * kPi) - kPi);
}

inline std::int64_t nearest_angle_index(const double* values, std::int64_t n, double x) {
    std::int64_t best = 0;
    double best_d = wrapped_angle_difference(x, values[0]);
    for (std::int64_t i = 1; i < n; ++i) {
        const double d = wrapped_angle_difference(x, values[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

inline double clampd(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

/**
 * Sample the Jones vector [J_pp, J_pq, J_qp, J_qq] of one beam model at (l, m)
 * for one frequency.  Identical to the Python sample_jones().
 */
inline void sample_jones(const BeamModel& bm, double l, double m, double freq,
                         double parallactic_angle, std::complex<double> jones[4]) {
    jones[0] = jones[1] = jones[2] = jones[3] = 0.0;
    const double rho = std::sqrt(l * l + m * m);
    if (rho >= bm.max_rad_1GHz / (freq / 1e9)) return;  // outside the beam cut

    if (bm.kind == 0) {
        const double resp = analytic_response(bm, l, m, freq);
        jones[0] = resp;
        jones[3] = resp;
    } else if (bm.kind == 1) {
        const std::int64_t i_chan = nearest_index(bm.poly_frequency, bm.n_poly_frequency, freq);
        const double* coef = bm.poly_coefficients + i_chan * bm.n_poly_coefficient;
        const double r_inc = (bm.max_rad_1GHz / (freq / 1e9)) / (kCasaAiryNSample - 1);
        const double r = (std::trunc(rho / r_inc) * r_inc) * (freq / 1e9) / kArcminToRad;
        double beam = 0.0;
        for (std::int64_t i = 0; i < bm.n_poly_coefficient; ++i) beam += coef[i] * std::pow(r, 2.0 * i);
        beam = beam < 0.0 ? 0.0 : std::sqrt(beam);
        jones[0] = beam;
        jones[3] = beam;
    } else {
        const std::int64_t i_pa = nearest_angle_index(bm.jones_parallactic_angle, bm.n_pa, parallactic_angle);
        const std::int64_t i_chan = nearest_index(bm.jones_frequency, bm.n_jones_frequency, freq);
        const double scale = freq / bm.jones_frequency[i_chan];
        const double delta_pa = parallactic_angle - bm.jones_parallactic_angle[i_pa];
        const double c = std::cos(delta_pa), s = std::sin(delta_pa);
        const double ls = l * scale, ms = m * scale;
        const double l_rot = c * ls + s * ms;
        const double m_rot = -s * ls + c * ms;
        const double x = clampd(l_rot / bm.cell_size_l + (bm.n_l / 2), 0.0, double(bm.n_l - 1));
        const double y = clampd(m_rot / bm.cell_size_m + (bm.n_m / 2), 0.0, double(bm.n_m - 1));
        const std::int64_t x0 = static_cast<std::int64_t>(std::floor(x));
        const std::int64_t y0 = static_cast<std::int64_t>(std::floor(y));
        const std::int64_t x1 = std::min(x0 + 1, bm.n_l - 1);
        const std::int64_t y1 = std::min(y0 + 1, bm.n_m - 1);
        const double fx = x - x0, fy = y - y0;
        const std::int64_t plane = bm.n_l * bm.n_m;
        const std::complex<double>* base = bm.jones + (i_pa * bm.n_jones_frequency + i_chan) * bm.n_jones_pol * plane;
        for (std::int64_t ip = 0; ip < bm.n_jones_pol; ++ip) {
            const std::complex<double>* img = base + ip * plane;
            const std::complex<double> v =
                (1 - fx) * (1 - fy) * img[x0 * bm.n_m + y0] + (1 - fx) * fy * img[x0 * bm.n_m + y1] +
                fx * (1 - fy) * img[x1 * bm.n_m + y0] + fx * fy * img[x1 * bm.n_m + y1];
            jones[bm.jones_polarization_index[ip]] = v;
        }
    }
}

inline void sin_project(double ra_o, double dec_o, double ra, double dec, double& l, double& m) {
    const double d_ra = ra - ra_o;
    l = std::cos(dec) * std::sin(d_ra);
    m = std::sin(dec) * std::cos(dec_o) - std::cos(dec) * std::sin(dec_o) * std::cos(d_ra);
}

}  // namespace

double bessel_j1(double x) {
    if (x < 0.0) return -bessel_j1(-x);
    if (x <= 5.0) {
        const double z = x * x;
        double w = polevl(z, RP, 3) / p1evl(z, RQ, 8);
        w = w * x * (z - Z1) * (z - Z2);
        return w;
    }
    const double w = 5.0 / x;
    const double z = w * w;
    double p = polevl(z, PP, 6) / polevl(z, PQ, 6);
    const double q = polevl(z, QP, 7) / p1evl(z, QQ, 7);
    const double xn = x - THPIO4;
    p = p * std::cos(xn) - w * q * std::sin(xn);
    return p * SQ2OPI / std::sqrt(x);
}

void calculate_visibilities(
    std::complex<double>* visibility, const double* uvw, const std::int64_t* antenna1,
    const std::int64_t* antenna2, const double* frequency, const std::int64_t* polarization_index,
    const std::complex<double>* flux, const double* k_vector, const double* inverse_n,
    const double* source_ra_dec, const double* pointing_ra_dec, const std::int64_t* beam_model_map,
    const std::vector<BeamModel>& beam_models, const double* parallactic_angle,
    const std::int64_t* mueller_selection, std::int64_t n_time, std::int64_t n_baseline,
    std::int64_t n_frequency, std::int64_t n_polarization, std::int64_t n_source,
    std::int64_t n_flux_time, std::int64_t n_flux_frequency, std::int64_t n_source_time,
    std::int64_t n_pointing_time, std::int64_t n_pointing_antenna, std::int64_t n_antenna,
    std::int64_t n_mueller, int n_threads) {
    if (n_threads <= 0) n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads < 1) n_threads = 1;
    n_threads = static_cast<int>(std::min<std::int64_t>(n_threads, std::max<std::int64_t>(n_baseline, 1)));

    const std::int64_t f_flux_time = (n_flux_time == 1) ? n_time : 1;
    const std::int64_t f_flux_freq = (n_flux_frequency == 1) ? n_frequency : 1;
    const std::int64_t f_source_time = (n_source_time == 1) ? n_time : 1;
    const std::int64_t f_pointing_time = (n_pointing_time == 1) ? n_time : 1;
    const std::int64_t f_pointing_ant = (n_pointing_antenna == 1) ? n_antenna : 1;

    auto worker = [&](std::int64_t b_begin, std::int64_t b_end) {
        std::vector<std::complex<double>> jones_all(static_cast<std::size_t>(n_antenna * n_frequency * 4));
        std::vector<std::complex<double>> flux_scaled(4);
        std::vector<double> lm_ant(static_cast<std::size_t>(n_antenna * 2));
        for (std::int64_t t = 0; t < n_time; ++t) {
            const double pa = parallactic_angle[t];
            for (std::int64_t s = 0; s < n_source; ++s) {
                const double ra = source_ra_dec[((t / f_source_time) * n_source + s) * 2 + 0];
                const double dec = source_ra_dec[((t / f_source_time) * n_source + s) * 2 + 1];
                // direction of the source relative to every antenna's pointing
                for (std::int64_t a = 0; a < n_antenna; ++a) {
                    const double* pt = pointing_ra_dec +
                        ((t / f_pointing_time) * n_pointing_antenna + (a / f_pointing_ant)) * 2;
                    sin_project(pt[0], pt[1], ra, dec, lm_ant[2 * a], lm_ant[2 * a + 1]);
                }
                // Jones of every antenna at every frequency
                for (std::int64_t a = 0; a < n_antenna; ++a) {
                    const BeamModel& bm = beam_models[static_cast<std::size_t>(beam_model_map[a])];
                    for (std::int64_t c = 0; c < n_frequency; ++c) {
                        sample_jones(bm, lm_ant[2 * a], lm_ant[2 * a + 1], frequency[c], pa,
                                     &jones_all[(a * n_frequency + c) * 4]);
                    }
                }
                const double* k = k_vector + (t * n_source + s) * 3;
                const double inv_n = inverse_n[t * n_source + s];
                for (std::int64_t b = b_begin; b < b_end; ++b) {
                    const double* uvw_b = uvw + (t * n_baseline + b) * 3;
                    const double phase = 2.0 * kPi * (uvw_b[0] * k[0] + uvw_b[1] * k[1] + uvw_b[2] * k[2]);
                    const std::int64_t a1 = antenna1[b], a2 = antenna2[b];
                    for (std::int64_t c = 0; c < n_frequency; ++c) {
                        const std::complex<double>* j1 = &jones_all[(a1 * n_frequency + c) * 4];
                        const std::complex<double>* j2 = &jones_all[(a2 * n_frequency + c) * 4];
                        const std::complex<double>* f =
                            flux + ((s * n_flux_time + t / f_flux_time) * n_flux_frequency + c / f_flux_freq) * 4;
                        flux_scaled[0] = flux_scaled[1] = flux_scaled[2] = flux_scaled[3] = 0.0;
                        for (std::int64_t im = 0; im < n_mueller; ++im) {
                            const int e = static_cast<int>(mueller_selection[im]);
                            const int ja = kMapMuellerToJones[e][0], jb = kMapMuellerToJones[e][1];
                            flux_scaled[e / 4] += j1[ja] * std::conj(j2[jb]) * f[e % 4];
                        }
                        const double arg = phase * frequency[c] / kSpeedOfLight;
                        const std::complex<double> fringe(std::cos(arg) * inv_n, std::sin(arg) * inv_n);
                        std::complex<double>* out = visibility + ((t * n_baseline + b) * n_frequency + c) * n_polarization;
                        for (std::int64_t p = 0; p < n_polarization; ++p) {
                            out[p] += flux_scaled[polarization_index[p]] * fringe;
                        }
                    }
                }
            }
        }
    };

    if (n_threads == 1) {
        worker(0, n_baseline);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    const std::int64_t per = (n_baseline + n_threads - 1) / n_threads;
    for (int i = 0; i < n_threads; ++i) {
        const std::int64_t b0 = i * per;
        const std::int64_t b1 = std::min(n_baseline, b0 + per);
        if (b0 >= b1) break;
        threads.emplace_back(worker, b0, b1);
    }
    for (auto& th : threads) th.join();
}

}  // namespace visibility_kernel
