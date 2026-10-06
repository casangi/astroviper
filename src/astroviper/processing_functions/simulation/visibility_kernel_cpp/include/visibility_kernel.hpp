#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace visibility_kernel {

/**
 * Flat description of one antenna beam model (see Python pack_beam_models()).
 *
 * kind: 0 = analytic, 1 = polynomial, 2 = Jones image.
 * All pointers refer to Python-owned, C-contiguous, read-only buffers.
 */
struct BeamModel {
    int kind = 0;
    double max_rad_1GHz = 0.0;

    // analytic: func 0 = none, 1 = casa_airy, 2 = airy
    int func = 0;
    double dish_diameter = 0.0;
    double blockage_diameter = 0.0;

    // polynomial: coefficients[n_frequency, n_coefficient]
    const double* poly_frequency = nullptr;
    const double* poly_coefficients = nullptr;
    std::int64_t n_poly_frequency = 0;
    std::int64_t n_poly_coefficient = 0;

    // Jones image: jones[n_pa, n_frequency, n_pol, n_l, n_m]
    const std::complex<double>* jones = nullptr;
    const double* jones_parallactic_angle = nullptr;
    const double* jones_frequency = nullptr;
    const std::int64_t* jones_polarization_index = nullptr;
    std::int64_t n_pa = 0;
    std::int64_t n_jones_frequency = 0;
    std::int64_t n_jones_pol = 0;
    std::int64_t n_l = 0;
    std::int64_t n_m = 0;
    double cell_size_l = 0.0;
    double cell_size_m = 0.0;
};

/**
 * Accumulate point-source visibilities into `visibility` (in place).
 *
 * visibility[n_time, n_baseline, n_frequency, n_polarization] (complex128, writeable)
 * uvw[n_time, n_baseline, 3]
 * antenna1/antenna2[n_baseline]
 * frequency[n_frequency]
 * polarization_index[n_polarization]            (0..3)
 * flux[n_source, n_flux_time, n_flux_frequency, 4]   (n_flux_* is 1 or the full size)
 * k_vector[n_time, n_source, 3]                 (rotation @ lmn_rot per time/source)
 * inverse_n[n_time, n_source]                   (1 / (1 - lmn_rot[2]))
 * source_ra_dec[n_source_time, n_source, 2]     (n_source_time is 1 or n_time)
 * pointing_ra_dec[n_pointing_time, n_pointing_antenna, 2]  (1 or full along each axis)
 * beam_model_map[n_antenna]
 * parallactic_angle[n_time]
 * mueller_selection[n_mueller]
 *
 * Threads split the baseline range; each thread samples the Jones vectors of
 * all antennas for every (time, source) itself, so no synchronization is needed.
 */
void calculate_visibilities(
    std::complex<double>* visibility,
    const double* uvw,
    const std::int64_t* antenna1,
    const std::int64_t* antenna2,
    const double* frequency,
    const std::int64_t* polarization_index,
    const std::complex<double>* flux,
    const double* k_vector,
    const double* inverse_n,
    const double* source_ra_dec,
    const double* pointing_ra_dec,
    const std::int64_t* beam_model_map,
    const std::vector<BeamModel>& beam_models,
    const double* parallactic_angle,
    const std::int64_t* mueller_selection,
    std::int64_t n_time,
    std::int64_t n_baseline,
    std::int64_t n_frequency,
    std::int64_t n_polarization,
    std::int64_t n_source,
    std::int64_t n_flux_time,
    std::int64_t n_flux_frequency,
    std::int64_t n_source_time,
    std::int64_t n_pointing_time,
    std::int64_t n_pointing_antenna,
    std::int64_t n_antenna,
    std::int64_t n_mueller,
    int n_threads);

/** Bessel function of the first kind of order one (Cephes algorithm, as used by SciPy). */
double bessel_j1(double x);

}  // namespace visibility_kernel
