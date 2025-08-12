#pragma once

#include <vector>
#include <tuple>
#include <cstddef>

namespace hogbom {

/**
 * @brief Structure to hold CLEAN algorithm parameters
 */
struct CleanParams {
    double gain = 0.1;        ///< Loop gain (fraction of peak to subtract)
    double threshold = 0.0;   ///< Cleaning threshold
    int max_iter = 100;       ///< Maximum number of iterations
    int x_begin = 0;          ///< Clean window start X
    int x_end = -1;           ///< Clean window end X (-1 = full width)
    int y_begin = 0;          ///< Clean window start Y  
    int y_end = -1;           ///< Clean window end Y (-1 = full height)
};

/**
 * @brief Structure to hold CLEAN results
 */
struct CleanResults {
    std::vector<double> component_flux;  ///< Flux of clean components
    std::vector<int> component_x;        ///< X positions of clean components
    std::vector<int> component_y;        ///< Y positions of clean components
    int iterations_performed;            ///< Number of iterations performed
    double final_peak;                   ///< Final peak value in residual
    double total_flux_cleaned;           ///< Total flux cleaned
};

/**
 * @brief Hogbom CLEAN algorithm implementation
 * 
 * @param dirty_map Input dirty map (modified in-place)
 * @param nx Width of dirty map
 * @param ny Height of dirty map
 * @param beam Clean beam (PSF)
 * @param beam_nx Width of beam
 * @param beam_ny Height of beam
 * @param params Cleaning parameters
 * @return CleanResults containing component information and statistics
 */
CleanResults hogbom_clean(
    double* dirty_map,
    int nx, int ny,
    const double* beam,
    int beam_nx, int beam_ny,
    const CleanParams& params = CleanParams{}
);

/**
 * @brief Multi-polarization Hogbom CLEAN algorithm
 * 
 * @param dirty_maps Array of dirty maps (npol x ny x nx), modified in-place
 * @param nx Width of dirty maps
 * @param ny Height of dirty maps  
 * @param npol Number of polarizations
 * @param beam Clean beam (PSF) - shared across polarizations
 * @param beam_nx Width of beam
 * @param beam_ny Height of beam
 * @param params Cleaning parameters
 * @return CleanResults containing component information and statistics
 */
CleanResults hogbom_clean_multipol(
    double* dirty_maps,
    int nx, int ny, int npol,
    const double* beam,
    int beam_nx, int beam_ny,
    const CleanParams& params = CleanParams{}
);

/**
 * @brief Find peak pixel in a 2D array within specified bounds
 * 
 * @param data Input data array
 * @param nx Width of data
 * @param ny Height of data
 * @param x_begin Start X coordinate  
 * @param x_end End X coordinate
 * @param y_begin Start Y coordinate
 * @param y_end End Y coordinate
 * @return Tuple of (peak_value, peak_x, peak_y)
 */
std::tuple<double, int, int> find_peak(
    const double* data,
    int nx, int ny,
    int x_begin, int x_end,
    int y_begin, int y_end
);

/**
 * @brief Subtract scaled beam from dirty map at specified location
 * 
 * @param dirty_map Dirty map to modify
 * @param nx Width of dirty map
 * @param ny Height of dirty map
 * @param beam Beam pattern
 * @param beam_nx Width of beam
 * @param beam_ny Height of beam
 * @param peak_x X location of peak
 * @param peak_y Y location of peak
 * @param flux_scale Scaling factor (gain * peak_flux)
 */
void subtract_beam(
    double* dirty_map,
    int nx, int ny,
    const double* beam,
    int beam_nx, int beam_ny,
    int peak_x, int peak_y,
    double flux_scale
);

} // namespace hogbom