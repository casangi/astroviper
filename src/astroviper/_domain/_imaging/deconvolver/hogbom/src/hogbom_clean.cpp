#include "hogbom_clean.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace hogbom {

std::tuple<double, int, int> find_peak(
    const double* data,
    int nx, int ny,
    int x_begin, int x_end,
    int y_begin, int y_end
) {
    if (x_end == -1) x_end = nx;
    if (y_end == -1) y_end = ny;
    
    // Validate bounds
    x_begin = std::max(0, x_begin);
    y_begin = std::max(0, y_begin);
    x_end = std::min(nx, x_end);
    y_end = std::min(ny, y_end);
    
    if (x_begin >= x_end || y_begin >= y_end) {
        throw std::invalid_argument("Invalid clean window bounds");
    }
    
    double peak_val = -std::numeric_limits<double>::infinity();
    int peak_x = x_begin;
    int peak_y = y_begin;
    
    for (int y = y_begin; y < y_end; ++y) {
        for (int x = x_begin; x < x_end; ++x) {
            double val = std::abs(data[y * nx + x]);
            if (val > peak_val) {
                peak_val = val;
                peak_x = x;
                peak_y = y;
            }
        }
    }
    
    // Return actual value (with sign) at peak location
    return std::make_tuple(data[peak_y * nx + peak_x], peak_x, peak_y);
}

void subtract_beam(
    double* dirty_map,
    int nx, int ny,
    const double* beam,
    int beam_nx, int beam_ny,
    int peak_x, int peak_y,
    double flux_scale
) {
    // Calculate beam center offsets
    int beam_center_x = beam_nx / 2;
    int beam_center_y = beam_ny / 2;
    
    // Calculate bounds for beam subtraction
    int start_x = std::max(0, peak_x - beam_center_x);
    int end_x = std::min(nx, peak_x - beam_center_x + beam_nx);
    int start_y = std::max(0, peak_y - beam_center_y);
    int end_y = std::min(ny, peak_y - beam_center_y + beam_ny);
    
    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            // Calculate corresponding beam indices
            int beam_x = x - peak_x + beam_center_x;
            int beam_y = y - peak_y + beam_center_y;
            
            // Bounds check for beam
            if (beam_x >= 0 && beam_x < beam_nx && beam_y >= 0 && beam_y < beam_ny) {
                double beam_val = beam[beam_y * beam_nx + beam_x];
                dirty_map[y * nx + x] -= flux_scale * beam_val;
            }
        }
    }
}

CleanResults hogbom_clean(
    double* dirty_map,
    int nx, int ny,
    const double* beam,
    int beam_nx, int beam_ny,
    const CleanParams& params
) {
    CleanResults results;
    results.iterations_performed = 0;
    results.total_flux_cleaned = 0.0;
    
    // Set clean window bounds
    int x_begin = params.x_begin;
    int y_begin = params.y_begin;
    int x_end = (params.x_end == -1) ? nx : params.x_end;
    int y_end = (params.y_end == -1) ? ny : params.y_end;
    
    for (int iter = 0; iter < params.max_iter; ++iter) {
        // Find peak in clean window
        auto [peak_val, peak_x, peak_y] = find_peak(
            dirty_map, nx, ny, x_begin, x_end, y_begin, y_end
        );
        
        double abs_peak = std::abs(peak_val);
        results.final_peak = abs_peak;
        
        // Check convergence
        if (abs_peak <= params.threshold) {
            break;
        }
        
        // Calculate flux to clean
        double flux_to_clean = params.gain * peak_val;
        
        // Store component
        results.component_flux.push_back(flux_to_clean);
        results.component_x.push_back(peak_x);
        results.component_y.push_back(peak_y);
        results.total_flux_cleaned += std::abs(flux_to_clean);
        
        // Subtract scaled beam from dirty map
        subtract_beam(dirty_map, nx, ny, beam, beam_nx, beam_ny, 
                     peak_x, peak_y, flux_to_clean);
        
        results.iterations_performed++;
    }
    
    return results;
}

CleanResults hogbom_clean_multipol(
    double* dirty_maps,
    int nx, int ny, int npol,
    const double* beam,
    int beam_nx, int beam_ny,
    const CleanParams& params
) {
    CleanResults results;
    results.iterations_performed = 0;
    results.total_flux_cleaned = 0.0;
    
    // Set clean window bounds
    int x_begin = params.x_begin;
    int y_begin = params.y_begin;
    int x_end = (params.x_end == -1) ? nx : params.x_end;
    int y_end = (params.y_end == -1) ? ny : params.y_end;
    
    const int map_size = nx * ny;
    
    for (int iter = 0; iter < params.max_iter; ++iter) {
        // Find peak across all polarizations (using Stokes I or first pol)
        double* stokes_i_map = dirty_maps; // Assume first polarization is Stokes I
        auto [peak_val, peak_x, peak_y] = find_peak(
            stokes_i_map, nx, ny, x_begin, x_end, y_begin, y_end
        );
        
        double abs_peak = std::abs(peak_val);
        results.final_peak = abs_peak;
        
        // Check convergence
        if (abs_peak <= params.threshold) {
            break;
        }
        
        // Calculate flux to clean
        double flux_to_clean = params.gain * peak_val;
        
        // Store component
        results.component_flux.push_back(flux_to_clean);
        results.component_x.push_back(peak_x);
        results.component_y.push_back(peak_y);
        results.total_flux_cleaned += std::abs(flux_to_clean);
        
        // Subtract scaled beam from all polarizations
        for (int pol = 0; pol < npol; ++pol) {
            double* pol_map = dirty_maps + pol * map_size;
            
            // For polarizations other than Stokes I, we might want to
            // scale the cleaning differently, but for now clean all equally
            subtract_beam(pol_map, nx, ny, beam, beam_nx, beam_ny,
                         peak_x, peak_y, flux_to_clean);
        }
        
        results.iterations_performed++;
    }
    
    return results;
}

} // namespace hogbom