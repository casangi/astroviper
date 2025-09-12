/*
 * Copyright (C) 1999,2000,2003
 * Associated Universities, Inc. Washington DC, USA.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 675 Massachusetts Ave, Cambridge,
 * MA 02139, USA.
 *
 * Correspondence concerning AIPS++ should be addressed as follows:
 *        Internet email: casa-feedback@nrao.edu.
 *        Postal address: AIPS++ Project Office
 *                        National Radio Astronomy Observatory
 *                        520 Edgemont Road
 *                        Charlottesville, VA 22903-2475 USA
 *
 * Converted from Fortran to C++ for modern usage
 * Templated version to support both float and double precision
 */

#include "../include/hclean.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <type_traits>

namespace hclean {

/**
 * Templated function to find minimum and maximum values in a 3D image array
 * 
 * @param limagestep 3D array [pol][ny][nx] input dirty image
 * @param domask flag indicating if mask is present (0 = no mask)
 * @param lmask 2D mask array [ny][nx] (used if domask != 0)
 * @param nx width of images
 * @param ny height of images  
 * @param npol number of polarizations
 * @param fmin output minimum value found
 * @param fmax output maximum value found
 */
template<typename T>
void maximg(const T* limagestep, int domask, const T* lmask, 
           int nx, int ny, int npol, T& fmin, T& fmax) {
    
    // Use appropriate large values for each type
    if constexpr (std::is_same_v<T, float>) {
        fmin = 1e20f;
        fmax = -1e20f;
    } else {
        fmin = 1e20;
        fmax = -1e20;
    }
    
    for (int pol = 0; pol < npol; ++pol) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                // Check mask condition
                T mask_threshold = static_cast<T>(0.5);
                if ((domask == 0) || (lmask[iy * nx + ix] > mask_threshold)) {
                    // Access 3D array: limagestep[pol][iy][ix] in Fortran becomes:
                    T wpeak = limagestep[pol * ny * nx + iy * nx + ix];
                    
                    if (wpeak > fmax) {
                        fmax = wpeak;
                    }
                    if (wpeak < fmin) {
                        fmin = wpeak;
                    }
                }
            }
        }
    }
}

/**
 * Templated Hogbom CLEAN algorithm implementation
 *
 * @param limage 3D output model image (clean components) [pol][ny][nx]
 * @param limagestep 3D input dirty image / output residual [pol][ny][nx]
 * @param lpsf 2D point spread function [ny][nx]
 * @param domask flag indicating if mask is present (0 = no mask)
 * @param lmask 2D input mask image [ny][nx]
 * @param nx width of images
 * @param ny height of images
 * @param npol number of polarizations
 * @param xbeg starting x coordinate for clean box (0-based)
 * @param xend ending x coordinate for clean box (0-based, exclusive)
 * @param ybeg starting y coordinate for clean box (0-based)
 * @param yend ending y coordinate for clean box (0-based, exclusive)
 * @param niter maximum allowed iterations
 * @param siter starting iteration number
 * @param iter output: last iteration number reached
 * @param gain clean loop gain
 * @param thres flux cleaning threshold
 * @param cspeedup if > 0, adaptive threshold: thres * 2^(iter/cspeedup)
 * @param msgput callback function for status messages
 * @param stopnow callback function to check if stopping is requested
 */
template<typename T>
void clean(T* limage, T* limagestep, const T* lpsf,
           int domask, const T* lmask, int nx, int ny, int npol,
           int xbeg, int xend, int ybeg, int yend,
           int niter, int siter, int& iter, T gain, T thres,
           T cspeedup,
           std::function<void(int, int, int, int, int, T)> msgput,
           std::function<void(int&)> stopnow) {
    
    int maxiter = siter;
    int yes = 0;
    
    // Process each polarization
    for (int pol = 0; pol < npol; ++pol) {
        
        // Find peak in current polarization within clean box
        T maxval = static_cast<T>(0);
        int px = 0;  // Convert from Fortran 1-based to 0-based
        int py = 0;

        // Main iteration loop
        for (iter = siter; iter < niter; ++iter) {
            for (int iy = ybeg; iy < yend; ++iy) {
                for (int ix = xbeg; ix < xend; ++ix) {
                    // Check mask condition
                    T mask_threshold = static_cast<T>(0.5);
                    if ((domask == 0) || (lmask[iy * nx + ix] > mask_threshold)) {
                        T val = std::abs(limagestep[pol * ny * nx + iy * nx + ix]);
                        if (val > maxval) {
                            px = ix;
                            py = iy;  
                            maxval = val;
                        }
                    }
                }
            }
            
            // Get actual value (with sign) at peak location
            maxval = limagestep[pol * ny * nx + py * nx + px];
            
            // Calculate adaptive threshold if requested
            T cthres;
            T zero_val = static_cast<T>(0);
            T two_val = static_cast<T>(2);
            if (cspeedup > zero_val) {
                cthres = thres * std::pow(two_val, static_cast<T>(iter - siter) / cspeedup);
            } else {
                cthres = thres;
            }
            
            // Check convergence criteria
            if ((yes == 1) || (std::abs(maxval) < cthres)) {
                break;  // goto 200 equivalent
            }
            
            // Output progress information
            int cycle = std::max(1, (niter - siter) / 10);
            if ((iter == siter) || ((iter % cycle) == 1)) {
                msgput(npol, pol, iter, px, py, maxval);
                stopnow(yes);
            }
            
            if (yes == 1) {
                break;  // goto 200 equivalent
            }
            
            // Calculate bounds for PSF subtraction
            // Note: In Fortran code, PSF appears to be centered, so we use nx/2, ny/2
            int x1 = std::max(0, px - nx / 2);
            int y1 = std::max(0, py - ny / 2);
            int x2 = std::min(nx - 1, px + nx / 2 - 1);
            int y2 = std::min(ny - 1, py + ny / 2 - 1);
            
            // Add component to model and subtract from residual
            T pv = gain * maxval;
            limage[pol * ny * nx + py * nx + px] += pv;
            
            // Subtract scaled PSF from residual
            for (int iy = y1; iy <= y2; ++iy) {
                for (int ix = x1; ix <= x2; ++ix) {
                    // PSF indexing: convert from Fortran centered indexing
                    // Fortran: lpsf(nx/2+ix-px+1, ny/2+iy-py+1)
                    // C++: lpsf[(ny/2 + iy - py) * nx + (nx/2 + ix - px)]
                    int psf_x = nx / 2 + ix - px;
                    int psf_y = ny / 2 + iy - py;

                    // Check if psf_x and psf_y are within bounds
                    if (psf_x < 0 || psf_x >= nx || psf_y < 0 || psf_y >= ny) {
                        continue; // Skip out-of-bounds
                    }
                    
                    limagestep[pol * ny * nx + iy * nx + ix] -= 
                        pv * lpsf[psf_y * nx + psf_x];
                }
            }
        }
        
        // Output final status for this polarization  
        if (iter > siter) {
            msgput(npol, pol, iter, px, py, maxval);
        }
        
        maxiter = std::max(iter, maxiter);
    }
    
    // Set final iteration count
    iter = maxiter;
    if (iter > niter) {
        iter = niter;
    }
}

// Explicit template instantiations for float and double
template void maximg<float>(const float* limagestep, int domask, const float* lmask, 
                           int nx, int ny, int npol, float& fmin, float& fmax);

template void maximg<double>(const double* limagestep, int domask, const double* lmask, 
                            int nx, int ny, int npol, double& fmin, double& fmax);

template void clean<float>(float* limage, float* limagestep, const float* lpsf,
                          int domask, const float* lmask, int nx, int ny, int npol,
                          int xbeg, int xend, int ybeg, int yend,
                          int niter, int siter, int& iter, float gain, float thres,
                          float cspeedup,
                          std::function<void(int, int, int, int, int, float)> msgput,
                          std::function<void(int&)> stopnow);

template void clean<double>(double* limage, double* limagestep, const double* lpsf,
                           int domask, const double* lmask, int nx, int ny, int npol,
                           int xbeg, int xend, int ybeg, int yend,
                           int niter, int siter, int& iter, double gain, double thres,
                           double cspeedup,
                           std::function<void(int, int, int, int, int, double)> msgput,
                           std::function<void(int&)> stopnow);

} // namespace hclean
