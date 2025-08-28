#pragma once

#include <functional>

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
           int nx, int ny, int npol, T& fmin, T& fmax);

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
           std::function<void(int&)> stopnow);

// Explicit template instantiation declarations
extern template void maximg<float>(const float* limagestep, int domask, const float* lmask, 
                                  int nx, int ny, int npol, float& fmin, float& fmax);

extern template void maximg<double>(const double* limagestep, int domask, const double* lmask, 
                                   int nx, int ny, int npol, double& fmin, double& fmax);

extern template void clean<float>(float* limage, float* limagestep, const float* lpsf,
                                 int domask, const float* lmask, int nx, int ny, int npol,
                                 int xbeg, int xend, int ybeg, int yend,
                                 int niter, int siter, int& iter, float gain, float thres,
                                 float cspeedup,
                                 std::function<void(int, int, int, int, int, float)> msgput,
                                 std::function<void(int&)> stopnow);

extern template void clean<double>(double* limage, double* limagestep, const double* lpsf,
                                  int domask, const double* lmask, int nx, int ny, int npol,
                                  int xbeg, int xend, int ybeg, int yend,
                                  int niter, int siter, int& iter, double gain, double thres,
                                  double cspeedup,
                                  std::function<void(int, int, int, int, int, double)> msgput,
                                  std::function<void(int&)> stopnow);

} // namespace hclean
