#pragma once

#include <functional>

namespace hclean {

/**
 * Templated function to find minimum and maximum values in a 2D image array.
 * The mask is a bool array; pixels with mask == true are considered.
 */
template<typename T>
void maximg(const T* limagestep, int domask, const bool* lmask,
           int nx, int ny, T& fmin, T& fmax);

/**
 * Templated Hogbom CLEAN algorithm implementation on a single 2D plane.
 * The mask is a bool array; pixels with mask == true are considered in the
 * peak search.
 */
template<typename T>
void clean(T* limage, T* limagestep, const T* lpsf,
           int domask, const bool* lmask, int nx, int ny,
           int xbeg, int xend, int ybeg, int yend,
           int niter, int siter, int& iter, T gain, T thres,
           T cspeedup,
           std::function<void(int, int, int, T)> msgput,
           std::function<void(int&)> stopnow);

/**
 * Templated Hogbom CLEAN over a full (nt, nf, np_img, ny, nx) image cube,
 * parallelized over (time, frequency, polarization) planes with std::thread.
 *
 * The residual_cube and model_cube buffers are modified in place: the
 * residual is updated as CLEAN components are subtracted, and the model
 * accumulates the found components. No intermediate copies of the plane
 * buffers are made; threads operate on disjoint (ny*nx) regions of the
 * supplied C-contiguous memory, so no inter-plane synchronization is
 * required beyond work-dispatch.
 *
 * The PSF cube may be provided with np_psf == np_img (one PSF per image
 * polarization) or np_psf == 1 (Stokes-I-only PSF broadcast across all
 * image polarizations).
 *
 * No Python-callable progress/stop callbacks are accepted: with multiple
 * worker threads these would require GIL-aware synchronization that
 * degrades throughput. Per-plane logging, if desired, should be done on
 * the Python side after the call returns.
 *
 * @param residual_cube 5D in-place residual buffer [nt, nf, np_img, ny, nx]
 * @param model_cube 5D in-place model buffer [nt, nf, np_img, ny, nx]
 * @param psf_cube 5D PSF buffer [nt, nf, np_psf, ny, nx], read-only
 * @param domask 0 if mask_cube is absent, 1 if present
 * @param mask_cube 5D mask buffer [nt, nf, np_img, ny, nx] or nullptr
 * @param nt number of time planes
 * @param nf number of frequency planes
 * @param np_img number of image polarization planes
 * @param np_psf number of PSF polarization planes (must be np_img or 1)
 * @param ny image height
 * @param nx image width
 * @param xbeg,xend,ybeg,yend per-plane clean box (0-based, exclusive upper)
 * @param niter maximum iterations per plane
 * @param gain clean loop gain
 * @param thres flux cleaning threshold
 * @param cspeedup adaptive threshold speedup (0 disables)
 * @param num_threads number of worker threads; clamped to [1, nplanes]
 * @param iter_out output iterations performed per plane, flat
 *                 array of length nt*nf*np_img in (t, f, p) C-order
 */
template<typename T>
void clean_cube(T* residual_cube, T* model_cube, const T* psf_cube,
                int domask, const bool* mask_cube,
                int nt, int nf, int np_img, int np_psf,
                int ny, int nx,
                int xbeg, int xend, int ybeg, int yend,
                int niter, T gain, T thres, T cspeedup,
                int num_threads, int* iter_out);

// Explicit template instantiation declarations
extern template void maximg<float>(const float* limagestep, int domask, const bool* lmask,
                                  int nx, int ny, float& fmin, float& fmax);

extern template void maximg<double>(const double* limagestep, int domask, const bool* lmask,
                                   int nx, int ny, double& fmin, double& fmax);

extern template void clean<float>(float* limage, float* limagestep, const float* lpsf,
                                 int domask, const bool* lmask, int nx, int ny,
                                 int xbeg, int xend, int ybeg, int yend,
                                 int niter, int siter, int& iter, float gain, float thres,
                                 float cspeedup,
                                 std::function<void(int, int, int, float)> msgput,
                                 std::function<void(int&)> stopnow);

extern template void clean<double>(double* limage, double* limagestep, const double* lpsf,
                                  int domask, const bool* lmask, int nx, int ny,
                                  int xbeg, int xend, int ybeg, int yend,
                                  int niter, int siter, int& iter, double gain, double thres,
                                  double cspeedup,
                                  std::function<void(int, int, int, double)> msgput,
                                  std::function<void(int&)> stopnow);

extern template void clean_cube<float>(float* residual_cube, float* model_cube,
                                       const float* psf_cube, int domask,
                                       const bool* mask_cube,
                                       int nt, int nf, int np_img, int np_psf,
                                       int ny, int nx,
                                       int xbeg, int xend, int ybeg, int yend,
                                       int niter, float gain, float thres, float cspeedup,
                                       int num_threads, int* iter_out);

extern template void clean_cube<double>(double* residual_cube, double* model_cube,
                                        const double* psf_cube, int domask,
                                        const bool* mask_cube,
                                        int nt, int nf, int np_img, int np_psf,
                                        int ny, int nx,
                                        int xbeg, int xend, int ybeg, int yend,
                                        int niter, double gain, double thres, double cspeedup,
                                        int num_threads, int* iter_out);

} // namespace hclean
