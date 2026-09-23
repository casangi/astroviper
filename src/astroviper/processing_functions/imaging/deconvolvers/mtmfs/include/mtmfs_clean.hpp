#pragma once

// Multi-Term Multi-Frequency Synthesis (MTMFS) CLEAN -- stateless kernel.
//
// The model-update cycle behind CASA tclean's `deconvolver='mtmfs'`
// (SDAlgorithmMSMFS driving MultiTermMatrixCleaner::mtclean), ported from
// CASA's casacore-free stdcleaner/StdMultiTermCleaner and exposed as free
// functions in the style of deconvolvers/hogbom (hclean::clean).
//
// There is no engine object and no cached setup. Each function takes every
// input it needs, allocates its own derived working memory for the duration
// of the call, and frees it on return. Concurrent calls on distinct buffers
// are safe.
//
// Memory contract (astroviper AGENTS.md section 6): residual, model, PSF and
// mask are caller-owned. The kernel never copies them. Residual and model are
// updated in place; PSF and mask are read-only. No caller pointer is retained
// after return.
//
// Layout: row-major (ny, nx), pixel (ix, iy) at iy*nx + ix. Taylor stacks
// are (nterms, ny, nx); the PSF stack is (2*nterms-1, ny, nx).

#include <cstddef>
#include <vector>

namespace mtmfs {

enum StopCode : int {
    kStopMaxIter = 0,
    kStopThreshold = 1,
    kStopNothingToClean = 2,
    kStopDiverged = -1,
};

constexpr int kSingularHessian = -2;
constexpr int kMaxTaylorTerms = 16;

template <typename T>
struct CleanResult {
    int iterations = 0;
    int stop_code = kStopMaxIter;
    T peak_residual = 0;
    T model_flux = 0;
    int psf_support = 0;
    float small_scale_bias = 0.0f;
    std::vector<float> scales;
    std::vector<double> hessian;
    std::vector<double> inverse_hessian;
};

// Empty -> {0}; sorted ascending and unique; sizes larger than half the image
// are dropped (MultiTermMatrixCleaner::verifyScaleSizes).
std::vector<float> effective_scales(const std::vector<float>& scales, int nx, int ny);

// SDAlgorithmMSMFS::initializeDeconvolver: clamp to [-1, 1].
float clamp_small_scale_bias(float bias);

// Per-scale Taylor Hessians and inverses. `scales` is replaced by the
// effective list. hessian / inverse_hessian are (nscales, nterms, nterms)
// row-major. Returns 0 or kSingularHessian.
template <typename T>
int taylor_hessian(const T* psf, int nterms, int nx, int ny, std::vector<float>& scales,
                   float small_scale_bias, std::vector<double>& hessian,
                   std::vector<double>& inverse_hessian, int& psf_support);

// One MTMFS model-update cycle in place (MultiTermMatrixCleaner::mtclean).
//   residual : (nterms, ny, nx) in/out
//   model    : (nterms, ny, nx) in/out
//   psf      : (2*nterms-1, ny, nx) read-only
//   mask     : (ny, nx) read-only, or nullptr
//   gain <= 0 selects casacore adaptive gain
//   mask_threshold > 0 binarises the scale-convolved mask at 0.1 (CASA)
template <typename T>
CleanResult<T> clean(T* residual, T* model, const T* psf, const T* mask, int nterms, int nx, int ny,
                     const std::vector<float>& scales, float small_scale_bias, int niter, T gain, T threshold,
                     T stop_fraction, T mask_threshold);

// residual[t1] = sum_t2 inverse_hessian[t1, t2] * residual[t2] in place
// (MultiTermMatrixCleaner::computeprincipalsolution). inverse_hessian is
// nterms x nterms row-major (the delta-function scale inverse).
template <typename T>
void principal_solution(T* residual, const double* inverse_hessian, int nterms, int nx, int ny);

extern template int taylor_hessian<float>(const float*, int, int, int, std::vector<float>&, float,
                                          std::vector<double>&, std::vector<double>&, int&);
extern template int taylor_hessian<double>(const double*, int, int, int, std::vector<float>&, float,
                                           std::vector<double>&, std::vector<double>&, int&);
extern template CleanResult<float> clean<float>(float*, float*, const float*, const float*, int, int, int,
                                                const std::vector<float>&, float, int, float, float, float, float);
extern template CleanResult<double> clean<double>(double*, double*, const double*, const double*, int, int, int,
                                                  const std::vector<float>&, float, int, double, double, double,
                                                  double);
extern template void principal_solution<float>(float*, const double*, int, int, int);
extern template void principal_solution<double>(double*, const double*, int, int, int);

}  // namespace mtmfs
