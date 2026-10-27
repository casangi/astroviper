#pragma once

// Multi-Term Multi-Frequency Synthesis (MTMFS) CLEAN -- stateless kernel.
//
// The model update behind CASA tclean's `deconvolver='mtmfs'`
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
// are (nterms, ny, nx); the PSF stack is (2*nterms-1, ny, nx). Residual
// spatial planes must be contiguous, but residual_term_stride may exceed
// ny*nx so a polarization slice of a larger image can be updated without a
// copy. Model and PSF stacks remain fully contiguous.

#include <cstddef>
#include <vector>

#include "../../include/plane_stop_test.hpp"

namespace mtmfs {

// The stop codes shared by the deconvolvers (plane_stop_test.hpp). Nothing
// left to clean in the search region counts as kStopThreshold.
enum StopCode : int {
    kStopNone = deconvolvers::STOP_NONE,
    kStopMaxIter = deconvolvers::STOP_MAX_ITER,
    kStopThreshold = deconvolvers::STOP_THRESHOLD,
    kStopDiverged = deconvolvers::STOP_DIVERGED,
};

constexpr int kSingularHessian = -2;
constexpr int kMaxTaylorTerms = 4;

template <typename T>
struct CleanResult {
    int iterations = 0;
    int stop_code = kStopNone;
    T peak_residual = 0;
    T model_flux = 0;
    int psf_support = 0;
    float small_scale_bias = 0.0f;
    std::vector<float> scales;
    std::vector<double> hessian;
    std::vector<double> inverse_hessian;
    std::vector<double> hessian_condition_number;
};

// Empty -> {0}; otherwise 0 must be present and is the smallest effective
// scale. Sorted ascending and unique; sizes larger than half the image are
// dropped (MultiTermMatrixCleaner::verifyScaleSizes).
std::vector<float> effective_scales(const std::vector<float>& scales, int nx, int ny);

// SDAlgorithmMSMFS::initializeDeconvolver: clamp to [-1, 1].
float clamp_small_scale_bias(float bias);

// Per-scale Taylor Hessians and inverses. `scales` is replaced by the
// effective list. hessian / inverse_hessian are (nscales, nterms, nterms)
// row-major. Returns 0 or kSingularHessian.
template <typename T>
int taylor_hessian(const T* psf, int nterms, int nx, int ny, std::vector<float>& scales,
                   float small_scale_bias, std::vector<double>& hessian,
                   std::vector<double>& inverse_hessian,
                   std::vector<double>& hessian_condition_number, int& psf_support);

// One MTMFS model update in place (MultiTermMatrixCleaner::mtclean).
//   residual : (nterms, ny, nx) in/out
//   model    : (nterms, ny, nx) in/out
//   psf      : (2*nterms-1, ny, nx) read-only
//   mask     : (ny, nx) read-only, or nullptr
//   gain <= 0 selects casacore adaptive gain
//   mask_threshold > 0 binarises the scale-convolved mask at 0.1 (CASA)
//   divergence: the divergence test shared with the other deconvolvers
//   (plane_stop_test.hpp); stops with kStopDiverged
template <typename T>
CleanResult<T> clean(T* residual, T* model, const T* psf, const T* mask, int nterms, int nx, int ny,
                     const std::vector<float>& scales, float small_scale_bias, int max_iter_per_cycle_updated, T gain,
                     T threshold_per_cycle, T mask_threshold,
                     const deconvolvers::DivergenceControl<T>& divergence, std::ptrdiff_t residual_term_stride = 0);

// residual[t1] = sum_t2 inverse_hessian[t1, t2] * residual[t2] in place
// (MultiTermMatrixCleaner::computeprincipalsolution). inverse_hessian is
// nterms x nterms row-major (the delta-function scale inverse).
template <typename T>
void principal_solution(T* residual, const double* inverse_hessian, int nterms, int nx, int ny);

extern template int taylor_hessian<float>(const float*, int, int, int, std::vector<float>&, float,
                                          std::vector<double>&, std::vector<double>&,
                                          std::vector<double>&, int&);
extern template int taylor_hessian<double>(const double*, int, int, int, std::vector<float>&, float,
                                           std::vector<double>&, std::vector<double>&,
                                           std::vector<double>&, int&);
extern template CleanResult<float> clean<float>(float*, float*, const float*, const float*, int, int, int,
                                                const std::vector<float>&, float, int, float, float, float,
                                                const deconvolvers::DivergenceControl<float>&, std::ptrdiff_t);
extern template CleanResult<double> clean<double>(double*, double*, const double*, const double*, int, int, int,
                                                  const std::vector<float>&, float, int, double, double, double,
                                                  const deconvolvers::DivergenceControl<double>&, std::ptrdiff_t);
extern template void principal_solution<float>(float*, const double*, int, int, int);
extern template void principal_solution<double>(double*, const double*, int, int, int);

}  // namespace mtmfs
