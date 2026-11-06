#pragma once

// The fast CPU Asp (Coral) model update (asp_clean_fast.cpp): the entry points
// of the reference's asp_clean.hpp (at e0c0d126, read from the reference's
// include directory), in namespace aspclean_fast, with the same arguments,
// results (aspclean::AspResult), options (aspclean::AspOptions) and
// exceptions, and outputs of the same bits.

#include "asp_clean.hpp"

namespace aspclean_fast {

using aspclean::AspOptions;
using aspclean::AspResult;

template <typename T>
AspResult aspclean_plane(T* residual, T* model, const T* psf, const T* mask, int nx, int ny, double gain,
                         double threshold_per_cycle, int max_iter_per_cycle_updated, double asp_fused_threshold,
                         double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                         int asp_scale_normalization_method, bool asp_stop_after_hogbom_run, bool verbose,
                         const deconvolvers::DivergenceControl<double>& divergence, int fft_threads = 1,
                         const AspOptions& options = AspOptions{});

template <typename T>
void aspclean_cube(T* residual, T* model, const T* psf, const T* mask, int nt, int nf, int np_img, int np_psf,
                   int nx, int ny, double gain, const double* threshold_per_cycle,
                   const int* max_iter_per_cycle_updated, double asp_fused_threshold, double asp_psf_width,
                   int asp_largest_scale, int asp_stop_point_mode, int asp_scale_normalization_method,
                   bool asp_stop_after_hogbom_run, int processing_function_threads,
                   const deconvolvers::DivergenceControl<double>& divergence, AspResult* results,
                   const AspOptions& options = AspOptions{});

}  // namespace aspclean_fast
