#pragma once

#include <cstddef>

#include "hclean.hpp"

namespace relaxed_hogbom {

struct Controls {
    double constraint_radius_beams = 0.5;
    int max_active_components = 100;
    double minimum_improvement = 0.001;
    bool adaptive_start = true;
    double rapid_progress_rms_ratio = 0.25;
    double stalled_progress_rms_ratio = 0.78;
    int rms_normalization_iterations = 100;
    int optimizer_max_iterations = 100;
};

template<typename T>
void clean_cube(
    T* residual_cube, T* model_cube, const T* psf_cube,
    int domask, const bool* mask_cube,
    int nt, int nf, int np_img, int np_psf, int ny, int nx,
    int xbeg, int xend, int ybeg, int yend,
    const int* max_iter_remaining, T gain, const T* threshold,
    int processing_function_threads,
    const hclean::DivergenceControl<T>& divergence,
    const Controls& controls,
    int* iter_out, int* stop_out, bool* attempted_out, bool* accepted_out,
    int* component_count_out, double* initial_norm_out,
    double* final_norm_out, double* rms_ratio_out);

extern template void clean_cube<float>(
    float*, float*, const float*, int, const bool*, int, int, int, int,
    int, int, int, int, int, int, const int*, float, const float*, int,
    const hclean::DivergenceControl<float>&, const Controls&, int*, int*,
    bool*, bool*, int*, double*, double*, double*);

extern template void clean_cube<double>(
    double*, double*, const double*, int, const bool*, int, int, int, int,
    int, int, int, int, int, int, const int*, double, const double*, int,
    const hclean::DivergenceControl<double>&, const Controls&, int*, int*,
    bool*, bool*, int*, double*, double*, double*);

}  // namespace relaxed_hogbom
