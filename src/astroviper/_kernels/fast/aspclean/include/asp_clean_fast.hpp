#pragma once

// The fast CPU Asp (Coral) model update (asp_clean_fast.cpp): the entry points
// of the reference's asp_clean.hpp (at e0c0d126, read from the reference's
// include directory), in namespace aspclean_fast, with the same arguments,
// results (aspclean::AspResult), options (aspclean::AspOptions) and
// exceptions, and outputs of the same bits.

#include <cstddef>
#include <cstring>
#include <functional>

#include "asp_clean.hpp"

namespace aspclean_fast {

using aspclean::AspOptions;
using aspclean::AspResult;

// A read-only array of T at any address (Dask can hand a worker an array at
// any byte offset of a frame it received, and reading a T through a
// misaligned T* is undefined): each item is read with std::memcpy, which
// compiles to a plain load. Null for an absent array (no mask).
template <typename T>
struct Unaligned {
    const unsigned char* bytes = nullptr;

    T operator[](std::size_t k) const {
        T v;
        std::memcpy(&v, bytes + k * sizeof(T), sizeof(T));
        return v;
    }
    // The array from item k on (null stays null).
    Unaligned at(std::size_t k) const { return {bytes ? bytes + k * sizeof(T) : nullptr}; }
    explicit operator bool() const { return bytes != nullptr; }
};

// The reference's checks of the controls (check_controls of asp_clean.cpp):
// std::invalid_argument, with its messages.
void check_controls(double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                    int asp_scale_normalization_method, double asp_min_mask_coverage);

// Whether a call must run the reference instead: its PSF or its model holds a
// value that is not finite or of magnitude 2^64 or more, or its residual holds
// one of magnitude 2^64 or more, or one that is not finite where the reference
// searches (where the mask is above 0.99, or anywhere without a mask; the
// reference sets such a value aside elsewhere). On a value that is not finite,
// pocketfft's SIMD lanes (this module's transforms) and its scalar path (the
// reference's, on the odd last line of each thread's share) can give a NaN
// another sign bit or payload, and so can two compilations of one sum: IEEE
// 754 leaves open which NaN an operation on two NaNs returns. Below 2^64 the
// transforms of the PSF and of the residual cannot overflow. A declined call
// has the reference's bits by construction, at its own thread count. `n` items
// in the residual, the model and the mask (when given), `npsf` in the PSF;
// `threads` share the scan.
template <typename T>
bool needs_reference(Unaligned<T> residual, Unaligned<T> model, Unaligned<T> psf, Unaligned<T> mask,
                     std::size_t n, std::size_t npsf, int threads);

// needs_reference of each plane of a cube (nt x nf x np_img planes of `plane`
// items, with nt x nf x np_psf PSF planes): declined[p] is 1 for a plane that
// needs the reference, else 0; returns their number. Each PSF plane is
// scanned once.
template <typename T>
int planes_needing_reference(Unaligned<T> residual, Unaligned<T> model, Unaligned<T> psf, Unaligned<T> mask, int nt,
                             int nf, int np_img, int np_psf, std::size_t plane, int threads, unsigned char* declined);

// The planes of a cube that the reference cleans. The cube's workers, as many
// as the reference's driver starts, take the planes in its order; a declined
// plane p goes to run(p, threads): the reference on that plane alone, its
// transforms on `threads` threads (the share the reference's driver gives a
// plane at that point), which returns the plane's result. The other planes
// keep their bits, which depend on no other plane and no thread count.
struct ReferencePlanes {
    const unsigned char* declined = nullptr;  // a flag per plane; null: none
    std::function<AspResult(int plane, int threads)> run;
};

template <typename T>
AspResult aspclean_plane(T* residual, T* model, Unaligned<T> psf, Unaligned<T> mask, int nx, int ny, double gain,
                         double threshold_per_cycle, int max_iter_per_cycle_updated, double asp_fused_threshold,
                         double asp_psf_width, int asp_largest_scale, int asp_stop_point_mode,
                         int asp_scale_normalization_method, bool asp_stop_after_hogbom_run, bool verbose,
                         const deconvolvers::DivergenceControl<double>& divergence, int fft_threads = 1,
                         const AspOptions& options = AspOptions{});

template <typename T>
void aspclean_cube(T* residual, T* model, Unaligned<T> psf, Unaligned<T> mask, int nt, int nf, int np_img,
                   int np_psf, int nx, int ny, double gain, const double* threshold_per_cycle,
                   const int* max_iter_per_cycle_updated, double asp_fused_threshold, double asp_psf_width,
                   int asp_largest_scale, int asp_stop_point_mode, int asp_scale_normalization_method,
                   bool asp_stop_after_hogbom_run, int processing_function_threads,
                   const deconvolvers::DivergenceControl<double>& divergence, AspResult* results,
                   const AspOptions& options = AspOptions{}, const ReferencePlanes& reference = {});

}  // namespace aspclean_fast
