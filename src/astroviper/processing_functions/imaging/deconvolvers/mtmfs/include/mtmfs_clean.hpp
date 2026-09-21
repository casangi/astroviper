#pragma once

// Multi-Term Multi-Frequency Synthesis (MTMFS) CLEAN engine.
//
// This is the algorithm behind CASA tclean's `deconvolver='mtmfs'`
// (synthesis::SDAlgorithmMSMFS driving MultiTermMatrixCleaner). It is ported
// from CASA's casacore-free `stdcleaner/StdMultiTermCleaner.{h,cc}`, which was
// verified against casacore MultiTermMatrixCleaner to ~1e-8 relative (Hessian
// ~1e-16), and follows the SDAlgorithmMSMFS lifecycle:
//
//   MultiTermCleaner c(nterms, scales, nx, ny, small_scale_bias);  // ctor
//   c.set_psf(psf);              // once per PSF: PSF/scale spectra + Hessian
//   c.clean(residual, model, mask, ...);   // one model-update (minor) cycle,
//                                          // repeated after every residual
//                                          // update (major) cycle
//   c.compute_principal_solution(residual);  // optional, before restoration
//
// Memory contract (astroviper AGENTS.md section 6):
//   * The residual, model, PSF and mask images are *caller-owned* buffers
//     (NumPy arrays on the Python side). The engine never copies them: it
//     reads the PSF/mask through const pointers and updates residual and
//     model in place through the pointers handed to clean(). No pointer to a
//     caller buffer is retained after the call returns.
//   * The only C++-owned memory is the engine's *derived working state* --
//     PSF/scale/mask spectra, the per-scale Taylor Hessians and their
//     convolution patches, the convolved-residual (RHS) and coefficient images
//     and the per-cycle model delta -- which has no NumPy counterpart. It is
//     allocated once per set_psf()/clean() and reused across calls.
//
// Layout: all images are row-major (ny, nx); multi-term stacks are
// (nterms, ny, nx) with term t at offset t*ny*nx. The PSF stack has
// 2*nterms-1 Taylor terms.

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace mtmfs {

template <typename T>
class MultiTermCleaner {
   public:
    using value_type = T;
    using complex_t = std::complex<T>;

    // nterms       : number of Taylor terms (>= 1); the PSF has 2*nterms-1 terms.
    // scales       : multi-scale sizes in pixels. Empty -> {0} (point source
    //                only), as SDAlgorithmMSMFS does. Scales larger than half
    //                the image are dropped (MultiTermMatrixCleaner::
    //                verifyScaleSizes); scales() returns the effective list.
    // nx, ny       : image dimensions.
    // small_scale_bias : clamped to [-1, 1] as SDAlgorithmMSMFS does.
    MultiTermCleaner(int nterms, const std::vector<float>& scales, int nx, int ny,
                     float small_scale_bias);

    int nterms() const { return ntaylor_; }
    int npsf_terms() const { return psf_ntaylor_; }
    int nscales() const { return static_cast<int>(scale_sizes_.size()); }
    const std::vector<float>& scales() const { return scale_sizes_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    float small_scale_bias() const { return small_scale_bias_; }
    int psf_support() const { return psf_support_; }
    bool has_psf() const { return has_psf_; }

    // Store the transforms of all PSF Taylor terms (psf is a caller-owned
    // (2*nterms-1, ny, nx) stack, read only), build the scale functions and
    // compute the per-scale Taylor Hessians and convolution patches.
    // Returns 0 on success or -2 if any scale Hessian is not invertible
    // (the MT-Cleaner "Non-invertible Hessian" error).
    int set_psf(const T* psf);

    // Run one multi-term minor cycle (MultiTermMatrixCleaner::mtclean) in
    // place. residual and model are caller-owned (nterms, ny, nx) stacks:
    // CLEAN components are added into model and the residual is updated with
    // the new components convolved with the PSF terms. mask is an optional
    // caller-owned (ny, nx) image (nullptr = no mask); mask_threshold mirrors
    // the SDAlgorithmMSMFS setMask() threshold argument.
    // Returns the number of iterations done (or -2 if set_psf() failed).
    // Throws std::logic_error if set_psf() has not been called.
    int clean(T* residual, T* model, const T* mask, T mask_threshold, int max_iter,
              T stop_fraction, T gain, T threshold);

    // Port of MultiTermMatrixCleaner::computeprincipalsolution(): replace the
    // caller-owned residual stack in place by the per-pixel principal solution
    // residual_t1 = sum_t2 invHessian_scale0(t1, t2) * residual_t2.
    // Throws std::logic_error if set_psf() has not been called.
    void compute_principal_solution(T* residual) const;

    // Per-scale nterms x nterms Taylor Hessian and its inverse (row-major).
    const std::vector<double>& hessian(int scale) const;
    const std::vector<double>& inverse_hessian(int scale) const;

   private:
    std::size_t nimg() const { return static_cast<std::size_t>(nx_) * ny_; }
    std::size_t nspec() const { return static_cast<std::size_t>(ny_) * ncx_; }
    int cross_index4(int t1, int t2, int s1, int s2) const;  // IND4
    int ind2(int taylor, int scale) const { return taylor * nscales() + scale; }

    void setup_scale_functions();
    int compute_hessian();
    void setup_user_mask();
    void compute_rhs();
    void build_image_patches();
    void solve_matrix_eqn(int scale, const std::array<int, 2>& blc, const std::array<int, 2>& trc);
    void choose_component(int scale, const std::array<int, 2>& blc, const std::array<int, 2>& trc);
    void update_model_and_rhs(T loopgain);
    int check_convergence(T& fluxlimit, T& loopgain);

    T* residual_term(int t) const { return residual_ + static_cast<std::size_t>(t) * nimg(); }
    T* model_term(int t) const { return model_ + static_cast<std::size_t>(t) * nimg(); }

    int nx_ = 0, ny_ = 0, ncx_ = 0;
    int ntaylor_ = 0, psf_ntaylor_ = 0;
    int psf_peak_x_ = 0, psf_peak_y_ = 0;
    int psf_support_ = 0;  // patch side length
    float small_scale_bias_ = 0.0f;
    bool has_psf_ = false;

    std::vector<float> scale_sizes_;
    std::vector<T> scale_bias_;
    std::vector<std::vector<complex_t>> psf_ft_;    // psf_ntaylor_
    std::vector<std::vector<complex_t>> scale_ft_;  // nscales
    std::vector<std::vector<T>> vec_scales_;        // psf_support^2 scale patches

    std::vector<std::vector<double>> mat_a_;      // per scale, ntaylor x ntaylor
    std::vector<std::vector<double>> inv_mat_a_;  // per scale, ntaylor x ntaylor
    std::vector<std::vector<T>> cube_a_;          // psf_support^2 patches, IND4

    std::vector<std::vector<T>> mat_r_;           // RHS, per (taylor,scale), full
    std::vector<std::vector<T>> mat_coeffs_;      // coeffs, per (taylor,scale), full
    std::vector<std::vector<T>> delta_model_;     // per taylor: model added this cycle
    std::vector<std::vector<T>> vec_scale_masks_;  // per scale, full
    std::vector<std::vector<T>> vec_work_;         // per scale scratch, full

    // Caller-owned buffers, valid only for the duration of clean().
    T* residual_ = nullptr;
    T* model_ = nullptr;
    const T* mask_ = nullptr;
    T mask_threshold_ = static_cast<T>(0.1);

    // minor-cycle state
    T global_max_val_ = static_cast<T>(0);
    std::array<int, 2> global_max_pos_{0, 0};
    int max_scale_index_ = 0;
    std::array<int, 2> blc_{0, 0}, trc_{0, 0}, blc_psf_{0, 0}, trc_psf_{0, 0};
    std::vector<T> max_scale_val_;
    std::vector<std::array<int, 2>> max_scale_pos_;
    T prev_max_ = static_cast<T>(1e10), min_max_ = static_cast<T>(1e10);
    int total_iters_ = 0, itercount_ = 0;
    T stop_fraction_ = static_cast<T>(0), input_gain_ = static_cast<T>(0),
      user_threshold_ = static_cast<T>(0);
    int max_iter_ = 0;
};

extern template class MultiTermCleaner<float>;
extern template class MultiTermCleaner<double>;

}  // namespace mtmfs
