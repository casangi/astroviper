"""Check deconvolver selection, Taylor order, and stopping thresholds."""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_boolean,
    require_finite_real,
    require_integer,
    require_mapping,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_primary_beam_limit,
    check_taylor_count,
)
from astroviper.utils.param_docs import shares_param_docs

# ========================================================================
# Configuration checks
# ========================================================================


@shares_param_docs
def check_model_update(config, *, mode, require_complete=False):
    """Validate model-update settings without modifying the configuration.

    Parameters
    ----------
    config : mapping
        Configuration containing ``deconvolver``, ``image_params``, and
        ``cycle_control_params`` (``deconvolve_params`` at the processing-function
        boundary). Only supplied keys are checked.
    mode : ImagingMode
        Imaging kind (cube or continuum) and field layout.

    require_complete : bool, optional
        Require all coordinates and cycle controls needed by a distributed driver.
        Lower-level entry points check only the configuration they receive.

    Raises
    ------
    KeyError
        If an explicit MT-MFS image configuration lacks its Taylor order.
    TypeError
        If Taylor order is not an integer or a threshold is not real.
    ValueError
        If the deconvolver, Taylor order, or thresholds are invalid.
    NotImplementedError
        If multifield imaging is requested.
    """
    # ========================================================================
    # 1. Validation context
    # ========================================================================

    # --- Supported imaging mode ---
    # Reject multifield requests until their model-update path is implemented.
    mode.require_supported()
    require_mapping("config", config)

    deconvolver = config.get("deconvolver")
    image_params = require_mapping("image_params", config.get("image_params", {}))

    # ========================================================================
    # 2. PSF fitting and helper inputs
    # ========================================================================

    # --- PSF fitting implementation ---
    # A misspelled fitter must fail before constructing its input PSF.
    if "psf_fitting_method" in config and config["psf_fitting_method"] not in (
        "astroviper",
        "casa",
    ):
        raise ValueError("psf_fitting_method must be 'astroviper' or 'casa'.")

    # --- Direct Gaussian-fitting controls ---
    # These are helper arguments, not additional top-level imaging options.
    for name in ("npix_window", "sampling"):
        if name in config:
            check_psf_fit_counts(name, config[name])
    if "cutoff" in config:
        cutoff = require_finite_real("cutoff", config["cutoff"], scalar=True)
        if cutoff < 0:
            raise ValueError("cutoff must be non-negative.")

    # --- Paired MVC helper arguments ---
    # A PSF cube cannot supply its weights without its normalization product.
    if (
        config.get("psf_cube") is not None
        and config.get("psf_normalization") is None
        and "psf_normalization" in config
    ):
        raise ValueError("psf_normalization is required when psf_cube is supplied.")

    # ========================================================================
    # 3. Deconvolver choice and Taylor basis
    # ========================================================================

    # --- Deconvolver selection ---
    # Each imaging kind has its own set of implemented algorithms.
    if "deconvolver" in config:
        check_deconvolver(deconvolver, mode=mode)

    # --- Cube spectral model ---
    # Cube planes are independent images; a multi-term continuum model is invalid.
    if mode.imaging_kind == "cube" and "n_taylor_terms" in image_params:
        check_taylor_count(image_params["n_taylor_terms"])
        if image_params["n_taylor_terms"] != 1:
            raise ValueError("Cube imaging supports only n_taylor_terms=1.")

    # --- Required Taylor order ---
    # An explicit Asp/MT-MFS image configuration must specify its number of terms.
    # Leaf model-update calls take image products instead of image_params.
    if (
        mode.imaging_kind == "continuum"
        and deconvolver in ("asp", "mtmfs")
        and "image_params" in config
        and "n_taylor_terms" not in image_params
    ):
        raise KeyError(
            f"image_params must contain 'n_taylor_terms' when deconvolver={deconvolver!r}."
        )

    if mode.imaging_kind == "continuum" and "n_taylor_terms" in image_params:
        n_taylor_terms = image_params["n_taylor_terms"]

        # --- Positive integer Taylor order ---
        # Booleans and fractional counts cannot define a Taylor basis.
        check_taylor_count(n_taylor_terms)

        # --- Single-term deconvolver compatibility ---
        # Högbom and Asp update only the constant term of a continuum model.
        if deconvolver in ("hogbom", "asp") and n_taylor_terms != 1:
            raise ValueError(
                f"Continuum {deconvolver} requires n_taylor_terms=1. "
                "Use deconvolver='mtmfs' for 2 to 4 Taylor terms."
            )

        # --- MT-MFS compatibility ---
        # The current continuum MT-MFS path supports two to four terms.
        if deconvolver == "mtmfs" and not 2 <= n_taylor_terms <= 4:
            raise ValueError("Continuum mtmfs requires n_taylor_terms between 2 and 4.")

    # ========================================================================
    # 4. Stopping controls and iteration budgets
    # ========================================================================

    # --- Stopping thresholds ---
    # The absolute floor and per-cycle thresholds must be finite and non-negative.
    # Per-cycle values may be arrays with one threshold for each image plane.
    if "cycle_control_params" in config:
        controls = require_mapping(
            "cycle_control_params", config["cycle_control_params"]
        )
    else:
        controls = require_mapping(
            "deconvolve_params", config.get("deconvolve_params", {})
        )

    for name in ("threshold", "threshold_per_cycle"):
        if name in controls:
            check_nonnegative(name, controls[name])

    # --- Per-plane control geometry ---
    # Valid values with the wrong number of planes cannot drive the controller.
    _check_control_shapes(controls, image_params, mode=mode)

    # --- Per-channel absolute thresholds ---
    # A two-dimensional cube threshold has one row per requested channel.
    if mode.imaging_kind == "cube" and "threshold" in controls:
        shape = np.shape(controls["threshold"])
        if len(shape) == 2 and all(
            name in image_params for name in ("frequency_coords", "polarization_coords")
        ):
            expected = (
                len(image_params["frequency_coords"]),
                len(image_params["polarization_coords"]),
            )
            if shape != expected:
                raise ValueError(
                    f"A per channel threshold has shape {shape}; expected "
                    f"(n_frequency, n_polarization) = {expected}."
                )

    # --- Required cycle controls ---
    # Application drivers need a complete controller configuration before I/O.
    if require_complete:
        for name in (
            "max_iter",
            "max_cycles",
            "threshold",
            "gain",
            "psf_sidelobe_factor",
            "min_psf_fraction",
            "max_psf_fraction",
            "max_iter_per_cycle",
        ):
            if name not in controls:
                raise KeyError(f"cycle_control_params must contain {name!r}.")

    # --- Recognized control names ---
    # Misspellings must fail here rather than leave a requested control unused.
    allowed_controls = {
        "gain",
        "max_iter",
        "max_cycles",
        "max_iter_per_cycle",
        "threshold",
        "threshold_sigma",
        "threshold_tolerance",
        "psf_sidelobe_factor",
        "min_psf_fraction",
        "max_psf_fraction",
        "max_iter_divergence",
        "max_cycles_divergence",
        "max_iter_per_cycle_updated",
        "threshold_per_cycle",
        "primary_beam_limit",
        "clean_box",
        "record_iteration_history",
        "exact_psf_subtraction",
        "subtraction_psf_cube",
        "scales",
        "small_scale_bias",
        "mask_threshold",
        "entropy_stop",
        "entropy_max_snr",
        "entropy_spatial_bins",
        "entropy_flux_bins",
        "asp_stop_after_hogbom_run",
        "asp_residual_components",
        "asp_scale_significance",
        "asp_residual_component_max_epsilon_fraction",
        "asp_residual_component_floor_at_threshold",
        "asp_residual_component_min_fraction",
        "asp_residual_component_noise_floor",
        "asp_residual_component_switch_factor",
        "asp_residual_component_positive_only",
        "asp_no_change_tolerance",
        "asp_min_mask_coverage",
        "asp_significance",
        "asp_significance_gain",
        "asp_fused_threshold",
        "asp_largest_scale",
        "asp_scale_normalization_method",
        "asp_psf_width",
        "asp_stop_point_mode",
    }
    unknown = set(controls) - allowed_controls
    if unknown:
        raise ValueError(f"Unknown cycle/deconvolution controls: {sorted(unknown)}.")

    # --- Fixed gain ---
    # High-level imaging supports fixed gain only. Adaptive gain remains an
    # explicitly documented option of direct native MT-MFS calls.
    if "gain" in controls:
        gain = require_finite_real("gain", controls["gain"], scalar=True)
        if not 0 < gain <= 1:
            raise ValueError(
                "Fixed CLEAN gain must satisfy 0 < gain <= 1; adaptive gain is not exposed by the imaging workflows."
            )

    # --- Iteration budgets ---
    # Zero iterations is a valid dirty-image request; -1 disables selected caps.
    for name in ("max_iter", "max_iter_per_cycle_updated"):
        if name in controls:
            values = np.asarray(controls[name])
            if values.dtype.kind not in "iu" or not values.size or np.any(values < 0):
                raise ValueError(f"{name} must contain non-negative integer counts.")
    if "max_iter_per_cycle" in controls:
        require_integer(
            "max_iter_per_cycle", controls["max_iter_per_cycle"], allow_minus_one=True
        )
    for name in ("max_iter_divergence", "max_cycles_divergence"):
        if name in controls:
            require_integer(name, controls[name], minimum=1, allow_minus_one=True)

    # --- Imaging-cycle limit ---
    # Automatic cycle prediction requires a contraction factor below one.
    if "max_cycles" in controls:
        check_max_cycles(controls["max_cycles"], controls.get("max_psf_fraction", 0.8))

    # --- Threshold and PSF policies ---
    # Fractions bound the adaptive cycle threshold; all multipliers are finite.
    for name in (
        "min_psf_fraction",
        "max_psf_fraction",
        "mask_threshold",
    ):
        if name in controls:
            value = require_finite_real(name, controls[name], scalar=True)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")
    if "primary_beam_limit" in controls:
        check_primary_beam_limit(controls["primary_beam_limit"])
    if controls.get("min_psf_fraction", 0.05) > controls.get("max_psf_fraction", 0.8):
        raise ValueError("min_psf_fraction must not exceed max_psf_fraction.")
    for name in ("threshold_sigma", "threshold_tolerance", "psf_sidelobe_factor"):
        if name in controls:
            check_nonnegative(name, controls[name], scalar=True)

    # ========================================================================
    # 5. Component geometry and spatial scales
    # ========================================================================

    # --- MT-MFS scale bias ---
    # Reject a bias the native implementation would otherwise clamp.
    if "small_scale_bias" in controls:
        value = require_finite_real(
            "small_scale_bias", controls["small_scale_bias"], scalar=True
        )
        if not -1 <= value <= 1:
            raise ValueError("small_scale_bias must be between -1 and 1.")

    # --- CLEAN rectangle ---
    # None and the all-minus-one sentinel explicitly select the whole image.
    # A plane has (l, m) = (y, x) axes. Bounds describe half-open slices:
    # xmax/ymax may equal the corresponding axis length, but the rectangle must
    # be nonempty. Reject invalid bounds instead of relying on native clipping.
    if "clean_box" in controls and controls["clean_box"] is not None:
        box = controls["clean_box"]
        if not isinstance(box, tuple) or len(box) != 4:
            raise ValueError(
                "clean_box must be a 4-tuple (xmin, xmax, ymin, ymax) or None."
            )
        for value in box:
            require_integer("clean_box coordinate", value, allow_minus_one=True)
        if box != (-1, -1, -1, -1):
            if any(value < 0 for value in box):
                raise ValueError(
                    "clean_box coordinates must be non-negative or use the whole-image tuple (-1, -1, -1, -1)."
                )
            xmin, xmax, ymin, ymax = box
            if xmin >= xmax or ymin >= ymax:
                raise ValueError("clean_box requires xmin < xmax and ymin < ymax.")
            if "image_size" in image_params:
                n_l, n_m = image_params["image_size"]
                if xmax > n_m or ymax > n_l:
                    raise ValueError(
                        "clean_box must fit the image geometry: xmax <= size of m, ymax <= size of l."
                    )

    # --- Requested scales ---
    # Validate the exact list before the kernel can sort or discard any scale.
    if "scales" in controls:
        check_scales(controls["scales"], image_size=image_params.get("image_size"))
        if deconvolver not in (None, "mtmfs"):
            raise ValueError("scales is supported only by the mtmfs deconvolver.")

    # --- MT-MFS-only controls ---
    # Högbom and Asp must not silently ignore a requested multi-scale MT-MFS option.
    if deconvolver is not None and deconvolver != "mtmfs":
        unused = set(controls) & {"small_scale_bias", "mask_threshold"}
        if unused:
            raise ValueError(
                f"MT-MFS controls require deconvolver='mtmfs': {sorted(unused)}."
            )

    # ========================================================================
    # 6. Algorithm options and entropy stopping
    # ========================================================================

    # --- Algorithm-specific Boolean controls ---
    # History is available for each current deconvolver; exact subtraction is
    # restricted to cube Högbom, whose graph supplies the larger subtraction PSF.
    boolean_controls = (
        "asp_stop_after_hogbom_run",
        "asp_residual_components",
        "asp_scale_significance",
        "asp_residual_component_floor_at_threshold",
        "asp_residual_component_positive_only",
    )
    for name in boolean_controls:
        if name in controls:
            require_boolean(name, controls[name])
    for name in ("record_iteration_history", "exact_psf_subtraction"):
        if name in controls:
            check_hogbom_option(deconvolver, name, controls[name], mode=mode)
    if deconvolver in ("hogbom", "hogbom_many_threads", "mtmfs"):
        unsupported = [name for name in controls if name.startswith("asp_")]
        if unsupported:
            raise ValueError(f"Asp controls require an Asp deconvolver: {unsupported}.")

    # --- Asp scale and normalization controls ---
    # Preserve documented -1 sentinels, but reject unsupported normalization modes.
    if "asp_largest_scale" in controls:
        require_integer(
            "asp_largest_scale", controls["asp_largest_scale"], allow_minus_one=True
        )
    if "asp_stop_point_mode" in controls:
        require_integer(
            "asp_stop_point_mode",
            controls["asp_stop_point_mode"],
            minimum=1,
            allow_minus_one=True,
        )
    if "asp_scale_normalization_method" in controls:
        require_integer(
            "asp_scale_normalization_method",
            controls["asp_scale_normalization_method"],
            minimum=1,
        )
        if controls["asp_scale_normalization_method"] not in (1, 2):
            raise ValueError("asp_scale_normalization_method must be 1 or 2.")
    for name in (
        "asp_fused_threshold",
        "asp_psf_width",
        "asp_significance",
        "asp_residual_component_noise_floor",
        "asp_residual_component_switch_factor",
    ):
        if name in controls:
            require_finite_real(name, controls[name], scalar=True)

    # --- Asp significance gain ---
    # Positive gains must not exceed one; nonpositive values deliberately use
    # the ordinary CLEAN gain and remain valid sentinels.
    if "asp_significance_gain" in controls:
        gain = require_finite_real(
            "asp_significance_gain", controls["asp_significance_gain"], scalar=True
        )
        if gain > 1:
            raise ValueError("asp_significance_gain must be <= 1.")

    # --- Residual-component fraction interval ---
    # The native fit fixes its upper fraction at 0.99. A larger lower bound
    # creates an empty fitting interval instead of a meaningful configuration.
    if "asp_residual_component_min_fraction" in controls:
        fraction = require_finite_real(
            "asp_residual_component_min_fraction",
            controls["asp_residual_component_min_fraction"],
            scalar=True,
        )
        if fraction > 0.99:
            raise ValueError("asp_residual_component_min_fraction must be <= 0.99.")

    # --- Residual-component smoothing interval ---
    # The native log-space fit fixes its lower epsilon fraction at 1e-3.
    # Its upper bound must preserve that interval and be positive for the log.
    if "asp_residual_component_max_epsilon_fraction" in controls:
        fraction = require_finite_real(
            "asp_residual_component_max_epsilon_fraction",
            controls["asp_residual_component_max_epsilon_fraction"],
            scalar=True,
        )
        if fraction < 1e-3:
            raise ValueError(
                "asp_residual_component_max_epsilon_fraction must be >= 1e-3."
            )

    if "asp_no_change_tolerance" in controls:
        check_nonnegative(
            "asp_no_change_tolerance", controls["asp_no_change_tolerance"], scalar=True
        )

    # --- Asp mask coverage ---
    # The native search keeps a Gaussian scale only where its share of flux on
    # the mask exceeds this value; a share cannot exceed 1, so 1 would keep none.
    if "asp_min_mask_coverage" in controls:
        coverage = require_finite_real(
            "asp_min_mask_coverage", controls["asp_min_mask_coverage"], scalar=True
        )
        if not 0 <= coverage < 1:
            raise ValueError("asp_min_mask_coverage must be at least 0 and below 1.")

    # --- Entropy stopping ---
    # Bin counts are positive integers; the SNR cutoff must be finite and positive.
    check_entropy_controls(controls)


# ========================================================================
# Shared control-shape and scale checks
# ========================================================================


def _check_control_shapes(controls, image_params, *, mode):
    """Check caller control arrays against the geometry known at this boundary.

    Parameters
    ----------
    controls : mapping
        Supplied scalar or per-plane cycle and iteration controls.
    image_params : mapping
        Known image dimensions and coordinates; unavailable extents are not
        inferred.
    mode : ImagingMode
        Cube or continuum layout used to interpret control-array axes.
    """
    if "threshold" in controls:
        shape = np.shape(controls["threshold"])
        allowed_ranks = (0, 1, 2) if mode.imaging_kind == "cube" else (0, 1)
        if len(shape) not in allowed_ranks:
            raise ValueError(
                "threshold must be scalar, one value per polarization, or a cube (frequency, polarization) array."
            )
        if (
            len(shape) == 1
            and "polarization_coords" in image_params
            and shape[0] != len(image_params["polarization_coords"])
        ):
            raise ValueError(
                "threshold must have one value per requested polarization."
            )

    # Per-cycle arrays use native (time, frequency, polarization) order.
    # Continuum's Taylor stack has one effective frequency plane.
    for name in ("max_iter_per_cycle_updated", "threshold_per_cycle", "max_iter"):
        if name not in controls or np.ndim(controls[name]) == 0:
            continue
        shape = np.shape(controls[name])
        if len(shape) != 3:
            raise ValueError(
                f"{name} must be scalar or have per-plane shape (time, frequency, polarization)."
            )
        for axis, dim in enumerate(("time", "frequency", "polarization")):
            if dim == "frequency" and mode.imaging_kind == "continuum":
                expected = 1
            elif f"{dim}_coords" in image_params:
                expected = len(image_params[f"{dim}_coords"])
            else:
                continue
            if shape[axis] != expected:
                raise ValueError(
                    f"{name} {dim} extent {shape[axis]} does not match {expected}."
                )


def check_scales(scales, *, image_size=None):
    """Validate an ordered MT-MFS scale list and return its native float values.

    Parameters
    ----------
    scales : array-like
        Nonempty, strictly increasing pixel scales beginning with zero.
    image_size : array-like, optional
        Spatial image dimensions. Scales must fit within half of each axis.

    Returns
    -------
    list of float
        Effective float32 scale values passed to the native kernel.
    """

    # --- Scale values ---
    # The C++ interface uses float32 scales, so check representability there too.
    values = require_finite_real("scales", scales)
    if (
        values.ndim != 1
        or np.any(values < 0)
        or np.any(values > np.finfo(np.float32).max)
    ):
        raise ValueError(
            "scales must be a nonempty one-dimensional list of finite, non-negative pixel sizes."
        )
    effective = values.astype(np.float32)
    if effective[0] != 0 or np.any(np.diff(effective) <= 0):
        raise ValueError(
            "scales must start with 0 and be strictly increasing at native float precision; scales are not sorted or deduplicated automatically."
        )

    # --- Scale support ---
    # Native MT-MFS discards scales larger than half an image axis; reject them.
    if image_size is not None:
        size = np.asarray(image_size)
        if size.shape != (2,) or np.any(size <= 0):
            raise ValueError("image_size must contain two positive pixel counts.")
        maximum_scale = int(size.min()) // 2
        if np.any(effective > maximum_scale):
            rejected = values[effective > maximum_scale].tolist()
            raise ValueError(
                f"scales {rejected} exceed half the image geometry {size.tolist()} (maximum {maximum_scale} pixels); no scales were dropped."
            )
    return effective.tolist()


# ========================================================================
# Approximate Taylor-Hessian checks before PSFs exist
# ========================================================================


def check_taylor_hessian_approximation(
    frequencies,
    effective_weights,
    *,
    n_taylor_terms,
    reference_frequency,
    longest_projected_baselines,
    image_size,
    cell_size,
    precision="float64",
    context=None,
):
    """Check a frequency-moment approximation before PSF products exist.

    Parameters
    ----------
    frequencies, effective_weights : array-like
        Selected frequencies in Hz and summed effective imaging weights after
        flags and correlation selection. No visibility values are required.
    n_taylor_terms : int
        Number of Taylor coefficients.
    reference_frequency : float
        Reference frequency in Hz.
    longest_projected_baselines : array-like or float
        Longest usable projected baselines in metres, one per frequency sample
        or a common scalar. Used for Gaussian FWHM = wavelength / baseline.
    image_size, cell_size : array-like
        Two image dimensions and signed angular pixel sizes in radians.
    precision : {"float32", "float64"}
        Working precision for the approximate inversion-safety thresholds.
    context : mapping, optional
        Field, spectral-window, selection, and polarization identifiers.

    Returns
    -------
    dict
        Equilibrated approximate Hessian, pivot/eigenvalue diagnostics, and
        assumption-based order-zero noise amplification. All values are JSON
        serializable; caller inputs remain unchanged.

    Notes
    -----
    This standalone diagnostic uses caller-supplied weights and resolution
    estimates. Coordinate-only distributed preflight does not collect these
    arrays or invoke this function. Local spectral guards use H_hat = A.T @ A.
    Spatial delta-component
    interactions are separately diagnosed with natural-resolution channel
    Gaussian PSFs on a coarse Fourier grid. Sidelobes, actual sampling gaps,
    gridding effects, and per-plane differences hidden by aggregated weights
    are absent; this cannot certify the later actual-PSF checks.
    """
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.config_check._spectral import (
        make_weighted_taylor_design,
        prepare_spectral_samples,
    )

    diagnostics = {
        **(context or {}),
        "approximation": "frequency_moment_gram",
        "n_taylor_terms": n_taylor_terms,
        "working_precision": np.dtype(precision).name,
        "covariance_assumption": "proportional_to_approximate_hessian",
    }

    def fail(reason):
        raise ValueError(
            f"Approximate Taylor-Hessian preflight: {reason}; {diagnostics}"
        )

    # ========================================================================
    # 1. Selected data and approximate Hessian construction
    # ========================================================================

    # --- Selected spectral metadata ---
    # Only positive-weight samples constrain the approximate Taylor system.
    try:
        frequencies, weights, reference, usable = prepare_spectral_samples(
            frequencies,
            effective_weights,
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference_frequency,
            precision=precision,
        )
    except (TypeError, ValueError) as error:
        fail(str(error))
    if np.unique(frequencies[usable]).size < n_taylor_terms:
        fail("fewer distinct positive-weight frequencies than Taylor terms")
    diagnostics["reference_frequency"] = reference
    diagnostics["n_taylor_terms"] = int(n_taylor_terms)
    diagnostics["usable_channel_count"] = int(usable.sum())

    # --- Frequency-moment Hessian ---
    # At point scale, spectral curvature is approximated by H_hat_kl =
    # sum(q_i * x_i**(k+l)), where x_i = (nu_i - nu_0) / nu_0.
    # Use the effective weights supplied by the caller without reading data or
    # preparing an imaging-weight density grid.
    # A common weight normalization does not affect any relative test below.
    try:
        design, _, _ = make_weighted_taylor_design(
            frequencies[usable],
            weights[usable],
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference,
        )
    except ValueError as error:
        fail(str(error))

    # Form D**(-1/2) H_hat D**(-1/2) by normalizing columns of A first.
    # This avoids squaring the raw Taylor dynamic range when forming A.T @ A.
    # Preliminary max normalization also avoids overflow in column norms.
    column_max = np.max(np.abs(design), axis=0)
    if np.any(column_max == 0):
        fail("a Taylor column has no representable spectral support")
    columns = design / column_max
    columns /= np.linalg.norm(columns, axis=0)
    equilibrated = columns.T @ columns
    epsilon = np.finfo(precision).eps

    # ========================================================================
    # 2. Local inversion, rank, and noise diagnostics
    # ========================================================================

    # --- 1. Approximate native inversion safety ---
    # A Cholesky failure predicts that even this spectral approximation cannot
    # support a safe full Taylor solve. Match the native diagonal-relative
    # margin, pivot > 100 * eps * diagonal, on the equilibrated system.
    # This early rejection is labelled approximate; a pass does not certify
    # the later scale-dependent native factorization.
    try:
        cholesky = np.linalg.cholesky(equilibrated)
    except np.linalg.LinAlgError as error:
        fail(f"frequency-moment Hessian is not positive definite ({error})")
    relative_pivots = np.diag(cholesky) ** 2 / np.diag(equilibrated)
    diagnostics["relative_cholesky_pivots"] = relative_pivots.tolist()
    diagnostics["relative_pivot_tolerance"] = float(100 * epsilon)
    if np.any(relative_pivots <= 100 * epsilon):
        fail(
            "frequency-moment Hessian has unsafe relative Cholesky pivots at working precision"
        )

    # --- 2. Approximate positive definiteness and numerical rank ---
    # Unit-diagonal equilibration separates narrow-band moment magnitudes from
    # dependence. Unresolved curvature directions would make the full Taylor
    # solution ambiguous; use the same dimension-scaled heuristic as for PSFs.
    eigenvalues = np.linalg.eigvalsh(equilibrated)
    tolerance = n_taylor_terms * epsilon * np.max(np.abs(eigenvalues))
    diagnostics["equilibrated_hessian"] = equilibrated.tolist()
    diagnostics["equilibrated_eigenvalues"] = eigenvalues.tolist()
    diagnostics["eigenvalue_tolerance"] = float(tolerance)
    if eigenvalues[0] <= tolerance:
        fail(
            "frequency-moment Hessian is numerically rank deficient at working precision"
        )

    # --- 3. Approximate order-zero noise amplification ---
    # Under C_R proportional to H_hat, compare the order-zero variance with a
    # constant-only fit: G_hat_0 = sqrt(H_hat_00 * inverse(H_hat)_00).
    # With unit diagonals this equals sqrt(inverse(equilibrated)_00), or
    # ||solve(L, e_0)||, so no explicit inverse is needed. This differs from
    # the spectral G0's equal-unweighted-channel-noise assumption.
    # As for actual Hessians, three is a permitted amplification factor, not
    # an equivalence to CLEAN stopping depth. The covariance is assumed rather
    # than measured, so exceeding the limit warns instead of rejecting.
    order_zero = np.zeros(n_taylor_terms)
    order_zero[0] = 1.0
    amplification = float(np.linalg.norm(np.linalg.solve(cholesky, order_zero)))
    diagnostics["order_zero_noise_amplification"] = amplification
    diagnostics["order_zero_noise_amplification_limit"] = 3.0
    if amplification > 3.0:
        logger.warning(
            f"Approximate Taylor-Hessian order-zero noise amplification exceeds 3.0: {diagnostics}",
        )

    # ========================================================================
    # 3. Interactions between component positions
    # ========================================================================

    # --- Spatial coupling between delta components ---
    # The local Gram matrix above cannot diagnose coupling between positions.
    # Build Gaussian channel PSF Fourier blocks with FWHM = lambda / Bmax;
    # the same block diagnostic will inspect actual PSFs in the append node.
    from astroviper.processing_functions.imaging.config_check._spatial_hessian import (
        check_gaussian_spatial_hessian,
    )

    baselines = np.broadcast_to(longest_projected_baselines, frequencies.shape)
    diagnostics["spatial_blocks"] = check_gaussian_spatial_hessian(
        frequencies[usable],
        weights[usable],
        baselines[usable],
        n_taylor_terms=n_taylor_terms,
        reference_frequency=reference,
        image_size=image_size,
        cell_size=cell_size,
        precision=precision,
        context=context,
    )
    return diagnostics


# ========================================================================
# Actual model-update products and Taylor Hessians
# ========================================================================


@shares_param_docs
@shares_param_docs
def check_model_products(
    img_xds,
    config,
    *,
    mode,
    image_data_group_in_name="residual",
    image_data_group_out_name="model",
    hessian_cache=None,
):
    """Validate model-update products and all MT-MFS Hessians before compute.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image products; validation never changes their arrays or configuration.
    config : mapping
        Deconvolver, image parameters, and cycle/deconvolution controls.
    mode : ImagingMode
        Imaging kind and field layout.
    image_data_group_in_name, image_data_group_out_name : str
        Data groups naming the residual/PSF/mask and any existing model.

    hessian_cache : mutable mapping or None, optional
        Workflow-owned mutable cache for content-verified PSF/Hessian diagnostics.
        Reuse requires matching PSF contents, coordinates, precision, Taylor basis,
        geometry, scales, scale bias, and cell sampling. Changing-product checks
        always run. None disables reuse; input_checked does not control this cache.

    Returns
    -------
    dict
        Validated effective inputs and separately named Taylor-Hessian diagnostics.
    """
    import toolviper.utils.logger as logger

    mode.require_supported()
    controls = config.get("cycle_control_params", config.get("deconvolve_params", {}))
    deconvolver = config.get("deconvolver", "hogbom")
    groups = img_xds.attrs.get("data_groups", {})
    if image_data_group_in_name not in groups:
        raise KeyError(f"Input data group {image_data_group_in_name!r} is missing.")
    residual_group = groups[image_data_group_in_name]
    residual_name = residual_group.get("sky", "SKY_RESIDUAL")
    psf_name = residual_group.get("point_spread_function", "POINT_SPREAD_FUNCTION")
    if residual_name not in img_xds or psf_name not in img_xds:
        raise KeyError(
            f"Model-update products require {residual_name!r} and {psf_name!r}."
        )
    residual = img_xds[residual_name]
    psf = img_xds[psf_name]
    term_dimension = "taylor_term" if mode.imaging_kind == "continuum" else "frequency"
    residual_dims = ("time", term_dimension, "polarization", "l", "m")
    psf_dims = (
        "time",
        "psf_taylor_order" if mode.imaging_kind == "continuum" else "frequency",
        "polarization",
        "l",
        "m",
    )

    # ========================================================================
    # 1. Product dimensions, Taylor basis, and precision
    # ========================================================================

    # --- Runtime product layouts and precision ---
    # Continuum processing owns its product contract; cube validation stays here.
    model_name = groups.get(image_data_group_out_name, {}).get("sky", "SKY_MODEL")
    if mode.imaging_kind == "continuum":
        from astroviper.processing_functions.imaging.image_continuum_single_field import (
            check_continuum_model_products,
        )

        check_continuum_model_products(
            residual,
            psf,
            deconvolver=deconvolver,
            model=img_xds[model_name] if model_name in img_xds else None,
        )
    else:
        if set(residual.dims) != set(residual_dims) or any(
            residual.sizes[dim] < 1 for dim in residual.dims
        ):
            raise ValueError(
                f"{residual_name} must have nonempty dimensions {residual_dims}."
            )
        if residual.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("Residual products must use float32 or float64.")
    if "single_precision_image" in config:
        expected_dtype = np.dtype(
            "float32" if config["single_precision_image"] else "float64"
        )
        if residual.dtype != expected_dtype:
            raise TypeError(
                f"Residual precision {residual.dtype} differs from configured {expected_dtype}."
            )

    # --- Exact Taylor-product counts ---
    # One common basis fixes the residual/model count and every PSF Taylor order.
    nterms = residual.sizes[term_dimension] if mode.imaging_kind == "continuum" else 1
    expected_psf_count = (
        2 * nterms - 1
        if mode.imaging_kind == "continuum"
        else residual.sizes["frequency"]
    )
    if mode.imaging_kind == "cube" and (
        set(psf.dims) != set(psf_dims)
        or psf.sizes.get(psf_dims[1]) != expected_psf_count
    ):
        raise ValueError(
            f"{psf_name} must have dimensions {psf_dims} with exactly {expected_psf_count} spectral/Taylor planes."
        )
    configured_nterms = config.get("image_params", {}).get("n_taylor_terms", nterms)
    if mode.imaging_kind == "continuum" and configured_nterms != nterms:
        raise ValueError(
            f"Residual has {nterms} Taylor planes; configuration requires exactly {configured_nterms}."
        )

    # --- Taylor-order labels and requested geometry ---
    # Correct counts must also refer to the exact Taylor basis that was checked.
    if mode.imaging_kind == "continuum":
        for product, dimension, count in (
            (residual, "taylor_term", nterms),
            (psf, "psf_taylor_order", expected_psf_count),
        ):
            if dimension in product.coords and not np.array_equal(
                product[dimension], np.arange(count)
            ):
                raise ValueError(
                    f"{dimension} coordinates must be exactly 0 through {count - 1}."
                )
        configured_reference = config.get("image_params", {}).get("reference_frequency")
        product_reference = img_xds.attrs.get("continuum_imaging", {}).get(
            "reference_frequency", img_xds.attrs.get("reference_frequency")
        )
        if (
            configured_reference is not None
            and product_reference is not None
            and configured_reference != product_reference
        ):
            raise ValueError(
                "Product reference frequency differs from the validated Taylor basis."
            )
    requested_geometry = config.get("image_params", {}).get("image_size")
    if requested_geometry is not None and list(requested_geometry) != [
        residual.sizes["l"],
        residual.sizes["m"],
    ]:
        raise ValueError(
            "Product image geometry differs from the validated image_size."
        )
    requested_polarization = config.get("image_params", {}).get("polarization_coords")
    if (
        requested_polarization is not None
        and "polarization" in residual.coords
        and not np.array_equal(requested_polarization, residual.polarization.values)
    ):
        raise ValueError(
            "Product polarization coordinates differ from the validated selection."
        )

    # ========================================================================
    # 2. Product alignment and mask validity
    # ========================================================================

    # --- Shared geometry, coordinates, and precision ---
    # A common shape alone is insufficient if coordinate labels disagree.
    spatial_dims = ("time", "polarization", "l", "m")
    if mode.imaging_kind == "cube":
        for name in (psf_name, model_name):
            if name not in img_xds:
                continue  # A first model update creates the model after validation.
            product = img_xds[name]
            if product.dtype != residual.dtype:
                raise TypeError(
                    f"{name} and residual must have the same working precision."
                )
            if name == model_name and set(product.dims) != set(residual_dims):
                raise ValueError(
                    f"{name} must have residual/model dimensions {residual_dims}."
                )
            if (
                name == model_name
                and product.sizes[term_dimension] != residual.sizes[term_dimension]
            ):
                raise ValueError(f"{name} must contain exactly {nterms} Taylor planes.")
            # Cube frequency labels must match the residual.
            if (
                term_dimension in product.coords
                and term_dimension in residual.coords
                and not np.array_equal(
                    product[term_dimension], residual[term_dimension]
                )
            ):
                raise ValueError(
                    f"{name} and residual have incompatible {term_dimension} coordinates."
                )
            for dim in spatial_dims:
                if dim not in product.dims or product.sizes[dim] != residual.sizes[dim]:
                    raise ValueError(
                        f"{name} and residual have incompatible {dim} geometry."
                    )
                if (
                    dim in product.coords
                    and dim in residual.coords
                    and not np.array_equal(product[dim], residual[dim])
                ):
                    raise ValueError(
                        f"{name} and residual have incompatible {dim} coordinates."
                    )

    # --- Mask compatibility ---
    # Boolean support masks are allowed; numerical masks use image precision.
    mask_name = residual_group.get("mask")
    if mask_name is not None:
        if mask_name not in img_xds:
            raise KeyError(f"Mask variable {mask_name!r} is missing.")
        mask = img_xds[mask_name]
        if not set(mask.dims).issubset(set(residual_dims)):
            raise ValueError("Mask dimensions must be compatible with the residual.")
        if mask.dtype != np.dtype(bool) and mask.dtype != residual.dtype:
            raise TypeError(
                "Floating masks must use the residual precision; Boolean masks are also accepted."
            )
        # A non-finite mask has no defined support. Check the actual values
        # even for trusted configuration, before any comparison or bool cast
        # can hide NaNs. Empty finite support remains valid for derived masks.
        if not np.all(np.isfinite(mask.values)):
            raise ValueError("Mask contains non-finite values.")
        for dim in mask.dims:
            if mask.sizes[dim] not in (1, residual.sizes[dim]):
                raise ValueError(f"Mask has incompatible {dim} geometry.")
            if (
                mask.sizes[dim] != 1
                and dim in mask.coords
                and dim in residual.coords
                and not np.array_equal(mask[dim], residual[dim])
            ):
                raise ValueError(f"Mask has incompatible {dim} coordinates.")

    # ========================================================================
    # 3. Runtime controls and effective scales
    # ========================================================================

    # --- Derived per-plane controls ---
    # Validate every runtime limit before the first image plane can be modified.
    if mode.imaging_kind == "continuum":
        from astroviper.processing_functions.imaging.image_continuum_single_field import (
            check_continuum_model_controls,
        )

        check_continuum_model_controls(controls, residual)
    else:
        expected_control_shape = (
            residual.sizes["time"],
            residual.sizes["frequency"],
            residual.sizes["polarization"],
        )
        for name in ("threshold_per_cycle", "max_iter_per_cycle_updated"):
            if name not in controls:
                continue
            values = require_finite_real(name, controls[name])
            if values.ndim != 0 and values.shape != expected_control_shape:
                raise ValueError(
                    f"{name} must be scalar or have per-plane shape {expected_control_shape}."
                )
            if np.any(values < 0) or (
                name == "max_iter_per_cycle_updated" and values.dtype.kind not in "iu"
            ):
                raise ValueError(f"{name} contains invalid per-plane limits.")

    # --- Validated effective scales ---
    # Record exactly the native scale values and geometry that were checked.
    scales = (
        check_scales(
            controls.get("scales", [0.0]),
            image_size=[residual.sizes["l"], residual.sizes["m"]],
        )
        if deconvolver == "mtmfs"
        else []
    )
    reference_frequency = config.get("image_params", {}).get(
        "reference_frequency", img_xds.attrs.get("reference_frequency")
    )
    diagnostics = {
        "deconvolver": deconvolver,
        "n_taylor_terms": nterms,
        "working_precision": residual.dtype.name,
        "image_geometry": [residual.sizes[dim] for dim in spatial_dims],
        "effective_scales": scales,
        # Persist plain scalars without changing the caller's configuration.
        "reference_frequency": (
            float(reference_frequency) if reference_frequency is not None else None
        ),
        "validated_deconvolution_inputs": {
            name: np.asarray(controls.get(name, default)).tolist()
            for name, default in (
                ("gain", 0.1),
                ("small_scale_bias", 0.0),
                ("mask_threshold", 0.9),
                ("threshold", 0.0),
                ("max_iter", 1000),
            )
        },
        "taylor_hessian_condition_number": [],
        "taylor_hessian_order_zero_noise_amplification": [],
        "taylor_hessian_spatial_blocks": [],
    }
    if deconvolver != "mtmfs":
        return diagnostics

    # ========================================================================
    # 4. Actual local and spatial Taylor-Hessian diagnostics
    # ========================================================================

    # --- Every per-scale Taylor Hessian ---
    # Validate all planes before any kernel can modify a residual or model plane.
    from astroviper.processing_functions.imaging.config_check._spatial_hessian import (
        check_psf_spatial_hessian,
    )
    from astroviper.processing_functions.imaging.deconvolvers import mtmfs

    psf_values = psf.transpose(*psf_dims).values
    hessian_fields = (
        "taylor_hessian_condition_number",
        "taylor_hessian_order_zero_noise_amplification",
        "taylor_hessian_spatial_blocks",
    )
    cache_key = None
    if hessian_cache is not None:
        from collections.abc import MutableMapping
        from copy import deepcopy

        from astroviper.processing_functions.imaging.config_check._hessian_cache import (
            model_hessian_cache_key,
        )

        if not isinstance(hessian_cache, MutableMapping):
            raise TypeError("hessian_cache must be a mutable mapping or None.")
        cache_key = model_hessian_cache_key(
            psf,
            psf_values,
            diagnostics,
            small_scale_bias=controls.get("small_scale_bias", 0.0),
            cell_size=config.get("image_params", {}).get("cell_size"),
        )
        if hessian_cache.get("key") == cache_key:
            # Only static diagnostics are reused. All changing-product checks
            # above have run on the actual arrays of this invocation.
            diagnostics.update(deepcopy(hessian_cache["diagnostics"]))
            return diagnostics

    epsilon = np.finfo(residual.dtype).eps
    for time_index in range(residual.sizes["time"]):
        for polarization_index in range(residual.sizes["polarization"]):
            context = f"time={time_index}, polarization={polarization_index}, nterms={nterms}, scales={scales}"

            # --- 1. Native inversion safety ---
            # The cleaner solves H * delta_c = R for each scale. Cholesky must
            # succeed before any model plane is changed; an invalid inverse
            # would corrupt the update regardless of its order-zero accuracy.
            # mtmfs.hessian uses the same native setup as CLEAN and rejects
            # pivots <= 100 * eps * H_ii. Comparing to each original diagonal
            # avoids rejecting a higher Taylor moment merely for being small.
            # The factor 100 is the existing native round-off safety margin,
            # not a scientific accuracy target; native guards remain active.
            try:
                result = mtmfs.hessian(
                    np.ascontiguousarray(psf_values[time_index, :, polarization_index]),
                    scales=scales,
                    small_scale_bias=controls.get("small_scale_bias", 0.0),
                )
            except (ValueError, RuntimeError) as error:
                raise ValueError(
                    f"Taylor-Hessian preflight failed ({context}): {error}"
                ) from error

            # --- 2. Positive definiteness and numerical rank ---
            # For a least-squares curvature matrix, every nonzero coefficient
            # direction must have positive curvature. Nonpositive directions
            # or numerical dependence make the full Taylor solve unsafe.
            # Taylor moments can differ greatly in magnitude in narrow bands.
            # With D = diag(H), the congruence D**(-1/2) H D**(-1/2) preserves
            # positive definiteness and gives unit diagonals, separating that
            # magnitude difference from dependence between coefficient columns.
            # Eigenvalues within p * eps * max(abs(eigenvalues)) of zero are
            # treated as unresolved. This is a dimension-scaled round-off
            # heuristic for the equilibrated matrix, not a science tolerance
            # or a bound on the forward error of the unscaled native solve.
            hessian = np.asarray(result["hessian"])
            diagonal = np.diagonal(hessian, axis1=1, axis2=2)
            if np.any(~np.isfinite(hessian)) or np.any(diagonal <= 0):
                raise ValueError(
                    f"Taylor-Hessian preflight: Hessians must be finite with positive diagonals ({context})."
                )
            diagonal_sqrt = np.sqrt(diagonal)
            equilibrated = (
                hessian / diagonal_sqrt[:, :, None] / diagonal_sqrt[:, None, :]
            )
            eigenvalues = np.linalg.eigvalsh(equilibrated)
            tolerance = nterms * epsilon * np.max(np.abs(eigenvalues), axis=1)
            if np.any(eigenvalues[:, 0] <= tolerance):
                raise ValueError(
                    f"Taylor-Hessian preflight: a per-scale Hessian is not positive definite and invertible at {residual.dtype} precision ({context}); equilibrated_eigenvalues={eigenvalues.tolist()}."
                )
            if list(result["scales"]) != scales:
                raise ValueError(
                    f"Taylor-Hessian preflight changed the checked effective scales ({context})."
                )
            condition_numbers = np.asarray(result["hessian_condition_number"])
            if np.any(~np.isfinite(condition_numbers)):
                raise ValueError(
                    f"Taylor-Hessian preflight produced non-finite condition numbers ({context})."
                )
            diagnostics["taylor_hessian_condition_number"].append(
                {
                    "time_index": time_index,
                    "polarization_index": polarization_index,
                    "per_scale": condition_numbers.tolist(),
                    "norm": "infinity",
                }
            )

            # --- 3. Order-zero noise amplification ---
            # A large full-Hessian condition number can be caused by higher
            # Taylor terms without compromising c_0. Target the first inverse
            # row: delta_c_0 = e_0.T H**(-1) delta_R. For residual covariance C,
            # Var(c_0) = (H**(-1) C H**(-T))_00. A constant-only update has
            # variance C_00 / H_00**2. If C = tau**2 * H, their standard-
            # deviation ratio simplifies to G_H0 = sqrt(H_00 * (H**(-1))_00).
            # This covariance relation is an explicit diagnostic assumption:
            # arbitrary imaging weights and scale filtering need not satisfy
            # it. We do not estimate C here. Reuse the already computed inverse
            # so this check needs no extra image passes or matrix inversion.
            inverse_hessian = np.asarray(result["inverse_hessian"])
            inverse_order_zero = inverse_hessian[:, 0, 0]
            if np.any(~np.isfinite(inverse_order_zero)) or np.any(
                inverse_order_zero <= 0
            ):
                raise ValueError(
                    f"Taylor-Hessian preflight: inverse order-zero diagonals must be finite and positive ({context})."
                )
            noise_amplification = np.sqrt(hessian[:, 0, 0]) * np.sqrt(
                inverse_order_zero
            )
            if np.any(~np.isfinite(noise_amplification)):
                raise ValueError(
                    f"Taylor-Hessian preflight produced non-finite order-zero noise amplification ({context})."
                )
            # As in spectral preflight, three is an allowed amplification
            # factor motivated by a 3-sigma residual target. It is not an
            # equivalence to CLEAN stopping depth or a prediction of final
            # residual noise. Warn, rather than reject, under this assumption.
            noise_amplification_limit = 3.0
            if np.any(noise_amplification > noise_amplification_limit):
                logger.warning(
                    f"Taylor-Hessian order-zero noise amplification exceeds {noise_amplification_limit} ({context}); covariance_assumption=proportional_to_hessian; per_scale={noise_amplification.tolist()}.",
                )
            diagnostics["taylor_hessian_order_zero_noise_amplification"].append(
                {
                    "time_index": time_index,
                    "polarization_index": polarization_index,
                    "per_scale": noise_amplification.tolist(),
                    "limit": noise_amplification_limit,
                    "covariance_assumption": "proportional_to_hessian",
                }
            )

            # --- Full spatial PSF blocks for delta components ---
            # Local inverses above protect the actual component update, but do
            # not retain interactions between different component positions.
            # Fourier blocks retain those interactions on a periodic patch of
            # paired PSF lags. Diagnose its supported modes before any plane
            # can change; missing spatial modes need not be invertible. The
            # actual-PSF helper handles unmatched edges and FFT roundoff.
            spatial_diagnostics = check_psf_spatial_hessian(
                psf_values[time_index, :, polarization_index],
                precision=residual.dtype,
                cell_size=config.get("image_params", {}).get("cell_size"),
                context={
                    "time_index": time_index,
                    "polarization_index": polarization_index,
                },
            )
            diagnostics["taylor_hessian_spatial_blocks"].append(
                {
                    "time_index": time_index,
                    "polarization_index": polarization_index,
                    **spatial_diagnostics,
                }
            )
    if hessian_cache is not None:
        # Publish only after every plane has passed. Failed checks never create
        # a reusable success record, and image metadata is not used as a cache.
        hessian_cache.clear()
        hessian_cache.update(
            key=cache_key,
            diagnostics={name: deepcopy(diagnostics[name]) for name in hessian_fields},
        )
    return diagnostics


# ========================================================================
# Shared stopping and algorithm-capability helpers
# ========================================================================


def check_max_cycles(value, max_psf_fraction=None, *, allow_integral_float=False):
    """Check cycle limits, retaining the standalone controller's legacy float API.

    Parameters
    ----------
    value : int or {'auto'}
        Cycle limit, including the -1 sentinel and zero; 'auto' requests a
        derived budget.
    max_psf_fraction : float or None, optional
        Upper cycle fraction; if supplied for auto, it must be below one.
    allow_integral_float : bool, optional
        Permit integral floating-point values for the standalone controller API.
    """
    if isinstance(value, str) and value == "auto":
        if max_psf_fraction is not None and max_psf_fraction >= 1:
            raise ValueError("max_cycles='auto' requires max_psf_fraction below 1.")
        return
    if (
        allow_integral_float
        and isinstance(value, float | np.floating)
        and float(value).is_integer()
    ):
        value = int(value)
    try:
        require_integer("max_cycles", value, allow_minus_one=True)
    except (ValueError, TypeError) as error:
        if allow_integral_float:
            raise ValueError(
                'max_cycles must be an integer >= -1 or "auto".'
            ) from error
        raise


def check_hogbom_option(deconvolver, name, value, *, mode=None):
    """Check history/subtraction capabilities, retaining the existing helper API.

    Parameters
    ----------
    deconvolver : str or None
        Selected algorithm, or None when the helper has no algorithm context.
    name : str
        Capability switch: record_iteration_history or exact_psf_subtraction.
    value : bool
        Whether the capability is enabled.
    mode : ImagingMode or None, optional
        Workflow context; continuum does not provide a double-sized subtraction
        PSF.
    """
    require_boolean(name, value)
    # Continuum graphs never provide the double-sized subtraction PSF.
    if (
        value
        and name == "exact_psf_subtraction"
        and mode is not None
        and mode.imaging_kind == "continuum"
    ):
        raise ValueError(
            "exact_psf_subtraction is not supported by continuum imaging "
            f"(deconvolver={deconvolver!r}): the continuum graph makes no PSF "
            "of twice the image size."
        )
    supported = (None, "hogbom", "hogbom_many_threads")
    if name == "record_iteration_history":
        supported = (*supported, "asp", "mtmfs")
    if value and deconvolver not in supported:
        implementations = (
            "Hogbom, Asp, and MT-MFS"
            if name == "record_iteration_history"
            else "Hogbom"
        )
        raise ValueError(
            f"{name} is supported only by {implementations} deconvolvers; received {deconvolver!r}."
        )


def check_entropy_controls(controls):
    """Check the small entropy histogram controls before allocating state.

    Parameters
    ----------
    controls : mapping
        Optional entropy stopping switch, histogram bin counts, and positive SNR
        limit.
    """
    if "entropy_stop" in controls:
        require_boolean("entropy_stop", controls["entropy_stop"])
    for name in ("entropy_spatial_bins", "entropy_flux_bins"):
        if name in controls:
            require_integer(name, controls[name], minimum=1)
    if "entropy_max_snr" in controls:
        require_finite_real(
            "entropy_max_snr", controls["entropy_max_snr"], positive=True, scalar=True
        )


def check_nonnegative(name, value, *, scalar=False):
    """Require a finite non-negative threshold or multiplier.

    Parameters
    ----------
    name : str
        Parameter label included in errors.
    value : array-like
        Finite real thresholds or multipliers, all nonnegative.
    scalar : bool, optional
        Require a zero-dimensional value.
    """
    values = require_finite_real(name, value, scalar=scalar)
    if np.any(values < 0):
        raise ValueError(f"{name} must be non-negative.")


def check_deconvolver(deconvolver, *, mode):
    """Require an implemented algorithm for this imaging mode.

    Parameters
    ----------
    deconvolver : str
        Algorithm name supported by the selected workflow.
    mode : ImagingMode
        Spectral imaging kind and supported field layout.

    Notes
    -----
    Restoration needs the algorithm name without the iteration controls and
    Taylor configuration required when actually updating a model.
    """
    mode.require_supported()
    if mode.imaging_kind == "continuum":
        allowed_deconvolvers = ("hogbom", "asp", "mtmfs")
    else:
        allowed_deconvolvers = (
            "hogbom",
            "hogbom_many_threads",
            "asp",
        )

    if deconvolver not in allowed_deconvolvers:
        raise ValueError(
            f"Deconvolution algorithm {deconvolver!r} not recognized; {mode.imaging_kind} deconvolver must be one of "
            f"{allowed_deconvolvers}; received {deconvolver!r}."
        )


def check_cycle_heuristic_parameters(config):
    """Validate supplied heuristic policy without checking measured residuals.

    Parameters
    ----------
    config : mapping
        Supplied gain, iteration, beam-cutoff, noise, and PSF-fraction controls.
        Measured residual values are checked by their owning heuristics.
    """
    require_mapping("heuristic controls", config)

    # --- Shared iteration policy ---
    # Validate the originals before helpers turn them into Python scalars.
    from astroviper.processing_functions.imaging.config_check._common import (
        CUBE_SINGLE_FIELD,
    )

    controls = {
        name: config[name]
        for name in ("gain", "max_iter", "primary_beam_limit")
        if name in config
    }
    check_model_update({"cycle_control_params": controls}, mode=CUBE_SINGLE_FIELD)

    # --- Noise-based stopping ---
    # A finite positive multiplier preserves the meaning of a noise threshold.
    if "n_sigma" in config:
        try:
            require_finite_real(
                "n_sigma", config["n_sigma"], positive=True, scalar=True
            )
        except ValueError as error:
            raise ValueError("n_sigma must be positive and finite.") from error
    if "noisy" in config:
        require_boolean("noisy", config["noisy"])
    if config.get("noisy", False):
        if config.get("thermal_noise") is None:
            raise ValueError("noisy=True needs a positive thermal_noise")
        require_finite_real(
            "thermal_noise", config["thermal_noise"], positive=True, scalar=True
        )

    # --- Predicted geometric decay ---
    # A fraction in (0, 1) gives a finite, positive logarithmic decay rate.
    if "psf_fraction" in config:
        fraction = require_finite_real(
            "psf_fraction", config["psf_fraction"], scalar=True
        )
        if not 0 < fraction < 1:
            raise ValueError("psf_fraction must lie in (0, 1).")


def check_psf_fit_counts(name, value):
    """Require a usable two-axis fitting grid before interpolation.

    Parameters
    ----------
    name : {'sampling', 'npix_window'}
        Fitting control whose bounds are checked.
    value : array-like of int
        Two counts: sampling requires at least two per axis; windows require
        positive odd counts.

    Notes
    -----
    Sampling needs two points per axis to define a spacing. Fitting windows
    need positive odd counts to retain a central pixel. Refuse fractional
    counts before NumPy grid construction can interpret or truncate them.
    """
    minimum = 2 if name == "sampling" else 1
    values = np.asarray(value)
    if values.shape != (2,) or values.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain two integer counts.")
    if np.any(values < minimum):
        message = (
            "sampling must contain at least two points per axis"
            if name == "sampling"
            else "npix_window must be positive"
        )
        raise ValueError(message)
    if name == "npix_window" and np.any(values % 2 == 0):
        raise ValueError("npix_window must contain odd counts.")
