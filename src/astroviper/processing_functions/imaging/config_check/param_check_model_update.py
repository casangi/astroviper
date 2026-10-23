"""Check deconvolver selection, Taylor order, and stopping thresholds."""

from numbers import Integral

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_boolean,
    require_finite_real,
    require_integer,
    require_mapping,
)
from astroviper.utils.param_docs import shares_param_docs


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
    # --- Supported imaging mode ---
    # Reject multifield requests until their model-update path is implemented.
    mode.require_supported()
    require_mapping("config", config)

    deconvolver = config.get("deconvolver")
    image_params = require_mapping("image_params", config.get("image_params", {}))

    # --- Deconvolver selection ---
    # Each imaging kind has its own set of implemented algorithms.
    if mode.imaging_kind == "continuum":
        allowed_deconvolvers = ("hogbom", "mtmfs")
    else:
        allowed_deconvolvers = (
            "hogbom",
            "hogbom_many_threads",
            "asp",
            "aspclean",
            "asp_clean",
        )

    if "deconvolver" in config and deconvolver not in allowed_deconvolvers:
        raise ValueError(
            f"Deconvolution algorithm {deconvolver!r} not recognized; {mode.imaging_kind} deconvolver must be one of "
            f"{allowed_deconvolvers}; received {deconvolver!r}."
        )

    # --- Cube spectral model ---
    # Cube planes are independent images; a multi-term continuum model is invalid.
    if mode.imaging_kind == "cube" and "n_taylor_terms" in image_params:
        require_integer("n_taylor_terms", image_params["n_taylor_terms"], minimum=1)
        if image_params["n_taylor_terms"] != 1:
            raise ValueError("Cube imaging supports only n_taylor_terms=1.")

    # --- Required Taylor order ---
    # An explicit MT-MFS image configuration must specify its number of terms.
    # Leaf model-update calls take image products instead of image_params.
    if (
        mode.imaging_kind == "continuum"
        and deconvolver == "mtmfs"
        and "image_params" in config
        and "n_taylor_terms" not in image_params
    ):
        raise KeyError(
            "image_params must contain 'n_taylor_terms' when deconvolver='mtmfs'."
        )

    if mode.imaging_kind == "continuum" and "n_taylor_terms" in image_params:
        n_taylor_terms = image_params["n_taylor_terms"]

        # --- Taylor-order type ---
        # A term count must be an integer; Boolean values are not counts.
        if isinstance(n_taylor_terms, bool) or not isinstance(n_taylor_terms, Integral):
            raise TypeError("n_taylor_terms must be an integer.")

        # --- Positive Taylor order ---
        # Every continuum model needs at least its constant term.
        if n_taylor_terms < 1:
            raise ValueError("n_taylor_terms must be positive.")

        # --- Högbom compatibility ---
        # Högbom updates only the constant term of a continuum model.
        if deconvolver == "hogbom" and n_taylor_terms != 1:
            raise ValueError("Continuum hogbom requires n_taylor_terms=1.")

        # --- MT-MFS compatibility ---
        # The current continuum MT-MFS path supports two to four terms.
        if deconvolver == "mtmfs" and not 2 <= n_taylor_terms <= 4:
            raise ValueError("Continuum mtmfs requires n_taylor_terms between 2 and 4.")

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
            values = require_finite_real(name, controls[name])
            if (values < 0).any():
                raise ValueError(f"{name} must be non-negative.")

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
        "asp_rc_eps_max",
        "asp_rc_floor_at_threshold",
        "asp_rc_frac_min",
        "asp_rc_noise_floor",
        "asp_rc_switch_factor",
        "asp_significance",
        "asp_significance_gain",
        "fusedthreshold",
        "largestscale",
        "norm_method",
        "psf_width",
        "stoppointmode",
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
        max_cycles = controls["max_cycles"]
        if isinstance(max_cycles, str) and max_cycles == "auto":
            if controls.get("max_psf_fraction", 0.8) >= 1:
                raise ValueError("max_cycles='auto' requires max_psf_fraction < 1.")
        else:
            require_integer("max_cycles", max_cycles, allow_minus_one=True)

    # --- Threshold and PSF policies ---
    # Fractions bound the adaptive cycle threshold; all multipliers are finite.
    for name in (
        "min_psf_fraction",
        "max_psf_fraction",
        "mask_threshold",
        "primary_beam_limit",
    ):
        if name in controls:
            value = require_finite_real(name, controls[name], scalar=True)
            if not 0 <= value <= 1 or (name == "primary_beam_limit" and value == 1):
                raise ValueError(
                    f"{name} must lie in [0, 1] (primary_beam_limit must be < 1)."
                )
    if controls.get("min_psf_fraction", 0.05) > controls.get("max_psf_fraction", 0.8):
        raise ValueError("min_psf_fraction must not exceed max_psf_fraction.")
    for name in ("threshold_sigma", "threshold_tolerance", "psf_sidelobe_factor"):
        if name in controls:
            value = require_finite_real(name, controls[name], scalar=True)
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

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
    if "clean_box" in controls and controls["clean_box"] is not None:
        box = controls["clean_box"]
        if not isinstance(box, tuple) or len(box) != 4:
            raise ValueError(
                "clean_box must be a 4-tuple (xmin, xmax, ymin, ymax) or None."
            )
        if box != (-1, -1, -1, -1):
            for value in box:
                require_integer("clean_box coordinate", value)
            if box[0] > box[1] or box[2] > box[3]:
                raise ValueError("clean_box lower bounds must not exceed upper bounds.")
            if "image_size" in image_params and (
                box[1] >= image_params["image_size"][0]
                or box[3] >= image_params["image_size"][1]
            ):
                raise ValueError("clean_box must fit the image geometry.")

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

    # --- Algorithm-specific Boolean controls ---
    # Iteration history and exact PSF subtraction belong to Högbom only.
    boolean_controls = (
        "record_iteration_history",
        "exact_psf_subtraction",
        "entropy_stop",
        "asp_stop_after_hogbom_run",
        "asp_residual_components",
        "asp_scale_significance",
        "asp_rc_floor_at_threshold",
    )
    for name in boolean_controls:
        if name in controls:
            require_boolean(name, controls[name])
    if deconvolver not in (None, "hogbom", "hogbom_many_threads"):
        for name in ("record_iteration_history", "exact_psf_subtraction"):
            if controls.get(name, False):
                raise ValueError(f"{name} is supported only by Hogbom deconvolvers.")
    if deconvolver in ("hogbom", "hogbom_many_threads", "mtmfs"):
        unsupported = [
            name
            for name in controls
            if name.startswith("asp_")
            or name
            in (
                "fusedthreshold",
                "psf_width",
                "largestscale",
                "stoppointmode",
                "norm_method",
            )
        ]
        if unsupported:
            raise ValueError(f"Asp controls require an Asp deconvolver: {unsupported}.")

    # --- Asp scale and normalization controls ---
    # Preserve documented -1 sentinels, but reject unsupported normalization modes.
    for name in ("largestscale", "stoppointmode"):
        if name in controls:
            require_integer(name, controls[name], allow_minus_one=True)
    if "norm_method" in controls:
        require_integer("norm_method", controls["norm_method"], minimum=1)
        if controls["norm_method"] not in (1, 2):
            raise ValueError("norm_method must be 1 or 2.")
    for name in (
        "fusedthreshold",
        "psf_width",
        "asp_significance",
        "asp_significance_gain",
        "asp_rc_eps_max",
        "asp_rc_frac_min",
        "asp_rc_noise_floor",
        "asp_rc_switch_factor",
    ):
        if name in controls:
            require_finite_real(name, controls[name], scalar=True)

    # --- Entropy stopping ---
    # Bin counts are positive integers; the SNR cutoff must be finite and positive.
    for name in ("entropy_spatial_bins", "entropy_flux_bins"):
        if name in controls:
            require_integer(name, controls[name], minimum=1)
    if "entropy_max_snr" in controls:
        require_finite_real(
            "entropy_max_snr", controls["entropy_max_snr"], positive=True, scalar=True
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


@shares_param_docs
def check_model_products(
    img_xds,
    config,
    *,
    mode,
    image_data_group_in_name="residual",
    image_data_group_out_name="model",
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

    Returns
    -------
    dict
        Validated effective inputs and separately named Taylor-Hessian diagnostics.
    """
    import warnings

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

    # --- Residual dimensions and working precision ---
    # Every time, polarization, and spatial plane participates in validation.
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
    configured_nterms = config.get("image_params", {}).get("n_taylor_terms", nterms)
    if mode.imaging_kind == "continuum" and configured_nterms != nterms:
        raise ValueError(
            f"Residual has {nterms} Taylor planes; configuration requires exactly {configured_nterms}."
        )
    expected_psf_count = (
        2 * nterms - 1
        if mode.imaging_kind == "continuum"
        else residual.sizes["frequency"]
    )
    if (
        set(psf.dims) != set(psf_dims)
        or psf.sizes.get(psf_dims[1]) != expected_psf_count
    ):
        raise ValueError(
            f"{psf_name} must have dimensions {psf_dims} with exactly {expected_psf_count} spectral/Taylor planes."
        )
    # The actual products must fit the backend even when configuration checks
    # were skipped. Check their derived order without revalidating configuration.
    if mode.imaging_kind == "continuum":
        if deconvolver == "hogbom" and nterms != 1:
            raise ValueError("Continuum hogbom requires n_taylor_terms=1.")
        if deconvolver == "mtmfs" and not 2 <= nterms <= 4:
            raise ValueError("Continuum mtmfs requires n_taylor_terms between 2 and 4.")

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
            "reference_frequency_hz", img_xds.attrs.get("reference_frequency")
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

    # --- Shared geometry, coordinates, and precision ---
    # A common shape alone is insufficient if coordinate labels disagree.
    spatial_dims = ("time", "polarization", "l", "m")
    model_name = groups.get(image_data_group_out_name, {}).get("sky", "SKY_MODEL")
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
        # Model Taylor labels and cube frequency labels must match the residual.
        if name == model_name or mode.imaging_kind == "cube":
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

    # --- Derived per-plane controls ---
    # Validate every runtime limit before the first image plane can be modified.
    expected_control_shape = (
        residual.sizes["time"],
        1 if mode.imaging_kind == "continuum" else residual.sizes["frequency"],
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
    diagnostics = {
        "deconvolver": deconvolver,
        "n_taylor_terms": nterms,
        "working_precision": residual.dtype.name,
        "image_geometry": [residual.sizes[dim] for dim in spatial_dims],
        "effective_scales": scales,
        "reference_frequency": config.get("image_params", {}).get(
            "reference_frequency", img_xds.attrs.get("reference_frequency")
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
    }
    if deconvolver != "mtmfs":
        return diagnostics

    # --- Every per-scale Taylor Hessian ---
    # Validate all planes before any kernel can modify a residual or model plane.
    from astroviper.processing_functions.imaging.deconvolvers import mtmfs

    psf_values = psf.transpose(*psf_dims).values
    epsilon = np.finfo(residual.dtype).eps
    for time_index in range(residual.sizes["time"]):
        for polarization_index in range(residual.sizes["polarization"]):
            context = f"time={time_index}, polarization={polarization_index}, nterms={nterms}, scales={scales}"
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
            eigenvalues = np.linalg.eigvalsh(result["hessian"])
            tolerance = nterms * epsilon * np.max(np.abs(eigenvalues), axis=1)
            if np.any(eigenvalues[:, 0] <= tolerance):
                raise ValueError(
                    f"Taylor-Hessian preflight: a per-scale Hessian is not positive definite and invertible at {residual.dtype} precision ({context}); eigenvalues={eigenvalues.tolist()}."
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
            if np.any(condition_numbers > 1 / np.sqrt(epsilon)):
                warnings.warn(
                    f"Poorly conditioned Taylor Hessian ({context}); taylor_hessian_condition_number={condition_numbers.tolist()}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            diagnostics["taylor_hessian_condition_number"].append(
                {
                    "time_index": time_index,
                    "polarization_index": polarization_index,
                    "per_scale": condition_numbers.tolist(),
                    "norm": "infinity",
                }
            )
    return diagnostics
