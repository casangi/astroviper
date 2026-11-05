"""Check frequency metadata and residual-imaging settings."""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    imaging_weighting_scope,
    require_boolean,
    require_finite_real,
    require_mapping,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_briggs_robust,
    check_taylor_count,
)
from astroviper.utils.param_docs import shares_param_docs

# ========================================================================
# Configuration checks
# ========================================================================


@shares_param_docs
def check_residual_update(config, *, mode, require_complete=False):
    """Validate residual-update settings without modifying the configuration.

    Parameters
    ----------
    config : mapping
        Configuration containing ``image_params``, ``imaging_weights_params``,
        and ``continuum_mode`` where applicable. Only supplied keys are checked.
        Frequencies are in Hz.
    mode : ImagingMode
        Imaging kind (cube or continuum) and field layout.

    require_complete : bool, optional
        Require all coordinates and cycle controls needed by a distributed driver.
        Lower-level entry points check only the configuration they receive.

    Raises
    ------
    TypeError
        If a supplied frequency is not real numerical data.
    ValueError
        If frequencies, continuum mode, or weighting scope are invalid.
    NotImplementedError
        If multifield imaging is requested.
    """
    # ========================================================================
    # 1. Supported execution choices
    # ========================================================================

    # --- Supported imaging mode ---
    # Reject multifield requests until their residual-update path is implemented.
    mode.require_supported()
    require_mapping("config", config)

    # --- FFT implementation ---
    # Fail before allocating grids if the requested transform is unavailable.
    if "fft_backend" in config:
        from astroviper.processing_functions.imaging.config_check._common import (
            check_fft_backend,
        )

        check_fft_backend(config["fft_backend"])

    # --- Implemented gridder ---
    # The single-field nodes implement anti-alias gridding only; no fallback.
    if "gridder" in config and config["gridder"] != "anti_alias":
        raise ValueError("Single-field imaging supports only gridder='anti_alias'.")

    # --- MVC model-update approximation ---
    # Each approximation needs its own static products; reject unknown names
    # before creating them rather than silently changing the selected model.
    if "mvc_model_update_approximation" in config:
        approximation = config["mvc_model_update_approximation"]
        if not isinstance(approximation, str) or approximation not in (
            "channel_psf",
            "local_beam_moments",
        ):
            raise ValueError(
                "mvc_model_update_approximation must be 'channel_psf' or "
                "'local_beam_moments'."
            )

    # --- Reusable MVC moments ---
    # This switch determines whether static moments are computed this cycle.
    if "include_local_beam_static" in config:
        require_boolean(
            "include_local_beam_static", config["include_local_beam_static"]
        )

    # ========================================================================
    # 2. Image and spectral configuration
    # ========================================================================

    # --- Required image coordinates ---
    # Full application calls need every coordinate before any output is created.
    image_params = require_mapping("image_params", config.get("image_params", {}))
    # Leaf gridders receive these same settings as separate keyword arguments.
    # A local view lets them reuse this validator without rewriting their inputs.
    image_params = {
        **image_params,
        **{
            name: config[name]
            for name in ("reference_frequency", "n_taylor_terms", "fft_padding")
            if config.get(name) is not None
        },
    }
    if require_complete:
        for name in (
            "image_size",
            "cell_size",
            "phase_direction",
            "frequency_coords",
            "time_coords",
            "polarization_coords",
        ):
            if name not in image_params:
                raise KeyError(f"image_params must contain {name!r}.")

    # --- Channel frequencies ---
    # Selected frequencies must be nonempty, finite, and strictly positive.
    if "frequency_coords" in image_params:
        frequencies = require_finite_real(
            "frequency_coords", image_params["frequency_coords"], positive=True
        )
        if frequencies.ndim != 1:
            raise ValueError("frequency_coords must be one-dimensional.")

    # --- Reference frequency ---
    # The Taylor reference must be one finite, strictly positive frequency.
    if "reference_frequency" in image_params:
        check_reference_frequency(image_params["reference_frequency"])

    # --- Direct Taylor-grid counts ---
    # Gridding helpers need positive counts even without a deconvolver choice.
    for name in ("n_taylor_terms", "n_psf_taylor_terms"):
        value = image_params.get(name, config.get(name))
        if value is not None:
            check_taylor_count(value, name=name)

    # --- Explicit helper precision ---
    # Native complex grids have exactly these two supported storage types.
    if config.get("complex_dtype") is not None:
        if np.dtype(config["complex_dtype"]) not in (
            np.dtype("complex64"),
            np.dtype("complex128"),
        ):
            raise TypeError("complex_dtype must be complex64 or complex128.")

    # --- Direct frequency-mapping policy ---
    # Helpers expose this policy; imaging drivers keep their existing choice.
    for name in ("matching", "frequency_matching"):
        if name in config:
            check_frequency_matching(config[name], name=name)

    # --- Continuum construction ---
    # Continuum residuals can be formed by either the MFS or MVC path.
    continuum_mode = config.get(
        "continuum_mode", image_params.get("continuum_mode", "mfs")
    )
    if mode.imaging_kind == "continuum":
        from astroviper.processing_functions.imaging.config_check._common import (
            check_continuum_mode,
        )

        check_continuum_mode(continuum_mode)

    # --- Cube weighting scope ---
    # Cube imaging does not implement a density shared across all channels.
    if mode.imaging_kind == "cube":
        imaging_weights_params = config.get("imaging_weights_params", {})
        if imaging_weights_params.get("weighting_scope") == "global":
            raise ValueError("Cube imaging does not support weighting_scope='global'.")

    # --- Canonical continuum basis ---
    # Require one explicit Taylor basis. Deprecated spellings are not rewritten.
    if mode.imaging_kind == "continuum" and "image_params" in config:
        if "nterms" in image_params:
            raise ValueError(
                "Use image_params['n_taylor_terms']; nterms is not accepted."
            )
        if require_complete:
            for name in ("reference_frequency", "n_taylor_terms"):
                if name not in image_params:
                    raise KeyError(
                        f"image_params must contain {name!r} for continuum imaging."
                    )

    # ========================================================================
    # 3. Image geometry and time sampling
    # ========================================================================

    # --- Image size ---
    # Both spatial dimensions are positive integer pixel counts.
    if "image_size" in image_params:
        size = np.asarray(image_params["image_size"])
        if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
            raise ValueError(
                "image_size must contain two positive integer pixel counts."
            )
        # PSF fitting reads a coordinate increment; MT-MFS needs two pixels too.
        if np.any(size < 2):
            raise ValueError(
                "Single-field imaging/PSF fitting needs at least 2 pixels per spatial axis."
            )

    # --- Angular cell size ---
    # Signed cells are allowed; both angular increments must be nonzero.
    if "cell_size" in image_params:
        cell_size = require_finite_real(
            "cell_size (radians)", image_params["cell_size"]
        )
        if cell_size.shape != (2,) or (cell_size == 0).any():
            raise ValueError(
                "cell_size must contain two finite, nonzero angles in radians."
            )

    # --- Phase direction ---
    # Declination is bounded by the poles; both angles are expressed in radians.
    if "phase_direction" in image_params:
        direction = require_finite_real(
            "phase_direction (radians)", image_params["phase_direction"]
        )
        if direction.shape != (2,) or abs(direction[1]) > 1.5707963267948966:
            raise ValueError(
                "phase_direction must be [right ascension, declination] in radians with |declination| <= pi/2."
            )

    # --- FFT padding ---
    # The supported padding range must not be changed by a later grid setup.
    if "fft_padding" in image_params:
        padding = require_finite_real(
            "fft_padding", image_params["fft_padding"], scalar=True
        )
        if not 1 <= padding <= 10:
            raise ValueError("fft_padding must be between 1 and 10.")

    # --- Image time coordinates ---
    # Empty or non-finite coordinates cannot define output image planes.
    if "time_coords" in image_params:
        time_coords = require_finite_real("time_coords", image_params["time_coords"])
        if time_coords.ndim != 1:
            raise ValueError("time_coords must be one-dimensional.")
        # The current MVC gridder accumulates all integrations into one plane.
        if (
            mode.imaging_kind == "continuum"
            and continuum_mode == "mvc"
            and time_coords.size != 1
        ):
            raise ValueError("Single-field MVC requires exactly one image time plane.")

    # ========================================================================
    # 4. Feeds, polarization, and primary beams
    # ========================================================================

    # --- Instrument feeds ---
    # A direct model-preparation call may supply the basis without Stokes labels.
    if "instrument_polarization_basis" in config and config[
        "instrument_polarization_basis"
    ] not in ("linear", "circular"):
        raise ValueError(
            "instrument_polarization_basis must be 'linear' or 'circular'."
        )

    # --- MVC beam support ---
    # The cutoff is known before a primary-beam image is available.
    if "primary_beam_limit" in config:
        from astroviper.processing_functions.imaging.config_check._predicates import (
            check_primary_beam_limit,
        )

        check_primary_beam_limit(config["primary_beam_limit"])

    # --- Requested Stokes planes ---
    # Validate supported feed/Stokes combinations when this boundary knows the feeds.
    if (
        "polarization_coords" in image_params
        and "instrument_polarization_basis" in config
    ):
        from astroviper.processing_functions.imaging.utils.imaging_polarization import (
            correlations_for_stokes,
        )

        correlations = correlations_for_stokes(
            image_params["polarization_coords"], config["instrument_polarization_basis"]
        )
        if continuum_mode == "mvc":
            from astroviper.processing_functions.imaging.utils.imaging_polarization import (
                is_four_correlation_basis,
            )

            # Real MVC Taylor images cannot retain the imaginary cross-hand plane.
            if is_four_correlation_basis(correlations):
                raise ValueError(
                    f"continuum_mode='mvc' does not image polarization_coords {list(image_params['polarization_coords'])}: request the two parallel hands or use continuum_mode='mfs'."
                )

    # --- Primary-beam prescription ---
    # Explicit choices and physical apertures are known without a PSF.
    from astroviper.processing_functions.imaging.config_check._primary_beam import (
        check_continuum_primary_beam,
        check_cube_primary_beam,
    )

    if mode.imaging_kind == "continuum":
        check_continuum_primary_beam(image_params)
    elif "primary_beam_params" in config:
        check_cube_primary_beam(config["primary_beam_params"])

    # ========================================================================
    # 5. Gridding kernels and imaging weights
    # ========================================================================

    # --- Kernel design ---
    # Validate schema, padding and cached image-edge correction before gridding.
    if "gridding_kernel_params" in config:
        from astroviper.processing_functions.imaging.config_check._gridding import (
            check_gridding_kernel,
        )

        check_gridding_kernel(
            config["gridding_kernel_params"],
            fft_padding=image_params.get("fft_padding"),
        )

    # --- Weighting method ---
    # Keep uniform as a public method; its kernel recipe must not rewrite inputs.
    weights = require_mapping(
        "imaging_weights_params", config.get("imaging_weights_params", {})
    )
    unknown_weight_parameters = set(weights) - {
        "weighting",
        "weighting_scope",
        "robust",
        "briggs_abs_noise",
        "casa_weighting_implementation",
    }
    if unknown_weight_parameters:
        raise ValueError(
            f"Unknown imaging-weight parameters: {sorted(unknown_weight_parameters)}."
        )
    weighting = weights.get("weighting", "natural")
    if weighting not in ("natural", "uniform", "briggs", "briggs_abs"):
        raise ValueError("weighting must be natural, uniform, briggs, or briggs_abs.")

    # --- Weight-density scope ---
    # Validate the supplied value even when natural weighting does not use it.
    # Resolution selects execution without rewriting the caller's dictionary.
    if "weighting_scope" in weights and weights["weighting_scope"] not in (
        "local",
        "global",
    ):
        raise ValueError("weighting_scope must be 'local' or 'global'.")
    # Absolute Briggs uses per-channel density; continuum Briggs may use global.
    scope = imaging_weighting_scope(weights, mode=mode)
    if weighting == "briggs_abs" and scope == "global":
        raise ValueError("briggs_abs requires weighting_scope='local'.")

    # --- Briggs robustness ---
    # A requested Briggs scheme must provide its finite, supported robustness.
    if weighting in ("briggs", "briggs_abs"):
        if "robust" not in weights:
            raise KeyError(
                "Briggs weighting requires imaging_weights_params['robust']."
            )
        check_briggs_robust(weights["robust"])

    # --- Absolute Briggs noise ---
    # The absolute-noise recipe requires a positive noise scale in Jy.
    if weighting == "briggs_abs":
        if "briggs_abs_noise" not in weights:
            raise KeyError(
                "briggs_abs requires imaging_weights_params['briggs_abs_noise']."
            )
        require_finite_real(
            "briggs_abs_noise (Jy)",
            weights["briggs_abs_noise"],
            positive=True,
            scalar=True,
        )

    # --- Correlation-weight equalization ---
    # Do not interpret strings or numbers as a request for CASA equalization.
    if "casa_weighting_implementation" in weights:
        require_boolean(
            "casa_weighting_implementation", weights["casa_weighting_implementation"]
        )


# ========================================================================
# Source metadata and inferred primary beams
# ========================================================================


def check_input_metadata(source, config, *, mode, measurement_set="input"):
    """Check selected source structure without loading visibility samples.

    Parameters
    ----------
    source : xarray.Dataset
        Lazy Measurement Set dataset. Only coordinates, dimensions and dtypes
        are inspected; visibility, flag, weight and UVW values are not read.
    config : mapping
        Image coordinates, input data-group name and continuum mode.
    mode : ImagingMode
        Explicit imaging kind and field layout.
    measurement_set : str
        Dataset name included in failures.
    """
    mode.require_supported()
    group_name = config.get("processing_set_data_group_name", "corrected")
    group = source.attrs.get("data_groups", {}).get(group_name)

    # --- Spectral units ---
    # Metadata are compared in Hz; never interpret another unit as Hz.
    units = source.frequency.attrs.get("units", ["Hz"])
    if set(np.atleast_1d(units)) != {"Hz"}:
        raise ValueError(
            f"Selected frequencies in {measurement_set!r} must be expressed in Hz; received units={units!r}."
        )

    # --- Role targets ---
    # A registered role is insufficient if its variable is absent on disk.
    if group is None:
        raise KeyError(f"{measurement_set}: data group {group_name!r} is missing.")
    for role in ("correlated_data", "weight", "flag", "uvw"):
        if role not in group or group[role] not in source:
            raise KeyError(
                f"{measurement_set}: data group {group_name!r} lacks {role!r} or its variable."
            )

    # --- Sample axes ---
    # These shapes are metadata, so mismatches need no visibility I/O to find.
    sample_dims = ("time", "baseline_id", "frequency", "polarization")
    reference = source[group["correlated_data"]]
    for role in ("correlated_data", "weight", "flag"):
        array = source[group[role]]
        if set(array.dims) != set(sample_dims) or array.ndim != 4:
            raise ValueError(
                f"{measurement_set}: {role} must have dimensions {sample_dims}."
            )
        if dict(array.sizes) != dict(reference.sizes):
            raise ValueError(
                f"{measurement_set}: {role} shape differs from correlated_data."
            )
    uvw = source[group["uvw"]]
    if uvw.dims != ("time", "baseline_id", "uvw_label") or uvw.sizes["uvw_label"] != 3:
        raise ValueError(
            f"{measurement_set}: uvw must have dimensions (time, baseline_id, uvw_label) with three components."
        )
    for dim in ("time", "baseline_id"):
        if uvw.sizes[dim] != reference.sizes[dim]:
            raise ValueError(
                f"{measurement_set}: uvw {dim} extent differs from correlated_data."
            )

    # --- Numerical storage ---
    # Inspect dtypes only; finite-value predicates require loaded sample arrays.
    for role, kinds in (
        ("correlated_data", "c"),
        ("weight", "iuf"),
        ("flag", "biu"),
        ("uvw", "iuf"),
    ):
        if source[group[role]].dtype.kind not in kinds:
            raise TypeError(
                f"{measurement_set}: unsupported {role} dtype {source[group[role]].dtype}."
            )
    for dim in ("frequency", "polarization"):
        if dim not in source.coords or source[dim].dims != (dim,):
            raise ValueError(
                f"{measurement_set}: {dim} must be a one-dimensional coordinate."
            )

    # --- Channel assignment ---
    # Use the production nearest-channel rule, preserving averaging/shifted axes.
    image = config.get("image_params", {})
    continuum_mode = config.get("continuum_mode", image.get("continuum_mode", "mfs"))
    if "frequency_coords" in image and (
        mode.imaging_kind == "cube" or continuum_mode == "mvc"
    ):
        from astroviper.processing_functions.imaging.config_check._common import (
            selected_frequency_mask,
        )
        from astroviper.processing_functions.imaging.utils.frequency_mapping import (
            map_visibility_frequencies_to_image,
        )

        frequencies = np.asarray(source.frequency.values)
        selected = selected_frequency_mask(
            frequencies, image["frequency_coords"], mode=mode
        )
        if np.any(selected):
            map_visibility_frequencies_to_image(
                frequencies[selected], image["frequency_coords"], input_checked=True
            )


# ========================================================================
# Selected visibility data predicates: require loaded sample arrays
# ========================================================================


def check_selected_fields(fields, *, mode):
    """Reject a multi-field selection before reading sample arrays."""
    mode.require_supported()
    if mode.field_layout == "single_field" and len(fields) > 1:
        raise ValueError(
            f"Single-field imaging selected multiple fields: {sorted(fields)}."
        )


def absent_visibility_rows(weights, uvw, *, time_centroid=None):
    """Identify padding without modifying the supplied arrays.

    Missing time and UVW markers must agree with non-contributing weights.
    XRADIO padding weights may be zero or NaN after sigma-to-weight conversion;
    positive, negative, or infinite weights must not hide corrupt observations.
    """
    if time_centroid is None:
        return np.zeros(weights.shape[:2], dtype=bool)
    return (
        ~np.isfinite(time_centroid)
        & np.isnan(uvw).all(axis=-1)
        & ((weights == 0) | np.isnan(weights)).all(axis=(2, 3))
    )


def check_statistical_weights(weights, active, *, context):
    """Require finite, non-negative statistical weights for active samples."""
    if np.any((~np.isfinite(weights) | (weights < 0)) & active):
        details = "; ".join(f"{name}={value!r}" for name, value in context.items())
        raise ValueError(
            "Spectral-design preflight: unflagged weights must be finite and non-negative; "
            f"{details}, usable/distinct/rank/condition=unavailable."
        )


def check_visibility_uvw(uvw, *, measurement_set):
    """Reject invalid UVW after known absent rows have been excluded."""
    if not np.all(np.isfinite(uvw)):
        raise ValueError(
            f"Spectral-design preflight: non-finite UVW coordinates in {measurement_set!r}."
        )


def check_effective_weights(weights, active, *, context):
    """Check computed imaging weights before accumulating spectral constraints."""
    if np.any((~np.isfinite(weights) | (weights < 0)) & active):
        raise ValueError(
            "Spectral-design preflight: negative or non-finite effective imaging weights "
            f"in {context['measurement_set']!r}; channel_selection={context['channel_selection']}, "
            f"polarization={context['polarization']}."
        )


def check_selected_samples(samples):
    """Require selected samples before constructing spectral diagnostics."""
    if not samples:
        raise ValueError(
            "Spectral-design preflight: no source channels match the requested frequency selection."
        )


def check_continuum_beam_metadata(image_params, antenna_xds, *, continuum_mode):
    """Check inferred continuum apertures before output creation.

    Parameters
    ----------
    image_params : mapping
        Unchanged caller configuration, including optional explicit apertures.
    antenna_xds : xarray.Dataset
        Small combined antenna metadata dataset.
    continuum_mode : {"mfs", "mvc"}
        Beam-selection context, shared by all frequency tasks.
    """
    from astroviper.processing_functions.imaging.config_check._primary_beam import (
        check_continuum_primary_beam,
    )
    from astroviper.processing_functions.imaging.primary_beam.continuum_primary_beam import (
        resolve_continuum_primary_beam,
    )

    # --- Physical antenna metadata ---
    # A missing/mixed aperture cannot be inferred from a single common dish.
    if (
        image_params.get("list_dish_diameters") is None
        and "ANTENNA_DISH_DIAMETER" in antenna_xds
    ):
        require_finite_real(
            "ANTENNA_DISH_DIAMETER",
            antenna_xds.ANTENNA_DISH_DIAMETER.values,
            positive=True,
        )

    # --- Inferred prescription ---
    # The existing resolver checks common diameters and ALMA aperture support.
    # It returns a copy; validate that copy without changing the caller config.
    resolved = resolve_continuum_primary_beam(
        image_params, antenna_xds, continuum_mode=continuum_mode
    )
    check_continuum_primary_beam(resolved)


# ========================================================================
# Spectral sampling and numerical diagnostics
# ========================================================================


def check_spectral_sampling(
    frequencies,
    effective_weights,
    *,
    n_taylor_terms,
    reference_frequency,
    precision="float64",
    context=None,
):
    """Check the weighted Taylor design and return reproducible diagnostics.

    Parameters
    ----------
    frequencies, effective_weights : array-like
        Selected frequencies in Hz and nonnegative relative sample weights.
        Metadata-only preflight supplies unit weights: flags and actual imaging
        weights are not known at that stage. Callers with prepared weights may
        supply them directly. Repeated frequencies add weight, not independent
        Taylor constraints. Neither input is modified.
    n_taylor_terms : int
        Number of Taylor terms required by the model.
    reference_frequency : float
        Reference frequency in Hz.
    precision : {"float32", "float64"}
        Requested image precision, recorded as context. Numerical rank is
        assessed in float64, independently of the image storage precision.
    context : mapping, optional
        Field, spectral-window, channel-selection, and polarization identifiers.

    Returns
    -------
    dict
        Channel counts, numerical rank, singular values, frequency range, and
        explicitly named 2-norm spectral-design condition number and reciprocal,
        plus order-zero noise amplification relative to a constant-only fit.
        Rank singular values and their tolerance describe the column-normalized
        distinct-frequency design. Condition numbers describe the original
        Taylor basis and are diagnostic, not rejection thresholds.

    Raises
    ------
    ValueError
        If metadata, weights, or the weighted design are invalid.
    """
    import warnings

    from astroviper.processing_functions.imaging.config_check._spectral import (
        make_weighted_taylor_design,
        prepare_spectral_samples,
    )

    # ========================================================================
    # 1. Diagnostic context and selected-data sanity
    # ========================================================================

    # --- Diagnostic context ---
    # Keep the same identifying information on successful and failed checks.
    diagnostics = {
        "field": None,
        "spectral_window": None,
        "channel_selection": None,
        "polarization": None,
        **(context or {}),
        "n_taylor_terms": n_taylor_terms,
        "reference_frequency": reference_frequency,
        "usable_channel_count": 0,
        "distinct_frequency_count": 0,
        "spectral_design_rank": None,
        "spectral_design_condition_number": None,
        "spectral_design_reciprocal_condition_number": None,
    }

    def fail(reason):
        raise ValueError(f"Spectral-design preflight: {reason}; {diagnostics}")

    # --- Frequency and weight metadata ---
    # Rank is meaningful only for finite, positive frequencies and valid weights.
    try:
        frequencies, weights, reference_value, usable = prepare_spectral_samples(
            frequencies,
            effective_weights,
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference_frequency,
            precision=precision,
        )
    except (TypeError, ValueError) as error:
        fail(str(error))
    diagnostics["reference_frequency"] = reference_value
    diagnostics["n_taylor_terms"] = int(n_taylor_terms)

    # ========================================================================
    # 2. Independent frequency constraints and numerical rank
    # ========================================================================

    # --- Independent usable samples ---
    # Zero-weight channels contribute neither constraints nor frequency support.
    usable_frequencies = frequencies[usable]
    diagnostics["usable_channel_count"] = int(usable.sum())
    diagnostics["distinct_frequency_count"] = int(np.unique(usable_frequencies).size)
    if not np.any(usable):
        fail("no selected channel has positive effective imaging weight")
    diagnostics["frequency_range_hz"] = [
        float(usable_frequencies.min()),
        float(usable_frequencies.max()),
    ]

    # --- Enough independent frequency constraints ---
    # A degree-(p-1) polynomial needs at least p distinct frequency constraints.
    # Repeating a frequency cannot remove an exact null direction. Reject an
    # underdetermined request rather than changing its Taylor order or basis.
    if diagnostics["distinct_frequency_count"] < n_taylor_terms:
        fail("fewer distinct positive-weight frequencies than requested Taylor terms")

    # --- Distinct-frequency design ---
    # Combine weights only for rank assessment. Equal-frequency rows are
    # proportional, and replacing them by one row with their summed weight
    # preserves A.T @ A. Repeating the complete input therefore leaves both
    # the normalized design and its rank tolerance unchanged.
    unique_frequencies, inverse = np.unique(usable_frequencies, return_inverse=True)
    usable_weights = weights[usable].astype(np.float64)
    usable_weights = usable_weights / usable_weights.max()
    combined_weights = np.zeros(unique_frequencies.size, dtype=np.float64)
    np.add.at(combined_weights, inverse, usable_weights)
    try:
        rank_design, _, _ = make_weighted_taylor_design(
            unique_frequencies,
            combined_weights,
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference_value,
        )
    except ValueError as error:
        fail(str(error))

    # --- Column normalization ---
    # A_ik = sqrt(q_i) * x_i**k maps physical coefficients c to observations.
    # With D_kk = ||A_:k||_2 and B = A D^-1, Ac = Bz for z = Dc. Nonzero
    # column scaling preserves rank while separating polynomial dependence from
    # the naturally different magnitudes of intensity, slope, and curvature.
    # Scaling by each column's maximum first keeps its norm representable.
    column_maximum = np.max(np.abs(rank_design), axis=0)
    if np.any(column_maximum == 0):
        fail("a Taylor design column has no representable frequency constraint")
    scaled_design = rank_design / column_maximum
    scaled_design /= np.linalg.norm(scaled_design, axis=0)
    rank_singular_values = np.linalg.svd(scaled_design, compute_uv=False)

    # --- Numerical resolution of the diagnostic ---
    # For B = U Sigma V.T, subtracting sigma_min * u_min * v_min.T removes
    # its weakest direction. Thus sigma_min is the spectral-norm distance to
    # loss of full column rank. Compare it with max(M, p) * eps64 * ||B||_2,
    # where M is the number of distinct frequencies and p the Taylor count.
    # This conventional SVD rank tolerance accounts approximately for roundoff
    # in the float64 diagnostic. The dimension factor is a numerical convention,
    # not a derived uncertainty bound for the imaging pipeline. Image precision,
    # visibility noise, flags, weighting, and spatial PSFs require their own
    # diagnostics; passing this check does not certify coefficient accuracy.
    tolerance = (
        max(scaled_design.shape) * np.finfo(np.float64).eps * rank_singular_values[0]
    )
    rank = int(np.count_nonzero(rank_singular_values > tolerance))
    diagnostics.update(
        spectral_design_rank=rank,
        spectral_design_rank_singular_values=rank_singular_values.tolist(),
        spectral_design_rank_tolerance=float(tolerance),
        spectral_design_rank_precision="float64",
        spectral_design_rank_scaling="unit_column_norm",
        working_precision=np.dtype(precision).name,
    )
    if rank != n_taylor_terms:
        fail("the column-normalized Taylor design is numerically rank deficient")

    # --- Conditioning in the physical Taylor basis ---
    # Retain every original sample for noise propagation: grouping repeated
    # frequencies for rank must not change the assumed independent noise model.
    try:
        design, normalized_weights, sqrt_weights = make_weighted_taylor_design(
            usable_frequencies,
            weights[usable],
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference_value,
        )
    except ValueError as error:
        fail(str(error))
    singular_values = np.linalg.svd(design, compute_uv=False)
    reciprocal_condition = float(singular_values[-1] / singular_values[0])
    diagnostics.update(
        spectral_design_singular_values=singular_values.tolist(),
        spectral_design_reciprocal_condition_number=reciprocal_condition,
        spectral_design_condition_number=(
            1.0 / reciprocal_condition if reciprocal_condition else None
        ),
    )

    # ========================================================================
    # 3. Noise amplification and reference placement
    # ========================================================================

    # --- Order-zero noise amplification ---
    # A small singular value may describe slope or curvature with little effect
    # on intensity at the reference frequency. Target c_0 instead of bounding
    # the error of the entire coefficient vector by 1 / sigma_min.
    # With A_ik = sqrt(q_i) * x_i**k, the first pseudoinverse row h gives
    # delta_c_0 = h @ delta_b. For independent, equal-noise unweighted channels
    # (standard deviation sigma), delta_b_i = sqrt(q_i) * noise_i, so
    # Var(c_0) = sigma**2 * sum(q_i * h_i**2).
    # A constant-only fit has Var(reference) = sigma**2 * sum(q_i**2) / W**2,
    # where W = sum(q_i). Their standard-deviation ratio is therefore
    # G_0 = W * sqrt(sum(q_i * h_i**2)) / sqrt(sum(q_i**2)).
    # Absolute flux and noise cancel. Compute h from the thin SVD, avoiding
    # normal equations, which square the condition number.
    # Solve in normalized coefficient coordinates and transform back: since
    # A = B D, A^+ = D^-1 B^+. This preserves the original coefficient/noise
    # meaning while avoiding division by raw tiny higher-order columns.
    sample_maximum = np.max(np.abs(design), axis=0)
    normalized_design = design / sample_maximum
    sample_norm = np.linalg.norm(normalized_design, axis=0)
    normalized_design /= sample_norm
    left_vectors, scaled_values, right_vectors_transposed = np.linalg.svd(
        normalized_design, full_matrices=False
    )
    order_zero_row = (
        (right_vectors_transposed[:, 0] / scaled_values) @ left_vectors.T
    ) / (sample_maximum[0] * sample_norm[0])
    noise_amplification = float(
        normalized_weights.sum()
        * np.linalg.norm(order_zero_row * sqrt_weights)
        / np.linalg.norm(normalized_weights)
    )
    # Three is an explicit noise-amplification policy motivated by a 3-sigma
    # residual target, not a derived equivalence to CLEAN stopping depth. Warn
    # rather than reject: unequal channel noise, correlations, model bias, and
    # spatial imaging effects are not represented by this spectral diagnostic.
    noise_amplification_limit = 3.0
    diagnostics.update(
        taylor_zero_noise_amplification=noise_amplification,
        taylor_zero_noise_amplification_limit=noise_amplification_limit,
    )
    if noise_amplification > noise_amplification_limit:
        warnings.warn(
            f"Taylor order-zero noise amplification exceeds {noise_amplification_limit}: {diagnostics}",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- Reference-frequency placement ---
    # Outside the usable frequency hull the Taylor model is extrapolating.
    if (
        reference_value < usable_frequencies.min()
        or reference_value > usable_frequencies.max()
    ):
        warnings.warn(
            f"Reference frequency is outside the usable frequency range: {diagnostics}",
            RuntimeWarning,
            stacklevel=2,
        )

    return diagnostics


# ========================================================================
# Shared Taylor-basis and reference-frequency helpers
# ========================================================================


def check_continuum_basis(image_params):
    """Require the common Taylor order and reference frequency for map helpers."""
    from astroviper.processing_functions.imaging.config_check._common import (
        CONTINUUM_SINGLE_FIELD,
    )

    require_mapping("image_params", image_params)
    check_residual_update({"image_params": image_params}, mode=CONTINUUM_SINGLE_FIELD)
    for name in ("n_taylor_terms", "reference_frequency"):
        if name not in image_params:
            raise KeyError(f"image_params must contain {name!r}.")


def check_reference_frequency(value, *, allow_one_element=False):
    """Check a shared reference, allowing legacy one-element helper inputs."""
    values = value
    if allow_one_element:
        values = np.asarray(value)
        if values.size == 1:
            values = values.reshape(())
    try:
        require_finite_real("reference_frequency", values, positive=True, scalar=True)
    except (ValueError, TypeError) as error:
        if allow_one_element:
            raise ValueError(
                "The reference frequency must be scalar, finite and positive."
            ) from error
        raise


def check_weight_density_method(params):
    """Require a density-based method for density gridding and degridding.

    Natural weighting has no density grid. Absolute Briggs weighting uses a
    different recipe; these continuum helpers implement Briggs and uniform only.
    General weighting validation remains in ``check_residual_update``.
    """
    require_mapping("imaging_weights_params", params)
    weighting = params.get("weighting", "natural")
    if weighting not in ("briggs", "uniform"):
        raise ValueError(
            "Weight-density operations require Briggs or uniform weighting; natural and briggs_abs are not supported."
        )


# ========================================================================
# Geometry heuristic arguments
# ========================================================================


def check_pixels_per_beam(value):
    """Require finite, positive sampling across the fitted minor beam axis."""
    require_finite_real("pixels_per_beam", value, positive=True, scalar=True)


def check_beam_level(value, *, name="level"):
    """A beam boundary must lie strictly between zero and peak power."""
    level = require_finite_real(name, value, scalar=True)
    if not 0 < level < 1:
        raise ValueError(f"{name} must lie in (0, 1).")


def check_telescope_prescription(telescope, *, default_robust=False):
    """Require a supported prescription after observatory metadata is interpreted.

    ngVLA has a beam model but no pipeline robustness prescription. Do not
    silently substitute another telescope's robustness.
    """
    if default_robust:
        if telescope not in ("alma", "aca", "vla"):
            raise ValueError(
                f"no default robust for telescope {telescope!r}; pass robust explicitly"
            )
    elif telescope not in ("ALMA", "ACA", "VLA", "EVLA", "NGVLA"):
        raise ValueError(
            f"no primary beam model for telescope {telescope!r}; expected ALMA, "
            "ACA, VLA, EVLA or ngVLA"
        )


def check_delphi_parameters(config):
    """Check supplied Delphi controls before metadata access or its PSF pass.

    Image geometry and cycle thresholds are derived later and must be checked
    as a new imaging configuration. This check never resolves defaults.
    """
    require_mapping("config", config)

    # --- Sampling and stopping policy ---
    # Both controls are positive physical multipliers, not zero sentinels.
    if "pixels_per_beam" in config:
        check_pixels_per_beam(config["pixels_per_beam"])
    if "n_sigma" in config:
        require_finite_real("n_sigma", config["n_sigma"], positive=True, scalar=True)

    # --- Field-of-view boundary ---
    # A beam level strictly between zero and its peak defines a finite radius.
    if "image_edge_primary_beam_limit" in config:
        check_beam_level(
            config["image_edge_primary_beam_limit"],
            name="image_edge_primary_beam_limit",
        )

    # --- Weighting recipe ---
    # Delphi derives missing Briggs robustness from the telescope metadata.
    if config.get("weighting", "briggs") not in ("natural", "uniform", "briggs"):
        raise ValueError("Delphi weighting must be natural, uniform, or briggs.")
    if config.get("robust") is not None:
        check_briggs_robust(config["robust"])


def check_frequency_matching(value, *, name="matching"):
    """Require the shared cube/continuum frequency-assignment policy.

    Checking this enum does not certify the coordinates: mapping still checks
    dimensions, finiteness, coverage, and uniqueness against the actual data.
    """
    if value not in ("nearest", "exact"):
        raise ValueError(f"{name} must be 'nearest' or 'exact'.")
