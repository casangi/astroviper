"""Check frequency metadata and residual-imaging settings."""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    imaging_weighting_scope,
    require_boolean,
    require_finite_real,
    require_mapping,
)
from astroviper.utils.param_docs import shares_param_docs


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
    # --- Supported imaging mode ---
    # Reject multifield requests until their residual-update path is implemented.
    mode.require_supported()
    require_mapping("config", config)

    # --- Required image coordinates ---
    # Full application calls need every coordinate before any output is created.
    image_params = require_mapping("image_params", config.get("image_params", {}))
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
        require_finite_real(
            "reference_frequency",
            image_params["reference_frequency"],
            positive=True,
            scalar=True,
        )

    # --- Continuum construction ---
    # Continuum residuals can be formed by either the MFS or MVC path.
    if mode.imaging_kind == "continuum" and "continuum_mode" in config:
        if config["continuum_mode"] not in ("mfs", "mvc"):
            raise ValueError("continuum_mode must be 'mfs' or 'mvc'.")

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
        if "reference_frequency_hz" in image_params:
            raise ValueError(
                "Use image_params['reference_frequency'] in Hz; reference_frequency_hz is not accepted."
            )
        if require_complete:
            for name in ("reference_frequency", "n_taylor_terms"):
                if name not in image_params:
                    raise KeyError(
                        f"image_params must contain {name!r} for continuum imaging."
                    )

    # --- Image size ---
    # Both spatial dimensions are positive integer pixel counts.
    if "image_size" in image_params:
        size = np.asarray(image_params["image_size"])
        if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
            raise ValueError(
                "image_size must contain two positive integer pixel counts."
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

    # --- Requested Stokes planes ---
    # Validate supported feed/Stokes combinations when this boundary knows the feeds.
    if (
        "polarization_coords" in image_params
        and "instrument_polarization_basis" in config
    ):
        from astroviper.processing_functions.imaging.utils.imaging_polarization import (
            correlations_for_stokes,
        )

        correlations_for_stokes(
            image_params["polarization_coords"], config["instrument_polarization_basis"]
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
    # Absolute Briggs uses per-channel density; continuum Briggs may use global.
    scope = imaging_weighting_scope(weights, mode=mode)
    if scope not in ("local", "global"):
        raise ValueError("weighting_scope must be 'local' or 'global'.")
    if weighting == "briggs_abs" and scope == "global":
        raise ValueError("briggs_abs requires weighting_scope='local'.")

    # --- Briggs robustness ---
    # A requested Briggs scheme must provide its finite, supported robustness.
    if weighting in ("briggs", "briggs_abs"):
        if "robust" not in weights:
            raise KeyError(
                "Briggs weighting requires imaging_weights_params['robust']."
            )
        robust = require_finite_real("robust", weights["robust"], scalar=True)
        if not -2 <= robust <= 2:
            raise ValueError("robust must be between -2 and 2.")

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
        Selected channel frequencies in Hz and summed effective imaging weights.
        Flags and correlation selection must already have been applied. Repeated
        frequencies are permitted, but do not add independent Taylor samples.
    n_taylor_terms : int
        Number of Taylor terms required by the model.
    reference_frequency : float
        Reference frequency in Hz.
    precision : {"float32", "float64"}
        Working image precision used for the numerical-rank tolerance.
    context : mapping, optional
        Field, spectral-window, channel-selection, and polarization identifiers.

    Returns
    -------
    dict
        Channel counts, numerical rank, singular values, frequency range, and
        explicitly named 2-norm spectral-design condition number and reciprocal.

    Raises
    ------
    ValueError
        If metadata, weights, or the weighted design are invalid.
    """
    import warnings

    from astroviper.processing_functions.imaging.config_check._common import (
        require_integer,
    )

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
        frequencies = require_finite_real(
            "selected frequencies", frequencies, positive=True
        )
        weights = require_finite_real("effective imaging weights", effective_weights)
        reference_value = require_finite_real(
            "reference_frequency", reference_frequency, positive=True, scalar=True
        )
        require_integer("n_taylor_terms", n_taylor_terms, minimum=1)
    except (TypeError, ValueError) as error:
        fail(str(error))
    if frequencies.ndim != 1 or weights.shape != frequencies.shape:
        fail(
            "frequencies and effective weights must be matching one-dimensional arrays"
        )
    if np.any(weights < 0):
        fail("effective imaging weights must be non-negative")
    if np.dtype(precision) not in (np.dtype("float32"), np.dtype("float64")):
        fail("working precision must be float32 or float64")

    # Use a scalar numerical value for arithmetic; an xarray scalar's ufunc
    # dispatch would otherwise try to retain its zero-dimensional coordinates.
    # The supplied reference object and caller configuration remain unchanged.
    reference_value = float(reference_value)
    diagnostics["reference_frequency"] = reference_value
    diagnostics["n_taylor_terms"] = int(n_taylor_terms)

    # --- Independent usable samples ---
    # Zero-weight channels contribute neither constraints nor frequency support.
    usable = weights > 0
    usable_frequencies = frequencies[usable]
    diagnostics["usable_channel_count"] = int(usable.sum())
    diagnostics["distinct_frequency_count"] = int(np.unique(usable_frequencies).size)
    if not np.any(usable):
        fail("no selected channel has positive effective imaging weight")
    diagnostics["frequency_range_hz"] = [
        float(usable_frequencies.min()),
        float(usable_frequencies.max()),
    ]

    # --- Weighted Taylor design ---
    # A common weight scale cancels from the singular-value ratio and avoids
    # overflowing the square root for otherwise valid large imaging weights.
    x = (usable_frequencies.astype(np.float64) - reference_value) / reference_value
    with np.errstate(over="ignore", invalid="ignore"):
        design = np.vander(x, N=n_taylor_terms, increasing=True)
        design *= np.sqrt(weights[usable] / weights[usable].max())[:, None]
    if not np.all(np.isfinite(design)):
        fail("the weighted Taylor design overflows at this reference frequency")
    singular_values = np.linalg.svd(design, compute_uv=False)
    epsilon = np.finfo(precision).eps
    tolerance = max(design.shape) * epsilon * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    reciprocal_condition = (
        float(singular_values[-1] / singular_values[0])
        if rank == n_taylor_terms
        else 0.0
    )
    diagnostics.update(
        spectral_design_rank=rank,
        spectral_design_singular_values=singular_values.tolist(),
        spectral_design_reciprocal_condition_number=reciprocal_condition,
        spectral_design_condition_number=(
            1.0 / reciprocal_condition if reciprocal_condition else None
        ),
        working_precision=np.dtype(precision).name,
        spectral_design_rank_tolerance=float(tolerance),
    )
    if diagnostics["distinct_frequency_count"] < n_taylor_terms:
        fail("fewer distinct positive-weight frequencies than requested Taylor terms")
    if rank != n_taylor_terms:
        fail("the weighted Taylor design is numerically rank deficient")

    # --- Conditioning warning ---
    # A square-root-epsilon warning exposes loss of precision without imposing
    # an arbitrary bandwidth cutoff on a design that is still full rank.
    if reciprocal_condition < np.sqrt(epsilon):
        warnings.warn(
            f"Poorly conditioned spectral design: {diagnostics}",
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
