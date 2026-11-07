"""Shared imaging rules, independent of the parameter-check entry points.

Callers decide when a rule applies and whether a value must be supplied. These
predicates validate original values without converting or changing configuration.
"""

from astroviper.processing_functions.imaging.config_check._common import (
    require_boolean,
    require_finite_real,
    require_integer,
    require_mapping,
)

# -----------------------------------------------------------------------------
# Fractional beam support and weighting controls
# -----------------------------------------------------------------------------


def check_primary_beam_limit(value):
    """Require a finite scalar cutoff in the inclusive interval [0, 1].

    Parameters
    ----------
    value : float
        Finite scalar primary-beam cutoff in [0, 1].
    """
    limit = require_finite_real("primary_beam_limit", value, scalar=True)
    if not 0 <= limit <= 1:
        raise ValueError("primary_beam_limit must lie in [0, 1].")


def check_briggs_robust(value):
    """Require a finite scalar Briggs robustness within the supported [-2, 2].

    Parameters
    ----------
    value : float
        Finite scalar robustness in [-2, 2].
    """
    robust = require_finite_real("robust", value, scalar=True)
    if not -2 <= robust <= 2:
        raise ValueError("robust must be between -2 and 2.")


# -----------------------------------------------------------------------------
# Least-squares solve
# -----------------------------------------------------------------------------


def check_least_squares_params(value):
    """Require ``least_squares_params`` with a ``max_steps`` of at least 1.

    Parameters
    ----------
    value : mapping
        The parameters of the least-squares solve.
    """
    require_mapping("least_squares_params", value)
    unknown = sorted(set(value) - {"max_steps", "smooth_fwhm_pixels"})
    if unknown:
        raise ValueError(f"Unknown least_squares_params keys: {unknown}.")
    if "max_steps" not in value:
        raise ValueError("least_squares_params needs max_steps.")
    require_integer("max_steps", value["max_steps"], minimum=1)
    if "smooth_fwhm_pixels" in value:
        require_finite_real(
            "smooth_fwhm_pixels",
            value["smooth_fwhm_pixels"],
            positive=True,
            scalar=True,
        )


# -----------------------------------------------------------------------------
# Automatic mask
# -----------------------------------------------------------------------------


def check_auto_mask_params(value):
    """Require known ``auto_mask_params`` keys with values in range.

    Keys left out take the defaults of
    :func:`~astroviper.processing_functions.image_analysis.make_auto_mask.make_auto_mask`.

    Parameters
    ----------
    value : mapping
        The automatic mask parameters.
    """
    from astroviper.processing_functions.image_analysis.make_auto_mask import (
        AUTO_MASK_DEFAULTS,
    )

    require_mapping("auto_mask_params", value)
    unknown = sorted(set(value) - set(AUTO_MASK_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown auto_mask_params keys: {unknown}.")

    for key in ("grow_prune", "fast_noise"):
        if key in value:
            require_boolean(key, value[key])
    if "grow_iterations" in value:
        require_integer("grow_iterations", value["grow_iterations"])

    # --- Numerical ranges ---
    # smooth_factor scales the smoothing kernel, which needs a width. cut_threshold is a
    # fraction of the smoothed peak: 0 keeps the whole reach of the kernel, 1 nothing.
    if "min_percent_change" in value:
        require_finite_real(
            "min_percent_change", value["min_percent_change"], scalar=True
        )
    if "smooth_factor" in value:
        require_finite_real(
            "smooth_factor", value["smooth_factor"], positive=True, scalar=True
        )
    for key in (
        "sidelobe_threshold",
        "noise_threshold",
        "low_noise_threshold",
        "negative_threshold",
        "min_beam_fraction",
    ):
        if key in value and require_finite_real(key, value[key], scalar=True) < 0:
            raise ValueError(f"{key} must not be negative.")
    if (
        "cut_threshold" in value
        and not 0
        < require_finite_real("cut_threshold", value["cut_threshold"], scalar=True)
        < 1
    ):
        raise ValueError("cut_threshold must lie in (0, 1).")


# -----------------------------------------------------------------------------
# Taylor-basis counts
# -----------------------------------------------------------------------------


def check_taylor_count(value, *, name="n_taylor_terms"):
    """Require a positive integer term count, rejecting Booleans and fractions.

    Parameters
    ----------
    value : int
        Strictly positive integral Taylor count; Booleans are rejected.
    name : str, optional
        Parameter label included in errors.

    Notes
    -----
    Backend-specific bounds are separate from this basic basis requirement.
    """
    try:
        require_integer(name, value, minimum=1)
    except ValueError as error:
        raise ValueError(f"{name} must be positive.") from error


# -----------------------------------------------------------------------------
# Shared restoration product selection
# -----------------------------------------------------------------------------


def check_restoration_output(value):
    """Require one restoration product selector for cube and continuum.

    Parameters
    ----------
    value : {"apparent", "primary_beam_corrected", "both"}
        Restored intensity products to compute. Booleans and aliases are rejected.

    Raises
    ------
    ValueError
        The selector is not one of the three supported strings.
    """
    if not isinstance(value, str) or value not in (
        "apparent",
        "primary_beam_corrected",
        "both",
    ):
        raise ValueError(
            "restoration_output must be 'apparent', 'primary_beam_corrected', or 'both'."
        )
