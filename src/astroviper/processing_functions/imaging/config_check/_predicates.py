"""Shared imaging rules, independent of the parameter-check entry points.

Callers decide when a rule applies and whether a value must be supplied. These
predicates validate original values without converting or changing configuration.
"""

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
    require_integer,
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
