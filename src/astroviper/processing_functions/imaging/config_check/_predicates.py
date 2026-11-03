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
    """Require a finite scalar cutoff in [0, 1) to retain beam support."""
    limit = require_finite_real("primary_beam_limit", value, scalar=True)
    if not 0 <= limit < 1:
        raise ValueError("primary_beam_limit must lie in [0, 1).")


def check_briggs_robust(value):
    """Require a finite scalar Briggs robustness within the supported [-2, 2]."""
    robust = require_finite_real("robust", value, scalar=True)
    if not -2 <= robust <= 2:
        raise ValueError("robust must be between -2 and 2.")


# -----------------------------------------------------------------------------
# Taylor-basis counts
# -----------------------------------------------------------------------------


def check_taylor_count(value, *, name="n_taylor_terms"):
    """Require a positive integer term count, rejecting Booleans and fractions.

    Backend-specific bounds are separate from this basic basis requirement.
    """
    try:
        require_integer(name, value, minimum=1)
    except ValueError as error:
        raise ValueError(f"{name} must be positive.") from error
