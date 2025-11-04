"""
Iteration Control Module

This module implements iteration control logic for deconvolution processes,
adapted from CASA's iteration control implementation in _gclean.py and
imager_return_dict.py.

Important difference to previous iteration control modules : The ability to
"checkpoint" a state. For massively parallel runs, passing a dict/some other
object containing state is more robust than holding state in the
IterationController object itself.
"""

from collections import namedtuple


# ============================================================================
# Stop Code Definition
# ============================================================================

# StopCode is a namedtuple with (major, minor) codes
StopCode = namedtuple("StopCode", ["major", "minor"])

# Major cycle stop codes
MAJOR_CONTINUE = 0  # Continue major cycles
MAJOR_ITER_LIMIT = 1  # Reached total iteration limit
MAJOR_THRESHOLD = 2  # Peak residual below threshold
MAJOR_CYCLE_LIMIT = 9  # Reached major cycle limit

# Minor cycle stop codes
MINOR_CONTINUE = 0  # Continue minor cycles
MINOR_ITER_LIMIT = 1  # Reached per-cycle iteration limit
MINOR_THRESHOLD = 2  # Peak residual below cyclethreshold
MINOR_DIVERGENCE = 4  # Possible divergence detected
MINOR_ZERO_MASK = 7  # Zero mask (no valid pixels)

# Stop code descriptions
STOPCODE_DESCRIPTIONS = {
    (MAJOR_CONTINUE, MINOR_CONTINUE): "Continue",
    (MAJOR_ITER_LIMIT, MINOR_CONTINUE): "Reached iteration limit",
    (MAJOR_THRESHOLD, MINOR_CONTINUE): "Reached threshold",
    (MAJOR_CYCLE_LIMIT, MINOR_CONTINUE): "Reached major cycle limit",
    (MAJOR_CONTINUE, MINOR_ITER_LIMIT): "Minor cycle iter limit",
    (MAJOR_CONTINUE, MINOR_THRESHOLD): "Minor cycle threshold",
    (MAJOR_CONTINUE, MINOR_DIVERGENCE): "Possible divergence",
    (MAJOR_CONTINUE, MINOR_ZERO_MASK): "Zero mask",
}


def merge_return_dicts():
    pass


def get_peak_residual_from_returndict():
    pass


def get_masksum_from_returndict():
    pass


def get_iterations_done_from_returndict():
    pass


def get_max_psf_sidelobe_from_returndict():
    pass


class IterationController:
    """
    Manages iteration control logic for deconvolution algorithms.

    Uses StopCode namedtuples with (major, minor) to support both
    major cycle and minor cycle convergence decisions.

    Attributes:
    -----------
    niter : int
        Maximum number of total iterations remaining
    nmajor : int
        Maximum number of major cycles remaining (-1 = unlimited)
    threshold : float
        Global stopping threshold (in Jy or image units)
    gain : float
        CLEAN loop gain (typically 0.1)
    cyclefactor : float
        Multiplier for PSF sidelobe to set cyclethreshold
    cycleniter : int
        Maximum iterations per minor cycle (-1 = use niter)

    Tracking:
    major_done : int
        Number of major cycles completed
    total_iter_done : int
        Total iterations completed

    State:
    stopcode : StopCode
        Current stop code (major, minor)
    """

    def __init__():
        pass

    def check_convergence():
        pass

    def update():
        pass

    def reset():
        pass

    def save_state():
        pass

    def load_state():
        pass
