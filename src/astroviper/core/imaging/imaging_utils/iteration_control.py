"""
Iteration Control Module for Deconvolution Algorithms

This module implements iteration control logic for deconvolution processes,
adapted from CASA's iteration control implementation in _gclean.py and
imager_return_dict.py. It has been streamlined to work with AstroViper's
ReturnDict structure while retaining all original functionality.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from collections import namedtuple


from astroviper.core.imaging.imaging_utils.return_dict import (
    ReturnDict,
    Key,
    FIELD_ACCUM,
    FIELD_SINGLE_VALUE,
)

HAVE_RETURNDICT = True

# Stop codes matching CASA's implementation
# See: https://casadocs.readthedocs.io/en/stable/notebooks/synthesis_imaging.html#Returned-Dictionary
# CASA Source: imager_return_dict.py:9-10, lines 463-565
#
# CASA uses a namedtuple StopCodes(major, minor) to separate major cycle and minor cycle
# convergence criteria. This "deals with the degeneracy in stopcode numbers, while still
# using the same definitions"

StopCode = namedtuple("StopCode", ["major", "minor"])

# Major cycle stop codes (global convergence)
MAJOR_CONTINUE = 0  # Continue major cycles
MAJOR_ITER_LIMIT = 1  # Reached total iteration limit (niter)
MAJOR_THRESHOLD = 2  # Peak residual below global threshold
MAJOR_ZERO_MASK = 7  # Zero mask (no valid pixels)
MAJOR_CYCLE_LIMIT = 9  # Reached major cycle limit (nmajor)

# Minor cycle stop codes (per-cycle convergence)
MINOR_CONTINUE = 0  # Continue minor cycles
MINOR_ITER_LIMIT = 1  # Reached per-cycle iteration limit (cycleniter)
MINOR_THRESHOLD = 2  # Peak residual below cyclethreshold
MINOR_DIVERGENCE = 4  # Possible divergence detected
MINOR_ZERO_MASK = 7  # Zero mask detected during minor cycle

# Stop code descriptions for major cycle codes
MAJOR_STOPCODE_DESCRIPTIONS = {
    MAJOR_CONTINUE: "Continue iterations",
    MAJOR_ITER_LIMIT: "Reached the iteration limit",
    MAJOR_THRESHOLD: "Reached global stopping threshold (within mask)",
    MAJOR_ZERO_MASK: "Zero mask",
    MAJOR_CYCLE_LIMIT: "Reached the major cycle limit (nmajor)",
}

# Stop code descriptions for minor cycle codes
MINOR_STOPCODE_DESCRIPTIONS = {
    MINOR_CONTINUE: "Continue minor cycle",
    MINOR_ITER_LIMIT: "Reached cycle iteration limit",
    MINOR_THRESHOLD: "Reached cycle threshold",
    MINOR_DIVERGENCE: "Possible divergence detected",
    MINOR_ZERO_MASK: "Zero mask",
}


# ============================================================================
# ReturnDict Utility Functions
# ============================================================================


def merge_return_dicts(
    return_dicts: List[ReturnDict],
    merge_strategy: str = "update",
) -> ReturnDict:
    """
    Merge multiple ReturnDict objects into a single ReturnDict.

    This is essential for dask workflows where each node processes a subset
    of (time, pol, chan) combinations and returns its own ReturnDict. Before
    making iteration control decisions, we need to merge all results.

    Merge Strategies:
    -----------------
    - "update" (default): Merge dictionaries at the value level if keys conflict. For
      FIELD_ACCUM fields (peakres, iter_done, etc.), concatenates history lists.
      For FIELD_SINGLE_VALUE fields, replaces with latest value. Use when
      different nodes may update different fields for the same plane.

    - "latest": If the same (time, pol, chan) key appears in multiple dicts,
      keep the value from the last dict in the list. Use when dicts represent
      sequential updates.

    - "error": Raise an error if any (time, pol, chan) key appears in multiple
      dicts. Use when each node should process unique planes.

    Parameters:
    -----------
    return_dicts : list of ReturnDict
        List of ReturnDict objects to merge

    merge_strategy : str, optional
        Strategy for handling conflicting keys. Options:
        - "update" (default): Merge value dicts, later entries overwrite earlier
        - "error": Raise error on conflicts
        - "latest": Use value from last dict with this key

    Returns:
    --------
    merged : ReturnDict
        Merged ReturnDict containing all data from input dicts

    Raises:
    -------
    ValueError
        If merge_strategy is "error" and conflicting keys are found

    Example:
    --------
    >>> # Dask workflow: 3 workers process different channels
    >>> rd1 = ReturnDict()
    >>> rd1.add({'peakres': 0.5, 'iter_done': 100}, time=0, pol=0, chan=0)
    >>>
    >>> rd2 = ReturnDict()
    >>> rd2.add({'peakres': 0.3, 'iter_done': 120}, time=0, pol=0, chan=1)
    >>>
    >>> rd3 = ReturnDict()
    >>> rd3.add({'peakres': 0.4, 'iter_done': 110}, time=0, pol=0, chan=2)
    >>>
    >>> # Merge results
    >>> merged = merge_return_dicts([rd1, rd2, rd3])
    >>> # merged now has 3 entries, one for each channel
    """
    if not return_dicts:
        return ReturnDict()

    merged = ReturnDict()

    for rd in return_dicts:
        for key, value in rd.data.items():
            if key in merged.data:
                # Key conflict - apply merge strategy
                if merge_strategy == "latest":
                    # Overwrite with latest value
                    merged.data[key] = value
                elif merge_strategy == "error":
                    raise ValueError(
                        f"Conflicting key found during merge: {key}. "
                        f"Use merge_strategy='latest' or 'update' to handle conflicts."
                    )
                elif merge_strategy == "update":
                    # Merge dictionaries - handle FIELD_ACCUM specially
                    if isinstance(merged.data[key], dict) and isinstance(value, dict):
                        # Merge field by field, concatenating lists for FIELD_ACCUM
                        for field, field_value in value.items():
                            if field in FIELD_ACCUM:
                                if field in merged.data[key]:
                                    existing = merged.data[key][field]

                                    if not isinstance(existing, list):
                                        existing = [existing]
                                    if not isinstance(field_value, list):
                                        field_value = [field_value]

                                    merged.data[key][field] = existing + field_value
                                else:
                                    # First occurrence - ensure it's a list
                                    merged.data[key][field] = (
                                        field_value
                                        if isinstance(field_value, list)
                                        else [field_value]
                                    )
                            else:
                                # Single-value field - replace
                                merged.data[key][field] = field_value
                    else:
                        merged.data[key] = value
                else:
                    raise ValueError(f"Unknown merge_strategy: {merge_strategy}")
            else:
                # No conflict - just add
                merged.data[key] = value

    return merged


def get_peak_residual_from_returndict(
    return_dict: ReturnDict,
    use_mask: bool = True,
    time: Optional[int] = None,
    pol: Optional[int] = None,
    chan: Optional[int] = None,
) -> float:
    """
    Extract peak residual from ReturnDict structure.

    This adapts CASA's get_peakres() from imager_return_dict.py
    to work with AstroViper's ReturnDict which uses (time, pol, chan) indexing.

    Parameters:
    -----------
    return_dict : ReturnDict
        ReturnDict instance containing deconvolution statistics

    use_mask : bool, optional
        If True, use 'peakres' (masked). If False, use 'peakres_nomask' (default: True)

    time : int, optional
        Filter by specific time index (None = all times)

    pol : int, optional
        Filter by specific polarization index (None = all pols)

    chan : int, optional
        Filter by specific channel index (None = all chans)

    Returns:
    --------
    peak_residual : float
        Maximum peak residual across selected planes (latest value from history)
        Returns 0.0 if no valid data found
    """
    selected = return_dict.sel(time=time, pol=pol, chan=chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    peak = 0.0
    key = "peakres" if use_mask else "peakres_nomask"

    for entry in selected:
        if entry is not None and key in entry:
            value = entry[key]
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                # Use latest value (last in list)
                value = value[-1]

            # Only consider planes with valid mask when using mask
            if use_mask and "masksum" in entry:
                masksum = entry["masksum"]
                # Handle masksum as list or single value
                if isinstance(masksum, list):
                    masksum = masksum[-1] if len(masksum) > 0 else 0
                if masksum == 0:
                    continue

            peak = max(peak, abs(value))

    return peak


def get_masksum_from_returndict(
    return_dict: ReturnDict,
    time: Optional[int] = None,
    pol: Optional[int] = None,
    chan: Optional[int] = None,
) -> float:
    """
    Calculate total mask sum from ReturnDict structure.

    Parameters:
    -----------
    return_dict : ReturnDict
        ReturnDict instance containing mask statistics

    time : int, optional
        Filter by specific time index (None = all times)

    pol : int, optional
        Filter by specific polarization index (None = all pols)

    chan : int, optional
        Filter by specific channel index (None = all chans)

    Returns:
    --------
    total_masksum : float
        Sum of latest mask values across selected planes
        Returns 0.0 if no mask data found
    """
    selected = return_dict.sel(time=time, pol=pol, chan=chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    total_masksum = 0.0
    for entry in selected:
        if entry is not None and "masksum" in entry:
            # Extract value (handle both list and single value for backward compatibility)
            value = entry["masksum"]
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                # Use latest value (last in list)
                value = value[-1]
            total_masksum += value

    return total_masksum


def get_iterations_done_from_returndict(
    return_dict: ReturnDict,
    time: Optional[int] = None,
    pol: Optional[int] = None,
    chan: Optional[int] = None,
) -> int:
    """
    Calculate total iterations done from ReturnDict structure.

    Parameters:
    -----------
    return_dict : ReturnDict
        ReturnDict instance containing iteration statistics

    time : int, optional
        Filter by specific time index (None = all times)

    pol : int, optional
        Filter by specific polarization index (None = all pols)

    chan : int, optional
        Filter by specific channel index (None = all chans)

    Returns:
    --------
    total_iterations : int
        Sum of all iterations done across entire history and selected planes
    """
    selected = return_dict.sel(time=time, pol=pol, chan=chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    total_iters = 0
    for entry in selected:
        if entry is not None and "iter_done" in entry:
            # Extract value (handle both list and single value for backward compatibility)
            value = entry["iter_done"]
            if isinstance(value, list):
                # Sum entire history
                total_iters += sum(value)
            else:
                # Single value (backward compatibility)
                total_iters += value

    return total_iters


def get_max_psf_sidelobe_from_returndict(
    return_dict: ReturnDict,
    time: Optional[int] = None,
    pol: Optional[int] = None,
    chan: Optional[int] = None,
) -> float:
    """
    Extract maximum PSF sidelobe level from ReturnDict.

    This should be populated by psf_fitting.py analysis and stored
    in the ReturnDict as 'max_psf_sidelobe'.

    Parameters:
    -----------
    return_dict : ReturnDict
        ReturnDict instance containing PSF analysis results

    time : int, optional
        Filter by specific time index (None = all times)

    pol : int, optional
        Filter by specific polarization index (None = all pols)

    chan : int, optional
        Filter by specific channel index (None = all chans)

    Returns:
    --------
    max_sidelobe : float
        Maximum PSF sidelobe level across selected planes
        Returns 0.2 (conservative default) if not found
    """
    selected = return_dict.sel(time=time, pol=pol, chan=chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    max_sidelobe = 0.0
    found_any = False

    for entry in selected:
        if entry is not None and "max_psf_sidelobe" in entry:
            max_sidelobe = max(max_sidelobe, entry["max_psf_sidelobe"])
            found_any = True

    # If no PSF sidelobe info found, return conservative default
    if not found_any:
        return 0.2  # Typical value for real observations

    return max_sidelobe


# ============================================================================
# IterationController Class
# ============================================================================


class IterationController:
    """
    Manages iteration control logic for deconvolution algorithms.

    The controller extracts needed statistics (peak residual, masksum, iterations
    done, etc.) from ReturnDict internally, so callers don't need to manually
    pass individual values.

    Attributes:
    -----------
    niter : int
        Maximum number of minor cycle iterations remaining
    nmajor : int
        Maximum number of major cycles remaining (-1 = unlimited)
    threshold : float
        Global stopping threshold (in Jy or image units)
    gain : float
        CLEAN loop gain (typically 0.1)
    cyclefactor : float
        Multiplier for PSF sidelobe to set cyclethreshold
    minpsffraction : float
        Minimum PSF fraction for cyclethreshold calculation
    maxpsffraction : float
        Maximum PSF fraction for cyclethreshold calculation
    cycleniter : int
        Maximum iterations per minor cycle (-1 = use niter)
    nsigma : float
        N-sigma threshold for stopping (0 = disabled)

    Major Cycle Tracking:
    ---------------------
    major_done : int
        Number of major cycles completed so far
    total_iter_done : int
        Total number of minor cycle iterations completed

    Convergence State:
    ------------------
    stopcode : StopCode
        Current stop code as namedtuple (major, minor)
        Access via stopcode.major and stopcode.minor
    stopdescription : str
        Human-readable description of stop reason
    """

    def __init__(
        self,
        niter: int = 1000,
        nmajor: int = -1,
        threshold: float = 0.0,
        gain: float = 0.1,
        cyclefactor: float = 1.0,
        minpsffraction: float = 0.05,
        maxpsffraction: float = 0.8,
        cycleniter: int = -1,
        nsigma: float = 0.0,
    ):
        """
        Initialize the iteration controller with deconvolution parameters.

        Parameters:
        -----------
        niter : int, optional
            Maximum total number of CLEAN iterations (default: 1000)

        nmajor : int, optional
            Maximum number of major cycles (default: -1 for unlimited)

        threshold : float, optional
            Global stopping threshold in Jy (default: 0.0)

        gain : float, optional
            CLEAN loop gain, range (0, 1] (default: 0.1)

        cyclefactor : float, optional
            Multiplier for adaptive cyclethreshold (default: 1.0)

        minpsffraction : float, optional
            Minimum PSF sidelobe fraction (default: 0.05)

        maxpsffraction : float, optional
            Maximum PSF sidelobe fraction (default: 0.8)

        cycleniter : int, optional
            Max iterations per minor cycle (default: -1)

        nsigma : float, optional
            N-sigma threshold for stopping (default: 0.0, disabled)
        """
        # Iteration limits
        self._initial_niter = niter
        self.niter = niter
        self.nmajor = nmajor

        # Threshold parameters
        self.threshold = threshold
        self.nsigma = nsigma

        # CLEAN parameters
        self.gain = gain
        self.cyclefactor = cyclefactor
        self.minpsffraction = minpsffraction
        self.maxpsffraction = maxpsffraction
        self.cycleniter = cycleniter

        # Tracking state
        self.major_done = 0
        self.total_iter_done = 0

        # Convergence state (namedtuple matching CASA)
        self.stopcode = StopCode(major=MAJOR_CONTINUE, minor=MINOR_CONTINUE)
        self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_CONTINUE]

    def calculate_cycle_controls(
        self,
        return_dict: ReturnDict,
        time: Optional[int] = None,
        pol: Optional[int] = None,
        chan: Optional[int] = None,
    ) -> Tuple[int, float]:
        """
        Calculate cycleniter and cyclethreshold for the next minor cycle.

        Logic:
        ------
        1. Extract max_psf_sidelobe and peak_residual from return_dict
        2. Start with remaining iterations (niter)
        3. If cycleniter is set (>= 0), use minimum of (cycleniter, niter)
        4. Calculate PSF fraction = max_psf_sidelobe * cyclefactor
        5. Clamp PSF fraction to [minpsffraction, maxpsffraction]
        6. cyclethreshold = max(psf_fraction * peak_residual, threshold)

        Parameters:
        -----------
        return_dict : ReturnDict
            ReturnDict containing deconvolution statistics including:
            - 'max_psf_sidelobe': Maximum PSF sidelobe level
            - 'peakres': Current peak residual

        time : int, optional
            Filter by specific time index

        pol : int, optional
            Filter by specific polarization index

        chan : int, optional
            Filter by specific channel index

        Returns:
        --------
        use_cycleniter : int
            Number of iterations to perform in this minor cycle

        cyclethreshold : float
            Stopping threshold for this minor cycle

        Example:
        --------
        >>> controller = IterationController(niter=1000, cyclefactor=1.5)
        >>> # return_dict populated by deconvolver and PSF analysis
        >>> cycleniter, cyclethresh = controller.calculate_cycle_controls(return_dict)
        """
        # Extract needed values from ReturnDict
        max_psf_sidelobe = get_max_psf_sidelobe_from_returndict(
            return_dict, time=time, pol=pol, chan=chan
        )
        peak_residual = get_peak_residual_from_returndict(
            return_dict, use_mask=True, time=time, pol=pol, chan=chan
        )

        # Start with all remaining iterations
        use_cycleniter = self.niter

        # If user forced a specific cycleniter, respect it
        if self.cycleniter >= 0:
            use_cycleniter = min(self.cycleniter, use_cycleniter)

        # Calculate adaptive PSF fraction for cyclethreshold
        psf_fraction = max_psf_sidelobe * self.cyclefactor

        # Clamp to user-specified bounds
        psf_fraction = max(psf_fraction, self.minpsffraction)
        psf_fraction = min(psf_fraction, self.maxpsffraction)

        # Set cyclethreshold as fraction of current peak residual
        cyclethreshold = psf_fraction * peak_residual
        cyclethreshold = max(cyclethreshold, self.threshold)

        return int(use_cycleniter), cyclethreshold

    def check_convergence(
        self,
        return_dict: ReturnDict,
        time: Optional[int] = None,
        pol: Optional[int] = None,
        chan: Optional[int] = None,
    ) -> Tuple[StopCode, str]:
        """
        Check if deconvolution has converged based on multiple criteria.

        This implements CASA's convergence checking from imager_return_dict.py:463-565.

        CASA uses a namedtuple StopCode(major, minor) to separate major cycle and
        minor cycle convergence criteria. This handles the "degeneracy in stopcode
        numbers" (CASA comment line 476-477).

        Major Cycle Stopping Criteria (in order of precedence):
        --------------------------------------------------------
        1. Zero mask (stopcode 7): No valid pixels to clean
        2. Iteration limit (stopcode 1): niter <= 0
        3. Threshold reached (stopcode 2): peak_residual <= threshold
        4. Major cycle limit (stopcode 9): nmajor == 0 (if not -1)

        Minor Cycle Stopping Criteria:
        -------------------------------
        - Checked by deconvolver (cycleniter, cyclethreshold)
        - Can be propagated via return_dict if needed

        Parameters:
        -----------
        return_dict : ReturnDict
            ReturnDict containing deconvolution statistics including:
            - 'peakres': Current peak residual
            - 'masksum': Sum of mask (number of valid pixels)

        time : int, optional
            Filter by specific time index

        pol : int, optional
            Filter by specific polarization index

        chan : int, optional
            Filter by specific channel index

        Returns:
        --------
        stopcode : StopCode
            Named tuple StopCode(major, minor) with integer codes
            Access via stopcode.major and stopcode.minor
            major=0, minor=0 means continue

        stopdescription : str
            Human-readable description of stop reason

        Example:
        --------
        >>> controller = IterationController(niter=100, threshold=0.01)
        >>> # After running deconvolution...
        >>> stopcode, desc = controller.check_convergence(return_dict)
        >>> if stopcode.major != 0:
        >>>     print(f"Converged: {desc}")
        >>> # Check both major and minor
        >>> if stopcode.major != 0 or stopcode.minor != 0:
        >>>     print(f"Stopped: major={stopcode.major}, minor={stopcode.minor}")
        """
        # Extract needed values from ReturnDict
        peak_residual = get_peak_residual_from_returndict(
            return_dict, use_mask=True, time=time, pol=pol, chan=chan
        )
        masksum = get_masksum_from_returndict(
            return_dict, time=time, pol=pol, chan=chan
        )

        # Initialize stop codes
        stopcode_maj = MAJOR_CONTINUE
        stopcode_min = MINOR_CONTINUE

        # Check major cycle stopping criteria (in priority order)

        # Priority 1: Check for zero mask
        if masksum == 0:
            stopcode_maj = MAJOR_ZERO_MASK
            self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_ZERO_MASK]
        # Priority 2: Check iteration limit
        elif self.niter <= 0:
            stopcode_maj = MAJOR_ITER_LIMIT
            self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_ITER_LIMIT]
        # Priority 3: Check threshold (with tolerance matching CASA)
        elif self.threshold > 0 and peak_residual <= self.threshold:
            stopcode_maj = MAJOR_THRESHOLD
            self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_THRESHOLD]
        # Priority 4: Check major cycle limit
        elif self.nmajor != -1 and self.nmajor <= 0:
            stopcode_maj = MAJOR_CYCLE_LIMIT
            self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_CYCLE_LIMIT]
        else:
            # No stopping criteria met
            self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_CONTINUE]

        # Update internal state
        self.stopcode = StopCode(major=stopcode_maj, minor=stopcode_min)

        return self.stopcode, self.stopdescription

    def update_counts(
        self,
        return_dict: ReturnDict,
        time: Optional[int] = None,
        pol: Optional[int] = None,
        chan: Optional[int] = None,
    ) -> None:
        """
        Update iteration counts after a major cycle completes.

        Updates:
        --------
        1. Extracts iterations_done from return_dict
        2. Decrements niter by iterations_done
        3. Decrements nmajor by 1 (if not -1)
        4. Increments major_done and total_iter_done
        5. Enforces floor values (no negatives)

        Parameters:
        -----------
        return_dict : ReturnDict
            ReturnDict containing iteration statistics including:
            - 'iter_done': Number of iterations completed in this major cycle

        time : int, optional
            Filter by specific time index

        pol : int, optional
            Filter by specific polarization index

        chan : int, optional
            Filter by specific channel index

        Example:
        --------
        >>> controller = IterationController(niter=1000, nmajor=5)
        >>> # After major cycle completes...
        >>> controller.update_counts(return_dict)
        >>> print(controller.niter, controller.nmajor, controller.major_done)
        900 4 1
        """
        # Only update if not converged (check both major and minor)
        if (
            self.stopcode.major != MAJOR_CONTINUE
            or self.stopcode.minor != MINOR_CONTINUE
        ):
            return

        # Extract iterations done from ReturnDict
        iterations_done = get_iterations_done_from_returndict(
            return_dict, time=time, pol=pol, chan=chan
        )

        # Decrement major cycle count (if not unlimited)
        if self.nmajor != -1:
            self.nmajor -= 1
            if self.nmajor < 0:
                self.nmajor = 0

        # Decrement iteration count
        self.niter -= iterations_done
        if self.niter < 0:
            self.niter = 0

        # Update tracking counters
        self.major_done += 1
        self.total_iter_done += iterations_done

    def update_parameters(
        self,
        niter: Optional[int] = None,
        cycleniter: Optional[int] = None,
        nmajor: Optional[int] = None,
        threshold: Optional[float] = None,
        cyclefactor: Optional[float] = None,
    ) -> Tuple[int, str]:
        """
        Update iteration control parameters with validation.

        This implements CASA's interactive parameter update from _gclean.py:127-180.
        Used in interactive clean workflows.

        Parameters:
        -----------
        niter : int, optional
            New maximum iteration count

        cycleniter : int, optional
            New iterations per minor cycle

        nmajor : int, optional
            New major cycle limit

        threshold : float or str, optional
            New stopping threshold (can include units like "10mJy")

        cyclefactor : float, optional
            New cycle factor for adaptive thresholding

        Returns:
        --------
        error_code : int
            0 if successful, -1 if validation failed

        error_message : str
            Empty string if successful, error description if failed
        """
        # Update and validate niter
        if niter is not None:
            try:
                niter_int = int(niter)
                if niter_int < -1:
                    return -1, "niter must be >= -1"
                self.niter = niter_int
            except (ValueError, TypeError):
                return -1, "niter must be an integer"

        # Update and validate cycleniter
        if cycleniter is not None:
            try:
                cycleniter_int = int(cycleniter)
                if cycleniter_int < -1:
                    return -1, "cycleniter must be >= -1"
                self.cycleniter = cycleniter_int
            except (ValueError, TypeError):
                return -1, "cycleniter must be an integer"

        # Update and validate nmajor
        if nmajor is not None:
            try:
                nmajor_int = int(nmajor)
                if nmajor_int < -1:
                    return -1, "nmajor must be >= -1"
                self.nmajor = nmajor_int
            except (ValueError, TypeError):
                return -1, "nmajor must be an integer"

        # Update and validate threshold
        if threshold is not None:
            try:
                if isinstance(threshold, str):
                    threshold_float = self._parse_threshold_string(threshold)
                    if threshold_float < 0:
                        return -1, "threshold must be >= 0"
                    self.threshold = threshold_float
                else:
                    threshold_float = float(threshold)
                    if threshold_float < 0:
                        return -1, "threshold must be >= 0"
                    self.threshold = threshold_float
            except (ValueError, TypeError):
                return (
                    -1,
                    "threshold must be a number, or a number with units (Jy/mJy/uJy)",
                )

        # Update and validate cyclefactor
        if cyclefactor is not None:
            try:
                cyclefactor_float = float(cyclefactor)
                if cyclefactor_float <= 0:
                    return -1, "cyclefactor must be > 0"
                self.cyclefactor = cyclefactor_float
            except (ValueError, TypeError):
                return -1, "cyclefactor must be a number"

        return 0, ""

    def _parse_threshold_string(self, threshold_str: str) -> float:
        """Parse threshold string with units (Jy, mJy, uJy) to float."""
        threshold_str = threshold_str.strip()

        if "uJy" in threshold_str:
            return float(threshold_str.replace("uJy", "")) / 1e6
        elif "mJy" in threshold_str:
            return float(threshold_str.replace("mJy", "")) / 1e3
        elif "Jy" in threshold_str:
            return float(threshold_str.replace("Jy", ""))
        else:
            raise ValueError(f"Unknown units in threshold string: {threshold_str}")

    def reset(self) -> None:
        """Reset the iteration controller to initial state."""
        self.niter = self._initial_niter
        self.major_done = 0
        self.total_iter_done = 0
        self.stopcode = StopCode(major=MAJOR_CONTINUE, minor=MINOR_CONTINUE)
        self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_CONTINUE]

    def reset_stopcode(self) -> None:
        """Reset the stopcode of the iteration controller"""
        self.stopcode = StopCode(major=MAJOR_CONTINUE, minor=MINOR_CONTINUE)
        self.stopdescription = MAJOR_STOPCODE_DESCRIPTIONS[MAJOR_CONTINUE]

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the iteration controller as a dictionary.

        Note: The stopcode is serialized as a dict with 'major' and 'minor' keys
        to preserve the namedtuple structure across serialization.
        """
        return {
            "niter": self.niter,
            "nmajor": self.nmajor,
            "initial_niter": self._initial_niter,
            "threshold": self.threshold,
            "nsigma": self.nsigma,
            "gain": self.gain,
            "cyclefactor": self.cyclefactor,
            "minpsffraction": self.minpsffraction,
            "maxpsffraction": self.maxpsffraction,
            "cycleniter": self.cycleniter,
            "major_done": self.major_done,
            "total_iter_done": self.total_iter_done,
            "stopcode": {"major": self.stopcode.major, "minor": self.stopcode.minor},
            "stopdescription": self.stopdescription,
        }


# ============================================================================
# Convergence Visualization
# ============================================================================


class ConvergencePlots:
    """
    Class for creating convergence visualization plots from ReturnDict.

    This class provides methods to create interactive HoloViews plots showing
    deconvolution convergence history, including peak residual evolution over
    iterations.

    Parameters
    ----------
    return_dict : ReturnDict
        ReturnDict object with convergence history (peakres, iter_done fields)

    Attributes
    ----------
    return_dict : ReturnDict
        The ReturnDict containing convergence data
    stokes_to_pol : dict
        Mapping from Stokes parameter names to polarization indices

    Examples
    --------
    >>> rd = ReturnDict()
    >>> for cycle in range(5):
    ...     rd.add({'peakres': 1.0 * 0.7**cycle, 'iter_done': 100},
    ...            time=0, pol=0, chan=0)
    >>> plotter = ConvergencePlots(rd)
    >>> plot = plotter.plot_history(time=0, stokes='I', chan=0)
    >>> plot  # Display in Jupyter notebook
    """

    def __init__(self, return_dict):
        """
        Initialize ConvergencePlots with a ReturnDict.

        Parameters
        ----------
        return_dict : ReturnDict
            ReturnDict object containing convergence history
        """
        self.return_dict = return_dict
        self.stokes_to_pol = {"I": 0, "Q": 1, "U": 2, "V": 3}

        # Default plotting parameters (set by plot_history)
        self.width = 700
        self.height = 400
        self.responsive = False
        self.time = 0

    def make_plot(self, stokes_sel, chan_sel):
        """
        Generate a convergence plot for given Stokes and channel selection.

        This method is called by HoloViews DynamicMap when widget values change.

        Parameters
        ----------
        stokes_sel : str
            Stokes parameter selection ('I', 'Q', 'U', 'V')
        chan_sel : int
            Channel index selection

        Returns
        -------
        holoviews.Curve
            Convergence history curve or empty curve with error message
        """
        # Lazy imports
        try:
            import holoviews as hv
            import numpy as np

            hv.extension("bokeh")
        except ImportError as e:
            raise ImportError(
                "ConvergencePlots requires holoviews and bokeh. "
                "Install with: pip install holoviews bokeh"
            ) from e

        pol_sel = self.stokes_to_pol.get(stokes_sel, 0)

        # Get data for this (time, pol, chan)
        key = Key(time=self.time, pol=pol_sel, chan=chan_sel)

        if key not in self.return_dict.data:
            # Show error message
            return hv.Curve([]).opts(
                title=f"No data for Time={self.time}, Stokes={stokes_sel}, Channel={chan_sel}",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
                width=self.width,
                height=self.height,
                show_grid=True,
            )

        data = self.return_dict.data[key]

        # Extract history
        peakres_history = data.get("peakres", [])
        iter_done_history = data.get("iter_done", [])

        # Handle single values (convert to list)
        if not isinstance(peakres_history, list):
            peakres_history = [peakres_history]
        if not isinstance(iter_done_history, list):
            iter_done_history = [iter_done_history]

        if not peakres_history or not iter_done_history:
            return hv.Curve([]).opts(
                title=f"No convergence history - Time={self.time}, Stokes={stokes_sel}, Channel={chan_sel}",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
                width=self.width,
                height=self.height,
                show_grid=True,
            )

        # Calculate cumulative iterations
        cumulative_iters = np.cumsum(iter_done_history)

        # Create curve data
        curve_data = list(zip(cumulative_iters, peakres_history))

        # Create plot
        curve = hv.Curve(
            curve_data, kdims=["Cumulative Iterations"], vdims=["Peak Residual (Jy)"]
        )

        return curve.opts(
            title=f"Convergence History - Time={self.time}, Stokes={stokes_sel}, Channel={chan_sel}",
            xlabel="Cumulative Iterations",
            ylabel="Peak Residual (Jy)",
            color="blue",
            line_width=2,
            width=self.width,
            height=self.height,
            show_grid=True,
            show_legend=True,
            tools=["hover"],
            responsive=self.responsive,
        )

    def plot_history(self, time=0, stokes="I", chan=0, **kwargs):
        """
        Plot interactive convergence history.

        Creates an interactive HoloViews plot showing peak residual evolution
        over iterations, with widgets to select Stokes parameter and channel.

        Parameters
        ----------
        time : int, optional
            Time index to plot (default: 0)
        stokes : str, optional
            Initial Stokes parameter to display: 'I', 'Q', 'U', or 'V' (default: 'I')
        chan : int, optional
            Initial channel index to display (default: 0)
        **kwargs : dict, optional
            Additional plotting options:
            - width : int, plot width in pixels (default: 700)
            - height : int, plot height in pixels (default: 400)
            - responsive : bool, make plot responsive (default: False)

        Returns
        -------
        holoviews.DynamicMap
            Interactive plot with Stokes and channel selector widgets

        Notes
        -----
        - Requires holoviews with bokeh backend
        - Uses lazy imports to avoid hard dependency
        """
        # Store plotting parameters as instance variables for access in make_plot
        self.time = time
        self.width = kwargs.get("width", 700)
        self.height = kwargs.get("height", 400)
        self.responsive = kwargs.get("responsive", False)

        # Lazy imports
        try:
            import holoviews as hv
            import numpy as np

            hv.extension("bokeh")
        except ImportError as e:
            raise ImportError(
                "ConvergencePlots requires holoviews and bokeh. "
                "Install with: pip install holoviews bokeh"
            ) from e

        # Extract available channels and stokes from ReturnDict
        available_keys = list(self.return_dict.data.keys())
        if not available_keys:
            return hv.Curve([]).opts(
                title="No data in ReturnDict",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
            )

        channels = sorted(set(k.chan for k in available_keys if k.time == time))
        pols = sorted(set(k.pol for k in available_keys if k.time == time))
        stokes_available = [s for s, p in self.stokes_to_pol.items() if p in pols]

        if not channels:
            return hv.Curve([]).opts(
                title=f"No data for time={time}",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
            )

        # Create widgets
        if stokes_available:
            stokes_default = (
                stokes if stokes in stokes_available else stokes_available[0]
            )
        else:
            stokes_default = "I"

        chan_default = chan if chan in channels else (channels[0] if channels else 0)

        # Create DynamicMap with widgets
        dmap = hv.DynamicMap(self.make_plot, kdims=["Stokes", "Channel"])
        dmap = dmap.redim.values(
            Stokes=stokes_available if stokes_available else ["I"],
            Channel=channels if channels else [0],
        )
        dmap = dmap.redim.default(Stokes=stokes_default, Channel=chan_default)

        return dmap


def plot_convergence_history(return_dict, time=0, stokes="I", chan=0, **kwargs):
    """
    Plot interactive convergence history from ReturnDict.

    Convenience function that wraps ConvergencePlots.plot_history() for
    backward compatibility and quick plotting.

    Parameters
    ----------
    return_dict : ReturnDict
        ReturnDict object with convergence history (peakres, iter_done fields)
    time : int, optional
        Time index to plot (default: 0)
    stokes : str, optional
        Initial Stokes parameter to display: 'I', 'Q', 'U', or 'V' (default: 'I')
    chan : int, optional
        Initial channel index to display (default: 0)
    **kwargs : dict, optional
        Additional plotting options (e.g., width, height, colors)

    Returns
    -------
    holoviews.DynamicMap
        Interactive plot with Stokes and channel selector widgets

    Examples
    --------
    >>> rd = ReturnDict()
    >>> for cycle in range(5):
    ...     rd.add({'peakres': 1.0 * 0.7**cycle, 'iter_done': 100},
    ...            time=0, pol=0, chan=0)
    >>> plot = plot_convergence_history(rd, time=0, stokes='I', chan=0)
    >>> plot  # Display in Jupyter notebook

    Notes
    -----
    - Requires holoviews with bokeh backend
    - Uses lazy imports to avoid hard dependency
    - Displays error message if selected (time, pol, chan) not found
    - For more control, use ConvergencePlots class directly
    """
    plotter = ConvergencePlots(return_dict)
    return plotter.plot_history(time=time, stokes=stokes, chan=chan, **kwargs)
