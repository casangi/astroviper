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


from astroviper.core.imaging.imaging_utils.return_dict import ReturnDict, Key

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
    merge_strategy: str = "latest",
) -> ReturnDict:
    """
    Merge multiple ReturnDict objects into a single ReturnDict.

    This is essential for dask workflows where each node processes a subset
    of (time, pol, chan) combinations and returns its own ReturnDict. Before
    making iteration control decisions, we need to merge all results.

    Merge Strategies:
    -----------------
    - "latest": If the same (time, pol, chan) key appears in multiple dicts,
      keep the value from the last dict in the list. Use when dicts represent
      sequential updates.

    - "error": Raise an error if any (time, pol, chan) key appears in multiple
      dicts. Use when each node should process unique planes.

    - "update": Merge dictionaries at the value level if keys conflict. Use
      when different nodes may update different fields for the same plane.

    Parameters:
    -----------
    return_dicts : list of ReturnDict
        List of ReturnDict objects to merge

    merge_strategy : str, optional
        Strategy for handling conflicting keys. Options:
        - "latest" (default): Use value from last dict with this key
        - "error": Raise error on conflicts
        - "update": Merge value dicts, later entries overwrite earlier

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
                    # Merge dictionaries - later values overwrite earlier
                    if isinstance(merged.data[key], dict) and isinstance(value, dict):
                        merged.data[key].update(value)
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
        Maximum peak residual across selected planes
        Returns 0.0 if no valid data found
    """
    selected = return_dict.sel(time=time, pol=pol, chan=chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    peak = 0.0
    key = "peakres" if use_mask else "peakres_nomask"

    for entry in selected:
        if entry is not None and key in entry:
            # Only consider planes with valid mask when using mask
            if use_mask and "masksum" in entry and entry["masksum"] == 0:
                continue
            peak = max(peak, abs(entry[key]))

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
        Sum of mask values across selected planes
        Returns 0.0 if no mask data found
    """
    selected = return_dict.sel(time=time, pol=pol, chan=chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    total_masksum = 0.0
    for entry in selected:
        if entry is not None and "masksum" in entry:
            total_masksum += entry["masksum"]

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
        Sum of iterations done across selected planes
    """
    selected = return_dict.sel(time=time, pol=pol, chan=chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    total_iters = 0
    for entry in selected:
        if entry is not None and "iter_done" in entry:
            total_iters += entry["iter_done"]

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
