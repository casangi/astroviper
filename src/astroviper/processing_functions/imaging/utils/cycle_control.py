"""
Cycle Control Module for Deconvolution Algorithms

This module implements cycle control logic for deconvolution processes,
adapted from CASA's iteration control implementation in _gclean.py and
imager_return_dict.py. It has been streamlined to work with AstroViper's
ImagingDict structure while retaining all original functionality.

Cycle control is the stop test between the residual update and the model
update, made on the residual the residual update has just made: it decides
whether a plane gets another imaging cycle and sets the limits of its next
model update. Iteration control is the stop test inside the model update,
made by the deconvolver at every iteration against those limits
(``max_iter_per_cycle_updated``, ``threshold_per_cycle``).

Per-plane cycle control
----------------------------
All cycle control is performed independently for every
``(time, chan(frequency), pol)`` plane. Concretely:

- ``max_iter_remaining`` is a per-plane array of the iterations left of
  ``max_iter`` (:attr:`CycleController.max_iter_remaining`, shape
  ``(ntime, nchan, npol)``);
- every stopping criterion (zero mask, iteration limit, threshold, residual
  entropy, imaging cycle limit, no progress) is evaluated per plane, and
  each plane carries its own stop code;
- thresholds may differ per plane — :meth:`CycleController.per_plane_threshold_per_cycle`
  produces a per-plane threshold_per_cycle array, and the deconvolvers accept
  per-plane ``max_iter_per_cycle_updated`` and ``threshold_per_cycle`` arrays.

The imaging cycle loop continues while *any* plane is still active.
"""

from collections import namedtuple
from typing import Any

import numpy as np

from astroviper.processing_functions.image_analysis import image_statistics as imgstats
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    EXTRA_CYCLES,
    NOISY_CYCLE_SAFETY,
    THRESHOLD_TOLERANCE,
    predicted_imaging_cycles,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    FIELD_ACCUM,
    ITERATION_HISTORY_FIELDS,
    ImagingDict,
    Key,
    ends_with_stop_test,
)

# Stop codes matching CASA's implementation
# See: https://casadocs.readthedocs.io/en/stable/notebooks/synthesis_imaging.html#Returned-Dictionary
# CASA Source: imager_return_dict.py:9-10, lines 463-565
#
# StopCode(imaging, model_update) separates the reason a plane's imaging cycles
# ended from the reason its last model update ended, using CASA's stop code
# numbering for both.

StopCode = namedtuple("StopCode", ["imaging", "model_update"])

# Imaging cycle stop codes (global convergence)
IMAGING_CONTINUE = 0  # Continue imaging cycles
IMAGING_MAX_ITER = 1  # Reached total iteration limit (max_iter)
IMAGING_THRESHOLD = 2  # Peak residual below global threshold
IMAGING_NO_PROGRESS = 4  # Model updates no longer do any iteration
IMAGING_DIVERGENCE = 5  # Last max_cycles_divergence model updates diverged
IMAGING_ZERO_MASK = 7  # Zero mask (no valid pixels)
IMAGING_MAX_CYCLES = 9  # Reached imaging cycle limit (max_cycles)
IMAGING_ENTROPY = 10  # Entropy of the residual passed its maximum (entropy_stop)

# Model update stop codes (per-cycle convergence): the iteration control stop
# code the deconvolver reports for every plane (deconvolvers/include/
# plane_stop_test.hpp)
MODEL_UPDATE_CONTINUE = 0  # No model update ran (no iterations allotted)
MODEL_UPDATE_MAX_ITER_PER_CYCLE = (
    1  # Reached per-cycle iteration limit (max_iter_per_cycle_updated)
)
MODEL_UPDATE_THRESHOLD_PER_CYCLE = 2  # Peak residual below threshold_per_cycle
MODEL_UPDATE_DIVERGENCE = 4  # Divergence detected by the deconvolver
MODEL_UPDATE_ALGORITHM = 5  # A stop test of the deconvolver itself (Asp)

# A plane whose model update did no iteration cannot change its residual, so a
# second identical imaging cycle would follow, and a third. After this many
# such model updates in a row the plane is stopped with IMAGING_NO_PROGRESS.
NO_PROGRESS_MODEL_UPDATES = 2

# max_cycles="auto": the limit is predicted from the dirty image (see
# CycleController.resolve_max_cycles).
AUTO_MAX_CYCLES = "auto"

# Stop code descriptions for imaging cycle codes
IMAGING_STOP_DESCRIPTIONS = {
    IMAGING_CONTINUE: "Continue imaging cycles",
    IMAGING_MAX_ITER: "Reached max_iter",
    IMAGING_THRESHOLD: "Reached threshold (peak residual within the mask)",
    IMAGING_NO_PROGRESS: "No progress (model updates did no iterations)",
    IMAGING_DIVERGENCE: "Model updates diverged max_cycles_divergence times in a row",
    IMAGING_ZERO_MASK: "Zero mask",
    IMAGING_MAX_CYCLES: "Reached max_cycles",
    IMAGING_ENTROPY: "Residual entropy passed its maximum",
}

# Stop code descriptions for model update codes
MODEL_UPDATE_STOP_DESCRIPTIONS = {
    MODEL_UPDATE_CONTINUE: "No model update ran",
    MODEL_UPDATE_MAX_ITER_PER_CYCLE: "Reached max_iter_per_cycle_updated",
    MODEL_UPDATE_THRESHOLD_PER_CYCLE: "Reached threshold_per_cycle",
    MODEL_UPDATE_DIVERGENCE: "Possible divergence detected",
    MODEL_UPDATE_ALGORITHM: "Stopped by a test of the deconvolver itself",
}


# ============================================================================
# ImagingDict Utility Functions
# ============================================================================


def _validate_imaging_dict_selection(
    imaging_dict: ImagingDict,
    selected: Any,
    time: int | None = None,
    pol: int | None = None,
    chan: int | None = None,
) -> None:
    """
    Validate that sel() returned data when explicit filters were provided.

    Raises KeyError if explicit time/pol/chan values were specified but
    no matching entries exist in the ImagingDict.

    Parameters:
    -----------
    imaging_dict : ImagingDict
        The ImagingDict that was queried
    selected : Any
        Result from imaging_dict.sel()
    time, pol, chan : Optional[int]
        The filter values that were used

    Raises:
    -------
    KeyError
        If explicit filters were specified but no matches found
    """
    # Only validate when at least one filter is explicitly specified
    has_explicit_filter = any(x is not None for x in (time, pol, chan))

    if not has_explicit_filter:
        return  # Wildcard behavior - no validation needed

    # Check if selection returned no results
    no_results = selected is None or (isinstance(selected, list) and len(selected) == 0)

    if no_results:
        # Build descriptive error message
        filter_parts = []
        if time is not None:
            filter_parts.append(f"time={time}")
        if pol is not None:
            filter_parts.append(f"pol={pol}")
        if chan is not None:
            filter_parts.append(f"chan={chan}")

        filter_str = ", ".join(filter_parts)

        # Get available keys for the error message
        available_keys = list(imaging_dict.data.keys())
        if len(available_keys) == 0:
            available_str = "ImagingDict is empty"
        elif len(available_keys) <= 10:
            available_str = f"Available keys: {available_keys}"
        else:
            available_str = f"Available keys (first 10 of {len(available_keys)}): {available_keys[:10]}"

        raise KeyError(f"No entries found for {filter_str}. {available_str}")


def merge_imaging_dicts(
    imaging_dicts: list[ImagingDict],
    merge_strategy: str = "update",
) -> ImagingDict:
    """
    Merge multiple ImagingDict objects into a single ImagingDict.

    This is essential for dask workflows where each node processes a subset
    of (time, pol, chan) combinations and returns its own ImagingDict. Before
    making cycle control decisions, we need to merge all results.

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
    imaging_dicts : list of ImagingDict
        List of ImagingDict objects to merge

    merge_strategy : str, optional
        Strategy for handling conflicting keys. Options:
        - "update" (default): Merge value dicts, later entries overwrite earlier
        - "error": Raise error on conflicts
        - "latest": Use value from last dict with this key

    Returns:
    --------
    merged : ImagingDict
        Merged ImagingDict containing all data from input dicts

    Raises:
    -------
    ValueError
        If merge_strategy is "error" and conflicting keys are found

    Example:
    --------
    >>> # Dask workflow: 3 workers process different channels
    >>> rd1 = ImagingDict()
    >>> rd1.add({'peakres': 0.5, 'iter_done': 100}, time=0, pol=0, chan=0)
    >>>
    >>> rd2 = ImagingDict()
    >>> rd2.add({'peakres': 0.3, 'iter_done': 120}, time=0, pol=0, chan=1)
    >>>
    >>> rd3 = ImagingDict()
    >>> rd3.add({'peakres': 0.4, 'iter_done': 110}, time=0, pol=0, chan=2)
    >>>
    >>> # Merge results
    >>> merged = merge_imaging_dicts([rd1, rd2, rd3])
    >>> # merged now has 3 entries, one for each channel
    """
    if not imaging_dicts:
        return ImagingDict()

    merged = ImagingDict()

    for rd in imaging_dicts:
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


def get_peak_residual_from_imaging_dict(
    imaging_dict: ImagingDict,
    use_mask: bool = True,
    time: int | None = None,
    pol: int | None = None,
    chan: int | None = None,
) -> float:
    """
    Extract peak residual from ImagingDict structure.

    This adapts CASA's get_peakres() from imager_return_dict.py
    to work with AstroViper's ImagingDict which uses (time, pol, chan) indexing.

    Parameters:
    -----------
    imaging_dict : ImagingDict
        ImagingDict instance containing deconvolution statistics

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
    selected = imaging_dict.sel(time=time, pol=pol, chan=chan)
    _validate_imaging_dict_selection(imaging_dict, selected, time, pol, chan)

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


def get_masksum_from_imaging_dict(
    imaging_dict: ImagingDict,
    time: int | None = None,
    pol: int | None = None,
    chan: int | None = None,
) -> float:
    """
    Calculate total mask sum from ImagingDict structure.

    Parameters:
    -----------
    imaging_dict : ImagingDict
        ImagingDict instance containing mask statistics

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
    selected = imaging_dict.sel(time=time, pol=pol, chan=chan)
    _validate_imaging_dict_selection(imaging_dict, selected, time, pol, chan)

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


def get_iterations_done_from_imaging_dict(
    imaging_dict: ImagingDict,
    time: int | None = None,
    pol: int | None = None,
    chan: int | None = None,
) -> int:
    """
    Calculate total iterations done from ImagingDict structure.

    Parameters:
    -----------
    imaging_dict : ImagingDict
        ImagingDict instance containing iteration statistics

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
    selected = imaging_dict.sel(time=time, pol=pol, chan=chan)
    _validate_imaging_dict_selection(imaging_dict, selected, time, pol, chan)

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


def get_max_psf_sidelobe_from_imaging_dict(
    imaging_dict: ImagingDict,
    time: int | None = None,
    pol: int | None = None,
    chan: int | None = None,
) -> float:
    """
    Extract maximum PSF sidelobe level from ImagingDict.

    This should be populated by psf_fitting.py analysis and stored
    in the ImagingDict as 'max_psf_sidelobe'.

    Parameters:
    -----------
    imaging_dict : ImagingDict
        ImagingDict instance containing PSF analysis results

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
    selected = imaging_dict.sel(time=time, pol=pol, chan=chan)
    _validate_imaging_dict_selection(imaging_dict, selected, time, pol, chan)

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


def get_model_flux_from_imaging_dict(
    imaging_dict: ImagingDict,
    time: int | None = None,
    pol: int | None = None,
    chan: int | None = None,
) -> float:
    """
    Extract cumulative model flux from ImagingDict structure.

    The 'model_flux' field tracks the cumulative flux in the CLEAN model
    at each imaging cycle. This function returns the latest (most recent)
    cumulative flux value from the history.

    **History Tracking**: The model_flux field is history-tracked (stored as
    a list). This function returns the latest value, which represents the
    current total flux in the model.

    Parameters:
    -----------
    imaging_dict : ImagingDict
        ImagingDict instance containing deconvolution statistics

    time : int, optional
        Filter by specific time index (None = all times)

    pol : int, optional
        Filter by specific polarization index (None = all pols)

    chan : int, optional
        Filter by specific channel index (None = all chans)

    Returns:
    --------
    model_flux : float
        Latest cumulative model flux across selected planes (Jy)
        Returns 0.0 if no valid data found

    Examples:
    ---------
    >>> rd = ImagingDict()
    >>> rd.add({'model_flux': 1.5}, time=0, pol=0, chan=0)
    >>> rd.add({'model_flux': 2.3}, time=0, pol=0, chan=0)
    >>> get_model_flux_from_imaging_dict(rd, time=0, pol=0, chan=0)
    2.3  # Latest cumulative flux
    """
    selected = imaging_dict.sel(time=time, pol=pol, chan=chan)
    _validate_imaging_dict_selection(imaging_dict, selected, time, pol, chan)

    if not isinstance(selected, list):
        selected = [selected] if selected is not None else []

    total_flux = 0.0

    for entry in selected:
        if entry is not None and "model_flux" in entry:
            # Extract value (handle both list and single value for backward compatibility)
            value = entry["model_flux"]
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                # Use latest value (last in list) - represents cumulative flux
                value = value[-1]

            total_flux += value

    return total_flux


ENTROPY_PARAM_DEFAULTS = {
    "entropy_stop": False,
    "entropy_max_snr": 6.0,
    "entropy_spatial_bins": 7,
    "entropy_flux_bins": 10,
}


def validate_entropy_params(cycle_control_params):
    """Entropy stop parameters of ``cycle_control_params``, with defaults.

    Parameters
    ----------
    cycle_control_params : dict or None
        Cycle control parameters. The keys ``entropy_stop``,
        ``entropy_max_snr``, ``entropy_spatial_bins`` and
        ``entropy_flux_bins`` are read; a missing key takes its default
        (``False``, ``6.0``, ``7`` and ``10``).

    Returns
    -------
    entropy_stop : bool
        Whether the entropy of the residual stops the imaging cycles.
    entropy_max_snr : float
        Ratio of the peak to the RMS of the residual below which the entropy
        is followed.
    entropy_spatial_bins : int
        Spatial bins per image axis.
    entropy_flux_bins : int
        Flux bins per unit of RMS.

    Raises
    ------
    ValueError
        If ``entropy_stop`` is not a bool, ``entropy_max_snr`` is not a
        positive number, or a number of bins is not a positive integer.
    """
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_entropy_controls,
    )

    params = {**ENTROPY_PARAM_DEFAULTS, **(cycle_control_params or {})}
    entropy = {name: params[name] for name in ENTROPY_PARAM_DEFAULTS}
    try:
        check_entropy_controls(entropy)
    except TypeError as error:
        raise ValueError(str(error)) from error
    return (
        bool(params["entropy_stop"]),
        float(params["entropy_max_snr"]),
        int(params["entropy_spatial_bins"]),
        int(params["entropy_flux_bins"]),
    )


def make_continuum_cycle_controller(cycle_control_params):
    """Build a cycle controller from the complete continuum control mapping."""
    entropy_stop, entropy_max_snr, entropy_spatial_bins, entropy_flux_bins = (
        validate_entropy_params(cycle_control_params)
    )
    return CycleController(
        max_iter=cycle_control_params["max_iter"],
        max_cycles=cycle_control_params["max_cycles"],
        threshold_sigma=cycle_control_params.get("threshold_sigma", 0.0),
        threshold=cycle_control_params["threshold"],
        gain=cycle_control_params["gain"],
        psf_sidelobe_factor=cycle_control_params["psf_sidelobe_factor"],
        min_psf_fraction=cycle_control_params["min_psf_fraction"],
        max_psf_fraction=cycle_control_params["max_psf_fraction"],
        max_iter_per_cycle=cycle_control_params["max_iter_per_cycle"],
        entropy_stop=entropy_stop,
        entropy_max_snr=entropy_max_snr,
        entropy_spatial_bins=entropy_spatial_bins,
        entropy_flux_bins=entropy_flux_bins,
        max_cycles_divergence=cycle_control_params.get("max_cycles_divergence", -1),
        threshold_tolerance=cycle_control_params.get(
            "threshold_tolerance",
            THRESHOLD_TOLERANCE,
        ),
    )


def _is_auto_max_cycles(max_cycles):
    """True for a ``max_cycles`` that is still ``"auto"`` (not resolved)."""
    return isinstance(max_cycles, str) and max_cycles == AUTO_MAX_CYCLES


def validate_max_cycles(max_cycles, max_psf_fraction=None):
    """``max_cycles`` checked: an integer of at least -1, or ``"auto"``.

    Parameters
    ----------
    max_cycles : int or str
        Maximum number of imaging cycles, ``-1`` for no limit, or ``"auto"``
        to predict it from the dirty image
        (:meth:`CycleController.resolve_max_cycles`). An integral float
        (``5.0``) is taken as the integer.
    max_psf_fraction : float, optional
        The upper clamp of the PSF fraction, which ``"auto"`` needs below 1
        (the cycles needed to clean down grow without bound as the PSF
        fraction approaches 1). Not checked when omitted.

    Returns
    -------
    int or str
        ``max_cycles`` as an ``int``, or ``"auto"``.

    Raises
    ------
    ValueError
        For anything else: a float that is not an integer, a bool, an integer
        below -1 or any other string; or for ``"auto"`` with a
        ``max_psf_fraction`` of 1 or more.
    """
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_max_cycles,
    )

    try:
        check_max_cycles(max_cycles, max_psf_fraction, allow_integral_float=True)
    except TypeError as error:
        raise ValueError(str(error)) from error
    return max_cycles if _is_auto_max_cycles(max_cycles) else int(max_cycles)


def validate_threshold_tolerance(threshold_tolerance):
    """``threshold_tolerance`` checked: a finite number >= 0.

    Parameters
    ----------
    threshold_tolerance : float
        Relative tolerance of the threshold stop
        (:meth:`CycleController.check_convergence`).

    Returns
    -------
    float
        ``threshold_tolerance`` as a ``float``.

    Raises
    ------
    ValueError
        For a bool, anything that is not a number, NaN, infinity or a value
        below 0.
    """
    from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
        check_nonnegative,
    )

    try:
        check_nonnegative("threshold_tolerance", threshold_tolerance, scalar=True)
    except TypeError as error:
        raise ValueError(str(error)) from error
    return float(threshold_tolerance)


# ============================================================================
# CycleController Class
# ============================================================================


class CycleController:
    """
    Manages cycle control logic for deconvolution algorithms.

    The controller extracts needed statistics (peak residual, masksum, iterations
    done, etc.) from ImagingDict internally, so callers don't need to manually
    pass individual values.

    Per-plane cycle control
    ---------------------------
    All cycle control is performed independently for every
    ``(time, chan(frequency), pol)`` plane. ``max_iter_remaining`` is a per-plane array,
    each plane carries its own stop code, and thresholds may differ per plane
    (:meth:`per_plane_threshold_per_cycle`). The deconvolvers are driven with
    per-plane ``max_iter_per_cycle_updated`` and ``threshold_per_cycle`` arrays. The imaging cycle loop
    continues while *any* plane is still active; the aggregate ``stopcode``
    reported by :meth:`check_convergence` is CONTINUE until every plane has
    stopped.

    Attributes:
    -----------
    max_iter_remaining : numpy.ndarray or None
        Per-plane iterations left of ``max_iter`` over all cycles, shape
        ``(ntime, nchan, npol)`` indexed ``(time, chan, pol)``. Allocated
        lazily the first time a ImagingDict is seen (the cube shape is unknown
        at construction); ``None`` until then. ``_max_iter`` holds the
        scalar per-plane budget.
    max_cycles : int or str
        Maximum number of imaging cycles remaining (-1 = unlimited). Imaging cycles
        are global (shared across all planes). ``"auto"`` until
        :meth:`resolve_max_cycles` has predicted it from the dirty image.
    max_cycles_auto : dict or None
        With ``max_cycles="auto"``, the limit predicted for every plane,
        ``{(time, chan, pol): int or None}`` (``None`` for a plane nothing
        could be predicted for), set by :meth:`resolve_max_cycles` and stamped
        into every checked ImagingDict entry as ``max_cycles_auto``. ``None``
        otherwise.
    threshold : float
        Global stopping threshold (in Jy or image units)
    threshold_tolerance : float
        Relative tolerance of the threshold stop: a plane stops once its
        peak residual is at most ``threshold * (1 + threshold_tolerance)``.
    gain : float
        CLEAN loop gain (typically 0.1)
    psf_sidelobe_factor : float
        Multiplier for PSF sidelobe to set threshold_per_cycle
    min_psf_fraction : float
        Minimum PSF fraction for threshold_per_cycle calculation
    max_psf_fraction : float
        Maximum PSF fraction for threshold_per_cycle calculation
    max_iter_per_cycle : int
        Maximum iterations one plane may run in a single model update
        cycle. ``-1`` lets the adaptive ``threshold_per_cycle`` govern the
        depth instead.
    threshold_sigma : float
        N-sigma threshold for stopping (0 = disabled)
    zero_iter_model_updates : numpy.ndarray or None
        Per-plane count of consecutive model updates that did no iteration,
        same shape as ``max_iter_remaining``. A plane is stopped with
        ``IMAGING_NO_PROGRESS`` when it reaches ``NO_PROGRESS_MODEL_UPDATES``.
    max_cycles_divergence : int
        Number of model updates in a row that may end diverged before the
        plane is stopped with ``IMAGING_DIVERGENCE``; ``-1`` disables it.
    diverged_model_updates : numpy.ndarray or None
        Per-plane count of consecutive model updates that ended diverged,
        same shape as ``max_iter_remaining``.
    divergence_stopped : numpy.ndarray or None
        Per-plane flag, set once the divergence stop has ended the imaging
        cycles of the plane. Such a plane gets no more iterations and keeps
        the stop code ``IMAGING_DIVERGENCE``.
    entropy_stop : bool
        Whether the entropy of the residual stops the imaging cycles of a
        plane (see :meth:`check_convergence`).
    entropy_max_snr : float
        The entropy of a plane is followed once the peak of its residual is at
        most this many times the RMS of the residual.
    entropy_spatial_bins : int
        Spatial bins per image axis of the entropy.
    entropy_flux_bins : int
        Flux bins per unit of RMS of the entropy.
    max_entropy : numpy.ndarray or None
        Per-plane highest entropy of the residual in the imaging cycles so
        far, not a number while none has been recorded. Same shape as
        ``max_iter_remaining``.
    entropy_stopped : numpy.ndarray or None
        Per-plane flag, set once the entropy stop has ended the imaging cycles
        of the plane. Such a plane gets no more iterations and keeps the stop
        code ``IMAGING_ENTROPY``.

    Imaging cycle Tracking:
    ---------------------
    cycles_done : int
        Number of imaging cycles completed so far
    total_iter_done : int
        Total number of model update iterations completed

    Convergence State:
    ------------------
    stopcode : StopCode
        Current stop code as namedtuple (imaging, model_update)
        Access via stopcode.imaging and stopcode.model_update
    stopdescription : str
        Human-readable description of stop reason
    """

    def __init__(
        self,
        max_iter: int = 1000,
        max_cycles: int | str = -1,
        threshold: float = 0.0,
        gain: float = 0.1,
        psf_sidelobe_factor: float = 1.0,
        min_psf_fraction: float = 0.05,
        max_psf_fraction: float = 0.8,
        max_iter_per_cycle: int = -1,
        threshold_sigma: float = 0.0,
        entropy_stop: bool = False,
        entropy_max_snr: float = 6.0,
        entropy_spatial_bins: int = 7,
        entropy_flux_bins: int = 10,
        max_cycles_divergence: int = -1,
        threshold_tolerance: float = THRESHOLD_TOLERANCE,
    ):
        """
        Initialize the cycle controller with deconvolution parameters.

        Parameters:
        -----------
        max_iter : int, optional
            Maximum CLEAN iterations for ONE plane, summed over all
            imaging cycles (default: 1000). Every plane is seeded
            with this full value; no budget is shared or split between
            planes. This is the deliberate difference from CASA's
            image-wide ``niter``.

        max_cycles : int or str, optional
            Maximum number of imaging cycles (default: -1 for unlimited), or
            ``"auto"`` to predict it from the dirty image (see
            :meth:`resolve_max_cycles`). An integral float (``5.0``) is taken
            as the integer.

        threshold : float, optional
            Global stopping threshold in Jy (default: 0.0)

        gain : float, optional
            CLEAN loop gain, range (0, 1] (default: 0.1)

        psf_sidelobe_factor : float, optional
            Multiplier for adaptive threshold_per_cycle (default: 1.0)

        min_psf_fraction : float, optional
            Minimum PSF sidelobe fraction (default: 0.05)

        max_psf_fraction : float, optional
            Maximum PSF sidelobe fraction (default: 0.8)

        max_iter_per_cycle : int, optional
            Max iterations per model update (default: -1)

        threshold_sigma : float, optional
            N-sigma threshold for stopping (default: 0.0, disabled)

        entropy_stop : bool, optional
            Stop the imaging cycles of a plane once the entropy of its
            residual has passed its maximum (default: False)

        entropy_max_snr : float, optional
            Ratio of the peak to the RMS of the residual below which the
            entropy is followed (default: 6.0)

        entropy_spatial_bins : int, optional
            Spatial bins per image axis of the entropy (default: 7)

        entropy_flux_bins : int, optional
            Flux bins per unit of RMS of the entropy (default: 10)

        max_cycles_divergence : int, optional
            Stop a plane once this many of its model updates in a row ended
            diverged (default: -1, disabled)

        threshold_tolerance : float, optional
            Relative tolerance of the threshold stop, finite and >= 0: a
            plane stops once its peak residual is at most
            ``threshold * (1 + threshold_tolerance)``; ``threshold_per_cycle``
            is not changed (default: ``THRESHOLD_TOLERANCE``, 0.02)

        Raises:
        -------
        ValueError
            If ``max_cycles_divergence`` is not -1 or a positive integer, if
            ``threshold_tolerance`` is not a finite number >= 0, if
            one of the entropy parameters has a wrong type or value, if
            ``max_cycles`` is neither an integer >= -1 nor ``"auto"``, or if
            it is ``"auto"`` with a ``max_psf_fraction`` of 1 or more (the
            cycles needed to clean down grow without bound as the PSF
            fraction approaches 1).
        """
        entropy_stop, entropy_max_snr, entropy_spatial_bins, entropy_flux_bins = (
            validate_entropy_params(
                {
                    "entropy_stop": entropy_stop,
                    "entropy_max_snr": entropy_max_snr,
                    "entropy_spatial_bins": entropy_spatial_bins,
                    "entropy_flux_bins": entropy_flux_bins,
                }
            )
        )
        # Iteration limits. max_iter is per-plane and allocated lazily (the cube
        # shape is not known until the first ImagingDict is seen); until then
        # _max_iter holds the scalar per-plane budget. See _ensure_state.
        self._max_iter = max_iter
        self.max_iter_remaining = None
        # Per-plane stop codes, same shape as self.max_iter_remaining (allocated lazily).
        self.stop_code_imaging = None
        self.stop_code_model_update = None
        # Per-plane consecutive model updates without any iteration.
        self.zero_iter_model_updates = None
        # Entropy stop: parameters and per-plane state (allocated lazily).
        self.entropy_stop = entropy_stop
        self.entropy_max_snr = entropy_max_snr
        self.entropy_spatial_bins = entropy_spatial_bins
        self.entropy_flux_bins = entropy_flux_bins
        self.max_entropy = None
        self.entropy_stopped = None
        # Divergence stop: parameter and per-plane state (allocated lazily).
        from astroviper.processing_functions.imaging.config_check._common import (
            require_integer,
        )
        from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
            check_nonnegative,
        )

        try:
            require_integer(
                "max_cycles_divergence",
                max_cycles_divergence,
                minimum=1,
                allow_minus_one=True,
            )
            check_nonnegative("threshold", threshold)
        except TypeError as error:
            raise ValueError(str(error)) from error
        self.max_cycles_divergence = int(max_cycles_divergence)
        self.diverged_model_updates = None
        self.divergence_stopped = None
        self.max_cycles = validate_max_cycles(max_cycles, max_psf_fraction)
        self.max_cycles_auto = None

        # Threshold parameters: one value for every plane, or one per
        # polarization plane (the floors of noise free Stokes planes scale
        # with their fluxes)
        if np.ndim(threshold) == 0:
            self.threshold = threshold
        else:
            self.threshold = tuple(float(value) for value in threshold)
        self.threshold_sigma = threshold_sigma
        self.threshold_tolerance = validate_threshold_tolerance(threshold_tolerance)

        # CLEAN parameters
        self.gain = gain
        self.psf_sidelobe_factor = psf_sidelobe_factor
        self.min_psf_fraction = min_psf_fraction
        self.max_psf_fraction = max_psf_fraction
        self.max_iter_per_cycle = max_iter_per_cycle

        # Tracking state
        self.cycles_done = 0
        self.total_iter_done = 0

        # Convergence state (namedtuple matching CASA). self.stopcode is the
        # aggregate over all planes; per-plane codes live in stop_code_imaging /
        # stop_code_model_update (allocated lazily alongside max_iter).
        self.stopcode = StopCode(
            imaging=IMAGING_CONTINUE, model_update=MODEL_UPDATE_CONTINUE
        )
        self.stopdescription = IMAGING_STOP_DESCRIPTIONS[IMAGING_CONTINUE]

    # ------------------------------------------------------------------
    # Per-plane state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(key, time, pol, chan):
        """True if a ImagingDict key matches the (time, pol, chan) selection."""
        return (
            (time is None or key.time == time)
            and (pol is None or key.pol == pol)
            and (chan is None or key.chan == chan)
        )

    @staticmethod
    def _key_index(key):
        """Map a ImagingDict Key(time, pol, chan) to a (time, chan, pol) index.

        The per-plane arrays follow the image-cube axis order
        ``(time, chan, pol)``, whereas ImagingDict keys are
        ``(time, pol, chan)``.
        """
        return (int(key.time), int(key.chan), int(key.pol))

    @staticmethod
    def _latest(entry, field, default=0.0):
        """Return the most recent value of a (possibly history-tracked) field."""
        if field not in entry:
            return default
        value = entry[field]
        if isinstance(value, list):
            return value[-1] if value else default
        return value

    def _plane_threshold(self, pol):
        """The absolute threshold of polarization plane ``pol`` (``None``: the lowest)."""
        if np.ndim(self.threshold) == 0:
            return self.threshold
        return min(self.threshold) if pol is None else self.threshold[pol]

    def _ensure_shape(self, needed):
        """Allocate (or grow) the per-plane arrays to the ``needed`` shape.

        ``needed`` is ``(ntime, nchan, npol)``. Newly allocated planes start at
        the full per-plane budget (``_max_iter``) with a CONTINUE stop
        code. Growing only ever enlarges the arrays (existing values kept).
        """
        if np.ndim(self.threshold) and len(self.threshold) != needed[2]:
            raise ValueError(
                f"threshold has {len(self.threshold)} values for {needed[2]} "
                "polarization planes; give one value or one per plane"
            )
        if self.max_iter_remaining is None:
            self.max_iter_remaining = np.full(needed, self._max_iter, dtype=int)
            self.stop_code_imaging = np.full(needed, IMAGING_CONTINUE, dtype=int)
            self.stop_code_model_update = np.full(
                needed, MODEL_UPDATE_CONTINUE, dtype=int
            )
            self.zero_iter_model_updates = np.zeros(needed, dtype=int)
            self.max_entropy = np.full(needed, np.nan, dtype=float)
            self.entropy_stopped = np.zeros(needed, dtype=bool)
            self.diverged_model_updates = np.zeros(needed, dtype=int)
            self.divergence_stopped = np.zeros(needed, dtype=bool)
        elif any(
            n > c for n, c in zip(needed, self.max_iter_remaining.shape, strict=False)
        ):
            grown = tuple(
                max(n, c)
                for n, c in zip(needed, self.max_iter_remaining.shape, strict=False)
            )
            sl = tuple(slice(0, d) for d in self.max_iter_remaining.shape)
            for attr, fill, dtype in (
                ("max_iter_remaining", self._max_iter, int),
                ("stop_code_imaging", IMAGING_CONTINUE, int),
                ("stop_code_model_update", MODEL_UPDATE_CONTINUE, int),
                ("zero_iter_model_updates", 0, int),
                ("max_entropy", np.nan, float),
                ("entropy_stopped", False, bool),
                ("diverged_model_updates", 0, int),
                ("divergence_stopped", False, bool),
            ):
                new = np.full(grown, fill, dtype=dtype)
                new[sl] = getattr(self, attr)
                setattr(self, attr, new)

    def ensure_planes(self, ntime, nchan, npol):
        """Pre-allocate per-plane state for an ``(ntime, nchan, npol)`` cube.

        Cycle control is performed independently for every
        ``(time, chan, pol)`` plane. Callers that know the cube shape up front
        (e.g. before the first deconvolution) call this so that ``max_iter_remaining`` is a
        fully-sized per-plane array by the time the deconvolver needs it.
        """
        self._ensure_shape((int(ntime), int(nchan), int(npol)))

    def _ensure_state(self, imaging_dict):
        """Allocate (or grow) the per-plane arrays to cover ``imaging_dict``.

        The cube shape is inferred from the ImagingDict keys (max index + 1 on
        each axis). A no-op when the ImagingDict is empty.
        """
        keys = list(imaging_dict.data.keys())
        if not keys:
            return
        needed = (
            max(k.time for k in keys) + 1,
            max(k.chan for k in keys) + 1,
            max(k.pol for k in keys) + 1,
        )
        self._ensure_shape(needed)

    def calculate_per_cycle_limits(
        self,
        imaging_dict: ImagingDict,
        time: int | None = None,
        pol: int | None = None,
        chan: int | None = None,
    ) -> tuple[int, float]:
        """
        Calculate max_iter_per_cycle_updated and threshold_per_cycle for the next model update.

        Logic:
        ------
        1. Extract max_psf_sidelobe and peak_residual from imaging_dict
        2. Start with the largest remaining budget over the planes (max_iter_remaining)
        3. If max_iter_per_cycle is set (>= 0), max_iter_per_cycle_updated = min(max_iter_per_cycle, that budget)
        4. Calculate PSF fraction = max_psf_sidelobe * psf_sidelobe_factor
        5. Clamp PSF fraction to [min_psf_fraction, max_psf_fraction]
        6. threshold_per_cycle = max(psf_fraction * peak_residual, threshold)

        Parameters:
        -----------
        imaging_dict : ImagingDict
            ImagingDict containing deconvolution statistics including:
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
        max_iter_per_cycle_updated : int
            Iteration limit of the next model update over the selected planes
            (:func:`get_per_cycle_limits` caps each plane's own
            ``max_iter_remaining`` with it)

        threshold_per_cycle : float
            Stopping threshold for this model update

        Example:
        --------
        >>> controller = CycleController(max_iter=1000, psf_sidelobe_factor=1.5)
        >>> # imaging_dict populated by deconvolver and PSF analysis
        >>> max_iter_per_cycle_updated, threshold_per_cycle = controller.calculate_per_cycle_limits(imaging_dict)
        """
        # Extract needed values from ImagingDict
        max_psf_sidelobe = get_max_psf_sidelobe_from_imaging_dict(
            imaging_dict, time=time, pol=pol, chan=chan
        )
        peak_residual = get_peak_residual_from_imaging_dict(
            imaging_dict, use_mask=True, time=time, pol=pol, chan=chan
        )

        # Start with all remaining iterations: the largest remaining budget
        # across planes (get_per_cycle_limits caps every plane's own budget
        # with the result). Before the per-plane array exists, fall back to
        # the initial per-plane budget.
        if self.max_iter_remaining is None:
            max_iter_per_cycle_updated = self._max_iter
        else:
            max_iter_per_cycle_updated = int(self.max_iter_remaining.max())

        # If user forced a specific max_iter_per_cycle, respect it
        if self.max_iter_per_cycle >= 0:
            max_iter_per_cycle_updated = min(
                self.max_iter_per_cycle, max_iter_per_cycle_updated
            )

        # Calculate adaptive PSF fraction for threshold_per_cycle
        psf_fraction = max_psf_sidelobe * self.psf_sidelobe_factor

        # Clamp to user-specified bounds
        psf_fraction = max(psf_fraction, self.min_psf_fraction)
        psf_fraction = min(psf_fraction, self.max_psf_fraction)

        # Set threshold_per_cycle as fraction of current peak residual
        threshold_per_cycle = psf_fraction * peak_residual
        threshold_per_cycle = max(threshold_per_cycle, self._plane_threshold(pol))

        return int(max_iter_per_cycle_updated), threshold_per_cycle

    def per_plane_threshold_per_cycle(
        self,
        imaging_dict: ImagingDict,
        time: int | None = None,
        pol: int | None = None,
        chan: int | None = None,
    ) -> "np.ndarray":
        """Compute the per-plane model update ``threshold_per_cycle`` array.

        The adaptive threshold_per_cycle is allowed to differ for every
        ``(time, chan, pol)`` plane. Each plane present in ``imaging_dict`` gets
        its own value::

            clamp(max_psf_sidelobe * psf_sidelobe_factor, min_psf_fraction, max_psf_fraction)
                * peak_residual,

        floored at the absolute user ``threshold``. Because each plane uses its
        own ``peak_residual``, the result is independent of how the cube was
        chunked across tasks. Planes not present in ``imaging_dict`` fall back to
        the representative scalar threshold_per_cycle from
        :meth:`calculate_per_cycle_limits`.

        Parameters
        ----------
        imaging_dict : ImagingDict
            Per-plane statistics (``peakres`` and ``max_psf_sidelobe``).
        time, pol, chan : int, optional
            Restrict the computation to a selection (otherwise all planes).

        Returns
        -------
        numpy.ndarray
            ``(ntime, nchan, npol)`` array of per-plane threshold_per_cycle values, indexed
            ``(time, chan, pol)`` to match :attr:`max_iter_remaining`.
        """
        self._ensure_state(imaging_dict)
        # Representative threshold_per_cycle, used only as the fallback for any plane
        # that has no entry in imaging_dict.
        _, fallback_threshold_per_cycle = self.calculate_per_cycle_limits(
            imaging_dict, time=time, pol=pol, chan=chan
        )
        threshold_per_cycle = np.full(
            self.max_iter_remaining.shape, fallback_threshold_per_cycle, dtype=float
        )
        for key, fields in imaging_dict.data.items():
            if not self._matches(key, time, pol, chan):
                continue
            idx = self._key_index(key)
            peak = abs(self._latest(fields, "peakres", 0.0))
            sidelobe = self._latest(fields, "max_psf_sidelobe", 0.2)
            frac = min(
                max(sidelobe * self.psf_sidelobe_factor, self.min_psf_fraction),
                self.max_psf_fraction,
            )
            threshold_per_cycle[idx] = max(frac * peak, self._plane_threshold(idx[2]))
        return threshold_per_cycle

    def resolve_max_cycles(self, imaging_dict, sensitivity=None, mask_pixels=None):
        """Predict ``max_cycles="auto"`` from the dirty image.

        A no-op unless ``max_cycles`` is ``"auto"``. ``imaging_dict`` holds
        the statistics of the residual of the first residual update, the
        dirty image (:func:`build_residual_imaging_dict`). For every plane:

        - ``P0`` is its peak residual inside the clean mask (``peakres``), the
          peak its first model update starts from and sets its
          ``threshold_per_cycle`` from;
        - ``f`` is its PSF fraction, ``clamp(max_psf_sidelobe *
          psf_sidelobe_factor, min_psf_fraction, max_psf_fraction)``;
        - ``L`` is its threshold if that is above 0, else its theoretical
          point source sensitivity (``sensitivity``);
        - ``n = predicted_imaging_cycles(P0, L, f)``, the imaging cycles that
          bring the peak down from ``P0`` to ``L`` when every model update
          cleans to ``f`` times the peak it starts from (one cycle for
          ``f = 0``).

        The limit of the plane is ``n + EXTRA_CYCLES`` when ``L`` is its
        threshold and ``NOISY_CYCLE_SAFETY * n`` when ``L`` is its
        sensitivity (the controls of
        :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
        The sensitivity is the noise of the plane when the weights are
        absolute (``WEIGHT = 1 / sigma**2``). With unit weights (``WEIGHT =
        1``, as a simulation without noise writes them) it is not a noise
        level but, for natural weighting, about 1 over the square root of the
        number of visibilities, in the units of the data (1e-3 for a
        million). The limit grows only with ``ln(P0 / L)``: on the same
        prediction, ``NOISY_CYCLE_SAFETY * n`` cycles bring the peak down to
        ``P0 * (L / P0)**NOISY_CYCLE_SAFETY``.
        ``max_cycles`` becomes the largest limit over the planes, at least 1,
        and from then on counts down like a number given by the caller. A
        plane nothing can be predicted for is left out: one without a pixel
        in its clean mask, with a peak or a level ``L`` that is 0 or not
        finite (an infinite sensitivity: no data), or with a PSF fraction
        that is not a number.

        Parameters
        ----------
        imaging_dict : ImagingDict
            Statistics of the dirty image of every plane: ``peakres``,
            ``max_psf_sidelobe`` and ``masksum``.
        sensitivity : numpy.ndarray, optional
            ``(ntime, nchan, npol)`` theoretical point source sensitivity of
            the planes, indexed ``(time, chan, pol)``: the ``SENSITIVITY`` of
            :func:`~astroviper.processing_functions.imaging.weight_statistics.calculate_weight_statistics_single_field`.
            Needed for the planes whose threshold is 0, which are left out
            without it.
        mask_pixels : numpy.ndarray, optional
            ``(ntime, nchan, npol)`` number of pixels in the clean mask of the
            planes, indexed ``(time, chan, pol)``; the ``masksum`` of the
            entries when omitted.

        Returns
        -------
        int
            ``max_cycles``, resolved.
        """
        if not _is_auto_max_cycles(self.max_cycles):
            return self.max_cycles
        limits = {}
        for key, fields in imaging_dict.data.items():
            idx = self._key_index(key)
            peak = abs(float(self._latest(fields, "peakres", 0.0)))
            pixels = (
                self._latest(fields, "masksum", 0)
                if mask_pixels is None
                else mask_pixels[idx]
            )
            threshold = self._plane_threshold(idx[2])
            if threshold > 0:
                level = float(threshold)
            else:
                level = np.nan if sensitivity is None else float(sensitivity[idx])
            fraction = min(
                max(
                    self._latest(fields, "max_psf_sidelobe", 0.2)
                    * self.psf_sidelobe_factor,
                    self.min_psf_fraction,
                ),
                self.max_psf_fraction,
            )
            if not (
                pixels > 0
                and np.isfinite(peak)
                and peak > 0
                and np.isfinite(level)
                and level > 0
                and np.isfinite(fraction)
            ):
                limits[idx] = None
                continue
            cycles = (
                1 if fraction <= 0 else predicted_imaging_cycles(peak, level, fraction)
            )
            limits[idx] = (
                cycles + EXTRA_CYCLES if threshold > 0 else NOISY_CYCLE_SAFETY * cycles
            )
        self.max_cycles_auto = limits
        self.max_cycles = max(
            (limit for limit in limits.values() if limit is not None), default=1
        )
        return self.max_cycles

    def _require_resolved_max_cycles(self):
        """Raise while ``max_cycles`` is still ``"auto"``."""
        if _is_auto_max_cycles(self.max_cycles):
            raise ValueError(
                f'max_cycles="{AUTO_MAX_CYCLES}" is not resolved yet: call '
                "resolve_max_cycles with the dirty image (resolve_auto_max_cycles "
                "in the imaging cycle loop) before the cycles are checked or "
                "counted."
            )

    def check_convergence(
        self,
        imaging_dict: ImagingDict,
        time: int | None = None,
        pol: int | None = None,
        chan: int | None = None,
        model_update_ran: bool = False,
    ) -> tuple[StopCode, str]:
        """
        Cycle control: the stop test on the residual of a residual update.

        The imaging cycle loop makes it after every residual update, the
        first included, on :func:`build_residual_imaging_dict` of the
        residual that update has just made, and a plane that continues gets
        another model update; the residual a model update ends with decides
        nothing (:meth:`record_model_update` books the model update).

        This implements CASA's convergence checking from imager_return_dict.py:463-565.

        StopCode(imaging, model_update) separates the imaging cycle and
        model update convergence criteria. This handles the "degeneracy in stopcode
        numbers" (CASA comment line 476-477).

        Imaging cycle Stopping Criteria (in order of precedence):
        --------------------------------------------------------
        1. Zero mask (stopcode 7): No valid pixels to clean
        2. Iteration limit (stopcode 1): max_iter_remaining <= 0
        3. Threshold reached (stopcode 2): peak_residual <= threshold *
           (1 + threshold_tolerance), with ``peak_residual`` the ``peakres``
           of the entry (the peak of the residual inside the clean mask)
        4. Residual entropy (stopcode 10), only with ``entropy_stop``: the
           entropy of the plane's residual (``entropy`` in the ImagingDict,
           see :func:`build_residual_imaging_dict`) is lower than the highest
           entropy of an earlier imaging cycle. The entropy rises while the
           clean removes emission and falls once the clean starts to fit
           noise (Homan, Roth and Pushkarev 2024, AJ 167, 11), so a fall means
           that the maximum has been passed. Only residuals whose peak is at
           most ``entropy_max_snr`` times their RMS count. The stop is final
           for the plane: it gets no more iterations and keeps this stop code.
           The test is made on the residual of a residual update
           (``model_update_ran`` False), so it notices the fall one model
           update after the maximum; the model of that last model update is
           kept.
        5. Divergence (stopcode 5), only with ``max_cycles_divergence`` > 0:
           the plane's last ``max_cycles_divergence`` model updates all ended
           diverged. The stop is final for the plane: it gets no more
           iterations and keeps this stop code.
        6. Imaging cycle limit (stopcode 9): max_cycles == 0 (if not -1). A
           ``max_cycles="auto"`` has to be resolved first
           (:meth:`resolve_max_cycles`).
        7. No progress (stopcode 4): the plane's last
           ``NO_PROGRESS_MODEL_UPDATES`` model updates did no iteration. Such
           a plane cannot change its residual, so without this stop it would
           cycle for ever whenever ``threshold`` is 0 and ``max_cycles`` is -1
           (for example an all-zero Stokes plane).

        Model update Stopping Criteria:
        -------------------------------
        - Checked by the deconvolver (iteration control:
          max_iter_per_cycle_updated, threshold_per_cycle, divergence, and
          for Asp tests of its own).
        - The deconvolver reports why every plane's model update ended through
          the ``stop_code`` placeholder of the ImagingDict entry (1
          ``MODEL_UPDATE_MAX_ITER_PER_CYCLE``, 2
          ``MODEL_UPDATE_THRESHOLD_PER_CYCLE``, 4 ``MODEL_UPDATE_DIVERGENCE``,
          5 ``MODEL_UPDATE_ALGORITHM``, 0 did not run).
          :meth:`record_model_update` makes it the plane's model update stop
          code, kept until a later model update runs for the plane: a check
          on the residual alone, or a model update that gave the plane no
          iterations, leaves it as it is. No model update stop code ends the
          plane's imaging cycles by itself: the next residual update
          recomputes the true residual and the stop test is made on it.

        Parameters:
        -----------
        imaging_dict : ImagingDict
            ImagingDict containing deconvolution statistics including:
            - 'peakres': Current peak residual
            - 'masksum': Sum of mask (number of valid pixels)

        time : int, optional
            Filter by specific time index

        pol : int, optional
            Filter by specific polarization index

        chan : int, optional
            Filter by specific channel index
        model_update_ran : bool, optional
            ``True`` when ``imaging_dict`` is the result of a model update:
            it is booked first (:meth:`record_model_update`) and then tested.
            The imaging cycle loop does not use this. ``False`` (default) for
            the stop test on the residual of a residual update; the entropy
            stop is decided on such a test.

        Returns:
        --------
        stopcode : StopCode
            Aggregate StopCode(imaging, model_update) across the selected planes.
            imaging=0 (CONTINUE) while *any* selected plane is still active;
            once every plane has stopped it is a representative nonzero code
            (the shared code if uniform, else the largest). Per-plane codes
            are available in ``stop_code_imaging`` / ``stop_code_model_update``.

        stopdescription : str
            Human-readable description of the aggregate stop reason.

        Side Effects:
        -------------
        Evaluates each selected ``(time, pol, chan)`` plane independently and
        writes that plane's own StopCode and description into the 'stop_code'
        and 'stop_description' fields of the corresponding ImagingDict entry,
        overwriting the placeholder set by the deconvolver. With
        ``max_cycles="auto"`` the entry also gets the plane's predicted limit
        as 'max_cycles_auto' (:attr:`max_cycles_auto`). Also allocates /
        updates the per-plane ``max_iter_remaining``, ``stop_code_imaging`` and
        ``stop_code_model_update`` arrays.

        Raises:
        -------
        ValueError
            While ``max_cycles`` is an unresolved ``"auto"``.

        Example:
        --------
        >>> controller = CycleController(max_iter=100, threshold=0.01)
        >>> # After a residual update...
        >>> stopcode, desc = controller.check_convergence(residual_imaging_dict)
        >>> if stopcode.imaging != 0:
        >>>     print(f"Converged: {desc}")
        >>> # Check both the imaging and the model update stop code
        >>> if stopcode.imaging != 0 or stopcode.model_update != 0:
        >>>     print(f"Stopped: imaging={stopcode.imaging}, model_update={stopcode.model_update}")
        """
        self._ensure_state(imaging_dict)
        self._require_resolved_max_cycles()
        if model_update_ran:
            self.record_model_update(imaging_dict, time=time, pol=pol, chan=chan)

        # Evaluate every selected plane independently and record its own stop
        # code. The aggregate returned to the caller is CONTINUE while any
        # plane is still active.
        plane_imaging_codes = []
        for key, fields in imaging_dict.data.items():
            if not self._matches(key, time, pol, chan):
                continue
            idx = self._key_index(key)
            peak_residual = abs(self._latest(fields, "peakres", 0.0))
            masksum = self._latest(fields, "masksum", 0)
            remaining = int(self.max_iter_remaining[idx])
            threshold = self._plane_threshold(idx[2])

            # Entropy of the fresh residual of this plane against the highest
            # entropy of its earlier imaging cycles.
            if self.entropy_stop and not model_update_ran:
                self._follow_entropy(idx, fields)

            model_update_code = int(self.stop_code_model_update[idx])

            # Imaging cycle stopping criteria, in priority order (per plane):
            #   1 zero mask, 2 iteration limit, 3 threshold, 4 residual entropy,
            #   5 divergence, 6 imaging cycle limit, 7 no progress
            if masksum == 0:
                maj = IMAGING_ZERO_MASK
            elif remaining <= 0:
                maj = IMAGING_MAX_ITER
            elif threshold > 0 and peak_residual <= threshold * (
                1 + self.threshold_tolerance
            ):
                maj = IMAGING_THRESHOLD
            elif self.entropy_stopped[idx]:
                maj = IMAGING_ENTROPY
            elif self.divergence_stopped[idx]:
                maj = IMAGING_DIVERGENCE
            elif self.max_cycles != -1 and self.max_cycles <= 0:
                maj = IMAGING_MAX_CYCLES
            elif self.zero_iter_model_updates[idx] >= NO_PROGRESS_MODEL_UPDATES:
                maj = IMAGING_NO_PROGRESS
            else:
                maj = IMAGING_CONTINUE

            self.stop_code_imaging[idx] = maj
            plane_imaging_codes.append(maj)

            # Stamp this plane's stop code/description into the ImagingDict,
            # replacing the placeholder set by the deconvolver. Written
            # directly (not via add()) so it stays a single value.
            fields["stop_code"] = StopCode(imaging=maj, model_update=model_update_code)
            fields["stop_description"] = IMAGING_STOP_DESCRIPTIONS[maj]
            if self.max_cycles_auto is not None:
                fields["max_cycles_auto"] = self.max_cycles_auto.get(idx)

        # Aggregate across the selected planes.
        if not plane_imaging_codes:
            # No matching planes (e.g. an empty ImagingDict): nothing left to
            # clean, so report a stop (matches the historical zero-mask result).
            agg_imaging = IMAGING_ZERO_MASK
            self.stopdescription = IMAGING_STOP_DESCRIPTIONS[IMAGING_ZERO_MASK]
        elif any(m == IMAGING_CONTINUE for m in plane_imaging_codes):
            agg_imaging = IMAGING_CONTINUE
            self.stopdescription = IMAGING_STOP_DESCRIPTIONS[IMAGING_CONTINUE]
        elif len(set(plane_imaging_codes)) == 1:
            agg_imaging = plane_imaging_codes[0]
            self.stopdescription = IMAGING_STOP_DESCRIPTIONS[agg_imaging]
        else:
            # All planes stopped, but for different reasons.
            agg_imaging = max(plane_imaging_codes)
            self.stopdescription = "All planes stopped (mixed reasons)"

        self.stopcode = StopCode(
            imaging=agg_imaging, model_update=MODEL_UPDATE_CONTINUE
        )
        return self.stopcode, self.stopdescription

    def record_model_update(
        self,
        imaging_dict: ImagingDict,
        time: int | None = None,
        pol: int | None = None,
        chan: int | None = None,
    ) -> None:
        """Book one model update for the stop test after the next residual update.

        For every selected plane: the run of model updates without an
        iteration (no progress stop), the model update stop code the
        deconvolver reported in the ``stop_code`` placeholder of the entry,
        and the run of diverged model updates (divergence stop). A plane given
        no iterations reports ``MODEL_UPDATE_CONTINUE`` and keeps its code and
        its run of diverged model updates. Nothing is decided here and
        ``stopcode`` is left as it is: :meth:`check_convergence` decides on
        the residual the next residual update recomputes.

        Parameters
        ----------
        imaging_dict : ImagingDict
            Result of the model update (``iter_done`` and ``stop_code``).
        time, pol, chan : int, optional
            Restrict the booking to a selection (otherwise all planes).

        Raises
        ------
        ValueError
            If a plane reports a model update stop code that is not in
            ``MODEL_UPDATE_STOP_DESCRIPTIONS``.
        """
        self._ensure_state(imaging_dict)
        for key, fields in imaging_dict.data.items():
            if not self._matches(key, time, pol, chan):
                continue
            idx = self._key_index(key)
            if int(self._latest(fields, "iter_done", 0)) == 0:
                self.zero_iter_model_updates[idx] += 1
            else:
                self.zero_iter_model_updates[idx] = 0
            placeholder = fields.get("stop_code", None)
            reported = (
                int(placeholder.model_update)
                if isinstance(placeholder, StopCode)
                else MODEL_UPDATE_CONTINUE
            )
            if reported not in MODEL_UPDATE_STOP_DESCRIPTIONS:
                raise ValueError(
                    f"Unknown model-update stop code {reported} for plane {key}."
                )
            if reported == MODEL_UPDATE_CONTINUE:
                continue
            self.stop_code_model_update[idx] = reported
            if reported == MODEL_UPDATE_DIVERGENCE:
                self.diverged_model_updates[idx] += 1
            else:
                self.diverged_model_updates[idx] = 0
            if (
                self.max_cycles_divergence > 0
                and self.diverged_model_updates[idx] >= self.max_cycles_divergence
            ):
                self.divergence_stopped[idx] = True

    def _follow_entropy(self, idx, fields):
        """Entropy stop of one plane, from the residual of a residual update.

        ``fields`` is the plane's ImagingDict entry. Its latest ``entropy`` is
        compared with the highest entropy of the plane's earlier imaging
        cycles (``max_entropy``): a lower value sets ``entropy_stopped``, a
        higher one becomes the new maximum. An entry without an entropy, or
        with one that is not a number (the peak of the residual is above
        ``entropy_max_snr`` times its RMS), changes nothing. Nor does any
        entropy of a plane that has been stopped already.
        """
        if self.entropy_stopped[idx]:
            return
        entropy = self._latest(fields, "entropy", np.nan)
        if entropy is None or not np.isfinite(entropy):
            return
        highest = self.max_entropy[idx]
        if np.isfinite(highest) and entropy < highest:
            self.entropy_stopped[idx] = True
        elif not np.isfinite(highest) or entropy > highest:
            self.max_entropy[idx] = entropy

    def update_counts(
        self,
        imaging_dict: ImagingDict,
        time: int | None = None,
        pol: int | None = None,
        chan: int | None = None,
    ) -> None:
        """
        Update iteration counts after a imaging cycle completes.

        Updates:
        --------
        1. Extracts iterations_done from imaging_dict
        2. Decrements max_iter_remaining by iterations_done
        3. Decrements max_cycles by 1 (if not -1)
        4. Increments cycles_done and total_iter_done
        5. Enforces floor values (no negatives)

        Parameters:
        -----------
        imaging_dict : ImagingDict
            ImagingDict containing iteration statistics including:
            - 'iter_done': Number of iterations completed in this imaging cycle

        time : int, optional
            Filter by specific time index

        pol : int, optional
            Filter by specific polarization index

        chan : int, optional
            Filter by specific channel index

        Raises:
        -------
        ValueError
            While ``max_cycles`` is an unresolved ``"auto"``
            (:meth:`resolve_max_cycles`).

        Example:
        --------
        >>> controller = CycleController(max_iter=1000, max_cycles=5)
        >>> # After imaging cycle completes...
        >>> controller.update_counts(imaging_dict)
        >>> # max_iter_remaining is an (ntime, nchan, npol) array, one
        >>> # remaining budget per plane -- not a scalar.
        >>> print(controller.max_iter_remaining, controller.max_cycles, controller.cycles_done)
        [[[900]]] 4 1
        """
        # Only update if not converged (check both the imaging and the model update code)
        if (
            self.stopcode.imaging != IMAGING_CONTINUE
            or self.stopcode.model_update != MODEL_UPDATE_CONTINUE
        ):
            return

        self._ensure_state(imaging_dict)
        self._require_resolved_max_cycles()

        # Decrement the global imaging cycle count (imaging cycles are shared
        # across planes) once per call.
        if self.max_cycles != -1:
            self.max_cycles = max(self.max_cycles - 1, 0)

        # Decrement each selected plane's remaining iterations by the work it
        # did this cycle. Planes that already stopped are left untouched.
        cycle_iters = 0
        for key, fields in imaging_dict.data.items():
            if not self._matches(key, time, pol, chan):
                continue
            idx = self._key_index(key)
            iters = int(self._latest(fields, "iter_done", 0))
            if self.stop_code_imaging[idx] == IMAGING_CONTINUE:
                self.max_iter_remaining[idx] = max(
                    int(self.max_iter_remaining[idx]) - iters, 0
                )
            cycle_iters += iters

        # Update tracking counters
        self.cycles_done += 1
        self.total_iter_done += cycle_iters

    def update_parameters(
        self,
        max_iter: int | None = None,
        max_iter_per_cycle: int | None = None,
        max_cycles: int | None = None,
        threshold: float | None = None,
        psf_sidelobe_factor: float | None = None,
    ) -> tuple[int, str]:
        """
        Update cycle control parameters with validation.

        This implements CASA's interactive parameter update from _gclean.py:127-180.
        Used in interactive clean workflows.

        Parameters:
        -----------
        max_iter : int, optional
            New maximum iteration count

        max_iter_per_cycle : int, optional
            New iterations per model update

        max_cycles : int, optional
            New imaging cycle limit (checked by :func:`validate_max_cycles`).
            ``"auto"`` is refused: it is given when the controller is
            created and resolved from the dirty image.

        threshold : float or str, optional
            New stopping threshold (can include units like "10mJy")

        psf_sidelobe_factor : float, optional
            New cycle factor for adaptive thresholding

        Returns:
        --------
        error_code : int
            0 if successful, -1 if validation failed

        error_message : str
            Empty string if successful, error description if failed
        """
        # Update and validate max_iter (the total budget) and reset the per-plane
        # max_iter_remaining array to it if that array has already been allocated.
        if max_iter is not None:
            try:
                max_iter_int = int(max_iter)
                if max_iter_int < -1:
                    return -1, "max_iter must be >= -1"
                self._max_iter = max_iter_int
                if self.max_iter_remaining is not None:
                    self.max_iter_remaining[...] = max_iter_int
            except (ValueError, TypeError):
                return -1, "max_iter must be an integer"

        # Update and validate max_iter_per_cycle
        if max_iter_per_cycle is not None:
            try:
                max_iter_per_cycle_int = int(max_iter_per_cycle)
                if max_iter_per_cycle_int < -1:
                    return -1, "max_iter_per_cycle must be >= -1"
                self.max_iter_per_cycle = max_iter_per_cycle_int
            except (ValueError, TypeError):
                return -1, "max_iter_per_cycle must be an integer"

        # Update and validate max_cycles
        if max_cycles is not None:
            if _is_auto_max_cycles(max_cycles):
                return (
                    -1,
                    f'max_cycles="{AUTO_MAX_CYCLES}" is given when the controller '
                    "is created (it is resolved from the dirty image), not updated",
                )
            try:
                self.max_cycles = validate_max_cycles(max_cycles)
            except ValueError:
                return -1, f"max_cycles must be an integer >= -1; got {max_cycles!r}"

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

        # Update and validate psf_sidelobe_factor
        if psf_sidelobe_factor is not None:
            try:
                psf_sidelobe_factor_float = float(psf_sidelobe_factor)
                if psf_sidelobe_factor_float <= 0:
                    return -1, "psf_sidelobe_factor must be > 0"
                self.psf_sidelobe_factor = psf_sidelobe_factor_float
            except (ValueError, TypeError):
                return -1, "psf_sidelobe_factor must be a number"

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
        """Reset the cycle controller to initial state."""
        if self.max_iter_remaining is not None:
            self.max_iter_remaining[...] = self._max_iter
            self.stop_code_imaging[...] = IMAGING_CONTINUE
            self.stop_code_model_update[...] = MODEL_UPDATE_CONTINUE
            self.zero_iter_model_updates[...] = 0
            self.max_entropy[...] = np.nan
            self.entropy_stopped[...] = False
            self.diverged_model_updates[...] = 0
            self.divergence_stopped[...] = False
        self.cycles_done = 0
        self.total_iter_done = 0
        self.stopcode = StopCode(
            imaging=IMAGING_CONTINUE, model_update=MODEL_UPDATE_CONTINUE
        )
        self.stopdescription = IMAGING_STOP_DESCRIPTIONS[IMAGING_CONTINUE]

    def reset_stopcode(self) -> None:
        """Reset the aggregate and per-plane stop codes of the controller."""
        self.stopcode = StopCode(
            imaging=IMAGING_CONTINUE, model_update=MODEL_UPDATE_CONTINUE
        )
        self.stopdescription = IMAGING_STOP_DESCRIPTIONS[IMAGING_CONTINUE]
        if self.stop_code_imaging is not None:
            self.stop_code_imaging[...] = IMAGING_CONTINUE
            self.stop_code_model_update[...] = MODEL_UPDATE_CONTINUE
            self.zero_iter_model_updates[...] = 0
            self.max_entropy[...] = np.nan
            self.entropy_stopped[...] = False
            self.diverged_model_updates[...] = 0
            self.divergence_stopped[...] = False

    def get_state(self) -> dict[str, Any]:
        """Get current state of the cycle controller as a dictionary.

        Note: The stopcode is serialized as a dict with 'imaging' and 'model_update' keys
        to preserve the namedtuple structure across serialization.
        """
        return {
            "max_iter_remaining": self.max_iter_remaining.tolist()
            if self.max_iter_remaining is not None
            else None,
            "max_cycles": self.max_cycles,
            "max_iter": self._max_iter,
            "threshold": list(self.threshold)
            if np.ndim(self.threshold)
            else self.threshold,
            "threshold_sigma": self.threshold_sigma,
            "threshold_tolerance": self.threshold_tolerance,
            "gain": self.gain,
            "psf_sidelobe_factor": self.psf_sidelobe_factor,
            "min_psf_fraction": self.min_psf_fraction,
            "max_psf_fraction": self.max_psf_fraction,
            "max_iter_per_cycle": self.max_iter_per_cycle,
            "entropy_stop": self.entropy_stop,
            "entropy_max_snr": self.entropy_max_snr,
            "entropy_spatial_bins": self.entropy_spatial_bins,
            "entropy_flux_bins": self.entropy_flux_bins,
            "max_cycles_divergence": self.max_cycles_divergence,
            "cycles_done": self.cycles_done,
            "total_iter_done": self.total_iter_done,
            "stopcode": {
                "imaging": self.stopcode.imaging,
                "model_update": self.stopcode.model_update,
            },
            "stopdescription": self.stopdescription,
        }


# ============================================================================
# Convergence Visualization
# ============================================================================


class ConvergencePlots:
    """
    Class for creating convergence visualization plots from ImagingDict.

    This class provides methods to create interactive HoloViews plots showing
    deconvolution convergence history, including peak residual evolution over
    iterations.

    Parameters
    ----------
    imaging_dict : ImagingDict
        ImagingDict object with convergence history (peakres, iter_done fields)

    Attributes
    ----------
    imaging_dict : ImagingDict
        The ImagingDict containing convergence data
    stokes_to_pol : dict
        Mapping from Stokes parameter names to polarization indices

    Examples
    --------
    >>> rd = ImagingDict()
    >>> for cycle in range(5):
    ...     rd.add({'peakres': 1.0 * 0.7**cycle, 'iter_done': 100},
    ...            time=0, pol=0, chan=0)
    >>> plotter = ConvergencePlots(rd)
    >>> plot = plotter.plot_history(time=0, stokes='I', chan=0)
    >>> plot  # Display in Jupyter notebook
    """

    def __init__(self, imaging_dict):
        """
        Initialize ConvergencePlots with a ImagingDict.

        Parameters
        ----------
        imaging_dict : ImagingDict
            ImagingDict object containing convergence history
        """
        self.imaging_dict = imaging_dict
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

        if key not in self.imaging_dict.data:
            # Show error message
            return hv.Curve([]).opts(
                title=f"No data for Time={self.time}, Stokes={stokes_sel}, Channel={chan_sel}",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
                width=self.width,
                height=self.height,
                show_grid=True,
            )

        data = self.imaging_dict.data[key]

        # Extract history
        peakres_history = data.get("peakres", [])
        iter_done_history = data.get("iter_done", [])
        model_flux_history = data.get("model_flux", [])

        # Handle single values (convert to list)
        if not isinstance(peakres_history, list):
            peakres_history = [peakres_history]
        if not isinstance(iter_done_history, list):
            iter_done_history = [iter_done_history]
        if not isinstance(model_flux_history, list):
            model_flux_history = [model_flux_history]

        if not peakres_history or not iter_done_history:
            return hv.Curve([]).opts(
                title=f"No convergence history - Time={self.time}, Stokes={stokes_sel}, Channel={chan_sel}",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
                width=self.width,
                height=self.height,
                show_grid=True,
            )

        # Prepend initial state at iteration 0
        start_peakres_history = data.get("start_peakres", [])
        start_model_flux_history = data.get("start_model_flux", [])
        if not isinstance(start_peakres_history, list):
            start_peakres_history = [start_peakres_history]
        if not isinstance(start_model_flux_history, list):
            start_model_flux_history = [start_model_flux_history]

        if start_peakres_history:
            peakres_history = [start_peakres_history[0]] + list(peakres_history)
        if start_model_flux_history:
            model_flux_history = [start_model_flux_history[0]] + list(
                model_flux_history
            )

        # Calculate cumulative iterations (with 0 prepended for initial state)
        cumulative_iters = np.concatenate([[0], np.cumsum(iter_done_history)])

        # Create peak residual curve (left y-axis)
        peakres_data = list(zip(cumulative_iters, peakres_history, strict=False))
        peakres_curve = hv.Curve(
            peakres_data, kdims=["Cumulative Iterations"], vdims=["Peak Residual (Jy)"]
        ).opts(
            color="blue",
            line_width=2,
            tools=["hover"],
        )

        # Create model flux curve (right y-axis) if available
        if model_flux_history and any(f is not None for f in model_flux_history):
            modelflux_data = list(
                zip(cumulative_iters, model_flux_history, strict=False)
            )
            modelflux_curve = hv.Curve(
                modelflux_data,
                kdims=["Cumulative Iterations"],
                vdims=["Model Flux (Jy)"],
            ).opts(
                color="red",
                line_width=2,
                tools=["hover"],
            )

            # Overlay both curves with dual y-axes
            overlay = (peakres_curve * modelflux_curve).opts(
                title=f"Convergence History - Time={self.time}, Stokes={stokes_sel}, Channel={chan_sel}",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
                width=self.width,
                height=self.height,
                show_grid=True,
                show_legend=True,
                responsive=self.responsive,
                multi_y=True,  # Enable dual y-axes
            )
            return overlay
        else:
            # Fall back to single plot if no model flux data
            return peakres_curve.opts(
                title=f"Convergence History - Time={self.time}, Stokes={stokes_sel}, Channel={chan_sel}",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
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

        Creates an interactive HoloViews plot showing peak residual and model flux
        evolution over iterations, with widgets to select Stokes parameter and channel.
        Displays dual y-axis plot with Peak Residual (left, blue) and Model Flux (right, red).

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
            - width : int, plot width in pixels per subplot (default: 700)
            - height : int, plot height in pixels (default: 400)
            - responsive : bool, make plot responsive (default: False)

        Returns
        -------
        holoviews.DynamicMap
            Interactive dual y-axis plot with Stokes and channel selector widgets

        Notes
        -----
        - Requires holoviews with bokeh backend
        - Uses lazy imports to avoid hard dependency
        - Falls back to single-axis plot if no model_flux data in ImagingDict
        """
        # Store plotting parameters as instance variables for access in make_plot
        self.time = time
        self.width = kwargs.get("width", 700)
        self.height = kwargs.get("height", 400)
        self.responsive = kwargs.get("responsive", False)

        # Lazy imports
        try:
            import holoviews as hv

            hv.extension("bokeh")
        except ImportError as e:
            raise ImportError(
                "ConvergencePlots requires holoviews and bokeh. "
                "Install with: pip install holoviews bokeh"
            ) from e

        # Extract available channels and stokes from ImagingDict
        available_keys = list(self.imaging_dict.data.keys())
        if not available_keys:
            return hv.Curve([]).opts(
                title="No data in ImagingDict",
                xlabel="Cumulative Iterations",
                ylabel="Peak Residual (Jy)",
            )

        channels = sorted({k.chan for k in available_keys if k.time == time})
        pols = sorted({k.pol for k in available_keys if k.time == time})
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


def plot_convergence_history(imaging_dict, time=0, stokes="I", chan=0, **kwargs):
    """
    Plot interactive convergence history from ImagingDict.

    Convenience function that wraps ConvergencePlots.plot_history() for
    backward compatibility and quick plotting. Displays dual y-axis plot
    of peak residual (blue, left) and model flux (red, right) evolution over iterations.

    Parameters
    ----------
    imaging_dict : ImagingDict
        ImagingDict object with convergence history (peakres, iter_done, model_flux fields)
    time : int, optional
        Time index to plot (default: 0)
    stokes : str, optional
        Initial Stokes parameter to display: 'I', 'Q', 'U', or 'V' (default: 'I')
    chan : int, optional
        Initial channel index to display (default: 0)
    **kwargs : dict, optional
        Additional plotting options (e.g., width, height)

    Returns
    -------
    holoviews.DynamicMap
        Interactive dual y-axis plot with Stokes and channel selector widgets

    Examples
    --------
    >>> rd = ImagingDict()
    >>> for cycle in range(5):
    ...     rd.add({'peakres': 1.0 * 0.7**cycle, 'iter_done': 100, 'model_flux': 0.5 * cycle},
    ...            time=0, pol=0, chan=0)
    >>> plot = plot_convergence_history(rd, time=0, stokes='I', chan=0)
    >>> plot  # Display in Jupyter notebook

    Notes
    -----
    - Requires holoviews with bokeh backend
    - Uses lazy imports to avoid hard dependency
    - Displays error message if selected (time, pol, chan) not found
    - Falls back to single-axis plot if no model_flux data in ImagingDict
    - For more control, use ConvergencePlots class directly
    """
    plotter = ConvergencePlots(imaging_dict)
    return plotter.plot_history(time=time, stokes=stokes, chan=chan, **kwargs)


# ============================================================================
# ImagingDict Pretty-Printing
# ============================================================================


def format_imaging_dict(combined_imaging_dict, float_format="{:.6g}"):
    """Return a human-readable string representation of a deconvolution ImagingDict.

    A deconvolution ImagingDict maps ``Key(time, pol, chan)`` planes to field
    dicts that mix constant parameters (``max_iter``, ``threshold``, ...) with
    per-imaging cycle history lists (``peakres``, ``iter_done``, ``model_flux``,
    ...). The default ``repr`` dumps each plane on one very long line, which is
    hard to read. This formatter groups each plane, separates scalar parameters
    from the per-cycle history, and lays the history out as an aligned table
    with one column per imaging cycle (the record of the imaging cycle loop
    ends with a column ``stop``, its closing stop test). Numpy scalars
    (``np.float64``, ``np.str_``) are unwrapped so they print as plain
    values; an array in the history (the
    iteration history of a model update, ``record_iteration_history``) prints
    as the number of its values.

    Parameters
    ----------
    combined_imaging_dict : ImagingDict or dict
        Either a ImagingDict instance or its underlying ``.data`` mapping of
        ``Key(time, pol, chan)`` -> field dict.
    float_format : str, optional
        Format string applied to floating point values (default ``"{:.6g}"``).

    Returns
    -------
    str
        The formatted, multi-line representation.
    """
    # Accept either a ImagingDict (exposes .data) or a plain mapping.
    data = getattr(combined_imaging_dict, "data", combined_imaging_dict)

    def to_py(v):
        # Unwrap numpy scalars (np.float64, np.str_, ...) to native Python types.
        return v.item() if hasattr(v, "item") else v

    def fmt(v):
        if isinstance(v, np.ndarray) and v.ndim:
            return f"<{v.size} values>"
        v = to_py(v)
        if v is None:
            return "None"
        if isinstance(v, float):
            return float_format.format(v)
        return str(v)

    if not data:
        return "<empty deconvolve dict>"

    lines = []
    for key, fields in data.items():
        # Split fields into per-cycle history (lists) and scalar parameters.
        history = {f: list(v) for f, v in fields.items() if isinstance(v, list)}
        scalars = {f: v for f, v in fields.items() if not isinstance(v, list)}

        time = getattr(key, "time", key[0] if len(key) > 0 else "?")
        pol = getattr(key, "pol", key[1] if len(key) > 1 else "?")
        chan = getattr(key, "chan", key[2] if len(key) > 2 else "?")
        stokes = scalars.get("stokes")
        header = f"Key(time={time}, pol={pol}, chan={chan})"
        if stokes is not None:
            header += f"  -  Stokes {to_py(stokes)}"

        lines.append("=" * 78)
        lines.append(header)
        lines.append("-" * 78)

        # Scalar parameters (skip stokes, already shown in the header).
        scalar_items = [(f, v) for f, v in scalars.items() if f != "stokes"]
        if scalar_items:
            label_w = max(len(f) for f, _ in scalar_items)
            lines.append("Parameters:")
            for f, v in scalar_items:
                lines.append(f"  {f:<{label_w}} : {fmt(v)}")

        # Per-cycle history table (one column per imaging cycle). The record of
        # the imaging cycle loop ends with the closing stop test, which has no
        # model update fields (start_peakres).
        if history:
            n_cycles = max(len(v) for v in history.values())
            label_w = max([len(f) for f in history] + [len("cycle")])
            # Format every cell, then right-justify to a uniform width.
            formatted = {f: [fmt(x) for x in v] for f, v in history.items()}
            closing = int(
                ends_with_stop_test(
                    history.get("iter_done", []), history.get("start_peakres", [])
                )
            )
            n_model_updates = n_cycles - closing
            cycle_labels = [str(i) for i in range(n_model_updates)] + ["stop"] * closing
            cell_w = max(
                [len(c) for cells in formatted.values() for c in cells]
                + [len(c) for c in cycle_labels]
            )

            def row(label, cells, cell_w=cell_w, label_w=label_w):
                padded = " ".join(f"{c:>{cell_w}}" for c in cells)
                return f"  {label:<{label_w}} : {padded}"

            plural = "s" if n_model_updates != 1 else ""
            stop = " and the closing stop test" if closing else ""
            lines.append(f"Per-cycle history ({n_model_updates} cycle{plural}{stop}):")
            lines.append(row("cycle", cycle_labels))
            for f, cells in formatted.items():
                lines.append(row(f, cells))
        lines.append("")

    return "\n".join(lines)


def print_imaging_dict(combined_imaging_dict, float_format="{:.6g}"):
    """Pretty-print a deconvolution ImagingDict. See :func:`format_imaging_dict`."""
    print(format_imaging_dict(combined_imaging_dict, float_format=float_format))


def _clean_mask(img_xds, image_data_group_in_name, primary_beam_limit):
    """The clean mask of a residual data group as a bool cube, or ``None``.

    The mask of the data group or, before the first model update has made
    it, the mask that model update makes
    (:func:`~astroviper.processing_functions.image_analysis.make_mask.make_mask`
    with ``primary_beam_limit``: the nonzero primary beam at or above
    ``primary_beam_limit`` times its peak over the cube), made without a
    float copy of the primary beam and without touching ``img_xds``.
    ``None`` when the data group has neither a mask nor a primary beam.
    """
    data_group = img_xds.attrs["data_groups"][image_data_group_in_name]
    mask_name = data_group.get("mask", None)
    if mask_name is not None and mask_name in img_xds:
        return img_xds[mask_name].values.astype(bool, copy=False)
    primary_beam_name = data_group.get("primary_beam", None)
    if primary_beam_name is None or primary_beam_name not in img_xds:
        return None
    primary_beam = img_xds[primary_beam_name].values
    mask = primary_beam >= primary_beam_limit * np.nanmax(primary_beam)
    mask &= primary_beam != 0
    return mask


def build_residual_imaging_dict(
    img_xds, image_data_group_in_name, cycle_control_params
):
    """Seed a per-plane :class:`ImagingDict` of peak-residual/masksum stats
    straight from the current residual image, no deconvolution involved.

    The record of the stop test after every residual update
    (:meth:`CycleController.check_convergence`) and the seed of the per cycle
    limits of the model update that follows it (:func:`get_per_cycle_limits`).
    The clean mask is the mask of the data group or, before the first model
    update has made it, the mask that model update makes
    (``primary_beam_limit``); ``img_xds`` is not modified. Without a mask and
    a primary beam every pixel counts.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset providing the residual image.
    image_data_group_in_name : str
        Name of the entry in ``img_xds.attrs["data_groups"]`` whose
        ``"sky"`` key resolves to the residual variable.
    cycle_control_params : dict
        Cycle control parameters; ``gain`` seeds the placeholder
        per-plane field and ``primary_beam_limit`` (default 0) gives the
        clean mask before the first model update. With ``entropy_stop`` the
        entropy of the residual is worked out as well (``entropy_max_snr``,
        ``entropy_spatial_bins``, ``entropy_flux_bins``).

    Returns
    -------
    ImagingDict
        Per-plane ``peakres`` (the absolute peak of the residual inside the
        clean mask, 0 for an empty mask), ``peakres_nomask`` (that of the
        whole plane), ``masksum`` (the pixels of the clean mask) and
        ``iter_done`` (0), indexed ``(time, chan, pol)``. With ``entropy_stop``
        also ``entropy`` and ``residual_snr``, the entropy of the whole plane
        and the ratio of the peak inside the same clean mask to the RMS of
        the plane
        (:func:`~astroviper.processing_functions.image_analysis.image_statistics.image_residual_entropy`).
        With ``record_iteration_history`` also an empty
        ``iteration_peak_residual`` and ``iteration_residual_rms`` (no
        iteration ran), so that a merged record holds one iteration history
        per entry of ``iter_done``.
    """
    entropy_stop, entropy_max_snr, entropy_spatial_bins, entropy_flux_bins = (
        validate_entropy_params(cycle_control_params)
    )
    record_iteration_history = cycle_control_params.get(
        "record_iteration_history", False
    )
    residual_data_group = img_xds.attrs["data_groups"][image_data_group_in_name]
    residual = img_xds[residual_data_group["sky"]].values
    # plane by plane and block by block: no copy of the residual cube
    plane_peak_nomask = imgstats.cube_plane_abs_max(residual)  # (ntime, nfreq, npol)
    ntime, nfreq, npol = plane_peak_nomask.shape
    mask = _clean_mask(
        img_xds,
        image_data_group_in_name,
        cycle_control_params.get("primary_beam_limit", 0.0),
    )
    if mask is None:
        plane_peak = plane_peak_nomask
        masksum = imgstats.get_image_masksum(
            img_xds, data_group_name=image_data_group_in_name
        )
    else:
        plane_peak = np.empty_like(plane_peak_nomask)
        for index in np.ndindex(plane_peak.shape):
            plane_peak[index] = imgstats.plane_abs_max(residual[index], mask[index])
        masksum = mask.sum(axis=(-2, -1))
    max_psf_sidelobe_arr = img_xds[
        residual_data_group["max_sidelobe_point_spread_function"]
    ].values  # (ntime, nfreq, npol)
    if entropy_stop:
        entropy, residual_snr = imgstats.image_residual_entropy(
            img_xds,
            data_group_name=image_data_group_in_name,
            spatial_bins=entropy_spatial_bins,
            flux_bins=entropy_flux_bins,
            max_snr=entropy_max_snr,
            mask=mask,
        )
    rd = ImagingDict()
    for tt in range(ntime):
        for nn in range(nfreq):
            for pp in range(npol):
                fields = {
                    "peakres": float(plane_peak[tt, nn, pp]),
                    "peakres_nomask": float(plane_peak_nomask[tt, nn, pp]),
                    "masksum": int(masksum[tt, nn, pp]),
                    "iter_done": 0,
                    "max_psf_sidelobe": float(max_psf_sidelobe_arr[tt, nn, pp]),
                    "gain": cycle_control_params["gain"],
                }
                if entropy_stop:
                    fields["entropy"] = float(entropy[tt, nn, pp])
                    fields["residual_snr"] = float(residual_snr[tt, nn, pp])
                if record_iteration_history:
                    for field in ITERATION_HISTORY_FIELDS:
                        fields[field] = np.empty(0)
                rd.add(fields, time=tt, pol=pp, chan=nn)
    return rd


def copy_residual_entropy(imaging_dict, residual_imaging_dict):
    """Record the entropy of the residual in the result of a model update.

    A model update reports what the deconvolver did. The entropy belongs to
    the residual the model update started from, which
    :func:`build_residual_imaging_dict` has measured. This copies ``entropy``
    and ``residual_snr`` of every plane into ``imaging_dict``, so that the
    merged record holds one value per imaging cycle.

    Parameters
    ----------
    imaging_dict : ImagingDict
        Result of the model update. Modified in place.
    residual_imaging_dict : ImagingDict
        Statistics of the residual the model update started from.
    """
    for key, residual_fields in residual_imaging_dict.data.items():
        if key not in imaging_dict.data:
            continue
        for field in ("entropy", "residual_snr"):
            if field in residual_fields:
                value = residual_fields[field]
                imaging_dict.data[key][field] = (
                    list(value) if isinstance(value, list) else [value]
                )


def get_per_cycle_limits(controller, residual_imaging_dict):
    """Compute the per-plane ``max_iter_per_cycle_updated`` and ``threshold_per_cycle`` for the next model update.

    The controls of every model update are derived from
    ``residual_imaging_dict``, the record of the stop test on the residual the
    residual update before it has just made (each plane's own peak residual
    inside the clean mask): the dirty image for the first model update, the
    fresh residual for a later one.  Both arrays are built per
    ``(time, frequency, polarization)`` plane from that plane's own state, so
    the result is independent of how the cube was chunked across tasks.

    Parameters
    ----------
    controller : CycleController
        Controller whose ``calculate_per_cycle_limits`` and
        ``per_plane_threshold_per_cycle`` drive the result.
    residual_imaging_dict : ImagingDict
        :func:`build_residual_imaging_dict` of the residual the latest
        residual update has made, the record of the stop test before this
        model update.

    Returns
    -------
    max_iter_per_cycle_updated : numpy.ndarray
        ``(time, frequency, polarization)`` int array: the iterations each
        plane may spend in the next model update,
        ``min(max_iter_per_cycle, max_iter_remaining)`` per plane (0 for a
        plane the entropy or divergence stop has ended, and with a
        ``threshold_tolerance`` above 0 for a plane the threshold stop has
        ended).
    threshold_per_cycle : numpy.ndarray
        ``(time, frequency, polarization)`` float array of per-plane stopping
        thresholds for the next model update.
    """
    # Per-plane threshold_per_cycle so each plane is cleaned to its own depth
    # (allocates the controller's per-plane state if needed).
    threshold_per_cycle = controller.per_plane_threshold_per_cycle(
        residual_imaging_dict
    )
    # Scalar cap = min(max_iter_per_cycle, largest remaining budget); clipping
    # the per-plane remaining budget with it gives each plane
    # min(max_iter_per_cycle, its own max_iter_remaining). .clip returns a
    # copy; the controller's own budget is decremented later by update_counts.
    max_iter_per_cycle_cap, _ = controller.calculate_per_cycle_limits(
        residual_imaging_dict
    )
    max_iter_per_cycle_updated = controller.max_iter_remaining.clip(
        max=max_iter_per_cycle_cap
    )
    # A plane that the entropy or divergence stop has ended is not cleaned any
    # further while the other planes of its channel carry on.
    if controller.entropy_stopped is not None:
        max_iter_per_cycle_updated[controller.entropy_stopped] = 0
    if controller.divergence_stopped is not None:
        max_iter_per_cycle_updated[controller.divergence_stopped] = 0
    # Nor is a plane the threshold stop has ended within a tolerance above 0,
    # whose peak can lie above its threshold_per_cycle; without a tolerance the
    # peak is at or below it and the deconvolver does no iteration.
    if controller.threshold_tolerance > 0 and controller.stop_code_imaging is not None:
        max_iter_per_cycle_updated[
            controller.stop_code_imaging == IMAGING_THRESHOLD
        ] = 0

    return max_iter_per_cycle_updated, threshold_per_cycle


def resolve_auto_max_cycles(
    controller,
    residual_imaging_dict,
    ps_xdt,
    img_xds,
    cycle_control_params,
    processing_set_data_group_name="corrected",
    image_data_group_in_name="residual",
):
    """Resolve ``max_cycles="auto"`` of ``controller`` from the dirty image.

    A no-op unless the controller's ``max_cycles`` is still ``"auto"``: the
    imaging cycle loop calls it after every residual update, and only the
    first one, which makes the dirty image, resolves it (see
    :meth:`CycleController.resolve_max_cycles`). The clean mask is the mask
    of the residual data group or, before the first model update has made it,
    the mask that model update makes
    (:func:`~astroviper.processing_functions.image_analysis.make_mask.make_mask`
    with ``primary_beam_limit``). The level of a plane whose threshold is 0
    is its theoretical point source sensitivity, the ``SENSITIVITY`` the
    point spread function pass reports, from the imaging weights of
    ``ps_xdt``
    (:func:`~astroviper.processing_functions.imaging.weight_statistics.calculate_weight_statistics_single_field`;
    the noise of the plane when the weights are absolute, see
    :meth:`CycleController.resolve_max_cycles`). It takes one pass over the
    weights, made only when such a plane has a pixel in its clean mask.
    ``ps_xdt`` and ``img_xds`` are not modified.

    Parameters
    ----------
    controller : CycleController
        Controller whose ``max_cycles`` is resolved.
    residual_imaging_dict : ImagingDict
        :func:`build_residual_imaging_dict` of the dirty image.
    ps_xdt : xarray.DataTree
        The visibilities imaged, whose data group holds the imaging weights
        (``weight_imaging``).
    img_xds : xarray.Dataset
        Image dataset holding the dirty image and the primary beam, in the
        Stokes basis.
    cycle_control_params : dict
        Cycle control parameters; ``primary_beam_limit`` (default 0) gives the
        clean mask of the first model update.
    processing_set_data_group_name : str, optional
        Measurement-set data group of the weights. Default ``"corrected"``.
    image_data_group_in_name : str, optional
        Image data group holding the residual image. Default ``"residual"``.

    Returns
    -------
    int
        The controller's ``max_cycles``.
    """
    if not _is_auto_max_cycles(controller.max_cycles):
        return controller.max_cycles
    from astroviper.processing_functions.imaging.weight_statistics import (
        calculate_weight_statistics_single_field,
    )

    mask = _clean_mask(
        img_xds,
        image_data_group_in_name,
        cycle_control_params.get("primary_beam_limit", 0.0),
    )
    mask_pixels = (
        imgstats.get_image_masksum(img_xds, data_group_name=image_data_group_in_name)
        if mask is None
        else mask.sum(axis=(-2, -1))
    )
    # The weights are read only for a plane whose level is its sensitivity
    # (threshold not above 0, as in resolve_max_cycles) and that has a clean
    # mask. Planes are (time, chan, pol), the order of the controller's
    # per-plane state; the sensitivity, (frequency, polarization), goes to
    # every time plane.
    sensitivity = None
    if any(
        not controller._plane_threshold(index[2]) > 0 and mask_pixels[index] > 0
        for index in np.ndindex(mask_pixels.shape)
    ):
        statistics = calculate_weight_statistics_single_field(
            ps_xdt, img_xds, processing_set_data_group_name
        )
        sensitivity = np.broadcast_to(
            statistics["SENSITIVITY"].values, mask_pixels.shape
        )
    return controller.resolve_max_cycles(
        residual_imaging_dict, sensitivity=sensitivity, mask_pixels=mask_pixels
    )
