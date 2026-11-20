"""
The imaging_dict, the convergence record of the imaging cycles.

Schema
------
An imaging_dict is a plain dictionary that maps ``Key(time, pol, chan)``, the
indices of an image plane, to a dictionary of fields of two kinds:

- a field in ``FIELD_ACCUM`` is a list of per-cycle measurements, to which
  :func:`add_to_imaging_dict` appends;
- a field in ``FIELD_SINGLE_VALUE`` is a single value, which
  :func:`add_to_imaging_dict` replaces.

A field in neither set is kept as a single value. Such fields are
``stop_code`` (a ``StopCode``, or None; ``create_imaging_dict``,
``check_convergence``, the native MT-MFS model update),
``stop_description`` (``check_convergence``, native MT-MFS), ``stokes``,
``frequency`` and ``time`` (``create_imaging_dict``, native MT-MFS),
``min_psf_fraction`` and ``max_psf_fraction`` (``create_imaging_dict``),
and ``scales``, ``hessian_condition_number`` and
``taylor_hessian_condition_number`` (native MT-MFS).
"""

from collections import namedtuple

from astroviper.processing_functions.image_analysis.image_statistics import (
    RESIDUAL_STATISTICS,
)

# Define the key structure
# A namedtuple is used for the keys to ensure immutability and easy comparison
# It can be treated as a regular tuple for indexing and comparisons
Key = namedtuple("Key", ["time", "pol", "chan"])

# The iteration history of the model updates (record_iteration_history): per
# cycle a 1-D array with one value per iteration of that cycle's model update,
# for every deconvolver.
ITERATION_HISTORY_FIELDS = (
    "iteration_peak_residual",  # Signed peak residual in the mask before the iteration
    "iteration_residual_rms",  # RMS of the residual in the mask after the iteration
    "iteration_model_flux",  # Model flux after the iteration (Jy)
    "iteration_component_l",  # Pixel of the component along l
    "iteration_component_m",  # Pixel of the component along m
    "iteration_component_scale",  # Scale of the component in pixels (0: a point)
    "iteration_component_flux",  # Flux of the component (Jy)
)

# The components of each kind a model update subtracted, per plane: points
# and scale components for every deconvolver, the rest for Asp (its residual
# shaped and significance components, its Hogbom runs and what started them).
COMPONENT_COUNT_FIELDS = (
    "n_point_components",
    "n_scale_components",
    "n_residual_components",
    "n_significance_components",
    "n_hogbom_runs",
    "n_switch_no_change",
    "n_switch_point_scales",
    "n_switch_fused",
)

# The sums of the positive and of the negative model pixels after each model
# update.
MODEL_FLUX_SIGN_FIELDS = ("model_flux_positive", "model_flux_negative")

# Seconds of the residual update a cycle starts with and of its model update.
CYCLE_TIMING_FIELDS = ("t_residual_update", "t_model_update")

# Fields that accumulate as lists (per-cycle measurements). The residual
# statistics (RESIDUAL_STATISTICS) are those of the residual each model
# update leaves and of the residual of the closing stop test, and with the
# prefix ``start_`` those of the residual each model update started from.
FIELD_ACCUM = {
    "peakres",  # Peak residual value (Jy)
    "peakres_nomask",  # Peak residual without mask (Jy)
    "iter_done",  # Iterations done in this cycle
    "masksum",  # Sum of mask values (valid pixels)
    "model_flux",  # Cumulative model flux (Jy)
    "start_peakres",  # Peak residual at start of each cycle (Jy)
    "start_peakres_nomask",  # Peak residual (no mask) at start of each cycle (Jy)
    "start_model_flux",  # Model flux at start of each cycle (Jy)
    "entropy",  # Entropy of the residual at the start of each cycle
    "residual_snr",  # Peak / RMS of the residual at the start of each cycle
    "model_update_stop_code",  # Stop code of the deconvolver in each model update
    "threshold_per_cycle",  # Threshold each model update cleaned to (Jy)
    *RESIDUAL_STATISTICS,
    *(f"start_{field}" for field in RESIDUAL_STATISTICS),
    *ITERATION_HISTORY_FIELDS,
    *COMPONENT_COUNT_FIELDS,
    *MODEL_FLUX_SIGN_FIELDS,
    *CYCLE_TIMING_FIELDS,
}

# Fields that remain single values (constant parameters)
FIELD_SINGLE_VALUE = {
    "max_psf_sidelobe",  # PSF characteristic (doesn't change per cycle)
    "gain",  # CLEAN gain parameter (constant)
    "max_iter",  # Max iterations requested (parameter, not measurement)
    "threshold",  # Threshold used (parameter, not measurement)
    "max_cycles_auto",  # Imaging cycles max_cycles="auto" gives the plane
}


def add_to_imaging_dict(imaging_dict, value, time, pol, chan):
    """
    Add value to imaging_dict with history tracking.

    For fields in FIELD_ACCUM (peakres, iter_done, masksum, peakres_nomask),
    values are appended to lists. For fields in FIELD_SINGLE_VALUE
    (max_psf_sidelobe, gain, max_iter, threshold), values replace previous values.

    Parameters:
    -----------
    imaging_dict : dict
        The imaging_dict the value is added to, in place
    value : dict
        Dictionary of field values to add
    time : int
        Time index
    pol : int
        Polarization index
    chan : int
        Channel index
    """
    key = Key(time, pol, chan)

    if key not in imaging_dict:
        # First time seeing this key - initialize
        imaging_dict[key] = {}
        for field, field_value in value.items():
            if field in FIELD_ACCUM:
                # Initialize as single-element list
                imaging_dict[key][field] = [field_value]
            else:
                # Store as single value
                imaging_dict[key][field] = field_value
    else:
        # Key exists - update with history tracking
        for field, field_value in value.items():
            if field in FIELD_ACCUM:
                # Append to history
                if field in imaging_dict[key]:
                    # Handle backward compatibility: convert to list if needed
                    if not isinstance(imaging_dict[key][field], list):
                        imaging_dict[key][field] = [imaging_dict[key][field]]
                    imaging_dict[key][field].append(field_value)
                else:
                    # First occurrence of this field
                    imaging_dict[key][field] = [field_value]
            else:
                # Replace single value
                imaging_dict[key][field] = field_value


def select_from_imaging_dict(imaging_dict, time=None, pol=None, chan=None):
    """Select the fields of the planes that match the given indices.

    Parameters
    ----------
    imaging_dict : dict
        The imaging_dict to select from.
    time, pol, chan : int or None
        Indices to match; None matches every value.

    Returns
    -------
    dict, list of dict or None
        The fields of the one matching plane, a list of the fields of
        several matching planes, or None when no plane matches.
    """
    matches = []
    for key, value in imaging_dict.items():
        if (
            (time is None or key.time == time)
            and (pol is None or key.pol == pol)
            and (chan is None or key.chan == chan)
        ):
            matches.append(value)

    if len(matches) == 1:
        return matches[0]
    return matches if matches else None


def ends_with_stop_test(iter_done, start_peakres):
    """Whether a plane's record ends with the closing stop test of the imaging
    cycle loop: an entry without iterations after one entry per model update
    (``start_peakres``, which only a model update reports)."""
    return len(start_peakres) == len(iter_done) - 1 and iter_done[-1] == 0


def imaging_dict_to_dataframe(imaging_dict):
    """Flatten a deconvolution ``imaging_dict`` to one row per plane.

    Used by the benchmark result saving (and any post-run analysis) to persist
    the per-plane convergence record in a tabular, feather-friendly form.

    Parameters
    ----------
    imaging_dict : dict
        The merged per-plane deconvolution dict returned by
        ``image_cube_single_field`` (``Key(time, pol, chan)`` ->
        parameters, per-cycle history lists and the stop code).

    Returns
    -------
    pandas.DataFrame
        One row per ``(time, pol, chan)`` plane. Scalar fields become plain
        columns (a ``StopCode`` namedtuple becomes ``stop_code_imaging`` /
        ``stop_code_model_update``); per-cycle history lists (``iter_done``,
        ``peakres``, ...) are kept as list-valued columns AND summarized into
        the derived columns ``n_cycles`` (the imaging cycles, i.e. the model
        updates: the entries of ``iter_done`` without the closing stop test
        of the record of the imaging cycle loop, see
        :func:`ends_with_stop_test`), ``iter_total`` (total
        CLEAN iterations over all cycles -- 0 means the plane was never
        deconvolved), ``peakres_start`` / ``peakres_final`` (for that record
        the absolute peak of the written residual inside the clean mask, 0
        for an empty mask; the entries of the model updates hold the signed
        value at the absolute peak, not a number for an empty mask) and
        ``model_flux_final``. The fields only a model update reports
        (``start_peakres``, ``start_model_flux``, ``model_flux``,
        ``model_update_stop_code``, ...) have
        one entry per model update, one fewer than ``iter_done`` and
        ``peakres`` in that record. The iteration history
        (``ITERATION_HISTORY_FIELDS``; only with ``record_iteration_history``)
        is a list of one 1-D array per cycle, which feather stores as a list
        of lists. Empty for an empty dict.
    """
    import pandas as pd

    rows = []
    for key, value in imaging_dict.items():
        row = {"time_index": key.time, "pol": key.pol, "chan": key.chan}
        for field, field_value in value.items():
            if isinstance(field_value, tuple) and hasattr(field_value, "_fields"):
                for name in field_value._fields:
                    row[f"{field}_{name}"] = getattr(field_value, name)
            elif isinstance(field_value, list):
                row[field] = list(field_value)
            else:
                row[field] = field_value
        iter_done = value.get("iter_done") or []
        start_peakres = value.get("start_peakres") or []
        row["n_cycles"] = len(iter_done) - ends_with_stop_test(iter_done, start_peakres)
        row["iter_total"] = int(sum(iter_done)) if iter_done else 0
        peakres = value.get("peakres") or []
        row["peakres_start"] = start_peakres[0] if start_peakres else None
        row["peakres_final"] = peakres[-1] if peakres else None
        model_flux = value.get("model_flux") or []
        row["model_flux_final"] = model_flux[-1] if model_flux else None
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["time_index", "pol", "chan", "n_cycles", "iter_total"]
        )
    return pd.DataFrame(rows).sort_values(
        ["time_index", "pol", "chan"], ignore_index=True
    )


# The columns imaging_dict_to_dataframe adds to the fields of a plane.
_DERIVED_COLUMNS = (
    "time_index",
    "pol",
    "chan",
    "n_cycles",
    "iter_total",
    "peakres_start",
    "peakres_final",
    "model_flux_final",
)


def imaging_dict_from_dataframe(frame):
    """Rebuild a deconvolution ``imaging_dict`` from its table.

    The inverse of :func:`imaging_dict_to_dataframe`: every row becomes the
    fields of the plane ``Key(time_index, pol, chan)``, in row order, set
    directly (not through :func:`add_to_imaging_dict`).

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per plane, as written by :func:`imaging_dict_to_dataframe`
        (for example read back from feather).

    Returns
    -------
    dict
        The derived columns (``n_cycles``, ``iter_total``, ``peakres_start``,
        ``peakres_final``, ``model_flux_final``) and the key columns are left
        out; ``stop_code_imaging`` and ``stop_code_model_update`` become
        ``stop_code``, a ``StopCode``. List, tuple and array cells become
        lists (the per-cycle arrays of the iteration history inside them stay
        arrays). A scalar cell that is None or NaN is left out, as the field
        of a plane that did not record it, so a recorded NaN scalar is lost.
    """
    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.utils.cycle_control import StopCode

    imaging_dict = {}
    for row in frame.to_dict("records"):
        fields = {}
        for column, value in row.items():
            if column in _DERIVED_COLUMNS or column == "stop_code_model_update":
                continue
            if column == "stop_code_imaging":
                model_update = row.get("stop_code_model_update")
                if not (pd.isna(value) or pd.isna(model_update)):
                    fields["stop_code"] = StopCode(
                        imaging=int(value), model_update=int(model_update)
                    )
            elif isinstance(value, list | tuple | np.ndarray):
                fields[column] = list(value)
            elif not pd.isna(value):
                fields[column] = value
        key = Key(int(row["time_index"]), int(row["pol"]), int(row["chan"]))
        imaging_dict[key] = fields
    return imaging_dict


def deconvolution_per_channel(return_dict, time_index=0):
    """The deconvolution record of a cube imaging run as per-channel arrays.

    Places every plane of the record (:func:`imaging_dict_to_dataframe`) at
    the row of its ``frequency`` field among the frequencies of the residual
    statistics, at the row of its channel number for a plane without one,
    and takes the final residual entropy and RMS from those statistics. The
    result holds the array arguments of
    :func:`~astroviper.utils.report_plots.plot_deconvolution_summary`.

    Parameters
    ----------
    return_dict : dict
        The return dict of ``image_cube_single_field``: its ``"deconvolution"``
        record and ``image_statistics["sky_residual"]`` are read.
    time_index : int, optional
        The time plane. Default 0.

    Returns
    -------
    dict or None
        ``frequency`` (Hz, the frequencies of the residual statistics),
        ``stokes`` (their polarization labels) and, per channel
        ``[n_frequency]`` or per plane ``[n_frequency, n_stokes]``:

        - ``imaging_stop_code``: the cycle control stop code of every plane;
        - ``model_update_stop_code_counts``: ``{code: [n_frequency]}``, the
          model updates of each channel (over its Stokes planes) that ended
          with ``code`` (a model update that did not run is not counted);
        - ``iterations``: the total iterations of every plane;
        - ``imaging_cycles``: the imaging cycles of every channel (the most
          over its Stokes planes);
        - ``peakres_start`` and ``peakres_final``: the signed peak residual
          inside the clean mask at the start of the first model update and
          the peak of the final residual (NaN without a model update);
        - ``threshold``: the ``threshold`` field of every plane, NaN where
          the record has none;
        - ``entropy`` and ``residual_rms``: the final residual entropy (NaN
          without one) and RMS of every plane.

        A channel or plane without a record is NaN. ``None`` when the run
        has no residual statistics or no record at ``time_index``.

    Raises
    ------
    ValueError
        If a plane of the record lies outside the channels or the Stokes
        planes of the residual statistics.
    """
    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.utils.cycle_control import (
        MODEL_UPDATE_CONTINUE,
    )

    residual = (return_dict.get("image_statistics") or {}).get("sky_residual")
    deconvolution = return_dict.get("deconvolution")
    if residual is None or deconvolution is None:
        return None
    record = imaging_dict_to_dataframe(deconvolution)
    record = record[record["time_index"] == time_index]
    if record.empty:
        return None
    residual = residual.isel(time=time_index).transpose("frequency", "polarization")
    stokes = [str(label) for label in residual["polarization"].values]
    frequency = np.asarray(residual["frequency"].values, dtype=np.float64)
    shape = (len(frequency), len(stokes))

    def column(name):
        values = record[name] if name in record else pd.Series(np.nan, record.index)
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)

    row_of_frequency = {value: index for index, value in enumerate(frequency)}
    channel = np.array(
        [
            row_of_frequency.get(value, chan)
            for value, chan in zip(column("frequency"), record["chan"], strict=True)
        ],
        dtype=int,
    )
    pol = record["pol"].to_numpy(dtype=int)
    outside = (channel < 0) | (channel >= shape[0]) | (pol < 0) | (pol >= shape[1])
    if outside.any():
        raise ValueError(
            f"deconvolution_per_channel: {int(outside.sum())} planes of the record "
            f"lie outside the {shape[0]} channels and {shape[1]} Stokes planes of "
            "the residual statistics"
        )

    per_plane = {}
    for name, source in (
        ("imaging_stop_code", "stop_code_imaging"),
        ("iterations", "iter_total"),
        ("peakres_start", "peakres_start"),
        ("peakres_final", "peakres_final"),
        ("threshold", "threshold"),
    ):
        values = np.full(shape, np.nan)
        values[channel, pol] = column(source)
        per_plane[name] = values
    imaging_cycles = np.full(shape[0], np.nan)
    np.fmax.at(imaging_cycles, channel, column("n_cycles"))
    recorded = np.zeros(shape[0], dtype=bool)
    recorded[channel] = True
    counts = {}
    if "model_update_stop_code" in record:
        for row, codes in zip(channel, record["model_update_stop_code"], strict=True):
            if not isinstance(codes, list):
                continue
            for code in codes:
                if int(code) != MODEL_UPDATE_CONTINUE:
                    counts.setdefault(int(code), np.zeros(shape[0]))[row] += 1
    for values in counts.values():
        values[~recorded] = np.nan
    return {
        "frequency": frequency,
        "stokes": stokes,
        "imaging_stop_code": per_plane["imaging_stop_code"],
        "model_update_stop_code_counts": dict(sorted(counts.items())),
        "iterations": per_plane["iterations"],
        "imaging_cycles": imaging_cycles,
        "peakres_start": per_plane["peakres_start"],
        "peakres_final": per_plane["peakres_final"],
        "threshold": per_plane["threshold"],
        "entropy": (
            np.asarray(residual["entropy"].values, dtype=np.float64)
            if "entropy" in residual
            else np.full(shape, np.nan)
        ),
        "residual_rms": np.asarray(residual["rms"].values, dtype=np.float64),
    }
