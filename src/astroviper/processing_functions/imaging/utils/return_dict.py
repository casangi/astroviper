"""
Module to hold the ReturnDict class. The ReturnDict class is a convenience
class around a regular nested dictionary, allowing for more flexible indexing
and seaerching. By keeping it as a dict underneath, it remains generic
and lightweight.

History Tracking (Added 2025-12-10):
-------------------------------------
The ReturnDict.add() method now maintains convergence history by tracking
certain fields as lists (appending values) while keeping others as single
values (replacing). This enables convergence visualization and monitoring
deconvolution progress across major/minor cycles.
"""

from collections import OrderedDict, namedtuple

# Define the key structure
# A namedtuple is used for the keys to ensure immutability and easy comparison
# It can be treated as a regular tuple for indexing and comparisons
Key = namedtuple("Key", ["time", "pol", "chan"])

# Fields that accumulate as lists (per-cycle measurements)
FIELD_ACCUM = {
    "peakres",  # Peak residual value (Jy)
    "peakres_nomask",  # Peak residual without mask (Jy)
    "iter_done",  # Iterations done in this cycle
    "masksum",  # Sum of mask values (valid pixels)
    "model_flux",  # Cumulative model flux (Jy)
    "start_peakres",  # Peak residual at start of each cycle (Jy)
    "start_peakres_nomask",  # Peak residual (no mask) at start of each cycle (Jy)
    "start_model_flux",  # Model flux at start of each cycle (Jy)
}

# Fields that remain single values (constant parameters)
FIELD_SINGLE_VALUE = {
    "max_psf_sidelobe",  # PSF characteristic (doesn't change per cycle)
    "loop_gain",  # CLEAN gain parameter (constant)
    "niter_per_plane",  # Max iterations requested (parameter, not measurement)
    "threshold",  # Threshold used (parameter, not measurement)
}


class ReturnDict:
    def __init__(self):
        self._data = OrderedDict()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def add(self, value, time, pol, chan):
        """
        Add value to ReturnDict with history tracking.

        For fields in FIELD_ACCUM (peakres, iter_done, masksum, peakres_nomask),
        values are appended to lists. For fields in FIELD_SINGLE_VALUE
        (max_psf_sidelobe, loop_gain, niter_per_plane, threshold), values replace previous values.

        Parameters:
        -----------
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

        if key not in self.data:
            # First time seeing this key - initialize
            self.data[key] = {}
            for field, field_value in value.items():
                if field in FIELD_ACCUM:
                    # Initialize as single-element list
                    self.data[key][field] = [field_value]
                else:
                    # Store as single value
                    self.data[key][field] = field_value
        else:
            # Key exists - update with history tracking
            for field, field_value in value.items():
                if field in FIELD_ACCUM:
                    # Append to history
                    if field in self.data[key]:
                        # Handle backward compatibility: convert to list if needed
                        if not isinstance(self.data[key][field], list):
                            self.data[key][field] = [self.data[key][field]]
                        self.data[key][field].append(field_value)
                    else:
                        # First occurrence of this field
                        self.data[key][field] = [field_value]
                else:
                    # Replace single value
                    self.data[key][field] = field_value

    def sel(self, time=None, pol=None, chan=None):
        """Get all items matching the given criteria"""
        matches = []
        for key, value in self.data.items():
            if (
                (time is None or key.time == time)
                and (pol is None or key.pol == pol)
                and (chan is None or key.chan == chan)
            ):
                matches.append(value)

        if len(matches) == 1:
            return matches[0]
        return matches if matches else None

    def to_dict(self):
        return self.data

    def __repr__(self):
        lines = []
        for key, value in self.data.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def return_dict_to_dataframe(return_dict):
    """Flatten a deconvolution :class:`ReturnDict` to one row per plane.

    Used by the benchmark result saving (and any post-run analysis) to persist
    the per-plane convergence record in a tabular, feather-friendly form.

    Parameters
    ----------
    return_dict : ReturnDict
        The merged per-plane deconvolution dict returned by
        ``image_cube_single_field`` (``Key(time, pol, chan)`` ->
        parameters, per-cycle history lists and the stop code).

    Returns
    -------
    pandas.DataFrame
        One row per ``(time, pol, chan)`` plane. Scalar fields become plain
        columns (a ``StopCode`` namedtuple becomes ``stop_code_major`` /
        ``stop_code_minor``); per-cycle history lists (``iter_done``,
        ``peakres``, ...) are kept as list-valued columns AND summarized into
        the derived columns ``n_cycles``, ``iter_total`` (total CLEAN
        iterations over all cycles -- 0 means the plane was never
        deconvolved), ``peakres_start`` / ``peakres_final`` and
        ``model_flux_final``. Empty for an empty dict (e.g. a niter_per_plane=0 run).
    """
    import pandas as pd

    rows = []
    for key, value in return_dict.data.items():
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
        row["n_cycles"] = len(iter_done)
        row["iter_total"] = int(sum(iter_done)) if iter_done else 0
        peakres = value.get("peakres") or []
        start_peakres = value.get("start_peakres") or []
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
