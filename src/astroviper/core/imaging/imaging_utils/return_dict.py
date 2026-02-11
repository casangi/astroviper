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

from collections import namedtuple, OrderedDict

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
    "niter",  # Max iterations requested (parameter, not measurement)
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
        (max_psf_sidelobe, loop_gain, niter, threshold), values replace previous values.

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
