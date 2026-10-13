import numpy as np
import xarray as xa
from xradio.measurement_set.measurement_set_xdt import MeasurementSetXdt


def unique(s):
    "Get the unique characters in a string in the order they occur in the original"
    return list(set(s))


def make_empty_cal_xds(xds: MeasurementSetXdt):
    """
    args: xds, the measurement set used to get the axes we need for our calibration quantum
    """
    pols_ant = unique("".join([c for c in "".join(xds.polarization.values)]))
    coord_dict = {
        "time": [xds.time.values[0]],
        "antenna_name": xds.antenna_xds.antenna_name.values,
        # We have to shift phase to that frequency, I guess,
        # because otherwise it will be wrong.
        "frequency": [xds.frequency.reference_frequency["data"]],
        "receptor_label": xds.antenna_xds.receptor_label,
    }
    quantum_coords = xa.Coordinates(coords=coord_dict)
    q = xa.Dataset(
        {
            "CALPARAM_PHASE": xa.DataArray(coords=quantum_coords),
            "CALPARAM_DELAY": xa.DataArray(coords=quantum_coords),
            "CALPARAM_RATE": xa.DataArray(coords=quantum_coords),
            "SNR": xa.DataArray(coords=quantum_coords),
            "FLAG": xa.DataArray(coords=quantum_coords).astype(bool),
            "ref_antenna": xa.DataArray(coords=(quantum_coords["time"],)).astype(
                np.dtype("<U2")
            ),
        }
    )
    q.attrs["spectral_window_name"] = xds.frequency.spectral_window_name
    return q


def nan_count(j):
    return np.sum(np.isnan(j))

def number_count(j):
    return np.sum(~np.isnan(j))

