from xradio.measurement_set.processing_set_xdt import ProcessingSetXdt
from xradio.measurement_set.measurement_set_xdt import MeasurementSetXdt

import dask
import numpy as np
import xarray as xa
import datetime


from .utils import (
    unique,
    multiply_scalar_in_place
    )

def get_xds_antenna_polarizations(xds):
    # (n_antenna, n_polarization_type)
    pol_types = xds.antenna_xds.polarization_type.values
    if not np.equal(pol_types, pol_types[0]).all:
        raise ValueError("We don't implement distinct polarization types yet")
    return pol_types[0]

def extract_several_fringefit_coord_dict_entries_from_xds(xds: MeasurementSetXdt):
    coord_dict = {
        # Which time should we use? 
        "time": [xds.time[0]],
        "antenna_name": xds.antenna_xds.antenna_name.values,
        "frequency": [xds.frequency.reference_frequency["data"]],
        "receptor_label": xds.antenna_xds.receptor_label.values,
    }
    return coord_dict
    
def make_empty_cal_xds(xds: MeasurementSetXdt):
    """
    args: xds, the measurement set used to get the axes we need for our calibration quantum
    """
    coord_dict = extract_several_fringefit_coord_dict_entries_from_xds(xds)
    quantum_coords = xa.Coordinates(coords=coord_dict)
    quantum_coords['frequency'].attrs = xds.frequency.attrs.copy()
    multiply_scalar_in_place(xds.frequency.shape[0], quantum_coords['frequency'].attrs['channel_width'])
    quantum_coords['time'].attrs = xds.time.attrs.copy()
    q = xa.Dataset(
        {
            "CALPARAM_PHASE" : xa.DataArray(coords=quantum_coords),
            "CALPARAM_DELAY" : xa.DataArray(coords=quantum_coords),
            "CALPARAM_RATE" : xa.DataArray(coords=quantum_coords),
            
            "CALERROR_PHASE": xa.DataArray(coords=quantum_coords),
            "CALERROR_DELAY": xa.DataArray(coords=quantum_coords),
            "CALERROR_RATE": xa.DataArray(coords=quantum_coords),
            "SNR": xa.DataArray(coords=quantum_coords),
            "ref_antenna": xa.DataArray(coords=(quantum_coords["time"],)).astype(
                np.dtype("<U2")
            ),
            "FLAGS": xa.DataArray(coords=quantum_coords).astype(bool),
            ## FIXME: We need to do something about field_name, but I am not 100% what
            # "field_name": xds.field_name # FIXME I think this might still be pretty bad!
        }
    )


    q.attrs["spectral_window_name"] = xds.frequency.spectral_window_name
    q.attrs["type"] = "Fringefit CalXds"
    q.attrs["type_version"] = "0.0"
    q.attrs["schema_version"] = "0.0"
    q.attrs["creator"] = {'software_name': 'astroviper.fringefit',
                          'version': '0.0'}
    q.attrs["creation_date"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return q


def nan_count(j):
    return np.sum(np.isnan(j))


def number_count(j):
    return np.sum(~np.isnan(j))


def square_up_last_dimension(v):
    s = v.shape[:-1] + (2, 2)
    v2 = np.reshape(v, s)
    return v2


def square_up_last_dimension(v):
    s = v.shape[:-1] + (2, 2)
    v2 = np.reshape(v, s)
    return v2

class SingleFringeJones(object):
    def __init__(self, corr_xds):
        """A class to handle applying the fringefit corrections

        This requires calculating left and right Jones matrices (or
        tensors with Jones matrix indices among others); once we have
        these we can calculate new visibilities as

           v_corr = left_jones @ v @ right_jones

        The version implemented here doesn't handle interpolation in
        time, however.
        """

        self.corr_xds = corr_xds

    def calc_antenna_jones(self, xds: MeasurementSetXdt):
        """Calculate the Jones matrices each antenna

        We calculate matrices for all the times and frequencies in xds for each antenna""" 
        frequency = xds.frequency

        ref_freq = self.corr_xds.frequency[0]
        dt = xds.time - self.corr_xds.time[0]
        df = frequency - ref_freq
        # We upscale the dimensions so that things broadcast nicely:
        # We want a matrix of (n_time, n_ant, n_freq, n_pol)
        coords = xa.Coordinates({'frequency' : xds.frequency,
                                 'time' : xds.time,
                                 })
        
        # (time, antenna_name, frequency, polarization, cal_parameter)
        self.phi0 = self.corr_xds.CALPARAM_PHASE.interp(coords, method='nearest', kwargs={'fill_value':'extrapolate'})
        self.tau = self.corr_xds.CALPARAM_DELAY.interp(coords, method='nearest', kwargs={'fill_value':'extrapolate'})
        self.r = self.corr_xds.CALPARAM_RATE.interp(coords,  method='nearest', kwargs={'fill_value':'extrapolate'})

        # Calculate phases:
        self.phi = self.phi0 + 2 * np.pi * self.tau * df + 2*np.pi * ref_freq * self.r * dt
        self.phi *= -1 # There's always a minus sign somewhere, and this one is here because it makes things better.
        # Shape: (n_time, n_ant, n_freq, n_pol)
        antenna_jones = np.exp(1j * self.phi, dtype=complex)
        return antenna_jones

    def get_left_jones(self, antenna_jones, left_indices):
        """
        Turn the antenna-based Jones matrices into baseline-indexed matrices

        We use the first antenna in the baseline here; see
        get_right_jones for the second antenna.
        """
        # antenna_jones has shape: (n_time, n_ant, n_freq, n_pol)
        # We map the left antenna in a baseline to the baseline dimension
        jones_diags_left0 = antenna_jones[:, left_indices, :, :]
        # We square up the polarization axis to (2,2) for ease of multiplication
        # (n_time, n_baseline, n_freq, 2, 2)
        jones_diags_left = np.expand_dims(jones_diags_left0, (-1,)) * np.identity(2)
        return jones_diags_left

    def get_right_jones(self, antenna_jones, right_indices):
        """
        Turn the antenna-based Jones matrices into baseline-indexed matrices

        We use the second antenna in the baseline here; see
        get_left_jones for the first antenna.
        """
        # antenna_jones has shape: (n_time, n_ant, n_freq, n_pol)
        # We map the left antenna in each baseline to the baseline dimension
        jones_diags_right0 = antenna_jones[:, right_indices, :, :]
        jones_diags_right = np.expand_dims(jones_diags_right0, (-1,)) * np.identity(2)
        # We square up the polarization axis to (2,2) for ease of multiplication
        # (n_time, n_baseline, n_freq, 2, 2)
        # and take the hermitian conjugate over the last two axes:
        jones_diags_right = (
            np.transpose(jones_diags_right, axes=(0, 1, 2, 4, 3))
        ).conj()
        return jones_diags_right

    def get_ant_to_baseline_maps(self, xds):
        """
        Get indices of antenna1 and antenna2 for each baseline

        The only reason this deserves its own method is that antenna1
        and antenna2 values are stored as names, which is very nice in
        and of itself but here we need to do matrix arithmetic and
        numerical indices are handier for that

        """
        # Both antenna?_name values are n_baseline long
        baseline_antenna1_name = xds.baseline_antenna1_name.values
        baseline_antenna2_name = xds.baseline_antenna2_name.values
        # We make a map of antenna name to number:
        antennas = xds.antenna_xds.antenna_name.values
        ant_num_map = {n: i for i, n in enumerate(antennas)}
        # Now we make n_baseline long list of left indices
        left_indices = [ant_num_map[n] for n in baseline_antenna1_name]
        # And the same for right indices
        right_indices = [ant_num_map[n] for n in baseline_antenna2_name]
        return left_indices, right_indices
    def apply_cal_vis(self, xds):
        antenna_jones = self.calc_antenna_jones(xds)
        left_indices, right_indices = self.get_ant_to_baseline_maps(xds)
        # v has shape (n_time, n_baseline, n_freq, 2, 2)
        v = square_up_last_dimension(xds.VISIBILITY.values)
        # Jones matrices for each visibility for left and right antennas on a baseline
        left_jones = self.get_left_jones(antenna_jones, left_indices)
        right_jones = self.get_right_jones(antenna_jones, right_indices)
        # We have a very concise form for calibration:
        v2 = left_jones @ v @ right_jones
        return v2
    
    def apply_cal(self, xds, group_name='calibrated'):
        """
        Apply the calibration from our CorrXDS to a single xds
        """
        if len(xds.time.values) == 0:
            return xds
        v2 = self.apply_cal_vis(xds)
        # Reshape the visibilities back to (n_time, n_baseline, n_freq, 4)
        s3 = v2.shape[:-2] + (4,)
        v2 = v2.reshape(s3)
        coords = {
            "time": xds.time,
            "baseline_id": xds.baseline_id,
            "frequency": xds.frequency,
            "polarization": xds.polarization,
        }
        da = xa.DataArray(coords=coords, data=v2)
        new_xds = xds.assign({"VISIBILITY_CORRECTED": da})
        # Now we add the new data group:
        new_xds.attrs["data_groups"][group_name] = {
            "correlated_data": "VISIBILITY_CORRECTED",
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }
        self.new_xds = new_xds
        self.da = da
        return new_xds
