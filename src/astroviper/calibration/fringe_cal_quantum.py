from xradio.measurement_set.processing_set import ProcessingSet

import dask
import numpy as np
import xarray as xa

# I am surprised this is not some kind of standard function, but the
# pandas version scolds me for using it on a string.
#
# This implementation is O(n^2) but if that is ever an issue we are
# doing something else very wrong.
def unique(s):
    "Get the unique characters in a string in the order they occur in the original"
    u = []
    for c in s:
        if c not in u:
            u.append(c)
    return u

def make_empty_cal_quantum(xds):
    # time_c = xa.Coordinates(coords = {'time' : [0.5*(xds.time[0]+xds.time[-1])]})
    pols_ant = unique(''.join([c for c in ''.join(xds.polarization.values)]))
    coord_dict = {
        'time' : [xds.time.values[0]],
        'antenna_name' : xds.antenna_xds.antenna_name.values,
        # We have to shift phase to that frequency, I guess,
        # because otherwise it will be wrong.
        'frequency' : [xds.frequency.reference_frequency['data']],
        'polarization' : pols_ant,
        'cal_parameter' : ['phase', 'delay', 'rate']
    }
    quantum_coords = xa.Coordinates(coords=coord_dict)
    q = xa.Dataset({'CALIBRATION_PARAMETER' : xa.DataArray(coords=quantum_coords),
                    'SNR' : xa.DataArray(coords=quantum_coords),
                    'ref_antenna' : xa.DataArray(coords=(quantum_coords['time'],)).astype(np.dtype('<U2'))
                    }
                   )
    q.attrs['spectral_window_name'] = xds.frequency.spectral_window_name
    return q

def nan_count(j):
    return np.sum(np.isnan(j))

def number_count(j):
    return np.sum(~np.isnan(j))

def square_up_last_dimension(v):
    s = v.shape[:-1] + (2,2)
    v2 = np.reshape(v, s)
    return v2
        
def square_up_last_dimension(v):
    s = v.shape[:-1] + (2,2)
    v2 = np.reshape(v, s)
    return v2

class SingleFringeJones(object):
    def __init__(self, q):
        """
"""
        self.q = q
    def calc_antenna_jones(self, xds):
        # We scrape all the metadata from an xds. Maybe this is wise, maybe not.
        frequency = xds.frequency
        ref_freq = self.q.frequency.values[0]
        # These aren't the right values.
        dt = xds.time.values - self.q.time.values[0]
        df = frequency.values - ref_freq
        # We upscale the dimensions so that things broadcast nicely:
        # We want a matrix of (n_time, n_ant, n_freq, n_pol)
        df_shaped = np.expand_dims(df, (0, 1, 3)) # frequency is dimension 2
        dt_shaped = np.expand_dims(dt, (1, 2, 3))
        cal = self.q.CALIBRATION_PARAMETER
        # (time, antenna_name, frequency, polarization, cal_parameter)
        phi0 = cal[:, :, :, :, 0].values
        tau  = cal[:, :, :, :, 1].values
        r    = cal[:, :, :, :, 2].values
        # These are compatible with the shape (n_time, n_ant, n_freq, n_pol)
        # Calculate phases: 
        phi = (phi0 + 2*np.pi*tau*df_shaped +
               2*np.pi*ref_freq*r*dt_shaped)
        # breakpoint()
        antenna_jones = np.exp(1J*phi, dtype=complex)
        return antenna_jones
    def get_left_jones(self, antenna_jones, left_indices):
        jones_diags_left0 = antenna_jones[:, left_indices, :, :]
        jones_diags_left = np.expand_dims(jones_diags_left0, (-1,)) * np.identity(2)
        return jones_diags_left
    def get_right_jones(self, antenna_jones, right_indices):
        jones_diags_right0 = antenna_jones[:, right_indices, :, :]
        jones_diags_right = np.expand_dims(jones_diags_right0, (-1,)) * np.identity(2)
        # Swap the last two axes only
        jones_diags_right = (np.transpose(jones_diags_right, axes=(0, 1, 2, 4, 3))).conj()
        return jones_diags_right
    def get_ant_to_baseline_maps(self, xds):
        baseline_antenna1_name = xds.baseline_antenna1_name.values
        baseline_antenna2_name = xds.baseline_antenna2_name.values
        antennas = xds.antenna_xds.antenna_name.values
        ant_num_map = {n: i for i, n in enumerate(antennas)} 
        left_indices = [ant_num_map[n] for n in baseline_antenna1_name]
        right_indices = [ant_num_map[n] for n in baseline_antenna2_name]
        return left_indices, right_indices
    def apply_cal(self, xds):
        antenna_jones = self.calc_antenna_jones(xds)
        left_indices, right_indices = self.get_ant_to_baseline_maps(xds)
        v = square_up_last_dimension(xds.VISIBILITY.values)
        pols_ant = self.q.polarization
        # Corrected visibilities:
        left_jones = self.get_left_jones(antenna_jones, left_indices)
        right_jones = self.get_right_jones(antenna_jones, right_indices)
        v2 = left_jones @ v @ right_jones
        # Pack this into a DataArray to match the xds
        s3 = v2.shape[:-2] + (4,)
        v2 = v2.reshape(s3)
        coords={'time' : xds.time,
                'baseline_id': xds.baseline_id,
                'frequency' : xds.frequency,
                'polarization' : xds.polarization}
        da = xa.DataArray( coords=coords, data=v2)
        new_xds = xds.assign({'FRINGEFIT' : da})
        return new_xds
       
def apply_cal_ps(ps, res, unixtime, interval):
    """I suspect this is the wrong API"""
    ps2 = ps.ms_sel(time=slice(unixtime, unixtime+interval)) 
    ps3 = {k : v for k, v in ps2.items() if len(v.time.values) != 0}
    ps4 = ProcessingSet({})
    for k, xds in ps3.items():
        cal = [q for q in res if q.spectral_window_name==xds.frequency.spectral_window_name][0]
        sfj = SingleFringeJones(cal)
        new_xds = sfj.apply_cal(xds)
        ps4[k] = new_xds
    return ps4

    
