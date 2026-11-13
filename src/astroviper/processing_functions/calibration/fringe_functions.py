import xarray as xr
import numpy as np
from typing import Union, Mapping
from xradio.measurement_set.measurement_set_xdt import MeasurementSetXdt
from astroviper.distributed_applications.calibration.utils import power_of_two_grid


def square_up_last_dimension(v):
    s = v.shape[:-1] + (2, 2)
    v2 = np.reshape(v, s)
    return v2


class SingleFringeJones:
    def __init__(self, q):
        """SingleFringeJones has the logic needed to appply a fringefit solution to an xds"""
        self.q = q

    def calc_antenna_jones(self, xds: MeasurementSetXdt):
        # We scrape all the metadata from an xds. Maybe this is wise, maybe not.
        frequency = xds.frequency
        # FIXME: this isn't the vibe
        ref_freq = self.q.frequency.values[0]
        dt = xds.time.values - self.q.time.values[0]
        df = frequency.values - ref_freq
        # We upscale the dimensions so that things broadcast nicely:
        # We want a matrix of (n_time, n_ant, n_freq, n_pol)
        df_shaped = np.expand_dims(df, (0, 1, 3))  # frequency is dimension 2
        dt_shaped = np.expand_dims(dt, (1, 2, 3))

        # (time, antenna_name, frequency, polarization)
        phi0 = self.q.CALPARAM_PHASE.values
        # phi0_shaped = np.expand_dims(phi0, (0, 1, 2, 3))
        tau = self.q.CALPARAM_DELAY.values
        r = self.q.CALPARAM_RATE.values
        # These are compatible with the shape (n_time, n_ant, n_freq, n_receptors)
        # Calculate phases:
        # FIXME this is driving me insane
        phi = -1 * (
            phi0 + 2 * np.pi * tau * df_shaped + 2 * np.pi * ref_freq * r * dt_shaped
        )
        # Shape: (n_time, n_ant, n_freq, n_pol)
        antenna_jones = np.exp(1j * phi, dtype=complex)
        return antenna_jones

    def get_left_jones(self, antenna_jones, left_indices):
        # antenna_jones has shape: (n_time, n_ant, n_freq, n_pol)
        # We map the left antenna in a baseline to the baseline dimension
        jones_diags_left0 = antenna_jones[:, left_indices, :, :]
        # We square up the polarization axis to (2,2) for ease of multiplication
        # (n_time, n_baseline, n_freq, 2, 2)
        jones_diags_left = np.expand_dims(jones_diags_left0, (-1,)) * np.identity(2)
        return jones_diags_left

    def get_right_jones(self, antenna_jones, right_indices):
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
        # Both antenna?_name values are n_baseline long
        baseline_antenna1_name = xds.baseline_antenna1_name.values
        baseline_antenna2_name = xds.baseline_antenna2_name.values
        baseline_id = xds.baseline_id.values
        # We make a map of antenna name to number.
        # Note that we need the names from the cal table but the baseline_id
        # from the xds itself.
        antennas = self.q.antenna_name.values
        ant_num_map = {n: i for i, n in enumerate(antennas)}
        # Now we make n_baseline long list of left indices
        # FIXME: if there's an antenna in the xds that isn't in the cal table
        # this will asplode
        left_indices, right_indices = [], []
        bl_found = []
        unfound1, unfound2 = set(), set()
        for b, n1, n2 in zip(
            baseline_id, baseline_antenna1_name, baseline_antenna2_name
        ):
            a1 = ant_num_map.get(n1, -1)
            a2 = ant_num_map.get(n2, -1)
            if a1 < 0 and n1 not in unfound1:
                unfound1.add(n1)
            elif a2 < 0 and n2 not in unfound2:
                unfound2.add(n1)
            else:
                left_indices.append(a1)
                right_indices.append(a2)
                bl_found.append(b)
        unfound = unfound1 & unfound2
        if unfound:
            print(f"No calibration entries for {', '.join(unfound)}")
        return np.array(bl_found), np.array(left_indices), np.array(right_indices)

    def apply_cal(
        self,
        xds,
        *,
        vis_key_in: str = "VISIBILITY",
        vis_key_out: str = "VISIBILITY_CORRECTED",
    ):
        if len(xds.time.values) == 0:
            return xds
        bl_found, left_indices, right_indices = self.get_ant_to_baseline_maps(xds)
        antenna_jones = self.calc_antenna_jones(xds)
        # v has shape (n_time, n_baselines, n_freq, 2, 2)
        v = square_up_last_dimension(xds[vis_key_in].values)
        # Jones matrices for each visibility for left and right antennas on a baseline
        left_jones = self.get_left_jones(antenna_jones, left_indices)
        right_jones = self.get_right_jones(antenna_jones, right_indices)
        # We have a fairly pleasant form for calibration:
        print(f"{xds.name=} {left_jones.shape=} {v.shape=} {right_jones.shape=}")
        v2 = left_jones @ v[:, bl_found, :, :] @ right_jones
        # Reshape the visibilities back to (n_time, n_baseline, n_freq, 4)
        s3 = v2.shape[:-2] + (4,)
        v2 = v2.reshape(s3)
        coords = {
            "time": xds.time,
            "baseline_id": bl_found,  # Is this allowed?
            "frequency": xds.frequency,
            "polarization": xds.polarization,
        }
        da = xr.DataArray(coords=coords, data=v2)
        new_xds = xds.assign({vis_key_out: da})
        # Now we add the new data group:
        new_xds.attrs["data_groups"][vis_key_out] = {
            "correlated_data": vis_key_out,
            "flag": "FLAG",
            "weight": "WEIGHT",
            "uvw": "UVW",
        }
        return new_xds
