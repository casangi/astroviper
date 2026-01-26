import numpy as np
import xarray as xr
import scipy
from scipy.optimize import minimize

from xradio.measurement_set.open_processing_set import open_processing_set
from astroviper.calibration.fringe_cal_quantum import make_empty_cal_quantum

import matplotlib.pyplot as plt


class XDSNewtonSolver():
    def __init__(self, xds, cal_q ,ref_ant='', ants=None, polarization=''): # xds : MeasurementSetXdt, cal_q: CalQuantum
        self.xds = xds # Just in case
        self.cal_q = cal_q
        self.ref_ant = ref_ant
        if ants == None:
            self.ants = set(xds.antenna_xds.antenna_name.values)
        else:
            self.ants = ants
        if self.ref_ant not in self.ants:
            raise ValueError(f"Ant {self.ref_ant} not in {self.ants}")
        self.num_ants = len(self.ants)
        self.polarization = polarization
        self.ordered_names = [name for name in xds.antenna_xds.antenna_name.values if name in self.ants]
        # The indices here are the ones we will use as the basis for vectors and arrays in this class
        self.ant_name_to_subset_array_ind = {name: i for i, name in enumerate(self.ordered_names)}
        self.bl_subset = [bl for (bl, (a1, a2)) in enumerate(zip(self.xds.baseline_antenna1_name.values,
                                                                 self.xds.baseline_antenna2_name.values))
                          if a1 in self.ants and a2 in self.ants and a1 != a2]
        # Many of the minimization routines seem to be bad at scaling,
        # so we take matters into our own hands
        self.xscale = np.zeros((3,), float)
        self.xscale[0] = 1 # phase
        self.xscale[1] = 1e-9 # delay 
        self.xscale[2] = 1e-3 # rate
        f_ref = self.xds.frequency.values[0]
        t_ref = self.xds.time.values[0]
        rtimes = 2*np.pi*(self.xds.time.values - t_ref)
        rfreqs = 2*np.pi*(self.xds.frequency.values - f_ref)
        # We upscale the dimensions so that things broadcast nicely:
        # We want a matrix of (n_time, n_ant, n_freq, n_pol)
        self.rfreqs_shaped = self.xscale[1]*np.expand_dims(rfreqs, (0, 1)) # => (1, 1, n_freq)
        self.rtimes_shaped = self.xscale[2]*np.expand_dims(rtimes, (1, 2)) # => (n_times, 1, 1)
        # FIXME: How do we handle pol_vals? 
        sel_dict = {'baseline_id':self.bl_subset,
                    'polarization':polarization}
        vises = self.xds.VISIBILITY.sel(sel_dict)
        self.phase_obs = np.angle(vises.values)
        self.weight_obs = self.xds.WEIGHT.sel(sel_dict).values
        self.flags_obs = self.xds.FLAG.sel(sel_dict).values
        # Apply flags:
        self.phase_obs[self.flags_obs] = np.nan
        # We make a map of antenna_name to number:
        baseline_antenna1_name = self.xds.baseline_antenna1_name.values[self.bl_subset]
        baseline_antenna2_name = self.xds.baseline_antenna2_name.values[self.bl_subset]
        # Now we make n_baseline long list of left indices 
        self.left_indices = [self.ant_name_to_subset_array_ind[n] for n in baseline_antenna1_name]
        # And the same for right indices
        self.right_indices = [self.ant_name_to_subset_array_ind[n] for n in baseline_antenna2_name]
    def get_left_vector_index(self, bl):
        """We want the index into the jacobian array, etc, so this is *excluding* the ref_ant, which itself gets index -1"""
        ant_name = self.xds.baseline_antenna1_name.values[bl]
        ind = self.ant_name_to_subset_array_ind[ant_name] if ant_name != self.ref_ant else -1
        if ind > self.ant_name_to_subset_array_ind[self.ref_ant]:
            ind -= 1 
        return ind
    def get_right_vector_index(self, bl):
        """We want the index into the jacobian array, etc, so this is *excluding* the ref_ant, which itself gets index -1"""
        ant_name = self.xds.baseline_antenna2_name.values[bl]
        ind = self.ant_name_to_subset_array_ind[ant_name] if ant_name != self.ref_ant else -1
        if ind > self.ant_name_to_subset_array_ind[self.ref_ant]:
            ind -= 1 
        return ind
    def add_ref_station(self, cal):
        # cal should have shape of ant_num, cal_param
        sh = list(cal.shape)
        sh[0] += 1
        new_cal = np.zeros(sh, float)
        ref_station_ind = self.ant_name_to_subset_array_ind[self.ref_ant]
        # breakpoint()
        new_cal[:ref_station_ind, :] = cal[:ref_station_ind, : ]
        new_cal[ref_station_ind+1:, :] = cal[ref_station_ind:, :]
        return new_cal
    def calc_antenna_phase(self, cal):
        # cal has shape (n_ant-1, 3)
        cal = self.add_ref_station(cal)
        # => (n_ant, n_param)
        cal_shaped = np.expand_dims(cal, (0, 2))
        # (1, n_ant, 1, cal_parameter)
        phi0 = cal_shaped[:, :, :, 0]
        tau  = cal_shaped[:, :, :, 1]
        r    = cal_shaped[:, :, :, 2]
        # Calculate phases: 
        phi = phi0 + (tau*self.rfreqs_shaped + r*self.rtimes_shaped)
        # => (n_time, n_ant, n_freq)
        return phi
    def calc_baseline_phase(self, cal):
        # cal has shape (n_ant-1, 3)
        antenna_phase = self.calc_antenna_phase(cal)
        # antenna_phase: [n_time, n_ant, n_freq]
        phase_left = antenna_phase[:, self.left_indices, :]
        phase_right = antenna_phase[:, self.right_indices, :]
        # phase: [n_time, n_baseline, n_freq]
        phase = phase_left - phase_right
        return phase
    def delta_ang(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi
    def calc_residual(self, cal):
        # cal has shape (3*(n_ant-1),)
        cal = np.reshape(cal, (-1, 3))
        # phase_model: (n_time, baseline, n_freq)
        phase_model = self.calc_baseline_phase(cal)
        delta_ang = self.delta_ang(phase_model - self.phase_obs)
        sins = np.nan_to_num(2*np.sin(delta_ang/2), 0)
        # This, we sum over all baselines
        return 0.5*np.sum(sins*sins)
    def calc_newt_jac(self, cal):
        # cal has shape (n_ant-1, 3)
        cal = np.reshape(cal, (-1, 3))
        phase_model = self.calc_baseline_phase(cal)
        (_, n_baseline, _) = phase_model.shape
        delta_ang = self.delta_ang(phase_model - self.phase_obs)
        sins = np.nan_to_num(np.sin(delta_ang), 0)
        res = np.zeros((3*(self.num_ants-1)), float)
        res_subs = np.zeros((3, n_baseline), float)
        res_subs[0] = np.sum(sins*1, axis=(0,2))
        res_subs[1] = np.sum(sins*self.rfreqs_shaped, axis=(0,2))
        res_subs[2] = np.sum(sins*self.rtimes_shaped, axis=(0,2))
        for i_bl, bl in enumerate(self.bl_subset):
            # These are the indices in jacobian vector
            i_l = self.get_left_vector_index(bl) # -1 for ref_ant
            i_r = self.get_right_vector_index(bl) # -1 for ref_ant
            if i_l >= 0:
                for i in range(3):
                    res[3*i_l + i] += res_subs[i, i_bl]
            if i_r >= 0:
                for i in range(3):
                    res[3*i_r + i] -= res_subs[i, i_bl]
        return res
    def calc_newt_hess(self, cal):
        # cal has shape (n_ant-1, 3)
        cal = np.reshape(cal, (-1, 3))
        res = np.zeros((3*(self.num_ants-1), 3*(self.num_ants-1)), float)
        phase_model = self.calc_baseline_phase(cal)
        (_, n_baseline, _) = phase_model.shape
        delta_ang = self.delta_ang(phase_model - self.phase_obs)
        coses = np.nan_to_num(np.cos(delta_ang), 0)
        res_subs = np.zeros((3, 3, n_baseline), float)
        res_subs[0, 0] = np.sum(coses*1*1, axis=(0,2))
        res_subs[0, 1] = np.sum(coses*1*self.rfreqs_shaped, axis=(0,2))
        res_subs[0, 2] = np.sum(coses*self.rtimes_shaped, axis=(0,2))
        res_subs[1, 1] = np.sum(coses*self.rfreqs_shaped*self.rfreqs_shaped, axis=(0,2))
        res_subs[1, 2] = np.sum(coses*self.rfreqs_shaped*self.rtimes_shaped, axis=(0,2))
        res_subs[2, 2] = np.sum(coses*self.rtimes_shaped*self.rtimes_shaped, axis=(0,2))
        res_subs[1, 0] = res_subs[0, 1]
        res_subs[2, 0] = res_subs[0, 2]
        res_subs[2, 1] = res_subs[1, 2]
        for i_bl, bl in enumerate(self.bl_subset):
            i_l = self.get_left_vector_index(bl)
            i_r = self.get_right_vector_index(bl)
            for i in range(3):
                for j in range(3):
                    if i_l>=0:
                        res[3*i_l+i, 3*i_l+j] += res_subs[i, j, i_bl]
                    if i_r >= 0:
                        res[3*i_r+i, 3*i_r+j] += res_subs[i, j, i_bl]
                    if (i_l >= 0) and (i_r >= 0):
                        res[3*i_r+i, 3*i_l+j] -= res_subs[i, j, i_bl]
                        res[3*i_l+i, 3*i_r+j] -= res_subs[i, j, i_bl]
        return res
    def callback(self, res):
        x = np.reshape(res, (-1, 3))
        print(f"{x}")
    def do_newton_solve(self):
        # We are recording antenna calibration parameters, based on baseline data.
        # So we need to map from e.g. 'RR' to just 'R'.
        # Note: Cross-pol data will break horribly!
        pol = self.polarization[0] 
        # We make a cal_quantum I guess
        cal = np.zeros((self.num_ants-1, 3), float)
        i = 0
        for n in self.ordered_names:
            if n == self.ref_ant:
                continue
            else:
                # FIXME: I have a strong vibe that there is a minus sign required somewhere!
                cal[i, :] = self.cal_q['CALIBRATION_PARAMETER'].sel({'antenna_name': n,
                                                                     'polarization': pol}).values
                i += 1
        cal /= self.xscale
        cal[:, 2] *= self.cal_q.frequency.values[0] # There is only 1 entry but
        print(f"Cal on the way in is {cal}")
        cal_flat = np.reshape(cal, (-1,))
        res = minimize(self.calc_residual, cal_flat,
                       method='trust-exact',
                       #method="BFGS",
                       #method="trust-ncg",
                       jac=self.calc_newt_jac,
                       hess=self.calc_newt_hess,
                       tol=1e-14,
                       callback=self.callback,
                       options = {
                           'maxiter' : 10,
                           'gtol' : 1e-3 # gradient size (for BFGS)
                       }
                       )
        self.res = res
        # FIXME! SNR calculation is currently slightly broken
        # snr = self.calc_snr(res)
        snr = 1e3
        # Report:
        cal_out = np.reshape(res.x, (-1, 3))
        cal_out = self.add_ref_station(cal_out)
        # FIXME! Where are the scales?
        cal_out *= self.xscale
        q = make_empty_cal_quantum(self.xds)
        cal_out[:, 2] /= q.frequency.values[0] # There is only 1 entry but
        q.ref_antenna[0] = self.ref_ant
        for i, n in enumerate(self.ordered_names):
            q.CALIBRATION_PARAMETER.loc[
                dict(antenna_name=n, polarization=pol)
            ] = cal_out[i, :]
            q.SNR.loc[
                dict(antenna_name=n, polarization=pol)
            ] = 3 * [snr]
        return q
    def calc_snr(self, res):
        # SNR stuff needs to be adapted still
        p = np.prod(res.x.shape)
        cal = np.reshape(res.x, (-1, 3))
        p_plane = self.calc_baseline_phase(cal)
        n = np.sum(np.logical_not(np.isnan(p_plane)))
        sumw_ = np.sum(self.weight_obs)
        sumww_ = np.sum(self.weight_obs * self.weights_obs)
        sum_sigma_squared = 2*self.newt_fun(cal)
        H = self.newt_hess(cal)
        H_inv = np.linalg.inv(H)
        p1 = sum_sigma_squared/(n-p)
        p2 = H_inv[2,2]
        # FIXME: There may be some unresolved issues here
        # snr = np.sqrt(n/4/np.pi)*np.sqrt(p2/p1)
        snr = 2*1/np.sqrt(p1*p2) # Doubling to see if it's that
        return snr

