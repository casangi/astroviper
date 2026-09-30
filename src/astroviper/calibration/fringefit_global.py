import numpy as np
import xarray as xr
import dask
from scipy.optimize import minimize

# Types
from collections.abc import Mapping, Iterable
from typing import Any, Union
from xarray.core.datatree import DataTree # As type

# xradio, graphviper and astroviper imports
from xradio.measurement_set.open_processing_set import (
    open_processing_set,
)
from xradio.measurement_set import ProcessingSetXdt 

from graphviper.graph_tools.map import map
from graphviper.graph_tools.reduce import reduce
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
)

from astroviper.calibration.fringe_cal_quantum import make_empty_cal_quantum

from .utils import (unique,
                    get_ps_polarization_coord,
                    make_cal_map
)

def _fringe_global_node_task(input_params: dict):
    """The dask task function for global fringefit

    Currently requires the selected data to be in a single xds, which is bad and should be fixed."""
    ps = input_params["ps"]
    data_selection = input_params["data_selection"]
    ref_ant = input_params["ref_ant"]
    cal_map = input_params["cal_map"]
    options = input_params["options"]
    # FIXME: for now we do single band
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    xds = xdses[0] 
    spw_name = xds.frequency.attrs['spectral_window_name']
    cal = cal_map[spw_name]
    nvis = np.prod(xds.VISIBILITY.shape)
    if nvis==0:
        raise ValueError("No data to process")
    name = xds.name #xds.relative_to(xds.parent)
    # SNR shape? time, antenna_name, frequency, recptor_label 
    good_ants0 = cal.antenna_name[np.any(cal.SNR[0, :, 0, :] > 0, axis=1)]
    good_ants = [str(n) for n in good_ants0.values]
    ns = XDSNewtonSolver(xds, cal, ref_ant=ref_ant, ants=good_ants, options=options)
    new_q = ns.do_newton_solve()
    return new_q


def _fringefit_global_single(
        ps: ProcessingSetXdt,
        node_task_data_mapping: Mapping,
        data_sub_selection: Mapping,
        ref_ant: str,
        cal_tree: DataTree,
        options: dict
):
    """The top level Dask routine for FFT fringefit.

    Currently each xds in a processing set is handled separately by the
    node task function, which is an important shortcoming that needs to
    be addressed soon.
    """
    input_params = {}
    input_params["data_sub_selection"] = data_sub_selection
    input_params["ps"] = ps
    input_params["ref_ant"] = ref_ant
    input_params["cal_map"] = make_cal_map(cal_tree)
    input_params["options"] = options
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_fringe_global_node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    dask_graph = generate_dask_workflow(graph)
    # Strip the singleton list wrapper at source
    # dask.visualize(dask_graph, filename='least-squares.svg', optimize_graph=True)
    [res] = dask.compute(dask_graph)
    return res

def fringefit_global_ps(ps: ProcessingSetXdt,
                        ref_ant: str,
                        unixtime: float,
                        interval: float,
                        cal_tree: DataTree,
                        options: dict):
    """Refines a fringefit calibration using an estimate from the fringefit stage

    This preliminary version handles spectral windows separately and maps
    the cal_tree spws to those in the data. The coupling between the
    fft stage and the global stage shouldn't really work like this and
    I should change it.

    """
    ps2 = ps.sel(time=slice(unixtime, unixtime + interval))
    parallel_coords = {}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps2, ps_partition=["spectral_window_name"]
    )
    subsel = {}
    res0 = _fringefit_global_single(ps2, node_task_data_mapping, subsel, ref_ant, cal_tree, options)
    # Now we collect these into a single DataTree!
    res_cal_tree = xr.DataTree(
        children={f"part{i}": xr.DataTree(dataset=q) for i, q in enumerate(res0)}
    )
    return res_cal_tree

class XDSNewtonSolver():
    """The gritty details of the global stage of the fringefit solver are implemented in this class.

    We minimize an objective function with a scipy routine using the
    scalar objective function itself, its derivative with respect to
    the parameters (the "Jacobian" here, which is a vector here) and
    the second derivatives (the "Hessian" matrix). Note that
    minimization (which we use here) has different terminology from
    least-squares methods. There is a lot of overlap between the two
    approaches -- least-squares methods typically work by minimizing an
    objective function too! -- but they are in different parts of scipy
    and they involve different points of view.

    In least-squares approaches one seeks to fit a *vector* function of
    parameters, and in that context the Jacobian is sometimes defined
    as the matrix of derivatives of the vector by the
    parameters. That's not what's going on here!
    """
    def __init__(self,
                 xds : xr.Dataset,
                 cal_q : xr.Dataset,
                 ref_ant : str ='' ,
                 ants : Union[str, None] = None,
                 receptor_labels : str = '',
                 options : Union[dict, None] = None
                 ):
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
        if receptor_labels == '':
            allpols = ''.join(xds.polarization.values)
            self.pol_hands = unique(allpols)
        else:
            self.pol_hands = receptor_labels
        if options == None:
            self.options = {}
        else:
            self.options = options
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
        sel_dict = {'baseline_id':self.bl_subset}
        
    def setup_for_pol_hand(self, pol_hand):
        par_pol = pol_hand + pol_hand
        sel_dict = {'baseline_id':self.bl_subset,
                    'polarization':par_pol}
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
        """We want the index into the jacobian vector, etc, so this is *excluding* the ref_ant, which itself gets index -1"""
        ant_name = self.xds.baseline_antenna1_name.values[bl]
        ind = self.ant_name_to_subset_array_ind[ant_name] if ant_name != self.ref_ant else -1
        if ind > self.ant_name_to_subset_array_ind[self.ref_ant]:
            ind -= 1 
        return ind
    def get_right_vector_index(self, bl):
        """We want the index into the jacobian vector, etc, so this is *excluding* the ref_ant, which itself gets index -1"""
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
        # cal has shape (n_ant-1, n_param)
        cal = self.add_ref_station(cal)
        # => (n_ant, n_param)
        cal_shaped = np.expand_dims(cal, (0, 2))
        # => (1, n_ant, 1, cal_parameter)
        phi0 = cal_shaped[:, :, :, 0]
        tau  = cal_shaped[:, :, :, 1]
        r    = cal_shaped[:, :, :, 2]
        # Calculate phases: 
        phi = phi0 + (tau*self.rfreqs_shaped + r*self.rtimes_shaped)
        # => (n_time, n_ant, n_freq)
        return phi
    def calc_baseline_phase(self, cal):
        # cal has shape (n_ant-1, n_param)
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
        # Recall that n_param=3 currently
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
        # print(f"{x}")
    def do_newton_solve(self):
        q = make_empty_cal_quantum(self.xds)
        q.SNR.loc[:] = 0 # Initialize
        # We only calculate one hand at a time so far, so the cal matrix,
        # which is used for intermediate results, shouldn't have a pol dimension
        cal = np.zeros((self.num_ants-1, 3), float)
        zero_rates = self.options.get('zerorates', False)
        for pol_ind, pol_hand in enumerate(self.pol_hands):
            print(f"Newton solver: {pol_hand=}, {pol_ind=}, {self.pol_hands=}")
            self.setup_for_pol_hand(pol_hand)
            pt = self.xds.antenna_xds.polarization_type
            receptor_indices = (pt==pol_hand).argmax(dim='receptor_label')
            receptor_labels = (pt==pol_hand).idxmax(dim='receptor_label')
            i = 0
            for n in self.ordered_names:
                if n == self.ref_ant:
                    continue
                else:
                    receptor_ind = receptor_indices.loc[n]
                    receptor_label = receptor_labels.loc[n]
                    # print(f"{pol_hand=}, {pol_ind=}, {str(n)=}, {receptor_ind.values=}, {receptor_label.values=}")
                    cal[i, 0] = self.cal_q['CALPARAM_PHASE'].sel({'antenna_name': n,
                                                                  'receptor_label': receptor_label}).values
                    cal[i, 1] = self.cal_q['CALPARAM_DELAY'].sel({'antenna_name': n,
                                                                  'receptor_label': receptor_label}).values
                    cal[i, 2] = self.cal_q['CALPARAM_RATE'].sel({'antenna_name': n,
                                                                 'receptor_label': receptor_label}).values
                    i += 1
            cal /= np.expand_dims(self.xscale, (0,))
            cal[:, 2] *= self.cal_q.frequency.values[0] # There is only 1 entry but
            # print(f"Cal for {pol_ind=} on the way in is {cal}")
            cal_flat = np.reshape(cal[:,:], (-1,))
            res = minimize(self.calc_residual, cal_flat,
                           method='trust-exact',
                           jac=self.calc_newt_jac,
                           hess=self.calc_newt_hess,
                           tol=1e-14,
                           callback=self.callback,
                           options = {
                               'maxiter' : 10,
                               'gtol' : 1e-3 # gradient size (for BFGS)
                           }
                           )
            print(f"{self.xds.name=} {res.success=} {res.nit=}")
            if res.success:
                snr = self.calc_snr(res)
                cal_out = np.reshape(res.x, (-1, 3))
                cal_out = self.add_ref_station(cal_out)
                # FIXME! Where are the scales?
                cal_out *= np.expand_dims(self.xscale, (0,))
                cal_out[:, 2] /= q.frequency.values[0] # There is only 1 entry so far
                q.ref_antenna[0] = self.ref_ant
                # print(f"Cal for {pol_ind=} on the way out is {cal_out}")
                j = 0
                for i, n in enumerate(self.ordered_names):
                    d = dict(antenna_name=n, receptor_label=receptor_labels.loc[n])
                    q.CALPARAM_PHASE.loc[d] = cal_out[i, 0]
                    q.CALPARAM_DELAY.loc[d] = cal_out[i, 1]
                    if not zero_rates:
                        q.CALPARAM_RATE.loc[d] = cal_out[i, 2]
                    else:
                        q.CALPARAM_RATE.loc[d] = 0
                    if n!=self.ref_ant:
                        q.SNR.loc[d] = snr[j] # We skip the ref antenna by hand here
                        j += 1
                    else:
                        q.SNR.loc[d] = 1e9
            else:
                # FIXME: Solver failed, we have to do something about that?
                pass
        return q
    def calc_snr(self, res):
        # p is the number of parameters
        p = np.prod(res.x.shape)
        cal = np.reshape(res.x, (-1, 3))
        n_groups = cal.shape[0]
        # n is the number of visibilities
        n = np.sum(np.logical_not(np.isnan(self.phase_obs)))
        sumw_ = np.sum(self.weight_obs)
        sumww_ = np.sum(self.weight_obs * self.weight_obs)
        sum_sigma_squared = 2*self.calc_residual(cal)
        H = self.calc_newt_hess(cal)
        H_inv = np.linalg.inv(H)
        H_inv_diag = np.diag(H_inv)
        print(f"{H_inv_diag[::3]=}")
        p1 = sum_sigma_squared/(n-p)/n_groups # Scalar
        p2 = H_inv_diag[0::3] # Vector of just the secular phase terms
        # We have a different scaling factor here from in the single baseline case, which is kinda odd.
        snr = 1/2*np.sqrt(2)/np.sqrt(p1*p2) 
        return snr
