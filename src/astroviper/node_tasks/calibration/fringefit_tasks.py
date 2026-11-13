import numpy as np
from scipy.special import iv, ive
from scipy.optimize import root_scalar
import xarray as xr
from scipy.optimize import minimize


from typing import (Union, Mapping)
from astroviper.distributed_applications.calibration.fringe_cal_xds import make_empty_cal_xds
from astroviper.distributed_applications.calibration.utils import combine_xdses

# Kring magic: scipy.special.ive is the "exponentially-scaled modified
# Bessel function of the first kind"
def kring_G(gamma : float) -> float:
    t1 = ive(0, gamma) + ive(1, gamma)
    t2 = np.sqrt(np.pi*gamma/2.0)
    return t1*t2

def kring_f(k : float, lhs : float) -> float:
    gamma = (k**2)/4
    return kring_G(gamma)**2 - lhs

def getFourierSpacings(xds: xr.Dataset, npad:int=1):
    f = xds.frequency.values
    df = (f[-1] - f[0]) / (len(f) - 1)
    dF = len(f) * df
    ddelay = 1 / dF / npad
    #
    t = xds.time.values
    dt = (t[-1] - t[0]) / (len(t) - 1)
    dT = len(t) * dt
    drate = 1 / dT / npad
    return (ddelay, drate)

def aips_snr(peak : float,
             xcount : int,
             sumw_ : float,
             sumww_ :float) -> float:
    x = np.pi / 2 * peak / sumw_
    if x > (0.99 * np.pi / 2):
        snr = 9999
    else:
        # Note that scaling ~ sqrt(xcount) if weights are approximately
        # constant
        scaling = np.sqrt(sumw_ / np.sqrt(sumww_ / xcount)) #
        snr = np.tan(x) ** 1.163 * scaling
    return snr

def kring_snr(peak : float,
              xcount : int,
              sumw_ : float,
              sumww_: float) -> float:
    n = xcount
    if n == 0:
        snr = 0
    else:
        ave_w = sumw_/n
        ave_w_sq = sumww_/n
        X = peak/n
        lhs = float((X**2*n - ave_w_sq)/(ave_w**2 * (n-1)))
        upper_k = 2
        # print(f"{X=} {n=} {lhs=}")
        while True:
            if kring_f(upper_k, lhs) > 0:
                break
            upper_k *= 2
        try:
            res = root_scalar(kring_f, args=(lhs,), bracket=(0,upper_k), method='bisect')
            snr = np.sqrt(n)*res.root
        except Exception as e:
            snr = 0
    return snr

def get_ref_baselines(xds : xr.Dataset, ref_ant : str) -> list[int]:
    ref_baselines = xds.baseline_id[
        (xds.baseline_id.baseline_antenna1_name == ref_ant).compute()
        | (xds.baseline_id.baseline_antenna2_name == ref_ant).compute()
    ]
    return ref_baselines

def weight_phasors(vis, weights, flags):
    ang = np.angle(vis)
    normed = np.exp(1j * ang) * weights  # May want a new name...
    normed = np.where(np.isnan(normed), 0, normed)
    return normed

def _fringe_fft_node_task(input_params: dict):
    ps = input_params["ps"]
    data_selection = input_params["data_selection"]
    ref_ant = input_params["ref_ant"]
    vis_key = input_params.get('vis_key', 'VISIBILITY')
    print(f"_fringe_fft_node_task: {vis_key=}")
    snr_threshold = input_params.get('snr_threshold', 5.0)
    print(f"{vis_key=}")
    npad = 8  
    # FIXME: for now we do single xds data
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    if len(xdses) < 1: 
        raise RuntimeError("No data in ProcessingSet")
    xds = xdses[0]
    name = xds.name
    q = make_empty_cal_xds(xds)
    pol_hands = "RL"  # FIXME: we should support data_sub_selection['polarization']
    par_hands = [p+p for p in pol_hands]
    xds2 = xds.sel(polarization=par_hands).isel(**data_selection[name])
    ddelay, drate = getFourierSpacings(xds2, npad)
    ref_baselines = get_ref_baselines(xds2, ref_ant)
    vis = xds2[vis_key].sel(baseline_id=ref_baselines)
    if 'uvw_label' in xds2.FLAG.dims:
        flags = xds2.FLAG.sel(baseline_id=ref_baselines).any(dim='uvw_label')
    else:
        flags = xds2.FLAG
    weights = xds2.WEIGHT.sel(baseline_id=ref_baselines)
    summed_w = weights.sum(dim=['time', 'frequency'], skipna=True)
    summed_ww = (weights*weights).sum(dim=['time', 'frequency'], skipna=True)
    xcounts = (np.logical_not(vis.isnull())).sum(dim=['time', 'frequency'])
    normed = weight_phasors(vis, weights, flags)
    s = list(normed.shape)
    nt, nf = s[0], s[2]
    s2 = (npad * nt, npad * nf)  # Pad in time and frequency
    fftvis = np.fft.fftshift(np.fft.fft2(normed, axes=(0, 2), s=s2), axes=(0, 2))
    freqs = xds2.frequency
    # xds.polarization has polarization values along a polarization axis.
    # Your guess is as good as mine why.
    vis_pols = xds2.polarization
    baseline_antenna1_name = xds2.baseline_antenna1_name
    baseline_antenna2_name = xds2.baseline_antenna2_name
    # Everything after this point should work fine for single _or_
    # multiple bands It's just grovelling over the FFT! And a lot of
    # bookkeeping! Which requires quite a lot of metadata! Which is
    # going to suck when we have to aggregate it by hand for the
    # multiband case!
    ref_freq = freqs.reference_frequency["data"]
    for p_ind, pol_hand in enumerate(pol_hands):
        par_pol = pol_hand + pol_hand
        # antenna_xds.polarization_type
        pt = xds2.antenna_xds.polarization_type
        receptor_indices = (pt==pol_hand).argmax(dim='receptor_label')
        receptor_labels = (pt==pol_hand).idxmax(dim='receptor_label')
        pol_index = (vis_pols==par_pol).argmax(dim='polarization')
        # Set ref_ant values:
        loc_dict = dict(antenna_name=ref_ant)
        q.CALPARAM_PHASE.loc[loc_dict] = 0
        q.CALPARAM_DELAY.loc[loc_dict] = 0
        q.CALPARAM_RATE.loc[loc_dict] = 0
        q.SNR.loc[loc_dict] = 2 * [1e9]
        for i, bl in enumerate(ref_baselines):
            if not bl: continue
            ant1 = xds2.baseline_antenna1_name.values[bl]
            ant2 = xds2.baseline_antenna2_name.values[bl]
            if ant1 == ref_ant and ant2 == ref_ant:
                continue
            if ant2 == ref_ant:
                ant, sign = ant1, +1
            else:
                ant, sign = ant2, -1
            receptor_ind = receptor_indices.loc[ant].values
            receptor_label = receptor_labels.loc[ant].values

            sumw_ = summed_w[i, pol_index]
            sumww_ = summed_ww[i, pol_index]
            xcount = xcounts[i, pol_index]
            ft = fftvis[:, i, :, pol_index].squeeze()
            a = np.abs(ft)
            # Deduce parameter values from FT
            ind = np.unravel_index(np.argmax(a), a.shape)
            ix, iy = ind
            nx, ny = a.shape
            phi0 = np.angle(ft[ind]) # NB: another sign choice!
            # The dimensions are time *then* frequency, so delay is y; rate is x.
            delay = sign * (iy - ny / 2) * ddelay
            phi0 -= (
                2 * np.pi * (delay * (freqs.values[0] - ref_freq))
            )  
            phi0 *= sign
            rate = sign * (ix - nx / 2) * drate / ref_freq
            # Calculate SNR of choice:
            peak = a[ind]
            if False: # FIXME!
                snr = aips_snr(peak, xcount, sumw_, sumww_)
            else: # Kring SNR
                snr = kring_snr(peak, xcount, sumw_, sumww_)
            if snr < snr_threshold:
                continue
            else:
                # Fill the table.
                loc_dict = dict(antenna_name=ant, receptor_label=receptor_label)
                q.CALPARAM_PHASE.loc[loc_dict] = phi0
                q.CALPARAM_DELAY.loc[loc_dict] = delay
                q.CALPARAM_RATE.loc[loc_dict] = rate
                q.SNR.loc[loc_dict] = snr
    print("Yeah we're done?")
    return q

def _fringe_fft_mbd_node_task(input_params: dict):
    """
    In the multiband case we just combine the xdses and pass them on to the single xds
    """
    ps = input_params["ps"]
    ## FIXME:
    ## We should be using input_params['data_selection'] generated by the parallelisation machinery
    ## for the individual xdses, but (a) that's a bunch of work and (b) the parallelisation machinery
    ## doesn't currently do what I want anyway, and that's *another* bunch of work to sort out
    combined_xds = combine_xdses(ps)
    combined_xds.name = 'combined'
    input_params["ps"] = dict(combined=combined_xds)
    input_params["data_selection"] = dict(combined={})
    calq = _fringe_fft_node_task(input_params)
    return calq

def _fringe_global_mbd_node_task(input_params: dict):
    """
    We try repeating the trick for the multiband case
    """
    ps = input_params["ps"]
    combined_xds = combine_xdses(ps)
    combined_xds.name = 'combined'
    input_params["ps"] = dict(combined=combined_xds)
    input_params["data_selection"] = dict(combined={})
    # breakpoint()
    calq = _fringe_global_node_task(input_params)
    return calq

def get_cal_for_xds(xds:xr.DataTree, cal_tree:xr.DataTree, spw_map:Union[dict, None]=None):
    if spw_map==None:
        spw_to_use = xds.frequency.spectral_window_name
    else:
        spw_to_use = spw_map.get(xds.frequency.spectral_window_name)
    for q in cal_tree.values():
        if q.spectral_window_name == spw_to_use:
            cal = q
            break
    else:
        raise KeyError(f"{spw_to_use} not found in Cal Tree")
    return cal

def _apply_cal_task(input_params: dict):
    ps = input_params["ps"]
    cal_tree = input_params["cal_tree"]
    data_selection = input_params["data_selection"]
    spw_map = input_params.get('spw_map', {})

    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    if len(xdses) < 1: 
        raise RuntimeError("No data in ProcessingSet")

    xds = xdses[0]
    name = xds.name
    cal = get_cal_for_xds(xds, cal_tree, spw_map)
    sfj = SingleFringeJones(cal)
    new_xds = sfj.apply_cal(xds)
    return new_xds
    
def _fringe_global_node_task(input_params: dict):
    """The dask task function for global fringefit

    Currently requires the selected data to be in a single xds, which is bad and should be fixed."""
    ps = input_params["ps"]
    data_selection = input_params["data_selection"]
    ref_ant = input_params["ref_ant"]
    cal_map = input_params["cal_map"]
    options = input_params["options"]
    vis_key = input_params.get('vis_key', 'VISIBILITY')
    print(f"_fringe_global_node_task: {vis_key=}")

    # FIXME: for now we do single band
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    if len(xdses) < 1: 
        raise RuntimeError("No data in ProcessingSet")
    xds = xdses[0] 
    spw_name = xds.frequency.attrs['spectral_window_name']
    cal = cal_map[spw_name]
    nvis = np.prod(xds.VISIBILITY.shape)
    if nvis==0:
        raise ValueError("No data to process")
    name = xds.name 
    # SNR shape: time, antenna_name, frequency, recptor_label 
    good_ants0 = cal.antenna_name[np.any(cal.SNR[0, :, 0, :] > 0, axis=1)]
    good_ants = [str(n) for n in good_ants0.values]
    ns = XDSNewtonSolver(xds, cal, ref_ant=ref_ant, ants=good_ants,
                         vis_key=vis_key, options=options)
    new_q = ns.do_newton_solve()
    return new_q


class XDSNewtonSolver():
    """The gritty details of the global stage of the fringefit solver
    are implemented in this class.

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
                 vis_key:str='VISIBILITY',
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
            allpols = set(''.join(xds.polarization.values))
            self.pol_hands = allpols
        else:
            self.pol_hands = receptor_labels
        self.vis_key = vis_key
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
        sel_dict = {'baseline_id' : self.bl_subset,
                    'polarization' : par_pol}
        vises = self.xds[self.vis_key].sel(sel_dict)
        self.phase_obs = np.angle(vises.values)
        self.weight_obs = self.xds.WEIGHT.sel(sel_dict).values
        if 'uvw_label' in self.xds.FLAG.dims:
            self.flags_obs = self.xds.FLAG.sel(sel_dict).any(dim='uvw_label')
        else:
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
        q = make_empty_cal_xds(self.xds)
        q.SNR.loc[:] = 0 # Initialize
        # We only calculate one hand at a time so far, so the cal matrix,
        # which is used for intermediate results, shouldn't have a pol dimension
        cal = np.zeros((self.num_ants-1, 3), np.float64)
        zero_rates = self.options.get('zerorates', False)
        for pol_ind, pol_hand in enumerate(self.pol_hands):
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
                    cal[i, 0] = self.cal_q['CALPARAM_PHASE'].sel({'antenna_name': n,
                                                                  'receptor_label': receptor_label}).values
                    cal[i, 1] = self.cal_q['CALPARAM_DELAY'].sel({'antenna_name': n,
                                                                  'receptor_label': receptor_label}).values.squeeze()
                    cal[i, 2] = self.cal_q['CALPARAM_RATE'].sel({'antenna_name': n,
                                                                 'receptor_label': receptor_label}).values
                    i += 1
            # Was this indentation wrong?!
            cal /= np.expand_dims(self.xscale, (0,))
            cal[:, 2] *= self.cal_q.frequency.values[0] # There is only 1 entry but
            cal_flat = np.reshape(cal[:,:], (-1,))
            res = minimize(self.calc_residual, cal_flat,
                           method='trust-exact',
                           jac=self.calc_newt_jac,
                           hess=self.calc_newt_hess,
                           tol=1e-14,
                           callback=self.callback,
                           options = {
                               'maxiter' : 20,
                               'gtol' : 1e-3 # gradient size (for BFGS)
                           }
                           )
            print(f"{self.xds.name=} {res.success=} {res.nit=}")
            if not res.success:
                print(f"Failed from starting point {cal}")
            if res.success:
                snr = self.calc_snr(res)
                cal_out = np.reshape(res.x, (-1, 3))
                cal_out = self.add_ref_station(cal_out)
                # FIXME! Where are the scales?
                cal_out *= np.expand_dims(self.xscale, (0,))
                cal_out[:, 2] /= q.frequency.values[0] # There is only 1 entry so far
                # I remove reference antenna from the cal_set
                # But I am too cowardly to remove it from here entirely in case I get bullied into restoring it
                # q.ref_antenna[0] = self.ref_ant
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
                for i, n in enumerate(self.ordered_names):
                    d = dict(antenna_name=n, receptor_label=receptor_labels.loc[n])
                    q.FLAG.loc[d] = True
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



