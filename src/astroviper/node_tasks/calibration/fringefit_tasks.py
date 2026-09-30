from scipy.special import iv, ive
from scipy.optimize import root_scalar

# Kring magic: scipy.special.ive is the "exponentially-scaled modified
# Bessel function of the first kind"
def kring_G(gamma):
    t1 = ive(0, gamma) + ive(1, gamma)
    t2 = np.sqrt(np.pi*gamma/2.0)
    return t1*t2

def kring_f(k, lhs):
    gamma = (k**2)/4
    return kring_G(gamma)**2 - lhs

def getFourierSpacings(xds, npad=1):
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

def aips_snr(peak, xcount, sumw_, sumww_):
    x = np.pi / 2 * peak / sumw_
    if x > (0.99 * np.pi / 2):
        snr = 9999
    else:
        # Note that scaling ~ sqrt(xcount) if weights are approximately
        # constant
        scaling = np.sqrt(sumw_ / np.sqrt(sumww_ / xcount)) #
        snr = np.tan(x) ** 1.163 * scaling

    return snr

def kring_snr(peak, xcount, sumw_, sumww_):
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

# FIXME: We should push some of this down to the processing functions module
def _fringe_fft_node_task(input_params: dict):
    ps = input_params["ps"]
    data_selection = input_params["data_selection"]
    ref_ant = input_params["ref_ant"]
    npad = 8  
    # FIXME: for now we do single band
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    xds = xdses[0]
    name = xds.name 
    q = make_empty_cal_quantum(xds)
    q.ref_antenna[0] = ref_ant
    pol_hands = "RL"  # FIXME: we should support data_sub_selection['polarization']
    par_hands = [p+p for p in pol_hands]
    xds2 = xds.sel(polarization=par_hands).isel(**data_selection[name])
    ddelay, drate = getFourierSpacings(xds2, npad)
    ref_baselines = xds2.baseline_id[
        (xds2.baseline_id.baseline_antenna1_name == ref_ant).compute()
        | (xds2.baseline_id.baseline_antenna2_name == ref_ant).compute()
    ]
    vis = xds2.VISIBILITY[:, ref_baselines, :]
    flags = xds2.FLAG[:, ref_baselines, :]
    weights = xds2.WEIGHT[:, ref_baselines, :]

    ang = np.angle(vis)
    normed = np.exp(1j * ang) * weights  # May want a new name...
    normed = np.where(np.isnan(normed), 0, normed)
    normed[flags] = 0
    s = list(normed.shape)
    nt = s[0]
    nf = s[2]
    s2 = (npad * nt, npad * nf)  # Pad in time  # Pad in frequency
    # FIXME: We should be using weights here!
    fftvis = np.fft.fftshift(np.fft.fft2(normed, axes=(0, 2), s=s2), axes=(0, 2))
    for p_ind, pol_hand in enumerate(pol_hands):
        par_pol = pol_hand + pol_hand
        pt = xds.antenna_xds.polarization_type
        receptor_indices = (pt==pol_hand).argmax(dim='receptor_label')
        receptor_labels = (pt==pol_hand).idxmax(dim='receptor_label')
        pol_index = (xds2.polarization==par_pol).argmax(dim='polarization')
        q.CALPARAM_PHASE.loc[
            dict(antenna_name=ref_ant)
        ] = 0
        q.CALPARAM_DELAY.loc[
            dict(antenna_name=ref_ant)
        ] = 0
        q.CALPARAM_RATE.loc[
            dict(antenna_name=ref_ant)
        ] = 0
        q.SNR.loc[dict(antenna_name=ref_ant)] = 2 * [1e9]
        for i, bl in enumerate(ref_baselines):
            if not bl:
                continue
            ant1 = xds2.baseline_antenna1_name.values[bl]
            ant2 = xds2.baseline_antenna2_name.values[bl]
            if ant2 == ref_ant:
                ant = ant1
                sign = +1
            else:
                ant = ant2
                sign = -1
            receptor_ind = receptor_indices.loc[ant].values
            receptor_label = receptor_labels.loc[ant].values
            these_weights = weights[:, i, :, pol_index]
            sumw_ = these_weights.sum(skipna=True)
            sumww_ = (these_weights * these_weights).sum(skipna=True)
            ft = fftvis[:, i, :, pol_index].squeeze()
            a = np.abs(ft)
            # Deduce parameter values from FT
            ind = np.unravel_index(np.argmax(a), a.shape)
            ix, iy = ind
            nx, ny = a.shape
            phi0 = np.angle(ft[ind]) # NB: another sign choice!
            # The dimensions are time *then* frequency, so delay is y; rate is x.
            delay = sign * (iy - ny / 2) * ddelay
            ref_freq = xds2.frequency.reference_frequency["data"]
            phi0 -= (
                2 * np.pi * (delay * (xds2.frequency.values[0] - ref_freq))
            )  # We'll fix the sign convention empirically
            phi0 *= sign
            rate = sign * (ix - nx / 2) * drate / ref_freq
            # Calculate SNR of choice:
            peak = a[ind]
            xcount = (np.logical_not(vis[:, i, :, pol_index].isnull())).sum()
            if False:
                snr = aips_snr(peak, xcount, sumw_, sumww_)
            if True: # Kring SNR
                snr = kring_snr(peak, xcount, sumw_, sumww_)
            # print(f"{name} {bl.values=}, {receptor_label=}")
            # Fill the table.
            q.CALPARAM_PHASE.loc[
                dict(antenna_name=ant, receptor_label=receptor_label)
            ] = phi0
            q.CALPARAM_DELAY.loc[
                dict(antenna_name=ant, receptor_label=receptor_label)
            ] = delay
            q.CALPARAM_RATE.loc[
                dict(antenna_name=ant, receptor_label=receptor_label)
            ] = rate
            q.SNR.loc[
                dict(antenna_name=ant, receptor_label=receptor_label)
            ] = snr
    return q


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


