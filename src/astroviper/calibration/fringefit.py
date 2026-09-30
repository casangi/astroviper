import dask
import numpy as np
import xarray as xa
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
    make_parallel_coord,
)
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from graphviper.graph_tools.map import map
from xradio.measurement_set.processing_set_xdt import ProcessingSetXdt

from astroviper.calibration.fringe_cal_quantum import (
    SingleFringeJones,
    make_empty_cal_quantum,
)

from scipy.special import iv, ive
from scipy.optimize import root_scalar

# Kring magic: scipy.special.ive is the "exponentially-scaled modified Bessel function of the first kind"
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


def _fringe_node_task(input_params: dict):
    ps = input_params["ps"]
    data_selection = input_params["data_selection"]
    ref_ant = input_params["ref_ant"]
    npad = 8  # To see if it's that
    # FIXME: for now we do single band
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    xds = xdses[0]
    name = xds.name # xds.relative_to(xds.parent)
    q = make_empty_cal_quantum(xds)
    q.ref_antenna[0] = ref_ant
    pol_hands = "RL"  # FIXME data_sub_selection['polarization']
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
            # p is meant to be the 
            these_weights = weights[:, i, :, pol_index]
            sumw_ = these_weights.sum(skipna=True)
            sumww_ = (these_weights * these_weights).sum(skipna=True)
            ft = fftvis[:, i, :, pol_index].squeeze()
            a = np.abs(ft)
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
            # Calculate AIPS SNR
            peak = a[ind]
            if False:
                x = np.pi / 2 * peak / sumw_
                if x > (0.99 * np.pi / 2):
                    snr = 9999
                else:
                    xcount = (np.logical_not(flag[:, i, :, pol_index])).sum()
                    # print(f"{peak=:6g} {x=:6g} {xcount=:6g}")
                    # Note that the weird weight stuff boils down to sqrt(xcount),
                    # if weights are approximately constant
                    snr = np.tan(x) ** 1.163 * np.sqrt(sumw_ / np.sqrt(sumww_ / xcount))
            if True: # Kring SNR
                # I just want to get an integer out of this and it's becoming a bit of a quest...
                n = int((np.logical_not(vis[:, i, :, pol_index].isnull())).sum())
                # print(f"{peak=} {n=}")
                if n == 0:
                    snr = 0
                else:
                    ave_w = sumw_/n
                    ave_w_sq = sumww_/n
                    X = peak/n
                    lhs = float((X**2*n - ave_w_sq)/(ave_w**2 * (n-1)))
                    upper_k = 2
                    print(f"{X=} {n=} {lhs=}")
                    while True:
                        if kring_f(upper_k, lhs) > 0:
                            break
                        upper_k *= 2
                    try:
                        res = root_scalar(kring_f, args=(lhs,), bracket=(0,upper_k), method='bisect')
                        snr = np.sqrt(n)*res.root
                    except Exception as e:
                        snr = 0
            # print(f"{name} {bl.values=}, {receptor_label=}")
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



def _fringefit_single(
    ps: ProcessingSetXdt,
    node_task_data_mapping: dict,
    sub_selection: dict,
    ref_ant: str,
):
    """Fringefits a single something. I should probably know what for something."""
    input_params = {}
    input_params["data_sub_selection"] = sub_selection
    input_params["ps"] = ps
    input_params["ref_ant"] = ref_ant
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_fringe_node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    # print(f"{graph=}")
    dask_graph = generate_dask_workflow(graph)
    # Strip the singleton list wrapper at source
    [res] = dask.compute(dask_graph)
    return res


def fringefit_ps(ps: ProcessingSetXdt, ref_ant: str, unixtime: float, interval: float):
    """Fringefits a ProcessingSetXdt for a time slice, handling each spw separately.

    Currently assumes that each there will be one xdt for each spw in the ProcessingSetXdt.
    """
    ps2 = ps.sel(time=slice(unixtime, unixtime + interval))
    # This returns *all* the xdses with that timerange filter. Many of
    # them will now have no entries in the time dimension, but that's
    # the ps.sel api and we deal with it.

    # I have a bad feeling about this...
    xds = ps.xr_ps.get_combined_antenna_xds()
    baselines = xds.baseline_id
    parallel_coords = {}
    parallel_coords["baseline_id"] = make_parallel_coord(
        coord=xds.baseline_id, n_chunks=1
    )
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps2, ps_partition=["spectral_window_name"]
    )
    subsel = {}
    res0 = _fringefit_single(ps2, node_task_data_mapping, subsel, ref_ant)
    # Now we collect these into a single DataTree!
    res_cal_tree = xa.DataTree(
        children={f"part{i}": xa.DataTree(dataset=q) for i, q in enumerate(res0)}
    )
    return res_cal_tree


def apply_cal_ps(
    ps: ProcessingSetXdt, cal_tree: xa.DataTree, unixtime: float, interval: float
):
    """Apply a cal_tree to a time slice of a processing set"""
    ps2 = ps.sel(time=slice(unixtime, unixtime + interval))
    for k, xds in ps2.items():
        cal = [
            q
            for q in cal_tree.values()
            if q.spectral_window_name == xds.frequency.spectral_window_name
        ][0]
        sfj = SingleFringeJones(cal)
        new_xds = sfj.apply_cal(xds)
        ps2[k] = new_xds
    return ps2


def _apply_cal_task(
        input_params: dict):
    ps = input_params["ps"]
    cal_tree = input_params["cal_tree"]
    data_selection = input_params["data_selection"]
    # print(f"{data_selection=} in _apply_cal_task")
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    xds = xdses[0]
    name = xds.name
    # name = xds.relative_to(xds.parent)
    cal = [
            q
            for q in cal_tree.values()
            if q.spectral_window_name == xds.frequency.spectral_window_name
        ][0]
    sfj = SingleFringeJones(cal)
    new_xds = sfj.apply_cal(xds)
    return new_xds

def apply_cal_dask(
    ps: ProcessingSetXdt, cal_tree: xa.DataTree
):
    parallel_coords = {}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps, ps_partition=["spectral_window_name", "scan_name"]
    )
    input_params = {"ps" : ps,
                    "cal_tree" : cal_tree}
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_apply_cal_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    print(f"Made a graph in apply_cal_dask")
    dask_graph = generate_dask_workflow(graph)
    # Strip the singleton list wrapper at source
    [res] = dask.compute(dask_graph)
    # Now we collect these into a single DataTree!
    for xds in res:
        name = xds.name # xds.relative_to(xds.parent)
        ps[name] = xds
    return ps
