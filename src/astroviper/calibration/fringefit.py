from xradio.measurement_set.processing_set_xdt import ProcessingSetXdt
from xradio.measurement_set.open_processing_set import open_processing_set

from graphviper.graph_tools.map import map
from graphviper.graph_tools.reduce import reduce
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
    make_parallel_coord,
    make_time_coord,
    make_frequency_coord,
)
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow

from astroviper.calibration.fringe_cal_quantum import make_empty_cal_quantum

from .fringe_cal_quantum import SingleFringeJones

from typing import Dict, Union
import dask
import numpy as np
import xarray as xa
import pandas as pd
import datetime


def unique(s):
    "Get the unique characters in a string in the order they occur in the original"
    u = []
    for c in s:
        if c not in u:
            u.append(c)
    return u


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


def _fringe_node_task(input_params: Dict):
    ps = input_params["ps"]
    data_selection = input_params["data_selection"]
    print(f"{data_selection=}")
    ref_ant = input_params["ref_ant"]
    npad = 8  # To see if it's that
    # FIXME: for now we do single band
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    xds = xdses[0]
    name = xds.relative_to(xds.parent)
    q = make_empty_cal_quantum(xds)
    q.ref_antenna[0] = ref_ant
    data_sub_selection = input_params["data_sub_selection"]
    allpols = "RL"  # FIXME data_sub_selection['polarization']
    xds2 = xds.isel(**data_selection[name])
    ddelay, drate = getFourierSpacings(xds2, npad)
    baselines = xds2.baseline_id[
        (xds2.baseline_id.baseline_antenna1_name == ref_ant)
        | (xds2.baseline_id.baseline_antenna2_name == ref_ant)
    ]
    vis = xds2.VISIBILITY[:, baselines, :, ::3]
    flags = xds2.FLAG[:, baselines, :, ::3]
    weights = xds2.WEIGHT[:, baselines, :, ::3]

    ang = np.angle(vis)
    normed = np.exp(1j * ang) * weights  # May want a new name...
    normed = np.where(np.isnan(normed), 0, normed)
    normed[flags] = 0
    s = list(normed.shape)
    nt = s[0]
    nf = s[2]
    s2 = (npad * nt, npad * nf)  # Pad in time  # Pad in frequency
    fftvis = np.fft.fftshift(np.fft.fft2(normed, axes=(0, 2), s=s2), axes=(0, 2))
    npols = normed.shape[-1]
    part_info = xds2.xr_ms.get_partition_info()
    spw = part_info["spectral_window_name"]
    t = xds2.time[0].values
    # Sometimes ref_ant has an autocorrelation; I think I'm allowed to
    # ignore it?  But if we set it up front and it gets overwritten by
    # actual data I guess that's fine too?
    for p in range(npols):
        q.CALIBRATION_PARAMETER.loc[
            dict(antenna_name=ref_ant, polarization=allpols[p])
        ] = [0, 0, 0]
        q.SNR.loc[dict(antenna_name=ref_ant, polarization=allpols[p])] = 3 * [1e9]
    for i, bl in enumerate(baselines):
        if not bl:
            continue
        ant1 = xds2.baseline_antenna1_name.values[bl]
        ant2 = xds2.baseline_antenna2_name.values[bl]
        print(f"Doing {ant1}-{ant2}")
        # FIXME: Surely we need to adjust signs too?
        if ant2 == ref_ant:
            ant = ant1
            sign = -1
        else:
            ant = ant2
            sign = +1
        for p in range(npols):
            these_weights = weights[:, i, :, p]
            sumw_ = np.sum(these_weights)
            sumww_ = np.sum(these_weights * these_weights)
            ft = fftvis[:, i, :, p].squeeze()
            a = np.abs(ft)
            ind = np.unravel_index(np.argmax(a), a.shape)
            ix, iy = ind
            nx, ny = a.shape
            phi0 = np.angle(ft[ind])
            # The dimensions are time *then* frequency, so delay is y; rate is x.
            delay = sign * (iy - ny / 2) * ddelay
            ref_freq = xds2.frequency.reference_frequency["data"]
            phi0 -= (
                2 * np.pi * (delay * (xds2.frequency.values[0] - ref_freq))
            )  # We'll fix the sign convention empirically
            phi0 *= sign
            rate = sign * (ix - nx / 2) * drate / ref_freq
            # Calculate SNR
            peak = a[ind]
            x = np.pi / 2 * peak / sumw_
            if x > (0.99 * np.pi / 2):
                snr = 9999
            else:
                xcount = np.sum(np.logical_not(flags[:, i, :, p]))
                print(f"{peak=:6g} {x=:6g} {xcount=:6g}")
                # Note that the weird weight stuff boils down to sqrt(xcount), if weights are approximately constant
                snr = np.tan(x) ** 1.163 * np.sqrt(sumw_ / np.sqrt(sumww_ / xcount))
            q.CALIBRATION_PARAMETER.loc[
                dict(antenna_name=ant, polarization=allpols[p])
            ] = [phi0, delay, rate]
            q.SNR.loc[dict(antenna_name=ant, polarization=allpols[p])] = [snr, snr, snr]
    return q


def _fringefit_single(
    ps: ProcessingSetXdt,
    node_task_data_mapping: Dict,
    sub_selection: Dict,
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
    print(f"{baselines.values=}")
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
    ps: ProcessingSetXdt, caltree: xa.DataTree, unixtime: float, interval: float
):
    """Apply a caltree to a processing set"""
    ps2 = ps.sel(time=slice(unixtime, unixtime + interval))
    for k, xds in ps2.items():
        cal = [
            q
            for q in caltree.values()
            if q.spectral_window_name == xds.frequency.spectral_window_name
        ][0]
        sfj = SingleFringeJones(cal)
        new_xds = sfj.apply_cal(xds)
        ps2[k] = new_xds
    return ps2
