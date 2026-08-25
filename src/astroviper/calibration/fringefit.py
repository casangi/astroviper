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

from astroviper.calibration.fringe_cal_quantum import (
    make_empty_cal_xds,
    get_xds_antenna_polarizations
)

from .fringe_cal_quantum import SingleFringeJones

from typing import Dict, Union 
import dask
import numpy as np
import xarray as xa
import pandas as pd
import datetime


def getFourierSpacings(xds, npad=1):
    f = xds.frequency.values
    df = (f[-1] - f[0]) / (len(f) - 1)
    dF = len(f) * df
    ddelay = -1 / dF / npad
    t = xds.time.values
    dt = (t[-1] - t[0]) / (len(t) - 1)
    dT = len(t) * dt
    drate = -1 / dT / npad
    return (ddelay, drate)


def _fringe_fft_node_task(input_params: Dict):
    ps = input_params["ps"]
    data_selection = input_params["data_selection"]
    ref_ant = input_params["ref_ant"]
    npad = 8  
    # FIXME: for now we do single band
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    if len(xdses) > 1:
        raise RuntimeError("We only do single xdses so far")
    xds = xdses[0]
    nvis = np.prod(xds.VISIBILITY.shape)
    if nvis==0:
        raise ValueError("No data to process")
    name = xds.relative_to(xds.parent)
    q = make_empty_cal_xds(xds)
    q.ref_antenna[0] = ref_ant
    data_sub_selection = input_params["data_sub_selection"]
    all_antenna_pols = get_xds_antenna_polarizations(xds)
    selected_pols = data_sub_selection['polarization']
    all_receptors = q.receptor_label.values
    xds2 = xds.isel(**data_selection[name])
    ddelay, drate = getFourierSpacings(xds2, npad)
    baselines = xds2.baseline_id[
        (xds2.baseline_id.baseline_antenna1_name == ref_ant).compute()
        | (xds2.baseline_id.baseline_antenna2_name == ref_ant).compute()
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
    antenna_pols = get_xds_antenna_polarizations(xds)
    npols = normed.shape[-1]
    assert npols == len(antenna_pols)
    part_info = xds2.xr_ms.get_partition_info()
    spw = part_info["spectral_window_name"]
    t = xds2.time[0].values
    # Sometimes ref_ant has an autocorrelation; I think I'm allowed to
    # ignore it.  But if we set it up front and it gets overwritten by
    # actual data I guess that's fine too.
    for p in range(npols):
        # print(f"{all_antenna_pols[p]} and {selected_pols} and {all_receptors[p]}")
        if all_antenna_pols[p] not in selected_pols:
            continue
        q.CALPARAM_PHASE.loc[
            dict(antenna_name=ref_ant, receptor_label=all_receptors[p])
        ] = 0
        q.CALPARAM_DELAY.loc[
            dict(antenna_name=ref_ant, receptor_label=all_receptors[p])
        ] = 0
        q.CALPARAM_RATE.loc[
            dict(antenna_name=ref_ant, receptor_label=all_receptors[p])
        ] = 0
        q.SNR.loc[dict(antenna_name=ref_ant, receptor_label=all_receptors[p])] = [1e9]
    for i, bl in enumerate(baselines):
        if not bl:
            continue
        ant1 = xds2.baseline_antenna1_name.values[bl]
        ant2 = xds2.baseline_antenna2_name.values[bl]
        if ant2 == ref_ant:
            ant = ant1
            sign = -1
        else:
            ant = ant2
            sign = +1
        for p in range(npols):
            if all_antenna_pols[p] not in selected_pols:
                continue
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
                if xcount == 0:
                    snr = 0
                else:
                    # print(f"{peak=:6g} {x=:6g} {xcount=:6g}")
                    # Note that the weird weight stuff boils down to sqrt(xcount),
                    # if weights are approximately constant
                    snr = np.tan(x) ** 1.163 * np.sqrt(sumw_ / np.sqrt(sumww_ / xcount))
            # print(f"{ant=}, {all_receptors[p]=}, {phi0=}")
            q.CALPARAM_PHASE.loc[
                dict(antenna_name=ant, receptor_label=all_receptors[p])
            ] = phi0
            q.CALPARAM_DELAY.loc[
                dict(antenna_name=ant, receptor_label=all_receptors[p])
            ] = delay
            q.CALPARAM_RATE.loc[
                dict(antenna_name=ant, receptor_label=all_receptors[p])
            ] = rate
            q.SNR.loc[dict(antenna_name=ant, receptor_label=all_receptors[p])] = snr
    return q


def _fringefit_fft_single(
    ps: ProcessingSetXdt,
    node_task_data_mapping: Dict,
    data_sub_selection: Dict,
    ref_ant: str,
):
    """Maps a dask task for FFT fringefit over the data mapping supplied.
    """
    input_params = {}
    input_params["data_sub_selection"] = data_sub_selection
    input_params["ps"] = ps
    input_params["ref_ant"] = ref_ant
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_fringe_fft_node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    dask_graph = generate_dask_workflow(graph)
    # dask.visualize(dask_graph, filename='fft.svg', optimize_graph=True)

    # Strip the singleton list wrapper at source
    [res] = dask.compute(dask_graph)
    return res

def fringefit_ps(ps: ProcessingSetXdt, ref_ant: str, unixtime: float, interval: float, polarization: str):
    """Top level fringefit of a processing set"""
    fft_caltree = fringefit_fft_ps(ps, ref_ant, unixtime, interval, polarization)
    return fft_caltree

def fringefit_fft_ps(ps: ProcessingSetXdt, ref_ant: str, unixtime: float, interval: float, polarization: str):
    """Fringefits a ProcessingSetXdt for a time slice with the FFT method, handling each spw separately.
    """
    ps2 = ps.sel(time=slice(unixtime, unixtime + interval))
    # This returns *all* the xdses with that timerange filter. Many of
    # them will now have no entries in the time dimension, but that's
    # the ps.sel api and we deal with it.
    xds = ps.xr_ps.get_combined_antenna_xds()
    baselines = xds.baseline_id
    parallel_coords = {}
    parallel_coords["baseline_id"] = make_parallel_coord(
        coord=xds.baseline_id, n_chunks=1
    )
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps2, ps_partition=["spectral_window_name"]
    )
    
    subsel = {'polarization' : polarization}
    res0 = _fringefit_fft_single(ps2, node_task_data_mapping, subsel, ref_ant)
    # Now we collect these into a single DataTree!
    res_cal_tree = xa.DataTree(
        children={f"part{i}": xa.DataTree(dataset=q) for i, q in enumerate(res0)}
    )
    return res_cal_tree

def apply_cal_ps(
    ps: ProcessingSetXdt, caltree: xa.DataTree, unixtime: float, interval: float
):
    """Apply a caltree to a processing set
    
The current implementation is a proof-of-concept stub that
    * assumes we are handling an interval
    * matches spectral windows in calibration table and data.
    
In the future, applycal will be part of a larger calibration framework
that works much more generally.
    """
    ps2 = ps.sel(time=slice(unixtime, unixtime + interval))
    for k, xds in ps2.items():
        cal = [
            q
            for q in caltree.values()
            if q.spectral_window_name == xds.frequency.spectral_window_name
        ][0]
        sfj = SingleFringeJones(cal)
        new_xds = sfj.apply_cal(xds, group_name='another_calibrated_set')
        ps2[k] = new_xds
    return ps2


def _apply_cal_task(
        input_params: Dict
):
    ps = input_params["ps"]
    caltree = input_params["caltree"]
    data_selection = input_params["data_selection"]
    # FIXME: for now we do single band
    xdses = [ps[k] for k in data_selection.keys() if ps[k].time.size > 0]
    #if len(xdses) > 1:
    #    raise RuntimeError(f"We only do single xdses so far {data_selection.keys()}")
    xds = xdses[0]
    new_xdses = []
    for x in xdses:
        cal_matches = [q for q in caltree.values()
                       if q.spectral_window_name == x.frequency.spectral_window_name
        ]
        if len(cal_matches)==0:
            print(f"Nothing found to match {x.frequency.spectral_window_name}")
            continue
        else:
            cal = cal_matches[0]
        sfj = SingleFringeJones(cal)
        new_xds = sfj.apply_cal(xds, group_name='another_calibrated_set')
        new_xdses.append(new_xds)
    return new_xdses

def _apply_cal_single(
        ps: ProcessingSetXdt,
        node_task_data_mapping: Dict,
        caltree: xa.DataTree
):
    """Maps a dask task for applycal over the data mapping supplied.
    """
    input_params = {}
    input_params["ps"] = ps
    input_params["caltree"] = caltree
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_apply_cal_task,
        input_params=input_params,
        in_memory_compute=True,
    )
    dask_graph = generate_dask_workflow(graph)
    print(f"Made a graph {dask_graph}")
    # Strip the singleton list wrapper at source
    [res] = dask.compute(dask_graph)
    return res
    
def apply_cal_dask(ps: ProcessingSetXdt, caltree: xa.DataTree):
    parallel_coords = {}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps, ps_partition=["spectral_window_name", "field_name"] # "scan_name"]
    )
    print(f"Got mapping, at least")
    res0 = _apply_cal_single(ps, node_task_data_mapping, caltree)
    return res0
