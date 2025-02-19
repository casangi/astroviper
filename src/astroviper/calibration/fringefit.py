from xradio.measurement_set.processing_set import ProcessingSet
from xradio.measurement_set.open_processing_set import open_processing_set

from graphviper.graph_tools.map import map
from graphviper.graph_tools.reduce import reduce
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from graphviper.graph_tools.coordinate_utils import (interpolate_data_coords_onto_parallel_coords,
                                                     make_parallel_coord, make_time_coord, make_frequency_coord)
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow

from astroviper.calibration.fringe_cal_quantum import make_empty_cal_quantum


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
    df = (f[-1] - f[0])/(len(f)-1)
    dF = len(f)*df
    ddelay = 1/dF/npad
    #
    t = xds.time.values
    dt = (t[-1] - t[0])/(len(t)-1)
    dT = len(t) * dt
    drate = 1/dT/npad
    return (ddelay, drate)


def _fringe_node_task(input_params: Dict):
    ps = input_params['ps'] 
    data_selection = input_params['data_selection']
    ref_ant = input_params['ref_ant']
    npad = 16 # To see if it's that
    # FIXME: for now we do single band
    if len(data_selection.keys()) > 1:
        print(f'{data_selection.keys()=}')
        raise RuntimeError("We only do single xdses so far")
    name = list(data_selection.keys())[0]
    xds = ps[name]
    q = make_empty_cal_quantum(xds)
    q.ref_antenna[0] = ref_ant
    data_sub_selection = input_params['data_sub_selection']
    allpols = 'RL' # FIXME data_sub_selection['polarization']
    xds2 = xds.isel(**data_selection[name])
    ddelay, drate = getFourierSpacings(xds2, npad)
    ref_bls = (xds.baseline_antenna1_name==ref_ant) | (xds.baseline_antenna2_name==ref_ant)
    vis = xds2.VISIBILITY.where(ref_bls)[:, :, :, ::3]
    ang = np.angle(vis)
    normed = np.exp(1J*ang)
    normed = np.where(np.isnan(normed), 0, normed)
    s = list(normed.shape)
    nt = s[0]
    nf = s[2]
    s[0] = npad*nt # Pad in time
    s[2] = npad*nf # Pad in frequency
    pad = np.zeros(s, complex)
    pad[:nt, :, :nf, :] = normed
    # Zero the NaNs
    fftvis = np.fft.fftshift(
        np.fft.fft2(pad, axes=(0,2)),
        axes=(0,2))
    npols = normed.shape[-1]
    spw = xds.partition_info['spectral_window_name']
    t = xds.time[0].values
    for i, bl in enumerate(ref_bls):
        if not bl: continue
        ant1 = xds2.baseline_antenna1_name.values[i]
        ant2 = xds2.baseline_antenna2_name.values[i]
        if ref_ant == ant1 and ref_ant==ant2:
            for p in range(npols):
                q.CALIBRATION_PARAMETER.loc[dict(antenna_name=ant1, polarization=allpols[p])] = [0, 0, 0]
                q.SNR.loc[dict(antenna_name=ant1, polarization=allpols[p])] = 3*[1e9]
        ant = ant1 if (ant2 == ref_ant) else ant2
        for p in range(npols):
            ft = fftvis[:, i, :, p]
            a = np.abs(ft) 
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            ix, iy = ind
            nx, ny = a.shape
            phi0 = np.angle(ft[ind])
            # The dimensions are time *then* frequency, so delay is y; rate is x.
            delay = (iy - ny/2)*ddelay
            ref_freq = xds.frequency.reference_frequency['data']
            phi0 -= 2*np.pi*(delay * (xds.frequency.values[0] - ref_freq)) # We'll fix the sign convention empirically
            rate = (ix - nx/2)*drate/ref_freq
            peak = np.abs(a[ind])
            q.CALIBRATION_PARAMETER.loc[dict(antenna_name=ant, polarization=allpols[p])] = [phi0, delay, rate]
            q.SNR.loc[dict(antenna_name=ant, polarization=allpols[p])] = [peak, peak, peak]
    return q

def _fringefit_single(ps: ProcessingSet, node_task_data_mapping: Dict, sub_selection: Dict, ref_ant: str):
    """Fringefits a 

"""    
    input_params = {}
    input_params['data_sub_selection'] = sub_selection
    input_params['ps'] = ps
    input_params['ref_ant'] = ref_ant
    graph = map(
        input_data = ps,
        node_task_data_mapping = node_task_data_mapping,
        node_task = _fringe_node_task, 
        input_params = input_params,
        in_memory_compute=False)
    dask_graph = generate_dask_workflow(graph)
    # Strip the singleton list wrapper at source
    [res] = dask.compute(dask_graph)
    return res

def fringefit_ps(ps: ProcessingSet, ref_ant: str, unixtime: float, interval: float):
    """Fringefits a ProcessingSet for a time slice, handling each spw separately.

Currently assumes that each there will be one xds for each spw in the ProcessingSet."""
    ps2 = ps.ms_sel(time=slice(unixtime, unixtime+interval))
    # We manually filter out the empty xdses:
    ps3 = {k : v for k, v in ps2.items() if len(v.time.values) != 0}
    # We need an xds to extract baseline_ids from; just grab the first one from ps3
    xds = next(iter(ps3.values())) 
    baselines = xds.baseline_id
    parallel_coords = {}
    parallel_coords['baseline_id'] = make_parallel_coord(coord=xds.baseline_id, n_chunks=1)
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords,
                                                                          ps3, ps_partition=['spectral_window_name'])
    subsel={} 
    res = _fringefit_single(ps3, node_task_data_mapping, subsel, ref_ant)
    return res
