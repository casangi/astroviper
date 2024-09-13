import numpy as np
import dask
import xarray as xr
from graphviper.graph_tools.map import map
from graphviper.graph_tools.reduce import reduce
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from typing import Dict, Union

from xradio.vis.read_processing_set import read_processing_set

import dask
import numpy as np
import xarray as xa
import pandas as pd
import datetime

## I should figure out where this belongs at some point
def unique(s):
    "Get the unique characters in a string in the order they occur in the original"
    u = []
    for c in s:
        if c not in u:
            u.append(c)
    return u




def getFourierSpacings(xds):
    f = xds.frequency.values
    df = (f[-1] - f[0])/(len(f)-1)
    dF = len(f)*df
    ddelay = 1/dF
    #
    t = xds.time.values
    dt = (t[-1] - t[0])/(len(t)-1)
    dT = len(t) * dt
    drate = 1/dT
    return (ddelay, drate)

def makeCalArray(xds, ref_ant):
    pols_ant = unique(''.join([c for c in ''.join(xds.polarization.values)]))
    quantumCoords = xa.Coordinates(coords={'antenna_name' : xds.antenna_xds.antenna_name,
                                           'polarization' : pols_ant,
                                           'parameter' : range(3)
                                           })
    q = xa.DataArray(coords=quantumCoords)
    ref_freq = xds.frequency.reference_frequency['data']
    # Should we choose this reference time?
    ref_time = xds.time[0]
    q.attrs['reference_frequency'] = ref_freq
    q.attrs['reference_time'] = ref_time
    q.attrs['reference_antenna'] = ref_ant
    return q

def _fringe_node_task(input_params: Dict):
    ps = input_params['ps'] 
    data_selection = input_params['data_selection']
    ref_ant = input_params['ref_ant']
    # FIXME: for now we do single band
    if len(data_selection.keys()) > 1:
        print(f'{data_selection.keys()=}')
        raise RuntimeError("We only do single xdses so far")
    name = list(data_selection.keys())[0]
    xds = ps[name]
    q = makeCalArray(xds, ref_ant)
    data_sub_selection = input_params['data_sub_selection']
    pols = data_sub_selection['polarization']
    # FIXME!
    pol = pols[0]
    xds2 = xds.isel(**data_selection[name])
    xds2 = xds2.sel(polarization=pols)
    ddelay, drate = getFourierSpacings(xds2)
    vis = xds2.VISIBILITY
    ang = np.angle(vis)
    nvis = np.exp(1J*ang)
    # Zero the NaNs
    nvis = np.where(np.isnan(vis), 0, nvis)
    fftvis = np.fft.fftshift(
         np.fft.fft2(
             nvis,
             axes=(0,2)
         ),
        axes=(0,2)
    )
    bl_slice = data_selection[name]["baseline_id"]
    baselines = xds2.baseline_id[bl_slice].values
    ant1s = xds2.baseline_antenna1_name.values
    ant2s = xds2.baseline_antenna2_name.values
    try:
        for i, (bl, ant1, ant2) in enumerate(zip(baselines, ant1s, ant2s)):
            if ref_ant not in [ant1, ant2]:
                # print(f"Skipping {ant1}-{ant2}")
                continue
            if ref_ant == ant1 and ref_ant==ant2:
                print("Skipping autos")
            # print(f"{ant1}-{ant2}")
            ant = ant1 if (ant2 == ref_ant) else ant2
            spw = xds.partition_info['spectral_window_name']
            t = xds.time[0].values
            print(f"{ant} {spw} {t}")
            a = np.abs(fftvis[:, i, :]) 
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            # breakpoint()
            ix, iy = ind
            phi0 = np.angle(a[ind])
            delay = ix*ddelay
            ref_freq = xds.frequency.reference_frequency['data']
            rate = iy*drate/ref_freq
            q.loc[dict(antenna_name=ant, polarization=pol)] = [phi0, delay, rate]
    except IndexError as e: 
        print(f'{xds2.baseline_antenna1_name.values}\n{baselines=}')
        raise e
    return q

def _fringefit_reduce(graph_inputs: xr.Dataset, input_params: Dict):
    merged = {}
    for e in graph_inputs:
        [t] = e.keys()
        rhs = e[t]
        if t in merged:
            merged[t].update(rhs)
        else:
            merged[t] = rhs
    return merged


def fringefit_single(ps, node_task_data_mapping: Dict, sub_selection: Dict, ref_ant: int):
    """
TODO!
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
    res = dask.compute(dask_graph)
    return res
