import numpy as np
import dask
import xarray as xr
from graphviper.graph_tools.map import map
from graphviper.graph_tools.reduce import reduce
from typing import Dict, Union

def _fringe_node_task(input_params: Dict):
    an_xds = input_params['xds']
    data_sub_selection = input_params['data_sub_selection']
    pol = data_sub_selection['polarization']
    baselines = input_params['task_coords']["baseline_id"]['data']
    t =  input_params['task_coords']["time"]['data']
    xds2 = an_xds.sel(time=t, baseline_id=baselines, polarization=pol)
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
    ref_ant = 1
    res = {}
    for i, bl in enumerate(baselines):
        ant1 = int(an_xds.baseline_antenna1_id[bl].values)
        ant2 = int(an_xds.baseline_antenna2_id[bl].values)
        a = np.abs(fftvis[:,i,:])
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        print(f"Time {t[0]} baseline {bl} ({ant1}-{ant2}), peak at {ind} val {a[ind]} out of {np.mean(a)}"
              f" , array{a.shape}")
        res.setdefault(t[0], {})[bl] = (ind, a[ind], a.shape)
    return res

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

def fringefit_single(an_xds: xr.Dataset, node_task_data_mapping: Dict, sub_selection: Dict):    
    input_params = {}
    input_params['xds'] = an_xds
    input_params['data_sub_selection'] = sub_selection
    graph = map(
        input_data = an_xds,
        node_task_data_mapping = node_task_data_mapping,
        node_task = _fringe_node_task, 
        input_params = input_params,
        in_memory_compute=False)
    graph_reduce = reduce(graph, _fringefit_reduce, {}, mode="single_node")
    res = dask.compute(graph_reduce)
    return res


