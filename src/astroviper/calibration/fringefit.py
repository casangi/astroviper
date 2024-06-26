import numpy as np
import dask
import xarray as xr
from graphviper.graph_tools.map import map
from graphviper.graph_tools.reduce import reduce
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from typing import Dict, Union

def _fringe_node_task(input_params: Dict):
    ps = input_params['ps']
    data_selection = input_params['data_selection']
    name = list(data_selection.keys())[0]
    xds = ps[name]
    # FIXME: for now we do single band
    data_sub_selection = input_params['data_sub_selection']
    # breakpoint()
    pol = data_sub_selection['polarization']
    xds2 = xds.isel(**input_params['data_selection'][name])
    xds2 = xds2.sel(polarization=pol)
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
    bl_slice = input_params['data_selection'][name]["baseline_id"]
    baselines = list(xds.baseline_id[bl_slice].values)
    for i, bl in enumerate(baselines):
        ant1 = int(xds2.baseline_antenna1_id[bl].values)
        ant2 = int(xds2.baseline_antenna2_id[bl].values)
        a = np.abs(fftvis[:,i,:])
        ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
        res.setdefault(xds2.time.values[0], {})[bl] = (ind, a[ind], a.shape)
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


def fringefit_single(ps, node_task_data_mapping: Dict, sub_selection: Dict):
    """
TODO!
"""    
    input_params = {}
    input_params['data_sub_selection'] = sub_selection
    input_params['ps'] = ps
    graph = map(
        input_data = ps,
        node_task_data_mapping = node_task_data_mapping,
        node_task = _fringe_node_task, 
        input_params = input_params,
        in_memory_compute=False)
    dask_graph = generate_dask_workflow(graph)
    res = dask.compute(dask_graph)
    return res
