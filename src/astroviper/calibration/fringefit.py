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
    # FIXME: for now we do single band
    if len(data_selection.keys())>1:
        print(f'{data_selection.keys()=}')
        raise RuntimeError("We only do single xdses so far")
    name = list(data_selection.keys())[0]
    xds = ps[name]
    data_sub_selection = input_params['data_sub_selection']
    pol = data_sub_selection['polarization']
    xds2 = xds.isel(**data_selection[name])
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
    bl_slice = data_selection[name]["baseline_id"]
    baselines = xds.baseline_id[bl_slice].values
    ant1s = xds2.baseline_antenna1_id.values
    ant2s = xds2.baseline_antenna1_id.values
    try:
        # FIXME:
        #
        # In the case of subcubes we *don't* get all the antenna1_id stuff!
        # antenna1_id.values: [ 6  6  6  7  7  7  7  7  8  8  8  8  9  9  9 10 10 11]
        # baselines :         [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77]
        for i, (bl, ant1, ant2) in enumerate(zip(baselines, ant1s, ant2s)):
            a = np.abs(fftvis[:,i,:])
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            res.setdefault(xds2.time.values[0], {})[bl] = (ind, a[ind], a.shape)
    except IndexError as e: 
        print(f'{xds2.baseline_antenna1_id.values}\n{baselines=}')
        raise e
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
