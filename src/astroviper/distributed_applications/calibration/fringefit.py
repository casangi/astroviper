import dask
import numpy as np
import xarray as xr

from typing import (Union, Mapping)


from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
    make_parallel_coord,
)
from graphviper.graph_tools.generate_dask_workflow import generate_dask_workflow
from graphviper.graph_tools.map import map
from xradio.measurement_set.processing_set_xdt import ProcessingSetXdt

from astroviper.distributed_applications.calibration.fringe_cal_xds import (
    make_empty_cal_xds,
)

from astroviper.processing_functions.calibration.fringe_functions import (
    SingleFringeJones,
)

from astroviper.node_tasks.calibration.fringefit_tasks import (
    _fringe_fft_node_task,
    _fringe_global_node_task
)

from .utils import make_cal_map


def _fringefit_fft_single(
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
        node_task=_fringe_fft_node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    # print(f"{graph=}")
    dask_graph = generate_dask_workflow(graph)
    # Strip the singleton list wrapper at source
    [res] = dask.compute(dask_graph)
    return res


def fringefit_fft_ps(ps: ProcessingSetXdt, ref_ant: str, unixtime: float, interval: float):
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
    res0 = _fringefit_fft_single(ps2, node_task_data_mapping, subsel, ref_ant)
    # Now we collect these into a single DataTree!
    res_cal_tree = xr.DataTree(
        children={f"part{i}": xr.DataTree(dataset=q) for i, q in enumerate(res0)}
    )
    return res_cal_tree

def _fringefit_global_single(
        ps: ProcessingSetXdt,
        node_task_data_mapping: Mapping,
        data_sub_selection: Mapping,
        ref_ant: str,
        cal_tree: xr.DataTree,
        options: dict
):
    """The top level Dask routine for FFT fringefit.

    Currently each xds in a processing set is handled separately by the
    node task function, which is an important shortcoming that needs to
    be addressed soon.
    """
    input_params = {}
    input_params["data_sub_selection"] = data_sub_selection
    input_params["ps"] = ps
    input_params["ref_ant"] = ref_ant
    input_params["cal_map"] = make_cal_map(cal_tree)
    input_params["options"] = options
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_fringe_global_node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    dask_graph = generate_dask_workflow(graph)
    [res] = dask.compute(dask_graph)
    return res

def fringefit_global_ps(ps: ProcessingSetXdt,
                        ref_ant: str,
                        unixtime: float,
                        interval: float,
                        cal_tree: xr.DataTree,
                        options: dict):
    """Refines a fringefit calibration using an estimate from the fringefit stage

    This preliminary version handles spectral windows separately and maps
    the cal_tree spws to those in the data. The coupling between the
    fft stage and the global stage shouldn't really work like this and
    I should change it.

    """
    ps2 = ps.sel(time=slice(unixtime, unixtime + interval))
    parallel_coords = {}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps2, ps_partition=["spectral_window_name"]
    )
    subsel = {}
    res0 = _fringefit_global_single(ps2, node_task_data_mapping, subsel, ref_ant, cal_tree, options)
    # Now we collect these into a single DataTree!
    res_cal_tree = xr.DataTree(
        children={f"part{i}": xr.DataTree(dataset=q) for i, q in enumerate(res0)}
    )
    return res_cal_tree

def apply_cal_ps(
    ps: ProcessingSetXdt, cal_tree: xr.DataTree, unixtime: float, interval: float
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
    ps: ProcessingSetXdt, cal_tree: xr.DataTree
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
