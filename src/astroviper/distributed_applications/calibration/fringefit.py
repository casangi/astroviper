import dask
import numpy as np
import xarray as xr

from typing import Union, Mapping


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
    _fringe_fft_mbd_node_task,
    _fringe_global_node_task,
    _fringe_global_mbd_node_task,
    _apply_cal_task,
    get_cal_for_xds,
)

from .utils import make_cal_map


def fringefit_fft_ps(
    ps: ProcessingSetXdt,
    ref_ant: str,
    selection: Union[dict, None] = None,
    options: Union[dict, None] = None,
):
    """Fringefits a ProcessingSetXdt, handling each spw separately.

    This is the top level function!
    Currently assumes that each there will be one xdt for each spw in the ProcessingSetXdt.
    """
    if selection == None:
        selection = {}
    if options == None:
        options = {}
    ps2 = ps.sel(**selection)
    # This returns *all* the xdses with that timerange filter
    # applied. Many of them will now have no entries in the time
    # dimension, but that's the ps.sel api and we deal with it.

    xds_ants = ps2.xr_ps.get_combined_antenna_xds()
    xds_fields = ps2.xr_ps.get_combined_field_and_source_xds()
    baselines = xds_ants.baseline_id
    parallel_coords = {}
    parallel_coords["baseline_id"] = make_parallel_coord(
        coord=xds_ants.baseline_id, n_chunks=1
    )
    if "solint" in options:
        # FIXME:
        # Seems like make_parallel_coord is about parallelisation,
        # not really about subdividing the data, so we just split by field name for now
        # and look sad about it.
        pass
    multiband = options.get("combine_spw", False)
    vis_key = options.get("vis_key", "VISIBILITY")
    if multiband:
        ## FIXME: we should be partitioning by field name, for sure,
        ## But right now we only want to do a single timerange and it's
        ## failing.
        ## ps_partition = ["field_name"]
        ps_partition = []
    else:
        ps_partition = ["spectral_window_name"]
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps2, ps_partition=ps_partition
    )
    subsel = {}
    res0 = _fringefit_fft(
        ps2,
        node_task_data_mapping,
        subsel,
        ref_ant=ref_ant,
        vis_key=vis_key,
        options=options,
    )
    # Now we collect these into a single DataTree!
    res_cal_tree = xr.DataTree(
        children={f"part{i}": xr.DataTree(dataset=q) for i, q in enumerate(res0)}
    )
    return res_cal_tree


def fringefit_global_ps(
    ps: ProcessingSetXdt,
    ref_ant: str,
    initial_cal_tree: xr.DataTree,
    selection: Union[dict, None] = None,
    options: Union[dict, None] = None,
):
    """Refines a fringefit calibration using an estimate from the fringefit stage"""
    if selection == None:
        selection = {}
    if options == None:
        options = {}
    ps2 = ps.sel(**selection)
    # It would be very cool to do solint!
    # Add solution interval, split on scan boundaries *optionally*, split on spw!
    parallel_coords = {}
    multiband = options.get("combine_spw", False)
    if multiband:
        ps_partition = []
    else:
        ps_partition = ["spectral_window_name"]
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps2, ps_partition=ps_partition
    )
    subsel = {}
    res0 = _fringefit_global(
        ps2, node_task_data_mapping, subsel, ref_ant, initial_cal_tree, options
    )
    # Now we collect these into a single DataTree!
    res_cal_tree = xr.DataTree(
        children={f"part{i}": xr.DataTree(dataset=q) for i, q in enumerate(res0)}
    )
    return res_cal_tree


def _fringefit_fft(
    ps: ProcessingSetXdt,
    node_task_data_mapping: dict,
    sub_selection: dict,
    ref_ant: str,
    vis_key: Union[str, None] = None,
    options: Union[dict, None] = None,
):
    """Fringefits via the FFT method (only).

    If options['combine_spw'] is true, do a multiband FFT"""

    input_params = {}
    input_params["data_sub_selection"] = sub_selection
    input_params["ps"] = ps
    input_params["ref_ant"] = ref_ant
    if vis_key != None:
        input_params["vis_key"] = vis_key
    if not options.get("combine_spw", False):
        node_task = _fringe_fft_node_task
    else:
        node_task = _fringe_fft_mbd_node_task
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    # print(f"{graph=}")
    dask_graph = generate_dask_workflow(graph)
    # Strip the singleton list wrapper at source
    [res] = dask.compute(dask_graph)
    return res


def _fringefit_global(
    ps: ProcessingSetXdt,
    node_task_data_mapping: Mapping,
    data_sub_selection: Mapping,
    ref_ant: str,
    cal_tree: xr.DataTree,
    options: Union[dict, None] = None,
):
    """The top level Dask routine for global fringefit."""
    if options == None:
        options = {}
    if options.get("combine_spw", False):
        print("Combining!")
        node_task = _fringe_global_mbd_node_task
    else:
        print(f"Not combining! {options=}")
        node_task = _fringe_global_node_task
    vis_key = options.get("vis_key", "VISIBILITY")
    input_params = {}
    input_params["data_sub_selection"] = data_sub_selection
    input_params["ps"] = ps
    input_params["ref_ant"] = ref_ant
    input_params["cal_map"] = make_cal_map(cal_tree)
    input_params["vis_key"] = vis_key
    input_params["options"] = options
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=node_task,
        input_params=input_params,
        in_memory_compute=False,
    )
    dask_graph = generate_dask_workflow(graph)
    [res] = dask.compute(dask_graph)
    return res


# This doesn't use the dask stuff!
# Deprecate it!
def apply_cal_ps(
    ps: ProcessingSetXdt,
    cal_tree: xr.DataTree,
    *,
    selection: Union[None, dict] = None,
    spw_map: Union[None, dict] = None,
    vis_key_in: str = "VISIBILITY",
    vis_key_out: str = "VISIBILTY_CORRECTED",
):
    """Apply a cal_tree to a processing set (or a selection from one)"""
    if selection == None:
        selection = {}

    ps2 = ps.sel(**selection)
    for k, xds in ps2.items():
        cal = get_cal_for_xds(xds, cal_tree, spw_map)
        sfj = SingleFringeJones(cal)
        new_xds = sfj.apply_cal(xds, vis_key_in=vis_key_in, vis_key_out=vis_key_out)
        ps2[k] = new_xds
    return ps2


def apply_cal_dask(
    ps: ProcessingSetXdt,
    cal_tree: xr.DataTree,
    *,
    selection: Union[None, dict] = None,
    spw_map: Union[None, dict] = None,
):
    if selection != None:
        ps2 = ps.sel(**selection)

    parallel_coords = {}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps, ps_partition=["spectral_window_name", "scan_name"]
    )
    input_params = {"ps": ps2, "cal_tree": cal_tree, "spw_map": spw_map}
    graph = map(
        input_data=ps2,
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
        name = xds.name
        ps2[name] = xds
    return ps2
