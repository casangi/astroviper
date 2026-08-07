"""Distributed ``imstatistics`` application for on-disk images.

The application applies the user selection before partitioning and builds a
GraphVIPER map/reduce workflow over only that selected coordinate range. Each
map node loads its pixel block and emits a compact ``min/max/sum/npts`` state;
image pixels never pass through the reduction tree.

If the partition dimension is reduced by ``axes``, partial states are merged
numerically. If it is retained, disjoint outputs are concatenated along that
dimension and sorted by coordinate. Thus partition count changes execution
granularity, not the numerical result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _reduce_statistics_states(input_data, input_params):
    """Adapt GraphVIPER reduction inputs to the state merger."""
    from astroviper.processing_functions.image_analysis.statistics import (
        merge_statistics_states,
    )

    return merge_statistics_states(
        input_data,
        partition_dim=input_params["partition_dim"],
        reduction_dims=input_params["reduction_dims"],
    )


def _automatic_partition_count(
    selected: xr.DataArray,
    *,
    partition_dim: str,
    memory_limit_gib: float | None,
    working_memory_factor: float,
) -> int:
    """Estimate partitions from the selected image's in-memory footprint.

    Uncompressed array bytes are multiplied by ``working_memory_factor`` to
    allow for masks and temporary reductions. The result is capped at the
    selected axis length, preventing empty partitions.
    """
    from astroviper.utils.data_partitioning import get_thread_info

    if memory_limit_gib is None:
        memory_limit_gib = 0.5 * float(get_thread_info()["memory_per_thread"])
    if memory_limit_gib <= 0:
        raise ValueError("memory_limit_gib must be positive")
    if working_memory_factor < 1:
        raise ValueError("working_memory_factor must be at least 1")
    selected_bytes = (
        int(np.prod(list(selected.sizes.values()))) * selected.dtype.itemsize
    )
    required_bytes = selected_bytes * working_memory_factor
    available_bytes = memory_limit_gib * 1024**3
    return max(
        1,
        min(
            selected.sizes[partition_dim],
            int(np.ceil(required_bytes / available_bytes)),
        ),
    )


def image_statistics(
    image: str | Path,
    *,
    data_variable: str | None = None,
    axes=-1,
    region="",
    box="",
    chans="",
    stokes="",
    timerange="",
    mask=None,
    stretch: bool = False,
    includepix=None,
    excludepix=None,
    statistics=("max", "min", "sum", "mean", "npts"),
    partition_dim: str = "frequency",
    n_partitions: int | None = None,
    memory_limit_gib: float | None = None,
    working_memory_factor: float = 2.5,
    reduce_mode: str = "tree",
    reduce_n_batch: int = 2,
) -> xr.Dataset:
    """Calculate image statistics with partitioned on-disk loading.

    Parameters
    ----------
    image : str or pathlib.Path
        On-disk XRADIO/Zarr image. In-memory images should call the node task.
    data_variable, axes, region, box, chans, stokes, timerange, mask, stretch,
    includepix, excludepix, statistics
        Same selection and statistics parameters as
        :func:`astroviper.node_tasks.image_analysis.image_statistics`.
        Distributed masks must be named variables in the image store.
    partition_dim : str, default "frequency"
        Dimension divided among map nodes.
    n_partitions : int, optional
        Explicit number of partitions. When omitted, it is estimated from the
        selected variable size and worker memory.
    memory_limit_gib : float, optional
        Per-task memory target used for automatic partitioning. By default half
        of the detected memory per worker thread is used.
    working_memory_factor : float, default 2.5
        Multiplicative allowance for masks and temporary reduction arrays.
    reduce_mode : {"tree", "single_node", "tree_n"}, default "tree"
        GraphVIPER reduction topology.
    reduce_n_batch : int, default 2
        Fan-in for ``tree_n`` reductions.

    Returns
    -------
    xarray.Dataset
        Final statistics. Map nodes return only compact partial states.

    Notes
    -----
    The application partitions only the effective user selection. Every map
    node calls the same directly callable node task and returns a compact
    mergeable state; image pixels never enter the reduction tree.

    Automatic partitioning estimates the uncompressed selected variable size,
    not the compressed Zarr size. ``n_partitions`` affects resource usage but
    must not affect coordinates or statistics.

    Examples
    --------
    Compute global statistics using four frequency partitions::

        image_statistics("image.zarr", n_partitions=4)

    Retain frequency while reducing every other canonical image dimension::

        image_statistics(
            "image.zarr",
            axes=("time", "polarization", "l", "m"),
            n_partitions=4,
        )
    """
    if not isinstance(image, (str, Path)):
        raise TypeError("The distributed application requires an on-disk image path")
    if mask is not None and not isinstance(mask, str):
        raise TypeError("Distributed on-disk masks must be named image variables")

    import dask
    from graphviper.graph_tools import generate_dask_workflow
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from graphviper.graph_tools.map import map
    from graphviper.graph_tools.reduce import reduce

    from astroviper.node_tasks.image_analysis.image_statistics import (
        _open_metadata,
        build_image_selection,
    )
    from astroviper.node_tasks.image_analysis.image_statistics import (
        image_statistics as image_statistics_node,
    )
    from astroviper.processing_functions.image_analysis.statistics import (
        finalize_statistics_state,
    )

    metadata, variable = _open_metadata(str(image), data_variable)
    source = metadata[variable]
    selection = build_image_selection(
        source,
        axes=axes,
        region=region,
        box=box,
        chans=chans,
        stokes=stokes,
        timerange=timerange,
    )
    if partition_dim not in source.dims:
        raise ValueError(f"Partition dimension {partition_dim!r} is not present")
    required = [variable]
    if isinstance(mask, str) and mask:
        if mask not in metadata:
            raise KeyError(f"Image has no mask variable {mask!r}")
        required.append(mask)
    selected_metadata = metadata[required].isel(selection.effective_indexers)
    size = selected_metadata.sizes[partition_dim]
    if size == 0:
        raise ValueError("The image selection is empty")
    if n_partitions is None:
        n_partitions = _automatic_partition_count(
            selected_metadata[variable],
            partition_dim=partition_dim,
            memory_limit_gib=memory_limit_gib,
            working_memory_factor=working_memory_factor,
        )
    if (
        not isinstance(n_partitions, int)
        or isinstance(n_partitions, bool)
        or n_partitions < 1
    ):
        raise ValueError("n_partitions must be a positive integer")
    n_partitions = min(n_partitions, size)

    parallel_coords = {
        partition_dim: make_parallel_coord(
            coord=selected_metadata[partition_dim], n_chunks=n_partitions
        )
    }
    partition_template = xr.Dataset(
        coords={
            name: coordinate for name, coordinate in selected_metadata.coords.items()
        }
    )
    input_data = {"image": partition_template}
    mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, input_data)
    input_params = {
        "image": str(image),
        "data_variable": variable,
        "axes": axes,
        "region": region,
        "box": box,
        "chans": chans,
        "stokes": stokes,
        "timerange": timerange,
        "mask": mask,
        "stretch": stretch,
        "includepix": includepix,
        "excludepix": excludepix,
        "statistics": statistics,
        "finalize": False,
    }
    graph = map(
        input_data=input_data,
        node_task_data_mapping=mapping,
        node_task=image_statistics_node,
        input_params=input_params,
        in_memory_compute=False,
    )
    graph = reduce(
        graph,
        _reduce_statistics_states,
        {
            "partition_dim": partition_dim,
            "reduction_dims": selection.reduction_dims,
        },
        mode=reduce_mode,
        n_batch=reduce_n_batch,
    )
    workflow = generate_dask_workflow(graph)
    (state,) = dask.compute(workflow)
    return finalize_statistics_state(state, statistics)


imstatistics = image_statistics
