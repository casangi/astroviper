"""Distributed ``imstatistics`` application for on-disk images.

The application applies the user selection before partitioning and builds a
GraphVIPER map/reduce workflow over only that selected coordinate range. Each
map node loads its pixel block and emits a mergeable numerical state. Most
statistics use a compact state; exact median statistics additionally retain
the selected samples through the reduction tree.

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
    data_variable : str, optional
        Image data variable to analyze. Required when the dataset does not
        contain exactly one image variable with the canonical image axes.
    axes : int, str, or sequence, default -1
        Dimensions reduced by every requested statistic. ``-1`` and ``None``
        reduce all dimensions. Names such as ``("l", "m")`` are preferred;
        integer axes follow the data variable's dimension order. Dimensions
        not listed here are retained in every returned variable.
    region : str, pathlib.Path, or dict, optional
        Pixel-coordinate region selection. Accepted forms are a record with
        ``blc`` and ``trc``, CRTF ``box[[...pix],[...pix]]`` text, or a CRTF
        file containing pixel boxes.
    box : str, optional
        CASA inclusive pixel-box syntax ``"x0,y0,x1,y1"``. Additional groups
        of four integers form a union of boxes. Specify either ``box`` or
        ``region``, not both.
    chans, timerange : str, optional
        Zero-based frequency or time selections. Scalars, inclusive ``a~b``
        ranges, stepped ``a~b^step`` ranges, comma/semicolon unions, and
        ``<``, ``<=``, ``>``, and ``>=`` expressions are supported.
    stokes : str, optional
        Polarization labels, concatenated or comma-separated.
    mask : str, optional
        Name of a Boolean mask variable in the on-disk image. Mask values of
        ``True`` include pixels. In-memory masks are supported by the node-task
        interface but cannot be serialized into this distributed application.
    stretch : bool, default False
        Permit a named mask with fewer or degenerate dimensions to broadcast
        across the selected image. Without stretching, mask dimensions and
        sizes must exactly match the selected image.
    includepix : pair of float, optional
        Inclusive ``[low, high]`` pixel-value range to retain after selection
        and masking.
    excludepix : pair of float, optional
        Inclusive ``[low, high]`` pixel-value range to exclude after selection
        and masking.
    statistics : sequence of str, default ("max", "min", "sum", "mean", "npts")
        Statistics to return. Supported names are:

        - ``"mean"``, ``"median"``, ``"min"``, ``"max"``, ``"sum"``,
          ``"sumsq"``, ``"npts"``, ``"sigma"``, and ``"rms"``;
        - ``"minpos"`` and ``"maxpos"`` for absolute pixel positions along
          the reduced dimensions; and
        - ``"medabsdevmed"`` for median absolute deviation from the median,
          with ``"mad"`` accepted as an alias.

        Only requested variables are returned. Exact ``median`` and median
        absolute deviation retain selected samples in the distributed state;
        all other statistics use compact partial summaries.
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
        One data variable per requested statistic. Unreduced dimensions and
        their coordinates are retained. Position variables add a
        ``statistics_axis`` dimension naming the reduced axes. Value
        statistics preserve the input variable's units; ``npts`` and position
        variables are unitless.

    Notes
    -----
    The application partitions only the effective user selection. Every map
    node calls the same directly callable node task and returns a mergeable
    state. Exact median statistics carry selected samples through that state;
    other requested statistics use compact partial summaries.

    Automatic partitioning estimates the uncompressed selected variable size,
    not the compressed Zarr size. ``n_partitions`` affects resource usage but
    must not affect coordinates or statistics.

    Examples
    --------
    Compute global statistics using four frequency partitions::

        image_statistics("image.zarr", n_partitions=4)

    Compute only robust noise and peak-location statistics::

        image_statistics(
            "image.zarr",
            axes=("l", "m"),
            mask="MASK_SKY",
            stretch=True,
            statistics=("medabsdevmed", "rms", "max", "maxpos"),
        )

    Retain frequency while reducing every other canonical image dimension::

        image_statistics(
            "image.zarr",
            axes=("time", "polarization", "l", "m"),
            n_partitions=4,
        )
    """
    if not isinstance(image, str | Path):
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
