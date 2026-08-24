"""Moments workflow driver (graph layer) -- AstroVIPER's CASA ``immoments``.

Builds and runs the GraphVIPER map graph that collapses an on-disk image along
one axis (``l``, ``m``, ``frequency``, ``polarization`` or ``time``) into a
set of moment maps.  It follows the same approach as
:func:`astroviper.distributed_applications.imaging.feather`:

1. create the empty output image on disk (coordinates with the moment axis
   collapsed to size 1, plus the per-moment data groups),
2. pre-allocate the ``SKY_MOMENT_*`` data variables on disk with
   :func:`astroviper.utils.io.create_empty_data_variables_on_disk`,
3. map the per-chunk node task
   :func:`astroviper.node_tasks.image_analysis.moments`, which loads only its
   own chunk of the sky (and optional mask) variables and writes its own
   slice of every moment map in parallel via
   :func:`astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.

The moment axis itself is never parallelized here (every moment needs the
full axis), so the graph is chunked along another axis: ``frequency`` by
default, or ``m`` when the moment axis *is* ``frequency``.

Memory / I-O strategy
---------------------
The cost of a moments run is set by two things the workflow must respect for
*any* on-disk layout: how much memory a task may use and how the store is
chunked (a partial read of an on-disk chunk decodes the whole chunk).

1. **Stream, never slab.** Every moment except ``median`` is computed by
   streaming the moment axis through output-map-sized accumulators
   (:class:`~astroviper.processing_functions.image_analysis.moments.MomentsAccumulator`).
   The node task reads *blocks* of planes straight from the store
   (:func:`~astroviper.node_tasks.image_analysis.moments.moment_axis_read_block`):
   a block spans a whole number of on-disk chunk lengths along the moment
   axis (so no chunk is decoded twice) and is capped at ``memory_budget_gb``.
   Per-task memory is therefore ``O(output maps) + block + decode transient``
   and does **not** grow with the length of the moment axis. Only ``median``
   (which needs each pixel's whole profile) falls back to holding the chunk
   slab, and only then does the moment-axis length enter the chunk count.
2. **Size chunks from that model, not from the slab.** The chunk count along
   the parallel axis is derived from the per-chunk estimate
   ``accumulators(map cells) + memory_budget_gb + decode transient`` for
   streamed moments (or the slab + median copies otherwise) and the
   available threads / memory (:func:`calculate_data_chunking`). With
   streaming, memory almost never constrains the chunk count, which frees
   the choice for I/O efficiency (next point).
3. **Align parallel chunks with on-disk chunks.** A task whose slice covers
   a fraction ``f`` of an on-disk chunk along the parallel axis still decodes
   the whole chunk, so total decode work is ``1/f`` x the store. Choose
   ``n_mapping_parallelism`` such that each parallel chunk is a whole number
   of on-disk chunks (``n_chunks = size / (k * chunk_len)``) whenever the
   task count allows; when it does not (few on-disk chunks along the
   parallel axis, many workers), the remaining amplification is
   ``chunk_len / slice_width`` and it is cheaper to add parallelism along the
   *moment* axis instead (point 4) than to split on-disk chunks further.
4. **Associative moments can also be split along the moment axis.** For
   single-pass moments (``moments_memory_model(...)["mergeable"]``) partial
   accumulators over disjoint moment-axis segments merge exactly
   (:meth:`MomentsAccumulator.merge`), so a map over (parallel-axis chunk x
   moment-axis segment -- ideally shard-aligned) plus a reduce reads every
   byte once with as many tasks as wanted. This driver does not build that
   graph yet; the accumulator API is the building block for it.
5. **Bound the decode transient explicitly.** Zarr decodes up to
   ``zarr.config["async.concurrency"]`` chunks at once; with large on-disk
   chunks this transient (compressed + decoded bytes per in-flight chunk)
   rather than the block can dominate, so set the concurrency from the chunk
   size and the budget (the Frontera script does this via
   ``ZARR_ASYNC__CONCURRENCY``).
"""

import os
import time

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter
import xarray as xr
from numcodecs import Blosc

# Import the node task directly from its module: the module and the function
# share the name "moments", so the package attribute
# node_tasks.image_analysis.moments is ambiguous (module vs function)
# depending on import order.
from astroviper.node_tasks.image_analysis.moments import (
    moments as _moments_node_task,
)
from astroviper.processing_functions.image_analysis.moments import (
    collapsed_moment_axis_coords,
    get_moments_data_variable_definitions,
    moment_axis_units,
    moment_data_variable_key,
    moment_units,
    moments_memory_model,
    normalize_dimension_flags,
    normalize_moments,
    normalize_pixel_range,
    resolve_moments_input_variables,
)
from astroviper.utils.data_group_tools import modify_data_groups_xds
from astroviper.utils.param_docs import shares_param_docs

# The toolviper parameter-check schema lives next to this module (rather than
# in a central config/ directory) so it is easy to find; point the validator
# at it.
_PARAM_CONFIG_DIR = os.path.dirname(__file__)

# Axes the graph may be chunked along (any combination, never the moment
# axis). Chunking is done in index space, so the decreasing ``l`` coordinate
# is fine; ``polarization`` is excluded because it is non-numeric and tiny.
ALLOWED_PARALLEL_AXES = ("frequency", "l", "m", "time")


def _open_input_image(input_image_store, selection):
    """Open the input image lazily and apply the user selection.

    Parameters
    ----------
    input_image_store : str
        Path to the on-disk input image (a ``.zarr`` store or any format
        ``xradio.image.open_image`` understands).
    selection : dict
        ``isel``-style ``{dim: slice}`` selection.

    Returns
    -------
    xarray.Dataset
        Lazily opened (data variables stay on disk), selected image dataset.
    """
    if not isinstance(input_image_store, str):
        raise TypeError(
            "input_image_store must be a string path to the on-disk image "
            "(each parallel node task reads its own chunk from disk)."
        )
    if "zarr" in input_image_store:
        img_xds = xr.open_zarr(input_image_store)
    else:
        from xradio.image import open_image

        img_xds = open_image(input_image_store)
    return img_xds.isel(selection)


def _collect_dataframes(results, frames):
    """Recursively collect the node tasks' timing frames from the compute result.

    Each node task returns ``{"timing_node_tasks": DataFrame}``; when the graph
    was built with ``monitor_resources_seconds`` graphviper adds a
    ``"resource_usage"`` dict-of-series which is folded into that task's
    one-row frame as list-valued columns (``time_seconds``, ``cpu_percent``,
    ``memory_rss_bytes``, ...) plus scalars (``sample_interval_seconds``,
    page-fault counts) -- the same layout as the imaging driver.
    """
    import pandas as pd

    if isinstance(results, dict) and "timing_node_tasks" in results:
        timing = results["timing_node_tasks"]
        usage = results.get("resource_usage")
        if usage is not None:
            timing = timing.copy()
            for key, value in usage.items():
                timing[key] = [value] if isinstance(value, list) else value
        frames.append(timing)
    elif isinstance(results, pd.DataFrame):
        frames.append(results)
    elif isinstance(results, list | tuple):
        for item in results:
            _collect_dataframes(item, frames)


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def moments(
    input_image_store: str,
    moments_image_store: str,
    moments: list = ["integrated"],  # noqa: B006 - param.json schema requires a list (not nullable); never mutated
    moment_axis: str = "frequency",
    image_data_group_in_name: str = "base",
    include_pixel_range: list | None = None,
    exclude_pixel_range: list | None = None,
    use_mask: bool = False,
    selection: dict | None = None,
    n_mapping_parallelism: dict[str, int | None] | None = None,
    thread_info: dict | None = None,
    compressor=None,
    overwrite: bool = False,
    memory_budget_gb: float = 1.0,
    monitor_resources_seconds: float | None = None,
    dimension_flags: dict | None = None,
):
    """Collapse an on-disk image along one axis into moment maps (CASA ``immoments``).

    The output is a single Zarr image store holding one ``SKY_MOMENT_<NAME>``
    data variable (and one ``moment_<name>`` data group) per requested moment,
    with the moment axis kept as a degenerate dimension of size 1 whose
    numeric coordinates are the mean of the input coordinates.  The compute is
    parallelized with a GraphVIPER map graph over the axis named in
    ``n_mapping_parallelism`` (never the moment axis); each node task loads
    only its own chunk of the sky (and optional mask) variables and writes its
    own slice of every moment map.

    Parameters
    ----------
    input_image_store : str
        Path to the on-disk input image (a ``.zarr`` store or any format
        ``xradio.image.open_image`` understands).
    moments_image_store : str
        Path of the output Zarr image store the moment maps are written to.
        Created up front by the distributed application and written
        chunk-by-chunk in parallel by the node tasks.
    moments : list of str or int, default ["integrated"]
        The moments to compute, as canonical names and/or CASA ``immoments``
        integer codes:

        - ``"mean"`` (CASA ``-1``) : mean value of the profile.
        - ``"integrated"`` (``0``) : integrated value ``sum(I * delta_v)``
          with ``delta_v`` the per-plane moment-axis coordinate width.
        - ``"weighted_coord"`` (``1``) : intensity-weighted coordinate
          ``sum(I * v) / sum(I)`` (e.g. velocity field).  Use
          ``include_pixel_range`` to restrict to positive flux for sensible
          results.
        - ``"weighted_dispersion_coord"`` (``2``) : intensity-weighted
          coordinate dispersion ``sqrt(sum(I * v^2)/sum(I) - m1^2)``.
        - ``"median"`` (``3``) : median value of the profile.
        - ``"median_coord"`` (``4``) : coordinate at which the cumulative
          profile crosses 50% of its total (only meaningful for
          predominantly positive profiles).
        - ``"standard_deviation"`` (``5``) : standard deviation about the
          profile mean.
        - ``"rms"`` (``6``) : root mean square of the profile.
        - ``"abs_mean_dev"`` (``7``) : mean absolute deviation from the
          profile mean.
        - ``"maximum"`` (``8``) / ``"maximum_coord"`` (``9``) : maximum of
          the profile and the coordinate at which it occurs.
        - ``"minimum"`` (``10``) / ``"minimum_coord"`` (``11``) : minimum of
          the profile and the coordinate at which it occurs.
    moment_axis : str, default "frequency"
        Image dimension to collapse: ``"l"``, ``"m"``, ``"frequency"``,
        ``"polarization"`` or ``"time"`` (AstroVIPER names for CASA's *ra*,
        *dec*, *spectral*, *stokes*).  The moment axis is never used for
        parallelism.  Coordinate-valued moments are expressed in the native
        coordinate units of this axis (Hz for ``frequency``, rad for
        ``l``/``m``); a non-numeric axis (``polarization``) uses the plane
        index.
    image_data_group_in_name : str, default "base"
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    include_pixel_range : list of float, optional
        Only pixel values inside ``[low, high]`` contribute.  A single value
        ``b`` means ``[-abs(b), abs(b)]`` (CASA convention).  Mutually
        exclusive with ``exclude_pixel_range``.
    exclude_pixel_range : list of float, optional
        Pixel values inside ``[low, high]`` do not contribute.  Same
        conventions as ``include_pixel_range``.
    use_mask : bool, default False
        If ``True``, pixels where the input data group's mask variable is
        ``False`` are excluded (XRADIO convention: mask ``True`` = include).
    selection : dict, optional
        ``xarray`` ``isel`` selection applied to the input image (e.g. a
        ``frequency`` channel range or an ``l``/``m`` sub-window), the
        AstroVIPER analogue of CASA's ``chans``/``stokes``/``box``.  Must not
        select along the parallel axis (chunking happens on the selected
        image).
    n_mapping_parallelism : dict, optional
        Mapping parallelism of the distributed graph, as a dict
        ``{parallel_axis: n_chunks | None}`` over any combination of
        ``"l"``, ``"m"``, ``"frequency"``, ``"time"`` (never the moment axis);
        one node task is mapped over every tile of the chunk grid, e.g.
        ``{"l": 18, "m": 18}`` = 324 tasks when collapsing ``frequency``. A
        ``None`` count is automatic: one chunk per on-disk chunk along that
        axis (tiles aligned with the store's inner chunks, so no chunk is
        decoded by more than one task), split further only if such a tile
        does not fit the per-thread memory. Default (``None``):
        ``{"l": None, "m": None}`` (or ``{"frequency": None}`` when the
        moment axis is ``l`` or ``m``).
    thread_info : dict, optional
        Thread information as returned by
        :func:`~astroviper.utils.data_partitioning.get_thread_info`; queried
        automatically when ``None``.
    compressor : numcodecs compressor, optional
        Compressor applied to each on-disk chunk of the moment maps.  Default
        ``Blosc(cname="lz4", clevel=5)``.
    overwrite : bool, default False
        If ``True`` an existing ``moments_image_store`` is overwritten;
        otherwise its presence raises ``RuntimeError``.
    memory_budget_gb : float, default 1.0
        Target size (GiB) of one decoded read block of a node task's chunk
        along the moment axis (see
        :func:`~astroviper.node_tasks.image_analysis.moments.moment_axis_read_block`):
        a block spans a whole number of on-disk chunk lengths along the
        moment axis and is capped at this size, so a task's peak memory is
        roughly the output maps + this block + the zarr decode transient.
        Also enters the distributed application's per-chunk memory estimate.
        Ignored by the ``median`` slab path.

    monitor_resources_seconds : float, optional
        If set, sample each node task's worker-process CPU / memory / I/O
        usage every this many seconds (graphviper's
        :func:`~graphviper.graph_tools.map.monitor_node_task`) and carry the
        series into the returned frame as list-valued columns
        (``time_seconds``, ``cpu_percent``, ``memory_rss_bytes`` and, on
        Linux, ``read_bytes``/``write_bytes``/``read_chars``/``write_chars``)
        plus the scalar ``sample_interval_seconds`` and page-fault counts.
        Requires ``psutil``. ``None`` (default) disables monitoring.

    Returns
    -------
    pandas.DataFrame
        Per-task timing frame (``task_id``, ``streamed``,
        ``n_read_block_planes``, ``T_load``, ``T_moments``, ``T_write``,
        ``T_moments_task``, ``start_unixtime``, ``hostname``, worker identity
        and, when monitored, the resource series), one row per graph node.

    Raises
    ------
    ValueError
        If a moment / axis choice is invalid, both pixel ranges are given, or
        ``selection`` selects along the parallel axis.
    RuntimeError
        If ``moments_image_store`` exists and ``overwrite=False``.
    """
    import dask
    import pandas as pd
    import zarr
    from graphviper.graph_tools import generate_dask_workflow
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from graphviper.graph_tools.map import map
    from xradio.image import write_image

    from astroviper.utils.data_partitioning import (
        calculate_data_chunking,
        get_thread_info,
    )
    from astroviper.utils.io import create_empty_data_variables_on_disk

    if selection is None:
        selection = {}
    if compressor is None:
        compressor = Blosc(cname="lz4", clevel=5)

    # --- Validate the moment / range / output parameters ----------------------
    moment_names = normalize_moments(moments)
    include_range = normalize_pixel_range(include_pixel_range, "include_pixel_range")
    exclude_range = normalize_pixel_range(exclude_pixel_range, "exclude_pixel_range")
    if include_range is not None and exclude_range is not None:
        raise ValueError(
            "Only one of include_pixel_range and exclude_pixel_range may be given."
        )
    if not overwrite and os.path.exists(moments_image_store):
        raise RuntimeError(
            f"Already existing file {moments_image_store} will not be overwritten. "
            "To overwrite it, set overwrite=True."
        )

    # --- Read input image metadata (lazy: no pixel data is loaded) ------------
    img_xds = _open_input_image(input_image_store, selection)
    sky_name, mask_name = resolve_moments_input_variables(
        img_xds, image_data_group_in_name, use_mask
    )
    sky = img_xds[sky_name]
    if moment_axis not in sky.dims:
        raise ValueError(
            f"moment_axis '{moment_axis}' is not a dimension of {sky_name} "
            f"(dims: {sky.dims})."
        )

    # --- Resolve and validate the mapping parallelism -------------------------
    # n_mapping_parallelism is {parallel_axis: n_chunks | None} over ANY
    # combination of non-moment axes (a 2-D (l, m) tiling is the natural
    # choice when collapsing frequency). None = auto: one parallel chunk per
    # on-disk chunk along that axis, so every task reads whole inner chunks
    # and no chunk is decoded by more than one task.
    if n_mapping_parallelism is None:
        if moment_axis in ("l", "m"):
            n_mapping_parallelism = {"frequency": None}
        else:
            n_mapping_parallelism = {"l": None, "m": None}
    if not isinstance(n_mapping_parallelism, dict) or not n_mapping_parallelism:
        raise ValueError(
            "n_mapping_parallelism must be a non-empty dict "
            f"{{parallel_axis: n_chunks | None}}; got {n_mapping_parallelism!r}."
        )
    parallel_axes = list(n_mapping_parallelism)
    for parallel_axis, n_parallel_chunks in n_mapping_parallelism.items():
        if n_parallel_chunks is not None and (
            isinstance(n_parallel_chunks, bool)
            or not isinstance(n_parallel_chunks, int)
            or n_parallel_chunks < 1
        ):
            raise ValueError(
                f"n_mapping_parallelism[{parallel_axis!r}] must be a positive "
                f"int or None (auto), got {n_parallel_chunks!r}."
            )
        if parallel_axis == moment_axis:
            raise ValueError(
                f"The moment axis '{moment_axis}' cannot be used for parallelism; "
                "choose different n_mapping_parallelism axes."
            )
        if parallel_axis not in ALLOWED_PARALLEL_AXES:
            raise ValueError(
                f"n_mapping_parallelism axis '{parallel_axis}' not in allowed axes "
                f"{ALLOWED_PARALLEL_AXES}."
            )
        if parallel_axis not in sky.dims:
            raise ValueError(
                f"n_mapping_parallelism axis '{parallel_axis}' is not a dimension "
                f"of {sky_name}."
            )
        if parallel_axis in selection:
            raise ValueError(
                "selection must not select along a parallel axis "
                f"('{parallel_axis}'); chunk indices are computed on the selected "
                "image, but node tasks apply the selection and the chunk slices "
                "to the on-disk image independently."
            )

    # --- Determine chunking along the parallel axes ---------------------------
    # On-disk chunk geometry of the sky variable (zarr stores expose it via
    # encoding; 1 = unknown / not chunked along that axis).
    start = time.time()
    on_disk_chunks = sky.encoding.get("chunks")
    if on_disk_chunks is not None and len(on_disk_chunks) == sky.ndim:
        chunk_len = {
            dim: int(c) for dim, c in zip(sky.dims, on_disk_chunks, strict=True)
        }
    else:
        chunk_len = {dim: 1 for dim in sky.dims}

    # Memory for the parallel-axes-size-1 cell (see the module docstring):
    # * streamed moments (everything but median): output-map-sized
    #   accumulators only -- up to ~10 float64/int64 maps (count, sums, running
    #   extrema + indices, pass-2 state) plus the output maps, i.e. ~100 B per
    #   map cell; the read block (memory_budget_gb) and the zarr decode
    #   transient (taken as another budget's worth) are chunk-size independent
    #   and enter as constant_memory.
    # * median: the loaded sky slab (full moment axis), the optional mask, plus
    #   the science function's extra copies (one working copy + numpy's
    #   internal partition copy when pixels are filtered).
    itemsize = sky.dtype.itemsize
    singleton_chunk_sizes = dict(sky.sizes)
    for parallel_axis in parallel_axes:
        del singleton_chunk_sizes[parallel_axis]
    singleton_cells = float(np.prod(np.array(list(singleton_chunk_sizes.values()))))
    fudge_factor = 1.3
    streamed = not moments_memory_model(moment_names)["requires_full_profile"]
    if not streamed:
        n_copies = 1.0
        if mask_name is not None:
            n_copies += 1.0 / itemsize
        filtering = include_range is not None or exclude_range is not None or use_mask
        n_copies += 2.0 if filtering else 1.0
        memory_singleton_chunk = (
            n_copies * singleton_cells * itemsize * fudge_factor / (1024**3)
        )
        constant_memory = 0.0
    if thread_info is None:
        thread_info = get_thread_info()
    logger.debug("Thread info " + str(thread_info))
    if streamed:
        map_cells = singleton_cells / sky.sizes[moment_axis]
        memory_singleton_chunk = map_cells * 100.0 * fudge_factor / (1024**3)
        # The read block may never take more than a quarter of a thread's
        # memory (block + decode transient + accumulators must all fit): clamp
        # the requested budget and hand the EFFECTIVE value to the node tasks.
        memory_budget_gb = float(
            min(memory_budget_gb, 0.25 * thread_info["memory_per_thread"])
        )
        constant_memory = 2.0 * memory_budget_gb
        logger.debug(f"moments: effective read-block budget {memory_budget_gb:.3f} GiB")

    n_chunks = {}
    auto_axes = [axis for axis in parallel_axes if n_mapping_parallelism[axis] is None]
    for parallel_axis in parallel_axes:
        if n_mapping_parallelism[parallel_axis] is not None:
            n_chunks[parallel_axis] = n_mapping_parallelism[parallel_axis]
    if auto_axes:
        # Aligned default: one task tile per on-disk chunk along each auto axis.
        aligned = {
            axis: max(1, -(-sky.sizes[axis] // chunk_len[axis])) for axis in auto_axes
        }
        # Memory check of that tile: cells of the fixed axes x the tile's extent
        # along the auto axes x per-cell cost (+ constant). If it does not fit
        # the per-thread memory, split the tile further along the auto axes
        # (calculate_data_chunking does that over the tile's extent).
        tile_cells = float(
            np.prod([min(chunk_len[axis], sky.sizes[axis]) for axis in auto_axes])
        )
        fixed_explicit = float(
            np.prod(
                [
                    -(-sky.sizes[axis] // n_chunks[axis])
                    for axis in parallel_axes
                    if axis not in auto_axes
                ]
            )
        )
        tile_memory = memory_singleton_chunk * tile_cells * fixed_explicit
        if tile_memory + constant_memory <= thread_info["memory_per_thread"]:
            n_chunks.update(aligned)
            logger.debug(
                f"moments: aligned tiling {aligned} (on-disk chunks {chunk_len}); "
                f"tile memory ~{tile_memory:.2f} GiB + {constant_memory:.2f} GiB"
            )
        else:
            logger.warning(
                f"moments: an on-disk-chunk-aligned tile needs ~{tile_memory:.1f} "
                f"GiB (+{constant_memory:.1f} GiB) but only "
                f"{thread_info['memory_per_thread']:.1f} GiB per thread is "
                "available; splitting tiles below the on-disk chunk size (each "
                "on-disk chunk will be decoded by several tasks)."
            )
            split = calculate_data_chunking(
                memory_singleton_chunk * fixed_explicit,
                {axis: sky.sizes[axis] for axis in auto_axes},
                thread_info,
                constant_memory=constant_memory,
                tasks_per_thread=1,
            )
            n_chunks.update({axis: split[axis] for axis in auto_axes})

    # Parallel coordinates. graphviper interpolates the chunk edges onto the
    # data coordinates, which requires monotonically increasing numeric
    # coordinates -- ``l`` decreases on sky images -- so the chunking is done
    # in INDEX space (np.arange) for every parallel axis: the resulting
    # data_selection / task_coords slices are index based and identical in
    # both spaces, and the node tasks only use those slices.
    index_xds = img_xds.assign_coords(
        {axis: np.arange(sky.sizes[axis], dtype=np.float64) for axis in parallel_axes}
    )
    parallel_coords = {
        parallel_axis: make_parallel_coord(
            coord=index_xds[parallel_axis], n_chunks=n_chunks[parallel_axis]
        )
        for parallel_axis in parallel_axes
    }
    n_tasks = int(
        np.prod([len(parallel_coords[axis]["data_chunks"]) for axis in parallel_axes])
    )
    logger.info(
        "Moments parallel chunks: "
        + ", ".join(
            f"{axis}={len(parallel_coords[axis]['data_chunks'])}"
            for axis in parallel_axes
        )
        + f" -> {n_tasks} node tasks (on-disk chunks "
        + str({axis: chunk_len[axis] for axis in parallel_axes})
        + ")"
    )
    T_determine_chunks = time.time() - start

    # --- Create the empty output image on disk --------------------------------
    # Coordinates are the input coordinates with the moment axis collapsed to
    # size 1; the per-moment data groups are registered on the store's attrs.
    start = time.time()
    moments_img_xds = xr.Dataset(
        coords=collapsed_moment_axis_coords(img_xds, moment_axis).coords
    )
    import copy

    moments_img_xds.attrs = copy.deepcopy(img_xds.attrs)
    moments_img_xds.attrs["data_groups"] = {}
    for name in moment_names:
        modify_data_groups_xds(
            moments_img_xds,
            data_group_out_name="moment_" + name,
            data_group_out={"sky": moment_data_variable_key(name).upper()},
            description=(
                f"Moment '{name}' of {sky_name} over the {moment_axis} axis "
                f"(immoments)."
            ),
        )
    # Drop encoding inherited from the source store: it can carry a compressor
    # spec Zarr v3's to_zarr rejects (mirrors the feather driver).
    for variable_name in moments_img_xds.variables:
        moments_img_xds[variable_name].encoding = {}
    write_image(
        moments_img_xds,
        imagename=moments_image_store,
        out_format="zarr",
        overwrite=overwrite,
    )

    # Pre-allocate the moment data variables (NaN-filled) so each map task can
    # lazily write its own slice in parallel.  Image-valued moments follow the
    # input image precision; coordinate-valued moments are always float64.
    single_precision_image = itemsize <= 4
    data_variable_definitions = get_moments_data_variable_definitions(
        moment_names, list(sky.dims), single_precision_image
    )
    sky_units = sky.attrs.get("units", "")
    axis_units = moment_axis_units(img_xds, moment_axis)
    for name in moment_names:
        units = moment_units(name, sky_units, axis_units)
        if units:
            data_variable_definitions[moment_data_variable_key(name)]["attrs"] = {
                "units": units
            }
    moments_data_variables = [moment_data_variable_key(name) for name in moment_names]
    create_empty_data_variables_on_disk(
        moments_image_store,
        moments_data_variables,
        shape_dict=dict(moments_img_xds.sizes),
        parallel_coords=parallel_coords,
        compressor=compressor,
        double_precision=not single_precision_image,
        data_variable_definitions=data_variable_definitions,
    )
    T_create_output = time.time() - start

    # --- Build and run the map graph ------------------------------------------
    # Interpolate in index space (see above); the graph's input_data only
    # carries coordinates, so either dataset works for the map call.
    input_data = {"img": index_xds}
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, input_data
    )

    input_params = {
        "input_image_store": input_image_store,
        "moments_image_store": moments_image_store,
        "moments": moment_names,
        "moment_axis": moment_axis,
        "image_data_group_in_name": image_data_group_in_name,
        "include_pixel_range": include_pixel_range,
        "exclude_pixel_range": exclude_pixel_range,
        "use_mask": use_mask,
        "selection": selection,
        "moments_data_variables": moments_data_variables,
        "memory_budget_gb": memory_budget_gb,
        # Full-image flags, validated here once; each node task slices them to
        # its chunk. Normalized so index-range lists reach workers as compact
        # bool arrays.
        "dimension_flags": normalize_dimension_flags(dimension_flags, dict(sky.sizes))
        or None,
    }

    start = time.time()
    viper_graph = map(
        input_data=input_data,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_moments_node_task,
        input_params=input_params,
        in_memory_compute=False,
        monitor_resources_seconds=monitor_resources_seconds,
    )
    dask_graph = generate_dask_workflow(viper_graph)
    logger.debug("Time to create moments graph " + str(time.time() - start) + "s")

    start = time.time()
    results = dask.compute(dask_graph)
    logger.info("Time to compute() moments " + str(time.time() - start) + "s")
    logger.debug(
        "Moments driver timings: determine chunks "
        + str(T_determine_chunks)
        + "s, create output "
        + str(T_create_output)
        + "s"
    )

    zarr.consolidate_metadata(moments_image_store)

    frames = []
    _collect_dataframes(results, frames)
    if frames:
        timing_df = pd.concat(frames, ignore_index=True)
        if "task_id" in timing_df:
            timing_df = timing_df.sort_values("task_id", ignore_index=True)
        return timing_df
    return pd.DataFrame()
