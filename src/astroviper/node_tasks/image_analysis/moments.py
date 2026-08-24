"""Node task for the moments (immoments) workflow.

Thin orchestration layer with a fully explicit, standalone-callable signature
(adapted to the graph calling convention by graphviper's
``make_graph_node_task``): it loads this task's chunk of the input image from
disk (only the sky and, optionally, mask variables -- nothing else is read
into memory), calls the science function
:func:`astroviper.processing_functions.image_analysis.moments`, and writes the
resulting moment maps into the pre-allocated Zarr store using the parallel
chunk-writer
:func:`astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.  This
mirrors the ``feather`` / ``image_cube_single_field`` node tasks.
"""

from astroviper.utils.param_docs import shares_param_docs


def _open_image_lazy(input_image_store):
    """Open an on-disk image lazily (metadata only; data stays on disk).

    Parameters
    ----------
    input_image_store : str
        Path to the input image (a ``.zarr`` store or any format
        ``xradio.image.open_image`` understands).

    Returns
    -------
    xarray.Dataset
        Lazily opened image dataset.
    """
    if "zarr" in input_image_store:
        import xarray as xr

        # chunks=None -> zarr-backed lazy arrays WITHOUT dask, so .load()
        # reads directly in this process. With the default (dask-backed)
        # arrays, .load() inside a distributed worker ships the chunk-read
        # graph to the ambient Client/scheduler -- tasks submitting tasks,
        # whose inner futures get cancelled (FutureCancelledError) when the
        # cluster is busy running the outer map tasks.
        return xr.open_zarr(input_image_store, chunks=None)
    from xradio.image import open_image

    return open_image(input_image_store)


def moment_axis_read_block(lazy_var, block_des, stream_axis, memory_budget_gb):
    """Number of ``stream_axis`` planes to read per request, from the on-disk
    chunk / shard geometry and a memory budget.

    A block is the unit read from the store (one zarr request) and streamed
    through the moments accumulator. Three constraints:

    * **Decode efficiency**: a partial read of an on-disk chunk decodes the
      WHOLE chunk, so a block should cover a whole number of chunk lengths
      along ``stream_axis`` (else the chunk is re-decoded once per block).
    * **Bounded decode transient**: zarr decodes the chunks of one request
      concurrently, and for SHARDED stores the concurrency limit applies per
      nesting level (shards x inner chunks per shard) -- a request spanning
      many shards can hold far more decoded chunks than the limit suggests
      (the 2026-08 MemoryErrors). A block therefore never exceeds ONE shard
      length along ``stream_axis``, so a request touches exactly one shard
      per on-disk chunk column of the task's tile.
    * **Memory**: the decoded block of this task's tile is at most
      ``memory_budget_gb``.

    Returns the shard length if it fits the budget, else the largest multiple
    of the chunk length that fits, else as many whole planes as fit (at least
    1); always clamped to the axis length. Non-zarr / unchunked stores count
    as chunk length 1 with no shards.
    """
    import numpy as np

    axis = lazy_var.dims.index(stream_axis)
    chunks = lazy_var.encoding.get("chunks")
    shards = lazy_var.encoding.get("shards")
    chunk_len = (
        max(1, int(chunks[axis]))
        if chunks is not None and len(chunks) == lazy_var.ndim
        else 1
    )
    shard_len = (
        max(chunk_len, int(shards[axis]))
        if shards is not None and len(shards) == lazy_var.ndim
        else None
    )
    selected = lazy_var.isel(block_des)
    n_planes = selected.sizes[stream_axis]
    plane_bytes = (
        float(np.prod([selected.sizes[d] for d in selected.dims if d != stream_axis]))
        * selected.dtype.itemsize
    )
    budget_bytes = max(memory_budget_gb, 0.0) * 1024**3
    planes_in_budget = int(budget_bytes // max(plane_bytes, 1))
    if shard_len is not None and planes_in_budget >= shard_len:
        block = shard_len
    elif planes_in_budget >= chunk_len:
        block = chunk_len * (planes_in_budget // chunk_len)
        if shard_len is not None:
            block = min(block, shard_len)
    else:
        # The budget cannot even hold one chunk length: whole planes that fit
        # (re-decoding the chunk per block; correct but slower) rather than
        # overrunning the budget.
        block = max(1, planes_in_budget)
    return max(1, min(block, n_planes))


def _make_plane_reader(lazy_xds, sky_name, mask_name, block_des, stream_axis, n_block):
    """Build ``read_planes(start, stop)`` over the lazily opened store.

    Each call reads ``[start, min(start + n_block, stop))`` of this task's
    chunk along ``stream_axis`` (the sky and, if present, the mask) directly
    from the zarr store into memory and returns them with ``stream_axis``
    first -- the contract of
    :func:`astroviper.processing_functions.image_analysis.moments.moments_streamed`.
    """
    import numpy as np

    sky = lazy_xds[sky_name].isel(block_des)
    mask = lazy_xds[mask_name].isel(block_des) if mask_name is not None else None
    axis = sky.dims.index(stream_axis)

    def read_planes(start, stop):
        stop = min(stop, start + n_block)
        sel = {stream_axis: slice(start, stop)}
        planes = np.moveaxis(sky.isel(sel).values, axis, 0)
        if mask is None:
            return planes, None
        if stream_axis in mask.dims:
            mask_planes = np.moveaxis(
                mask.isel(sel).values, mask.dims.index(stream_axis), 0
            )
        else:
            mask_planes = np.broadcast_to(mask.values, planes.shape)
        return planes, mask_planes

    return read_planes


def _load_chunk_streaming(lazy_xds, variables, block_des, stream_axis, n_block=1):
    """Load this task's whole chunk into memory, reading ``n_block`` planes
    of ``stream_axis`` at a time (the ``median`` path).

    ``median`` needs every pixel's full profile at once, so the full chunk
    slab (all ``stream_axis`` planes of this task's slice) must be held in
    memory; the distributed application sizes the chunks for that. Reading
    it in blocks (rather than one bulk ``.load()``) bounds the decode
    transient to the chunks touched by one block instead of letting the
    zarr sharding codec materialize whole shards.

    Returns an in-memory ``xarray.Dataset`` equivalent to
    ``lazy_xds[variables].isel(block_des).load()``.
    """
    import numpy as np
    import xarray as xr

    selected = lazy_xds[variables].isel(block_des)
    loaded = xr.Dataset(coords=selected.coords, attrs=selected.attrs)
    for name in variables:
        var = selected[name]
        if stream_axis not in var.dims:
            loaded[name] = var.load()
            continue
        axis = var.dims.index(stream_axis)
        out = np.empty(var.shape, dtype=var.dtype)
        index = [slice(None)] * var.ndim
        n = var.sizes[stream_axis]
        for start in range(0, n, n_block):
            index[axis] = slice(start, min(start + n_block, n))
            out[tuple(index)] = var.isel({stream_axis: index[axis]}).values
        loaded[name] = (var.dims, out)
        loaded[name].attrs = dict(var.attrs)
    return loaded


@shares_param_docs
def moments(
    input_image_store: str,
    moments_image_store: str,
    moments=["integrated"],  # noqa: B006 - mirrors the distributed application signature; never mutated
    moment_axis: str = "frequency",
    image_data_group_in_name: str = "base",
    include_pixel_range=None,
    exclude_pixel_range=None,
    use_mask: bool = False,
    selection=None,
    moments_data_variables=None,
    task_coords=None,
    data_selection=None,
    task_id=None,
    graph_mode: bool = True,
    memory_budget_gb: float = 1.0,
    dimension_flags=None,
):
    """Compute the moment maps of one image chunk and write them to Zarr.

    Memory strategy: every moment except ``median`` is computed by
    **streaming** the chunk along the moment axis straight from the store
    (:func:`~astroviper.processing_functions.image_analysis.moments.moments_streamed`):
    blocks of planes -- sized by :func:`moment_axis_read_block` to a whole
    number of on-disk chunk lengths that fit ``memory_budget_gb`` -- are read,
    folded into output-map-sized accumulators and dropped, so the task's
    memory is O(output maps) + one block, independent of the length of the
    moment axis. Only ``median`` (which needs every pixel's full profile)
    loads the whole chunk slab (:func:`_load_chunk_streaming`).

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
    moments_data_variables : list of str, optional
        Lowercase keys of the moment variables to write
        (e.g. ``["sky_moment_integrated"]``).  Defaults to the keys implied
        by ``moments``.
    task_coords : dict, optional
        Per-chunk coordinate mapping; ``task_coords[<parallel dim>]`` supplies
        this chunk's parallel coordinate values (``"data"``) and its
        ``"slice"`` into the full output array (for cube imaging the
        parallel dim is ``frequency``).
    data_selection : dict, optional
        Injected by the graph framework: ``{"img": {dim: slice}}`` selecting
        this task's chunk of the input image.
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    graph_mode : bool, default True
        If ``True`` the moment maps are written into the pre-allocated Zarr
        store with
        :func:`~astroviper.utils.io.write_result_chunk_to_disk_using_zarr`;
        if ``False`` the chunk dataset is returned instead of written (used
        for standalone testing).
    memory_budget_gb : float, default 1.0
        Target size (GiB) of one decoded read block of a node task's chunk
        along the moment axis (see
        :func:`~astroviper.node_tasks.image_analysis.moments.moment_axis_read_block`):
        a block spans a whole number of on-disk chunk lengths along the
        moment axis and is capped at this size, so a task's peak memory is
        roughly the output maps + this block + the zarr decode transient.
        Also enters the distributed application's per-chunk memory estimate.
        Ignored by the ``median`` slab path.

    Returns
    -------
    dict or xarray.Dataset
        ``{"timing_node_tasks": pandas.DataFrame}`` -- a one-row timing frame with ``T_load`` (streaming: time to open the
        store and plan the read; median: the slab load), ``T_moments``
        (streaming reads + accumulation), ``T_write`` and the total
        ``T_moments_task`` (plus ``task_id``, ``n_read_block_planes`` and
        ``streamed``) when ``graph_mode=True``; the per-chunk moments dataset
        when ``graph_mode=False``.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger
    from toolviper.utils.memory_management import free_memory, get_rss_gb

    # Import the science function directly from its module: the module and the
    # function share the name "moments", so the package attribute
    # pf.image_analysis.moments is ambiguous (module vs function) depending on
    # import order.
    from astroviper.processing_functions.image_analysis.moments import (
        moment_data_variable_key,
        moments_memory_model,
        moments_streamed,
        normalize_moments,
    )
    from astroviper.processing_functions.image_analysis.moments import (
        moments as moments_processing_function,
    )

    task_start = time.time()
    # No per-task allocator tuning (mallopt pins the mmap threshold and
    # disables glibc's dynamic adaptation -- the production policy since the
    # 2026-08 Frontera drift investigation is a clean allocator environment
    # set once per worker, not per task).
    logger.debug("Memory at start of moments node task: " + str(get_rss_gb()) + " GB")

    if selection is None:
        selection = {}
    if data_selection is None:
        data_selection = {"img": {}}

    # Load only this chunk's sky (and mask) variables into memory.  The needed
    # variable names are resolved on the lazily opened store (metadata only)
    # so no other data variable is ever read (memory efficiency).
    start = time.time()
    from astroviper.processing_functions.image_analysis.moments import (
        resolve_moments_input_variables,
    )

    block_des = {**selection, **data_selection["img"]}
    lazy_xds = _open_image_lazy(input_image_store)
    # dimension_flags are FULL-IMAGE flags (built once by the driver/caller);
    # restrict each dimension's flag vector to this task's chunk with the
    # same selection the data is loaded with.
    if dimension_flags:
        import numpy as np

        chunk_flags = {}
        for dim, flags in dimension_flags.items():
            flags = np.asarray(flags)
            if flags.dtype != bool:
                from astroviper.processing_functions.image_analysis.moments import (
                    normalize_dimension_flags,
                )

                flags = normalize_dimension_flags({dim: flags}, dict(lazy_xds.sizes))[
                    dim
                ]
            chunk_flags[dim] = flags[block_des[dim]] if dim in block_des else flags
        dimension_flags = chunk_flags

    sky_name, mask_name = resolve_moments_input_variables(
        lazy_xds, image_data_group_in_name, use_mask
    )
    variables = [sky_name] + ([mask_name] if mask_name is not None else [])
    moment_names = normalize_moments(moments)
    streamed = not moments_memory_model(moment_names)["requires_full_profile"]
    n_block = moment_axis_read_block(
        lazy_xds[sky_name], block_des, moment_axis, memory_budget_gb
    )
    img_xds = None
    if not streamed:
        # 'median' needs the whole profile: load the chunk slab (in blocks).
        img_xds = _load_chunk_streaming(
            lazy_xds, variables, block_des, moment_axis, n_block=n_block
        )
    T_load = time.time() - start
    logger.debug(
        f"moments task {task_id}: {'streaming' if streamed else 'slab'} path, "
        f"{n_block} plane(s) per read block"
    )

    start = time.time()
    if streamed:
        selected = lazy_xds[variables].isel(block_des)
        read_planes = _make_plane_reader(
            lazy_xds, sky_name, mask_name, block_des, moment_axis, n_block
        )
        moments_img_xds = moments_streamed(
            read_planes,
            coords_xds=selected,
            sky_dims=selected[sky_name].dims,
            sky_name=sky_name,
            sky_attrs=dict(selected[sky_name].attrs),
            attrs=selected.attrs,
            moments=moment_names,
            moment_axis=moment_axis,
            include_pixel_range=include_pixel_range,
            exclude_pixel_range=exclude_pixel_range,
            dimension_flags=dimension_flags,
        )
        selected = None
        read_planes = None
    else:
        moments_img_xds = moments_processing_function(
            img_xds,
            moments=moment_names,
            moment_axis=moment_axis,
            image_data_group_in_name=image_data_group_in_name,
            include_pixel_range=include_pixel_range,
            exclude_pixel_range=exclude_pixel_range,
            use_mask=use_mask,
            dimension_flags=dimension_flags,
        )
    lazy_xds = None
    T_moments = time.time() - start

    if not graph_mode:
        return moments_img_xds

    start = time.time()
    if moments_data_variables is None:
        moments_data_variables = [
            moment_data_variable_key(name) for name in normalize_moments(moments)
        ]
    from astroviper.utils.io import write_result_chunk_to_disk_using_zarr

    write_result_chunk_to_disk_using_zarr(
        moments_image_store,
        moments_data_variables,
        task_coords,
        moments_img_xds,
    )
    T_write = time.time() - start

    moments_img_xds = None
    img_xds = None
    free_memory()
    logger.debug("Memory after moments node task: " + str(get_rss_gb()) + " GB")

    # Execution identity + wall-clock anchor, mirroring the imaging node task:
    # start_unixtime places the task on the run timeline (task-stream /
    # cluster-usage plots); pid/thread/worker name give exact lane assignment.
    import os
    import socket
    import threading

    try:
        from dask.distributed import get_worker

        worker_name = str(get_worker().name)
    except Exception:
        worker_name = None
    timing_df = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "streamed": streamed,
                "n_read_block_planes": n_block,
                "start_unixtime": task_start,
                "T_load": T_load,
                "T_moments": T_moments,
                "T_write": T_write,
                "T_moments_task": time.time() - task_start,
                "hostname": socket.gethostname(),
                "process_pid": os.getpid(),
                "thread_native_id": threading.get_native_id(),
                "worker_name": worker_name,
            }
        ]
    )
    # Single-dict return convention (as the imaging node task) so graphviper's
    # optional resource monitor can attach its "resource_usage" series.
    return {"timing_node_tasks": timing_df}
