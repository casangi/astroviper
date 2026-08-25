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

The moment axis itself can never be parallelized (every moment needs the full
axis), so the graph is chunked along another axis: ``frequency`` by default,
or ``m`` when the moment axis *is* ``frequency``.  The number of chunks is
derived from a per-chunk memory estimate that accounts for the streaming
science function's memory model (see
:mod:`astroviper.processing_functions.image_analysis.moments`).
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

# Axes the graph may be chunked along.  ``l`` is excluded because GraphVIPER's
# interpolate_data_coords_onto_parallel_coords requires a monotonically
# increasing coordinate and the ``l`` coordinate decreases; ``polarization``
# is excluded because its coordinate is non-numeric.
ALLOWED_PARALLEL_AXES = ("frequency", "m", "time")


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
    """Recursively collect the node tasks' timing DataFrames from the compute result."""
    import pandas as pd

    if isinstance(results, pd.DataFrame):
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
    parallel_axis: str | None = None,
    n_chunks: int | None = None,
    thread_info: dict | None = None,
    compressor=None,
    overwrite: bool = False,
):
    """Collapse an on-disk image along one axis into moment maps (CASA ``immoments``).

    The output is a single Zarr image store holding one ``SKY_MOMENT_<NAME>``
    data variable (and one ``moment_<name>`` data group) per requested moment,
    with the moment axis kept as a degenerate dimension of size 1 whose
    numeric coordinates are the mean of the input coordinates.  The compute is
    parallelized with a GraphVIPER map graph over ``parallel_axis`` (never the
    moment axis); each node task loads only its own chunk of the sky (and
    optional mask) variables and writes its own slice of every moment map.

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
    parallel_axis : str, optional
        Image dimension the graph is chunked along; never the moment axis.
        One of ``"frequency"``, ``"m"`` or ``"time"``.  Default (``None``):
        ``"frequency"``, or ``"m"`` when the moment axis is ``"frequency"``.
    n_chunks : int, optional
        Number of chunks along ``parallel_axis``.  If ``None`` (default), the
        chunk count is auto-determined from the per-chunk memory estimate and
        the available threads/memory.
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

    Returns
    -------
    pandas.DataFrame
        Per-task timing frame (``task_id``, ``T_load``, ``T_moments``,
        ``T_write``, ``T_moments_task``), one row per graph node.

    Raises
    ------
    ValueError
        If a moment / axis choice is invalid, both pixel ranges are given, or
        ``selection`` selects along ``parallel_axis``.
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

    # --- Choose and validate the parallel axis --------------------------------
    if parallel_axis is None:
        parallel_axis = "m" if moment_axis == "frequency" else "frequency"
    if parallel_axis == moment_axis:
        raise ValueError(
            f"The moment axis '{moment_axis}' cannot be used for parallelism; "
            f"choose a different parallel_axis."
        )
    if parallel_axis not in ALLOWED_PARALLEL_AXES:
        raise ValueError(
            f"parallel_axis '{parallel_axis}' not in allowed axes "
            f"{ALLOWED_PARALLEL_AXES} (the parallel coordinate must be numeric "
            "and monotonically increasing)."
        )
    if parallel_axis not in sky.dims:
        raise ValueError(
            f"parallel_axis '{parallel_axis}' is not a dimension of {sky_name}."
        )
    if parallel_axis in selection:
        raise ValueError(
            "selection must not select along the parallel_axis "
            f"('{parallel_axis}'); chunk indices are computed on the selected "
            "image, but node tasks apply the selection and the chunk slices to "
            "the on-disk image independently."
        )

    # --- Determine chunking along the parallel axis ---------------------------
    # Memory for a parallel-axis-size-1 chunk: the loaded sky slab (full moment
    # axis), the optional mask, plus the science function's extra copies (one
    # working copy + numpy's internal partition copy when a median-family
    # moment filters pixels; see the processing-function module docstring).
    # The streaming accumulators are map-sized and covered by the fudge factor.
    start = time.time()
    itemsize = sky.dtype.itemsize
    singleton_chunk_sizes = dict(sky.sizes)
    del singleton_chunk_sizes[parallel_axis]
    n_copies = 1.0
    if mask_name is not None:
        n_copies += 1.0 / itemsize
    if "median" in moment_names:
        filtering = include_range is not None or exclude_range is not None or use_mask
        n_copies += 2.0 if filtering else 1.0
    fudge_factor = 1.3
    memory_singleton_chunk = (
        n_copies
        * float(np.prod(np.array(list(singleton_chunk_sizes.values()))))
        * itemsize
        * fudge_factor
        / (1024**3)
    )
    if thread_info is None:
        thread_info = get_thread_info()
    logger.debug("Thread info " + str(thread_info))
    if n_chunks is None:
        n_chunks_dict = calculate_data_chunking(
            memory_singleton_chunk,
            {parallel_axis: sky.sizes[parallel_axis]},
            thread_info,
            constant_memory=0,
            # Moment tasks are light (one streaming reduction per chunk), so
            # per-task overhead matters: one task per thread balances the graph
            # without drowning small cubes in scheduling/IO overhead.
            tasks_per_thread=1,
        )
        n_chunks = n_chunks_dict[parallel_axis]

    parallel_coords = {
        parallel_axis: make_parallel_coord(
            coord=img_xds[parallel_axis], n_chunks=n_chunks
        )
    }
    logger.info(
        "Number of "
        + parallel_axis
        + " chunks: "
        + str(len(parallel_coords[parallel_axis]["data_chunks"]))
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
    input_data = {"img": img_xds}
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
    }

    start = time.time()
    viper_graph = map(
        input_data=input_data,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_moments_node_task,
        input_params=input_params,
        in_memory_compute=False,
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
