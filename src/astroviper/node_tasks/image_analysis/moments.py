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

        return xr.open_zarr(input_image_store)
    from xradio.image import open_image

    return open_image(input_image_store)


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
):
    """Compute the moment maps of one image chunk and write them to Zarr.

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

    Returns
    -------
    pandas.DataFrame or xarray.Dataset
        One-row timing frame with ``T_load``, ``T_moments``, ``T_write`` and
        the total ``T_moments_task`` (plus ``task_id``) when
        ``graph_mode=True``; the per-chunk moments dataset when
        ``graph_mode=False``.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger
    from toolviper.utils.memory_management import free_memory, get_rss_gb, memory_setup

    # Import the science function directly from its module: the module and the
    # function share the name "moments", so the package attribute
    # pf.image_analysis.moments is ambiguous (module vs function) depending on
    # import order.
    from astroviper.processing_functions.image_analysis.moments import (
        moment_data_variable_key,
        normalize_moments,
    )
    from astroviper.processing_functions.image_analysis.moments import (
        moments as moments_processing_function,
    )

    task_start = time.time()
    # Pin the mmap threshold before any large allocation so big buffers are
    # returned to the OS on free (mirrors the imaging node tasks).
    memory_setup(131072)
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
    sky_name, mask_name = resolve_moments_input_variables(
        lazy_xds, image_data_group_in_name, use_mask
    )
    variables = [sky_name] + ([mask_name] if mask_name is not None else [])
    img_xds = lazy_xds[variables].isel(block_des).load()
    lazy_xds = None
    T_load = time.time() - start

    start = time.time()
    moments_img_xds = moments_processing_function(
        img_xds,
        moments=moments,
        moment_axis=moment_axis,
        image_data_group_in_name=image_data_group_in_name,
        include_pixel_range=include_pixel_range,
        exclude_pixel_range=exclude_pixel_range,
        use_mask=use_mask,
    )
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

    return pd.DataFrame(
        [
            {
                "task_id": task_id,
                "T_load": T_load,
                "T_moments": T_moments,
                "T_write": T_write,
                "T_moments_task": time.time() - task_start,
            }
        ]
    )
