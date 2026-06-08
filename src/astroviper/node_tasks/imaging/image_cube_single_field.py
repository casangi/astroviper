def _remap_deconvolve_dict_to_global_channels(combined_deconvolve_dict, data_selection):
    """Shift a chunk-local deconvolve ReturnDict onto global channel numbers.

    The per-chunk ``image_cube_single_field`` labels channels ``0..N-1`` within
    the chunk. The global channel offset for this chunk is the start of the
    ``frequency`` slice in ``data_selection`` (frequency and channel are the
    same axis), e.g. ``{'ms_name': {'frequency': slice(2, 4)}}`` -> offset 2.

    Returns a new ReturnDict whose ``Key.chan`` values are global channel
    numbers. A no-op returning the input unchanged when the offset is 0 (e.g. a
    single chunk starting at channel 0) or no frequency slice is present.
    """
    from astroviper.processing_functions.imaging.return_dict import ReturnDict, Key

    chan_offset = 0
    for sel in (data_selection or {}).values():
        freq_sel = sel.get("frequency") if isinstance(sel, dict) else None
        if isinstance(freq_sel, slice) and freq_sel.start is not None:
            chan_offset = int(freq_sel.start)
            break

    if chan_offset == 0:
        return combined_deconvolve_dict

    remapped = ReturnDict()
    for key, value in combined_deconvolve_dict.data.items():
        remapped.data[Key(time=key.time, pol=key.pol, chan=key.chan + chan_offset)] = (
            value
        )
    return remapped


def image_cube_single_field(input_params, graph_mode=True):
    """Image one frequency chunk of a single-field cube and write it to disk.

    Thin node task (one ``input_params`` dict in, one ``return_dict`` out):
    pins the malloc mmap threshold, builds the empty per-chunk image, loads (or
    receives) this chunk's visibilities, runs the science
    :func:`~astroviper.processing_functions.imaging.image_cube_single_field.image_cube_single_field`,
    writes the result slice to the Zarr image store and returns the timing and
    deconvolution metadata.

    Parameters
    ----------
    input_params : dict
        Parameters injected by the graph framework (``task_coords``,
        ``data_selection``, ``task_id``, ``input_data``, ...) merged with the
        imaging parameters set by the driver (``image_params``, ``image_store``,
        ``image_data_variables_keep``, ``memory_mode``,
        ``processing_set_data_group_name``, ``deconvolver``, ...).
    graph_mode : bool, optional
        If ``True`` (the default) each kept variable's slice is written into the
        pre-allocated Zarr store with
        :func:`~astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.
        If ``False`` the whole chunk image is written with ``to_zarr``.

    Returns
    -------
    dict
        Single dict with two keys:

        * ``"timing"`` : one-row :class:`pandas.DataFrame` with a ``T_*`` column
          per processing function (load, image build, weights, PSF, primary
          beam, gridding, FFT normalization, degridding, deconvolution, write,
          ...) plus ``task_id``, ``n_channels``, ``n_major_cycles`` and the
          total ``T_image_cube_task``.
        * ``"deconvolution"`` : the per-plane deconvolution
          :class:`~astroviper.processing_functions.imaging.return_dict.ReturnDict`,
          with channels remapped to global channel numbers.
    """
    print("######### Starting image_cube_single_field_node_task ############")
    import time
    import toolviper.utils.logger as logger
    from xradio.image import make_empty_sky_image
    from toolviper.utils.memory_management import memory_setup, free_memory, get_rss_gb
    import astroviper.processing_functions as pf

    task_start = time.time()
    # Pin the mmap threshold BEFORE any large allocations so they use mmap and
    # are returned to the OS immediately on free (no heap fragmentation). Must
    # run at the start of the task, not after, or fragmentation is already done.
    memory_setup(131072)

    logger.debug(
        "Memory usage at start of image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    assert (
        input_params["memory_mode"] == "in_memory"
    ), "Currently only memory_mode='in_memory' is implemented."

    # Build the empty per-chunk image in the correlation basis the gridder works
    # in. NB: the correlation polarization basis is currently hard-coded to two
    # linear feeds ("XX", "YY"); the image is transformed to the Stokes output
    # basis (image_params["polarization_coords"]) inside the science function.
    image_params = input_params["image_params"]
    start = time.time()
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=input_params["task_coords"]["frequency"]["data"],
        pol_coords=["XX", "YY"],
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )
    T_make_empty_image = time.time() - start

    start = time.time()
    if input_params.get("input_data") is not None:
        # Data was pre-loaded by the data loading layer (disk-chunk granularity
        # I/O coalescing). The framework has already applied the task-level
        # sub-selection, so use the dict directly.
        ps_xdt = input_params["input_data"]
    else:
        from xradio.measurement_set.load_processing_set import load_processing_set

        ps_xdt = load_processing_set(
            input_params["input_data_store"],
            sel_parms=input_params["data_selection"],
            data_group_name=input_params["processing_set_data_group_name"],
            load_sub_datasets=False,
        )
    T_load = time.time() - start

    img_xds, timing_df, combined_deconvolve_dict = pf.imaging.image_cube_single_field(
        input_params, ps_xdt, img_xds
    )

    # The deconvolve dict's channels are chunk-local (0-based); remap them to
    # global channel numbers so the reduce can merge chunks correctly.
    combined_deconvolve_dict = _remap_deconvolve_dict_to_global_channels(
        combined_deconvolve_dict, input_params["data_selection"]
    )

    start = time.time()
    if graph_mode:
        from astroviper.utils.io import write_result_chunk_to_disk_using_zarr

        write_result_chunk_to_disk_using_zarr(
            input_params["image_store"],
            input_params["image_data_variables_keep"],
            input_params["task_coords"],
            img_xds,
        )
    else:
        img_xds.to_zarr(input_params["image_store"], consolidated=True)
    T_write = time.time() - start

    img_xds = None
    ps_xdt = None
    free_memory()

    logger.debug(
        "Memory usage after image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    # Fold the node-task timings (image build, load, write, total) into the
    # per-chunk timing frame produced by the science function.
    timing_df["T_make_empty_image"] = T_make_empty_image
    timing_df["T_load"] = T_load
    timing_df["T_write"] = T_write
    timing_df["T_image_cube_task"] = time.time() - task_start

    return {"timing": timing_df, "deconvolution": combined_deconvolve_dict}
