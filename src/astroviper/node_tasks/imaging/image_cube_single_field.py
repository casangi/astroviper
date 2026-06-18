from astroviper.utils.param_docs import shares_param_docs


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
    from astroviper.processing_functions.imaging.utils.return_dict import (
        ReturnDict,
        Key,
    )

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


@shares_param_docs
def image_cube_single_field(
    image_params,
    imaging_weights_params,
    iteration_control_params,
    task_coords,
    data_selection,
    image_store,
    input_data_store,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    restore=False,
    memory_mode="in_memory",
    skunk_works=False,
    data_group=None,
    task_id=0,
    input_data=None,
    graph_mode=True,
):
    """Image one frequency chunk of a single-field cube and write it to disk.

    Thin node task: pins the malloc mmap threshold, builds the empty per-chunk
    image in the correlation (instrument) polarization basis, loads (or receives)
    this chunk's visibilities, runs the science
    :func:`~astroviper.processing_functions.imaging.image_cube_single_field.image_cube_single_field`,
    writes the result slice to the Zarr image store, and returns the timing and
    deconvolution metadata.

    This function has a fully spelled-out signature so it can be called directly
    (standalone) outside of a graph.  When driven by
    :func:`graphviper.graph_tools.map.map`, graphviper adapts it automatically
    (via :func:`graphviper.graph_tools.map.make_graph_node_task`), expanding the
    single ``input_params`` dict it passes into these keyword arguments.

    Parameters
    ----------
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    iteration_control_params : dict
        CLEAN iteration controls: ``niter``, ``nmajor``, ``threshold`` (the
        deconvolver stopping threshold), ``primary_beam_limit`` (primary-beam
        mask cutoff as a fraction of the peak primary beam, distinct from
        ``threshold``), ``gain``, ``cyclefactor``, ``cycleniter``,
        ``minpsffraction`` and ``maxpsffraction``.
    task_coords : dict
        Per-chunk coordinate mapping; ``task_coords["frequency"]["data"]``
        supplies this chunk's frequency axis.
    data_selection : dict
        Per-chunk ``{ms_name: {dim: slice}}`` selection injected by graphviper;
        used to load this chunk's data and to remap chunk-local channel numbers
        to global ones.
    image_store : str
        Path/URL of the on-disk Zarr image cube.
    input_data_store : str
        Path/URL of the processing-set Zarr store to load this chunk's
        visibilities from (used only when ``input_data`` is ``None``).
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    deconvolver : str, optional
        Deconvolution algorithm for the minor cycle. One of ``"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``
        (numba, threaded across *and* within planes -- faster when there are
        few planes, e.g. single-channel imaging) or ``"asp"``.
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` (``XX``/``YY``) or ``"circular"`` (``RR``/``LL``). The
        output image is always produced in the Stokes basis.
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / Numba /
        FFT) kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    restore : bool, optional
        If ``True`` produce a restored image after deconvolution: the model
        convolved with the clean beam (the Gaussian fit to the PSF) plus the
        residual, written to the ``sky_restored`` (``SKY_RESTORED``) variable.
    memory_mode : str, optional
        Only ``"in_memory"`` is implemented.  Default ``"in_memory"``.
    skunk_works : bool, optional
        If ``True`` use the experimental performance I/O path: load only the
        data group's data variables straight from the Zarr chunk blobs with
        :func:`~astroviper.node_tasks.imaging.utils.load_processing_set_skunk_works`
        (reconstructing coordinates from the inputs) and write each result chunk
        with
        :func:`~astroviper.node_tasks.imaging.utils.write_result_chunk_to_disk_using_zarr_skunk_works`.
        Both the skunk-works load and write spread their per-array / per-variable
        I/O and (de)compression concurrently across ``processing_function_threads``
        threads.  Requires ``data_group``.  Default ``False`` (production I/O).
    data_group : dict, optional
        Resolved role->variable mapping for ``processing_set_data_group_name``
        (e.g. ``{"correlated_data": "VISIBILITY", "uvw": "UVW", ...}``), supplied
        by the distributed graph; only used when ``skunk_works`` is ``True``.
    task_id : int, optional
        Identifier of the frequency chunk being imaged.
    input_data : dict, optional
        Pre-loaded data for this chunk (supplied by the data-loading layer); when
        ``None`` (default) the data is loaded from ``input_data_store``.
    graph_mode : bool, optional
        If ``True`` (default) each kept variable's slice is written into the
        pre-allocated Zarr store with
        :func:`~astroviper.utils.io.write_result_chunk_to_disk_using_zarr`.  If
        ``False`` the whole chunk image is written with ``to_zarr``.

    Returns
    -------
    dict
        Single dict with two keys:

        * ``"timing_node_tasks"`` : one-row :class:`pandas.DataFrame` with a
          ``T_*`` column per processing function (load, image build, weights,
          PSF, primary beam, gridding, FFT normalization, degridding,
          deconvolution, write, ...) plus ``task_id``, ``n_channels``,
          ``n_major_cycles`` and the total ``T_image_cube_task``.
        * ``"deconvolution"`` : the per-plane deconvolution
          :class:`~astroviper.processing_functions.imaging.utils.return_dict.ReturnDict`,
          with channels remapped to global channel numbers.
    """
    print("######### Starting image_cube_single_field_node_task ############")
    import time
    import toolviper.utils.logger as logger
    from xradio.image import make_empty_sky_image
    from toolviper.utils.memory_management import get_rss_gb
    import astroviper.processing_functions as pf

    task_start = time.time()

    logger.debug(
        "Memory usage at start of image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    assert (
        memory_mode == "in_memory"
    ), "Currently only memory_mode='in_memory' is implemented."

    if image_data_variables_keep is None:
        image_data_variables_keep = [
            "sky_residual",
            "point_spread_function",
            "primary_beam",
        ]

    # Build the empty per-chunk image in the correlation (instrument)
    # polarization basis the gridder works in. The two-feed correlation labels
    # follow ``instrument_polarization_basis`` ("linear" -> XX/YY,
    # "circular" -> RR/LL); the image is transformed to the Stokes output basis
    # (image_params["polarization_coords"]) inside the science function.
    correlation_pol_coords = {
        "linear": ["XX", "YY"],
        "circular": ["RR", "LL"],
    }[instrument_polarization_basis]
    start = time.time()
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=task_coords["frequency"]["data"],
        pol_coords=correlation_pol_coords,
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )
    T_make_empty_image = time.time() - start

    start = time.time()
    if input_data is not None:
        # Data was pre-loaded by the data loading layer (disk-chunk granularity
        # I/O coalescing). The framework has already applied the task-level
        # sub-selection, so use the dict directly.
        ps_xdt = input_data
    elif skunk_works:
        # Experimental performance path: read only this chunk's data-group
        # variables straight from the Zarr chunk blobs and reconstruct the
        # coordinates from the inputs (no datatree/coords/sub-datasets open).
        from astroviper.node_tasks.imaging.utils import (
            load_processing_set_skunk_works,
        )

        ps_xdt = load_processing_set_skunk_works(
            input_data_store,
            sel_parms=data_selection,
            data_group=data_group,
            processing_set_data_group_name=processing_set_data_group_name,
            frequency_coords=task_coords["frequency"]["data"],
            instrument_polarization_basis=instrument_polarization_basis,
            num_threads=processing_function_threads,
        )
    else:
        from xradio.measurement_set.load_processing_set import load_processing_set

        ps_xdt = load_processing_set(
            input_data_store,
            sel_parms=data_selection,
            data_group_name=processing_set_data_group_name,
            load_sub_datasets=False,
        )
    T_load = time.time() - start

    img_xds, timing_df, combined_deconvolve_dict = pf.imaging.image_cube_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        iteration_control_params,
        processing_set_data_group_name=processing_set_data_group_name,
        deconvolver=deconvolver,
        instrument_polarization_basis=instrument_polarization_basis,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
        restore=restore,
        task_id=task_id,
    )

    # The deconvolve dict's channels are chunk-local (0-based); remap them to
    # global channel numbers so the reduce can merge chunks correctly.
    combined_deconvolve_dict = _remap_deconvolve_dict_to_global_channels(
        combined_deconvolve_dict, data_selection
    )

    start = time.time()
    if graph_mode and skunk_works:
        # Experimental performance path: encode and write only this chunk's
        # blob(s) directly to the pre-created Zarr image store (no open_group).
        from astroviper.node_tasks.imaging.utils import (
            write_result_chunk_to_disk_using_zarr_skunk_works,
        )

        write_result_chunk_to_disk_using_zarr_skunk_works(
            image_store,
            image_data_variables_keep,
            task_coords,
            img_xds,
            num_threads=processing_function_threads,
        )
    elif graph_mode:
        from astroviper.utils.io import write_result_chunk_to_disk_using_zarr

        write_result_chunk_to_disk_using_zarr(
            image_store,
            image_data_variables_keep,
            task_coords,
            img_xds,
        )
    else:
        img_xds.to_zarr(image_store, consolidated=True)
    T_write = time.time() - start

    img_xds = None
    ps_xdt = None

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

    # Debug: phase-grouped timing breakdown for this chunk. The generic
    # formatter lives in the top-level utils; the imaging phase layout
    # parameterizes it.
    from astroviper.utils.timing import print_timing_summary
    from astroviper.processing_functions.imaging.utils import (
        IMAGING_TIMING_PHASES,
        IMAGING_TIMING_TOTAL_KEY,
    )

    print_timing_summary(
        timing_df,
        IMAGING_TIMING_PHASES,
        total_key=IMAGING_TIMING_TOTAL_KEY,
        printer=logger.debug,
    )

    return {"timing_node_tasks": timing_df, "deconvolution": combined_deconvolve_dict}
