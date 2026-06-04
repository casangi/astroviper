from time import time


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
        remapped.data[
            Key(time=key.time, pol=key.pol, chan=key.chan + chan_offset)
        ] = value
    return remapped


def image_cube_single_field(input_params, graph_mode=True):
    import toolviper.utils.logger as logger
    import time
    from xradio.image import make_empty_sky_image
    from toolviper.utils.memory_management import memory_setup, free_memory, get_rss_gb    
    import astroviper.processing_functions as pf
    
    
    start = time.time()
    # Pin the mmap threshold BEFORE any large allocations so they use mmap
    # and are returned to the OS immediately on free (no heap fragmentation).
    # Must run at the start of the task, not after, or fragmentation is already done.
    memory_setup(131072)

    logger.debug(
        "Memory usage at start of image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )

    # Handle initial stokes (transform to corr).

    image_params = input_params["image_params"]
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=input_params["task_coords"]["frequency"]["data"],
        # pol_coords=image_params["polarization_coords"],
        pol_coords=["XX", "YY"],
        time_coords=image_params["time_coords"],
        do_sky_coords=False,
    )

    if input_params["memory_mode"] == "in_memory":
        in_memory = True
    else:
        in_memory = False

    assert (
        in_memory
    ), "Currently only in_memory is supported for memory_mode is implemented."

    start = time.time()
    if input_params.get("input_data") is not None:
        # Data was pre-loaded by the data loading layer (disk-chunk granularity
        # I/O coalescing).  The framework has already applied the task-level
        # sub-selection, so use the dict directly.
        ps_xdt = input_params["input_data"]
    elif in_memory:
        from xradio.measurement_set.load_processing_set import load_processing_set

        ps_xdt = load_processing_set(
            input_params["input_data_store"],
            sel_parms=input_params["data_selection"],
            data_group_name=input_params["processing_set_data_group_name"],
            load_sub_datasets=False,
        )
    else:
        raise ValueError("in_memory=False is not currently supported.")
        # from xradio.measurement_set.open_processing_set import open_processing_set
        # ps_xdt = open_processing_set(...)
        # need to work on data selection.
    T_load = time.time() - start
    
    print("&&&&"*10,input_params["data_selection"])

    img_xds, return_df, combined_deconvolve_dict = pf.imaging.image_cube_single_field(
        input_params, ps_xdt, img_xds
    )

    # The deconvolve dict's channels are chunk-local (0-based); remap them to
    # global channel numbers so the reduce can merge chunks correctly.
    combined_deconvolve_dict = _remap_deconvolve_dict_to_global_channels(
        combined_deconvolve_dict, input_params["data_selection"]
    )

    start_write = time.time()
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
    T_write = time.time() - start_write


    return_df["T_load"] = T_load
    return_df["T_write"] = T_write
    img_xds = None
    ps_xdt = None
    free_memory()
    
    logger.debug(
        "Memory usage after image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )
    
    image_cube_task_time = time.time() - start
    return_df["image_cube_task_time"] = image_cube_task_time
    
    import pandas as pd
    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None,
        "display.width", None,
        "display.float_format", "{:.6g}".format,
    ):
        print(return_df.to_string())



    return return_df, combined_deconvolve_dict
