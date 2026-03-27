def NT_image_cube_single_field(input_params,graph_mode=True):
    import toolviper.utils.logger as logger
    import time
    from xradio.image import make_empty_sky_image
    from toolviper.utils.memory_management import memory_setup, free_memory, get_rss_gb
    from astroviper.core.imaging.image_cube_single_field import PF_image_cube_single_field

    # Pin the mmap threshold BEFORE any large allocations so they use mmap
    # and are returned to the OS immediately on free (no heap fragmentation).
    # Must run at the start of the task, not after, or fragmentation is already done.
    memory_setup(131072)

    logger.debug(
        "Memory usage at start of NT_image_cube_single_field_node_task: "
        + str(get_rss_gb())
        + " GB"
    )
    
    #Handle initial stokes (transform to corr).

    image_params = input_params["image_params"]
    img_xds = make_empty_sky_image(
        phase_center=image_params["phase_direction"],
        image_size=image_params["image_size"],
        cell_size=image_params["cell_size"],
        frequency_coords=input_params["task_coords"]["frequency"]["data"],
        #pol_coords=image_params["polarization_coords"],
        pol_coords=["XX","YY"],
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
    if in_memory:
        from xradio.measurement_set.load_processing_set import load_processing_set
        ps_xdt = load_processing_set(input_params["input_data_store"], 
                                        sel_parms=input_params["data_selection"],
                                        data_group_name=input_params["processing_set_data_group_name"],
                                        load_sub_datasets=False,)  
    else:
        raise ValueError("in_memory=False is not currently supported.")
        # from xradio.measurement_set.open_processing_set import open_processing_set
        # ps_xdt = open_processing_set(...)
        # need to work on data selection.
    T_load = time.time() - start
    logger.debug("Time to load data " + str(T_load))

    
    logger.debug("Processing set iterator created with partitions.")
    img_xds, return_df = PF_image_cube_single_field(input_params, ps_xdt, img_xds)
    
    
    start_write = time.time()
    if graph_mode:
        from astroviper.utils.io import write_result_chunk_to_disk_using_zarr
        write_result_chunk_to_disk_using_zarr(input_params["image_store"],input_params["image_data_variables_keep"],input_params["task_coords"],img_xds)
    else:
        img_xds.to_zarr(input_params["image_store"], consolidated=True)
    T_write = time.time() - start_write
    logger.debug("Time to write to disk " + str(T_write))
    
    return_df["T_load"] = T_load
    return_df["T_write"] = T_write

    logger.debug(
        "Memory usage after completing node task, before releasing references: "
        + str(get_rss_gb())
        + " GB"
    )

    img_xds = None
    ps_xdt = None
    free_memory()

    logger.debug(
        "Memory usage after releasing references: " + str(get_rss_gb()) + " GB"
    )
    return return_df
