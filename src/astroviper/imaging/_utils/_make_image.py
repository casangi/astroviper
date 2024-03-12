def _make_image(input_params):
    import time
    from xradio.vis.load_processing_set import load_processing_set
    import graphviper.utils.logger as logger

    #print(input_params.keys())

    start_total = time.time()
    logger.debug(
        "Processing chunk " + str(input_params["task_id"]) 
    )

    start_0 = time.time()
    import numpy as np
    from astroviper._domain._visibility._phase_shift import _phase_shift_vis_ds
    from astroviper._domain._imaging._make_imaging_weights import _make_imaging_weights
    from astroviper._domain._imaging._make_gridding_convolution_function import (
        _make_gridding_convolution_function,
    )
    from astroviper._domain._imaging._make_aperture_grid import _make_aperture_grid
    from astroviper._domain._imaging._make_uv_sampling_grid import (
        _make_uv_sampling_grid,
    )
    from xradio.image import make_empty_sky_image
    from astroviper._domain._imaging._make_visibility_grid import _make_visibility_grid
    from astroviper._domain._imaging._fft_norm_img_xds import _fft_norm_img_xds

    from xradio.vis.load_processing_set import load_processing_set, processing_set_iterator
    import xarray as xr
    logger.debug("0. Imports " + str(time.time()-start_0))

    start_1 = time.time()
    grid_params = input_params["grid_params"]

    shift_params = {}
    shift_params["new_phase_direction"] = grid_params["phase_direction"]
    shift_params["common_tangent_reprojection"] = True
    image_freq_coord = input_params["task_coords"]["frequency"]["data"]

    if input_params["polarization"] is not None:
        image_polarization_coord = input_params["polarization"]
    else:
        image_polarization_coord = input_params["task_coords"]["polarization"]["data"]

    if input_params["time"] is not None:
        image_time_coord = input_params["time"]
    else:
        image_time_coord = input_params["task_coords"]["time"]["data"]

    img_xds = make_empty_sky_image(
        phase_center=grid_params["phase_direction"]["data"],
        image_size=grid_params["image_size"],
        cell_size=grid_params["cell_size"],
        chan_coords=image_freq_coord,
        pol_coords=image_polarization_coord,
        time_coords=image_time_coord,
    )
    img_xds.attrs["data_groups"] = {"mosaic": {}}

    logger.debug("1. Empty Image "+ str(time.time()-start_1))

    gcf_xds = xr.Dataset()
    T_compute =0.0
    T_load = 0.0
    T_phase_shift = 0.0
    T_weights = 0.0
    T_gcf = 0.0
    T_aperture_grid = 0.0
    T_uv_sampling_grid = 0.0
    T_vis_grid = 0.0

    ps_iter = processing_set_iterator(input_params["data_selection"], input_params["input_data_store"], input_params["input_data"])

    start_2 = time.time()
    for ms_xds in ps_iter:

        start_compute = time.time()
        start_3 = time.time()
        data_group_out = _phase_shift_vis_ds(
            ms_xds, shift_parms=shift_params, sel_parms={}
        )
        T_phase_shift = T_phase_shift + time.time() - start_3

        start_4 = time.time()
        data_group_out = _make_imaging_weights(
            ms_xds,
            grid_parms=grid_params,
            imaging_weights_parms={"weighting": "briggs", "robust": 0.6},
            sel_parms={"data_group_in": data_group_out},
        )
        T_weights = T_weights + time.time() - start_4

        start_5 = time.time()
        gcf_params = {}
        gcf_params["function"] = "casa_airy"
        gcf_params["list_dish_diameters"] = np.array([10.7])
        gcf_params["list_blockage_diameters"] = np.array([0.75])

        unique_ant_indx = ms_xds.attrs["antenna_xds"].DISH_DIAMETER.values
        unique_ant_indx[unique_ant_indx == 12.0] = 0

        gcf_params["unique_ant_indx"] = unique_ant_indx.astype(int)
        gcf_params["phase_direction"] = grid_params["phase_direction"]
        _make_gridding_convolution_function(
            gcf_xds,
            ms_xds,
            gcf_params,
            grid_params,
            sel_parms={"data_group_in": data_group_out},
        )
        T_gcf = T_gcf + time.time() - start_5

        start_6 = time.time()
        _make_aperture_grid(
            ms_xds,
            gcf_xds,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "mosaic"},
            grid_parms=grid_params,
        )
        T_aperture_grid = T_aperture_grid + time.time()-start_6

        start_7 = time.time()
        _make_uv_sampling_grid(
            ms_xds,
            gcf_xds,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "mosaic"},
            grid_parms=grid_params,
        ) #Will become the PSF.
        T_uv_sampling_grid = T_uv_sampling_grid + time.time() - start_7

        start_8 = time.time()
        _make_visibility_grid(
            ms_xds,
            gcf_xds,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "mosaic"},
            grid_parms=grid_params,
        )
        T_vis_grid = T_vis_grid + time.time() - start_8
        T_compute = T_compute + time.time() - start_compute

    T_load = time.time()-start_2-T_compute

    logger.debug("2. Load "+ str(T_load))
    logger.debug("3. Weights "+ str(T_weights))
    logger.debug("4. Phase_shift "+ str(T_phase_shift))
    logger.debug("5. make_gridding_convolution_function "+ str(T_uv_sampling_grid))
    logger.debug("6. Aperture grid "+ str(T_aperture_grid))
    logger.debug("7. UV sampling grid "+ str(T_uv_sampling_grid))
    logger.debug("8. Visibility grid "+ str(T_vis_grid))
    logger.debug("Compute "+ str(T_compute))


    start_9 = time.time()
    _fft_norm_img_xds(
        img_xds,
        gcf_xds,
        grid_params,
        norm_parms={},
        sel_parms={"data_group_in": "mosaic", "data_group_out": "mosaic"},
    )
    logger.debug("9. fft norm "+ str(time.time()-start_9))

    # Tranform uv-space -> lm-space (sky)

    start_10 = time.time()
    parallel_dims_chunk_id = dict(
        zip(input_params["parallel_dims"], input_params["chunk_indices"])
    )

    from xradio.image._util._zarr.zarr_low_level import (
        write_chunk
    )
    import os
  
    img_xds = img_xds.transpose('polarization','frequency',...).expand_dims(dim='dummy',axis=0)
    if input_params["to_disk"]:
        for data_variable, meta in input_params["zarr_meta"].items():
            write_chunk(img_xds,meta,parallel_dims_chunk_id,input_params["compressor"],input_params["image_file"])
    else:
        return img_xds
    logger.debug("10. to disk "+ str(time.time()-start_10))
    logger.debug("Completed task " + str(input_params['task_id']) + " in " + str(time.time()-start_total) + " s.")
    logger.debug("***"*20)
