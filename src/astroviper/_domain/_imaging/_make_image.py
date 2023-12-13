def _make_image(input_parms):
    import time
    from xradio.vis.load_processing_set import load_processing_set

    start_total = time.time()
    # from astroviper._utils._logger import _get_logger
    # logger = _get_logger()
    # logger.debug(
    #     "Processing chunk " + str(input_parms["chunk_id"]) + " " + str(logger.level)
    # )

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

    from xradio.vis.load_processing_set import load_processing_set
    import xarray as xr
    # print("0. Imports",time.time()-start_0)

    start_1 = time.time()
    grid_parms = input_parms["grid_parms"]

    shift_parms = {}
    shift_parms["new_phase_direction"] = grid_parms["phase_direction"]
    shift_parms["common_tangent_reprojection"] = True

    if input_parms["grid_parms"]["frequency"] is not None:
        image_freq_coord = input_parms["grid_parms"]["frequency"]
    else:
        image_freq_coord = input_parms["task_coords"]["frequency"]["data"]

    if input_parms["grid_parms"]["polarization"] is not None:
        image_polarization_coord = input_parms["grid_parms"]["polarization"]
    else:
        image_polarization_coord = input_parms["task_coords"]["polarization"]["data"]

    if input_parms["grid_parms"]["time"] is not None:
        image_time_coord = input_parms["grid_parms"]["time"]
    else:
        image_time_coord = input_parms["task_coords"]["time"]["data"]

    img_xds = make_empty_sky_image(
        phase_center=grid_parms["phase_direction"]["data"],
        image_size=grid_parms["image_size"],
        cell_size=grid_parms["cell_size"],
        chan_coords=image_freq_coord,
        pol_coords=image_polarization_coord,
        time_coords=image_time_coord,
    )
    img_xds.attrs["data_groups"] = {"mosaic": {}}

    # print("1. Empty Image",time.time()-start_1)

    gcf_xds = xr.Dataset()



    for ms_v4_name, slice_description in input_parms["data_selection"].items():
        start_2 = time.time()
 
        if input_parms["input_data"] is None:
            ps = load_processing_set(
                    ps_name=input_parms["input_data_store"],
                    sel_parms={ms_v4_name: slice_description},
                )
            ms_xds = ps.get(0)
        else:
            img_xds = input_parms["input_data"][ms_v4_name] #In memory

        # print("2. Load",time.time()-start_2)

        start_3 = time.time()
        data_group_out = _phase_shift_vis_ds(
            ms_xds, shift_parms=shift_parms, sel_parms={}
        )
        # print("3. phase_shift",time.time()-start_3)

        start_4 = time.time()
        data_group_out = _make_imaging_weights(
            ms_xds,
            grid_parms=grid_parms,
            imaging_weights_parms={"weighting": "briggs", "robust": 0.6},
            sel_parms={"data_group_in": data_group_out},
        )

        gcf_parms = {}
        gcf_parms["function"] = "casa_airy"
        gcf_parms["list_dish_diameters"] = np.array([10.7])
        gcf_parms["list_blockage_diameters"] = np.array([0.75])

        # print("4. Phase_shift ",time.time()-start_4)

        start_4_1 = time.time()
        # print(ms_xds.attrs['antenna_xds'])
        unique_ant_indx = ms_xds.attrs["antenna_xds"].DISH_DIAMETER.values
        unique_ant_indx[unique_ant_indx == 12.0] = 0

        gcf_parms["unique_ant_indx"] = unique_ant_indx.astype(int)
        gcf_parms["phase_direction"] = grid_parms["phase_direction"]
        _make_gridding_convolution_function(
            gcf_xds,
            ms_xds,
            gcf_parms,
            grid_parms,
            sel_parms={"data_group_in": data_group_out},
        )
        # print("4.1 make_gridding_convolution_function ",time.time()-start_4_1)

        start_4_2 = time.time()
        _make_aperture_grid(
            ms_xds,
            gcf_xds,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "mosaic"},
            grid_parms=grid_parms,
        )

        _make_uv_sampling_grid(
            ms_xds,
            gcf_xds,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "mosaic"},
            grid_parms=grid_parms,
        )

        _make_visibility_grid(
            ms_xds,
            gcf_xds,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "mosaic"},
            grid_parms=grid_parms,
        )

        # print("4.2 rest ",time.time()-start_4_2)
    

    start_5 = time.time()
    _fft_norm_img_xds(
        img_xds,
        gcf_xds,
        grid_parms,
        norm_parms={},
        sel_parms={"data_group_in": "mosaic", "data_group_out": "mosaic"},
    )
    # print("5. fft norm",time.time()-start_5)

    # Tranform uv-space -> lm-space (sky)

    start_6 = time.time()
    parallel_dims_chunk_id = dict(
        zip(input_parms["parallel_dims"], input_parms["chunk_indices"])
    )

    from xradio.image._util._zarr.zarr_low_level import (
        pad_array_with_nans,
        write_binary_blob_to_disk,
    )
    import os

    if input_parms["to_disk"]:
        for data_varaible, meta in input_parms["zarr_meta"].items():
            dims = meta["dims"]
            dtype = meta["dtype"]
            data_varaible_name = meta["name"]
            chunks = meta["chunks"]
            shape = meta["shape"]
            chunk_name = ""
            if data_varaible_name in img_xds:
                for d in img_xds[data_varaible_name].dims:
                    if d in parallel_dims_chunk_id:
                        chunk_name = chunk_name + str(parallel_dims_chunk_id[d]) + "."
                    else:
                        chunk_name = chunk_name + "0."
                chunk_name = chunk_name[:-1]

                if list(img_xds[data_varaible_name].shape) != list(chunks):
                    array = pad_array_with_nans(
                        img_xds[data_varaible_name].values,
                        output_shape=chunks,
                        dtype=dtype,
                    )
                else:
                    array = img_xds[data_varaible_name].values

                write_binary_blob_to_disk(
                    array,
                    file_path=os.path.join(
                        input_parms["image_file"], data_varaible_name, chunk_name
                    ),
                    compressor=input_parms["compressor"],
                )
    else:
        return img_xds
    # print("6. to disk",time.time()-start_6)
    # print("Done chunk",input_parms['chunk_id'],time.time()-start_total)
    # print("***"*20)
