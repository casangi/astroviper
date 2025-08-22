def make_image_single_field(input_params):
    import time
    from xradio.correlated_data.load_processing_set import ProcessingSetIterator
    import toolviper.utils.logger as logger
    import dask

    # #print(input_params.keys())

    start_total = time.time()
    logger.debug("Processing chunk " + str(input_params["task_id"]))

    # dask.distributed.print('2 Input data ',input_params['task_id'], input_params['chunk_indices'], ' ***** ')

    start_0 = time.time()
    import numpy as np
    from astroviper.core.imaging._make_imaging_weights import make_imaging_weights
    from astroviper.core.imaging._make_uv_sampling_grid import (
        make_uv_sampling_grid_single_field,
    )
    from xradio.image import make_empty_sky_image
    from astroviper.core.imaging._make_visibility_grid import (
        make_visibility_grid_single_field,
    )
    from astroviper.core.imaging.fft_norm_img_xds import fft_norm_img_xds

    import xarray as xr

    logger.debug("0. Imports " + str(time.time() - start_0))

    start_1 = time.time()
    grid_params = input_params["grid_params"]

    image_freq_coord = input_params["task_coords"]["frequency"]["data"]

    if input_params["polarization"] is not None:
        image_polarization_coord = input_params["polarization"]
    else:
        image_polarization_coord = input_params["task_coords"]["polarization"]["data"]

    if input_params["time"] is not None:
        image_time_coord = input_params["time"]
    else:
        image_time_coord = input_params["task_coords"]["time"]["data"]

    logger.debug("Creating empty image ")
    img_xds = make_empty_sky_image(
        phase_center=grid_params["phase_direction"].values,
        image_size=grid_params["image_size"],
        cell_size=grid_params["cell_size"],
        chan_coords=image_freq_coord,
        pol_coords=image_polarization_coord,
        time_coords=image_time_coord,
    )
    img_xds.attrs["data_groups"] = {"single_field": {}}

    T_empty_image = time.time() - start_1
    logger.debug("1. Empty Image " + str(time.time() - start_1))

    T_compute = 0.0
    T_load = 0.0
    T_weights = 0.0
    T_gcf = 0.0
    T_aperture_grid = 0.0
    T_uv_sampling_grid = 0.0
    T_vis_grid = 0.0

    data_group = input_params["data_group"]
    if data_group == "base":
        data_variables = ["FLAG", "UVW", "VISIBILITY", "WEIGHT"]
    elif data_group == "corrected":
        data_variables = ["FLAG", "UVW", "VISIBILITY_CORRECTED", "WEIGHT"]

    ps_iter = ProcessingSetIterator(
        input_params["data_selection"],
        input_params["input_data_store"],
        input_params["input_data"],
        data_variables=data_variables,
        load_sub_datasets=True,
    )
    logger.debug("1.5 Created ProcessingSetIterator ")

    from astroviper.core._imaging._imaging_utils.gcf_prolate_spheroidal import (
        _create_prolate_spheroidal_kernel_1D,
    )

    cgk_1D = _create_prolate_spheroidal_kernel_1D(100, 7)

    start_2 = time.time()
    for ms_xds in ps_iter:
        start_compute = time.time()

        # Create a mask where baseline_antenna1_name does not equal baseline_antenna2_name
        mask = ms_xds["baseline_antenna1_name"] != ms_xds["baseline_antenna2_name"]
        # Apply the mask to the Dataset
        ms_xds = ms_xds.where(mask, drop=True)

        start_4 = time.time()
        data_group_out = make_imaging_weights(
            ms_xds,
            grid_parms=grid_params,
            imaging_weights_parms={"weighting": "briggs", "robust": 0.6},
            sel_parms={"data_group_in": data_group},
        )
        T_weights = T_weights + time.time() - start_4

        start_7 = time.time()
        make_uv_sampling_grid_single_field(
            ms_xds,
            cgk_1D,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "single_field"},
            grid_parms=grid_params,
        )  # Will become the PSF.
        T_uv_sampling_grid = T_uv_sampling_grid + time.time() - start_7

        start_8 = time.time()
        make_visibility_grid_single_field(
            ms_xds,
            cgk_1D,
            img_xds,
            vis_sel_parms={"data_group_in": data_group_out},
            img_sel_parms={"data_group_in": "single_field"},
            grid_parms=grid_params,
        )
        T_vis_grid = T_vis_grid + time.time() - start_8
        T_compute = T_compute + time.time() - start_compute

    # print(img_xds)
    from astroviper.core.imaging._imaging_utils.make_pb_symmetric import (
        airy_disk_rorder,
    )

    pb_parms = {}
    pb_parms["list_dish_diameters"] = np.array([10.7])
    pb_parms["list_blockage_diameters"] = np.array([0.75])
    pb_parms["ipower"] = 1

    grid_params["image_center"] = (np.array(grid_params["image_size"]) // 2).tolist()
    # (1, 1, len(pol), 1, 1))
    # print(_airy_disk_rorder(ms_xds.frequency.values, ms_xds.polarization.values, pb_parms, grid_params).shape)

    # img_xds["PRIMARY_BEAM"] =  xr.DataArray(_airy_disk_rorder(ms_xds.frequency.values, ms_xds.polarization.values, pb_parms, grid_params)[0,...], dims=("frequency", "polarization", "l", "m"))
    img_xds["PRIMARY_BEAM"] = xr.DataArray(
        np.ones(img_xds.UV_SAMPLING.shape), dims=("frequency", "polarization", "l", "m")
    )

    T_load = time.time() - start_2 - T_compute

    logger.debug("2. Load " + str(T_load))
    logger.debug("3. Weights " + str(T_weights))
    logger.debug("5. make_gridding_convolution_function " + str(T_gcf))
    logger.debug("6. Aperture grid " + str(T_aperture_grid))
    logger.debug("7. UV sampling grid " + str(T_uv_sampling_grid))
    logger.debug("8. Visibility grid " + str(T_vis_grid))
    logger.debug("Compute " + str(T_compute))

    start_9 = time.time()

    gcf_xds = xr.Dataset()
    gcf_xds.attrs["oversampling"] = [100, 100]
    gcf_xds.attrs["SUPPORT"] = [7, 7]
    from astroviper.core._imaging._imaging_utils.gcf_prolate_spheroidal import (
        _create_prolate_spheroidal_kernel,
    )

    _, ps_corr_image = _create_prolate_spheroidal_kernel(
        100, 7, n_uv=img_xds["UV_SAMPLING"].shape[-2:]
    )

    # print(ps_corr_image.shape)
    gcf_xds["PS_CORR_IMAGE"] = xr.DataArray(ps_corr_image, dims=("l", "m"))

    fft_norm_img_xds(
        img_xds,
        gcf_xds,
        grid_params,
        norm_parms={},
        sel_parms={"data_group_in": "single_field", "data_group_out": "single_field"},
    )
    T_fft = time.time() - start_9
    logger.debug("9. fft norm " + str(time.time() - start_9))

    # # Tranform uv-space -> lm-space (sky)

    start_10 = time.time()
    parallel_dims_chunk_id = dict(
        zip(input_params["parallel_dims"], input_params["chunk_indices"])
    )

    from xradio.image._util._zarr.zarr_low_level import write_chunk
    import os

    # dask.distributed.print('Writing results to Lustre for task ',input_params['task_id'],' ***** ')
    img_xds = img_xds.transpose("polarization", "frequency", ...).expand_dims(
        dim="dummy", axis=0
    )

    if input_params["to_disk"]:
        for data_variable, meta in input_params["zarr_meta"].items():
            write_chunk(
                img_xds,
                meta,
                parallel_dims_chunk_id,
                input_params["compressor"],
                input_params["image_file"],
            )
    else:
        return img_xds

    # mini_cube_name = os.path.join(input_params["image_file"], 'cube_chunk_' + str(input_params["task_id"]))
    # logger.debug('The mini_cube name ' + mini_cube_name)
    # img_xds.attrs.clear()
    # with dask.config.set(scheduler="synchronous"):
    #     img_xds.to_zarr(mini_cube_name)

    T_to_disk = time.time() - start_10
    logger.debug("10. to disk " + str(time.time() - start_10))
    # dask.distributed.print('Done writing results to Lustre for task ',input_params['task_id'], T_to_disk,' ***** ')

    logger.debug(
        "Completed task "
        + str(input_params["task_id"])
        + " in "
        + str(time.time() - start_total)
        + ", "
        + "Empty_image, Load, Weights, GCF, Aperture, UV, Vis, Total Loop, FFT, To_disk "
        + str(T_empty_image)
        + ", "
        + str(T_load)
        + ", "
        + str(T_weights)
        + ", "
        + str(T_gcf)
        + ", "
        + str(T_aperture_grid)
        + ", "
        + str(T_uv_sampling_grid)
        + ", "
        + str(T_vis_grid)
        + ", "
        + str(T_compute)
        + ", "
        + str(T_fft)
        + ", "
        + str(T_to_disk)
    )
    # logger.debug("Completed task " + str(input_params['task_id']) + " in " + str(time.time()-start_total) + " s.")
    # logger.debug("***"*20)
    import pandas as pd

    return_dict = {
        "task_id": [input_params["task_id"]],
        "n_channels": [len(input_params["task_coords"]["frequency"]["data"])],
        "T_load": T_load,
        "T_empty_image": T_empty_image,
        "T_load": T_load,
        "T_weights": T_weights,
        "T_aperture_grid": T_aperture_grid,
        "T_uv_sampling_grid": T_uv_sampling_grid,
        "T_vis_grid": T_vis_grid,
        "T_compute": T_compute,
        "T_fft": T_fft,
        "T_to_disk": T_to_disk,
    }
    df = pd.DataFrame(return_dict)

    # import pandas as pd
    # df = pd.DataFrame()

    return df
