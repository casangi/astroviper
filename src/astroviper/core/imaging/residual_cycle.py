import time


# from memory_profiler import profile
# @profile(precision=1)
def residual_cycle_cube_single_field(ps_xdt, img_xds, input_params, is_n_iter_0):
    """_summary_

    Parameters
    ----------
    ps_xdt : _type_
        _description_
    img_xds : _type_
        _description_
    input_params : _type_
        _description_
    is_n_iter_0 : _type_
        _description_
    """
    import toolviper.utils.logger as logger
    import xarray as xr
    import time

    start_0 = time.time()
    import numpy as np
    from astroviper.core.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )
    from astroviper.core.imaging.fft_normalize_prolate_spheriodal_gridder import (
        ifft_norm_img_xds,
    )
    from astroviper.core.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )

    from astroviper.core.image_analysis.point_spread_function_gaussian_fit import (
        point_spread_function_gaussian_fit,
    )

    img_data_group_name = "single_field"

    ps_data_group_name = input_params["processing_set_data_group_name"]
    img_xds.attrs["type"] = "image_dataset"
    img_xds = img_xds.xr_img.add_data_group(
        new_data_group_name=img_data_group_name,
        new_data_group={"description": "test", "date": "2026"},
    )
    logger.debug("img_xds size " + str(img_xds.nbytes / 1e9) + " GB")

    T_weights = 0.0
    if is_n_iter_0:
        T_start_weight = time.time()
        data_group_out = calculate_imaging_weights(
            ps_xdt,
            img_xds,
            imaging_weights_params=input_params["imaging_weights_params"],
            return_weight_density_grid=False,
            ms_data_group_in_name=ps_data_group_name,
            ms_data_group_out_name=ps_data_group_name,
            ms_data_group_out_modified={"weight_imaging": "WEIGHT_IMAGING"},
        )
        T_weights = time.time() - T_start_weight
        logger.debug("Calculate imaging weights " + str(time.time() - T_start_weight))
    # Nb Nb handle data_group_out correctly for not is_n_iter_0.

    T_start_gcf = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(100, 7)
    T_gcf = time.time() - T_start_gcf

    T_start_make_uv_images_single_field = time.time()
    img_xds, make_uv_images_single_field_return_df = make_uv_images_single_field(
        ps_xdt,
        img_xds,
        input_params["image_params"],
        cgk_1D,
        is_n_iter_0,
        ms_data_group_in_name=ps_data_group_name,
        img_data_group_out_name=img_data_group_name,
    )
    T_make_uv_images_single_field = time.time() - T_start_make_uv_images_single_field

    # Creation of primary beam
    start = time.time()
    pb_parms = {}
    pb_parms["list_dish_diameters"] = np.array([10.7])
    pb_parms["list_blockage_diameters"] = np.array([0.75])
    pb_parms["ipower"] = 1

    input_params["image_params"]["image_center"] = (
        np.array(input_params["image_params"]["image_size"]) // 2
    ).tolist()

    from astroviper.core.imaging.make_pb_symmetric import (
        airy_disk_rorder,
        airy_disk_rorder_v2,
    )

    img_xds["PRIMARY_BEAM"] = xr.DataArray(
        airy_disk_rorder_v2(
            img_xds.frequency.values,
            img_xds.polarization.values,
            pb_parms,
            input_params["image_params"],
        )[0, ...][
            None, ...
        ],  # Select first since we only have one dish diameter and add time axis.
        dims=("time", "frequency", "polarization", "l", "m"),
    )
    logger.debug("Calculate primary beam " + str(time.time() - start))

    # Temp: Add singleton time dim to img_xds for FFT normalization. Need to fix gridders to not require this.
    # del img_xds["time"]
    # img_xds = img_xds.expand_dims(dim="time", axis=0)

    print("1 img_xds ", img_xds)
    print("*************" * 10)

    start_fft_norm = time.time()
    img_xds = ifft_norm_img_xds(
        img_xds,
        image_params=input_params["image_params"],
        image_data_group_in_name=img_data_group_name,
        image_data_group_out_name=img_data_group_name,
        image_data_group_out_modified={
            "sky": "SKY_RESIDUAL",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        },
        image_data_variables_keep=input_params["image_data_variables_keep"],
    )
    T_fft_norm = time.time() - start_fft_norm
    logger.debug("9.  fft norm " + str(time.time() - start_fft_norm))

    start = time.time()
    img_xds = point_spread_function_gaussian_fit(
        img_xds,
        image_data_group_in_name="single_field",
        image_data_group_out_name="single_field",
        image_data_group_out_modified={
            "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
        },
        overwrite=True,
    )
    T_psf_fit = time.time() - start

    print("2 img_xds ", img_xds)

    return_dict = {
        "T_weights": [T_weights],
        "T_make_uv_images_single_field": [T_make_uv_images_single_field],
        "T_fft_norm": [T_fft_norm],
        "T_psf_fit": [T_psf_fit],
    }
    import pandas as pd

    return_df = pd.DataFrame(return_dict)

    # Add the return dict from make_uv_images_single_field
    return_df = pd.concat([return_df, make_uv_images_single_field_return_df], axis=1)

    return img_xds, return_df


def make_uv_images_single_field(
    ps_xdt,
    img_xds,
    image_params,
    cgk_1D,
    is_n_iter_0,
    ms_data_group_in_name="corrected",
    img_data_group_out_name="single_field",
):
    from astroviper.core.imaging.add_uv_sampling_grid import (
        add_uv_sampling_grid_single_field,
    )
    from astroviper.core.imaging.add_visibility_grid import (
        add_visibility_grid_single_field,
    )

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    T_vis_grid = 0.0

    T_start_add_to_grid = time.time()
    for ms_name, ms_xdt in ps_xdt.items():
        T_start_vis_mask = time.time()
        # Create a mask where baseline_antenna1_name does not equal baseline_antenna2_name
        mask = ms_xdt["baseline_antenna1_name"] != ms_xdt["baseline_antenna2_name"]
        # Apply the mask to the Dataset
        ms_xdt.ds = ms_xdt.ds.where(mask, drop=True)
        T_vis_mask = T_vis_mask + time.time() - T_start_vis_mask

        if is_n_iter_0:
            T_start_uv = time.time()
            add_uv_sampling_grid_single_field(
                ms_xdt,
                cgk_1D,
                img_xds,
                ms_data_group_in_name=ms_data_group_in_name,
                img_data_group_out_name=img_data_group_out_name,
                img_data_group_out_modified={
                    "uv_sampling": "UV_SAMPLING",
                    "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
                },
                overwrite=True,
                chan_mode="cube",
                fft_padding=image_params["fft_padding"],
            )  # Will become the PSF.
            T_uv_sampling_grid = T_uv_sampling_grid + time.time() - T_start_uv

        T_start_vis = time.time()
        add_visibility_grid_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            img_data_group_out_name=img_data_group_out_name,
            img_data_group_out_modified={
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
            },
            overwrite=True,
            chan_mode="cube",
            fft_padding=image_params["fft_padding"],
        )
        T_vis_grid = T_vis_grid + time.time() - T_start_vis

    return_dict = {
        "T_vis_mask": [T_vis_mask],
        "T_uv_sampling_grid": [T_uv_sampling_grid],
        "T_vis_grid": [T_vis_grid],
    }
    import pandas as pd

    return_df = pd.DataFrame(return_dict)

    return img_xds, return_df
