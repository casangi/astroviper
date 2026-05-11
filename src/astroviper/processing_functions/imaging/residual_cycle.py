import time

from astroviper.processing_functions.imaging.get_visibility_grid import get_visibility_grid_single_field
from astroviper.utils.data_group_tools import create_data_groups_in_and_out, modify_data_groups_xds


# from memory_profiler import profile
# @profile(precision=1)
def residual_cycle_cube_single_field(ps_xdt, img_xds, input_params, is_n_iter_0, img_residual_data_group_name = "residual", img_model_data_group_name = "model", last_residual_cycle=False):
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
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        ifft_norm_img_xds, fft_norm_img_xds
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )

    from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
        point_spread_function_gaussian_fit,
    )
    
    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )

    ps_data_group_name = input_params["processing_set_data_group_name"]
    
    T_start_gcf = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(100, 7)
    T_gcf = time.time() - T_start_gcf
    
    #Degrid and calculate residual visibilities.
    if not is_n_iter_0:
        #print("1. Data vars", img_xds.data_vars)
        residual_data_group = img_xds.attrs["data_groups"][img_residual_data_group_name]
        img_xds.xr_img.delete_data_variables(variables=[residual_data_group["sky"]]) #Deletes the SKY_RESIDUAL.
        
        #Stokes to corr for model visibilities.
        # NB NB NB To do: Determine new_polarization_basis based on input data polarization basis. Don't hard code "linear".
        img_xds = transform_polarization_basis( 
            img_xds, new_polarization_basis="linear", overwrite=True
        ) 
        
   
        # ifft_norm_img_xds will have already transformed to stokes, so transform back to corr for degridding.
        # cgk_1D = create_prolate_spheroidal_kernel_1D(100, 7)
        # img_casa_xds = xr.open_zarr("twhya_selfcal_5chans_lsrk_niter_99_nmajor_1_briggs.img.zarr")
        # img_xds["SKY_MODEL"].values = img_casa_xds.SKY_MODEL.values
        
        
        # print("2. Data vars", img_xds.data_vars)
        # print("2.1 Keep", input_params["image_data_variables_keep"])
        start_fft_norm = time.time()
        img_xds = fft_norm_img_xds(
            img_xds,
            image_params=input_params["image_params"],
            image_data_group_in_name=img_model_data_group_name,
            image_data_group_out_name=img_model_data_group_name,
            image_data_group_out_modified={
                "visibility": "VISIBILITY_MODEL",
            },
            image_data_variables_keep=['sky'],
            num_threads=input_params["processing_function_threads"],
        )
        T_fft_norm = time.time() - start_fft_norm
        
        #print("3. Data vars", img_xds.data_vars)
        
        make_visibility_model_single_field( ps_xdt,
            img_xds,
            cgk_1D,
            ms_data_group_out_name = "model",
            ms_data_group_out_modified = {
                                "correlated_data": "VISIBILITY_MODEL",
                            },
            img_data_group_in_name = "model",
            num_threads=1,)
        
        # from xradio.measurement_set.load_processing_set import load_processing_set
        # ps_xdt2 = load_processing_set(
        #         input_params["input_data_store"],
        #         sel_parms=input_params["data_selection"],
        #         load_sub_datasets=False,
        #     )  
        # model2 = ps_xdt2["twhya_selfcal_lsrk_5chans_0"].ds.VISIBILITY_MODEL.values
        # model_av = ps_xdt["twhya_selfcal_lsrk_5chans_0"].ds.VISIBILITY_MODEL.values
        # print("1^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(model2[0:5,0,:,0])
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(model_av[0:5,0,:,0])
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(model2[0:5,0,:,0]-model_av[0:5,0,:,0])
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(np.abs(model2[0:5,0,:,0])/np.abs(model_av[0:5,0,:,0]))
        # print(np.abs(model_av[0:5,0,:,0])/np.abs(model2[0:5,0,:,0]))
        # print(model2.shape,model_av.shape)
        # print("Max abs diff in model visibilities: " + str(np.nanmax(np.abs(model2-model_av))))
        # print("Max abs diff in model visibilities: " + str(np.nanmean(np.abs(model2-model_av))))
        # print(img_xds.VISIBILITY_NORMALIZATION.values)
        # print(img_casa_xds.VISIBILITY_NORMALIZATION.values)
        # print("2^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        calculate_residual_visibilities(ps_xdt, ms_data_group_residual_name="residual", ms_data_group_model_name="model", ms_data_group_original_name=ps_data_group_name)
        
        ps_data_group_name="residual" #After the first iteration, the residual data group becomes the new "input" data group for the next iteration.

    if img_residual_data_group_name not in img_xds.attrs["data_groups"]:
        img_xds.attrs["type"] = "image_dataset"
        img_xds = img_xds.xr_img.add_data_group(
            new_data_group_name=img_residual_data_group_name,
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
    # Nb Nb handle data_group_out correctly for not is_n_iter_0.

    T_start_make_uv_images_single_field = time.time()
    img_xds, make_uv_images_single_field_return_df = make_uv_images_single_field(
        ps_xdt,
        img_xds,
        input_params["image_params"],
        cgk_1D,
        is_n_iter_0,
        ms_data_group_in_name=ps_data_group_name,
        img_data_group_out_name=img_residual_data_group_name,
        num_threads=input_params["processing_function_threads"],
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

    from astroviper.processing_functions.imaging.make_pb_symmetric import (
        airy_disk_rorder,
        airy_disk_rorder_v2,
    )
    
    
    if img_xds.attrs["data_groups"][img_residual_data_group_name].get("primary_beam", None) is None:
        image_data_group_in, image_data_group_out = create_data_groups_in_and_out(
            img_xds,
            data_group_in_name=img_residual_data_group_name,
            data_group_out_name=img_residual_data_group_name,
            data_group_out_modified={"primary_beam": "PRIMARY_BEAM"},
            overwrite=False,
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
        
        img_xds["PRIMARY_BEAM"].attrs['type'] = "primary_beam"
        img_xds["PRIMARY_BEAM"].attrs['method'] = "airy_disk"
        
        modify_data_groups_xds(
            img_xds,
            data_group_out_name=img_residual_data_group_name,
            data_group_out=image_data_group_out,
            description="Added primary beam to data group.",
        )
        
        T_primary_beam = time.time() - start
    else:
        T_primary_beam = 0.0

    # Temp: Add singleton time dim to img_xds for FFT normalization. Need to fix gridders to not require this.
    # del img_xds["time"]
    # img_xds = img_xds.expand_dims(dim="time", axis=0)
    
    #print("Before ifff img_xds", img_xds)
    
    if is_n_iter_0:
        ifft_norm_image_data_group_out_modified ={
            "sky": "SKY_RESIDUAL",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        }
    else:
        ifft_norm_image_data_group_out_modified ={
            "sky": "SKY_RESIDUAL",
        }

    start_fft_norm = time.time()
    img_xds = ifft_norm_img_xds(
        img_xds,
        image_params=input_params["image_params"],
        image_data_group_in_name=img_residual_data_group_name,
        image_data_group_out_name=img_residual_data_group_name,
        image_data_group_out_modified=ifft_norm_image_data_group_out_modified,
        image_data_variables_keep=input_params["image_data_variables_keep"],
        num_threads=input_params["processing_function_threads"],
    )
    T_fft_norm = time.time() - start_fft_norm
    
    from toolviper.utils.memory_management import get_rss_gb
    
    logger.debug("Memory usage after residual cycle " + str(get_rss_gb()) + " GB")
    start = time.time()
    img_xds = transform_polarization_basis(
        img_xds, new_polarization_basis="stokes", overwrite=True
    )
    T_transform_pol = time.time() - start
    logger.debug("Memory usage after transform polarization " + str(get_rss_gb()) + " GB")

    if is_n_iter_0:
        start = time.time()
        img_xds = point_spread_function_gaussian_fit(
            img_xds,
            image_data_group_in_name=img_residual_data_group_name,
            image_data_group_out_name=img_residual_data_group_name,
            image_data_group_out_modified={
                "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
            },
            overwrite=True,
            num_threads=input_params["processing_function_threads"],
        )
        T_psf_fit = time.time() - start
    else:
        T_psf_fit = 0.0

    return_dict = {
        "T_transform_pol": [T_transform_pol],
        "T_weights": [T_weights],
        "T_make_uv_images_single_field": [T_make_uv_images_single_field],
        "T_gcf": [T_gcf],
        "T_primary_beam": [T_primary_beam],
        "T_fft_norm": [T_fft_norm],
        "T_psf_fit": [T_psf_fit],
    }
    import pandas as pd

    return_df = pd.DataFrame(return_dict)

    # Add the return dict from make_uv_images_single_field
    return_df = pd.concat([return_df, make_uv_images_single_field_return_df], axis=1)

    return img_xds, return_df

def make_visibility_model_single_field( ps_xdt,
    img_xds,
    cgk_1D,
    ms_data_group_out_name = "model",
    ms_data_group_out_modified = {
                        "correlated_data": "VISIBILITY_MODEL",
                    },
   img_data_group_in_name = "model",
    num_threads=1,):
    
    for ms_name, ms_xdt in ps_xdt.items():
        get_visibility_grid_single_field(
                    ms_xdt,
                    cgk_1D,
                    img_xds,
                    ms_data_group_out_name = ms_data_group_out_name,
                    ms_data_group_out_modified = ms_data_group_out_modified,
                    img_data_group_in_name = img_data_group_in_name,
                    overwrite = True,
                    chan_mode = "cube",
                    fft_padding = 1.2,
                    num_threads = num_threads,
                )
        


def calculate_residual_visibilities(ps_xdt, ms_data_group_residual_name="residual", ms_data_group_model_name="model", ms_data_group_original_name="base"):
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )
    
    for ms_name, ms_xdt in ps_xdt.items():
        ms_data_group_model = ms_xdt.attrs["data_groups"][ms_data_group_model_name]
        ms_data_group_original = ms_xdt.attrs["data_groups"][ms_data_group_original_name]
        
        ms_data_group_original, ms_data_group_residual = create_data_groups_in_and_out(
                    ms_xdt,
                    data_group_in_name=ms_data_group_original_name,
                    data_group_out_name=ms_data_group_residual_name,
                    data_group_out_modified={
                        "correlated_data": "VISIBILITY_RESIDUAL",
                    },
                    overwrite=True,
                )
        
        import numpy as np
        #print(ms_data_group_residual["correlated_data"], ms_data_group_original["correlated_data"], ms_data_group_model["correlated_data"])
        # print("$$$$$$$$$$$$$$$ Original visibilities:")
        # print(np.abs(ms_xdt[ms_data_group_original["correlated_data"]].values))
        # print("$$$$$$$$$$$$$$$ Model visibilities:")
        # print(np.abs(ms_xdt[ms_data_group_model["correlated_data"]].values))
              
        
        ms_xdt[ms_data_group_residual["correlated_data"]] = ms_xdt[ms_data_group_original["correlated_data"]] - ms_xdt[ms_data_group_model["correlated_data"]]

        modify_data_groups_xds(
            ms_xdt,
            data_group_out_name=ms_data_group_residual_name,
            data_group_out=ms_data_group_residual,
            description="Calculated residual visibilities by subtracting model visibilities from original visibilities.",
        )

def make_uv_images_single_field(
    ps_xdt,
    img_xds,
    image_params,
    cgk_1D,
    is_n_iter_0,
    ms_data_group_in_name="corrected",
    img_data_group_out_name="residual",
    num_threads=1,
):
    from astroviper.processing_functions.imaging.add_uv_sampling_grid import (
        add_uv_sampling_grid_single_field,
    )
    from astroviper.processing_functions.imaging.add_visibility_grid import (
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
                img_data_group_in_name=img_data_group_out_name,
                img_data_group_out_name=img_data_group_out_name,
                img_data_group_out_modified={
                    "uv_sampling": "UV_SAMPLING",
                    "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
                },
                overwrite=True,
                chan_mode="cube",
                fft_padding=image_params["fft_padding"],
                num_threads=num_threads,
            )  # Will become the PSF.
            T_uv_sampling_grid = T_uv_sampling_grid + time.time() - T_start_uv

        T_start_vis = time.time()
        add_visibility_grid_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            img_data_group_in_name=img_data_group_out_name,
            img_data_group_out_name=img_data_group_out_name,
            img_data_group_out_modified={
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
            },
            overwrite=True,
            chan_mode="cube",
            fft_padding=image_params["fft_padding"],
            num_threads=num_threads,
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
