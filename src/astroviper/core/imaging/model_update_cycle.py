

def model_update_cycle_cube_single_field(img_xds, input_params, is_n_iter_0, num_threads=1, img_data_group_name = "single_field"):

    # img_data_group_in_name: str = "base",
    # img_data_group_out_name: str = "imaging",
    # img_data_group_out_modified: dict = {"weight_imaging": "WEIGHT_IMAGING"}
    
    print(img_xds)
    print("************" * 10)
    print(img_xds.attrs["data_groups"])
    print("************" * 10)
    from astroviper.core.imaging.deconvolution import deconvolve
    
    deconvolve_dict = deconvolve(
            img_xds=img_xds,
            algorithm='hogbom',
            deconvolve_params=input_params["iteration_control_params"],
            image_data_group_in_name = "residual",
            image_data_group_out_name = "model",
            num_threads=num_threads
        )

    print("************" * 10)
    print(deconvolve_dict)
    print("************" * 10)
