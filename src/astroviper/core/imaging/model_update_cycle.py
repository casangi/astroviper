



def print_nested_dict(d, name=None):
    import pprint
    if name is not None:
        print(f"{name}:")
    pprint.pprint(d, width=100, sort_dicts=False)


def model_update_cycle_cube_single_field(img_xds, input_params, is_n_iter_0, num_threads=1,  img_data_group_in_name = "residual", img_data_group_out_name = "model"):

    # img_data_group_in_name: str = "base",
    # img_data_group_out_name: str = "imaging",
    # img_data_group_out_modified: dict = {"weight_imaging": "WEIGHT_IMAGING"}
    
    from astroviper.core.imaging.deconvolution import deconvolve
    from astroviper.core.image_analysis.make_mask import make_mask
    

    # Adding PB mask to img_data_group_in_name.
    mask_name = img_xds.attrs["data_groups"][img_data_group_in_name].get("mask", None)
    if mask_name is None:
        make_mask(img_xds, threshold=input_params["iteration_control_params"]["threshold"], image_data_group_in_name=img_data_group_in_name, image_data_group_out_name=img_data_group_in_name, combine_mask=False, overwrite=False)
    
    deconvolve_dict = deconvolve(
            img_xds=img_xds,
            algorithm='hogbom',
            deconvolve_params=input_params["iteration_control_params"],
            image_data_group_in_name = img_data_group_in_name,
            image_data_group_out_name =  img_data_group_out_name,
            num_threads=num_threads
        )
    
    #img_xds.to_zarr("test123.zarr", mode="w")
    import numpy as np
    # import matplotlib.pyplot as plt
    # for i in range(len(img_xds.frequency)):
    #     plt.figure(figsize=(20,10)) 
    #     print("%%%%%%%%%%%%%%%%%%%% the max abs value in SKY_MODEL:", np.max(np.abs(img_xds["SKY_MODEL"].isel(polarization=1,time=0, frequency=i)).values))
    #     plt.imshow(img_xds["SKY_MODEL"].isel(polarization=1,time=0, frequency=i).values)
    #     plt.colorbar()
    # plt.show()
    
