
from distributed import Client

from astroviper.alma_wsu_pdr.technology_demonstrators\
    .single_dish_cube_imaging.science_code.single_dish_cube_imaging import (
        select_msv4s,
        initialize_output_images,
        compute_data_coordinates,
        image_single_dish_cube_slice
    )


def make_single_dish_images(client: Client, imaging_parameters: dict):
    # Select msv4s
    msv4_selection = client.submit(
        select_msv4s,
        ps_store=imaging_parameters["ps_store"],
        data_selection=imaging_parameters["data_selection"]
    ).result()
    # Initialize output images
    initialization_result = client.submit(
        initialize_output_images,
        ps_store=imaging_parameters["ps_store"],
        image_store=imaging_parameters["image_store"],
        channels_chunk=imaging_parameters["channels_chunk"],
        msv4_selection=msv4_selection,
        image_definition=imaging_parameters["image_definition"],
        pure=False
    ).result()
    # Build the single dish cube imaging tasks graph
    future_data_directions = {
        antenna_name: client.submit(
            compute_data_coordinates,
            ps_store=imaging_parameters["ps_store"],
            antenna_name=antenna_name,
            msv4_name=msv4_name,
            image_definition=imaging_parameters["image_definition"]
        )
        for antenna_name, msv4_name in msv4_selection.items()
    }
    image_kind_path = initialization_result["image_kind_path"]
    slices_info = initialization_result["slices_info"]
    future_slices_results = [
        client.submit(
            image_single_dish_cube_slice,
            ps_store=imaging_parameters["ps_store"],
            msv4_selection=msv4_selection,
            image_kind_path=image_kind_path,
            image_definition=imaging_parameters["image_definition"],
            slice_info=slice_info,
            data_directions=future_data_directions,
            pure=False
        )
        for slice_info in slices_info
    ]
    # Compute the graph across dask workers
    client.gather(future_slices_results)
