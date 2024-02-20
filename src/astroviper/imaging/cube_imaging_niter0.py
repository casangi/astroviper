from numcodecs import Blosc


def cube_imaging_niter0(
    ps_name,
    image_name,
    grid_params,
    n_chunks,
    data_variables=["sky", "point_spread_function", "primary_beam"],
    intents=["OBSERVE_TARGET#ON_SOURCE"],
    compressor=Blosc(cname="lz4", clevel=5),
):
    import numpy as np
    import xarray as xr
    import dask
    import os
    from xradio.vis.read_processing_set import read_processing_set
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord
    from graphviper.graph_tools import map
    from astroviper.imaging._utils import _make_image
    from xradio.image import make_empty_sky_image
    from xradio.image import write_image
    import zarr

    # Get metadata
    ps = read_processing_set(ps_name, intents=intents)
    summary_df = ps.summary()

    # Get phase center of mosaic if field_id given.
    if isinstance(grid_params["phase_direction"], int):
        ms_xds_name = summary_df.loc[
            summary_df["field_id"] == grid_params["phase_direction"]
        ].name.values[0]
        grid_params["phase_direction"] = ps[ms_xds_name].attrs["field_info"][
            "phase_direction"
        ]

    #Make Parallel Coords
    parallel_coords = {}
    ms_xds = ps.get(0)
    parallel_coords["frequency"] = make_parallel_coord(
        coord=ms_xds.frequency, n_chunks=n_chunks
    )

    #Create empty image
    img_xds = make_empty_sky_image(
        phase_center=grid_params["phase_direction"]["data"],
        image_size=grid_params["image_size"],
        cell_size=grid_params["cell_size"],
        chan_coords=parallel_coords["frequency"]["data"],
        pol_coords=ms_xds.polarization.values,
        time_coords=[0],
    )

    #print(img_xds)

    write_image(img_xds, imagename=image_name, out_format="zarr")

    from xradio.image._util._zarr.zarr_low_level import (
        create_data_variable_meta_data_on_disk, image_data_varaibles_and_dims
    )

    xds_dims = dict(img_xds.dims)
    data_varaibles_and_dims_sel = {
        key: image_data_varaibles_and_dims[key] for key in data_variables
    }
    zarr_meta = create_data_variable_meta_data_on_disk(
        image_name, data_varaibles_and_dims_sel, xds_dims, parallel_coords, compressor
    )

    sel_parms = {}
    sel_parms["intents"] = intents
    sel_parms["fields"] = None

    input_parms = {}
    input_parms["grid_params"] = grid_params
    input_parms["zarr_meta"] = zarr_meta
    input_parms["to_disk"] = True
    input_parms["grid_params"]["polarization"] = ms_xds.polarization.values
    input_parms["grid_params"]["frequency"] = None
    input_parms["grid_params"]["time"] = [0]
    input_parms["compressor"] = compressor
    input_parms["image_file"] = image_name
    input_parms["input_data_store"]=ps_name

    from graphviper.graph_tools.coordinate_utils import interpolate_data_coords_onto_parallel_coords
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, ps)

    #Create Map Graph
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_make_image,
        input_params=input_parms,
        in_memory_compute=False
    )
    input_parms = {}

    #Compute cube
    dask.compute(graph)

    zarr.consolidate_metadata(image_name)






