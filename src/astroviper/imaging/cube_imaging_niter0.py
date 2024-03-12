from numcodecs import Blosc


def cube_imaging_niter0(
    ps_name,
    image_name,
    grid_params,
    polarization_coord=None, 
    frequency_coord=None, 
    n_chunks=None,
    data_variables=["sky", "point_spread_function", "primary_beam"],
    intents=["OBSERVE_TARGET#ON_SOURCE"],
    compressor=Blosc(cname="lz4", clevel=5),
    data_group='base',
    double_precission=True,
    thread_info=None
):
    import numpy as np
    import xarray as xr
    import dask
    import os
    from xradio.vis.read_processing_set import read_processing_set
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord
    from graphviper.graph_tools import generate_dask_workflow
    from graphviper.graph_tools import map
    from astroviper.imaging._utils import _make_image
    from xradio.image import make_empty_sky_image
    from xradio.image import write_image
    import zarr
    import graphviper.utils.logger as logger

    # Get metadata
    ps = read_processing_set(ps_name, intents=intents, data_group=data_group)
    summary_df = ps.summary()

    # Get phase center of mosaic if field_id given.
    if isinstance(grid_params["phase_direction"], int):
        ms_xds_name = summary_df.loc[
            summary_df["field_id"] == grid_params["phase_direction"]
        ].name.values[0]

        vis_name = ps[ms_xds_name].attrs["data_groups"][data_group]["visibility"]
        grid_params["phase_direction"] = ps[ms_xds_name][vis_name].attrs["field_info"][
            "phase_direction"
        ]

    # Create Cube frequency axis
    if frequency_coord is None:
        frequency_values = []
        spw_ids = []
        frame = ps.get(0).frequency.frame #Should all be the same frame.
        for ms_xds in ps.values():
            if frame != ms_xds.frequency.frame:
                logger.exception('Frequency reference frame must be the same for all MS v4.')
            
            if ms_xds.frequency.spw_id not in spw_ids:
                frequency_values.append(ms_xds.frequency.values)
                spw_ids.append(ms_xds.frequency.spw_id)
        frequency_values = np.sort(np.unique(np.concatenate(frequency_values)))
        frequency_coord = xr.DataArray(frequency_values,dims='frequency')
        freq_attrs = {'frame':frame,'units':ps.get(0).frequency.attrs['units'],'type':'spectral_coord'}
        frequency_coord.attrs = freq_attrs

    if polarization_coord is None:
        polarization_coord = ps.get(0).polarization

    #Create empty image
    img_xds = make_empty_sky_image(
        phase_center=grid_params["phase_direction"]["data"],
        image_size=grid_params["image_size"],
        cell_size=grid_params["cell_size"],
        chan_coords=frequency_coord.data,
        pol_coords=polarization_coord.data,
        time_coords=[0],
    )
    write_image(img_xds, imagename=image_name, out_format="zarr")

    if n_chunks is None:
        #Calculate n_chunks
        from astroviper._utils.data_partitioning import bytes_in_dtype
        ## Determine the amount of memory required by the node task if all dimensions that chunking will occur on are singleton.
        ## For example cube_imaging does chunking only only frequency, so memory_singleton_chunk should be the amount of memory requered by _feather when there is a single frequency channel.

        n_pixels_single_frequency = img_xds.sizes['l']*img_xds.sizes['m']*img_xds.sizes['polarization']*img_xds.sizes['time']
        fudge_factor=1.2
        if double_precission:
            memory_singleton_chunk = fudge_factor*(3*n_pixels_single_frequency*bytes_in_dtype['complex64']/(1024**3) + 3*n_pixels_single_frequency*bytes_in_dtype['float32']/(1024**3))
        else:
            memory_singleton_chunk = fudge_factor*(3*n_pixels_single_frequency*bytes_in_dtype['complex128']/(1024**3) + 3*n_pixels_single_frequency*bytes_in_dtype['float64']/(1024**3))

        logger.debug("Memory required for a single frequency channel: " + str(memory_singleton_chunk) + " GiB")

        chunking_dims_sizes = {'frequency':img_xds.sizes['frequency']} #Need to know how many frequency channels there are.
        from astroviper._utils.data_partitioning import calculate_data_chunking, get_thread_info
        if thread_info is None:
            thread_info = get_thread_info()
            logger.debug("Thread info " + str(thread_info))
        n_chunks = calculate_data_chunking(memory_singleton_chunk, chunking_dims_sizes, thread_info, constant_memory=0, tasks_per_thread=4)['frequency']


    #Make Parallel Coords
    parallel_coords = {}
    ms_xds = ps.get(0)
    parallel_coords["frequency"] = make_parallel_coord(
        coord=ms_xds.frequency, n_chunks=n_chunks
    )
    logger.info('Number of frequency chunks: ' + str(len(parallel_coords['frequency']['data_chunks'])))

    from xradio.image._util._zarr.zarr_low_level import create_data_variable_meta_data_on_disk

    if double_precission:
        from xradio.image._util._zarr.zarr_low_level import image_data_variables_and_dims_double_precision as image_data_varaibles_and_dims
    else:
        from xradio.image._util._zarr.zarr_low_level import image_data_variables_and_dims_single_precision as image_data_varaibles_and_dims

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
    input_parms["polarization"] = polarization_coord.data
    input_parms["time"] = [0]
    input_parms["compressor"] = compressor
    input_parms["image_file"] = image_name
    input_parms["input_data_store"]=ps_name

    from graphviper.graph_tools.coordinate_utils import interpolate_data_coords_onto_parallel_coords
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, ps)

    #Create Map Graph
    viper_graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=_make_image,
        input_params=input_parms,
        in_memory_compute=False
    )
    input_parms = {}
    dask_graph = generate_dask_workflow(viper_graph)

    #Compute cube
    dask.compute(dask_graph)

    zarr.consolidate_metadata(image_name)






