def grid_flag(ps_store, grid_flag_params):
    from xradio.vis.read_processing_set import read_processing_set
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord
    from astroviper._domain._flagging._gridflag_histogram import (
        compute_uv_histogram,
        merge_uv_grids,
        apply_flags,
        generate_flags,
    )
    from graphviper.graph_tools import map, reduce
    import dask

    ps = read_processing_set(ps_store)
    print(ps.summary())

    # uvrange = [0,12500]
    # uvcell = 10
    # nhistbin = 100
    # nsigma = 5

    npixu = 2 * grid_flag_params["uvrange"][1] // grid_flag_params["uvcell"]
    npixv = grid_flag_params["uvrange"][1] // grid_flag_params["uvcell"]

    input_params = grid_flag_params
    input_params["npixels"] = [npixu, npixv]
    input_params["input_data_store"] = ps_store

    # Make Parallel Coords.
    # Needs to be reworked for multiple spw.
    # Development: Only selecting the first 5 channels so it does not take too long.
    n_chunks = 5
    parallel_coords = {}
    ms_xds = ps.get(0)
    parallel_coords["frequency"] = make_parallel_coord(
        coord=ms_xds.frequency[0:5], n_chunks=n_chunks
    )
    print(len(parallel_coords["frequency"]["data"]))

    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
    )

    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps
    )

    # Create Map Graph
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=compute_uv_histogram,
        input_parms=input_params,
        in_memory_compute=False,
    )

    # input_parms = {}
    graph = reduce(graph, merge_uv_grids, input_params, mode="single_node")

    input_params = {}
    input_params["accum_uv_hist"] = graph

    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=generate_flags,
        input_parms=input_params,
        in_memory_compute=False,
    )

    # Inspect graph.
    dask.visualize(graph, filename="gridflag_graph.png")

    result = dask.compute(graph)
    print(result)
