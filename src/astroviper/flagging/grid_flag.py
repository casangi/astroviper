def grid_flag(ps_store, grid_flag_params):

    from xradio.vis.read_processing_set import read_processing_set

    from graphviper.graph_tools import map, reduce, generate_dask_workflow
    from graphviper.graph_tools.coordinate_utils import (
        make_parallel_coord,
        interpolate_data_coords_onto_parallel_coords)

    from astroviper._domain._flagging._gridflag_histogram import (
        accumulate_uv_points,
        merge_uv_grids,
        apply_flags)

    import dask
    import numpy as np

    # ----

    ps = read_processing_set(ps_store)

    for msv4 in ps:
        nflag = np.count_nonzero(ps[msv4].FLAG)
        frac = nflag/ps[msv4].FLAG.size
        print(f"{msv4} % flags {frac*100.:.3f}")

    # uvrange = [0,12500]
    # uvcell = 10
    # nhistbin = 100
    # nsigma = 5

    npixu = 2 * grid_flag_params["uvrange"][1] // grid_flag_params["uvcell"]
    npixv = grid_flag_params["uvrange"][1] // grid_flag_params["uvcell"]

    input_params = grid_flag_params
    input_params["npixels"] = [npixu, npixv]
    input_params["input_data_store"] = ps_store
    input_params["nhistbin"] = grid_flag_params['nhistbin']

    # Make Parallel Coords.
    # Needs to be reworked for multiple spw.
    n_chunks = 4
    parallel_coords = {}
    ms_xds = ps.get(0)
    parallel_coords["frequency"] = make_parallel_coord(
        coord=ms_xds.frequency, n_chunks=n_chunks
    )

    n_chunks = 4
    parallel_coords["baseline_id"] = make_parallel_coord(
        coord=ms_xds.baseline_id, n_chunks=n_chunks
    )

    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, ps
    )

    # Generate map graph
    graph = map(
        input_data=ps,
        node_task_data_mapping=node_task_data_mapping,
        node_task=accumulate_uv_points,
        input_params=input_params,
        in_memory_compute=False,
    )

    # Generate reduce graph
    graph = reduce(graph, merge_uv_grids, input_params, mode="tree")
    dask_graph = generate_dask_workflow(graph)

    input_params = {}
    input_params["accum_uv_hist"] = dask_graph

    #graph = map(
    #    input_data=ps,
    #    node_task_data_mapping=node_task_data_mapping,
    #    node_task=apply_flags,
    #    input_params=input_params,
    #    in_memory_compute=False,
    #)

    #dask_graph = generate_dask_workflow(graph)
    # Inspect graph.
    #dask.visualize(dask_graph, filename="gridflag_graph_raw.png", ranked='LR')

    result = dask.compute(dask_graph, num_workers=3)
    #for msv4 in ps:
    #    nflag = np.count_nonzero(ps[msv4].FLAG)
    #    frac = nflag/ps[msv4].FLAG.size
    #    print(f"{msv4} % flags {frac*100.:.3f}")

    print("DONE")

    return dask_graph
