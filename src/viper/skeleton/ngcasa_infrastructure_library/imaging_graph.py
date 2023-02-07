'''
The graph-generating functions construct graphs using the graph-building tools where the graph nodes are the functions in the ngCASA: Science Library.
'''


def major_cycle(vis_mds_name, graph_coords, chunk_function_input_parms):
    '''
    Run only the major cycle. Needed for interactive CLEAN.
    '''
    
    from ngcasa.ngcasa_science_library.imaging import  run_major_cycle_chunk, sum_img_ds_chunk, combine_return_dict
    from ngcasa.ngcasa_infrastructure_library.graph_tools import scatter_graph, gather_graph

    if chunk_function_input_parms['cube']:
        chunk_function_input_parms['return_data'] = False
        scattered_graph = scatter_graph(vis_mds_name, graph_coords, chunk_function=run_major_cycle_chunk, chunk_function_input_parms['major'])
        major_cycle_graph = gather_graph(scattered_graph , chunk_function=combine_return_dict,chunk_function_input_parms)

    elif chunk_function_input_parms['cont']:
        chunk_function_input_parms['return_data'] = False
        scattered_graph = scatter_graph(vis_mds_name, graph_coords, chunk_function=run_major_cycle_chunk, chunk_function_input_parms['major'])
        major_cycle_graph = gather_graph(scattered_graph,chunk_function=sum_img_ds_chunk,chunk_function_input_parms)

    result_dict = dask.compute(major_cycle_graph)
    return result_dict


def minor_cycle(img_ds_name, graph_coords, chunk_function_input_parms):
    '''
    Run only the minor cycle. Needed for interactive CLEAN.
    '''
    from ngcasa.ngcasa_science_library.imaging import  run_minor_cycle, combine_return_dict
    from ngcasa.ngcasa_infrastructure_library.graph_tools import scatter_graph, gather_graph

    chunk_function_input_parms['return_data'] = False
    
    #For continuum the graph will only have a single branch.
    scattered_graph = scatter_graph(img_ds_name, graph_coords, chunk_function=run_minor_cycle_chunk, chunk_function_input_parms['minor'])
    minor_cycle_graph = gather_graph(scattered_graph , chunk_function=combine_return_dict,chunk_function_input_parms)

    result_dict = dask.compute(minor_cycle_graph)
    return result_dict


def reconstruct_cube_image(mds_name, graph_coords, chunk_function_input_parms):
    '''
        Run cube image reconstruction.
    '''
    from ngcasa.ngcasa_science_library.imaging import  reconstruct_cube_image_chunk, combine_return_dict
    from ngcasa.ngcasa_infrastructure_library.graph_tools import scatter_graph, gather_graph

    chunk_function_input_parms['return_data'] = False
    ## The major/minor while loop happens inside the reconstruct_cube_image_chunk.
    scattered_graph = scatter_graph(mds_name, graph_coords, chunk_function= reconstruct_cube_image_chunk, chunk_function_input_parms)
    cube_graph = gather_graph(scattered_graph, chunk_function=combine_return_dict, chunk_function_input_parms)

    return_dict = dask.compute(cube_graph)
    return return_dict


def reconstruct_cont_image(mds_name, chunk_function_input_parms):
    '''
        Run continuum image reconstruction.
    '''

    from ngcasa.ngcasa_science_library.imaging import  run_major_cycle_chunk, run_minor_cycle_chunk, sum_img_ds_chunk, combine_return_dict
    from ngcasa.ngcasa_infrastructure_library.graph_tools import scatter_graph, gather_graph, append_graph


    results_dict = None
    while(! has_converged(results_dict)):
        chunk_function_input_parms['return_data'] = True
        scattered_graph = scatter_graph(mds_name, graph_coords, chunk_function= run_major_cycle_chunk, chunk_function_input_parms['major'])
        gathered_graph = gather_graph(scattered_graph,chunk_function=sum_img_ds_chunk,hunk_function_input_parms)
        chunk_function_input_parms['return_data'] = False
        cont_graph = append_graph(gathered_graph,chunk_function=run_minor_cycle_chunk, chunk_function_input_parms['minor']) #End up with a single node, no gather required.

        results_dict = dask.compute(cont_graph)
    
    return results_dict
        

    




