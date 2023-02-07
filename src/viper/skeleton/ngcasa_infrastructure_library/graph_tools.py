'''
Graph building tools help generate Dask graphs by wrapping chunk functions from the ngCASA:
Science Library in dask.delayed. A non-exhaustive list of the required functionality for the graph-
generating tools are:
- Mapping the data coordinates in each selected ds of a mds to the input chunked graph coords
(the chunked coordinates which define the parallelism of the Dask graph).
- Creating a chunk function graph node, using dask.delayed, for each chunk in graph coords.
- Labeling of graph nodes with resource requirements using dask.annotate. Examples of resource
requirements are that a Dask Worker must have a GPU or local storage.
- Gathering of graph node outputs.
'''


def scatter_graph(mds_name, graph_coords, chunk_function, chunk_function_input_parms):
    '''
    Constructs a perfectly parallel graph where the number of nodes is specified by the graphs coords and each node contains the chunk function.
    '''
    mds_sel_parms = calc_mds_sel_parms(mds_name, ...) #Describes the subselection/chunk of the mds for each graph node.
    
    graph = []
    for chunk in graph_coord:
        mds_sel_parms_chunk = mds_sel_parms[chunk]

        graph.append(dask.delayed(wrap_func_with_dio)(mds_name, chunk_function,  chunk_function_input_parms, mds_sel_parms_chunk))
        
    return graph
    
    
def gather_graph(scattered_graph , chunk_function, chunk_function_input_parms, mode='single'/'tree'):
    '''
    Gathers an existing graph to a single node. Different gathering strategies will be available for example gathering all outputs to a single node or using a binary tree structure.
    '''
    
    
def append_graph(graph, chunk_function, chunk_function_input_parms):
    '''
    Grows the graph by appending another node to each of the last nodes.
    '''


def wrap_func_with_dio(mds, chunk_function,  chunk_function_input_parms, mds_sel_parms_chunk):
    '''
        Wraps chunk function to include loading/saving/caching of data.
    '''
    
    if mds is Dataset:
        mds_chunk = mds
    else:
        #Load the data from cache or common storage. If data loads from common storage it can be cache.
        mds_chunk = transfer_mds_chunk(mds_name, mds_sel_parms_chunk)
    
    #Run the chunk function on the above selected mds_chunk.
    mds_result_chunk = chunk_function(mds_chunk, chunk_function_input_parms)
    
    
    if  chunk_function_input_parms['return_data']:
        return mds_result_chunk
    else:
        ## Save the results
        transfer_mds_chunk(mds_result_chunk)
        results_dict = mds_result_chunk['results_dict']
        return results_dict

