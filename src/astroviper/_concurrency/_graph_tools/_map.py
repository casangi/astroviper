
from xradio.vis.read_processing_set import read_processing_set
import itertools, functools, operator
import numpy as np
import dask
import math
import os
import datetime
from astroviper._utils._logger import _get_logger
import ipaddress


def get_unique_resource_ip(workers_info):
    nodes = []
    for worker, wi in workers_info.items():
        worker_ip = worker[worker.rfind("/") + 1 : worker.rfind(":")]
        assert worker_ip in list(
            wi["resources"].keys()
        ), "local_cache enabled but workers have not been annotated. Make sure that local_cache has been set to True during client setup."
        if worker_ip not in nodes:
            nodes.append(worker_ip)
    return nodes


def _make_iter_chunks_indxs(parallel_coords):
    parallel_dims = []
    list_chunk_indxs = []
    n_chunks = 1
    for dim, pc in parallel_coords.items():
        chunk_indxs = list(pc["data_chunks"].keys())
        n_chunks = n_chunks*len(chunk_indxs)
        list_chunk_indxs.append(chunk_indxs)
        parallel_dims.append(dim)
        
    iter_chunks_indxs = itertools.product(*list_chunk_indxs)
    return iter_chunks_indxs, parallel_dims
    

def _generate_chunk_slices(parallel_coords, ps):
    """
    """
    from scipy.interpolate import interp1d

    #Construct an interpolator for each parallel dim:
    interp1d_dict = {}
    for dim, pc in parallel_coords.items():
        interp1d_dict[dim] = interp1d(
                pc["data"],
                np.arange(len(pc["data"])),
                kind="nearest",
                bounds_error=False,
                fill_value=-1
                #fill_value="extrapolate",
                #assume_sorted=True
            ) #Should we use fill_value='extrapolate' in interp1d?

    chunk_slice_dict = {}
    for xds_key in ps:
        for dim, pc in parallel_coords.items():
            interp_indx = interp1d_dict[dim](ps[xds_key][dim].values).astype(int)
            #print(dim,interp_indx,len(interp_indx))
            
            #print(pc['data'][interp_indx])
            
            pc_chunk_start = 0
            pc_chunk_end = 0
            chunk_indx_start_stop={}
            for chunk_key in np.arange(len(pc['data_chunks'])): #ensure that keys are ordered.
                pc_chunk_start = pc_chunk_end
                pc_chunk_end = len(pc['data_chunks'][chunk_key]) + pc_chunk_end

                start_slice = np.where(interp_indx==pc_chunk_start)[0]
                if start_slice.size != 0:
                    start_slice=start_slice[0]
                else:
                    start_slice = None

                end_slice = np.where(interp_indx==pc_chunk_end-1)[0]
            
                if end_slice.size != 0:
                    end_slice=end_slice[-1]+1
                else:
                    end_slice=len(interp_indx)
                    
                if start_slice is None:
                    chunk_indx_start_stop[chunk_key] = slice(None)
                else:
                    chunk_indx_start_stop[chunk_key] = slice(start_slice,end_slice)

            if xds_key in chunk_slice_dict:
                chunk_slice_dict[xds_key][dim] = chunk_indx_start_stop
            else:
                chunk_slice_dict[xds_key] = {dim: chunk_indx_start_stop}
        
    return chunk_slice_dict
  

def _map(
    ps_name, sel_parms, parallel_coords, func_chunk, client
):
    """
    Builds a perfectly parallel graph where func_chunk node task is created for each chunk defined in parallel_coords. The data in the ps is mapped to each parallel_coords chunk.
    """

    logger = _get_logger()
    ps = read_processing_set(ps_name, sel_parms["intents"], sel_parms["fields"])
    #ps = {list(ps.keys())[0]:ps[list(ps.keys())[0]]}
    
    
    iter_chunks_indxs, parallel_dims = _make_iter_chunks_indxs(parallel_coords)
    
    chunk_slice_dict = _generate_chunk_slices(parallel_coords, ps)
    #print(chunk_slice_dict)
    
    input_parms = {"ps_name": ps_name}

    if "VIPER_LOCAL_DIR" in os.environ:
        local_cache = True
        input_parms["viper_local_dir"] = os.environ["VIPER_LOCAL_DIR"]

        if "date_time" in sel_parms:
            input_parms["date_time"] = sel_parms["date_time"]
        else:
            input_parms["date_time"] = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    else:
        local_cache = False
        input_parms["viper_local_dir"] = None
        input_parms["date_time"] = None

    if local_cache:
        workers_info = client.scheduler_info()["workers"]
        nodes_ip_list = get_unique_resource_ip(workers_info)
        n_nodes = len(nodes_ip_list)

        chunks_per_node = math.floor(n_chunks / n_nodes + 0.5)
        if chunks_per_node == 0:
            chunks_per_node = 1
        chunk_to_node_map = np.repeat(np.arange(n_nodes), chunks_per_node)

        if len(chunk_to_node_map) < n_chunks:
            n_pad = n_chunks - len(chunk_to_node_map)
            chunk_to_node_map = np.concatenate(
                [chunk_to_node_map, np.array([chunk_to_node_map[-1]] * n_pad)]
            )

    graph_list = []
    for i_chunk, chunk_indx in enumerate(iter_chunks_indxs):
        #print("chunk_indx", i_chunk, chunk_indx)

        single_chunk_slice_dict = {}
        for xds_id in ps.keys():
            single_chunk_slice_dict[xds_id] = {}
            empty_chunk = False
            for i, chunk_id in enumerate(chunk_indx):
                if chunk_id in chunk_slice_dict[xds_id][parallel_dims[i]]:
                    single_chunk_slice_dict[xds_id][
                        parallel_dims[i]
                    ] = chunk_slice_dict[xds_id][parallel_dims[i]][chunk_id]
                    
                    if chunk_slice_dict[xds_id][parallel_dims[i]][chunk_id] == slice(None):
                        empty_chunk = True
                else:
                    empty_chunk = True

            if (
                empty_chunk
            ):  # The xds with xds_id has no data for the parallel chunk (no slice on one of the dims).
                single_chunk_slice_dict.pop(xds_id, None)
        #print(single_chunk_slice_dict)
        if single_chunk_slice_dict:
            input_parms["data_sel"] = single_chunk_slice_dict
            input_parms["chunk_coords"] = {}
            for i_dim,dim in enumerate(parallel_dims):
                chunk_coords={}
                chunk_coords["data"] = parallel_coords[dim]["data_chunks"][chunk_indx[i_dim]]
                chunk_coords["dims"] = parallel_coords[dim]["dims"]
                chunk_coords["attrs"] = parallel_coords[dim]["attrs"]
                input_parms["chunk_coords"][dim] = chunk_coords
            input_parms["chunk_indx"] = chunk_indx
            input_parms["chunk_id"] = chunk_id
            input_parms["parallel_dims"] = parallel_dims

            if local_cache:
                node_ip = nodes_ip_list[chunk_to_node_map[i_chunk]]
                logger.debug(
                    "Task with chunk id "
                    + str(chunk_id)
                    + " is assigned to ip "
                    + str(node_ip)
                )
                input_parms["node_ip"] = node_ip
                with dask.annotate(resources={node_ip: 1}):
                    graph_list.append(dask.delayed(func_chunk)(dask.delayed(input_parms)))
            else:
                #print("input_parms",input_parms)
                graph_list.append(dask.delayed(func_chunk)(dask.delayed(input_parms)))

        """
        a = np.ravel_multi_index([127,0,0,1],[256,256,256,256])
        #a = np.ravel_multi_index([255,255,255,254],[256,256,256,256])
        print(a)
        """

    return graph_list, input_parms["date_time"]
