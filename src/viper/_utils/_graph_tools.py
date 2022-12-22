#  CASA Next Generation Infrastructure
#  Copyright (C) 2021 AUI, Inc. Washington DC, USA
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from viper._utils._io import _load_mxds
import itertools
import numpy as np
import dask


def _make_iter_chunks_indxs(parallel_coords, parallel_dims):
    chunks_dict = parallel_coords.chunks

    parallel_chunks_dict = {}
    n_chunks = {}
    chunk_indxs = []

    for dim in parallel_dims:
        parallel_chunks_dict[dim] = chunks_dict[dim]
        n_chunks[dim] = len(chunks_dict[dim])
        chunk_indxs.append(np.arange(n_chunks[dim]))

    iter_chunks_indxs = itertools.product(*chunk_indxs)
    return iter_chunks_indxs


def find_start_stop(chunk_indx):
    chunk_indx_start_stop = {}
    diff = np.diff(chunk_indx)
    non_zeros_indx = np.nonzero(diff)[0]

    if non_zeros_indx.size:
        prev_nz_indx = 0
        for nz_indx in non_zeros_indx:
            chunk_indx_start_stop[chunk_indx[nz_indx]] = slice(
                prev_nz_indx, nz_indx + 1
            )
            prev_nz_indx = nz_indx + 1
        chunk_indx_start_stop[chunk_indx[-1]] = slice(
            prev_nz_indx, len(chunk_indx)
        )  # Last chunk
    else:
        chunk_indx_start_stop = {
            chunk_indx[0]: slice(None)
        }  # all data maps to same parallel chunk

    return chunk_indx_start_stop


def generate_chunk_slices(parallel_coords, mxds, parallel_dims):
    """
    Questions: Should we use fill_value='extrapolate' in interp1d?
    """
    from scipy.interpolate import interp1d

    chunk_slice_dict = {}

    # Construct an interpolator for each parallel dim:
    interp1d_dict = {}
    for pC_dim in parallel_coords.coords:
        if pC_dim in parallel_dims:
            interp1d_dict[pC_dim] = interp1d(
                parallel_coords[pC_dim].values,
                np.arange(len(parallel_coords[pC_dim])),
                kind="nearest",
                fill_value="extrapolate",
            )

    for xds_key in mxds:
        for pC_dim in parallel_coords.coords:
            if pC_dim in parallel_dims:
                interp_indx = interp1d_dict[pC_dim](mxds[xds_key][pC_dim].values)
                chunk_indx = (interp_indx // parallel_coords.chunks[pC_dim][0]).astype(
                    int
                )
                chunk_indx_start_stop = find_start_stop(chunk_indx)

                if xds_key in chunk_slice_dict:
                    chunk_slice_dict[xds_key][pC_dim] = chunk_indx_start_stop
                else:
                    chunk_slice_dict[xds_key] = {pC_dim: chunk_indx_start_stop}

    return chunk_slice_dict


def sel_parallel_coords_chunk(parallel_coords, i_chunks, parallel_dims):
    dim_slices_dict = {}
    for i, dim in enumerate(parallel_dims):
        end = np.sum(parallel_coords.chunks[dim][: i_chunks[i] + 1])
        start = end - parallel_coords.chunks[dim][i_chunks[i]]
        dim_slices_dict[dim] = slice(start, end)

    parallel_coords_chunk = parallel_coords.isel(dim_slices_dict)
    return parallel_coords_chunk


def _build_perfectly_parallel_graph(mxds_name, sel_parms, parallel_coords, parallel_dims, func_chunk, client):
    """
    Builds a perfectly parallel graph where func_chunk node task is created for each chunk defined in parallel_coords. The data in the mxds is mapped to each parallel_coords chunk.
    """
    
    #if local cache is required
    #wait for workers to join
    print(client.scheduler_info())

    mxds = _load_mxds(mxds_name, sel_parms)
    iter_chunks_indxs = _make_iter_chunks_indxs(parallel_coords, parallel_dims)

    chunk_slice_dict = generate_chunk_slices(parallel_coords, mxds, parallel_dims)

    input_parms = {"mxds_name": mxds_name}
    graph_list = []
    for i_chunks in iter_chunks_indxs:
        # print('i_chunks',i_chunks)

        parallel_coords_chunk = sel_parallel_coords_chunk(
            parallel_coords, i_chunks, parallel_dims
        )
        parallel_coords_chunk = parallel_coords_chunk.drop_vars(
            list(parallel_coords_chunk.keys())
        )

        single_chunk_slice_dict = {}
        for xds_id in mxds.keys():
            single_chunk_slice_dict[xds_id] = {}
            empty_chunk = False
            for i, chunk_id in enumerate(i_chunks):
                if chunk_id in chunk_slice_dict[xds_id][parallel_dims[i]]:
                    single_chunk_slice_dict[xds_id][
                        parallel_dims[i]
                    ] = chunk_slice_dict[xds_id][parallel_dims[i]][chunk_id]
                else:
                    empty_chunk = True

            if (
                empty_chunk
            ):  # The xds with xds_id has no data for the parallel chunk (no slice on one of the dims).
                single_chunk_slice_dict.pop(xds_id, None)

        input_parms["data_sel"] = single_chunk_slice_dict
        input_parms["parallel_coords"] = parallel_coords_chunk
        input_parms["chunk_id"] = i_chunks
        input_parms["parallel_dims"] = parallel_dims
        
        
        graph_list.append(dask.delayed(func_chunk)(dask.delayed(input_parms)))
        
        '''
        a = np.ravel_multi_index([127,0,0,1],[256,256,256,256])
        #a = np.ravel_multi_index([255,255,255,254],[256,256,256,256])
        print(a)
        '''

        '''
        #a = np.ravel_multi_index([127,0,0,1],[256,256,256,256])
        if chunk_id == 0:
            with dask.annotate(resources={'127.0.0.1': 1}):
                graph_list.append(dask.delayed(func_chunk)(dask.delayed(input_parms)))
        else:
            with dask.annotate(resources={'127.0.0.2': 1}):
                graph_list.append(dask.delayed(func_chunk)(dask.delayed(input_parms)))
        '''
    
    return graph_list

'''
{'type': 'Scheduler', 'id': 'Scheduler-42869e5b-f743-4b1f-9b28-068c7f361c2c', 'address': 'tcp://127.0.0.1:61159', 'services': {'dashboard': 8787}, 'started': 1671722207.311133, 'workers': {'tcp://127.0.0.1:61172': {'type': 'Worker', 'id': 0, 'host': '127.0.0.1', 'resources': {'ravel_ip': 2130706433}, 'local_directory': '/var/folders/l_/788x61bx0dg0fg4hg3zl9zf40000gp/T/dask-worker-space/worker-d7w_r8f1', 'name': 0, 'nthreads': 1, 'memory_limit': 2000000000, 'services': {'dashboard': 61177}, 'status': 'init', 'nanny': 'tcp://127.0.0.1:61165'}, 'tcp://127.0.0.1:61170': {'type': 'Worker', 'id': 2, 'host': '127.0.0.1', 'resources': {'ravel_ip': 2130706433}, 'local_directory': '/var/folders/l_/788x61bx0dg0fg4hg3zl9zf40000gp/T/dask-worker-space/worker-zn3c1mfb', 'name': 2, 'nthreads': 1, 'memory_limit': 2000000000, 'services': {'dashboard': 61175}, 'status': 'init', 'nanny': 'tcp://127.0.0.1:61162'}, 'tcp://127.0.0.1:61173': {'type': 'Worker', 'id': 3, 'host': '127.0.0.1', 'resources': {'ravel_ip': 2130706433}, 'local_directory': '/var/folders/l_/788x61bx0dg0fg4hg3zl9zf40000gp/T/dask-worker-space/worker-398bn6_u', 'name': 3, 'nthreads': 1, 'memory_limit': 2000000000, 'services': {'dashboard': 61174}, 'status': 'init', 'nanny': 'tcp://127.0.0.1:61163'}, 'tcp://127.0.0.1:61171': {'type': 'Worker', 'id': 1, 'host': '127.0.0.1', 'resources': {'ravel_ip': 2130706433}, 'local_directory': '/var/folders/l_/788x61bx0dg0fg4hg3zl9zf40000gp/T/dask-worker-space/worker-5iec4ntz', 'name': 1, 'nthreads': 1, 'memory_limit': 2000000000, 'services': {'dashboard': 61176}, 'status': 'init', 'nanny': 'tcp://127.0.0.1:61164'}}}

'''
