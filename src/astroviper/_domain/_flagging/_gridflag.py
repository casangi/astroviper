import numpy as np
import numba as nb

from xradio.vis.read_processing_set import read_processing_set
import xarray as xr

from astroviper._domain._flagging._gridflag_histogram import compute_uv_histogram, merge_uv_grids, apply_flags

import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
dask.config.set(scheduler='synchronous')

def create_chunks(baseline_id, frequencies, baseline_chunk_size, freq_chunk_size):
    """
    Given the input baseline_ids and frequencies, return a list of the start and end slices
    to parallelize over.

    Inputs:
    baseline_id             : Baseline IDs, np.array
    frequencies             : Frequencies, np.array
    baseline_chunk_size     : Number of baseline IDs to process per chunk, int
    freq_chunk_size         : Number of frequencies to process per chunk, int

    Returns:
    baseline_chunks         : List of start and end slices for baseline IDs
    freq_chunks             : List of start and end slices for frequencies
    """

    baseline_chunks = []
    freq_chunks = []

    nbaseline = len(baseline_id)
    nfreq = len(frequencies)

    for bb in range(0, len(baseline_id), baseline_chunk_size):
        if bb+baseline_chunk_size > nbaseline:
            baseline_chunks.append(slice(baseline_id[bb], None))
        else:
            baseline_chunks.append(slice(baseline_id[bb],baseline_id[bb+baseline_chunk_size]))

    for ff in range(0, len(frequencies), freq_chunk_size):
        if ff+freq_chunk_size > nfreq:
            freq_chunks.append(slice(frequencies[ff], None))
        else:
            freq_chunks.append(slice(frequencies[ff],frequencies[ff+freq_chunk_size]))

    return baseline_chunks, freq_chunks



#@click.command(context_settings=ctx)
#@click.argument('zarr', type=click.Path(exists=True))
#@click.option('--uvrange', nargs=2, type=int, default=[None, None], show_default=True, help='UV range within which to flag')
#@click.option('--uvcell', default=None, type=int, show_default=True, help='UV cell size to use for flagging.')
#@click.option('--nhistbin', default=100, type=int, show_default=True, help='Number of bins in the visibility histogram per bin')
#@click.option('--nsigma', default=5, type=int, show_default=True, help='Sigma threshold for flagging')
def gridflag(zarr, uvrange, uvcell, nhistbin, nsigma):
    """
    Given the input visibility data set (in Zarr format), flags the data using
    the GRIDflag algorithm.

    Please always specify --uvrange and --uvcell. Future versions will attempt
    to auto-discover these parameters but it is currently not implemented.
    """

    baseline_chunk_size = 218
    freq_chunk_size = 22

    #client = Client()
    #client = Client(LocalCluster(n_workers=2, threads_per_worker=1))

    # Lazy load the processing set
    ps = read_processing_set(zarr)

    npixu = 2*uvrange[1]//uvcell
    npixv = uvrange[1]//uvcell

    input_params = {}
    input_params['uvrange'] = uvrange
    input_params['uvcell'] = uvcell
    input_params['nhistbin'] = nhistbin
    input_params['npixels'] = [npixu, npixv]
    input_params['input_data'] = zarr

    output = []
    for idx in range(len(ps)):
        # Chunk up each MSv4 in the processing set independently
        ds = ps.get(idx)
        baseline_id = ds.coords['baseline_id']
        frequencies = ds.coords['frequency']

        baseline_chunks, freq_chunks = create_chunks(baseline_id, frequencies, baseline_chunk_size, freq_chunk_size)

        nchunk = len(baseline_chunks) * len(freq_chunks)
        input_params['nchunks'] = nchunk

        for bb in range(len(baseline_chunks)):
            for ff in range(len(freq_chunks)):
                #print(bb, ff)
                loc_dict = {'baseline_id':baseline_chunks[bb], 'frequency':freq_chunks[ff]}
                this_ds = ds.loc[loc_dict]

                this_output = compute_uv_histogram(this_ds, input_params)
                output.append(this_output)

    result = dask.compute(output)

    uv_med, uv_std = merge_uv_grids(result, npixu, npixv, nhistbin)

    import uuid
    tmpname = str(uuid.uuid4()) + '_histogram.npz'
    np.savez(tmpname, vis_hist_med=uv_med, vis_hist_std=uv_std)

    min_thresh = uv_med - nsigma*uv_std
    max_thresh = uv_med + nsigma*uv_std

    # Apply flags to chunks
    output = []
    for idx in range(len(ps)):
        # Chunk up each MSv4 in the processing set independently
        ds = ps.get(idx)
        baseline_id = ds.coords['baseline_id']
        frequencies = ds.coords['frequency']

        baseline_chunks, freq_chunks = create_chunks(baseline_id, frequencies, baseline_chunk_size, freq_chunk_size)

        nchunk = len(baseline_chunks) * len(freq_chunks)
        input_params['nchunks'] = nchunk

        for bb in range(len(baseline_chunks)):
            for ff in range(len(freq_chunks)):
                print(bb, ff)
                loc_dict = {'baseline_id':baseline_chunks[bb], 'frequency':freq_chunks[ff]}
                this_ds = ds.loc[loc_dict]

                this_output = apply_flags(this_ds, input_params, min_thresh, max_thresh)
                output.append(this_output)

    result = dask.compute(output)
    print("Done")

    # Write back to disk...
    #print("Updating dataset")
    #count = 0
    #for idx in range(len(ps)):
    #    for bb in range(len(baseline_chunks)):
    #        for ff in range(len(freq_chunks)):
    #            print(bb, ff)
    #            #loc_dict = {'baseline_id':baseline_chunks[bb], 'frequency':freq_chunks[ff]}
    #            #print(result[0][idx])
    #            #print(ds)
    #            #ds.loc[loc_dict] = result[0][idx]
    #            ds.update(result[0][count])
    #            count += 1

    #ps.to_zarr("flagged.zarr", mode='w')
