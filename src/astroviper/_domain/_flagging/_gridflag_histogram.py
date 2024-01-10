"""
Module to hold all the histogram related functions.
"""

import dask

import numba as nb
import numpy as np

from scipy.stats import median_abs_deviation


#@dask.delayed
def compute_uv_histogram(input_params):
    """
    Read in the input XDS and calculate a histogram per pixel in the UV plane.
    """
    print('Processing task with id: ', input_params['task_id'])
    from xradio.vis.load_processing_set import load_processing_set

    uvrange = np.asarray(sorted(input_params['uvrange']))
    uvcell = input_params['uvcell']
    nhistbin = input_params['nhistbin']
    npixu, npixv = input_params['npixels']

    accum_uv_hist_med = np.zeros((npixu, npixv))
    accum_uv_hist_std = np.zeros((npixu, npixv))
    vis_accum = np.empty((npixu, npixv), dtype=object)

    for ms_v4_name, slice_description in input_params["data_selection"].items():
        if input_params["input_data"] is None:
            ps = load_processing_set(
                    ps_name=input_params["input_data_store"],
                    sel_parms={ms_v4_name: slice_description},
                )
        else:
            ps = input_params["input_data"][ms_v4_name].isel(slice_description) #In memory

        ms_xds = ps.get(0)

        #ref_freq = float(ms_xds.frequency.attrs['reference_frequency']['data'])
        ref_freq = np.mean(ms_xds.frequency).values 
   
        min_baseline = ms_xds.baseline_id.min().data
        max_baseline = ms_xds.baseline_id.max().data

        # Drop rows that are outside the selected UV range
        ms_xds = ms_xds.where(((ms_xds.UVW[...,0]**2 + ms_xds.UVW[...,1]**2 > uvrange[0]*uvrange[0]) & (ms_xds.UVW[...,0]**2 + ms_xds.UVW[...,1]**2 < uvrange[1]*uvrange[1])), drop=True)

        uvw = ms_xds.UVW.data
        vis = ms_xds.VISIBILITY.data
        flag = ms_xds.FLAG.data.astype(bool)
        freq = ms_xds.frequency.data

        # Flip flags and replace NaNs with zeros, so they flag the corresponding visibilities
        #flag = np.nan_to_num(~flag).astype(bool)
        vis = np.nan_to_num(vis)
        # Flag visibilities
        #vis = np.asarray(vis*~flag)

        uvw = np.nan_to_num(uvw)
        uv_scaled = scale_uv_freq(np.asarray(uvw), np.asarray(freq), ref_freq)

        # Create a histogram per UV pixel - some might be entirely zeros, with no data.
        # Manually verified that the reshape works for a handful of random indices
        uv_histogram(vis_accum, uv_scaled.reshape([-1,2]), vis.reshape([-1,2]), uvrange, uvcell, npixu, npixv)


    return vis_accum


#@nb.jit(nopython=True, nogil=True, cache=True)
def merge_uv_grids(results, input_parms):
    """
    Given the list of results from compute_uv_histogram, merge the UV grids together to compute stats

    Inputs:
    results      : np.array(list) - All points falling within a UV cell
    npixu        : Number of pixels in U
    npixv        : Number of pixels in V
    nhistbin     : Number of histogram bins

    Returns:
    accum_uv_hist_med   : np.array - Median of the histogram per pixel
    accum_uv_hist_std   : np.array - Standard deviation of the histogram per pixel
    """

    npixu = input_parms['npixu']
    npixv = input_parms['npixv']
    nhistbin = input_parms['nhistbin']

    nchunk = len(results)
    accum_uv_hist_med = np.zeros((npixu, npixv))
    accum_uv_hist_std = np.zeros((npixu, npixv))

    #for nn in range(nchunk):
    #    print(results[0][nn].shape)
    #    for uu in range(npixu):
    #        for vv in range(npixv):
    #            if len(results[0][nn][uu,vv]) > 0:
    #                print(f"nn {nn} uu {uu} vv {vv}")
    #                print(len(results[0][nn][uu,vv]))
    #                input()


    for uu in range(npixu):
        for vv in range(npixv):
            concat_list = []
            for nn in range(nchunk):
                if len(results[nn][uu,vv]) > 0:
                    concat_list.extend(results[nn][uu,vv])

            if len(concat_list) > 0:
                concat_list = np.nan_to_num(concat_list)
                accum_uv_hist_med[uu, vv] = np.median(concat_list[concat_list != 0])
                accum_uv_hist_std[uu, vv] = median_abs_deviation(concat_list[concat_list != 0])

    return [accum_uv_hist_med, accum_uv_hist_std]



@nb.jit(nopython=True, nogil=True, cache=True)
def hermitian_conjugate(uv, vis):
    """
    Given the input UV coordinates and visibilities, calculate the hermitian conjugate
    of the visibilities.
    """

    nvis = vis.shape[0]
    for nn in range(nvis):
        if uv[nn, 1] < 0:
            uv[nn, 1] *= -1
            vis[nn] = vis[nn].conj()

    return uv, vis


#@profile
#@nb.jit(nopython=True, nogil=True, cache=True)
def uv_histogram(vis_hist,uv, vis, uvrange, uvcell, npixu, npixv):
    """
    Generate a histogram per UV pixel, given the input UV coordinates & visibilities.
    """

    uvrange = sorted(uvrange)
    uv, vis = hermitian_conjugate(np.asarray(uv), np.asarray(vis))

    #vis_hist = np.zeros((npixu, npixv), dtype=object)

    stokesI = np.abs((vis[...,0] + vis[...,-1])/2.)

    # Initialize empty lists
    for uu in range(npixu):
        for vv in range(npixv):
            vis_hist[uu][vv] = []

    for idx, dat in enumerate(stokesI):
        uvdist = np.sqrt(uv[idx, 0]**2 + uv[idx, 1]**2)

        if uvdist > uvrange[1] or uvdist < uvrange[0]:
            continue

        ubin = int((uv[idx, 0] + uvrange[1])//uvcell)
        vbin = int(uv[idx, 1]//uvcell)

        if dat != 0:
            vis_hist[ubin,vbin].append(dat)

    #return vis_hist


@nb.jit(nopython=True, nogil=True, cache=True)
def scale_uv_freq(uvw, frequency, ref_freq):
    """
    Given the input UVW and frequency coordinates, scale the UVW coordinates
    correctly per frequency.

    This only returns the scaled U and V values - it drops W.
    """

    shape = uvw.shape
    uv_scaled = np.zeros((shape[0], shape[1], frequency.size, 2))

    for ffidx, ff in enumerate(frequency):
        delta_nu = (ff - ref_freq)/ref_freq
        uv_scaled[:,:,ffidx,0] = uvw[:,:,0] * (1 + delta_nu/ff)
        uv_scaled[:,:,ffidx,1] = uvw[:,:,1] * (1 + delta_nu/ff)

    return uv_scaled



@nb.jit(nopython=True, nogil=True, cache=True)
def _loop_and_apply_flags(uv_scaled, vis, flag, uvrange, uvcell, min_thresh, max_thresh):
    """
    Loop over the ND data and apply flags if the data falls outside the thresholds.

    Inputs:
    uv_scaled   : np.array - Scaled UV coordinates
    vis         : np.array - Visibilities
    flag        : np.array - Flags
    uvrange     : np.array - UV range
    uvcell      : float - UV cell size
    min_thresh  : np.array - Minimum threshold per pixel
    max_thresh  : np.array - Maximum threshold per pixel

    Returns:
    flag        : np.array - Flags
    """

    for ii in range(uv_scaled.shape[0]):
        for jj in range(uv_scaled.shape[1]):
            for kk in range(uv_scaled.shape[2]):
                # Hermitian conjugate
                if uv_scaled[ii,jj,kk,1] < 0:
                    uv_scaled[ii,jj,kk,1] *= -1
                    vis[ii,jj,kk] = vis[ii,jj,kk].conj()

                uvdist = np.sqrt(uv_scaled[ii,jj,kk,0]**2 + uv_scaled[ii,jj,kk,1]**2)

                if uvdist > uvrange[1] or uvdist < uvrange[0]:
                    continue

                ubin = int((uv_scaled[ii,jj,kk,0] + uvrange[1])//uvcell)
                vbin = int(uv_scaled[ii,jj,kk,1]//uvcell)

                stokesI = np.abs((vis[ii,jj,kk,0] + vis[ii,jj,kk,-1])/2.)

                if stokesI < min_thresh[ubin, vbin] or stokesI > max_thresh[ubin, vbin]:
                    flag[ii,jj,kk] = True

    return flag



#@dask.delayed
def apply_flags(ms_xds, input_params, min_thresh, max_thresh):
    """
    Apply flags to the input XDS, given the min and max thresholds.
    The flags are applied in-place.

    Inputs:
    ms_xds      : xarray Dataset - Input XDS
    input_params: dict - Dictionary of input parameters
    min_thresh  : np.array - Minimum threshold per pixel
    max_thresh  : np.array - Maximum threshold per pixel

    Returns:
    None
    """

    # in Hz
    ref_freq = float(ms_xds.frequency.attrs['reference_frequency']['data'])

    uvrange = np.asarray(sorted(input_params['uvrange']))
    uvcell = input_params['uvcell']
    nhistbin = input_params['nhistbin']
    npixu, npixv = input_params['npixels']
    nchunks = input_params['nchunks']

    # Drop rows that are outside the selected UV range
    ms_xds = ms_xds.where(((ms_xds.UVW[...,0]**2 + ms_xds.UVW[...,1]**2 > uvrange[0]*uvrange[0]) & (ms_xds.UVW[...,0]**2 + ms_xds.UVW[...,1]**2 < uvrange[1]*uvrange[1])), drop=True)

    uvw = ms_xds.UVW.data
    vis = ms_xds.VISIBILITY.data
    flag = ms_xds.FLAG.data.astype(bool)
    freq = ms_xds.frequency.data

    uvw = np.nan_to_num(uvw)
    uv_scaled = scale_uv_freq(np.asarray(uvw), np.asarray(freq), ref_freq)

    #print("uvw shape ", uvw.shape)
    #print("uv_scaled shape ", uv_scaled.shape)
    #print("vis shape ", vis.shape)
    #print("flag shape ", flag.shape)

    flag = _loop_and_apply_flags(uv_scaled, vis, flag, uvrange, uvcell, min_thresh, max_thresh)

    ms_xds.FLAG.data = flag

    return ms_xds


