"""
Module to hold all the histogram related functions.
"""

import dask
import time

import numba as nb
from numba.typed import List
import numpy as np
import gc
#from memory_profiler import profile


@nb.njit(nogil=True, cache=True, fastmath=True)
def _merge_vis_lists(ps_vis_accum, vis_accum, npixu, npixv):

    print("start _merge_vis_lists")

    for uu in range(npixu):
        for vv in range(npixv):
            if len(vis_accum[uu][vv]) > 0:
                ps_vis_accum[uu][vv].extend(vis_accum[uu][vv])

    print("end _merge_vis_lists ")

    return ps_vis_accum


#@profile
def accumulate_uv_points(input_params):
    """
    Read in the input XDS and calculate a histogram per pixel in the UV plane.
    """
    gc.collect()

    from xradio.vis.load_processing_set import load_processing_set

    uvrange = np.asarray(sorted(input_params['uvrange']))
    uvcell = input_params['uvcell']
    nhistbin = input_params['nhistbin']
    npixu, npixv = input_params['npixels']

    ps_vis_accum = List([List([List.empty_list(nb.f8) for y in range(npixv)]) for z in range(npixu)])

    print(input_params["data_selection"].items())

    for ms_v4_name, slice_description in input_params["data_selection"].items():
        if input_params["input_data"] is None:
            ps = load_processing_set(input_params["input_data_store"],
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

        print(uvw.shape, vis.shape, flag.shape, freq.shape)

        def getsize(arr):
            return round(arr.nbytes/1024/1024, 2)

        print("UVW, Vis, flag, freq in MB",  getsize(uvw), getsize(vis), getsize(flag), getsize(freq))

        import sys, psutil
        process = psutil.Process()

        print("Size of PS, ms_xds in bytes ", sys.getsizeof(ps), sys.getsizeof(ms_xds))
        print("Total process memory in MB ", process.memory_info().rss/1024/1024)

        del ps, ms_xds
        gc.collect()
        print("Total process memory in MB after GC", process.memory_info().rss/1024/1024)

        vis = np.nan_to_num(vis)
        # Apply previously computed flags
        vis = np.asarray(vis*~flag)

        uvw = np.nan_to_num(uvw)
        t1 = time.time()
        uv_scaled = scale_uv_freq(np.asarray(uvw), np.asarray(freq), ref_freq)
        t2 = time.time()
        print(f"scale_uv_freq time {t2-t1} s")

        npt = uv_scaled.reshape([-1,2]).shape[0]

        # Create a list of visibilities per UV pixel - some might be entirely zeros, with no data.
        # Manually verified that the reshape works for a handful of random indices
        t1 = time.time()
        vis_accum = vis_per_uv_pixel(uv_scaled.reshape([-1,2]), vis.reshape([-1,2]), uvrange, uvcell, npixu, npixv, npt)
        t2 = time.time()
        print(f"vis_per_uv_pixel time {t2-t1} s")

        t1 = time.time()
        ps_vis_accum = _merge_vis_lists(ps_vis_accum, vis_accum, npixu, npixv)
        t2 = time.time()
        print(f"_merge_vis_lists time {t2-t1} s")

    #ps_vis_accum = np.asarray(ps_vis_accum)
    t1 = time.time()
    uv_med_grid, uv_std_grid, uv_npt_grid = calc_uv_stats(ps_vis_accum, npixu, npixv)
    t2 = time.time()
    print(f"calc_uv_stats time {t2-t1} s")

    return uv_med_grid, uv_std_grid, uv_npt_grid



@nb.njit(cache=True, nogil=True, fastmath=True)
def mad_std(data):
    """
    Calculate the median absolute deviation of the data. Cannot use a "built-in" function like
    astropy.stats.mad_std because we want to call this function inside numba.jit

    Inputs:
    data    : np.array - Data

    Returns:
    mad_std     : float - Median absolute deviation
    """

    median = np.median(data)
    mad = np.median(np.abs(data - median))
    std = 1.4826 * mad

    return std

    return np.median(np.abs(data - np.median(data)))


@nb.njit(cache=True, nogil=True, fastmath=True)
def calc_uv_stats(ps_vis_accum, npixu, npixv):
    """
    Calculate the median and standard deviation of the visibilities per UV pixel.

    Inputs:
    ps_vis_accum    : np.array - Visibilities
    npixu           : int - Number of pixels in U
    npixv           : int - Number of pixels in V

    Returns:
    uv_med_grid     : np.array - Median of the histogram per pixel
    uv_std_grid     : np.array - Standard deviation of the histogram per pixel
    """

    print("start calc_uv_stats")

    uv_med_grid = np.zeros((npixu, npixv))
    uv_std_grid = np.zeros((npixu, npixv))
    uv_npt_grid = np.zeros((npixu, npixv), dtype=np.int64)

    for uu in range(npixu):
        for vv in range(npixv):
            if len(ps_vis_accum[uu][vv]) > 0:
                uv_med_grid[uu, vv] = np.median(np.asarray(ps_vis_accum[uu][vv]))
                uv_std_grid[uu, vv] = mad_std(np.asarray(ps_vis_accum[uu][vv]))
                uv_npt_grid[uu, vv] = len(ps_vis_accum[uu][vv])
            else:
                uv_med_grid[uu, vv] = 0
                uv_std_grid[uu, vv] = 0

    print("end calc_uv_stats")
    return uv_med_grid, uv_std_grid, uv_npt_grid



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
@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def vis_per_uv_pixel(uv, vis, uvrange, uvcell, npixu, npixv, npt):
    """
    Accumulate list of visibilities per UV pixel
    """

    uvrange = sorted(uvrange)
    uv, vis = hermitian_conjugate(np.asarray(uv), np.asarray(vis))
    nptuv = np.zeros((npixu, npixv))

    # numba hack : Create an empty typed list
    vis_accum = List([List([List([float(x) for x in range(0)]) for y in range(npixv)]) for z in range(npixu)])

    stokesI = np.abs((vis[...,0] + vis[...,-1])/2.)

    idx = 0
    for didx, dat in enumerate(stokesI):
        if dat == 0:
            continue

        uvdist = np.sqrt(uv[didx, 0]**2 + uv[didx, 1]**2)

        if uvdist > uvrange[1] or uvdist < uvrange[0]:
            continue

        ubin = int((uv[didx, 0] + uvrange[1])//uvcell)
        vbin = int(uv[didx, 1]//uvcell)

        vis_accum[ubin][vbin].append(dat)
        nptuv[ubin, vbin] += 1

    print("Number of points appended ", np.sum(nptuv))
    print("Approx size ", np.sum(nptuv)*8/1024/1024)
    print("end vis_per_uv_pixel")
    return vis_accum



@nb.njit(nogil=True, cache=True, fastmath=True)
def _accum_means(mean1, npt1, mean2, npt2):
    """
    Calculate the aggregate mean given two input mean values.

    Inputs:
    mean1   : float - Mean 1
    npt1    : int - Number of points for mean 1
    mean2   : float - Mean 2
    npt2    : int - Number of points for mean 2

    Returns:
    mean    : float - Aggregate mean
    npt     : int - Number of points
    """

    npt = npt1 + npt2
    if npt == 0:
        return 0, 0

    mean = (mean1*npt1 + mean2*npt2)/npt

    return mean, npt


@nb.njit(nogil=True, cache=True, fastmath=True)
def _accum_std(mean1, std1, npt1, mean2, std2, npt2):
    """
    Calculate the aggregate standard deviation given two input standard deviation values.

    Inputs:
    mean1   : float - Mean 1
    std1    : float - Standard deviation 1
    npt1    : int - Number of points for mean 1
    mean2   : float - Mean 2
    std2    : float - Standard deviation 2
    npt2    : int - Number of points for mean 2

    Returns:
    std     : float - Aggregate standard deviation
    npt     : int - Number of points
    """

    npt = npt1 + npt2
    if npt < 2:
        return 0, 0

    var1 = ((npt1 - 1)*std1**2 + (npt2-1)*std2**2)/(npt1 + npt2 - 1)
    var2 = ((npt1*npt2) * (mean1 - mean2)**2)/((npt1 + npt2)*(npt1 + npt2 - 1))

    std = np.sqrt(var1 + var2)

    return std, npt


#@nb.jit(nopython=True, nogil=True, cache=True)
def merge_uv_grids(graph, input_params):
    """
    Given the list of results from accumulate_uv_points, merge the UV grids together to compute stats

    Inputs:
    graph        : list(np.array, np.array) - Each element contains the median, std dev and npt UV grid
    npixu        : Number of pixels in U
    npixv        : Number of pixels in V
    nhistbin     : Number of histogram bins

    Returns:
    accum_uvmed   : np.array - Median of the histogram per pixel
    accum_uvstd   : np.array - Standard deviation of the histogram per pixel
    """

    npixu = input_params['npixels'][0]
    npixv = input_params['npixels'][1]
    nhistbin = input_params['nhistbin']

    # Graph is a tuple of 2 nested elements : (median, std, npt) from each node
    # of the input DAG merge_uv_grids should accumulate these grids onto a
    # single one, and return it.

    nchunk = len(graph)

    accum_uv_med = np.zeros((npixu, npixv))
    accum_uv_std = np.zeros((npixu, npixv))
    uvnpt = np.zeros((npixu, npixv), dtype=np.int64)

    med0 = graph[0][0]
    std0 = graph[0][1]
    npt0 = graph[0][2]

    med1 = graph[1][0]
    std1 = graph[1][1]
    npt1 = graph[1][2]

    for uu in range(npixu):
        for vv in range(npixv):
            accum_uv_med[uu,vv], uvnpt[uu,vv] = _accum_means(med0[uu,vv], npt0[uu,vv], med1[uu,vv], npt1[uu,vv])
            accum_uv_std[uu,vv], __ = _accum_std(med0[uu,vv], std0[uu,vv], npt0[uu,vv], med1[uu,vv], std1[uu,vv], npt1[uu,vv])


    np.savez('accum_uv_med.npz', accum_uv_med, accum_uv_std, uvnpt)
    return accum_uv_med, accum_uv_std, uvnpt



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
        uv_scaled[:,:,ffidx,0] = uvw[:,:,0] * (1 + delta_nu)
        uv_scaled[:,:,ffidx,1] = uvw[:,:,1] * (1 + delta_nu)

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


