import numpy as np
from numba import jit, njit, prange
import xarray
from astropy import constants
from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import *


def sgrid(
    uvw: np.ndarray,
    dphase: float,
    freq: float,
    c: float,
    scale: np.ndarray,
    offset: np.ndarray,
    sampling: int,
    pos: np.ndarray,
    loc,
    off,
):
    """
    uvw is a 3 element array
    """
    pos[:2] = scale[:2] * uvw[:2] * freq / c + (offset[:2])
    loc[:2] = np.round(pos[:2])
    off[:2] = np.round((loc[:2] - pos[:2]) * sampling)
    phase = -2.0 * np.pi * dphase * freq / c
    phasor = complex(np.cos(phase), np.sin(phase))
    return phasor


### This is a translation of fgridft::dgrid
def dgrid(
    uvw: np.ndarray,
    dphase: np.ndarray,
    values: np.ndarray,
    flag,
    scale,
    offset,
    grid,
    freq,
    c,
    support,
    sampling,
    convFunc,
    chanmap,
    polmap,
):
    """
    uvw should have the shape (nt, nb, 3)
    values has shape (nt, nb, nvispol, nvischan) # Corrected understanding of shape
    flag has shape (nt, nb, nvispol, nvischan)   # Corrected understanding of shape
    dphase should have the shape (nt,nb,)
    freq should have shape (nvischan,) # Corrected understanding of shape
    scale (3) matching u,v,w
    offset(3) where is u,v,w =0 on the pixel grid
    convfunc (support//2 * sampling)
    chanmap (nvischan)
    polmap (nvispol)
    """
    nvispol = values.shape[3]
    nvischan = values.shape[2]
    nchan = grid.shape[0]
    nx = grid.shape[2]
    ny = grid.shape[3]
    npol = grid.shape[1]
    nt = values.shape[0]
    nb = values.shape[1]
    supp_centre = support // 2
    supp_beg = -supp_centre
    supp_end = supp_beg + support

    for t in range(nt):
        for b in range(nb):
            # Corrected loop order to match values and flag shape
            for ipol in range(nvispol):
                apol = polmap[ipol]
                if 0 <= apol < npol:
                    for ichan in range(nvischan):
                        achan = chanmap[ichan]
                        if 0 <= achan < nchan:
                            pos = np.zeros(2)
                            loc = np.zeros(2, dtype=int)
                            off = np.zeros(2, dtype=int)

                            # Pass the correct frequency for the current channel
                            phasor = sgrid(
                                uvw[t, b],
                                dphase[t, b],
                                freq[ichan],
                                c,
                                scale,
                                offset,
                                sampling,
                                pos,
                                loc,
                                off,
                            )

                            # Corrected indexing for flag and values
                            if not flag[
                                t, b, ichan, ipol
                            ]:  # Access flag using ipol then ichan
                                ogrid = (
                                    ((loc[0] + supp_beg) >= 0)
                                    and ((loc[0] + supp_end) < nx)
                                    and ((loc[1] + supp_beg) >= 0)
                                    and ((loc[1] + supp_end) < ny)
                                )

                                if ogrid:
                                    nvalue = 0.0
                                    norm = 0.0
                                    for iy in range(supp_beg, supp_end):
                                        iloc2 = abs(sampling * iy + off[1])
                                        # Ensure iloc2 is within bounds for convFunc
                                        if iloc2 < len(convFunc):
                                            wty = convFunc[iloc2]
                                            for ix in range(supp_beg, supp_end):
                                                iloc1 = abs(sampling * ix + off[0])
                                                # Ensure iloc1 is within bounds for convFunc
                                                if iloc1 < len(convFunc):
                                                    wtx = convFunc[iloc1]
                                                    wt = wtx * wty
                                                    norm += wt
                                                    # Access grid using achan then apol, and corrected loc indexing
                                                    grid_val = grid[
                                                        achan,
                                                        apol,
                                                        loc[0] + ix,
                                                        loc[1] + iy,
                                                    ]
                                                    nvalue += wt * grid_val
                                    if norm != 0.0:
                                        # Access values using ipol then ichan
                                        values[t, b, ichan, ipol] += (
                                            nvalue * np.conj(phasor)
                                        ) / norm
    return


###optimizied with help of Gemini


@njit
def dgrid_optimized(
    uvw,
    dphase,
    values,
    flag,
    scale,
    offset,
    grid,
    freq,
    c,
    support,
    sampling,
    convFunc,
    chanmap,
    polmap,
):

    nvispol = values.shape[3]
    nvischan = values.shape[2]
    nchan = grid.shape[0]
    nx = grid.shape[2]
    ny = grid.shape[3]
    npol = grid.shape[1]
    nt = values.shape[0]
    nb = values.shape[1]

    supp_centre = support // 2
    supp_beg = -supp_centre
    supp_end = supp_beg + support
    # Pre-calculate sgrid components for all time and baseline points
    # This replaces the loop over t and b and the call to sgrid inside.
    # Expanding sgrid logic:
    # pos = (
    #    scale[:2] * uvw[:, :, :2] * (freq[0] / c) + offset[:2]
    # )  # Using freq[0] as a placeholder, need to handle freq dependency per channel later
    # loc = np.round(pos).astype(np.int64)
    # off = np.round((loc - pos) * sampling).astype(np.int64)

    # Recalculate pos, loc, off, and phasor per channel dependency outside the inner loops.
    pos_chan = np.empty((nt, nb, nvischan, 2), dtype=np.float64)
    loc_chan = np.empty((nt, nb, nvischan, 2), dtype=np.int64)
    off_chan = np.empty((nt, nb, nvischan, 2), dtype=np.int64)
    phasor_chan = np.empty((nt, nb, nvischan), dtype=np.complex128)

    for ichan in range(nvischan):
        achan = chanmap[ichan]
        if 0 <= achan < nchan:
            # sgrid logic applied across nt and nb
            pos_chan[:, :, ichan, :] = (
                scale[:2] * uvw[:, :, :2] * (freq[ichan] / c) + offset[:2]
            )
            loc_chan[:, :, ichan, :] = np.round(pos_chan[:, :, ichan, :]).astype(
                np.int64
            )
            off_chan[:, :, ichan, :] = np.round(
                (loc_chan[:, :, ichan, :] - pos_chan[:, :, ichan, :]) * sampling
            ).astype(np.int64)
            phase = -2.0 * np.pi * dphase * freq[ichan] / c
            phasor_chan[:, :, ichan] = np.cos(phase) + 1j * np.sin(phase)

    # Pre-calculate convolution kernel weights for all possible offsets
    # The offsets for x and y dimensions are independent, so we can calculate 1D kernels and combine them.
    max_offset_abs = (
        support // 2 + 1
    ) * sampling  # Maximum possible absolute offset in sampling units
    conv_kernel_1d = np.empty(max_offset_abs + 1, dtype=convFunc.dtype)
    for i in range(max_offset_abs):
        conv_kernel_1d[i] = convFunc[i]

    # Loop over time, baseline, channel, and polarization
    for t in range(nt):
        for b in range(nb):
            for ichan in range(nvischan):
                achan = chanmap[ichan]
                if 0 <= achan < nchan:
                    loc_tbi = loc_chan[t, b, ichan, :]
                    off_tbi = off_chan[t, b, ichan, :]
                    phasor_tbi = phasor_chan[t, b, ichan]

                    ogrid = (
                        ((loc_tbi[0] + supp_beg) >= 0)
                        and ((loc_tbi[0] + supp_end) < nx)
                        and ((loc_tbi[1] + supp_beg) >= 0)
                        and ((loc_tbi[1] + supp_end) < ny)
                    )

                    if ogrid:
                        for ipol in range(nvispol):
                            apol = polmap[ipol]
                            if (not flag[t, b, ichan, ipol]) and (0 <= apol < npol):
                                # Vectorized convolution
                                # Extract grid patch
                                grid_patch = grid[
                                    achan,
                                    apol,
                                    loc_tbi[0] + supp_beg : loc_tbi[0] + supp_end,
                                    loc_tbi[1] + supp_beg : loc_tbi[1] + supp_end,
                                ]

                                # Create 2D kernel from pre-calculated 1D kernels based on offset
                                offset_x = abs(off_tbi[0])
                                offset_y = abs(off_tbi[1])
                                kernel_x = conv_kernel_1d[
                                    sampling * np.abs(np.arange(supp_beg, supp_end))
                                    + offset_x
                                ]
                                kernel_y = conv_kernel_1d[
                                    sampling * np.abs(np.arange(supp_beg, supp_end))
                                    + offset_y
                                ]
                                conv_kernel_2d = kernel_x.reshape(
                                    -1, 1
                                ) * kernel_y.reshape(1, -1)

                                # Perform element-wise multiplication and sum for nvalue and norm
                                nvalue = np.sum(conv_kernel_2d * grid_patch)
                                norm = np.sum(
                                    conv_kernel_2d
                                )  # The sum of weights is the norm

                                if norm != 0.0:
                                    values[t, b, ichan, ipol] += (
                                        nvalue * np.conj(phasor_tbi)
                                    ) / norm
    return


# some manual optimization of dgrid


def dgrid2(
    uvw,
    dphase,
    values,
    flag,
    scale,
    offset,
    grid,
    freq,
    c,
    support,
    sampling,
    convFunc,
    chanmap,
    polmap,
):
    nvispol = values.shape[3]
    nvischan = values.shape[2]
    nchan, npol, nx, ny = grid.shape
    nt, nb = values.shape[:2]
    supp_centre = support // 2
    supp_beg = -supp_centre
    supp_end = supp_beg + support

    for t in range(nt):
        for b in range(nb):
            for ipol in range(nvispol):
                apol = polmap[ipol]
                if 0 <= apol < npol:
                    for ichan in range(nvischan):
                        achan = chanmap[ichan]
                        if 0 <= achan < nchan and not flag[t, b, ichan, ipol]:
                            pos = np.empty(2)
                            loc = np.empty(2, dtype=int)
                            off = np.empty(2, dtype=int)

                            phasor = sgrid(
                                uvw[t, b],
                                dphase[t, b],
                                freq[ichan],
                                c,
                                scale,
                                offset,
                                sampling,
                                pos,
                                loc,
                                off,
                            )

                            x0, y0 = loc
                            if (
                                (x0 + supp_beg >= 0)
                                and (x0 + supp_end < nx)
                                and (y0 + supp_beg >= 0)
                                and (y0 + supp_end < ny)
                            ):
                                nvalue = 0.0
                                norm = 0.0
                                for iy in range(supp_beg, supp_end):
                                    iloc2 = abs(sampling * iy + off[1])
                                    if iloc2 < len(convFunc):
                                        wty = convFunc[iloc2]
                                        for ix in range(supp_beg, supp_end):
                                            iloc1 = abs(sampling * ix + off[0])
                                            if iloc1 < len(convFunc):
                                                wtx = convFunc[iloc1]
                                                wt = wtx * wty
                                                norm += wt
                                                nvalue += (
                                                    wt
                                                    * grid[
                                                        achan,
                                                        apol,
                                                        x0 + ix,
                                                        y0 + iy,
                                                    ]
                                                )
                                if norm != 0.0:
                                    values[t, b, ichan, ipol] += (
                                        nvalue * np.conj(phasor)
                                    ) / norm


# jit version of dgrid2


@njit
def sgrid_numba(uvw, dphase, freq, c, scale, offset, sampling):
    pos = np.empty(2, dtype=np.float64)
    loc = np.empty(2, dtype=np.int32)
    off = np.empty(2, dtype=np.int32)

    pos[0] = scale[0] * uvw[0] * freq / c + offset[0]
    pos[1] = scale[1] * uvw[1] * freq / c + offset[1]

    loc[0] = int(np.round(pos[0]))
    loc[1] = int(np.round(pos[1]))

    off[0] = int(np.round((loc[0] - pos[0]) * sampling))
    off[1] = int(np.round((loc[1] - pos[1]) * sampling))

    phase = -2.0 * np.pi * dphase * freq / c
    phasor = np.cos(phase) + 1j * np.sin(phase)

    return phasor, loc, off


@njit
def dgrid_numba(
    uvw,
    dphase,
    values,
    flag,
    scale,
    offset,
    grid,
    freq,
    c,
    support,
    sampling,
    convFunc,
    chanmap,
    polmap,
):
    nvispol = values.shape[3]
    nvischan = values.shape[2]
    nchan, npol, ny, nx = grid.shape
    nt, nb = values.shape[:2]
    supp_centre = support // 2
    supp_beg = -supp_centre
    supp_end = supp_beg + support
    print(
        "polmap",
        polmap,
        "chanmap",
        chanmap,
        "nvispol",
        nvispol,
        "nvischan",
        nvischan,
    )
    for t in prange(nt):
        for b in range(nb):
            for ipol in range(nvispol):
                apol = polmap[ipol]
                if 0 <= apol < npol:
                    for ichan in range(nvischan):
                        achan = chanmap[ichan]
                        if 0 <= achan < nchan and not flag[t, b, ichan, ipol]:
                            phasor, loc, off = sgrid_numba(
                                uvw[t, b],
                                dphase[t, b],
                                freq[ichan],
                                c,
                                scale,
                                offset,
                                sampling,
                            )

                            x0, y0 = loc
                            if (
                                (x0 + supp_beg >= 0)
                                and (x0 + supp_end < nx)
                                and (y0 + supp_beg >= 0)
                                and (y0 + supp_end < ny)
                            ):
                                nvalue = 0.0
                                norm = 0.0
                                for iy in range(supp_beg, supp_end):
                                    iloc2 = abs(sampling * iy + off[1])
                                    if iloc2 < len(convFunc):
                                        wty = convFunc[iloc2]
                                        for ix in range(supp_beg, supp_end):
                                            iloc1 = abs(sampling * ix + off[0])
                                            if iloc1 < len(convFunc):
                                                wtx = convFunc[iloc1]
                                                wt = wtx * wty
                                                norm += wt
                                                nvalue += (
                                                    wt
                                                    * grid[
                                                        achan,
                                                        apol,
                                                        x0 + iy,
                                                        y0 + ix,
                                                    ]
                                                )
                                if norm != 0.0:
                                    values[t, b, ichan, ipol] += (
                                        nvalue * np.conj(phasor)
                                    ) / norm


def degrid_spheroid_ms4(
    vis: xarray.core.datatree.DataTree,
    grid: np.ndarray,
    pixelincr: np.ndarray,
    support: int = 7,
    sampling: int = 100,
    incremental: bool = False,
    whichFunc=0,
):
    """

    Parameters
    ----------
    vis : xarray.core.datatree.DataTree
        ms4 xradio datatree.
    grid : np.ndarray
        model gridded visibilities.
    pixelincr : np.ndarray
        [dx,dy] 2 values in radians
    support : int, optional
        Support of the prolate spheroidal conv func.
        The default is 7.
    sampling : int, optional
        oversampling of prolate spheroidal conv func .
        The default is 100.
    incremental : bool, optional
        if True add model visibilities to existing VISIBILITY_MODEL.
        The default is False.
    whichFunc : TYPE, optional
        0 use degrid vectorized and jit
        1 use degrid loop and jit
        The default is 0.

    TODO: The chanmap and polmap may be totally wrong

    Returns
    -------
    None.

    """
    func = dgrid_optimized
    if whichFunc == 1:
        func = dgrid_numba
    uvw = vis.UVW.data
    dims = vis.dims
    # as we are not gridding away from phase_center
    dphase = np.zeros((dims["time"], dims["baseline_id"]), dtype=float)

    if incremental or "VISIBILITY_MODEL" not in vis:
        modvis = np.zeros(
            (
                dims["time"],
                dims["baseline_id"],
                dims["frequency"],
                dims["polarization"],
            ),
            dtype=np.complex128(),
        )
    else:
        modvis = vis.VISIBILITY_MODEL.data
    flag = vis.FLAG.data
    # lets flag NaN UVW
    ###testoo
    flag[:, :, :, :] = False
    ######
    uvwmask = np.isnan(vis.UVW)
    print("number of nan UVW", np.sum(uvwmask) / 3)
    nan_it, nan_ib = np.where(uvwmask.any(dim="uvw_label"))
    print("sum of flag bef", np.sum(flag))
    # flag[nan_it, nan_ib, :, :] = True
    for k in range(len(nan_it)):
        flag[nan_it[k], nan_ib[k], :, :] = True
    print("sum of flags aft", np.sum(flag))
    nx, ny = grid.shape[-2:]
    # for sake of completeness making size 3 ofr uvw scales and offset
    scale = np.array([-pixelincr[0] * nx, -pixelincr[1] * ny, 0.0]).astype(np.float64)
    offset = np.array([nx / 2, ny / 2, 0.0]).astype(float)
    freq = vis.frequency.data
    c = constants.c.value
    convFunc = create_prolate_spheroidal_kernel_1D(sampling, support)
    nchan = grid.shape[0]
    npol = grid.shape[1]
    ## we need to pass chan/pol maps or determine them from info
    chanmap = np.round(np.arange(dims["frequency"]) / dims["frequency"] * nchan).astype(
        int
    )  # Channel mapping
    polmap = np.round(np.arange(dims["polarization"]) % npol).astype(
        int
    )  # this is wrong most probably
    #    dgrid_numba(
    func(
        uvw,
        dphase,
        modvis,
        flag,
        scale,
        offset,
        grid,
        freq,
        c,
        support,
        sampling,
        convFunc,
        chanmap,
        polmap,
    )
    datdims = vis.VISIBILITY.dims
    datcoords = vis.VISIBILITY.coords
    print("MODVIS", np.max(modvis))
    modvis_da = xarray.DataArray(modvis, coords=datcoords, dims=datdims)
    vis["VISIBILITY_MODEL"] = modvis_da
    print("post mod", np.max(vis.VISIBILITY_MODEL))
