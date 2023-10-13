from numba import jit
import numpy as np
import math


def _standard_grid_numpy_wrap(vis_data, uvw, weight, freq_chan, cgk_1D, grid_parms):
    """
    Wraps the jit gridder code.

    Parameters
    ----------
    grid : complex array
        (n_chan, n_pol, n_u, n_v)
    sum_weight : float array
        (n_chan, n_pol)
    uvw  : float array
        (n_time, n_baseline, 3)
    freq_chan : float array
        (n_chan)
    weight : float array
        (n_time, n_baseline, n_vis_chan)
    cgk_1D : float array
        (oversampling*(support//2 + 1))
    grid_parms : dictionary
        keys ('image_size','cell','oversampling','support')

    Returns
    -------
    grid : complex array
        (1,n_imag_chan,n_imag_pol,n_u,n_v)
    """

    n_chan = weight.shape[2]
    if grid_parms["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = grid_parms["image_size_padded"]
    delta_lm = grid_parms["cell_size"]
    oversampling = grid_parms["oversampling"]
    support = grid_parms["support"]

    if grid_parms["complex_grid"]:
        grid = np.zeros(
            (n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128
        )
    else:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)

    do_psf = grid_parms["do_psf"]
    do_imaging_weight = False
    _standard_grid_jit(
        grid,
        sum_weight,
        do_psf,
        do_imaging_weight,
        vis_data,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        weight,
        cgk_1D,
        n_uv,
        delta_lm,
        support,
        oversampling,
    )

    return grid, sum_weight


def _standard_grid_psf_numpy_wrap(uvw, weight, freq_chan, cgk_1D, grid_parms):
    """
    Wraps the jit gridder code.

    Parameters
    ----------
    grid : complex array
        (n_chan, n_pol, n_u, n_v)
    sum_weight : float array
        (n_chan, n_pol)
    uvw  : float array
        (n_time, n_baseline, 3)
    freq_chan : float array
        (n_chan)
    weight : float array
        (n_time, n_baseline, n_vis_chan)
    cgk_1D : float array
        (oversampling*(support//2 + 1))
    grid_parms : dictionary
        keys ('image_size','cell','oversampling','support')

    Returns
    -------
    grid : complex array
        (1,n_imag_chan,n_imag_pol,n_u,n_v)
    """

    n_chan = weight.shape[2]
    if grid_parms["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = grid_parms["image_size_padded"]
    delta_lm = grid_parms["cell_size"]
    oversampling = grid_parms["oversampling"]
    support = grid_parms["support"]

    grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)

    do_imaging_weight = grid_parms["do_imaging_weight"]

    do_psf = grid_parms["do_psf"]
    vis_data = np.zeros(
        (1, 1, 1, 1), dtype=bool
    )  # This 0 bool array is needed to pass to _standard_grid_jit so that the code can be resued and to keep numba happy.
    # print('do_imaging_weight',do_imaging_weight)
    _standard_grid_jit(
        grid,
        sum_weight,
        do_psf,
        do_imaging_weight,
        vis_data,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        weight,
        cgk_1D,
        n_uv,
        delta_lm,
        support,
        oversampling,
    )
    # print('new code')
    return grid, sum_weight


import numpy as np


# When jit is used round is repolaced by standard c++ round that is different to python round
@jit(nopython=True, cache=True, nogil=True)
def _standard_grid_jit(
    grid,
    sum_weight,
    do_psf,
    do_imaging_weight,
    vis_data,
    uvw,
    freq_chan,
    chan_map,
    pol_map,
    weight,
    cgk_1D,
    n_uv,
    delta_lm,
    support,
    oversampling,
):
    """
    Parameters
    ----------
    grid : complex array
        (n_chan, n_pol, n_u, n_v)
    sum_weight : float array
        (n_chan, n_pol)
    vis_data : complex array
        (n_time, n_baseline, n_vis_chan, n_pol)
    uvw  : float array
        (n_time, n_baseline, 3)
    freq_chan : float array
        (n_chan)
    chan_map : int array
        (n_chan)
    pol_map : int array
        (n_pol)
    weight : float array
        (n_time, n_baseline, n_vis_chan)
    cgk_1D : float array
        (oversampling*(support//2 + 1))
    grid_parms : dictionary
        keys ('n_imag_chan','n_imag_pol','n_uv','delta_lm','oversampling','support')

    Returns
    -------
    """

    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c

    # oversampling_center = int(oversampling // 2)
    support_center = int(support // 2)
    uv_center = n_uv // 2

    start_support = -support_center
    end_support = (
        support - support_center
    )  # end_support is larger by 1 so that python range() gives correct indices

    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)
    n_pol = len(pol_map)

    n_u = n_uv[0]
    n_v = n_uv[1]

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]

                    u_pos_conj = -u + uv_center[0]
                    v_pos_conj = -v + uv_center[1]

                    # Doing round as int(x+0.5) since u_pos/v_pos should always positive and this matices fortran and gives consistant rounding.
                    # u_center_indx = int(u_pos + 0.5)
                    # v_center_indx = int(v_pos + 0.5)

                    # Do not use numpy round
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)

                    u_center_indx_conj = int(u_pos_conj + 0.5)
                    v_center_indx_conj = int(v_pos_conj + 0.5)

                    if (
                        (u_center_indx + support_center < n_u)
                        and (v_center_indx + support_center < n_v)
                        and (u_center_indx - support_center >= 0)
                        and (v_center_indx - support_center >= 0)
                    ):
                        u_offset = u_center_indx - u_pos
                        u_center_offset_indx = math.floor(u_offset * oversampling + 0.5)
                        v_offset = v_center_indx - v_pos
                        v_center_offset_indx = math.floor(v_offset * oversampling + 0.5)

                        for i_pol in range(n_pol):
                            if do_psf:
                                if (n_pol >= 2) and do_imaging_weight:
                                    weighted_data = (
                                        weight[i_time, i_baseline, i_chan, 0]
                                        + weight[i_time, i_baseline, i_chan, 1]
                                    ) / 2.0
                                    sel_weight = weighted_data
                                else:
                                    sel_weight = weight[
                                        i_time, i_baseline, i_chan, i_pol
                                    ]
                                    weighted_data = weight[
                                        i_time, i_baseline, i_chan, i_pol
                                    ]
                            else:
                                sel_weight = weight[i_time, i_baseline, i_chan, i_pol]
                                weighted_data = (
                                    vis_data[i_time, i_baseline, i_chan, i_pol]
                                    * weight[i_time, i_baseline, i_chan, i_pol]
                                )

                            # print('1. u_center_indx, v_center_indx', u_center_indx, v_center_indx, vis_data[i_time, i_baseline, i_chan, i_pol], weight[i_time, i_baseline, i_chan, i_pol])

                            if ~np.isnan(weighted_data) and (weighted_data != 0.0):
                                a_pol = pol_map[i_pol]
                                norm = 0.0

                                for i_v in range(start_support, end_support):
                                    v_indx = v_center_indx + i_v
                                    v_offset_indx = np.abs(
                                        oversampling * i_v + v_center_offset_indx
                                    )
                                    conv_v = cgk_1D[v_offset_indx]

                                    if do_imaging_weight:
                                        v_indx_conj = v_center_indx_conj + i_v

                                    for i_u in range(start_support, end_support):
                                        u_indx = u_center_indx + i_u
                                        u_offset_indx = np.abs(
                                            oversampling * i_u + u_center_offset_indx
                                        )
                                        conv_u = cgk_1D[u_offset_indx]
                                        conv = conv_u * conv_v

                                        grid[a_chan, a_pol, u_indx, v_indx] = (
                                            grid[a_chan, a_pol, u_indx, v_indx]
                                            + conv * weighted_data
                                        )
                                        norm = norm + conv

                                        if do_imaging_weight:
                                            u_indx_conj = u_center_indx_conj + i_u
                                            grid[
                                                a_chan, a_pol, u_indx_conj, v_indx_conj
                                            ] = (
                                                grid[
                                                    a_chan,
                                                    a_pol,
                                                    u_indx_conj,
                                                    v_indx_conj,
                                                ]
                                                + conv * weighted_data
                                            )

                                sum_weight[a_chan, a_pol] = (
                                    sum_weight[a_chan, a_pol] + sel_weight * norm
                                )

                                if do_imaging_weight:
                                    sum_weight[a_chan, a_pol] = (
                                        sum_weight[a_chan, a_pol] + sel_weight * norm
                                    )

    return


# TransformMachines/FTMachine.tcc
# template <class T> void FTMachine::getGrid(casacore::Array<T>& thegrid)
############################################################################################################################################################################################################################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################################################################################################################################################################################################################


def _standard_imaging_weight_degrid_numpy_wrap(
    grid_imaging_weight,
    uvw,
    natural_imaging_weight,
    briggs_factors,
    freq_chan,
    grid_parms,
):
    n_chan = natural_imaging_weight.shape[2]
    n_imag_chan = n_chan

    #    if grid_parms['chan_mode'] == 'cube':
    #        n_imag_chan = n_chan
    #        chan_map = (np.arange(0, n_chan)).astype(int)
    #    else:  # continuum
    #        n_imag_chan = 1
    #        chan_map = (np.zeros(n_chan)).astype(int)
    # Should always be cube for image weights
    n_imag_chan = n_chan
    chan_map = (np.arange(0, n_chan)).astype(int)

    n_imag_pol = natural_imaging_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = grid_parms["image_size_padded"]
    delta_lm = grid_parms["cell_size"]

    imaging_weight = np.zeros(natural_imaging_weight.shape, dtype=np.double)

    _standard_imaging_weight_degrid_jit(
        imaging_weight,
        grid_imaging_weight,
        briggs_factors,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        natural_imaging_weight,
        n_uv,
        delta_lm,
    )

    return imaging_weight


@jit(nopython=True, cache=True, nogil=True)
def _standard_imaging_weight_degrid_jit(
    imaging_weight,
    grid_imaging_weight,
    briggs_factors,
    uvw,
    freq_chan,
    chan_map,
    pol_map,
    natural_imaging_weight,
    n_uv,
    delta_lm,
):
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c

    uv_center = n_uv // 2

    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)
    n_pol = len(pol_map)
    n_imag_chan = chan_map.shape[0]

    n_u = n_uv[0]
    n_v = n_uv[1]

    # print('Degrid operation')

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]

                    # Doing round as int(x+0.5) since u_pos/v_pos should always be positive and  fortran and gives consistant rounding.
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)

                    # print('f uv', freq_chan[i_chan], uvw[i_time, i_baseline, 0],uvw[i_time, i_baseline, 1])
                    if (
                        (u_center_indx < n_u)
                        and (v_center_indx < n_v)
                        and (u_center_indx >= 0)
                        and (v_center_indx >= 0)
                    ):
                        # print('u_center_indx, v_center_indx',  u_center_indx, v_center_indx)

                        for i_pol in range(n_pol):
                            a_pol = pol_map[i_pol]

                            if n_pol == 2:
                                imaging_weight[i_time, i_baseline, i_chan, i_pol] = (
                                    natural_imaging_weight[
                                        i_time, i_baseline, i_chan, 0
                                    ]
                                    + natural_imaging_weight[
                                        i_time, i_baseline, i_chan, 1
                                    ]
                                ) / 2.0
                            else:
                                imaging_weight[
                                    i_time, i_baseline, i_chan, i_pol
                                ] = natural_imaging_weight[
                                    i_time, i_baseline, i_chan, i_pol
                                ]

                            if ~np.isnan(
                                natural_imaging_weight[
                                    i_time, i_baseline, i_chan, i_pol
                                ]
                            ) and (
                                natural_imaging_weight[
                                    i_time, i_baseline, i_chan, i_pol
                                ]
                                != 0.0
                            ):
                                if ~np.isnan(
                                    grid_imaging_weight[
                                        a_chan, a_pol, u_center_indx, v_center_indx
                                    ]
                                ) and (
                                    grid_imaging_weight[
                                        a_chan, a_pol, u_center_indx, v_center_indx
                                    ]
                                    != 0.0
                                ):
                                    briggs_grid_imaging_weight = (
                                        briggs_factors[0, a_chan, a_pol]
                                        * grid_imaging_weight[
                                            a_chan, a_pol, u_center_indx, v_center_indx
                                        ]
                                        + briggs_factors[1, a_chan, a_pol]
                                    )
                                    imaging_weight[
                                        i_time, i_baseline, i_chan, i_pol
                                    ] = (
                                        imaging_weight[
                                            i_time, i_baseline, i_chan, i_pol
                                        ]
                                        / briggs_grid_imaging_weight
                                    )

    return
