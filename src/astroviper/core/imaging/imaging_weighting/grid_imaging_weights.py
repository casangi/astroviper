from numba import jit
import numpy as np
import math


def grid_imaging_weights(grid, sum_weight, uvw, weight, freq_chan, grid_parms):
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
    grid_parms : dictionary
        keys ('image_size','cell','oversampling','support')

    Returns
    -------
    grid : complex array
        (1,n_imag_chan,n_imag_pol,n_u,n_v)
    """

    n_chan = weight.shape[2]
    chan_map = (np.arange(0, n_chan)).astype(int)

    # Always per chan weight so no longer do:
    # if grid_parms["chan_mode"] == "cube":
    #     n_imag_chan = n_chan
    #     chan_map = (np.arange(0, n_chan)).astype(int)
    # else:  # continuum
    #     n_imag_chan = 1  # Making only one continuum image.
    #     chan_map = (np.zeros(n_chan)).astype(int)

    n_imag_pol = weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = grid_parms["image_size_padded"]
    delta_lm = grid_parms["cell_size"]

    assert weight.shape[3] < 3, "Polarization should be PP or PP, QQ."

    # print("chan map ", chan_map)

    grid_imaging_weights_jit(
        grid,
        sum_weight,
        uvw,
        freq_chan,
        chan_map,
        weight,
        n_uv,
        delta_lm,
    )


import numpy as np


# When jit is used round is repolaced by standard c++ round that is different to python round
@jit(nopython=True, cache=True, nogil=True)  # fastmath=True
def grid_imaging_weights_jit(
    grid,
    sum_weight,
    uvw,
    freq_chan,
    chan_map,
    data_weights,
    n_uv,
    delta_lm,
):
    """
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
    chan_map : int array
        (n_chan)
    weight : float array
        (n_time, n_baseline, n_vis_chan)

    Returns
    -------
    """
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c
    uv_center = n_uv // 2

    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(chan_map)

    n_u = n_uv[0]
    n_v = n_uv[1]

    n_pol = data_weights.shape[3]

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

                    # Doing round as int(x+0.5) since u_pos/v_pos should always positive and this matrices fortran and gives consistant rounding.
                    # Do ~use numpy round
                    u_indx = int(u_pos + 0.5)
                    v_indx = int(v_pos + 0.5)

                    u_indx_conj = int(u_pos_conj + 0.5)
                    v_indx_conj = int(v_pos_conj + 0.5)

                    if (
                        (u_indx < n_u)
                        and (v_indx < n_v)
                        and (u_indx >= 0)
                        and (v_indx >= 0)
                    ):
                        # if n_pol >= 2:  # NB NB Check polarization order
                        #     weight = (
                        #         data_weights[i_time, i_baseline, i_chan, 0]
                        #         + data_weights[i_time, i_baseline, i_chan, 1]
                        #     ) / 2.0
                        # else:
                        weight = data_weights[i_time, i_baseline, i_chan, 0]

                        if ~np.isnan(weight):
                            norm = 0.0

                            grid[a_chan, 0, u_indx, v_indx] = (
                                grid[a_chan, 0, u_indx, v_indx] + weight
                            )

                            grid[a_chan, 0, u_indx_conj, v_indx_conj] = (
                                grid[a_chan, 0, u_indx_conj, v_indx_conj] + weight
                            )

                            sum_weight[a_chan, 0] = sum_weight[a_chan, 0] + 2 * weight
    return


# TransformMachines/FTMachine.tcc
# template <class T> void FTMachine::getGrid(casacore::Array<T>& thegrid)
############################################################################################################################################################################################################################################################################################################################################################################################################################################################
############################################################################################################################################################################################################################################################################################################################################################################################################################################################


def degrid_imaging_weights(
    grid_imaging_weight,
    uvw,
    natural_imaging_weight,
    briggs_factors,
    freq_chan,
    grid_parms,
):
    n_imag_chan = natural_imaging_weight.shape[2]

    #    if grid_parms['chan_mode'] == 'cube':
    #        n_imag_chan = n_chan
    #        chan_map = (np.arange(0, n_chan)).astype(int)
    #    else:  # continuum
    #        n_imag_chan = 1
    #        chan_map = (np.zeros(n_chan)).astype(int)
    # Should always be cube for image weights
    chan_map = (np.arange(0, n_imag_chan)).astype(int)

    n_imag_pol = natural_imaging_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    n_uv = grid_parms["image_size_padded"]
    delta_lm = grid_parms["cell_size"]

    imaging_weight = np.zeros(natural_imaging_weight.shape, dtype=np.double)

    degrid_imaging_weights_jit(
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
def degrid_imaging_weights_jit(
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
                                imaging_weight[i_time, i_baseline, i_chan, i_pol] = (
                                    natural_imaging_weight[
                                        i_time, i_baseline, i_chan, i_pol
                                    ]
                                )

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
                                # print("a_chan, a_pol, u_center_indx, v_center_indx", a_chan, a_pol, u_center_indx, v_center_indx, grid_imaging_weight.shape)
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
