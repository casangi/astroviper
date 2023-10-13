from numba import jit
import numpy as np
import math

# from numba import gdb


def ndim_list(shape):
    return [ndim_list(shape[1:]) if len(shape) > 1 else None for _ in range(shape[0])]


@jit(nopython=True, cache=True, nogil=True)
def _aperture_grid_jit(
    grid,
    sum_weight,
    uvw,
    freq_chan,
    chan_map,
    pol_map,
    cf_baseline_map,
    cf_chan_map,
    cf_pol_map,
    imaging_weight,
    conv_kernel_convolved,
    n_uv,
    delta_lm,
    weight_support,
    oversampling,
    phase_gradient,
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

    n_u = n_uv[0]
    n_v = n_uv[1]

    u_center = uv_center[0]
    v_center = uv_center[1]

    max_support_center = np.max(weight_support)

    conv_v_center = conv_kernel_convolved.shape[-1] // 2
    conv_u_center = conv_kernel_convolved.shape[-2] // 2

    # print(phase_gradient.shape)
    # print(conv_kernel_convolved.shape)

    conv_kernel_convolved_phase_gradient = (
        conv_kernel_convolved * phase_gradient[0, :, :]
    )

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            cf_baseline = cf_baseline_map[i_baseline]
            for i_chan in range(n_chan):
                cf_chan = cf_chan_map[i_chan]
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]

                    # Do not use numpy round
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)

                    if (
                        (u_center_indx + max_support_center < n_u)
                        and (v_center_indx + max_support_center < n_v)
                        and (u_center_indx - max_support_center >= 0)
                        and (v_center_indx - max_support_center >= 0)
                    ):
                        # u_offset = u_center_indx - u_pos
                        # u_center_offset_indx = math.floor(u_offset * oversampling[0] + 0.5)
                        # v_offset = v_center_indx - v_pos
                        # v_center_offset_indx = math.floor(v_offset * oversampling[1] + 0.5)

                        for i_pol in range(n_pol):
                            weighted_data = imaging_weight[
                                i_time, i_baseline, i_chan, i_pol
                            ]

                            if ~np.isnan(weighted_data) and (weighted_data != 0.0):
                                cf_pol = cf_pol_map[i_pol]
                                a_pol = pol_map[i_pol]
                                norm = 0.0

                                """
                                support = weight_support[cf_baseline,cf_chan,cf_pol,:]
                                #support = np.array([13,13])
                                support_center = support // 2
                                start_support = - support_center
                                end_support = support - support_center # end_support is larger by 1 so that python range() gives correct indices
                                """

                                support_u = weight_support[
                                    cf_baseline, cf_chan, cf_pol, 0
                                ]
                                support_v = weight_support[
                                    cf_baseline, cf_chan, cf_pol, 1
                                ]

                                support_center_u = support_u // 2
                                support_center_v = support_v // 2

                                start_support_u = -support_center_u
                                start_support_v = -support_center_v

                                end_support_u = support_u - support_center_u
                                end_support_v = support_v - support_center_v

                                for i_v in range(start_support_v, end_support_v):
                                    v_indx = v_center + i_v
                                    cf_v_indx = oversampling[1] * i_v + conv_v_center

                                    for i_u in range(start_support_u, end_support_u):
                                        u_indx = u_center + i_u
                                        cf_u_indx = (
                                            oversampling[0] * i_u + conv_u_center
                                        )

                                        conv = conv_kernel_convolved_phase_gradient[
                                            cf_baseline,
                                            cf_chan,
                                            cf_pol,
                                            cf_u_indx,
                                            cf_v_indx,
                                        ]

                                        grid[a_chan, a_pol, u_indx, v_indx] = (
                                            grid[a_chan, a_pol, u_indx, v_indx]
                                            + conv * weighted_data
                                        )
                                        norm = norm + conv

                                # print("The norm",norm)
                                sum_weight[a_chan, a_pol] = sum_weight[
                                    a_chan, a_pol
                                ] + weighted_data * np.real(norm)

    return


def _mosaic_grid_numpy_wrap(
    vis_data,
    uvw,
    imaging_weight,
    field,
    cf_baseline_map,
    cf_chan_map,
    cf_pol_map,
    conv_kernel,
    weight_support,
    phase_gradient,
    freq_chan,
    grid_parms,
):
    # print('imaging_weight ', imaging_weight.shape)
    import time

    n_chan = imaging_weight.shape[2]
    if grid_parms["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(np.int)

    n_imag_pol = imaging_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms["image_size_padded"]
    delta_lm = grid_parms["cell_size"]
    oversampling = grid_parms["oversampling"]

    if grid_parms["complex_grid"]:
        grid = np.zeros(
            (n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128
        )
    else:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)

    do_psf = grid_parms["do_psf"]
    field_id = grid_parms["field_id"]

    # print('vis_data', vis_data.shape , 'grid ', grid.shape, 'sum_weight', sum_weight.shape, 'cf_chan_map ', cf_chan_map.shape, ' cf_baseline_map', cf_baseline_map.shape, 'cf_pol_map', cf_pol_map.shape, ' conv_kernel',  conv_kernel.shape, 'phase_gradient', phase_gradient.shape, 'field', field.shape,  )

    # start = time.time()
    _aperture_grid_jit(
        grid,
        sum_weight,
        do_psf,
        vis_data,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        cf_baseline_map,
        cf_chan_map,
        cf_pol_map,
        imaging_weight,
        conv_kernel,
        n_uv,
        delta_lm,
        weight_support,
        oversampling,
        field,
        field_id,
        phase_gradient,
    )
    # time_to_grid = time.time() - start
    # print("time to grid ", time_to_grid)

    return grid, sum_weight


def _mosaic_psf_grid_numpy_wrap(
    uvw,
    imaging_weight,
    field,
    cf_baseline_map,
    cf_chan_map,
    cf_pol_map,
    conv_kernel,
    weight_support,
    phase_gradient,
    freq_chan,
    grid_parms,
):
    # print('imaging_weight ', imaging_weight.shape)
    import time

    n_chan = imaging_weight.shape[2]
    if grid_parms["chan_mode"] == "cube":
        n_imag_chan = n_chan
        chan_map = (np.arange(0, n_chan)).astype(np.int)
    else:  # continuum
        n_imag_chan = 1  # Making only one continuum image.
        chan_map = (np.zeros(n_chan)).astype(np.int)

    n_imag_pol = imaging_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(np.int)

    n_uv = grid_parms["image_size_padded"]
    delta_lm = grid_parms["cell_size"]
    oversampling = grid_parms["oversampling"]

    if grid_parms["complex_grid"]:
        grid = np.zeros(
            (n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.complex128
        )
    else:
        grid = np.zeros((n_imag_chan, n_imag_pol, n_uv[0], n_uv[1]), dtype=np.double)
    sum_weight = np.zeros((n_imag_chan, n_imag_pol), dtype=np.double)

    do_psf = grid_parms["do_psf"]
    field_id = grid_parms["field_id"]

    # print('vis_data', vis_data.shape , 'grid ', grid.shape, 'sum_weight', sum_weight.shape, 'cf_chan_map ', cf_chan_map.shape, ' cf_baseline_map', cf_baseline_map.shape, 'cf_pol_map', cf_pol_map.shape, ' conv_kernel',  conv_kernel.shape, 'phase_gradient', phase_gradient.shape, 'field', field.shape,  )

    vis_data = np.zeros((1, 1, 1, 1), dtype=np.bool)
    # start = time.time()
    _mosaic_grid_jit(
        grid,
        sum_weight,
        do_psf,
        vis_data,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        cf_baseline_map,
        cf_chan_map,
        cf_pol_map,
        imaging_weight,
        conv_kernel,
        n_uv,
        delta_lm,
        weight_support,
        oversampling,
        field,
        field_id,
        phase_gradient,
    )
    # time_to_grid = time.time() - start
    # print("time to grid ", time_to_grid)

    return grid, sum_weight


# Important changes to be made https://github.com/numba/numba/issues/4261
# debug=True and gdb()
@jit(nopython=True, cache=True, nogil=True)
def _mosaic_grid_jit(
    grid,
    sum_weight,
    do_psf,
    vis_data,
    uvw,
    freq_chan,
    chan_map,
    pol_map,
    cf_baseline_map,
    cf_chan_map,
    cf_pol_map,
    imaging_weight,
    conv_kernel,
    n_uv,
    delta_lm,
    weight_support,
    oversampling,
    phase_gradient,
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

    n_u = n_uv[0]
    n_v = n_uv[1]

    u_center = uv_center[0]
    v_center = uv_center[1]

    max_support_center = np.max(weight_support)

    conv_v_center = conv_kernel.shape[-1] // 2
    conv_u_center = conv_kernel.shape[-2] // 2

    conv_size = np.array(conv_kernel.shape[-2:])

    # print('conv_size',conv_size)

    # print('sizes ',conv_kernel.shape, conv_u_center, conv_v_center)

    # print(phase_gradient.shape)
    # print(conv_kernel_convolved.shape)

    conv_kernel_phase_gradient = conv_kernel * phase_gradient[0, :, :]

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            cf_baseline = cf_baseline_map[i_baseline]
            for i_chan in range(n_chan):
                cf_chan = cf_chan_map[i_chan]
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]

                    # Do not use numpy round
                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)

                    if (
                        (u_center_indx + max_support_center < n_u)
                        and (v_center_indx + max_support_center < n_v)
                        and (u_center_indx - max_support_center >= 0)
                        and (v_center_indx - max_support_center >= 0)
                    ):
                        u_offset = u_center_indx - u_pos
                        u_center_offset_indx = (
                            math.floor(u_offset * oversampling[0] + 0.5) + conv_u_center
                        )
                        v_offset = v_center_indx - v_pos
                        v_center_offset_indx = (
                            math.floor(v_offset * oversampling[1] + 0.5) + conv_v_center
                        )

                        for i_pol in range(n_pol):
                            if do_psf:
                                weighted_data = imaging_weight[
                                    i_time, i_baseline, i_chan, i_pol
                                ]
                            else:
                                weighted_data = (
                                    vis_data[i_time, i_baseline, i_chan, i_pol]
                                    * imaging_weight[i_time, i_baseline, i_chan, i_pol]
                                )

                            if ~np.isnan(weighted_data) and (weighted_data != 0.0):
                                cf_pol = cf_pol_map[i_pol]
                                a_pol = pol_map[i_pol]
                                norm = 0.0

                                """
                                support = weight_support[cf_baseline,cf_chan,cf_pol,:]
                                #support = np.array([13,13])
                                support_center = support // 2
                                start_support = - support_center
                                end_support = support - support_center # end_support is larger by 1 so that python range() gives correct indices
                                """

                                support_u = weight_support[
                                    cf_baseline, cf_chan, cf_pol, 0
                                ]
                                support_v = weight_support[
                                    cf_baseline, cf_chan, cf_pol, 1
                                ]

                                support_center_u = support_u // 2
                                support_center_v = support_v // 2

                                start_support_u = -support_center_u
                                start_support_v = -support_center_v

                                end_support_u = support_u - support_center_u
                                end_support_v = support_v - support_center_v

                                # print(support)
                                ###############
                                #                                resized_conv_size = (support  + 1)*oversampling
                                #                                start_indx = conv_size//2 - resized_conv_size//2
                                #                                end_indx = start_indx + resized_conv_size
                                #                                normalize_factor = np.real(np.sum(conv_kernel[cf_baseline,cf_chan,cf_pol,start_indx[0]:end_indx[0],start_indx[1]:end_indx[1]])/(oversampling[0]*oversampling[1]))
                                #
                                #                                conv_kernel_phase_gradient = conv_kernel*phase_gradient[field[i_time],:,:]/normalize_factor
                                #                                print(normalize_factor)
                                ##############

                                for i_v in range(start_support_v, end_support_v):
                                    v_indx = v_center_indx + i_v
                                    cf_v_indx = (
                                        oversampling[1] * i_v + v_center_offset_indx
                                    )

                                    for i_u in range(start_support_u, end_support_u):
                                        u_indx = u_center_indx + i_u
                                        cf_u_indx = (
                                            oversampling[0] * i_u + u_center_offset_indx
                                        )

                                        conv = conv_kernel_phase_gradient[
                                            cf_baseline,
                                            cf_chan,
                                            cf_pol,
                                            cf_u_indx,
                                            cf_v_indx,
                                        ]

                                        grid[a_chan, a_pol, u_indx, v_indx] = (
                                            grid[a_chan, a_pol, u_indx, v_indx]
                                            + conv * weighted_data
                                        )
                                        norm = norm + conv
                                if do_psf:
                                    sum_weight[a_chan, a_pol] = sum_weight[
                                        a_chan, a_pol
                                    ] + imaging_weight[
                                        i_time, i_baseline, i_chan, i_pol
                                    ] * np.real(
                                        norm
                                    )
                                else:
                                    sum_weight[a_chan, a_pol] = sum_weight[
                                        a_chan, a_pol
                                    ] + imaging_weight[
                                        i_time, i_baseline, i_chan, i_pol
                                    ] * np.real(
                                        norm**2
                                    )  # *np.real(norm**2)#* np.real(norm) #np.abs(norm**2) #**2 term is needed since the pb is in the image twice (one naturally and another from the gcf)

    return


def _tree_sum_list(list_to_sum):
    import dask.array as da

    while len(list_to_sum) > 1:
        new_list_to_sum = []
        for i in range(0, len(list_to_sum), 2):
            if i < len(list_to_sum) - 1:
                lazy = da.add(list_to_sum[i], list_to_sum[i + 1])
            else:
                lazy = list_to_sum[i]
            new_list_to_sum.append(lazy)
        list_to_sum = new_list_to_sum
    return list_to_sum
