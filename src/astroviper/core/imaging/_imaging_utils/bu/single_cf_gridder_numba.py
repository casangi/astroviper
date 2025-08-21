# Enable fastmath (don't check dims).
from numba import jit
import numpy as np
import math
import time


@jit(nopython=True, cache=True, nogil=True, fastmath=True)
def _standard_grid_jit(
    grid,
    grid_shape,
    sum_weight,
    vis_data,
    vis_shape,
    uvw,
    freq_chan,
    chan_map,
    pol_map,
    weight,
    cgk_1D,
    delta_lm,
    support,
    oversampling,
):

    n_time = vis_shape[0]
    n_baseline = vis_shape[1]
    n_chan = vis_shape[2]
    n_pol = vis_shape[3]

    n_u = grid_shape[2]
    n_v = grid_shape[3]
    c = 299792458.0

    support = 7
    oversampling = 100

    support_center = int(support // 2)
    u_center = n_u // 2
    v_center = n_v // 2

    start_support = -support_center
    end_support = (
        support - support_center
    )  # end_support is larger by 1 so that python range() gives correct indices

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * (
                    -(freq_chan[i_chan] * delta_lm[0] * n_u) / c
                )
                v = uvw[i_time, i_baseline, 1] * (
                    -(freq_chan[i_chan] * delta_lm[1] * n_v) / c
                )

                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + u_center
                    v_pos = v + v_center

                    u_pos_conj = -u + u_center
                    v_pos_conj = -v + v_center

                    # Doing round as int(x+0.5) since u_pos/v_pos should always positive and this matices fortran and gives consistant rounding.
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
                            sel_weight = weight[i_time, i_baseline, i_chan, i_pol]
                            weighted_data = (
                                vis_data[i_time, i_baseline, i_chan, i_pol]
                                * weight[i_time, i_baseline, i_chan, i_pol]
                            )

                            if ~np.isnan(weighted_data) and (weighted_data != 0.0):
                                a_pol = pol_map[i_pol]
                                norm = 0.0

                                for i_v in range(start_support, end_support):
                                    v_indx = v_center_indx + i_v
                                    v_offset_indx = np.abs(
                                        oversampling * i_v + v_center_offset_indx
                                    )
                                    conv_v = cgk_1D[v_offset_indx]

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

                                sum_weight[a_chan, a_pol] = (
                                    sum_weight[a_chan, a_pol] + sel_weight * norm
                                )

    return


def _create_prolate_spheroidal_kernel(oversampling, support, n_uv):
    """
    Create PSWF to serve as gridding kernel

    Parameters
    ----------
    oversampling : int
        oversampling//2 is the index of the zero value of the oversampling value
    support : int
        support//2 is the index of the zero value of the support values
    n_uv: int array
        (2)
        number of pixels in u,v space

    Returns
    -------
    kernel : numpy.ndarray
    kernel_image : numpy.ndarray

    """
    # support//2 is the index of the zero value of the support values
    # oversampling//2 is the index of the zero value of the oversampling value
    support_center = support // 2
    oversampling_center = oversampling // 2

    support_values = np.arange(support) - support_center
    if (oversampling % 2) == 0:
        oversampling_values = (
            (np.arange(oversampling + 1) - oversampling_center) / oversampling
        )[:, None]
        kernel_points_1D = (
            np.broadcast_to(support_values, (oversampling + 1, support))
            + oversampling_values
        )
    else:
        oversampling_values = (
            (np.arange(oversampling) - oversampling_center) / oversampling
        )[:, None]
        kernel_points_1D = (
            np.broadcast_to(support_values, (oversampling, support))
            + oversampling_values
        )

    kernel_points_1D = kernel_points_1D / support_center

    _, kernel_1D = _prolate_spheroidal_function(kernel_points_1D)
    # kernel_1D /= np.sum(np.real(kernel_1D[oversampling_center,:]))

    if (oversampling % 2) == 0:
        kernel = np.zeros(
            (oversampling + 1, oversampling + 1, support, support), dtype=np.double
        )  # dtype=np.complex128
    else:
        kernel = np.zeros(
            (oversampling, oversampling, support, support), dtype=np.double
        )

    for x in range(oversampling):
        for y in range(oversampling):
            kernel[x, y, :, :] = np.outer(kernel_1D[x, :], kernel_1D[y, :])

    # norm = np.sum(np.real(kernel))
    # kernel /= norm

    # Gridding correction function (applied after dirty image is created)
    kernel_image_points_1D_u = np.abs(2.0 * _coordinates(n_uv[0]))
    kernel_image_1D_u = _prolate_spheroidal_function(kernel_image_points_1D_u)[0]

    kernel_image_points_1D_v = np.abs(2.0 * _coordinates(n_uv[1]))
    kernel_image_1D_v = _prolate_spheroidal_function(kernel_image_points_1D_v)[0]

    kernel_image = np.outer(kernel_image_1D_u, kernel_image_1D_v)

    # kernel_image[kernel_image > 0.0] = kernel_image.max() / kernel_image[kernel_image > 0.0]

    # kernel_image =  kernel_image/kernel_image.max()
    return kernel, kernel_image


def _create_prolate_spheroidal_kernel_1D(oversampling, support):
    support_center = support // 2
    oversampling_center = oversampling // 2
    u = np.arange(oversampling * (support_center)) / (support_center * oversampling)
    # print(u)

    long_half_kernel_1D = np.zeros(oversampling * (support_center + 1))
    _, long_half_kernel_1D[0 : oversampling * (support_center)] = (
        _prolate_spheroidal_function(u)
    )

    # print(_prolate_spheroidal_function(u))
    return long_half_kernel_1D


def _prolate_spheroidal_function(u):
    """
    Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.

    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.

    The griddata function is (1-NU**2)*GRDSF(NU) where NU is the distance
    to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    is now the distance to the edge of the image.
    """
    p = np.array(
        [
            [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
            [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
        ]
    )
    q = np.array(
        [
            [1.0000000e0, 8.212018e-1, 2.078043e-1],
            [1.0000000e0, 9.599102e-1, 2.918724e-1],
        ]
    )

    _, n_p = p.shape
    _, n_q = q.shape

    u = np.abs(u)
    uend = np.zeros(u.shape, dtype=np.float64)
    part = np.zeros(u.shape, dtype=np.int64)

    part[(u >= 0.0) & (u < 0.75)] = 0
    part[(u >= 0.75) & (u <= 1.0)] = 1
    uend[(u >= 0.0) & (u < 0.75)] = 0.75
    uend[(u >= 0.75) & (u <= 1.0)] = 1.0

    delusq = u**2 - uend**2

    top = p[part, 0]
    for k in range(1, n_p):  # small constant size loop
        top += p[part, k] * np.power(delusq, k)

    bot = q[part, 0]
    for k in range(1, n_q):  # small constant size loop
        bot += q[part, k] * np.power(delusq, k)

    grdsf = np.zeros(u.shape, dtype=np.float64)
    ok = bot > 0.0
    grdsf[ok] = top[ok] / bot[ok]
    ok = np.abs(u > 1.0)
    grdsf[ok] = 0.0

    # Return the correcting image and the gridding kernel value
    return grdsf, (1 - u**2) * grdsf


def _coordinates(npixel: int):
    """1D array which spans [-.5,.5[ with 0 at position npixel/2"""
    return (np.arange(npixel) - npixel // 2) / npixel


def _coordinates2(npixel: int):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension
    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    return (np.mgrid[0:npixel, 0:npixel] - npixel // 2) / npixel
