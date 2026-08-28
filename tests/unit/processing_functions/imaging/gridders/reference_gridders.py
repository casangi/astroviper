"""Pure-Python reference implementations of the prolate spheroidal gridders.

These are exact, de-jitted copies of the retired numba kernels
(``prolate_spheroidal_grid_jit``, ``prolate_spheroidal_grid_uv_sampling_jit``,
``prolate_spheroidal_degrid_jit``) kept solely as independent test oracles for
the C++ extension (`prolate_spheroidal_grid_cpp`). They are slow — pure Python
loops — but the unit-test inputs are small.

Do not use these in production code; the C++ kernels are the only supported
implementation.
"""

import math

import numpy as np


def prolate_spheroidal_grid_reference(
    grid,
    normalization,
    vis_data,
    uvw,
    frequency_coord,
    frequency_map,
    time_map,
    pol_map,
    weight,
    cgk_1D,
    n_uv,
    delta_lm,
    support,
    oversampling,
):
    """Grid weighted visibilities onto a UV plane (reference implementation).

    Accumulates weighted, convolved visibility samples into ``grid`` and the
    corresponding sum of convolution weights into ``normalization`` in place.
    Sub-pixel offsets are rounded with ``int(x + 0.5)`` to match the
    Fortran/C++ rounding convention.
    """
    c = 299792458.0
    uv_scale = np.zeros((2, len(frequency_coord)), dtype=np.double)
    uv_scale[0, :] = -(frequency_coord * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(frequency_coord * delta_lm[1] * n_uv[1]) / c

    support_center = int(support // 2)
    uv_center = n_uv // 2

    start_support = -support_center
    end_support = support - support_center

    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(frequency_map)
    n_pol = len(pol_map)

    n_u = n_uv[0]
    n_v = n_uv[1]

    for i_time in range(n_time):
        a_time = time_map[i_time]
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = frequency_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if np.isnan(u) or np.isnan(v):
                    continue

                u_pos = u + uv_center[0]
                v_pos = v + uv_center[1]

                # Do not use numpy round (int(x + 0.5) matches Fortran/C++).
                u_center_indx = int(u_pos + 0.5)
                v_center_indx = int(v_pos + 0.5)

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

                        if (not np.isnan(weighted_data)) and (weighted_data != 0.0):
                            a_pol = pol_map[i_pol]
                            norm = 0.0

                            for i_u in range(start_support, end_support):
                                u_indx = u_center_indx + i_u
                                u_offset_indx = np.abs(
                                    oversampling * i_u + u_center_offset_indx
                                )
                                conv_u = cgk_1D[u_offset_indx]

                                for i_v in range(start_support, end_support):
                                    v_indx = v_center_indx + i_v
                                    v_offset_indx = np.abs(
                                        oversampling * i_v + v_center_offset_indx
                                    )
                                    conv_v = cgk_1D[v_offset_indx]
                                    conv = conv_u * conv_v

                                    grid[a_time, a_chan, a_pol, u_indx, v_indx] = (
                                        grid[a_time, a_chan, a_pol, u_indx, v_indx]
                                        + conv * weighted_data
                                    )
                                    norm = norm + conv

                            normalization[a_time, a_chan, a_pol] = (
                                normalization[a_time, a_chan, a_pol] + sel_weight * norm
                            )
    return


def prolate_spheroidal_grid_uv_sampling_reference(
    grid,
    normalization,
    uvw,
    frequency_coord,
    frequency_map,
    time_map,
    pol_map,
    weight,
    cgk_1D,
    n_uv,
    delta_lm,
    support,
    oversampling,
):
    """Grid imaging weights onto a UV plane (PSF numerator; reference)."""
    c = 299792458.0
    uv_scale = np.zeros((2, len(frequency_coord)), dtype=np.double)
    uv_scale[0, :] = -(frequency_coord * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(frequency_coord * delta_lm[1] * n_uv[1]) / c

    support_center = int(support // 2)
    uv_center = n_uv // 2

    start_support = -support_center
    end_support = support - support_center

    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(frequency_map)
    n_pol = len(pol_map)

    n_u = n_uv[0]
    n_v = n_uv[1]

    for i_time in range(n_time):
        a_time = time_map[i_time]
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = frequency_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if np.isnan(u) or np.isnan(v):
                    continue

                u_pos = u + uv_center[0]
                v_pos = v + uv_center[1]

                # Do not use numpy round (int(x + 0.5) matches Fortran/C++).
                u_center_indx = int(u_pos + 0.5)
                v_center_indx = int(v_pos + 0.5)

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
                        weight_data = weight[i_time, i_baseline, i_chan, i_pol]

                        if (not np.isnan(weight_data)) and (weight_data != 0.0):
                            a_pol = pol_map[i_pol]
                            norm = 0.0

                            for i_u in range(start_support, end_support):
                                u_indx = u_center_indx + i_u
                                u_offset_indx = np.abs(
                                    oversampling * i_u + u_center_offset_indx
                                )
                                conv_u = cgk_1D[u_offset_indx]

                                for i_v in range(start_support, end_support):
                                    v_indx = v_center_indx + i_v
                                    v_offset_indx = np.abs(
                                        oversampling * i_v + v_center_offset_indx
                                    )
                                    conv_v = cgk_1D[v_offset_indx]
                                    conv = conv_u * conv_v

                                    grid[a_time, a_chan, a_pol, u_indx, v_indx] = (
                                        grid[a_time, a_chan, a_pol, u_indx, v_indx]
                                        + conv * weight_data
                                    )
                                    norm = norm + conv

                            normalization[a_time, a_chan, a_pol] = (
                                normalization[a_time, a_chan, a_pol]
                                + weight_data * norm
                            )
    return


def prolate_spheroidal_degrid_reference(
    grid,
    vis_data,
    uvw,
    frequency_coord,
    frequency_map,
    time_map,
    pol_map,
    cgk_1D,
    n_uv,
    delta_lm,
    support,
    oversampling,
):
    """Sample (degrid) a model UV grid at visibility coordinates (reference).

    Writes the convolution-interpolated model value (divided by the kernel
    norm) into ``vis_data`` in place. Samples where ``vis_data`` is NaN are
    skipped, so flags must be applied (as NaN) by the caller beforehand.
    """
    c = 299792458.0
    uv_scale = np.zeros((2, len(frequency_coord)), dtype=np.double)
    uv_scale[0, :] = -(frequency_coord * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(frequency_coord * delta_lm[1] * n_uv[1]) / c

    support_center = int(support // 2)
    uv_center = n_uv // 2

    start_support = -support_center
    end_support = support - support_center

    n_time = uvw.shape[0]
    n_baseline = uvw.shape[1]
    n_chan = len(frequency_map)
    n_pol = len(pol_map)

    n_u = n_uv[0]
    n_v = n_uv[1]

    for i_time in range(n_time):
        a_time = time_map[i_time]
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = frequency_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if np.isnan(u) or np.isnan(v):
                    continue

                u_pos = u + uv_center[0]
                v_pos = v + uv_center[1]

                # Do not use numpy round (int(x + 0.5) matches Fortran/C++).
                u_center_indx = int(u_pos + 0.5)
                v_center_indx = int(v_pos + 0.5)

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
                        a_pol = pol_map[i_pol]
                        norm = 0.0
                        degrid_value = 0.0

                        if not np.isnan(vis_data[i_time, i_baseline, i_chan, i_pol]):
                            for i_u in range(start_support, end_support):
                                u_indx = u_center_indx + i_u
                                u_offset_indx = np.abs(
                                    oversampling * i_u + u_center_offset_indx
                                )
                                conv_u = cgk_1D[u_offset_indx]

                                for i_v in range(start_support, end_support):
                                    v_indx = v_center_indx + i_v
                                    v_offset_indx = np.abs(
                                        oversampling * i_v + v_center_offset_indx
                                    )
                                    conv_v = cgk_1D[v_offset_indx]
                                    conv = conv_u * conv_v
                                    degrid_value += (
                                        conv
                                        * grid[a_time, a_chan, a_pol, u_indx, v_indx]
                                    )
                                    norm = norm + conv

                            if norm != 0.0:
                                vis_data[i_time, i_baseline, i_chan, i_pol] = (
                                    degrid_value / norm
                                )

    return
