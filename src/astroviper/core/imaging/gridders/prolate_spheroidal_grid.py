from numba import jit
import numpy as np
import numpy.typing as npt
import math


# When jit is used round is repolaced by standard c++ round that is different to python round
@jit(nopython=True, cache=True, nogil=True)  # fastmath=True
def prolate_spheroidal_grid_jit(
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
    """Grid weighted visibilities onto a UV plane using a prolate spheroidal convolution kernel.

    Accumulates weighted, convolved visibility samples into ``grid`` and the
    corresponding sum of convolution weights into ``normalization``.  Both
    arrays are modified in place so that repeated calls (e.g. over multiple
    measurement sets) accumulate contributions.

    Time and frequency mapping allow the input visibility axes to be collapsed
    or reordered onto the image axes.

    Convention note — axis prefixes:
      * ``m_`` denotes an image-domain axis size.
      * ``n_`` denotes a visibility-domain axis size.

    Parameters
    ----------
    grid : complex ndarray, shape (m_time, m_chan, m_pol, m_u, m_v)
        UV grid accumulator.  Modified in place.
    normalization : float ndarray, shape (m_time, m_chan, m_pol)
        Per-cell sum of ``weight * convolution_kernel`` values, used to
        normalise the grid after all visibilities have been accumulated.
        Modified in place.
    vis_data : complex ndarray, shape (n_time, n_baseline, n_vis_chan, n_pol)
        Correlated visibility data.
    uvw : float ndarray, shape (n_time, n_baseline, 3)
        Baseline coordinates in metres.  The third axis is ordered (U, V, W).
    frequency_coord : float ndarray, shape (n_vis_chan,)
        Channel centre frequencies in Hz, used to convert UVW from metres to
        wavelengths.
    frequency_map : int ndarray, shape (n_vis_chan,)
        Maps each visibility channel index to its target image channel index
        ``a_chan`` in ``grid``.
    time_map : int ndarray, shape (n_time,)
        Maps each visibility time index to its target image time index
        ``a_time`` in ``grid``.
    pol_map : int ndarray, shape (n_pol,)
        Maps each visibility polarization index to its target image
        polarization index ``a_pol`` in ``grid``.
    weight : float ndarray, shape (n_time, n_baseline, n_vis_chan, n_pol)
        Imaging weights applied to each visibility sample before gridding.
        Samples with ``NaN`` weight or zero weighted data are skipped.
    cgk_1D : float ndarray, shape (oversampling * (support // 2 + 1),)
        Oversampled 1-D prolate spheroidal wave function (PSWF) kernel.
        The 2-D kernel is formed as the outer product of two 1-D lookups.
    n_uv : int ndarray, shape (2,)
        Padded UV-grid dimensions ``[n_u, n_v]`` in pixels.
    delta_lm : float ndarray, shape (2,)
        Image cell size in radians ``[delta_l, delta_m]``.
    support : int
        Full convolution support width in grid pixels.  The kernel spans
        ``[-support//2, support - support//2)`` pixels around each sample.
    oversampling : int
        Oversampling factor of the 1-D kernel ``cgk_1D``.  A higher value
        reduces interpolation error at the cost of a larger kernel table.

    Returns
    -------
    None
        ``grid`` and ``normalization`` are modified in place; nothing is
        returned.

    Notes
    -----
    - Visibilities whose U or V coordinate is ``NaN`` are skipped entirely.
    - Samples whose weighted data value is ``NaN`` or zero are skipped.
    - The sub-pixel offset from the nearest grid point is rounded using
      ``int(x + 0.5)`` rather than ``numpy.round`` to match Fortran/C++
      rounding behaviour (round-half-away-from-zero).
    - Hardcoding ``support`` and ``oversampling`` as literals (currently
      commented out) allows Numba to unroll the inner support loops,
      yielding a significant speed-up.
    """

    # By hardcoding the support and oversampling values, the innermost for loops can be unrolled by the compiler leading to significantly faster code.
    # support = 7
    # oversampling = 100

    c = 299792458.0
    uv_scale = np.zeros((2, len(frequency_coord)), dtype=np.double)
    uv_scale[0, :] = -(frequency_coord * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(frequency_coord * delta_lm[1] * n_uv[1]) / c

    # oversampling_center = int(oversampling // 2)
    support_center = int(support // 2)
    uv_center = n_uv // 2

    start_support = -support_center
    end_support = (
        support - support_center
    )  # end_support is larger by 1 so that python range() gives correct indices

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

                if ~np.isnan(u) and ~np.isnan(v):
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]

                    # u_pos_conj = -u + uv_center[0]
                    # v_pos_conj = -v + uv_center[1]

                    # Doing round as int(x+0.5) since u_pos/v_pos should always positive and this matices fortran and gives consistant rounding.
                    # u_center_indx = int(u_pos + 0.5)
                    # v_center_indx = int(v_pos + 0.5)

                    # Do not use numpy round
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

                                        grid[a_time, a_chan, a_pol, u_indx, v_indx] = (
                                            grid[a_time, a_chan, a_pol, u_indx, v_indx]
                                            + conv * weighted_data
                                        )
                                        norm = norm + conv

                                normalization[a_time, a_chan, a_pol] = (
                                    normalization[a_time, a_chan, a_pol]
                                    + sel_weight * norm
                                )
    return


# Gridder used to create PSF (UV-sampling grid / point spread function)
# When jit is used round is repolaced by standard c++ round that is different to python round
@jit(nopython=True, cache=True, nogil=True)  # fastmath=True
def prolate_spheroidal_grid_uv_sampling_jit(
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
    """Grid imaging weights onto a UV plane to form the UV-sampling function (PSF numerator).

    This is the PSF variant of :func:`prolate_spheroidal_grid_jit`.  Instead
    of gridding weighted visibility data, it grids the imaging weights alone,
    accumulating the UV-sampling function that describes the PSF of the
    interferometric array.  The result is used to form the dirty beam (point
    spread function) and to normalise the dirty image.

    Both ``grid`` and ``normalization`` are modified in place so that repeated
    calls over multiple measurement sets accumulate contributions.

    Convention note — axis prefixes:
      * ``m_`` denotes an image-domain axis size.
      * ``n_`` denotes a visibility-domain axis size.

    Parameters
    ----------
    grid : complex ndarray, shape (m_time, m_chan, m_pol, m_u, m_v)
        UV-sampling grid accumulator.  Modified in place.
    normalization : float ndarray, shape (m_time, m_chan, m_pol)
        Per-cell sum of ``weight * convolution_kernel`` values.  Modified in
        place.
    uvw : float ndarray, shape (n_time, n_baseline, 3)
        Baseline coordinates in metres.  The third axis is ordered (U, V, W).
    frequency_coord : float ndarray, shape (n_vis_chan,)
        Channel centre frequencies in Hz, used to convert UVW from metres to
        wavelengths.
    frequency_map : int ndarray, shape (n_vis_chan,)
        Maps each visibility channel index to its target image channel index
        ``a_chan`` in ``grid``.
    time_map : int ndarray, shape (n_time,)
        Maps each visibility time index to its target image time index
        ``a_time`` in ``grid``.
    pol_map : int ndarray, shape (n_pol,)
        Maps each visibility polarization index to its target image
        polarization index ``a_pol`` in ``grid``.
    weight : float ndarray, shape (n_time, n_baseline, n_vis_chan, n_pol)
        Imaging weights.  Samples with ``NaN`` weight or zero weight are
        skipped.
    cgk_1D : float ndarray, shape (oversampling * (support // 2 + 1),)
        Oversampled 1-D prolate spheroidal wave function (PSWF) kernel.
        The 2-D kernel is formed as the outer product of two 1-D lookups.
    n_uv : int ndarray, shape (2,)
        Padded UV-grid dimensions ``[n_u, n_v]`` in pixels.
    delta_lm : float ndarray, shape (2,)
        Image cell size in radians ``[delta_l, delta_m]``.
    support : int
        Full convolution support width in grid pixels.  The kernel spans
        ``[-support//2, support - support//2)`` pixels around each sample.
    oversampling : int
        Oversampling factor of the 1-D kernel ``cgk_1D``.

    Returns
    -------
    None
        ``grid`` and ``normalization`` are modified in place; nothing is
        returned.

    Notes
    -----
    - Visibilities whose U or V coordinate is ``NaN`` are skipped entirely.
    - Samples with ``NaN`` or zero weight are skipped.
    - Sub-pixel offsets are rounded with ``int(x + 0.5)`` rather than
      ``numpy.round`` to match Fortran/C++ rounding behaviour
      (round-half-away-from-zero).
    - Hardcoding ``support`` and ``oversampling`` as literals (currently
      commented out) allows Numba to unroll the inner support loops,
      yielding a significant speed-up.
    """

    # By hardcoding the support and oversampling values, the innermost for loops can be unrolled by the compiler leading to significantly faster code.
    # support = 7
    # oversampling = 100

    c = 299792458.0
    uv_scale = np.zeros((2, len(frequency_coord)), dtype=np.double)
    uv_scale[0, :] = -(frequency_coord * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(frequency_coord * delta_lm[1] * n_uv[1]) / c

    # oversampling_center = int(oversampling // 2)
    support_center = int(support // 2)
    uv_center = n_uv // 2

    start_support = -support_center
    end_support = (
        support - support_center
    )  # end_support is larger by 1 so that python range() gives correct indices

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
                            weight_data = weight[i_time, i_baseline, i_chan, i_pol]

                            if ~np.isnan(weight_data) and (weight_data != 0.0):
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
