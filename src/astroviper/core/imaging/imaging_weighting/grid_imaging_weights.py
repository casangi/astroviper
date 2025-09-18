from numba import jit
import numpy as np
import math


def grid_imaging_weights(
    grid: np.ndarray,
    sum_weight: np.ndarray,
    uvw: np.ndarray,
    data_weight: np.ndarray,
    freq_chan: np.ndarray,
    grid_parms: dict,
):
    """
    Grid per-visibility *data weights* onto a UV grid.

    This is a thin Python wrapper that prepares mapping arrays and basic parameters,
    and then calls the Numba-jitted inner kernel :func:`grid_imaging_weights_jit`.

    Parameters
    ----------
    grid : np.ndarray (float64/32), shape (n_chan, n_pol, n_u, n_v)
        Output UV-plane grid of *accumulated data weights*. Updated in-place.
        For each (channel, polarization), the kernel adds the per-visibility
        data weight at the nearest (u, v) pixel and its conjugate location.
    sum_weight : np.ndarray (float64), shape (n_chan, n_pol)
        Per-(channel, polarization) sum of gridded data weights. Updated in-place.
        The kernel adds ``2 * weight`` for each successfully gridded visibility
        (accounting for the conjugate update).
    uvw : np.ndarray (float64), shape (n_time, n_baseline, 3)
        UVW coordinates in meters for each time and baseline. Only the
        u and v components ([..., 0] and [..., 1]) are used here.
    dataweight : np.ndarray (float64), shape (n_time, n_baseline, n_vis_chan, n_pol)
        Per-visibility *data weights* (e.g., 1/variance). The kernel currently
        uses polarization index 0 (see Notes).
    freq_chan : np.ndarray (float64), shape (n_chan,)
        Sky frequencies (Hz) for each visibility channel used to scale
        meters -> wavelengths and to compute UV pixel coordinates.
    grid_parms : dict
        Dictionary of gridding parameters with required keys:
        - ``"image_size"`` : tuple(int, int)
            Target padded image size in pixels along (u, v). This is also
            the UV grid size.
        - ``"cell_size"`` : tuple(float, float)
            Pixel scale (Δl, Δm) in radians along the two image axes.

    Returns
    -------
    None
        The function operates in-place on ``grid`` and ``sum_weight``.

    Notes
    -----
    * Polarization handling: the kernel enforces
      ``assert weight.shape[3] < 3`` and currently grids only polarization 0.
      If you intend to combine polarizations (e.g., average PP and QQ),
      adjust the polarization logic in the jitted function accordingly.
    * Rounding: to match historical Fortran/CASA behavior, UV pixel indices are
      computed via ``int(x + 0.5)`` (rather than ``np.round``).

    See Also
    --------
    grid_imaging_weights_jit : Numba-jitted inner kernel that performs the work.
    """

    n_chan = data_weight.shape[2]  # number of *visibility* channels
    chan_map = (np.arange(0, n_chan)).astype(int)  # identity channel map

    n_imag_pol = data_weight.shape[3]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)  # not used by the kernel yet

    n_uv = grid_parms["image_size"]
    delta_lm = grid_parms["cell_size"]

    # Only PP or (PP, QQ) is supported here; adjust if more pols are added later.
    assert data_weight.shape[3] < 3, "Polarization should be PP or PP, QQ."

    # Dispatch to the jitted inner kernel (updates grid and sum_weight in-place).
    grid_imaging_weights_jit(
        grid,
        sum_weight,
        uvw,
        freq_chan,
        chan_map,
        data_weight,
        n_uv,
        delta_lm,
    )


# When Numba JIT is used, Python's round is not used; we explicitly mimic legacy
# Fortran/C rounding via int(x + 0.5) where x is non-negative (see below).
@jit(nopython=True, cache=True, nogil=True)  # fastmath can be enabled if desired
def grid_imaging_weights_jit(
    grid: np.ndarray,
    sum_weight: np.ndarray,
    uvw: np.ndarray,
    freq_chan: np.ndarray,
    chan_map: np.ndarray,
    data_weights: np.ndarray,
    n_uv: list,
    delta_lm: list,
):
    """
    Jitted inner kernel to grid per-visibility *data weights* to a UV grid.

    Parameters
    ----------
    grid : np.ndarray (float64/32), shape (n_chan, n_pol, n_u, n_v)
        UV grid to accumulate weights into. Updated in-place.
    sum_weight : np.ndarray (float64), shape (n_chan, n_pol)
        Accumulator for per-(chan, pol) total of weights. Updated in-place.
    uvw : np.ndarray (float64), shape (n_time, n_baseline, 3)
        Per-visibility UVW baseline vectors in meters.
    freq_chan : np.ndarray (float64), shape (n_chan,)
        Frequencies (Hz) of visibility channels.
    chan_map : np.ndarray (int64), shape (n_chan,)
        Mapping from visibility channel index to imaging channel index.
        Here typically ``chan_map[i] == i`` (identity).
    data_weights : np.ndarray (float64), shape (n_time, n_baseline, n_vis_chan, n_pol)
        Per-visibility *data weights* to be gridded.
        Currently only polarization index 0 is used (see Notes).
    n_uv : tuple(int, int)
        UV grid dimensions (n_u, n_v).
    delta_lm : tuple(float, float)
        Image-plane pixel scale (Δl, Δm) in radians.

    Returns
    -------
    None
        Operates in-place on ``grid`` and ``sum_weight``.

    Notes
    -----
    * The kernel applies Hermitian symmetry: for each (u, v) hit, it also
      updates (-u, -v). This ensures the synthesized image is real-valued.
    * Pixel indexing uses ``int(x + 0.5)`` on non-negative ``x`` to emulate
      historical Fortran/C behavior and to match CASA rounding semantics.
    """

    # Speed of light (m/s).
    c = 299792458.0

    # uv_scale maps meters -> UV pixels for each frequency and image pixel scale.
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c

    # UV grid center in pixel coordinates (for rounding to nearest pixel).
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

                # Convert baseline meters -> fractional pixels at this frequency.
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if (not np.isnan(u)) and (not np.isnan(v)):
                    # Shift so that (0,0) baseline maps to grid center.
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]

                    # Conjugate-symmetric location (-u, -v).
                    u_pos_conj = -u + uv_center[0]
                    v_pos_conj = -v + uv_center[1]

                    # Fortran/CASA-compatible rounding to nearest pixel (u_pos, v_pos >= 0).
                    u_indx = int(u_pos + 0.5)
                    v_indx = int(v_pos + 0.5)

                    u_indx_conj = int(u_pos_conj + 0.5)
                    v_indx_conj = int(v_pos_conj + 0.5)

                    # Bounds check for both direct and conjugate pixels.
                    if (
                        (u_indx < n_u)
                        and (v_indx < n_v)
                        and (u_indx >= 0)
                        and (v_indx >= 0)
                    ):
                        weight = data_weights[i_time, i_baseline, i_chan, 0]

                        if not np.isnan(weight):
                            # Accumulate weight at (u, v) and its conjugate (-u, -v).
                            grid[a_chan, 0, u_indx, v_indx] = (
                                grid[a_chan, 0, u_indx, v_indx] + weight
                            )
                            grid[a_chan, 0, u_indx_conj, v_indx_conj] = (
                                grid[a_chan, 0, u_indx_conj, v_indx_conj] + weight
                            )

                            # Track total contribution (factor 2 for conjugate update).
                            sum_weight[a_chan, 0] = sum_weight[a_chan, 0] + 2.0 * weight
    return


def degrid_imaging_weights(
    grid_imaging_weight,
    uvw,
    data_weight,
    briggs_factors,
    freq_chan,
    grid_parms,
):
    """
    Sample a UV *imaging weight grid* at each visibility's (u, v) to form
    per-visibility imaging weights (e.g., natural/Briggs/robust).

    Parameters
    ----------
    grid_imaging_weight : np.ndarray (float64/32), shape (n_chan, n_pol, n_u, n_v)
        UV-plane *imaging weight grid* (already constructed; typically real).
    uvw : np.ndarray (float64), shape (n_time, n_baseline, 3)
        UVW coordinates in meters.
    data_weight : np.ndarray (float64), shape (n_time, n_baseline, n_vis_chan, n_pol)
        Used as the starting point before applying the UV-grid-based reweighting.
    briggs_factors : np.ndarray (float64), shape (2, n_chan, n_pol)
        Pre-computed Briggs/robust factors. The per-sample denominator is:
        ``briggs_factors[0, chan, pol] * grid_imaging_weight + briggs_factors[1, chan, pol]``.
    freq_chan : np.ndarray (float64), shape (n_chan,)
        Frequencies (Hz) of visibility channels.
    grid_parms : dict
        Dictionary with required keys:
        - ``"image_size"`` : tuple(int, int), UV grid size.
        - ``"cell_size"`` : tuple(float, float), image pixel scale (Δl, Δm) in radians.

    Returns
    -------
    imaging_weight : np.ndarray (float64), shape like ``data_weight``
        The per-visibility imaging weights after sampling the UV grid and
        applying the Briggs/robust denominator.

    Notes
    -----
    * Channel and polarization maps are currently identity and single-pol
      respectively in the jitted kernel. If you have more polarizations, extend
      ``pol_map`` accordingly.
    * The kernel checks bounds and NaNs and leaves samples unchanged if the UV
      location falls outside the grid or if the corresponding grid value is 0/NaN.
    """

    n_imag_chan = data_weight.shape[2]

    # Always imaging as a cube for imaging weights (identity channel map).
    chan_map = (np.arange(0, n_imag_chan)).astype(int)

    # Single-pol degrid by default (extend to >1 if needed).
    pol_map = (np.arange(0, 1)).astype(int)

    n_uv = grid_parms["image_size"]
    delta_lm = grid_parms["cell_size"]

    # Output array mirrors data_weight shape.
    imaging_weight = np.zeros(data_weight.shape, dtype=np.double)

    # Dispatch to the jitted kernel (in-place update of imaging_weight).
    degrid_imaging_weights_jit(
        imaging_weight,
        grid_imaging_weight,
        briggs_factors,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        data_weight,
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
    data_weight,
    n_uv,
    delta_lm,
):
    """
    Jitted kernel to sample a UV imaging-weight grid at each visibility and
    compute per-visibility imaging weights (with Briggs/robust denominator).

    Parameters
    ----------
    imaging_weight : np.ndarray (float64), shape like ``data_weight``
        Output array. Updated in-place with the final imaging weights.
    grid_imaging_weight : np.ndarray (float64/32), shape (n_chan, n_pol, n_u, n_v)
        UV imaging-weight grid to be sampled at rounded UV pixel coordinates.
    briggs_factors : np.ndarray (float64), shape (2, n_chan, n_pol)
        Coefficients ``a`` and ``b`` such that the denominator is
        ``a * grid_imaging_weight + b`` at the sampled UV pixel.
    uvw : np.ndarray (float64), shape (n_time, n_baseline, 3)
        Baseline coordinates in meters. Only u and v are used.
    freq_chan : np.ndarray (float64), shape (n_chan,)
        Frequencies (Hz) of visibility channels.
    chan_map : np.ndarray (int64), shape (n_chan,)
        Mapping from visibility channels to imaging channels.
    pol_map : np.ndarray (int64), shape (n_pol_out,)
        Polarization mapping for output (currently single-pol).
    data_weight : np.ndarray (float64), shape (n_time, n_baseline, n_vis_chan, n_pol)
        Input per-visibility natural imaging weights (starting point).
    n_uv : tuple(int, int)
        UV grid dimensions (n_u, n_v).
    delta_lm : tuple(float, float)
        Image-plane pixel scale (Δl, Δm) in radians.

    Returns
    -------
    None
        Operates in-place on ``imaging_weight``.

    Notes
    -----
    * The per-visibility imaging weight is initialized from
      ``data_weight`` (averaging pols if two are present—see code),
      then divided by the Briggs denominator sampled from the UV grid.
    * Pixel indices are computed using the same rounding convention as the
      gridding kernel (``int(x + 0.5)``).
    """

    c = 299792458.0

    # uv_scale maps baseline meters -> UV pixels for each frequency & pixel scale.
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

    # Loop over all visibilities and channels.
    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(n_chan):
                a_chan = chan_map[i_chan]

                # Convert baseline meters -> fractional pixels.
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]

                if (not np.isnan(u)) and (not np.isnan(v)):
                    # Shift to grid-centered coordinates and round to nearest pixel.
                    u_pos = u + uv_center[0]
                    v_pos = v + uv_center[1]

                    u_center_indx = int(u_pos + 0.5)
                    v_center_indx = int(v_pos + 0.5)

                    # Bounds/validity checks before sampling the grid.
                    if (
                        (u_center_indx < n_u)
                        and (v_center_indx < n_v)
                        and (u_center_indx >= 0)
                        and (v_center_indx >= 0)
                    ):
                        for i_pol in range(n_pol):
                            a_pol = pol_map[i_pol]

                            # # Initialize from natural weights:
                            # # If two pols exist, average them; else keep the same pol.
                            # if data_weight.shape[3] == 2:
                            #     imaging_weight[i_time, i_baseline, i_chan, i_pol] = (
                            #         data_weight[
                            #             i_time, i_baseline, i_chan, 0
                            #         ]
                            #         + data_weight[
                            #             i_time, i_baseline, i_chan, 1
                            #         ]
                            #     ) / 2.0
                            # else:
                            #     imaging_weight[i_time, i_baseline, i_chan, i_pol] = (
                            #         data_weight[
                            #             i_time, i_baseline, i_chan, i_pol
                            #         ]
                            #     )

                            imaging_weight[i_time, i_baseline, i_chan, i_pol] = (
                                data_weight[i_time, i_baseline, i_chan, i_pol]
                            )

                            # Only proceed if the natural weight is finite and non-zero.
                            if not np.isnan(
                                data_weight[i_time, i_baseline, i_chan, i_pol]
                            ) and (
                                data_weight[i_time, i_baseline, i_chan, i_pol] != 0.0
                            ):
                                gij = grid_imaging_weight[
                                    a_chan, a_pol, u_center_indx, v_center_indx
                                ]

                                # Require finite/non-zero grid weight as well.
                                if (not np.isnan(gij)) and (gij != 0.0):

                                    if np.isnan(
                                        briggs_factors[0, a_chan, a_pol]
                                    ) or np.isnan(briggs_factors[1, a_chan, a_pol]):
                                        print(
                                            "NaN in briggs_factors", briggs_factors, gij
                                        )
                                        raise ValueError("NaN in briggs_factors")
                                    # Briggs denominator: a * G + b
                                    denom = (
                                        briggs_factors[0, a_chan, a_pol] * gij
                                        + briggs_factors[1, a_chan, a_pol]
                                    )

                                    # Apply robust/Briggs reweighting.
                                    imaging_weight[
                                        i_time, i_baseline, i_chan, i_pol
                                    ] = (
                                        imaging_weight[
                                            i_time, i_baseline, i_chan, i_pol
                                        ]
                                        / denom
                                    )

    return
