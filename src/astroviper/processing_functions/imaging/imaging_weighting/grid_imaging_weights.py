import numpy as np


def _resolve_frequency_map(frequency_map, n_vis_chan, grid):
    """Return a validated ``int64`` channel map for the weight kernels.

    ``None`` yields the identity map. Every entry must name a plane of
    ``grid`` (first axis) and there must be one entry per visibility channel,
    because the C++ kernels index the grid with these values unchecked.
    """
    if frequency_map is None:
        chan_map = np.arange(n_vis_chan, dtype=np.int64)
    else:
        chan_map = np.ascontiguousarray(frequency_map, dtype=np.int64)
    if chan_map.shape != (n_vis_chan,):
        raise ValueError(
            "frequency_map must contain one grid-plane index per visibility "
            f"channel; received shape {chan_map.shape} for {n_vis_chan} channels."
        )
    n_chan_grid = grid.shape[0]
    if chan_map.size and (chan_map.min() < 0 or chan_map.max() >= n_chan_grid):
        raise ValueError(
            "frequency_map contains an index outside the imaging-weight grid "
            f"({n_chan_grid} channel planes)."
        )
    return chan_map


def grid_imaging_weights(
    grid: np.ndarray,
    sum_weight: np.ndarray,
    uvw: np.ndarray,
    data_weight: np.ndarray,
    freq_chan: np.ndarray,
    # grid_parms: dict,
    n_uv: list,
    delta_lm: list,
    # Currently unused: the weight gridding kernel stays serial so weight sums
    # are bit-reproducible; accepted for API consistency across the stack.
    processing_function_threads: int = 1,
    frequency_map: np.ndarray | None = None,
):
    """
    Grid per-visibility *data weights* onto a UV grid.

    This is a thin Python wrapper that prepares mapping arrays and basic parameters,
    and then calls the serial C++ kernel
    :func:`~astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights_cpp.grid_imaging_weights`.

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
    frequency_map : np.ndarray (int64), shape (n_vis_chan,), optional
        Grid plane (image channel) each visibility channel accumulates into.
        ``None`` (default) uses the identity map, i.e. the grid has one plane
        per visibility channel. Pass the map returned by
        :func:`~astroviper.processing_functions.imaging.utils.frequency_mapping.map_visibility_frequencies_to_image`
        when the visibility channels are a subset of, or offset from, the
        image frequency axis.

    n_uv : tuple(int, int)
            Target padded image size in pixels along (u, v). This is also
            the UV grid size.
    cell_size : tuple(float, float)
            Pixel scale (Δl, Δm) in radians along the two image axes.

    Returns
    -------
    None
        The function operates in-place on ``grid`` and ``sum_weight``.

    Notes
    -----
    * Polarization handling: the wrapper enforces
      ``assert weight.shape[3] < 3`` and the kernel currently grids only
      polarization 0. If you intend to combine polarizations (e.g., average PP
      and QQ), adjust the polarization logic in the C++ kernel accordingly.
    * Rounding: to match historical Fortran/CASA behavior, UV pixel indices are
      computed by rounding to the nearest pixel (``floor(x + 0.5)``).
    """
    from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights_cpp import (
        grid_imaging_weights as grid_imaging_weights_cpp,
    )

    chan_map = _resolve_frequency_map(frequency_map, data_weight.shape[2], grid)

    # Only PP or (PP, QQ) is supported here; adjust if more pols are added later.
    assert data_weight.shape[3] < 3, "Polarization should be PP or PP, QQ."

    # Dispatch to the C++ kernel (updates grid and sum_weight in-place).
    grid_imaging_weights_cpp(
        grid,
        sum_weight,
        uvw,
        freq_chan,
        chan_map,
        data_weight,
        np.asarray(n_uv, dtype=np.int64),
        np.asarray(delta_lm, dtype=np.float64),
        processing_function_threads=processing_function_threads,
    )


def degrid_imaging_weights(
    grid_imaging_weight,
    uvw,
    data_weight,
    briggs_factors,
    freq_chan,
    n_uv,
    delta_lm,
    # Currently unused: the weight degridding kernel stays serial so weight
    # sums are bit-reproducible; accepted for API consistency across the stack.
    processing_function_threads: int = 1,
    frequency_map: np.ndarray | None = None,
):
    """
    Sample a UV *imaging weight grid* at each visibility's (u, v) to form
    per-visibility imaging weights (e.g., natural/Briggs/robust).

    This is a thin Python wrapper around the serial C++ kernel
    :func:`~astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights_cpp.degrid_imaging_weights`.

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
    frequency_map : np.ndarray (int64), shape (n_vis_chan,), optional
        Grid plane (image channel) each visibility channel samples; must be the
        same map used to grid the weights. ``None`` (default) uses the
        identity map.
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
    * The polarization map is currently single-pol. If you have more
      polarizations, extend ``pol_map`` accordingly.
    * The kernel checks bounds and NaNs and leaves samples unchanged if the UV
      location falls outside the grid or if the corresponding grid value is 0/NaN.
    """
    from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights_cpp import (
        degrid_imaging_weights as degrid_imaging_weights_cpp,
    )

    chan_map = _resolve_frequency_map(
        frequency_map, data_weight.shape[2], grid_imaging_weight
    )

    # Single-pol degrid by default (extend to >1 if needed).
    pol_map = (np.arange(0, 1)).astype(int)

    # Output array mirrors data_weight shape.
    imaging_weight = np.zeros(data_weight.shape, dtype=np.double)

    # Dispatch to the C++ kernel (in-place update of imaging_weight).
    degrid_imaging_weights_cpp(
        imaging_weight,
        grid_imaging_weight,
        briggs_factors,
        uvw,
        freq_chan,
        chan_map,
        pol_map,
        data_weight,
        np.asarray(n_uv, dtype=np.int64),
        np.asarray(delta_lm, dtype=np.float64),
        processing_function_threads=processing_function_threads,
    )

    return imaging_weight
