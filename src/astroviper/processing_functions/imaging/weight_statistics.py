"""Theoretical sensitivity and weight statistics of the imaging weights.

Everything here is computed from what the point spread function pass already
holds in memory (the natural weights ``WEIGHT``, the imaging weights
``WEIGHT_IMAGING``, ``FLAG`` and ``UVW``); no visibility is read.

Sensitivity definition. For a correlation ``c`` the dirty image at any pixel
is ``sum_k W_ck Re(V_ck e^(i phi_k)) / sum_k W_ck``, so its thermal noise is

``sigma_c = sqrt(sum_k W_ck**2 / w_ck) / sum_k W_ck``

over the valid cross-correlation samples ``k`` (unique visibilities, no
conjugate doubling), with ``w`` the natural weight ``WEIGHT = 1 / sigma_vis**2``
of one real or imaginary component and ``W`` the imaging weight. The noise of
the different correlations is independent, so a Stokes plane ``S`` formed with
the imager's instrument basis to Stokes matrix ``M`` (for example
``I = (XX + YY) / 2``) has ``sigma_S**2 = sum_c |M_Sc|**2 sigma_c**2``. The
natural sensitivity takes ``W = w`` on the same samples, which gives
``sigma_c = 1 / sqrt(sum_k w_ck)``. These are absolute noise levels in Jy/beam
only when the weights are absolute (a simulation with noise writes them so; a
simulation without noise writes ``WEIGHT = 1``).
"""

import numpy as np

SPEED_OF_LIGHT = 299792458.0  # m/s

# Samples (time x baseline x frequency x polarization) per block of whole
# integrations in calculate_weight_statistics_single_field; the copies and
# temporaries of a block take about 60 B per sample.
BLOCK_SAMPLES = 2**22

# Per (image channel, correlation) sums accumulated over the valid samples, in
# the order accumulate_correlation_weight_sums returns them.
CORRELATION_WEIGHT_SUM_NAMES = (
    "sum_weight_imaging",
    "sum_weight_imaging_squared",
    "sum_weight_imaging_squared_over_weight",
    "sum_weight",
    "n_valid",
)


def valid_sample_mask(weight, weight_imaging, flag):
    """Samples that enter the image: not flagged, a finite positive natural
    weight and a finite imaging weight (the imager never grids a ``NaN``
    imaging weight, for example the equalized weight of a parallel-hand pair
    with one flagged hand).

    Parameters
    ----------
    weight, weight_imaging : numpy.ndarray
        Natural and imaging weights, any (identical) shape.
    flag : numpy.ndarray of bool
        Flags, same shape.

    Returns
    -------
    numpy.ndarray of bool
        Same shape as the inputs.
    """
    with np.errstate(invalid="ignore"):
        return (
            ~np.asarray(flag, dtype=bool)
            & np.isfinite(weight)
            & (weight > 0)
            & np.isfinite(weight_imaging)
        )


def accumulate_correlation_weight_sums(
    weight, weight_imaging, flag, image_channel_map, n_image_channels, sums=None
):
    """Add the per image channel and correlation weight sums of one measurement
    set, or of a block of its integrations, to ``sums``.

    Parameters
    ----------
    weight : numpy.ndarray, shape (time, baseline_id, frequency, polarization)
        Natural weights ``w`` (one real or imaginary component), cross
        correlations only.
    weight_imaging : numpy.ndarray
        Imaging weights ``W``, same shape.
    flag : numpy.ndarray of bool
        Flags, same shape.
    image_channel_map : numpy.ndarray of int, shape (frequency,)
        Image channel of every visibility channel.
    n_image_channels : int
        Number of image channels.
    sums : dict, optional
        Sums of earlier calls to add to (updated in place); ``None`` starts
        from zero.

    Returns
    -------
    dict
        ``{name: numpy.ndarray of shape (n_image_channels, polarization)}`` for
        the names in :data:`CORRELATION_WEIGHT_SUM_NAMES`, plus ``n_samples``
        and ``n_flagged`` of shape ``(n_image_channels,)``: every sample
        (time, baseline, visibility channel, correlation) and those excluded
        from the image (flagged, or a non-finite or non-positive natural
        weight).
    """
    weight = np.asarray(weight, dtype=np.float64)
    weight_imaging = np.asarray(weight_imaging, dtype=np.float64)
    n_pol = weight.shape[-1]
    if sums is None:
        sums = {
            name: np.zeros((n_image_channels, n_pol))
            for name in CORRELATION_WEIGHT_SUM_NAMES
        }
        sums["n_samples"] = np.zeros(n_image_channels)
        sums["n_flagged"] = np.zeros(n_image_channels)

    valid = valid_sample_mask(weight, weight_imaging, flag)
    w = np.where(valid, weight, 0.0)
    w_imaging = np.where(valid, weight_imaging, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_imaging_squared_over_w = np.where(valid, w_imaging**2 / weight, 0.0)
    # Reduce over time and baseline, then add every visibility channel onto
    # its image channel.
    per_visibility_channel = {
        "sum_weight_imaging": w_imaging.sum(axis=(0, 1)),
        "sum_weight_imaging_squared": (w_imaging**2).sum(axis=(0, 1)),
        "sum_weight_imaging_squared_over_weight": w_imaging_squared_over_w.sum(
            axis=(0, 1)
        ),
        "sum_weight": w.sum(axis=(0, 1)),
        "n_valid": valid.sum(axis=(0, 1)),
    }
    for name, value in per_visibility_channel.items():
        np.add.at(sums[name], image_channel_map, value)

    with np.errstate(invalid="ignore"):
        excluded = np.asarray(flag, dtype=bool) | ~np.isfinite(weight) | ~(weight > 0)
    np.add.at(
        sums["n_samples"],
        image_channel_map,
        np.full(weight.shape[2], weight.shape[0] * weight.shape[1] * n_pol),
    )
    np.add.at(sums["n_flagged"], image_channel_map, excluded.sum(axis=(0, 1, 3)))
    return sums


def accumulate_uv_extent(
    uvw, frequency, valid, image_channel_map, n_image_channels, extent=None
):
    """Add the uv extent (in wavelengths) of the valid samples of one
    measurement set, or of a block of its integrations, to ``extent``.

    Parameters
    ----------
    uvw : numpy.ndarray, shape (time, baseline_id, 3)
        Baseline coordinates in metres.
    frequency : numpy.ndarray, shape (frequency,)
        Visibility channel frequencies in Hz.
    valid : numpy.ndarray of bool, shape (time, baseline_id, frequency, polarization)
        Valid samples (:func:`valid_sample_mask`); a (time, baseline,
        visibility channel) sample counts when any correlation is valid.
    image_channel_map : numpy.ndarray of int, shape (frequency,)
        Image channel of every visibility channel.
    n_image_channels : int
        Number of image channels.
    extent : dict, optional
        Extent of earlier calls (updated in place); ``None`` starts empty.

    Returns
    -------
    dict
        ``uv_distance_max``, ``uv_distance_min`` and ``abs_w_max``, each of
        shape ``(n_image_channels,)``; ``NaN`` for a channel without valid
        samples.
    """
    if extent is None:
        extent = {
            "uv_distance_max": np.full(n_image_channels, -np.inf),
            "uv_distance_min": np.full(n_image_channels, np.inf),
            "abs_w_max": np.full(n_image_channels, -np.inf),
        }
    uvw = np.asarray(uvw, dtype=np.float64)
    valid_any = np.asarray(valid).any(axis=-1)  # (time, baseline, frequency)
    uv_distance = np.hypot(uvw[..., 0], uvw[..., 1])[..., np.newaxis]
    abs_w = np.abs(uvw[..., 2])[..., np.newaxis]
    wavelengths_per_metre = np.asarray(frequency, dtype=np.float64) / SPEED_OF_LIGHT
    has_valid = valid_any.any(axis=(0, 1))
    uv_max = np.where(valid_any, uv_distance, -np.inf).max(axis=(0, 1))
    uv_min = np.where(valid_any, uv_distance, np.inf).min(axis=(0, 1))
    w_max = np.where(valid_any, abs_w, -np.inf).max(axis=(0, 1))
    channel = image_channel_map[has_valid]
    scale = wavelengths_per_metre[has_valid]
    np.maximum.at(extent["uv_distance_max"], channel, uv_max[has_valid] * scale)
    np.minimum.at(extent["uv_distance_min"], channel, uv_min[has_valid] * scale)
    np.maximum.at(extent["abs_w_max"], channel, w_max[has_valid] * scale)
    return extent


def stokes_weight_statistics(sums, correlation_labels, stokes_labels):
    """Combine the per-correlation weight sums into per Stokes plane statistics.

    Parameters
    ----------
    sums : dict
        Output of :func:`accumulate_correlation_weight_sums`.
    correlation_labels : list of str
        Correlation of every column of the sums (for example ``["XX", "YY"]``).
    stokes_labels : list of str
        Stokes planes to return (for example ``["I", "Q"]``), each formed with
        the instrument basis to Stokes matrix of
        :func:`~astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`.

    Returns
    -------
    dict
        ``{name: numpy.ndarray of shape (n_image_channels, len(stokes_labels))}``:

        - ``sensitivity``: ``sqrt(sum_c |M_Sc|**2 sigma_c**2)`` with
          ``sigma_c = sqrt(sum W**2 / w) / sum W``.
        - ``sensitivity_natural``: the same with ``W = w``.
        - ``sum_of_weights``: ``sum W`` of the correlations forming the plane,
          the mean per correlation.
        - ``effective_number_of_visibilities``: ``(sum W)**2 / sum W**2`` per
          correlation, the mean over the correlations forming the plane.

        A plane with a correlation without valid samples has an infinite
        sensitivity.
    """
    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        _select_transform_matrix,
    )

    matrix, in_labels, out_labels = _select_transform_matrix(
        frozenset(correlation_labels), "stokes"
    )
    rows = [out_labels.index(stokes) for stokes in stokes_labels]
    columns = [in_labels.index(correlation) for correlation in correlation_labels]
    matrix_squared = np.abs(np.asarray(matrix)[np.ix_(rows, columns)]) ** 2
    forms_plane = matrix_squared > 0
    n_forming = forms_plane.sum(axis=1)

    sum_weight_imaging = sums["sum_weight_imaging"]
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(
            sum_weight_imaging > 0,
            sums["sum_weight_imaging_squared_over_weight"] / sum_weight_imaging**2,
            np.inf,
        )
        variance_natural = np.where(
            sums["sum_weight"] > 0, 1.0 / sums["sum_weight"], np.inf
        )
        effective_number = np.where(
            sum_weight_imaging > 0,
            sum_weight_imaging**2 / sums["sum_weight_imaging_squared"],
            0.0,
        )
    return {
        "sensitivity": _combine_variances(variance, matrix_squared),
        "sensitivity_natural": _combine_variances(variance_natural, matrix_squared),
        "sum_of_weights": sum_weight_imaging @ forms_plane.T / n_forming,
        "effective_number_of_visibilities": effective_number
        @ forms_plane.T
        / n_forming,
    }


def _combine_variances(variance, matrix_squared):
    """``sqrt(sum_c |M_Sc|**2 variance_c)`` per channel and Stokes plane, with
    correlations that do not form a plane left out (``inf * 0`` would be
    ``NaN``)."""
    contributions = np.where(
        matrix_squared[np.newaxis] > 0,
        variance[:, np.newaxis, :] * matrix_squared[np.newaxis],
        0.0,
    )
    return np.sqrt(contributions.sum(axis=-1))


def _cross_correlation_values(data_array, cross, order):
    """Values of a measurement-set variable on the cross-correlation baselines
    ``cross``, as (time, baseline_id, frequency, polarization) with the
    correlations in ``order``, or (time, baseline_id, ...) without a
    polarization axis."""
    if "polarization" in data_array.dims:
        data_array = data_array.transpose(
            "time", "baseline_id", "frequency", "polarization"
        )
        return data_array.values[:, cross][..., order]
    return data_array.transpose("time", "baseline_id", ...).values[:, cross]


def calculate_weight_statistics_single_field(
    ps_xdt, img_xds, processing_set_data_group_name="corrected"
):
    """Theoretical sensitivity and weight statistics per image channel and
    Stokes plane.

    Call it after the imaging weights are computed (the data group then holds
    ``weight_imaging``); see the module docstring for the definitions.
    Autocorrelations are left out (without modifying ``ps_xdt``), and every
    visibility channel counts for the image channel the gridders map it onto.
    The samples are copied and reduced in blocks of whole integrations of at
    most :data:`BLOCK_SAMPLES` samples (one integration if it has more), so
    the memory this adds to ``ps_xdt`` does not grow with the number of
    integrations.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or dict
        Measurement sets ``{ms_name: node}`` with the data group's ``weight``,
        ``weight_imaging``, ``flag`` and ``uvw`` roles.
    img_xds : xarray.Dataset
        Image whose ``frequency`` coordinate gives the image channels and
        whose ``polarization`` coordinate gives the Stokes planes to return.
    processing_set_data_group_name : str, optional
        Measurement-set data group to read the roles from.

    Returns
    -------
    xarray.Dataset
        Dims ``(frequency, polarization)``: ``SENSITIVITY`` and
        ``SENSITIVITY_NATURAL`` (Jy/beam for absolute weights),
        ``SUM_OF_WEIGHTS``, ``EFFECTIVE_NUMBER_OF_VISIBILITIES``; dims
        ``(frequency,)``: ``UV_DISTANCE_MAX``, ``UV_DISTANCE_MIN`` and
        ``ABS_W_MAX`` (wavelengths, ``NaN`` without valid samples) and
        ``FLAGGED_FRACTION``.
    """
    import xarray as xr

    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        map_visibility_frequencies_to_image,
    )

    image_frequencies = img_xds.frequency.values
    n_image_channels = image_frequencies.size
    correlation_labels = None
    sums = None
    extent = None
    for ms_xdt in ps_xdt.values():
        data_group = ms_xdt.attrs["data_groups"][processing_set_data_group_name]
        cross = (
            ms_xdt["baseline_antenna1_name"].values
            != ms_xdt["baseline_antenna2_name"].values
        )
        labels = [str(label) for label in ms_xdt.polarization.values]
        if correlation_labels is None:
            correlation_labels = labels
        order = [labels.index(label) for label in correlation_labels]
        variables = [
            ms_xdt[data_group[role]]
            for role in ("weight", "weight_imaging", "flag", "uvw")
        ]
        image_channel_map = map_visibility_frequencies_to_image(
            ms_xdt.frequency.values, image_frequencies
        )
        samples_per_time = int(cross.sum()) * ms_xdt.sizes["frequency"] * len(order)
        times_per_block = max(1, BLOCK_SAMPLES // max(samples_per_time, 1))
        for start in range(0, ms_xdt.sizes["time"], times_per_block):
            weight, weight_imaging, flag, uvw = (
                _cross_correlation_values(
                    variable.isel(time=slice(start, start + times_per_block)),
                    cross,
                    order,
                )
                for variable in variables
            )
            sums = accumulate_correlation_weight_sums(
                weight, weight_imaging, flag, image_channel_map, n_image_channels, sums
            )
            extent = accumulate_uv_extent(
                uvw,
                ms_xdt.frequency.values,
                valid_sample_mask(weight, weight_imaging, flag),
                image_channel_map,
                n_image_channels,
                extent,
            )

    stokes_labels = [str(label) for label in img_xds.polarization.values]
    statistics = stokes_weight_statistics(sums, correlation_labels, stokes_labels)
    for name, value in extent.items():
        extent[name] = np.where(np.isfinite(value), value, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        flagged_fraction = np.where(
            sums["n_samples"] > 0, sums["n_flagged"] / sums["n_samples"], np.nan
        )

    plane_dims = ("frequency", "polarization")
    return xr.Dataset(
        {
            "SENSITIVITY": (
                plane_dims,
                statistics["sensitivity"],
                {"units": "Jy/beam"},
            ),
            "SENSITIVITY_NATURAL": (
                plane_dims,
                statistics["sensitivity_natural"],
                {"units": "Jy/beam"},
            ),
            "SUM_OF_WEIGHTS": (plane_dims, statistics["sum_of_weights"]),
            "EFFECTIVE_NUMBER_OF_VISIBILITIES": (
                plane_dims,
                statistics["effective_number_of_visibilities"],
            ),
            "UV_DISTANCE_MAX": (
                ("frequency",),
                extent["uv_distance_max"],
                {"units": "wavelength"},
            ),
            "UV_DISTANCE_MIN": (
                ("frequency",),
                extent["uv_distance_min"],
                {"units": "wavelength"},
            ),
            "ABS_W_MAX": (
                ("frequency",),
                extent["abs_w_max"],
                {"units": "wavelength"},
            ),
            "FLAGGED_FRACTION": (("frequency",), flagged_fraction),
        },
        coords={"frequency": image_frequencies, "polarization": stokes_labels},
    )
