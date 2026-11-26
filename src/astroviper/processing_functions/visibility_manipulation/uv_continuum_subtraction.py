"""Continuum subtraction in the uv domain, the port of the pipeline's ``hif_uvcontsub``.

Every spectrum of a measurement set (one time, baseline and polarization) has
a polynomial in frequency fitted to its continuum channels and subtracted from
all of its channels, as CASA ``uvcontsub`` does with ``fitmethod="gsl"``
(``UVContSubTVI``, ``DenoisingLib``): a weighted least-squares fit, with the
data weights, of the real and imaginary parts to the unflagged continuum
channels, whose order is lowered to ``n - 1`` for a spectrum with
``n <= fit_order`` such channels. The polynomial is a function of the channel
frequency of the measurement set, as in CASA.

Departures from CASA and the pipeline:

- The continuum channels are found at every time: a channel is a continuum
  channel at the times its frequency, converted to the frame of the continuum
  ranges, lies in a range. The pipeline converts each range to one TOPO
  channel selection per measurement set, the channels that overlap the range
  at any time (``synthesisutils.advisechansel``), and fits those at all times.
- A spectrum without a valid continuum channel is flagged in the output; CASA
  copies it unsubtracted and unflagged.
- A sample of zero weight, or with a non-finite weight or visibility, is not
  a valid channel; CASA counts a zero-weight sample for the order and fits it
  with zero weight.
- The fit takes the weights of the input data group (XRADIO's ``WEIGHT``,
  from ``WEIGHT_SPECTRUM`` or ``WEIGHT``). CASA weights a fit of the ``DATA``
  column by ``1 / SIGMA_SPECTRUM**2`` and one of ``CORRECTED_DATA`` by
  ``WEIGHT_SPECTRUM``.
- The chi-squared of every fit is CASA's (the weighted sum of the squared
  residuals of the valid samples, for the real and imaginary parts); the
  statistics aggregate those of all fitted spectra. CASA's
  ``goodness_of_fit`` keeps one fit per buffer of rows and orders the
  extremes by the norm of the complex chi-squared.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import xarray as xr

from astroviper.utils.param_docs import shares_param_docs

VISIBILITY_DIMS = ("time", "baseline_id", "frequency", "polarization")
# Frames the continuum ranges may have for a TOPO measurement set (those of
# ms_spectral_frame_conversion.doppler_factor).
FRAMES_FROM_TOPO = ("lsrk", "lsrd", "BARY")
DEFAULT_MS_DATA_GROUP_OUT_MODIFIED = {
    "correlated_data": "VISIBILITY_CONTINUUM_SUBTRACTED",
    "flag": "FLAG_CONTINUUM_SUBTRACTED",
}
# Image frames (casacore names) whose XRADIO observer name differs
# (xradio.schema.measures.AllowedSpectralCoordFrames).
_OBSERVER_OF_IMAGE_FRAME = {"LSRK": "lsrk", "LSRD": "lsrd", "GEO": "gcrs"}
# Statistics of the fits per polarization, summed (counts and chi-squared
# sums) or taken as extremes over the spectra.
FIT_STATISTICS_SUMS = (
    "n_spectra",
    "n_fitted",
    "n_order_reduced",
    "n_without_continuum",
    "n_flagged_without_continuum",
    "chi_squared_sum_real",
    "chi_squared_sum_imag",
)
FIT_STATISTICS_MINIMA = ("chi_squared_min_real", "chi_squared_min_imag")
FIT_STATISTICS_MAXIMA = ("chi_squared_max_real", "chi_squared_max_imag")
# Normal equations of order 2 or more whose condition number exceeds this are
# fitted by least squares of the weighted samples instead.
NORMAL_EQUATIONS_CONDITION_LIMIT = 1e10
# OpenBLAS built with OpenMP returns wrong products when threads of one process
# call it at once (node tasks of a threaded scheduler): one call at a time.
_LINEAR_ALGEBRA_LOCK = threading.Lock()


def continuum_fit_from_find_continuum(find_continuum_result: dict) -> dict:
    """The continuum fit of a spectral window, from its ``find_continuum_ultra`` result.

    As ``hif_uvcontsub`` takes the fit from ``cont.dat`` since PL2024: the
    continuum frequency ranges and fit order 1, or 0 when the continuum finding
    flagged a low bandwidth or a low spread of the continuum channels.

    Parameters
    ----------
    find_continuum_result : dict
        Return dict of
        :func:`~astroviper.distributed_applications.imaging.find_continuum_ultra.find_continuum_ultra`
        for the spectral window: ``continuum_frequency_ranges``,
        ``frequency_frame`` (the cube's frame by its casacore name),
        ``low_bandwidth`` and ``low_spread``.

    Returns
    -------
    dict
        ``continuum_frequency_ranges``, ``continuum_frequency_frame`` (the
        frame by its XRADIO observer name: ``"LSRK"`` is ``"lsrk"``) and
        ``fit_order``, the keyword arguments of :func:`uv_continuum_subtraction`.
    """
    frame = find_continuum_result["frequency_frame"]
    low = find_continuum_result["low_bandwidth"] or find_continuum_result["low_spread"]
    return {
        "continuum_frequency_ranges": [
            [float(low_edge), float(high_edge)]
            for low_edge, high_edge in find_continuum_result[
                "continuum_frequency_ranges"
            ]
        ],
        "continuum_frequency_frame": _OBSERVER_OF_IMAGE_FRAME.get(frame, frame),
        "fit_order": 0 if low else 1,
    }


def check_continuum_fit(
    continuum_frequency_ranges,
    continuum_frequency_frame,
    fit_order,
    ms_data_group_out_modified=None,
) -> np.ndarray:
    """Check the configuration of a continuum subtraction.

    Parameters
    ----------
    continuum_frequency_ranges : array-like
        ``[low, high]`` pairs (Hz): at least one, finite, ``low <= high``.
    continuum_frequency_frame : str
        Spectral frame of the ranges, an XRADIO observer name.
    fit_order : int
        Polynomial order, at least 0.
    ms_data_group_out_modified : dict, optional
        Must name a different variable for each of exactly the roles
        ``correlated_data`` and ``flag``.

    Returns
    -------
    numpy.ndarray
        The ranges, shape ``(n_ranges, 2)``, float64.

    Raises
    ------
    ValueError
        If a check fails.
    """
    try:
        ranges = np.asarray(continuum_frequency_ranges, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "continuum_frequency_ranges must be [low, high] pairs of frequencies (Hz)."
        ) from error
    if ranges.ndim != 2 or ranges.shape[1] != 2 or ranges.shape[0] == 0:
        raise ValueError(
            "continuum_frequency_ranges must be a non-empty list of [low, high] "
            f"pairs (Hz), not an array of shape {ranges.shape}."
        )
    if not np.all(np.isfinite(ranges)) or np.any(ranges[:, 0] > ranges[:, 1]):
        raise ValueError(
            "continuum_frequency_ranges must be finite with low <= high in every pair."
        )
    if not isinstance(continuum_frequency_frame, str):
        raise ValueError("continuum_frequency_frame must be a string.")
    if isinstance(fit_order, bool) or not isinstance(fit_order, (int, np.integer)):
        raise ValueError(f"fit_order must be an integer, not {fit_order!r}.")
    if fit_order < 0:
        raise ValueError(f"fit_order must be at least 0, not {fit_order}.")
    if ms_data_group_out_modified is not None:
        if set(ms_data_group_out_modified) != set(DEFAULT_MS_DATA_GROUP_OUT_MODIFIED):
            raise ValueError(
                "ms_data_group_out_modified must name the output variables of "
                f"exactly the roles {sorted(DEFAULT_MS_DATA_GROUP_OUT_MODIFIED)}, "
                f"not {sorted(ms_data_group_out_modified)}."
            )
        if len(set(ms_data_group_out_modified.values())) != len(
            ms_data_group_out_modified
        ):
            raise ValueError(
                "ms_data_group_out_modified must name a different variable for "
                f"each role, not {ms_data_group_out_modified}."
            )
    return ranges


def continuum_channel_mask(
    frequency: np.ndarray,
    continuum_frequency_ranges,
    doppler_factors: np.ndarray | None = None,
) -> np.ndarray:
    """The channels whose frequency lies in a continuum range, at every time.

    Parameters
    ----------
    frequency : numpy.ndarray
        Channel frequencies of the measurement set (Hz), shape ``(frequency,)``.
    continuum_frequency_ranges : array-like
        ``[low, high]`` pairs (Hz) in the frame of ``frequency`` times
        ``doppler_factors``; they may overlap.
    doppler_factors : numpy.ndarray, optional
        Ratio of the frequency in the frame of the ranges to the frequency of
        the measurement set at every time, shape ``(time,)``. ``None``: the
        ranges are in the frame of the measurement set.

    Returns
    -------
    numpy.ndarray of bool
        Shape ``(time, frequency)``, or ``(1, frequency)`` without
        ``doppler_factors``: true where the channel frequency, converted at
        that time, lies in a range, edges included.
    """
    ranges = np.asarray(continuum_frequency_ranges, dtype=np.float64).reshape(-1, 2)
    ranges = ranges[np.argsort(ranges[:, 0], kind="stable")]
    lows, highs = ranges[:, 0], np.maximum.accumulate(ranges[:, 1])
    frequency = np.asarray(frequency, dtype=np.float64)
    converted = (
        frequency[np.newaxis, :]
        if doppler_factors is None
        else np.asarray(doppler_factors, dtype=np.float64)[:, np.newaxis]
        * frequency[np.newaxis, :]
    )
    # The last range starting at or below a frequency; the running maximum of
    # the upper edges covers the ranges it overlaps.
    index = np.searchsorted(lows, converted, side="right") - 1
    return (index >= 0) & (converted <= highs[np.maximum(index, 0)])


def subtract_polynomial_continuum(
    visibility: np.ndarray,
    flag: np.ndarray,
    weight: np.ndarray,
    frequency: np.ndarray,
    continuum_channels: np.ndarray,
    fit_order: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Subtract a polynomial fitted to the continuum channels of every spectrum.

    A sample is valid when it is unflagged, has a positive, finite weight and a
    finite visibility, and is a continuum channel at its time. Each spectrum
    (one time, baseline and polarization) with ``n`` valid samples has a
    polynomial of order ``min(fit_order, n - 1)`` in the channel frequency
    fitted to them by weighted least squares, the real and imaginary parts
    with the same weights, and subtracted from all of its channels. A spectrum
    without a valid sample is kept and flagged.

    The normal equations of all spectra come from one matrix product of the
    weights with the products of the basis polynomials, over the channels
    valid for some spectrum: Legendre polynomials of the frequency scaled to
    ``[-1, 1]`` over those channels, which span the polynomials of CASA's
    ``x = (f - f_mid) / (f_min - f_mid)``. A spectrum whose normal equations of
    order 2 or more are ill-conditioned (condition number above
    :data:`NORMAL_EQUATIONS_CONDITION_LIMIT`, valid samples clustered by
    flags) is fitted by least squares of its weighted samples instead. An
    output sample that overflows the dtype of ``visibility`` (a high order
    extrapolated far from the fitted channels) is flagged.

    Parameters
    ----------
    visibility : numpy.ndarray
        Complex visibilities, ``(time, baseline_id, frequency, polarization)``.
    flag : numpy.ndarray of bool
        Flags, same shape.
    weight : numpy.ndarray
        Weights, same shape.
    frequency : numpy.ndarray
        Channel frequencies (Hz), ``(frequency,)``.
    continuum_channels : numpy.ndarray of bool
        Continuum channels, ``(time, frequency)`` or ``(1, frequency)``
        (:func:`continuum_channel_mask`).
    fit_order : int
        Polynomial order of the fit.

    Returns
    -------
    subtracted : numpy.ndarray
        The visibilities less the polynomials, the dtype of ``visibility``.
    flag_out : numpy.ndarray of bool
        ``flag`` with the spectra without a valid sample and the overflowing
        samples flagged.
    statistics : dict
        Per polarization (arrays of shape ``(polarization,)``): ``n_spectra``,
        ``n_fitted``, ``n_order_reduced`` (fitted with an order below
        ``fit_order``), ``n_without_continuum`` (no valid sample),
        ``n_flagged_without_continuum`` (of those, the ones with an unflagged
        sample, flagged by the subtraction), and over the fitted spectra the
        sum, minimum and maximum of the chi-squared of the fits of the real
        and imaginary parts (``chi_squared_{sum,min,max}_{real,imag}``, the
        weighted sums of the squared residuals of the valid samples, which
        CASA computes for each fit); without a fitted spectrum the sums are 0
        and the minimum and maximum NaN.
    """
    from numpy.polynomial import legendre

    n_time, n_baseline, n_frequency, n_polarization = visibility.shape

    def rows(array):
        """(time, baseline, frequency, polarization) -> (spectrum, frequency)."""
        return np.ascontiguousarray(np.moveaxis(array, 2, -1)).reshape(-1, n_frequency)

    continuum_channels = np.asarray(continuum_channels, dtype=bool)
    flag_rows = rows(flag)
    continuum = np.broadcast_to(
        continuum_channels[:, np.newaxis, np.newaxis, :],
        (n_time, n_baseline, n_polarization, n_frequency),
    ).reshape(-1, n_frequency)
    weight_rows = rows(weight).astype(np.float64, copy=False)
    data = rows(visibility)
    valid = (
        continuum
        & ~flag_rows
        & (weight_rows > 0)
        & np.isfinite(weight_rows)
        & np.isfinite(data)
    )
    fit_weight = np.where(valid, weight_rows, 0.0)
    del weight_rows, continuum

    n_valid = valid.sum(axis=1)
    order = np.minimum(fit_order, n_valid - 1)  # -1: no valid sample
    # The terms some spectrum uses, whatever fit_order asks for.
    n_terms = max(int(order.max(initial=0)), 0) + 1
    frequency = np.asarray(frequency, dtype=np.float64)
    # The fit sees only the channels valid for some spectrum: the basis is
    # scaled to [-1, 1] over them and the normal equations are formed there.
    fitted_channels = valid.any(axis=0)
    columns = slice(None) if fitted_channels.all() else fitted_channels
    span = frequency[fitted_channels] if fitted_channels.any() else frequency
    low, high = float(span.min()), float(span.max())
    x = (
        (2.0 * frequency - (low + high)) / (high - low)
        if high > low
        else np.zeros(n_frequency)
    )
    basis = legendre.legvander(x, n_terms - 1)  # (frequency, n_terms)
    fit_basis = basis[columns]
    products = (fit_basis[:, :, np.newaxis] * fit_basis[:, np.newaxis, :]).reshape(
        -1, n_terms * n_terms
    )
    fit_weight_columns = fit_weight[:, columns]
    valid_columns = valid[:, columns]
    data_columns = data[:, columns]

    normal = _matmul(fit_weight_columns, products).reshape(-1, n_terms, n_terms)
    # Invalid samples may be NaN: zero them, as their zero weight would not.
    projection_real = _matmul(
        fit_weight_columns * np.where(valid_columns, data_columns.real, 0.0), fit_basis
    )
    projection_imag = _matmul(
        fit_weight_columns * np.where(valid_columns, data_columns.imag, 0.0), fit_basis
    )

    coefficients = np.zeros((data.shape[0], n_terms, 2))
    for effective_order in range(n_terms):
        selected = order == effective_order
        if not selected.any():
            continue
        k = effective_order + 1
        right_hand_side = np.stack(
            [projection_real[selected, :k], projection_imag[selected, :k]], axis=-1
        )
        coefficients[selected, :k] = _solve_normal_equations(
            normal[selected, :k, :k], right_hand_side
        )
        if k > 2:
            ill_conditioned = np.flatnonzero(selected)[
                _condition_numbers(normal[selected, :k, :k])
                > NORMAL_EQUATIONS_CONDITION_LIMIT
            ]
            if ill_conditioned.size:
                coefficients[ill_conditioned, :k] = _weighted_least_squares(
                    fit_basis[:, :k],
                    data_columns[ill_conditioned],
                    fit_weight_columns[ill_conditioned],
                    valid_columns[ill_conditioned],
                )
    del normal, projection_real, projection_imag
    del fit_weight_columns, valid_columns, data_columns

    model_real = _matmul(coefficients[:, :, 0], basis.T)
    residual_real = data.real - model_real
    del model_real
    chi_squared_real = np.einsum(
        "rf,rf->r", fit_weight, np.where(valid, residual_real, 0.0) ** 2
    )
    model_imag = _matmul(coefficients[:, :, 1], basis.T)
    residual_imag = data.imag - model_imag
    del model_imag
    chi_squared_imag = np.einsum(
        "rf,rf->r", fit_weight, np.where(valid, residual_imag, 0.0) ** 2
    )
    del fit_weight, valid

    fitted = order >= 0
    subtracted_rows = np.empty_like(data)
    with np.errstate(over="ignore", invalid="ignore"):
        subtracted_rows.real = residual_real
        subtracted_rows.imag = residual_imag
    del residual_real, residual_imag
    subtracted_rows[~fitted] = data[~fitted]
    # A high order extrapolated far from its channels can exceed the dtype.
    overflow = ~np.isfinite(subtracted_rows) & np.isfinite(data)
    flag_out_rows = flag_rows | ~fitted[:, np.newaxis] | overflow

    shape = (n_time, n_baseline, n_polarization, n_frequency)
    subtracted = np.ascontiguousarray(
        np.moveaxis(subtracted_rows.reshape(shape), -1, 2)
    )
    flag_out = np.ascontiguousarray(np.moveaxis(flag_out_rows.reshape(shape), -1, 2))

    statistics = _fit_statistics(
        shape,
        fitted,
        order,
        fit_order,
        flag_rows.all(axis=1),
        chi_squared_real,
        chi_squared_imag,
    )
    return subtracted, flag_out, statistics


@shares_param_docs
def uv_continuum_subtraction(
    ms_xdt: xr.DataTree,
    continuum_frequency_ranges,
    continuum_frequency_frame: str,
    fit_order: int = 1,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "continuum_subtracted",
    ms_data_group_out_modified: dict | None = None,
    overwrite: bool = False,
    *,
    doppler_factors: np.ndarray | None = None,
    input_checked: bool = False,
) -> dict:
    """Subtract the continuum of every visibility spectrum of a measurement set.

    The port of the pipeline's ``hif_uvcontsub`` (CASA ``uvcontsub``) on one
    measurement set: :func:`continuum_channel_mask` gives the continuum
    channels at every time and :func:`subtract_polynomial_continuum` fits and
    subtracts the polynomials. The continuum-subtracted visibilities and their
    flags are added to ``ms_xdt`` and registered as the output data group,
    which keeps the other roles of the input group (the weights are not
    changed).

    Parameters
    ----------
    ms_xdt : xarray.DataTree
        Measurement set (MSv4) with the dimensions ``(time, baseline_id,
        frequency, polarization)``, its frequencies in Hz. It is modified in
        place. Continuum ranges in another frame than the measurement set's
        need its ``antenna_xds`` and field and source dataset.
    continuum_frequency_ranges : list or numpy.ndarray
        The continuum ranges, ``[low, high]`` pairs of frequencies (Hz) in
        ``continuum_frequency_frame``; a channel is a continuum channel at the
        times its frequency, converted to that frame, lies in a range, edges
        included (the ``continuum_frequency_ranges`` of ``find_continuum_ultra``
        are the outer edges of its continuum channels).
    continuum_frequency_frame : str
        Spectral frame of ``continuum_frequency_ranges``, by its XRADIO
        observer name: the frame of the measurement set, or, for a ``"TOPO"``
        measurement set, ``"lsrk"``, ``"lsrd"`` or ``"BARY"``, to which the
        channel frequencies are converted at every time with the Doppler factor
        of ``ms_spectral_frame_conversion.doppler_factor`` (the telescope site
        at rest on the Earth, the phase centre of the first field).
    fit_order : int, default 1
        Order of the polynomial fitted to the continuum channels of every
        spectrum [CASA ``fitorder``]; lowered to ``n - 1`` for a spectrum with
        ``n <= fit_order`` valid continuum channels.
    ms_data_group_in_name : str, default "base"
        Measurement-set data group whose correlated data are continuum
        subtracted, with its flags and weights.
    ms_data_group_out_name : str, default "continuum_subtracted"
        Measurement-set data group of the continuum-subtracted visibilities.
    ms_data_group_out_modified : dict, optional
        The output variables of the roles ``correlated_data`` and ``flag``;
        ``None`` gives ``{"correlated_data": "VISIBILITY_CONTINUUM_SUBTRACTED",
        "flag": "FLAG_CONTINUUM_SUBTRACTED"}``. Neither may be a variable of
        another data group, the input group included.
    overwrite : bool, default False
        Replace an earlier output data group and its variables
        (:func:`check_output_data_group`).
    doppler_factors : numpy.ndarray, optional
        The Doppler factors of the times of ``ms_xdt`` from the frame of the
        measurement set to ``continuum_frequency_frame``
        (:func:`measurement_set_doppler_factors`), when the caller has them;
        computed from ``ms_xdt`` when ``None`` and the frames differ.
    input_checked : bool, default False
        Skip the configuration checks done by the calling layer. The checks of
        the measurement sets always run.

    Returns
    -------
    dict
        ``fit_statistics``, per polarization (see
        :func:`subtract_polynomial_continuum`), with the ``polarization``
        labels; ``n_continuum_channels_min`` and ``n_continuum_channels_max``,
        the fewest and most continuum channels of a time;
        ``continuum_channel_fraction``, the fraction of the channels of all
        times that are continuum channels; and the timing
        ``T_continuum_channel_mask`` and ``T_fit_and_subtract`` (seconds).

    Raises
    ------
    ValueError
        If the configuration fails :func:`check_continuum_fit`; the input
        data group is not in the measurement set or lacks the role
        ``correlated_data``, ``flag`` or ``weight``; the output data group or
        variables fail :func:`check_output_data_group`; the correlated data
        have other dimensions or the frequencies other units;
        ``doppler_factors`` does not have one value per time; or the frame of
        the ranges cannot be converted to the measurement set's.
    """
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    if input_checked:
        ranges = np.asarray(continuum_frequency_ranges, dtype=np.float64)
    else:
        ranges = check_continuum_fit(
            continuum_frequency_ranges,
            continuum_frequency_frame,
            fit_order,
            ms_data_group_out_modified,
        )
    out_modified = dict(
        ms_data_group_out_modified or DEFAULT_MS_DATA_GROUP_OUT_MODIFIED
    )
    data_groups = ms_xdt.attrs.get("data_groups", {})
    if ms_data_group_in_name not in data_groups:
        raise ValueError(
            f"Data group {ms_data_group_in_name!r} not in the measurement set "
            f"(data groups {sorted(data_groups)})."
        )
    data_group_in = data_groups[ms_data_group_in_name]
    for role in ("correlated_data", "flag", "weight"):
        if role not in data_group_in:
            raise ValueError(
                f"Data group {ms_data_group_in_name!r} has no {role!r} role."
            )
    check_output_data_group(
        data_groups,
        {name: variable.dims for name, variable in ms_xdt.ds.variables.items()},
        ms_data_group_in_name,
        ms_data_group_out_name,
        out_modified,
        overwrite,
    )
    _, data_group_out = create_data_groups_in_and_out(
        ms_xdt,
        data_group_in_name=ms_data_group_in_name,
        data_group_out_name=ms_data_group_out_name,
        data_group_out_modified=out_modified,
        overwrite=True,
    )
    visibility = ms_xdt[data_group_in["correlated_data"]]
    if visibility.dims != VISIBILITY_DIMS:
        raise ValueError(
            f"The correlated data must have the dimensions {VISIBILITY_DIMS}, not "
            f"{visibility.dims}."
        )
    units = ms_xdt.frequency.attrs.get("units")
    if units != "Hz":
        raise ValueError(f"The frequency axis must be in Hz, not {units!r}.")

    start = time.perf_counter()
    if doppler_factors is None:
        doppler_factors = measurement_set_doppler_factors(
            ms_xdt, continuum_frequency_frame
        )
    elif np.shape(doppler_factors) != (ms_xdt.sizes["time"],):
        raise ValueError(
            f"doppler_factors must have one value per time ({ms_xdt.sizes['time']}), "
            f"not the shape {np.shape(doppler_factors)}."
        )
    frequency = ms_xdt.frequency.values
    continuum_channels = continuum_channel_mask(frequency, ranges, doppler_factors)
    n_continuum = continuum_channels.sum(axis=1)
    timing = {"T_continuum_channel_mask": time.perf_counter() - start}

    start = time.perf_counter()
    subtracted, flag_out, statistics = subtract_polynomial_continuum(
        visibility.values,
        ms_xdt[data_group_in["flag"]].values,
        ms_xdt[data_group_in["weight"]].values,
        frequency,
        continuum_channels,
        fit_order,
    )
    flag = ms_xdt[data_group_in["flag"]]
    ms_xdt[data_group_out["correlated_data"]] = xr.DataArray(
        subtracted, dims=visibility.dims, attrs=dict(visibility.attrs)
    )
    ms_xdt[data_group_out["flag"]] = xr.DataArray(
        flag_out, dims=flag.dims, attrs=dict(flag.attrs)
    )
    timing["T_fit_and_subtract"] = time.perf_counter() - start

    modify_data_groups_xds(
        ms_xdt,
        data_group_out_name=ms_data_group_out_name,
        data_group_out=data_group_out,
        description=(
            f"Continuum subtracted from {ms_data_group_in_name}: polynomial of "
            f"order {fit_order} fitted to the channels in "
            f"{len(ranges)} continuum frequency ranges ({continuum_frequency_frame})."
        ),
    )
    return {
        "fit_statistics": {
            "polarization": [str(label) for label in ms_xdt.polarization.values],
            **statistics,
        },
        "n_continuum_channels_min": int(n_continuum.min()),
        "n_continuum_channels_max": int(n_continuum.max()),
        "continuum_channel_fraction": float(continuum_channels.mean()),
        **timing,
    }


def measurement_set_doppler_factors(
    ms_xdt: xr.DataTree, continuum_frequency_frame: str
) -> np.ndarray | None:
    """Ratio of the frequency in the frame of the continuum ranges to the measurement set's.

    Parameters
    ----------
    ms_xdt : xarray.DataTree
        Measurement set (MSv4).
    continuum_frequency_frame : str
        Spectral frame of the continuum ranges, an XRADIO observer name.

    Returns
    -------
    numpy.ndarray or None
        The Doppler factor at every time, shape ``(time,)``; ``None`` when the
        ranges are in the frame of the measurement set.

    Raises
    ------
    ValueError
        If the measurement set is not TOPO and the frames differ, or the frame
        of the ranges is not one of :data:`FRAMES_FROM_TOPO`.
    """
    ms_frame = ms_xdt.frequency.attrs.get("observer")
    if continuum_frequency_frame == ms_frame:
        return None
    if ms_frame != "TOPO" or continuum_frequency_frame not in FRAMES_FROM_TOPO:
        raise ValueError(
            f"Continuum ranges in {continuum_frequency_frame!r} cannot be converted "
            f"to the {ms_frame!r} frequencies of the measurement set: use the frame "
            f"of the measurement set, or one of {list(FRAMES_FROM_TOPO)} for a "
            "'TOPO' measurement set."
        )
    from astroviper.processing_functions.visibility_manipulation.ms_spectral_frame_conversion import (
        doppler_factor,
    )

    return doppler_factor(ms_xdt, continuum_frequency_frame)


def weighted_spectrum_sums(
    visibility: np.ndarray, flag: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sums over the times and baselines of the valid visibilities.

    Parameters
    ----------
    visibility, flag, weight : numpy.ndarray
        ``(time, baseline_id, frequency, polarization)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``sum(w V)`` (complex128) and ``sum(w)`` (float64) over the unflagged
        samples with a positive, finite weight and a finite visibility,
        ``(frequency, polarization)``; their ratio is the weighted mean
        spectrum.
    """
    weight = weight.astype(np.float64, copy=False)
    valid = ~flag & (weight > 0) & np.isfinite(weight) & np.isfinite(visibility)
    valid_weight = np.where(valid, weight, 0.0)
    return (
        np.einsum("tbfp,tbfp->fp", valid_weight, np.where(valid, visibility, 0.0)),
        valid_weight.sum(axis=(0, 1)),
    )


def check_output_data_group(
    data_groups: dict,
    variable_dims: dict,
    ms_data_group_in_name: str,
    ms_data_group_out_name: str,
    ms_data_group_out_modified: dict,
    overwrite: bool,
) -> None:
    """Check that the output data group and variables can be written.

    The output group is a new group, unless ``overwrite`` replaces an earlier
    output group of that name. An output variable may not be a variable of
    any other data group, the input group included, and replaces an existing
    variable only with ``overwrite`` and when that variable has the
    dimensions of the visibilities.

    Parameters
    ----------
    data_groups : dict
        The data groups of the measurement set.
    variable_dims : dict
        ``{name: dimensions}`` of the variables of the measurement set.
    ms_data_group_in_name, ms_data_group_out_name : str
        The input and output data groups.
    ms_data_group_out_modified : dict
        The output variables of the roles ``correlated_data`` and ``flag``.
    overwrite : bool
        Replace an earlier output data group and its variables.

    Raises
    ------
    ValueError
        If the output group is the input group, exists without ``overwrite``,
        or an output variable is a variable of another data group, exists
        without ``overwrite`` or does not have the dimensions of the
        visibilities.
    """
    if ms_data_group_out_name == ms_data_group_in_name:
        raise ValueError(
            f"The output data group {ms_data_group_out_name!r} is the input data group."
        )
    if not overwrite and ms_data_group_out_name in data_groups:
        raise ValueError(
            f"Data group {ms_data_group_out_name!r} exists in the measurement set; "
            "set overwrite=True to replace it."
        )
    for role, name in ms_data_group_out_modified.items():
        for group_name, group in data_groups.items():
            if group_name == ms_data_group_out_name:
                continue
            roles = {
                variable_role: variable
                for variable_role, variable in group.items()
                if variable_role not in ("date", "description")
            }
            if name in roles.values():
                raise ValueError(
                    f"The output {role} variable {name!r} is a variable of the data "
                    f"group {group_name!r}."
                )
        if name in variable_dims:
            if not overwrite:
                raise ValueError(
                    f"Output variable {name!r} exists in the measurement set; set "
                    "overwrite=True to replace it."
                )
            if tuple(variable_dims[name]) != VISIBILITY_DIMS:
                raise ValueError(
                    f"Output variable {name!r} exists with the dimensions "
                    f"{tuple(variable_dims[name])}, not those of the visibilities."
                )


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b``, one product at a time in the process."""
    with _LINEAR_ALGEBRA_LOCK:
        return a @ b


def _solve_normal_equations(normal: np.ndarray, right_hand_side: np.ndarray):
    """Solve stacked normal equations, by least squares where one is singular."""
    with _LINEAR_ALGEBRA_LOCK:
        try:
            return np.linalg.solve(normal, right_hand_side)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(normal) @ right_hand_side


def _condition_numbers(normal: np.ndarray) -> np.ndarray:
    """The condition numbers of stacked symmetric normal matrices (inf when
    not positive definite)."""
    with _LINEAR_ALGEBRA_LOCK:
        eigenvalues = np.linalg.eigvalsh(normal)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            eigenvalues[..., 0] > 0, eigenvalues[..., -1] / eigenvalues[..., 0], np.inf
        )


def _weighted_least_squares(
    design: np.ndarray, samples: np.ndarray, weights: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """The coefficients ``(spectrum, term, real/imaginary)`` of the weighted
    least-squares fits of the valid complex ``samples`` of every spectrum with
    the columns of ``design``."""
    coefficients = np.empty((samples.shape[0], design.shape[1], 2))
    with _LINEAR_ALGEBRA_LOCK:
        for row, use in enumerate(valid):
            root = np.sqrt(weights[row, use])
            right_hand_side = (
                np.stack([samples[row, use].real, samples[row, use].imag], axis=-1)
                * root[:, None]
            )
            coefficients[row] = np.linalg.lstsq(
                design[use] * root[:, None], right_hand_side, rcond=None
            )[0]
    return coefficients


def _fit_statistics(
    shape,
    fitted,
    order,
    fit_order,
    all_flagged,
    chi_squared_real,
    chi_squared_imag,
) -> dict:
    """Statistics of the fits per polarization (the last axis of ``shape``'s rows)."""
    n_polarization = shape[2]

    def per_polarization(values):
        return np.moveaxis(values.reshape(shape[:3]), 2, 0).reshape(n_polarization, -1)

    fitted_p = per_polarization(fitted)
    statistics = {
        "n_spectra": np.full(n_polarization, fitted_p.shape[1], dtype=np.int64),
        "n_fitted": fitted_p.sum(axis=1),
        "n_order_reduced": per_polarization(fitted & (order < fit_order)).sum(axis=1),
        "n_without_continuum": (~fitted_p).sum(axis=1),
        "n_flagged_without_continuum": per_polarization(~fitted & ~all_flagged).sum(
            axis=1
        ),
    }
    any_fitted = fitted_p.any(axis=1)
    for part, chi_squared in (("real", chi_squared_real), ("imag", chi_squared_imag)):
        values = per_polarization(chi_squared)
        statistics[f"chi_squared_sum_{part}"] = np.where(fitted_p, values, 0.0).sum(
            axis=1
        )
        for name, empty, reduce in (("min", np.inf, np.min), ("max", -np.inf, np.max)):
            extreme = reduce(np.where(fitted_p, values, empty), axis=1)
            statistics[f"chi_squared_{name}_{part}"] = np.where(
                any_fitted, extreme, np.nan
            )
    return statistics
