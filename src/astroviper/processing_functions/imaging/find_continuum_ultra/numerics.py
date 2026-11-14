"""findContinuum's arithmetic that the steps of find_continuum_ultra share.

Levels rounded as findContinuum writes them as text, median absolute
deviations, Chauvenet statistics, the one-event sigma, runs and ranges of
channels, and the line fit, each computed as findContinuum computes it. ``fC``
line numbers are those of the pipeline's ``findContinuum.py`` (pipeline
2026.3.0.7).
"""

import decimal

import numpy as np
import scipy.optimize
import scipy.special

from astroviper.processing_functions.image_analysis.make_auto_mask import (
    _chauvenet_sample,
)

# The divisor of findContinuum's scaled median absolute deviation (fC 8591).
MAD_DIVISOR = 0.6745


def round_half_up(value: float) -> float:
    """The integer nearest to the binary value of ``value``, ties away from
    zero [findContinuum ``round_half_up``] (fC 398-399).

    Parameters
    ----------
    value : float
        Value to round.

    Returns
    -------
    float
        The rounded value.
    """
    return float(
        decimal.Decimal(float(value)).to_integral_value(rounding=decimal.ROUND_HALF_UP)
    )


def round_to_decimals(value: float, n_decimals: int) -> float:
    """``value`` written with ``n_decimals`` decimals and read back, as
    findContinuum writes levels with ``%f`` (6) and ``%.9f`` (9).

    Parameters
    ----------
    value : float
        Value to round.
    n_decimals : int
        Decimals written.

    Returns
    -------
    float
        The value of the text of ``value`` with ``n_decimals`` decimals.
    """
    return float(f"{value:.{n_decimals}f}")


def round_to_significant_digits(value: float, n_digits: int) -> float:
    """``value`` written with ``n_digits`` significant digits and read back,
    as findContinuum writes levels with ``%g`` (fC 8503-8505).

    Parameters
    ----------
    value : float
        Value to round.
    n_digits : int
        Significant digits written.

    Returns
    -------
    float
        The value of the text of ``value`` with ``n_digits`` significant
        digits.
    """
    return float(f"{value:.{n_digits}g}")


def median_absolute_deviation(values: np.ndarray) -> float:
    """Median absolute deviation of the finite values, unscaled
    (fC 7889-7899).

    Parameters
    ----------
    values : numpy.ndarray
        Values of any shape.

    Returns
    -------
    float
        ``median(|v - median(v)|)`` over the finite values ``v``.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(np.abs(values - np.median(values))))


def scaled_median_absolute_deviation(values: np.ndarray) -> float:
    """Median absolute deviation of the values that are not NaN, divided by
    ``MAD_DIVISOR`` [findContinuum ``MAD``] (fC 8591-8604).

    Parameters
    ----------
    values : numpy.ndarray
        Values of any shape.

    Returns
    -------
    float
        ``median(|v - median(v)|) / 0.6745`` over the values ``v`` that are not
        NaN; NaN if there are none.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(np.abs(values - np.median(values)))) / MAD_DIVISOR


def chauvenet_statistics(values: np.ndarray) -> tuple[float, float, float]:
    """Median, median absolute deviation and standard deviation of the values
    that Chauvenet's criterion keeps [CASA ``imstat(algorithm="chauvenet",
    maxiter=5)``].

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional values, gathered in C order of the map: the clipping
        means and standard deviations depend on the order.

    Returns
    -------
    median : float
    median_absolute_deviation : float
        Unscaled.
    standard_deviation : float
        With ``ddof=1``; 0.0 for one kept value.
    """
    sample = _chauvenet_sample(np.asarray(values, dtype=np.float64))
    median = np.median(sample)
    return (
        float(median),
        float(np.median(np.abs(sample - median))),
        float(sample.std(ddof=1)) if sample.size > 1 else 0.0,
    )


def one_event_sigma(n_values: float, n_events: float) -> float:
    """The sigma above which a Gaussian population of ``n_values`` values has
    ``n_events`` positive events [findContinuum ``oneEvent``] (fC 8335-8338).

    Parameters
    ----------
    n_values : float
        Size of the population.
    n_events : float
        Events allowed.

    Returns
    -------
    float
        ``sqrt(2) erfinv(1 - 2 n_events / n_values)``, computed in
        findContinuum's order.
    """
    odds = n_events / float(n_values)
    odds *= 2
    return float((2**0.5) * scipy.special.erfinv(1 - odds))


def contiguous_runs(channels: np.ndarray) -> list[np.ndarray]:
    """The runs of consecutive channels, in the order given
    [findContinuum ``splitListIntoContiguousLists``] (fC 8606-8624).

    Parameters
    ----------
    channels : numpy.ndarray
        One-dimensional integer channels.

    Returns
    -------
    list of numpy.ndarray
        ``channels`` split where a channel is not the previous one plus one;
        no run for no channel.
    """
    channels = np.asarray(channels)
    if channels.size == 0:
        return []
    return np.split(channels, np.flatnonzero(np.diff(channels) != 1) + 1)


def channels_of_ranges(ranges: list) -> np.ndarray:
    """The channels of channel ranges, in list order and with duplicates
    [findContinuum ``convertSelectionIntoChannelList``] (fC 8751-8767).

    Parameters
    ----------
    ranges : list
        ``[start, stop)`` pairs.

    Returns
    -------
    numpy.ndarray
        int64 channels.
    """
    return np.array(
        [channel for start, stop in ranges for channel in range(start, stop)],
        dtype=np.int64,
    )


def ranges_of_channels(channels: np.ndarray) -> list[list[int]]:
    """The channel ranges of the runs of consecutive channels.

    Parameters
    ----------
    channels : numpy.ndarray
        One-dimensional integer channels.

    Returns
    -------
    list of list of int
        ``[start, stop)`` of every run of :func:`contiguous_runs`.
    """
    return [[int(run[0]), int(run[-1]) + 1] for run in contiguous_runs(channels)]


def range_count(ranges: list) -> int:
    """The number of channel ranges as findContinuum counts them: 1 for no
    range (fC 2000, 7008).

    Parameters
    ----------
    ranges : list
        ``[start, stop)`` pairs.

    Returns
    -------
    int
        ``max(len(ranges), 1)``.
    """
    return max(len(ranges), 1)


def line_fit(x: np.ndarray, y: np.ndarray, y_error: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of findContinuum's line fit [findContinuum
    ``linfit``] (fC 9366-9391).

    ``scipy.optimize.leastsq`` from ``[0, 0]`` on findContinuum's residuals
    ``(y - (intercept + slope x)) / (y_error / y)**2``.

    Parameters
    ----------
    x, y : numpy.ndarray
        Abscissae and values.
    y_error : numpy.ndarray
        Uncertainties of ``y``, or one for every value.

    Returns
    -------
    slope, intercept : float
    """
    x, y = np.array(x), np.array(y)
    if type(y_error) not in (list, np.ndarray):
        y_error = np.ones(len(x)) * y_error
    fit = scipy.optimize.leastsq(
        lambda p, x, y, error: (y - (p[1] + p[0] * x)) / (error**2),
        [0, 0],
        args=(x, y, y_error / y),
        full_output=1,
    )
    slope, intercept = fit[0]
    return float(slope), float(intercept)
