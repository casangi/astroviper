"""Shared NumPy statistics for local planes and distributed image summaries.

The last axis contains samples; leading axes identify independent output groups.
All value calculations use float64, exclude NaNs, and leave caller arrays intact.
Counts remain integers internally. Public value results are float64, with NaN
for empty groups except for a zero pixel count. No xarray, storage, or graph
objects are used here; the wrappers own coordinates, masks, and sample retention.
"""

import warnings

import numpy as np

MEAN_NAME = "__statistics_mean__"
M2_NAME = "__statistics_m2__"
STATE_NAMES = ("min", "max", "sum", "sumsq", "n_pixels", MEAN_NAME, M2_NAME)


def summarize_samples(samples, *, assume_valid=False):
    """Build compact summaries over the final sample axis.

    Parameters
    ----------
    samples : numpy.ndarray
        Real samples with arbitrary leading group dimensions.
    assume_valid : bool, default False
        Skip NaN detection for callers that already removed invalid samples.

    Returns
    -------
    dict of str to numpy.ndarray or numpy scalar
        Mergeable moments plus first minimum/maximum sample indices. Indices
        are local to the sample axis; empty groups use -1.
    """
    values = np.asarray(samples, dtype=np.float64)
    shape = values.shape[:-1]
    size = values.shape[-1]
    if size == 0:
        return {
            "min": np.full(shape, np.nan),
            "max": np.full(shape, np.nan),
            "sum": np.zeros(shape),
            "sumsq": np.zeros(shape),
            "n_pixels": np.zeros(shape, dtype=np.int64),
            MEAN_NAME: np.full(shape, np.nan),
            M2_NAME: np.zeros(shape),
            "min_index": np.full(shape, -1, dtype=np.int64),
            "max_index": np.full(shape, -1, dtype=np.int64),
        }
    valid = None if assume_valid else ~np.isnan(values)
    if valid is None or valid.all():
        count = np.full(shape, size, dtype=np.int64)
        total = values.sum(axis=-1)
        mean = total / count
        deviations = values - np.expand_dims(mean, -1)
        return {
            "min": values.min(axis=-1),
            "max": values.max(axis=-1),
            "sum": total,
            "sumsq": (values * values).sum(axis=-1),
            "n_pixels": count,
            MEAN_NAME: mean,
            M2_NAME: (deviations * deviations).sum(axis=-1),
            "min_index": values.argmin(axis=-1),
            "max_index": values.argmax(axis=-1),
        }
    count = valid.sum(axis=-1, dtype=np.int64)
    filled = np.where(valid, values, 0.0)
    total = filled.sum(axis=-1)
    mean = np.divide(total, count, out=np.full(shape, np.nan), where=count > 0)
    deviations = np.where(valid, values - np.expand_dims(mean, -1), 0.0)
    minimum = np.where(valid, values, np.inf).min(axis=-1)
    maximum = np.where(valid, values, -np.inf).max(axis=-1)
    # Match against valid samples so a NaN sentinel cannot win an infinity tie.
    min_index = (valid & (values == np.expand_dims(minimum, -1))).argmax(axis=-1)
    max_index = (valid & (values == np.expand_dims(maximum, -1))).argmax(axis=-1)
    return {
        "min": np.where(count > 0, minimum, np.nan),
        "max": np.where(count > 0, maximum, np.nan),
        "sum": total,
        "sumsq": (filled * filled).sum(axis=-1),
        "n_pixels": count,
        MEAN_NAME: mean,
        M2_NAME: (deviations * deviations).sum(axis=-1),
        "min_index": np.where(count > 0, min_index, -1),
        "max_index": np.where(count > 0, max_index, -1),
    }


def merge_summaries(summaries):
    """Combine aligned compact summaries with Chan's stable variance formula.

    Parameters
    ----------
    summaries : sequence of dict
        At least one summary, with identically ordered output groups. Positions
        and median samples are combined separately by the coordinate wrapper.

    Returns
    -------
    dict
        Combined moments. Empty groups contribute zero count and moments.
    """
    result = {name: np.asarray(summaries[0][name]).copy() for name in STATE_NAMES}
    for other in summaries[1:]:
        count = result["n_pixels"]
        other_count = other["n_pixels"]
        total = count + other_count
        mean = np.where(count > 0, result[MEAN_NAME], 0.0)
        other_mean = np.where(other_count > 0, other[MEAN_NAME], 0.0)
        delta = np.where((count > 0) & (other_count > 0), other_mean - mean, 0.0)
        weight = other_count / np.where(total > 0, total, 1)
        combined_mean = np.where(count == 0, other_mean, mean + delta * weight)
        result[M2_NAME] = (
            np.where(count > 0, result[M2_NAME], 0.0)
            + np.where(other_count > 0, other[M2_NAME], 0.0)
            + delta * delta * (count * weight)
        )
        result[MEAN_NAME] = np.where(total > 0, combined_mean, np.nan)
        result["n_pixels"] = total
        result["min"] = np.fmin(result["min"], other["min"])
        result["max"] = np.fmax(result["max"], other["max"])
        result["sum"] = result["sum"] + other["sum"]
        result["sumsq"] = result["sumsq"] + other["sumsq"]
    return result


def _median_samples(values):
    """Ignore NaNs while treating empty/all-NaN groups as ordinary empty outputs."""
    if values.shape[-1] == 0:
        return np.full(values.shape[:-1], np.nan)
    if not np.isnan(values).any():
        return np.median(values, axis=-1)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        return np.nanmedian(values, axis=-1)


def finalize_summary(summary, statistics, *, samples=None, minimum_first=None):
    """Evaluate requested statistics once, reusing the median for scaled MAD.

    Parameters
    ----------
    summary : dict
        Compact moments from a local or merged summary.
    statistics : sequence of str
        Value statistics requested by the wrapper, in output order.
    samples : numpy.ndarray, optional
        Required for median/MAD; final axis holds all samples in each group.
    minimum_first : numpy.ndarray of bool, optional
        Whether the minimum precedes the maximum in absolute pixel order.
        Defaults to comparing local summary indices for plane calculations.

    Returns
    -------
    dict
        Float64 values: population std, RMS, Gaussian-scaled MAD (1.4826),
        and signed peak with first-pixel tie breaking.
    """
    count = np.asarray(summary["n_pixels"])
    nonempty = count > 0
    denominator = np.where(nonempty, count, 1)
    median = None
    if {"median", "mad_sigma"}.intersection(statistics):
        if samples is None:
            raise ValueError("Median statistics require retained samples")
        values = np.asarray(samples, dtype=np.float64)
        median = _median_samples(values)
    result = {}
    for name in statistics:
        if name in ("min", "max", "sum", "sumsq"):
            value = summary[name]
        elif name == "n_pixels":
            result[name] = count.astype(np.float64)
            continue
        elif name == "mean":
            value = summary["sum"] / denominator
        elif name == "rms":
            value = np.sqrt(summary["sumsq"] / denominator)
        elif name == "std":
            value = np.sqrt(summary[M2_NAME] / denominator)
        elif name == "peak":
            if minimum_first is None:
                minimum_first = summary["min_index"] < summary["max_index"]
            minimum, maximum = summary["min"], summary["max"]
            choose_minimum = (np.abs(minimum) > np.abs(maximum)) | (
                (np.abs(minimum) == np.abs(maximum)) & minimum_first
            )
            value = np.where(choose_minimum, minimum, maximum)
        elif name == "median":
            value = median
        elif name == "mad_sigma":
            deviations = np.abs(values - np.expand_dims(median, -1))
            value = 1.4826 * _median_samples(deviations)
        else:
            raise ValueError(f"Unknown statistic: {name!r}")
        result[name] = np.asarray(np.where(nonempty, value, np.nan), dtype=np.float64)
    return result
