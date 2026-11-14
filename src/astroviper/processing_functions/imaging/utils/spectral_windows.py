"""Spectral windows of an image's frequency axis."""

import numpy as np


def linear_channel_ranges(
    frequency: np.ndarray, tolerance: float = 1e-3
) -> list[list[int]]:
    """Runs of channels on which the frequency axis is linear (spectral windows).

    A run that starts at channel ``s`` with the increment
    ``d = f[s + 1] - f[s]`` extends over the channels ``i`` with
    ``|f[i] - (f[s] + (i - s) d)| <= tolerance |d|``; the first channel that
    breaks this starts the next run. The last run may hold one channel.

    Parameters
    ----------
    frequency : array_like
        Channel frequencies.
    tolerance : float, default 1e-3
        Largest departure from the run's linear axis, in increments.

    Returns
    -------
    list of list of int
        ``[start, stop)`` of every run, in order.
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    ranges, start = [], 0
    while start < frequency.size:
        stop = start + 1
        if stop < frequency.size:
            increment = frequency[start + 1] - frequency[start]
            offsets = np.arange(frequency.size - start)
            linear = np.abs(
                frequency[start:] - (frequency[start] + offsets * increment)
            ) <= tolerance * abs(increment)
            stop = start + (linear.size if linear.all() else int(np.argmin(linear)))
        ranges.append([start, stop])
        start = stop
    return ranges
