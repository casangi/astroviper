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


def window_rest_frequency(
    frequency: np.ndarray, channel_range: list, rest_frequency: float
) -> float:
    """The rest frequency of a spectral window of a cube, of the radio
    velocity that weights findContinuum's moment 0.

    findContinuum scales moment 0 by the window's mean frequency (fC 1757 of
    the pipeline's ``findContinuum.py``), so its moment difference keeps
    ``1 - mean_frequency / rest_frequency`` of the continuum. A cube of one
    spectral window keeps its rest frequency. A cube of several has one, the
    middle channel of all of them (``xradio.image.make_empty_sky_image``),
    outside most windows; a window then takes its own middle channel, the rest
    frequency of a cube of that window alone.

    Parameters
    ----------
    frequency : array_like
        Frequencies of the cube's channels (Hz).
    channel_range : sequence of int
        ``[start, stop)``, the channels of the window in the cube.
    rest_frequency : float
        Rest frequency of the cube (Hz).

    Returns
    -------
    float
        The rest frequency of the window (Hz).
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    if len(linear_channel_ranges(frequency)) == 1:
        return float(rest_frequency)
    start, stop = channel_range
    return float(frequency[start + (stop - start) // 2])
