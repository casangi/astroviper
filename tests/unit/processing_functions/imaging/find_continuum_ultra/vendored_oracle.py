"""The vendored findContinuum as the oracle of the find_continuum_ultra tests.

The tests import the vendored ``findContinuum.py`` (the pipeline's, as the
simple port ``find_continuum`` runs it) only through this module:
``findContinuum`` exposes its functions, :func:`vendored_block` compiles lines
of one of its functions as a function of that function's locals, and
:func:`quiet` silences what it prints and logs. ``random_spectrum``,
``random_entry`` and ``random_ranges`` generate cases. ``fC`` line numbers are
those of the pipeline's ``findContinuum.py`` (pipeline 2026.3.0.7); the vendored
file has other line numbers.
"""

import contextlib
import copy
import io
import logging

import numpy as np

from astroviper.distributed_applications.imaging.find_continuum import findContinuum

N_CHANNELS = [
    8,
    9,
    11,
    16,
    40,
    64,
    74,
    75,
    120,
    128,
    193,
    240,
    241,
    257,
    480,
    480,
    750,
    960,
    1001,
    1920,
    3840,
]
CHANNEL_WIDTHS = [
    976562.5,
    488281.25,
    244140.625,
    7.8125e6,
    15.625e6,
    31.25e6,
    1.953125e6,
]


def vendored_block(first_line, first_text, last_line, last_text, parameter_names):
    """fC lines ``first_line`` to ``last_line`` of the vendored findContinuum as
    a function.

    The block is found where the stripped texts of its first and last lines are
    ``first_text`` and ``last_text``, at the smallest offset from the fC line
    numbers. It runs in a copy of the vendored module's globals (its
    ``__globals__``), with its line numbers in the vendored file.

    Parameters
    ----------
    first_line, last_line : int
        fC line numbers of the block's first and last lines.
    first_text, last_text : str
        Their stripped texts.
    parameter_names : sequence of str
        The locals the block reads.

    Returns
    -------
    callable
        Takes ``parameter_names`` and returns the block's locals as a dict
        (or what a ``return`` of the block returns).

    Raises
    ------
    AssertionError
        If the vendored file has no such block.
    """
    with open(findContinuum.__file__) as stream:
        lines = stream.read().split("\n")
    offsets = sorted(range(1 - first_line, len(lines) - last_line + 1), key=abs)
    offset = next(
        (
            offset
            for offset in offsets
            if lines[first_line + offset - 1].strip() == first_text
            and lines[last_line + offset - 1].strip() == last_text
        ),
        None,
    )
    assert offset is not None, (
        f"fC {first_line} {first_text!r} to {last_line} {last_text!r} is not in "
        f"{findContinuum.__file__}"
    )
    first, last = first_line + offset, last_line + offset
    block = lines[first - 1 : last]
    indent = block[0][: len(block[0]) - len(block[0].lstrip())]
    body = "\n".join(
        "    " + line[len(indent) :] if line.startswith(indent) else line
        for line in block
    )
    source = (
        f"def _block({', '.join(parameter_names)}):\n{body}\n"
        "    return dict(locals())\n"
    )
    namespace = dict(vars(findContinuum))
    exec(
        compile("\n" * (first - 2) + source, findContinuum.__file__, "exec"), namespace
    )
    return namespace["_block"]


@contextlib.contextmanager
def quiet():
    """Silence what findContinuum prints and logs."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def random_spectrum(rng, n):
    """A random spectrum of ``n`` channels of one of 13 kinds.

    Parameters
    ----------
    rng : numpy.random.Generator
    n : int

    Returns
    -------
    numpy.ndarray
        float64 ``(n,)``.
    """
    kind = rng.integers(0, 13)
    x = np.arange(n, dtype=np.float64)
    if kind == 9 and n > 4:  # two levels: every channel is edge-valued (fC:6672)
        s = np.full(n, rng.normal())
        s[int(rng.integers(1, n - 1)) :] = rng.normal()
        return s
    if kind in (10, 11, 12):
        noise = rng.normal(0, 1, n) * 10 ** rng.uniform(-4, -2)
        sigma = np.std(noise)
        if (
            kind == 10
        ):  # narrow strong lines on a strong continuum (lineSNR, fC:6875-6902)
            s = noise + rng.uniform(1000, 5000) * sigma
            n_lines, amplitudes, widths = int(rng.integers(3, 9)), (50, 500), (0.5, 2.0)
        elif (
            kind == 11
        ):  # many narrow lines on a moderate continuum (trim 13, fC:6981-6990)
            s = noise + rng.uniform(130, 250) * sigma
            n_lines, amplitudes, widths = (
                int(rng.integers(20, 80)),
                (20, 200),
                (0.5, 2.0),
            )
        else:  # absorption-dominated (high baseline, fC:6714-6738, 6915-6917)
            s = noise + rng.uniform(5, 50) * sigma
            n_lines, amplitudes, widths = (
                int(rng.integers(3, 30)),
                (-200, -5),
                (0.5, max(0.6, n / 20)),
            )
        for _ in range(n_lines):
            centre = rng.uniform(0, n)
            width = rng.uniform(*widths)
            s += (
                rng.uniform(*amplitudes)
                * sigma
                * np.exp(-0.5 * ((x - centre) / width) ** 2)
            )
        return s
    noise = rng.normal(0, 1, n)
    s = noise * 10 ** rng.uniform(-4, -1)
    sigma = np.std(s) if n > 1 else 1.0
    if rng.random() < 0.6:
        s += rng.uniform(-3, 30) * sigma  # continuum offset
    n_lines = int(rng.choice([0, 1, 2, 3, 5, 8, 12, 20, 40])) if kind != 0 else 0
    for _ in range(n_lines):
        centre = rng.uniform(0, n)
        width = rng.uniform(0.3, max(1.0, n / rng.choice([5, 20, 80])))
        amplitude = rng.uniform(1, 200) * sigma
        if kind in (2, 3) and rng.random() < 0.6:
            amplitude = -amplitude  # absorption
        s += amplitude * np.exp(-0.5 * ((x - centre) / width) ** 2)
    if kind == 4:  # single-channel spikes
        for c in rng.integers(0, n, rng.integers(2, 8)):
            s[c] += rng.uniform(5, 40) * sigma
    if kind == 5 and n > 4:  # constant edge blocks (flagged edges)
        k = int(rng.integers(1, max(2, n // 6)))
        s[:k] = s[k]
        s[-k:] = s[-k - 1] if rng.random() < 0.5 else s.min()
    if kind == 6 and n > 4:  # zero blocks
        k = int(rng.integers(1, max(2, n // 3)))
        start = int(rng.integers(0, n - k))
        s[start : start + k] = 0.0
    if kind == 7:  # slope and a broad line (smooth, rich spectrum)
        s += rng.uniform(-50, 50) * sigma * x / n + rng.uniform(100, 1000) * sigma
    if kind == 8 and n > 4:  # repeated minimum at an end
        k = int(rng.integers(1, 4))
        if rng.random() < 0.5:
            s[:k] = s.min()
        else:
            s[-k:] = s.min()
    if rng.random() < 0.05:
        s = np.round(s, int(rng.integers(3, 6)))  # ties
    return s


def pick(rng, options):
    """One element of a Python list, keeping its Python type."""
    return options[int(rng.integers(0, len(options)))]


def random_entry(rng, template):
    """runFindContinuum's locals of ``template`` with a random spectrum, axis
    and round (the inputs of fC 5228-5862).

    Parameters
    ----------
    rng : numpy.random.Generator
    template : dict
        runFindContinuum's locals at fC 5228, by name.

    Returns
    -------
    dict
    """
    entry = copy.deepcopy(template)
    n = pick(rng, N_CHANNELS)
    s = random_spectrum(rng, n)
    width = pick(rng, CHANNEL_WIDTHS)
    first = pick(rng, [86e9, 230e9, 345e9, 650e9])
    entry.update(
        avgSpectrumNansReplaced=s,
        nchan=n,
        channelWidth=np.float64(width),
        firstFreq=np.float64(first),
        lastFreq=np.float64(first + (n - 1) * width),
        nBaselineChannels=max(2, int(np.floor(0.19 * n + 0.5))),
        sigmaFindContinuum=pick(rng, ["auto", "auto", "auto", 2.5, 4.2, 7.0, 8.0, -1]),
        sigmaFindContinuumMode=pick(rng, ["auto", "auto", "autolower"]),
        signalRatioTier1=pick(rng, [0.967, 1.0]),
        amendMaskIterationName=pick(
            rng,
            [
                ".original",
                ".original",
                "",
                ".amendedMask",
                ".extraMask",
                ".onlyExtraMask",
                ".autoLower",
                ".autoLower2",
            ],
        ),
        momDiffSNR=np.float64(pick(rng, [-1, 5.0, 14.9, 15.0, 30.0])),
        numberPixelsInMom8Mask=pick(rng, [0, 0, 3, 4, 100]),
        mom0peak=np.float64(rng.uniform(0, 10)),
        mom8peak=np.float64(rng.uniform(0, 0.2)),
        singleContinuum=bool(rng.random() < 0.15),
        idxNonZeroMAD=None,
        amendMask=bool(rng.random() < 0.9),
        maxMadRatioForSFCAdjustment=1.18,
        narrow="auto",
        trimChannels="auto",
        maxTrim=20,
        RangesDropped=0,
        nanmin=None if rng.random() < 0.7 else 0.0,
    )
    return entry


def random_ranges(rng, n_channels, max_ranges=6):
    """Up to ``max_ranges`` sorted ``[start, stop)`` ranges of random channels.

    Parameters
    ----------
    rng : random.Random
    n_channels : int
    max_ranges : int, default 6

    Returns
    -------
    list of list of int
    """
    chosen = sorted(rng.sample(range(n_channels), rng.randint(1, min(n_channels, 40))))
    ranges = []
    for channel in chosen:
        if ranges and channel == ranges[-1][1]:
            ranges[-1][1] += 1
        else:
            ranges.append([channel, channel + 1])
    return ranges[: rng.randint(1, max_ranges)]
