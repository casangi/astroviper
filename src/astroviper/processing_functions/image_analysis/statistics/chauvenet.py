"""Chauvenet's criterion for outliers [casacore ``ChauvenetCriterionStatistics``].

The clipping of CASA ``imstat`` and ``ia.statistics`` with
``algorithm="chauvenet"`` and ``maxiter=5``.
"""

import math

import numpy as np
import scipy.special

CHAUVENET_MAX_ITERATIONS = 5


def chauvenet_sample(values):
    """``values`` without the outliers of Chauvenet's criterion.

    As casacore's ``ChauvenetCriterionStatistics`` with five iterations: up
    to six passes each remove the values farther from the mean than ``z``
    standard deviations (``ddof=1``), where ``n * erfc(z / sqrt(2)) = 0.5``
    for the ``n`` values the pass starts from. The passes stop when one
    removes nothing or fewer than two values are left.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional finite values. The arithmetic is done in their dtype.

    Returns
    -------
    numpy.ndarray
        The values kept, in their order; ``values`` itself when none is
        removed.

    Notes
    -----
    casacore's ``z`` is any value in ``[z(n), z(n + 1))``, found by bisection
    from a table, and it applies each range to all the values; here ``z`` is
    ``z(n)`` and each pass filters the values the previous one kept. The
    results differ when a value lies between the two ``z`` ranges, or when a
    later casacore range reaches a value an earlier pass removed. A value
    between the ``z`` ranges changes the later passes too: in samples of tens
    of values the kept values can differ by several and the RMS by tens of
    percent, in samples of 1000 values by one and less than 1 percent.
    """
    for _ in range(CHAUVENET_MAX_ITERATIONS + 1):
        if values.size < 2:
            break
        mean, deviation = values.mean(), values.std(ddof=1)
        z_score = math.sqrt(2.0) * float(scipy.special.erfcinv(0.5 / values.size))
        kept = values[np.abs(values - mean) <= z_score * deviation]
        if kept.size == values.size:
            break
        values = kept
    return values
