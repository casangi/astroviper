"""Cube reductions of the continuum channel finder (``find_continuum``).

Every reduction of the dirty cube that the pipeline's findContinuum makes with
CASA (``immoments``, ``imstat`` and ``image.getprofile``) runs on one chunk of
channels at a time: moment images merge partial accumulators, per-channel
statistics and spectra gather one value per channel, and whole-cube statistics
merge their sums and extrema. Masks are CASA lattice expressions over the cube,
its primary beam and two-dimensional images, evaluated in CASA axis order
``(l, m, polarization, frequency)``.

``find_continuum`` and ``find_continuum_fused`` reduce whole chunks
(:func:`find_continuum_cube_reductions`). ``find_continuum_ultra`` streams
batches of planes (:func:`find_continuum_batch_reductions`) with masks split
into a two-dimensional part and cube terms (:func:`split_lattice_conjuncts`)
and the primary-beam masks of the first round (:func:`primary_beam_masks`).
"""

import re
import time

import numpy as np

from astroviper.processing_functions.image_analysis.make_auto_mask import (
    _chauvenet_sample,
)

SPEED_OF_LIGHT_KM_S = 299792.458
# casacore's factor from the median absolute deviation to a Gaussian sigma.
MAD_TO_SIGMA = 1.482602218505602
CASA_MOMENTS = {-1: "mean", 0: "integrated", 8: "maximum", 10: "minimum"}
CHANNEL_STATISTICS = (
    "npts",
    "sum",
    "sumsq",
    "mean",
    "sigma",
    "rms",
    "min",
    "max",
    "median",
    "medabsdevmed",
    "q1",
    "q3",
    "quartile",
)
MERGEABLE_STATISTICS = ("npts", "sum", "sumsq", "min", "max")
# The primary-beam level of findContinuum's spectrum mask ("pb">0.21).
PROFILE_LEVEL = 0.21
_CUBE_ROLES = ("sky", "primary_beam")
# Rows of the blocks that planes are reduced in; whole-cube sums depend on it.
_BLOCK_ROWS = 625

_TOKEN = re.compile(
    r"\s*(?:(?P<number>\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)"
    r"|(?P<name>\"[^\"]*\"|'[^']*')"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<op>&&|\|\||==|!=|>=|<=|[-+*/()<>,!]))"
)


def _token_matches(expression):
    position = 0
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if match is None or match.end() == position:
            raise ValueError(f"Cannot parse lattice expression {expression!r}")
        yield match
        position = match.end()


def _tokenize(expression):
    tokens = []
    for match in _token_matches(expression.strip()):
        kind = match.lastgroup
        value = match.group(kind)
        tokens.append((kind, value[1:-1] if kind == "name" else value))
    return tokens


class _ExpressionParser:
    """Recursive-descent evaluator of the CASA lattice expressions findContinuum writes."""

    def __init__(self, tokens, arrays):
        self.tokens = tokens
        self.position = 0
        self.arrays = arrays

    def peek(self):
        return (
            self.tokens[self.position]
            if self.position < len(self.tokens)
            else (None, None)
        )

    def take(self, value=None):
        token = self.peek()
        if value is not None and token[1] != value:
            raise ValueError(
                f"Expected {value!r} in lattice expression, got {token[1]!r}"
            )
        self.position += 1
        return token

    def parse(self):
        result = self.logical_or()
        if self.position != len(self.tokens):
            raise ValueError(f"Unexpected {self.peek()[1]!r} in lattice expression")
        return result

    def logical_or(self):
        result = self.logical_and()
        while self.peek()[1] == "||":
            self.take()
            result = np.logical_or(result, self.logical_and())
        return result

    def logical_and(self):
        result = self.comparison()
        while self.peek()[1] == "&&":
            self.take()
            result = np.logical_and(result, self.comparison())
        return result

    def comparison(self):
        left = self.additive()
        operator = self.peek()[1]
        if operator not in ("==", "!=", ">=", "<=", ">", "<"):
            return left
        self.take()
        right = self.additive()
        compare = {
            "==": np.equal,
            "!=": np.not_equal,
            ">=": np.greater_equal,
            "<=": np.less_equal,
            ">": np.greater,
            "<": np.less,
        }[operator]
        with np.errstate(invalid="ignore"):
            result = compare(left, right)
        # A masked (NaN) pixel fails every comparison, as in CASA.
        return result & ~(np.isnan(left) | np.isnan(right))

    def additive(self):
        result = self.multiplicative()
        while self.peek()[1] in ("+", "-"):
            operator = self.take()[1]
            other = self.multiplicative()
            result = result + other if operator == "+" else result - other
        return result

    def multiplicative(self):
        result = self.unary()
        while self.peek()[1] in ("*", "/"):
            operator = self.take()[1]
            other = self.unary()
            result = result * other if operator == "*" else result / other
        return result

    def unary(self):
        operator = self.peek()[1]
        if operator == "-":
            self.take()
            return -self.unary()
        if operator == "!":
            self.take()
            return np.logical_not(self.unary())
        return self.primary()

    def primary(self):
        kind, value = self.take()
        if kind == "number":
            return float(value)
        if kind == "name":
            return self.lookup(value)
        if kind == "ident":
            if self.peek()[1] == "(":
                self.take("(")
                arguments = [self.logical_or()]
                while self.peek()[1] == ",":
                    self.take()
                    arguments.append(self.logical_or())
                self.take(")")
                if value.lower() != "iif" or len(arguments) != 3:
                    raise ValueError(
                        f"Lattice expression function {value!r} is not supported"
                    )
                return np.where(arguments[0], arguments[1], arguments[2])
            if value in ("T", "True"):
                return True
            if value in ("F", "False"):
                return False
            return self.lookup(value)
        if value == "(":
            result = self.logical_or()
            self.take(")")
            return result
        raise ValueError(f"Unexpected {value!r} in lattice expression")

    def lookup(self, name):
        key = name.rstrip("/")
        if key not in self.arrays:
            raise KeyError(f"Lattice expression refers to unknown image {name!r}")
        return self.arrays[key]


def lattice_expression_names(expression):
    """Image names a CASA lattice expression refers to (quoted names only).

    Parameters
    ----------
    expression : str
        Lattice expression.

    Returns
    -------
    list of str
        The quoted image names, without trailing slashes, in order of first
        appearance.
    """
    names = []
    for kind, value in _tokenize(expression):
        if kind == "name" and value.rstrip("/") not in names:
            names.append(value.rstrip("/"))
    return names


def evaluate_lattice_expression(expression, arrays):
    """Evaluate a CASA lattice expression on NumPy arrays.

    Supports the subset findContinuum writes: numbers, quoted image names and
    identifiers (``IM0``, ...), ``+ - * /``, comparisons, ``&&``, ``||``, ``!``
    and ``iif(condition, a, b)``. Arrays broadcast against each other (the
    ``stretch`` of CASA); masked pixels are NaN and fail every comparison.

    Parameters
    ----------
    expression : str
        Lattice expression.
    arrays : dict
        Image name (without trailing slashes) to array.

    Returns
    -------
    numpy.ndarray or float
        The evaluated expression.

    Raises
    ------
    ValueError
        If the expression does not parse or uses an unsupported function.
    KeyError
        If the expression refers to an image not in ``arrays``.
    """
    return _ExpressionParser(_tokenize(expression), arrays).parse()


def split_lattice_conjuncts(expression, cube_names):
    """Split a mask at its top-level ``&&`` into a two-dimensional part and cube terms.

    Each operand of the top-level ``&&`` chain is a comparison of the parser
    and carries its own NaN guard, so the AND of the operands, each evaluated
    alone and taken as booleans, is the whole mask exactly.

    Parameters
    ----------
    expression : str
        Lattice expression, ``""`` for no mask.
    cube_names : iterable of str
        Names of the cubes; every other name is a two-dimensional image.

    Returns
    -------
    tuple or None
        ``(two_d, cube_terms)``: the operands that name no cube joined with
        ``" && "`` (``None`` if there are none), and the operands that name a
        cube, in order. ``None`` if the expression has a top-level ``||``.

    Raises
    ------
    ValueError
        If the expression does not tokenize or an operand is empty.
    """
    expression = expression.strip()
    operands, depth, start, end = [], 0, None, None

    def operand():
        if start is None:
            raise ValueError(f"Empty operand in lattice expression {expression!r}")
        return expression[start:end]

    for match in _token_matches(expression):
        kind = match.lastgroup
        value = match.group(kind)
        if kind == "op" and depth == 0 and value in ("&&", "||"):
            if value == "||":
                return None
            operands.append(operand())
            start = None
            continue
        if kind == "op":
            depth += {"(": 1, ")": -1}.get(value, 0)
        if start is None:
            start = match.start(kind)
        end = match.end()
    if operands or start is not None:
        operands.append(operand())
    cubes = {name.rstrip("/") for name in cube_names}
    two_d, cube_terms = [], []
    for text in operands:
        (cube_terms if cubes & set(lattice_expression_names(text)) else two_d).append(
            text
        )
    return (" && ".join(two_d) if two_d else None), cube_terms


def channel_statistics(values, overwrite_input=False):
    """CASA ``imstat`` statistics of one set of pixel values (``classic``).

    The order statistics come from one partition of the values at every index
    that ``numpy.median`` and ``numpy.percentile`` read, so every statistic
    equals the one NumPy's functions give.

    Parameters
    ----------
    values : numpy.ndarray
        Valid pixel values (no NaN, masked pixels removed), one dimension.
    overwrite_input : bool, default False
        Partition a float64 ``values`` in place instead of a copy.

    Returns
    -------
    dict
        The ``CHANNEL_STATISTICS`` keys; all NaN except ``npts`` when
        ``values`` is empty.
    """
    values = (np.asarray if overwrite_input else np.array)(values, dtype=np.float64)
    npts = values.size
    if npts == 0:
        return {key: (0.0 if key == "npts" else np.nan) for key in CHANNEL_STATISTICS}
    sums = {
        "npts": float(npts),
        "sum": float(values.sum()),
        "sumsq": float(np.dot(values, values)),
        "mean": float(values.mean()),
        "sigma": float(values.std(ddof=1)) if npts > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }
    half = npts // 2
    position_1, position_3 = 0.25 * (npts - 1), 0.75 * (npts - 1)
    low_1, low_3 = int(position_1), int(position_3)
    high_1, high_3 = min(low_1 + 1, npts - 1), min(low_3 + 1, npts - 1)
    values.partition(sorted({max(half - 1, 0), half, low_1, high_1, low_3, high_3}))
    median = values[half] if npts % 2 else (values[half - 1] + values[half]) / 2.0
    q1 = _lerp(values[low_1], values[high_1], position_1 - low_1)
    q3 = _lerp(values[low_3], values[high_3], position_3 - low_3)
    np.subtract(values, median, out=values)
    np.abs(values, out=values)
    return {
        **sums,
        "rms": float(np.sqrt(sums["sumsq"] / npts)),
        "median": float(median),
        "medabsdevmed": float(np.median(values, overwrite_input=True)),
        "q1": float(q1),
        "q3": float(q3),
        "quartile": float(q3 - q1),
    }


def _lerp(low, high, fraction):
    """numpy.percentile's linear interpolation between two order statistics."""
    difference = high - low
    if fraction >= 0.5:
        return high - difference * (1 - fraction)
    return low + difference * fraction


def image_statistics(values, algorithm="classic", maxiter=-1):
    """CASA ``imstat`` statistics with the ``classic`` or ``chauvenet`` algorithm.

    Parameters
    ----------
    values : numpy.ndarray
        Valid pixel values.
    algorithm : {"classic", "chauvenet"}, default "classic"
        ``chauvenet`` first removes the outliers of Chauvenet's criterion
        (casacore ``ChauvenetCriterionStatistics``).
    maxiter : int, default -1
        Iterations of ``chauvenet``; only 5, the value findContinuum uses, is
        supported.

    Returns
    -------
    dict
        See :func:`channel_statistics`.

    Raises
    ------
    ValueError
        For another algorithm, or ``chauvenet`` with ``maxiter`` other than 5.
    """
    values = np.asarray(values, dtype=np.float64)
    if algorithm == "chauvenet":
        if maxiter != 5:
            raise ValueError("Only maxiter=5 is supported with algorithm='chauvenet'")
        values = _chauvenet_sample(values)
    elif algorithm != "classic":
        raise ValueError(f"imstat algorithm {algorithm!r} is not supported")
    return channel_statistics(values)


def _profile_value(values, function):
    if values.size == 0:
        return 0.0 if function == "npts" else np.nan
    if function == "mean":
        return float(values.mean())
    if function == "median":
        return float(np.median(values))
    if function == "max":
        return float(values.max())
    if function == "min":
        return float(values.min())
    if function == "npts":
        return float(values.size)
    if function == "xmadm":
        return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))
    if function == "max/xmadm":
        mad = MAD_TO_SIGMA * np.median(np.abs(values - np.median(values)))
        return float(values.max() / mad)
    raise ValueError(f"getprofile function {function!r} is not supported")


def plane_statistics(values, valid, rows=_BLOCK_ROWS):
    """Count, sum, sum of squares, minimum and maximum of the valid pixels of a plane.

    The plane is taken in blocks of ``rows`` rows, in order; the sums are float64
    sums of the blocks' pairwise sums (no BLAS), so equal planes, validities
    and ``rows`` give bitwise equal results.

    Parameters
    ----------
    values : numpy.ndarray
        Pixel values of one plane, ``(l, m)``.
    valid : numpy.ndarray
        Boolean plane of the pixels to include.
    rows : int, default 625
        Rows per block.

    Returns
    -------
    numpy.ndarray
        float64 ``[npts, sum, sumsq, min, max]``; min and max are ``inf`` and
        ``-inf`` without a valid pixel.
    """
    statistics = np.array([0.0, 0.0, 0.0, np.inf, -np.inf])
    for start in range(0, values.shape[0], rows):
        block = slice(start, start + rows)
        _add_statistics(statistics, values[block][valid[block]])
    return statistics


def _add_statistics(statistics, values):
    if values.size:
        values = values.astype(np.float64, copy=False)
        statistics[0] += values.size
        statistics[1] += values.sum()
        statistics[2] += np.square(values).sum()
        statistics[3] = min(statistics[3], values.min())
        statistics[4] = max(statistics[4], values.max())


def merge_channel_rows(channels, rows):
    """Whole-cube sums from per-channel rows, summed in channel order.

    Parameters
    ----------
    channels : array_like of int
        Channel of every row.
    rows : array_like
        ``(n, 5)`` rows of :func:`plane_statistics`.

    Returns
    -------
    dict
        ``npts``, ``sum``, ``sumsq`` (sequential sums over the rows sorted by
        channel), ``min`` and ``max``; independent of how the rows were split
        and ordered.
    """
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    rows = rows[np.argsort(np.asarray(channels), kind="stable")]
    sums = np.cumsum(rows[:, :3], axis=0)[-1] if len(rows) else np.zeros(3)
    return {
        "npts": float(sums[0]),
        "sum": float(sums[1]),
        "sumsq": float(sums[2]),
        "min": float(rows[:, 3].min()) if len(rows) else np.inf,
        "max": float(rows[:, 4].max()) if len(rows) else -np.inf,
    }


def _whole_cube_statistics(sums):
    npts = sums["npts"]
    mean = sums["sum"] / npts if npts else np.nan
    variance = (sums["sumsq"] - npts * mean**2) / (npts - 1) if npts > 1 else np.nan
    return {
        **{key: sums[key] for key in MERGEABLE_STATISTICS},
        "min": sums["min"] if npts else np.nan,
        "max": sums["max"] if npts else np.nan,
        "mean": mean,
        "sigma": float(np.sqrt(max(variance, 0.0))) if npts > 1 else np.nan,
        "rms": float(np.sqrt(sums["sumsq"] / npts)) if npts else np.nan,
        **{key: np.nan for key in ("median", "medabsdevmed", "q1", "q3", "quartile")},
    }


def frequency_from_axis(frequency_axis):
    """Frequency of every channel (Hz) of a linear axis, as CASA computes it.

    Parameters
    ----------
    frequency_axis : sequence
        ``(reference_value, increment, reference_pixel, n_channels)``.

    Returns
    -------
    numpy.ndarray
        float64 frequencies.
    """
    reference_value, increment, reference_pixel, n_channels = frequency_axis
    return reference_value + increment * (np.arange(int(n_channels)) - reference_pixel)


def radio_velocity_km_s(frequency, rest_frequency):
    """Radio velocity of every frequency (km/s), the moment-axis coordinate of CASA."""
    return SPEED_OF_LIGHT_KM_S * (
        1.0 - np.asarray(frequency, dtype=np.float64) / rest_frequency
    )


def moment_velocity_widths(frequency_axis, rest_frequency):
    """Channel widths in radio velocity (km/s) that weight moment 0.

    Parameters
    ----------
    frequency_axis : sequence
        See :func:`frequency_from_axis`.
    rest_frequency : float
        Rest frequency (Hz).

    Returns
    -------
    numpy.ndarray
        ``abs(gradient(velocity))``, ones for a single channel, as
        :class:`~astroviper.processing_functions.image_analysis.moments.MomentsAccumulator`.
    """
    velocity = radio_velocity_km_s(frequency_from_axis(frequency_axis), rest_frequency)
    return np.abs(np.gradient(velocity)) if velocity.size > 1 else np.ones(1)


def linear_channel_ranges(frequency, tolerance=1e-3):
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


def linear_frequency_axis(frequency):
    """The linear frequency axis findContinuum sees for a run of channels.

    Parameters
    ----------
    frequency : array_like
        Channel frequencies of one linear run.

    Returns
    -------
    tuple
        ``(f[0], f[1] - f[0], 0.0, n)``, with increment 1.0 for one channel.
    """
    frequency = np.asarray(frequency)
    return (
        float(frequency[0]),
        float(frequency[1] - frequency[0]) if frequency.size > 1 else 1.0,
        0.0,
        int(frequency.size),
    )


class FindContinuumMoments:
    """findContinuum's moments -1, 0, 8 and 10 of the planes added so far.

    Only the maps the moments need are kept, allocated at the first plane:
    ``INTEGRATED`` (float64, moment 0), ``MAXIMUM`` and ``MINIMUM`` (in the
    plane dtype, where they are exact; moments 8 and 10), ``SUM`` (float64,
    moment -1) and ``COUNT`` (valid channels per pixel, for moment -1 and for
    moment 0 without 8 or 10; uint16, uint32 from 65536 channels). The finalized
    maps equal those of
    :class:`~astroviper.processing_functions.image_analysis.moments.MomentsAccumulator`
    fed the same planes cast to float64.

    Parameters
    ----------
    moments : list of int
        CASA moment numbers (-1, 0, 8, 10).
    velocity_widths : numpy.ndarray
        Width of every channel in velocity (:func:`moment_velocity_widths`).
    map_shape : tuple of int
        Shape of the maps, ``(l, m)``.

    Raises
    ------
    ValueError
        For another moment.
    """

    def __init__(self, moments, velocity_widths, map_shape):
        unsupported = set(moments) - set(CASA_MOMENTS)
        if unsupported:
            raise ValueError(f"Moments {sorted(unsupported)} are not supported")
        self.moments = list(moments)
        self.velocity_widths = np.asarray(velocity_widths, dtype=np.float64)
        self.map_shape = tuple(map_shape)
        self.maps = None

    def _fields(self):
        moments = set(self.moments)
        fields = [
            field
            for field, moment in (
                ("INTEGRATED", 0),
                ("MAXIMUM", 8),
                ("MINIMUM", 10),
                ("SUM", -1),
            )
            if moment in moments
        ]
        if -1 in moments or not moments & {8, 10}:
            fields.append("COUNT")
        return fields

    def add_plane(self, channel, plane, valid, rows=slice(None), scratch=None):
        """Add the valid pixels of one plane (or of a block of its rows).

        Parameters
        ----------
        channel : int
            Index of the plane's channel in ``velocity_widths``.
        plane : numpy.ndarray
            Pixel values of the maps' ``rows``.
        valid : numpy.ndarray
            Boolean, same shape: the pixels to add (finite values only).
        rows : slice or tuple of slice, default slice(None)
            The part of the maps that ``plane`` covers.
        scratch : numpy.ndarray, optional
            Contiguous float64 work array of at least ``plane.size`` values.
        """
        if self.maps is None:
            count = np.uint16 if self.velocity_widths.size < 2**16 else np.uint32
            fill = {
                "INTEGRATED": (0.0, np.float64),
                "MAXIMUM": (-np.inf, plane.dtype),
                "MINIMUM": (np.inf, plane.dtype),
                "SUM": (0.0, np.float64),
                "COUNT": (0, count),
            }
            self.maps = {
                field: np.full(self.map_shape, *fill[field]) for field in self._fields()
            }
        maps = self.maps
        if "COUNT" in maps:
            count = maps["COUNT"][rows]
            count += valid
        if "SUM" in maps:
            total = maps["SUM"][rows]
            np.add(total, plane, out=total, where=valid)
        if "INTEGRATED" in maps:
            product = (
                np.empty(plane.shape)
                if scratch is None
                else scratch.reshape(-1)[: plane.size].reshape(plane.shape)
            )
            np.multiply(
                plane, self.velocity_widths[channel], out=product, dtype=np.float64
            )
            integrated = maps["INTEGRATED"][rows]
            np.add(integrated, product, out=integrated, where=valid)
        if "MAXIMUM" in maps:
            maximum = maps["MAXIMUM"][rows]
            np.maximum(maximum, plane, out=maximum, where=valid)
        if "MINIMUM" in maps:
            minimum = maps["MINIMUM"][rows]
            np.minimum(minimum, plane, out=minimum, where=valid)

    def merge(self, other):
        """Add the maps of ``other`` (disjoint channels; ``None`` is the identity).

        Parameters
        ----------
        other : FindContinuumMoments or None

        Returns
        -------
        FindContinuumMoments
            ``self``, holding the sum; it may share arrays with ``other``.
        """
        if other is None or other.maps is None:
            return self
        if self.maps is None:
            self.maps = other.maps
            return self
        for field, mine in self.maps.items():
            combine = {"MAXIMUM": np.maximum, "MINIMUM": np.minimum}.get(field, np.add)
            combine(mine, other.maps[field], out=mine)
        return self

    def state(self):
        """The maps by field name (``INTEGRATED``, ``MAXIMUM``, ``MINIMUM``,
        ``SUM``, ``COUNT``), or ``None`` before the first plane."""
        return None if self.maps is None else dict(self.maps)

    @classmethod
    def from_state(cls, moments, velocity_widths, map_shape, state):
        """An accumulator holding ``state`` (as :meth:`state` returns it).

        The arrays are used as given, and :meth:`merge` and :meth:`add_plane`
        update them in place.
        """
        moments_maps = cls(moments, velocity_widths, map_shape)
        if state is not None:
            moments_maps.maps = {
                field: state[field] for field in moments_maps._fields()
            }
        return moments_maps

    def finalize(self):
        """CASA moment number to its float64 map, NaN where no channel
        contributed (everywhere before the first plane)."""
        if self.maps is None:
            return {moment: np.full(self.map_shape, np.nan) for moment in self.moments}
        maps = self.maps
        if "MAXIMUM" in maps:
            empty = maps["MAXIMUM"] == -np.inf
        elif "MINIMUM" in maps:
            empty = maps["MINIMUM"] == np.inf
        else:
            empty = maps["COUNT"] == 0
        results = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for moment in self.moments:
                if moment == -1:
                    value = maps["SUM"] / maps["COUNT"]
                elif moment == 0:
                    value = maps["INTEGRATED"].copy()
                elif moment == 8:
                    value = maps["MAXIMUM"].astype(np.float64)
                else:
                    value = maps["MINIMUM"].astype(np.float64)
                value[empty] = np.nan
                results[moment] = value
        return results

    @property
    def nbytes(self):
        """Bytes of the maps."""
        return 0 if self.maps is None else sum(a.nbytes for a in self.maps.values())


def merge_in_tree_order(leaves):
    """Merge partial moments as GraphVIPER's binary tree reduce does.

    The tree pairs neighbours level by level and carries an odd last node up.
    The leaves are merged as they arrive, holding at most
    ``ceil(log2(n)) + 1`` partials, with the same pairs and order
    (``left += right``), so the result equals the tree's bit for bit.

    Parameters
    ----------
    leaves : iterable
        :class:`FindContinuumMoments` or ``None`` (a leaf without channels,
        the identity) per slot, in slot order.

    Returns
    -------
    FindContinuumMoments or None
        The merged moments; ``None`` without a non-``None`` leaf.
    """
    stack = []

    def merge_top():
        size, right = stack.pop()
        left = stack[-1][1]
        stack[-1] = [stack[-1][0] + size, right if left is None else left.merge(right)]

    for leaf in leaves:
        stack.append([1, leaf])
        while len(stack) > 1 and stack[-1][0] == stack[-2][0]:
            merge_top()
    while len(stack) > 1:
        merge_top()
    return stack[0][1] if stack else None


def primary_beam_masks(primary_beam, primary_beam_limit, profile_level=PROFILE_LEVEL):
    """The primary-beam limit mask and findContinuum's spectrum mask ``"pb">0.21``.

    ``inside`` is ``~(primary_beam < limit)`` with the limit as a Python float,
    which compares in the beam's dtype, as the cubes are blanked (a NaN beam is
    inside). ``profile`` is ``inside & (primary_beam > float64(profile_level))``,
    the lattice expression on the blanked float64 cube.

    Parameters
    ----------
    primary_beam : numpy.ndarray
        ``(time, frequency, polarization, l, m)``.
    primary_beam_limit : float or None
        Limit of the blanking; ``None``: every pixel is inside.
    profile_level : float, default PROFILE_LEVEL
        Level of the spectrum mask.

    Returns
    -------
    inside, profile : numpy.ndarray
        Boolean, shaped as ``primary_beam``.
    """
    if primary_beam_limit is None:
        inside = np.ones(primary_beam.shape, dtype=bool)
    else:
        inside = np.less(primary_beam, float(primary_beam_limit))
        np.logical_not(inside, out=inside)
    profile = np.greater(primary_beam, np.float64(profile_level))
    profile &= inside
    return inside, profile


def first_round_statistics(sky, primary_beam, inside):
    """Per-channel statistics of the cube and of its primary beam inside the limit.

    Parameters
    ----------
    sky, primary_beam : numpy.ndarray
        ``(time, frequency, polarization, l, m)``, one time and polarization.
    inside : numpy.ndarray
        Boolean primary-beam limit mask, same shape.

    Returns
    -------
    numpy.ndarray
        float64 ``(frequency, 2, 5)``: (sky, primary beam) x
        :func:`plane_statistics` over the finite pixels inside the limit.

    Raises
    ------
    ValueError
        If there is more than one time or polarization plane.
    """
    _check_planes(sky)
    statistics = np.empty((sky.shape[1], 2, 5))
    valid = np.empty(sky.shape[-2:], dtype=bool)
    for channel in range(sky.shape[1]):
        for index, cube in enumerate((sky, primary_beam)):
            plane = cube[0, channel, 0]
            np.isfinite(plane, out=valid)
            valid &= inside[0, channel, 0]
            statistics[channel, index] = plane_statistics(plane, valid)
    return statistics


def _check_planes(array):
    if array.shape[0] != 1 or array.shape[2] != 1:
        raise ValueError("find_continuum reduces one time and one polarization plane")


def tile_occupancy(mask, tile_shape):
    """Whether each tile of each plane holds a ``True`` pixel.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean ``(time, frequency, polarization, l, m)``.
    tile_shape : tuple of int
        ``(l, m)`` size of the tiles from the first pixel; edge tiles may be
        smaller.

    Returns
    -------
    numpy.ndarray
        Boolean ``(time, frequency, polarization, n_tiles_l, n_tiles_m)``.
    """
    rows = np.logical_or.reduceat(
        mask, np.arange(0, mask.shape[-2], tile_shape[0]), axis=-2
    )
    return np.logical_or.reduceat(
        rows, np.arange(0, mask.shape[-1], tile_shape[1]), axis=-1
    )


def pack_plane_masks(mask):
    """Boolean masks packed eight pixels per byte along ``m`` (``numpy.packbits``)."""
    return np.packbits(mask, axis=-1)


def unpack_plane_masks(packed, n_m, out=None):
    """Masks of :func:`pack_plane_masks`.

    Parameters
    ----------
    packed : numpy.ndarray
        uint8 ``(..., l, ceil(n_m / 8))``.
    n_m : int
        Pixels along ``m``.
    out : numpy.ndarray, optional
        Boolean ``(..., l, n_m)`` to unpack into, a block of rows at a time.

    Returns
    -------
    numpy.ndarray
        Boolean ``(..., l, n_m)``.
    """
    if out is None:
        return np.unpackbits(packed, axis=-1, count=n_m).view(bool)
    for start in range(0, packed.shape[-2], _BLOCK_ROWS):
        rows = slice(start, start + _BLOCK_ROWS)
        out[..., rows, :] = np.unpackbits(
            packed[..., rows, :], axis=-1, count=n_m
        ).view(bool)
    return out


def find_continuum_first_round_requests(
    frequency_axis, rest_frequency, cube, primary_beam
):
    """The reductions of findContinuum's first round that need no earlier result.

    The moment 0 and 8 images of the cube where it is nonzero, the statistics
    of the cube and of the primary beam, and the mean primary beam at five
    channels, as findContinuum asks for them.

    Parameters
    ----------
    frequency_axis : sequence
        Linear frequency axis of the cube, see :func:`frequency_from_axis`.
    rest_frequency : float
        Rest frequency of the moment axis (Hz).
    cube, primary_beam : str
        Names of the cube and of the primary beam in the requests.

    Returns
    -------
    list of dict
        Requests for :func:`find_continuum_cube_reductions`.
    """
    n_channels = int(frequency_axis[3])
    channels = (n_channels // 8, n_channels // 4, n_channels // 2)
    channels += (3 * n_channels // 4, 7 * n_channels // 8)
    moments = dict(
        kind="moments",
        frequency_axis=tuple(frequency_axis),
        rest_frequency=rest_frequency,
    )
    return [
        dict(
            moments,
            image=primary_beam,
            mask="",
            channels=None if n_channels < 16 else sorted(set(channels)),
            moments=[-1],
        ),
        dict(moments, image=cube, mask=f'"{cube}"!=0', channels=None, moments=[0, 8]),
        dict(kind="statistics", image=cube, mask="", channels=None, per_channel=False),
        dict(
            kind="statistics",
            image=primary_beam,
            mask="",
            channels=None,
            per_channel=False,
        ),
    ]


def find_continuum_cube_reductions(
    img_xds,
    requests,
    variables,
    images,
    channel_index,
    primary_beam=None,
    primary_beam_limit=None,
):
    """Partial reductions of one chunk of channels of the dirty cube.

    The chunk is reduced one plane at a time: each cube a request needs is cast
    to float64 and blanked below the primary beam limit once per plane, each
    mask is evaluated once per plane (once per chunk when it refers to
    two-dimensional images only), and moments keep only the maps they need
    (:class:`FindContinuumMoments`). Whole-cube statistics gather the chunk's
    valid values pixel by pixel (``l``, ``m``, then channel), as a sum over
    the chunk.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Chunk of the image, dims ``(time, frequency, polarization, l, m)``, with
        one time and the Stokes I plane only, holding the data variables named
        in ``variables``.
    requests : list of dict
        Reductions to make. Every request has ``kind`` (``"moments"``,
        ``"profile"`` or ``"statistics"``), ``image`` (the name of the reduced
        cube), ``mask`` (a lattice expression, or ``""``) and ``channels``
        (channel indices, or ``None`` for all). ``moments`` requests add
        ``moments`` (CASA moment numbers), ``frequency_axis`` (the linear
        frequency axis of the channels, see :func:`frequency_from_axis`) and
        ``rest_frequency`` (Hz); ``profile`` requests add ``function``;
        ``statistics`` requests add ``per_channel``.
    variables : dict
        Cube name to the data variable of ``img_xds`` that holds it.
    images : dict
        Name of every two-dimensional image a mask refers to, to its values in
        CASA axis order ``(l, m, 1, 1)`` with masked pixels NaN.
    channel_index : numpy.ndarray
        Channel index (as in the requests) of every frequency of ``img_xds``.
    primary_beam : str, optional
        Data variable of ``img_xds`` holding the primary beam; needed with
        ``primary_beam_limit``.
    primary_beam_limit : float, optional
        Pixels where the primary beam is below this level are masked in every
        cube, as in the cubes ``tclean`` makes with ``pblimit``. ``None``
        (default): the cubes as they are.

    Returns
    -------
    list
        One partial result per request: a :class:`FindContinuumMoments`
        (``None`` when the chunk holds none of the request's channels), a dict
        of per-channel values with their ``channels``, or a dict of mergeable
        sums.

    Raises
    ------
    ValueError
        If ``img_xds`` has more than one time or polarization plane.
    """
    if img_xds.sizes["time"] != 1 or img_xds.sizes["polarization"] != 1:
        raise ValueError("find_continuum reduces one time and one polarization plane")
    shape = (img_xds.sizes["l"], img_xds.sizes["m"])
    # (frequency, l, m) planes of every cube, in the dtype of the store.
    raw = {
        name: np.asarray(img_xds[variable].transpose(..., "l", "m").values)[0, :, 0]
        for name, variable in variables.items()
    }
    beam = None
    if primary_beam_limit is not None:
        beam = np.asarray(img_xds[primary_beam].transpose(..., "l", "m").values)[
            0, :, 0
        ]
    partials = [_new_partial(request) for request in requests]

    def mask_of(expression, arrays):
        result = evaluate_lattice_expression(expression, arrays)
        return np.broadcast_to(np.asarray(result, dtype=bool), shape + (1, 1))[
            ..., 0, 0
        ]

    # The cubes each request reads in a plane: its own and those its mask
    # names. A mask of two-dimensional images only is evaluated once here.
    cube_names, masks_2d = [], {}
    for request in requests:
        mask_cubes = set(lattice_expression_names(request["mask"])) & raw.keys()
        cube_names.append(mask_cubes | {request["image"]})
        if request["mask"] and not mask_cubes:
            masks_2d[request["mask"]] = mask_of(request["mask"], images)
    channel_index = np.asarray(channel_index)
    selections = [
        None
        if request["channels"] is None
        else np.isin(channel_index, request["channels"])
        for request in requests
    ]
    gathered = [[] for _ in requests]
    stacks = {}
    scratch = np.empty(shape)
    for position, channel in enumerate(channel_index.tolist()):
        needed = [k for k, s in enumerate(selections) if s is None or s[position]]
        if not needed:
            continue
        outside = None if beam is None else beam[position] < primary_beam_limit
        planes = {}
        for name in set().union(*(cube_names[k] for k in needed)):
            planes[name] = raw[name][position].astype(np.float64)
            if outside is not None:
                planes[name][outside] = np.nan
        finite = {name: np.isfinite(plane) for name, plane in planes.items()}
        cube_masks = {}
        for k in needed:
            request = requests[k]
            plane = planes[request["image"]]
            valid = finite[request["image"]]
            expression = request["mask"]
            if expression in masks_2d:
                valid = valid & masks_2d[expression]
            elif expression:
                if expression not in cube_masks:
                    arrays = {
                        **images,
                        **{n: planes[n][:, :, None, None] for n in cube_names[k]},
                    }
                    cube_masks[expression] = mask_of(expression, arrays)
                valid = valid & cube_masks[expression]
            kind = request["kind"]
            if kind == "moments":
                if partials[k] is None:
                    partials[k] = FindContinuumMoments(
                        request["moments"],
                        moment_velocity_widths(
                            request["frequency_axis"], request["rest_frequency"]
                        ),
                        shape,
                    )
                partials[k].add_plane(
                    channel, raw[request["image"]][position], valid, scratch=scratch
                )
            elif kind == "profile":
                gathered[k].append(_profile_value(plane[valid], request["function"]))
            elif request["per_channel"]:
                gathered[k].append(
                    channel_statistics(plane[valid], overwrite_input=True)
                )
            else:
                values = raw[request["image"]][position]
                if k not in stacks:
                    n = int(
                        channel_index.size
                        if selections[k] is None
                        else selections[k].sum()
                    )
                    stacks[k] = [
                        np.empty(shape + (n,), values.dtype),
                        np.empty(shape + (n,), bool),
                        0,
                    ]
                stack = stacks[k]
                stack[0][..., stack[2]] = values
                stack[1][..., stack[2]] = valid
                stack[2] += 1
    for k, (values, valid, _) in stacks.items():
        values = values[valid].astype(np.float64)
        partials[k] = {
            "npts": float(values.size),
            "sum": float(values.sum()),
            "sumsq": float(np.dot(values, values)),
            "min": float(values.min()) if values.size else np.inf,
            "max": float(values.max()) if values.size else -np.inf,
        }
    for k, request in enumerate(requests):
        if request["kind"] == "moments" or (
            request["kind"] == "statistics" and not request["per_channel"]
        ):
            continue
        selected = (
            channel_index if selections[k] is None else channel_index[selections[k]]
        )
        partials[k]["channels"] = np.concatenate([partials[k]["channels"], selected])
        if request["kind"] == "profile":
            partials[k]["values"] = np.concatenate(
                [partials[k]["values"], np.asarray(gathered[k], dtype=np.float64)]
            )
        else:
            for key in CHANNEL_STATISTICS:
                partials[k]["values"][key] = np.concatenate(
                    [
                        partials[k]["values"][key],
                        np.array([row[key] for row in gathered[k]], dtype=np.float64),
                    ]
                )
    return partials


def _new_partial(request):
    kind = request["kind"]
    if kind == "moments":
        return None
    if kind == "profile":
        return {"channels": np.empty(0, dtype=int), "values": np.empty(0)}
    if kind == "statistics" and request["per_channel"]:
        return {
            "channels": np.empty(0, dtype=int),
            "values": {key: np.empty(0) for key in CHANNEL_STATISTICS},
        }
    if kind == "statistics":
        return {"npts": 0.0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf}
    raise ValueError(f"Unknown find_continuum reduction {kind!r}")


def merge_find_continuum_reductions(partials, other):
    """Merge the partial reductions of two disjoint sets of channels.

    Parameters
    ----------
    partials, other : list
        Partial results of :func:`find_continuum_cube_reductions` for the same
        requests; a ``None`` moments partial is the identity.

    Returns
    -------
    list
        The merged partial results (``partials`` is updated in place).
    """
    for index, (mine, theirs) in enumerate(zip(partials, other, strict=True)):
        if mine is None or isinstance(mine, FindContinuumMoments):
            partials[index] = theirs if mine is None else mine.merge(theirs)
        elif "channels" in mine:
            values = mine["values"]
            if isinstance(values, dict):
                merged_values = {
                    key: np.concatenate([values[key], theirs["values"][key]])
                    for key in values
                }
            else:
                merged_values = np.concatenate([values, theirs["values"]])
            partials[index] = {
                "channels": np.concatenate([mine["channels"], theirs["channels"]]),
                "values": merged_values,
            }
        else:
            mine["npts"] += theirs["npts"]
            mine["sum"] += theirs["sum"]
            mine["sumsq"] += theirs["sumsq"]
            mine["min"] = min(mine["min"], theirs["min"])
            mine["max"] = max(mine["max"], theirs["max"])
    return partials


def finalize_find_continuum_reductions(partials, requests):
    """Final results of merged partial reductions.

    Parameters
    ----------
    partials : list
        Merged partial results, one per request.
    requests : list of dict
        The requests of :func:`find_continuum_cube_reductions`.

    Returns
    -------
    list
        Per request: for ``moments`` a dict of CASA moment number to its map
        ``(l, m)`` (NaN where no channel contributed); for ``profile`` the
        per-channel values in channel order; for per-channel ``statistics`` a
        dict of the ``CHANNEL_STATISTICS`` arrays in channel order; for
        whole-cube ``statistics`` a dict of the mergeable statistics and those
        derived from them (``mean``, ``sigma``, ``rms``); the order statistics
        (``median``, ``medabsdevmed``, quartiles) of a whole cube do not merge
        and are NaN (findContinuum only writes them to a diagnostics file).

    Raises
    ------
    ValueError
        If a moments request selected no channel of the cube.
    """
    results = []
    for partial, request in zip(partials, requests, strict=True):
        if request["kind"] == "moments":
            if partial is None:
                raise ValueError(f"No channel of the cube is selected by {request}")
            maps = partial.finalize()
            results.append({moment: maps[moment] for moment in request["moments"]})
        elif "channels" in partial:
            order = np.argsort(partial["channels"], kind="stable")
            values = partial["values"]
            if isinstance(values, dict):
                results.append({key: values[key][order] for key in values})
            else:
                results.append(values[order])
        else:
            results.append(_whole_cube_statistics(partial))
    return results


def new_streamed_partials(requests):
    """Empty partial results of :func:`find_continuum_batch_reductions`.

    Parameters
    ----------
    requests : list of dict
        Encoded requests (see :func:`find_continuum_batch_reductions`).

    Returns
    -------
    list
        Per request: ``None`` for moments (allocated at the request's first
        plane); ``{"channels", "values"}`` for a profile, with ``values`` a
        dict of the ``CHANNEL_STATISTICS`` for per-channel statistics;
        ``{"channels", "rows"}`` (rows of :func:`plane_statistics`) for
        whole-cube statistics.

    Raises
    ------
    ValueError
        For an unknown request kind.
    """
    partials = []
    for request in requests:
        kind = request["kind"]
        channels = np.empty(0, dtype=np.int64)
        if kind == "moments":
            partials.append(None)
        elif kind == "profile":
            partials.append({"channels": channels, "values": np.empty(0)})
        elif kind == "statistics" and request["per_channel"]:
            values = {key: np.empty(0) for key in CHANNEL_STATISTICS}
            partials.append({"channels": channels, "values": values})
        elif kind == "statistics":
            partials.append({"channels": channels, "rows": np.empty((0, 5))})
        else:
            raise ValueError(f"Unknown find_continuum reduction {kind!r}")
    return partials


def find_continuum_batch_reductions(
    sky,
    channels,
    requests,
    partials,
    inside=None,
    profile=None,
    two_d_masks=None,
    primary_beam=None,
    images=None,
    buffers=None,
):
    """Add a batch of planes to the partial results of encoded requests.

    Per plane, the validity of every cube a request uses is computed once:
    finite and inside the primary-beam limit. A request ANDs it with its
    two-dimensional mask and its cube terms, or with its whole mask (the
    general path), evaluated on the cubes blanked in float64 a block of rows
    at a time. Moments add each block of rows (:class:`FindContinuumMoments`);
    profiles and per-channel statistics gather the valid values in C order into
    one reused float64 buffer from its start; whole-cube statistics add one
    row of :func:`plane_statistics` per channel. Planes are only processed
    within the bounding box of the two-dimensional masks of the requests that
    select them. The results equal those of
    :func:`find_continuum_cube_reductions` for the same masks.

    Parameters
    ----------
    sky : numpy.ndarray
        ``(1, B, 1, l, m)`` planes of the cube, in the dtype of the store.
    channels : array_like of int
        Channel of every plane, increasing over the calls of a task.
    requests : list of dict
        Encoded requests, each with ``kind``, ``image`` (``"sky"`` or
        ``"primary_beam"``), ``channels`` (sorted channels in the numbering of
        ``channels``, or ``None`` for all), ``two_d_mask`` (an index into
        ``two_d_masks``, or ``None``), ``cube_terms`` (a list of
        ``("nonzero", cube)``, ``("profile",)`` for the stored ``"pb">0.21``
        mask, or ``("expression", text)``) and ``expression`` (the whole mask
        for the general path, or ``None``); the lattice expressions name the
        cubes ``"sky"`` and ``"primary_beam"``. Moments requests add
        ``moments``, ``frequency_axis`` and ``rest_frequency`` (of the
        channels findContinuum sees) and ``channel_offset`` (the channel
        number of its channel 0); profiles add ``function``; statistics add
        ``per_channel``.
    partials : list
        One partial result per request (:func:`new_streamed_partials`),
        updated in place.
    inside : numpy.ndarray, optional
        Boolean primary-beam limit mask, shaped as ``sky``; ``None``: no limit.
    profile : numpy.ndarray, optional
        Boolean ``"pb">0.21`` mask (:func:`primary_beam_masks`), shaped as
        ``sky``; needed by ``("profile",)`` terms.
    two_d_masks : list of numpy.ndarray, optional
        Boolean ``(l, m)`` masks.
    primary_beam : numpy.ndarray, optional
        ``(1, B, 1, l, m)`` planes of the primary beam, when a request uses it.
    images : dict, optional
        Name to values ``(l, m, ...)`` (masked pixels NaN) of the
        two-dimensional images the lattice expressions name.
    buffers : dict, optional
        Work arrays kept between the calls of one task (its two-dimensional
        masks fixed), and the seconds spent so far in ``T_mask``,
        ``T_moments``, ``T_statistics`` and ``T_profile``; the function fills
        it. ``None``: a new dict per call.

    Raises
    ------
    ValueError
        If there is more than one time or polarization plane, or ``channels``
        does not have one channel per plane.
    """
    _check_planes(sky)
    channels = np.asarray(channels)
    if channels.shape != (sky.shape[1],):
        raise ValueError("find_continuum needs one channel per plane of the batch")
    buffers = {} if buffers is None else buffers
    n_l, n_m = sky.shape[-2:]
    cubes = {"sky": sky, "primary_beam": primary_beam}
    start = time.perf_counter()
    bounds = buffers.setdefault("bounds", {})
    for request in requests:
        index = request["two_d_mask"]
        if index is not None and index not in bounds:
            bounds[index] = _mask_bounds(two_d_masks[index])
    _add_seconds(buffers, "T_mask", start)
    selected = [
        np.ones(channels.size, dtype=bool)
        if request["channels"] is None
        else np.isin(channels, request["channels"])
        for request in requests
    ]
    request_cubes = [_request_cubes(request) for request in requests]
    batch_rows = [[] for _ in requests]
    for position, channel in enumerate(channels.tolist()):
        needed = [k for k in range(len(requests)) if selected[k][position]]
        if not needed:
            continue
        start = time.perf_counter()
        r0, r1, c0, c1 = _union_bounds(
            [requests[k]["two_d_mask"] for k in needed], bounds, (n_l, n_m)
        )
        window = (slice(r0, r1), slice(c0, c1))
        shape = (r1 - r0, c1 - c0)
        blocks = _row_blocks(r0, r1)
        planes = {
            name: cubes[name][0, position, 0][window]
            for name in set().union(*(request_cubes[k] for k in needed))
        }
        inside_plane = None if inside is None else inside[0, position, 0][window]
        base = {}
        for name in {requests[k]["image"] for k in needed}:
            base[name] = np.isfinite(
                planes[name], out=_work_array(buffers, f"base_{name}", shape, bool)
            )
            if inside_plane is not None:
                base[name] &= inside_plane
        _add_seconds(buffers, "T_mask", start)
        for k in needed:
            request = requests[k]
            start = time.perf_counter()
            valid = base[request["image"]]
            if (
                request["two_d_mask"] is not None
                or request["cube_terms"]
                or request["expression"] is not None
            ):
                valid = _work_array(buffers, "valid", shape, bool)
                np.copyto(valid, base[request["image"]])
            if request["two_d_mask"] is not None:
                valid &= two_d_masks[request["two_d_mask"]][window]
            expressions = []
            for term in request["cube_terms"]:
                if term[0] == "nonzero":
                    valid &= planes[term[1]] != 0
                    if term[1] != request["image"]:
                        valid &= ~np.isnan(planes[term[1]])
                elif term[0] == "profile":
                    valid &= profile[0, position, 0][window]
                else:
                    expressions.append(term[1])
            if request["expression"] is not None:
                expressions.append(request["expression"])
            if expressions:
                names = set().union(*map(lattice_expression_names, expressions))
                for rows, local in blocks:
                    arrays = {}
                    for name in names:
                        if name in _CUBE_ROLES:
                            block = planes[name][local].astype(np.float64)
                            if inside_plane is not None:
                                block[~inside_plane[local]] = np.nan
                        else:
                            block = images[name][rows, c0:c1].reshape(
                                rows.stop - rows.start, c1 - c0
                            )
                        arrays[name] = block
                    for text in expressions:
                        valid[local] &= np.asarray(
                            evaluate_lattice_expression(text, arrays), dtype=bool
                        )
            _add_seconds(buffers, "T_mask", start)
            start = time.perf_counter()
            plane = planes[request["image"]]
            kind = request["kind"]
            if kind == "moments":
                if partials[k] is None:
                    partials[k] = FindContinuumMoments(
                        request["moments"],
                        moment_velocity_widths(
                            request["frequency_axis"], request["rest_frequency"]
                        ),
                        (n_l, n_m),
                    )
                scratch = _work_array(
                    buffers, "scratch", (_BLOCK_ROWS * shape[1],), float
                )
                for rows, local in blocks:
                    partials[k].add_plane(
                        channel - request["channel_offset"],
                        plane[local],
                        valid[local],
                        rows=(rows, slice(c0, c1)),
                        scratch=scratch,
                    )
                _add_seconds(buffers, "T_moments", start)
            elif kind == "statistics" and not request["per_channel"]:
                statistics = np.array([0.0, 0.0, 0.0, np.inf, -np.inf])
                for _, local in blocks:
                    _add_statistics(statistics, plane[local][valid[local]])
                batch_rows[k].append((channel, statistics))
                _add_seconds(buffers, "T_statistics", start)
            else:
                values = _work_array(
                    buffers, "gather", (np.count_nonzero(valid),), float
                )
                values[...] = plane[valid]
                if kind == "profile":
                    batch_rows[k].append(
                        (channel, _profile_value(values, request["function"]))
                    )
                    _add_seconds(buffers, "T_profile", start)
                else:
                    batch_rows[k].append(
                        (channel, channel_statistics(values, overwrite_input=True))
                    )
                    _add_seconds(buffers, "T_statistics", start)
    for partial, rows in zip(partials, batch_rows, strict=True):
        if not rows:
            continue
        new_channels = np.array([channel for channel, _ in rows], dtype=np.int64)
        partial["channels"] = np.concatenate([partial["channels"], new_channels])
        if "rows" in partial:
            partial["rows"] = np.concatenate(
                [partial["rows"], [row for _, row in rows]]
            )
        elif isinstance(partial["values"], dict):
            for key, column in partial["values"].items():
                partial["values"][key] = np.concatenate(
                    [column, [row[key] for _, row in rows]]
                )
        else:
            partial["values"] = np.concatenate(
                [partial["values"], [value for _, value in rows]]
            )


def _request_cubes(request):
    names = {request["image"]}
    texts = [request["expression"]] if request["expression"] is not None else []
    for term in request["cube_terms"]:
        if term[0] == "nonzero":
            names.add(term[1])
        elif term[0] == "expression":
            texts.append(term[1])
    for text in texts:
        names.update(set(lattice_expression_names(text)) & set(_CUBE_ROLES))
    return names


def _mask_bounds(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(columns[0]), int(columns[-1]) + 1


def _union_bounds(indices, bounds, shape):
    if any(index is None for index in indices):
        return 0, shape[0], 0, shape[1]
    boxes = [bounds[index] for index in indices if bounds[index] is not None]
    if not boxes:
        return 0, 0, 0, 0
    r0, r1, c0, c1 = zip(*boxes, strict=True)
    return min(r0), max(r1), min(c0), max(c1)


def _row_blocks(start, stop):
    """Blocks of rows aligned to multiples of _BLOCK_ROWS, cut to [start, stop),
    with their slices relative to start."""
    blocks = []
    for row in range(start - start % _BLOCK_ROWS, stop, _BLOCK_ROWS):
        rows = slice(max(row, start), min(row + _BLOCK_ROWS, stop))
        blocks.append((rows, slice(rows.start - start, rows.stop - start)))
    return blocks


def _work_array(buffers, name, shape, dtype):
    size = int(np.prod(shape))
    flat = buffers.get(name)
    if flat is None or flat.size < size or flat.dtype != dtype:
        flat = buffers[name] = np.empty(size, dtype=dtype)
    return flat[:size].reshape(shape)


def _add_seconds(buffers, key, start):
    buffers[key] = buffers.get(key, 0.0) + time.perf_counter() - start


def finalize_streamed_reductions(merged, requests):
    """Final results of merged :func:`find_continuum_batch_reductions` partials.

    Parameters
    ----------
    merged : list
        Per request, the partial results of every task concatenated (rows in
        any order); for moments a :class:`FindContinuumMoments` or ``None``.
    requests : list of dict
        The encoded requests.

    Returns
    -------
    list
        As :func:`finalize_find_continuum_reductions`; whole-cube statistics
        from the rows summed in channel order (:func:`merge_channel_rows`).
        A moments request with a ``None`` partial gives ``None``.
    """
    results = []
    for partial, request in zip(merged, requests, strict=True):
        kind = request["kind"]
        if kind == "moments":
            maps = None if partial is None else partial.finalize()
            results.append(
                None if maps is None else {m: maps[m] for m in request["moments"]}
            )
        elif kind == "statistics" and not request["per_channel"]:
            results.append(
                _whole_cube_statistics(
                    merge_channel_rows(partial["channels"], partial["rows"])
                )
            )
        else:
            order = np.argsort(partial["channels"], kind="stable")
            values = partial["values"]
            if isinstance(values, dict):
                results.append({key: values[key][order] for key in values})
            else:
                results.append(values[order])
    return results
