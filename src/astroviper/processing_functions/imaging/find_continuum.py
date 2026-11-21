"""Cube reductions of the continuum channel finder (``find_continuum``).

Every reduction of the dirty cube that the pipeline's findContinuum makes with
CASA (``immoments``, ``imstat`` and ``image.getprofile``) runs on one chunk of
channels at a time: moment images merge partial accumulators, per-channel
statistics and spectra gather one value per channel, and whole-cube statistics
merge their sums and extrema. Masks are CASA lattice expressions over the cube,
its primary beam and two-dimensional images, evaluated in CASA axis order
``(l, m, polarization, frequency)``.

``find_continuum`` reduces whole chunks (:func:`find_continuum_cube_reductions`).
"""

import re

import numpy as np

from astroviper.processing_functions.image_analysis.statistics.chauvenet import (
    chauvenet_sample,
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
        values = chauvenet_sample(values)
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

    def add_plane(self, channel, plane, valid, scratch=None):
        """Add the valid pixels of one plane.

        Parameters
        ----------
        channel : int
            Index of the plane's channel in ``velocity_widths``.
        plane : numpy.ndarray
            Pixel values, shaped as the maps.
        valid : numpy.ndarray
            Boolean, same shape: the pixels to add (finite values only).
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
            maps["COUNT"] += valid
        if "SUM" in maps:
            np.add(maps["SUM"], plane, out=maps["SUM"], where=valid)
        if "INTEGRATED" in maps:
            product = (
                np.empty(plane.shape)
                if scratch is None
                else scratch.reshape(-1)[: plane.size].reshape(plane.shape)
            )
            np.multiply(
                plane, self.velocity_widths[channel], out=product, dtype=np.float64
            )
            np.add(maps["INTEGRATED"], product, out=maps["INTEGRATED"], where=valid)
        if "MAXIMUM" in maps:
            np.maximum(maps["MAXIMUM"], plane, out=maps["MAXIMUM"], where=valid)
        if "MINIMUM" in maps:
            np.minimum(maps["MINIMUM"], plane, out=maps["MINIMUM"], where=valid)

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
