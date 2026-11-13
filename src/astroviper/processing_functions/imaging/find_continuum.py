"""Cube reductions of the continuum channel finder (``find_continuum``).

Every reduction of the dirty cube that the pipeline's findContinuum makes with
CASA (``immoments``, ``imstat`` and ``image.getprofile``) runs on one chunk of
channels at a time: moment images merge partial accumulators, per-channel
statistics and spectra gather one value per channel, and whole-cube statistics
merge their sums and extrema. Masks are CASA lattice expressions over the cube,
its primary beam and two-dimensional images, evaluated in CASA axis order
``(l, m, polarization, frequency)``.
"""

import re

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

_TOKEN = re.compile(
    r"\s*(?:(?P<number>\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)"
    r"|(?P<name>\"[^\"]*\"|'[^']*')"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<op>&&|\|\||==|!=|>=|<=|[-+*/()<>,!]))"
)


def _tokenize(expression):
    tokens, position = [], 0
    expression = expression.strip()
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if match is None or match.end() == position:
            raise ValueError(f"Cannot parse lattice expression {expression!r}")
        kind = match.lastgroup
        value = match.group(kind)
        if kind == "name":
            value = value[1:-1]
        tokens.append((kind, value))
        position = match.end()
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


def channel_statistics(values):
    """CASA ``imstat`` statistics of one set of pixel values (``classic``).

    Parameters
    ----------
    values : numpy.ndarray
        Valid pixel values (no NaN, masked pixels removed).

    Returns
    -------
    dict
        The ``CHANNEL_STATISTICS`` keys; all NaN except ``npts`` when
        ``values`` is empty.
    """
    values = np.array(values, dtype=np.float64)
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
    # The order statistics of numpy.median and numpy.percentile (linear), from
    # one partition of the copy at every index they read.
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


def frequency_axis_values(frequency_axis):
    """Frequency of every channel (Hz) of a linear axis ``[reference_value,
    increment, reference_pixel, n_channels]``, as CASA computes it."""
    reference_value, increment, reference_pixel, n_channels = frequency_axis
    return reference_value + increment * (np.arange(int(n_channels)) - reference_pixel)


def radio_velocity_km_s(frequency, rest_frequency):
    """Radio velocity of every frequency (km/s), the moment-axis coordinate of CASA."""
    return SPEED_OF_LIGHT_KM_S * (
        1.0 - np.asarray(frequency, dtype=np.float64) / rest_frequency
    )


class MomentMaps:
    """CASA moments -1, 0, 8 and 10 of the planes added so far.

    Only the maps the moments need are kept, allocated at the first plane: the
    number of valid channels (moment -1) or a flag of contributing pixels,
    float64 sums for moments -1 and 0, and the running maximum and minimum in
    the cube's dtype, where they are exact. The finalized maps equal those of
    :class:`~astroviper.processing_functions.image_analysis.moments.MomentsAccumulator`.

    Parameters
    ----------
    moments : sequence of int
        CASA moment numbers (-1, 0, 8, 10).
    velocity : numpy.ndarray
        Radio velocity of every channel of the cube (km/s); moment 0 sums the
        planes times the channel widths in velocity.
    shape : tuple of int
        Map shape ``(l, m)``.
    """

    def __init__(self, moments, velocity, shape):
        self.moments = tuple(moments)
        velocity = np.asarray(velocity, dtype=np.float64)
        self.widths = np.abs(np.gradient(velocity)) if velocity.size > 1 else np.ones(1)
        self.shape = tuple(shape)
        self.count = self.total = self.integrated = None
        self.maximum = self.minimum = None

    def add_plane(self, channel, plane, valid, scratch):
        """Add the valid pixels of one plane of channel ``channel``; ``scratch``
        is a float64 plane the call may overwrite."""
        if self.count is None:
            self.count = np.zeros(
                self.shape, dtype=np.uint32 if -1 in self.moments else bool
            )
            if -1 in self.moments:
                self.total = np.zeros(self.shape)
            if 0 in self.moments:
                self.integrated = np.zeros(self.shape)
            if 8 in self.moments:
                self.maximum = np.full(self.shape, -np.inf, dtype=plane.dtype)
            if 10 in self.moments:
                self.minimum = np.full(self.shape, np.inf, dtype=plane.dtype)
        if self.count.dtype == bool:
            np.logical_or(self.count, valid, out=self.count)
        else:
            self.count += valid
        if self.total is not None:
            np.add(self.total, plane, out=self.total, where=valid)
        if self.integrated is not None:
            np.multiply(plane, self.widths[channel], out=scratch, dtype=np.float64)
            np.add(self.integrated, scratch, out=self.integrated, where=valid)
        if self.maximum is not None:
            np.maximum(self.maximum, plane, out=self.maximum, where=valid)
        if self.minimum is not None:
            np.minimum(self.minimum, plane, out=self.minimum, where=valid)

    def merge(self, other):
        """Fold ``other``'s maps (disjoint channels) into these, in place."""
        if other.count is None:
            return self
        if self.count is None:
            return other
        if self.count.dtype == bool:
            np.logical_or(self.count, other.count, out=self.count)
        else:
            self.count += other.count
        for name, combine in (
            ("total", np.add),
            ("integrated", np.add),
            ("maximum", np.maximum),
            ("minimum", np.minimum),
        ):
            if getattr(self, name) is not None:
                combine(
                    getattr(self, name), getattr(other, name), out=getattr(self, name)
                )
        return self

    def finalize(self):
        """CASA moment number to its float64 map, NaN where no channel contributed."""
        if self.count is None:
            return {moment: np.full(self.shape, np.nan) for moment in self.moments}
        empty = self.count == 0
        maps = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for moment in self.moments:
                if moment == -1:
                    value = self.total / self.count
                elif moment == 0:
                    value = self.integrated.copy()
                elif moment == 8:
                    value = self.maximum.astype(np.float64)
                else:
                    value = self.minimum.astype(np.float64)
                value[empty] = np.nan
                maps[moment] = value
        return maps


def find_continuum_first_round_requests(
    n_channels, first_frequency, channel_width, rest_frequency, cube, primary_beam
):
    """The reductions of findContinuum's first round that need no earlier result.

    The moment 0 and 8 images of the cube where it is nonzero, the statistics
    of the cube and of the primary beam, and the mean primary beam at five
    channels, as findContinuum asks for them.

    Parameters
    ----------
    n_channels : int
        Channels of the cube.
    first_frequency, channel_width : float
        Frequency of channel 0 and the channel width (Hz).
    rest_frequency : float
        Rest frequency of the moment axis (Hz).
    cube, primary_beam : str
        Names of the cube and of the primary beam in the requests.

    Returns
    -------
    list of dict
        Requests for :func:`find_continuum_cube_reductions`.
    """
    channels = (n_channels // 8, n_channels // 4, n_channels // 2)
    channels += (3 * n_channels // 4, 7 * n_channels // 8)
    moments = dict(
        kind="moments",
        frequency_axis=[first_frequency, channel_width, 0.0, n_channels],
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
    partials=None,
):
    """Partial reductions of one chunk of channels of the dirty cube.

    The chunk is reduced one plane at a time: each cube a request needs is cast
    to float64 and blanked below the primary beam limit once per plane, each
    mask is evaluated once per plane (once per chunk when it refers to
    two-dimensional images only), and moments keep only the maps they need
    (:class:`MomentMaps`).

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
        (global channel indices, or ``None`` for all). ``moments`` requests add
        ``moments`` (CASA moment numbers), ``frequency_axis`` (the cube's linear
        frequency axis, see :func:`frequency_axis_values`) and
        ``rest_frequency`` (Hz); ``profile`` requests add
        ``function``; ``statistics`` requests add ``per_channel``.
    variables : dict
        Cube name to the data variable of ``img_xds`` that holds it.
    images : dict
        Name of every two-dimensional image a mask refers to, to its values in
        CASA axis order ``(l, m, 1, 1)`` with masked pixels NaN.
    channel_index : numpy.ndarray
        Global channel index of every frequency of ``img_xds``.
    primary_beam : str, optional
        Data variable of ``img_xds`` holding the primary beam; needed with
        ``primary_beam_limit``.
    primary_beam_limit : float, optional
        Pixels where the primary beam is below this level are masked in every
        cube, as in the cubes ``tclean`` makes with ``pblimit``. ``None``
        (default): the cubes as they are.
    partials : list, optional
        Partial results of other channels for the same requests, to which this
        chunk is added in place. ``None`` (default): new partial results.

    Returns
    -------
    list
        One partial result per request: a :class:`MomentMaps`, a dict of
        per-channel values with their ``channels``, or a dict of mergeable sums.

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
    if partials is None:
        partials = [_new_partial(request, shape) for request in requests]

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
            request, partial = requests[k], partials[k]
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
                partial.add_plane(
                    channel, raw[request["image"]][position], valid, scratch
                )
            elif kind == "profile":
                gathered[k].append(_profile_value(plane[valid], request["function"]))
            elif request["per_channel"]:
                gathered[k].append(channel_statistics(plane[valid]))
            else:
                values = plane[valid]
                partial["npts"] += float(values.size)
                partial["sum"] += float(values.sum())
                partial["sumsq"] += float(np.dot(values, values))
                if values.size:
                    partial["min"] = min(partial["min"], float(values.min()))
                    partial["max"] = max(partial["max"], float(values.max()))
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


def _new_partial(request, shape):
    kind = request["kind"]
    if kind == "moments":
        return MomentMaps(
            request["moments"],
            radio_velocity_km_s(
                frequency_axis_values(request["frequency_axis"]),
                request["rest_frequency"],
            ),
            shape,
        )
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
        requests.

    Returns
    -------
    list
        The merged partial results (``partials`` is updated in place).
    """
    for index, (mine, theirs) in enumerate(zip(partials, other, strict=True)):
        if isinstance(mine, MomentMaps):
            partials[index] = mine.merge(theirs)
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
    """
    results = []
    for partial, request in zip(partials, requests, strict=True):
        if isinstance(partial, MomentMaps):
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
            npts = partial["npts"]
            mean = partial["sum"] / npts if npts else np.nan
            variance = (
                (partial["sumsq"] - npts * mean**2) / (npts - 1) if npts > 1 else np.nan
            )
            results.append(
                {
                    **{key: partial[key] for key in MERGEABLE_STATISTICS},
                    "min": partial["min"] if npts else np.nan,
                    "max": partial["max"] if npts else np.nan,
                    "mean": mean,
                    "sigma": float(np.sqrt(max(variance, 0.0))) if npts > 1 else np.nan,
                    "rms": float(np.sqrt(partial["sumsq"] / npts)) if npts else np.nan,
                    **{
                        key: np.nan
                        for key in ("median", "medabsdevmed", "q1", "q3", "quartile")
                    },
                }
            )
    return results
