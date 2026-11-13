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
from astroviper.processing_functions.image_analysis.moments import MomentsAccumulator

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
    values = np.asarray(values, dtype=np.float64)
    npts = values.size
    if npts == 0:
        return {key: (0.0 if key == "npts" else np.nan) for key in CHANNEL_STATISTICS}
    median = np.median(values)
    q1, q3 = np.percentile(values, [25.0, 75.0])
    sumsq = float(np.dot(values, values))
    return {
        "npts": float(npts),
        "sum": float(values.sum()),
        "sumsq": sumsq,
        "mean": float(values.mean()),
        "sigma": float(values.std(ddof=1)) if npts > 1 else 0.0,
        "rms": float(np.sqrt(sumsq / npts)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(median),
        "medabsdevmed": float(np.median(np.abs(values - median))),
        "q1": float(q1),
        "q3": float(q3),
        "quartile": float(q3 - q1),
    }


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


def radio_velocity_km_s(frequency, rest_frequency):
    """Radio velocity of every frequency (km/s), the moment-axis coordinate of CASA."""
    return SPEED_OF_LIGHT_KM_S * (
        1.0 - np.asarray(frequency, dtype=np.float64) / rest_frequency
    )


def find_continuum_cube_reductions(img_xds, requests, variables, images, channel_index):
    """Partial reductions of one chunk of channels of the dirty cube.

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
        ``moments`` (CASA moment numbers), ``frequency`` (every channel of the
        cube, Hz) and ``rest_frequency`` (Hz); ``profile`` requests add
        ``function``; ``statistics`` requests add ``per_channel``.
    variables : dict
        Cube name to the data variable of ``img_xds`` that holds it.
    images : dict
        Name of every two-dimensional image a mask refers to, to its values in
        CASA axis order ``(l, m, 1, 1)`` with masked pixels NaN.
    channel_index : numpy.ndarray
        Global channel index of every frequency of ``img_xds``.

    Returns
    -------
    list
        One partial result per request: a ``MomentsAccumulator``, a dict of
        per-channel values with their ``channels``, or a dict of mergeable sums.

    Raises
    ------
    ValueError
        If ``img_xds`` has more than one time or polarization plane.
    """
    if img_xds.sizes["time"] != 1 or img_xds.sizes["polarization"] != 1:
        raise ValueError("find_continuum reduces one time and one polarization plane")
    cubes = {
        name: np.transpose(img_xds[variable].values[0], (2, 3, 1, 0)).astype(np.float64)
        for name, variable in variables.items()
    }
    arrays = {**images, **cubes}
    channel_index = np.asarray(channel_index)
    partials = []
    for request in requests:
        cube = cubes[request["image"]]
        selected = (
            np.ones(channel_index.size, dtype=bool)
            if request["channels"] is None
            else np.isin(channel_index, request["channels"])
        )
        valid = np.isfinite(cube)
        if request["mask"]:
            valid &= np.broadcast_to(
                np.asarray(
                    evaluate_lattice_expression(request["mask"], arrays), dtype=bool
                ),
                cube.shape,
            )
        kind = request["kind"]
        if kind == "moments":
            accumulator = MomentsAccumulator(
                [CASA_MOMENTS[moment] for moment in request["moments"]],
                radio_velocity_km_s(request["frequency"], request["rest_frequency"]),
                cube.shape[:2],
            )
            for position in np.flatnonzero(selected):
                accumulator.add_plane(
                    int(channel_index[position]),
                    cube[:, :, 0, position],
                    valid[:, :, 0, position],
                )
            partials.append(accumulator)
        elif kind == "profile":
            positions = np.flatnonzero(selected)
            partials.append(
                {
                    "channels": channel_index[positions],
                    "values": np.array(
                        [
                            _profile_value(
                                cube[..., p][valid[..., p]], request["function"]
                            )
                            for p in positions
                        ]
                    ),
                }
            )
        elif kind == "statistics" and request["per_channel"]:
            positions = np.flatnonzero(selected)
            rows = [channel_statistics(cube[..., p][valid[..., p]]) for p in positions]
            partials.append(
                {
                    "channels": channel_index[positions],
                    "values": {
                        key: np.array([row[key] for row in rows])
                        for key in CHANNEL_STATISTICS
                    },
                }
            )
        elif kind == "statistics":
            values = cube[..., selected][valid[..., selected]]
            partials.append(
                {
                    "npts": float(values.size),
                    "sum": float(values.sum()),
                    "sumsq": float(np.dot(values, values)),
                    "min": float(values.min()) if values.size else np.inf,
                    "max": float(values.max()) if values.size else -np.inf,
                }
            )
        else:
            raise ValueError(f"Unknown find_continuum reduction {kind!r}")
    return partials


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
        if isinstance(mine, MomentsAccumulator):
            mine.merge(theirs)
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
        if isinstance(partial, MomentsAccumulator):
            maps = partial.finalize()
            results.append(
                {moment: maps[CASA_MOMENTS[moment]] for moment in request["moments"]}
            )
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
