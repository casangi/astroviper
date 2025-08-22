from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Tuple, Union, Sequence
from numbers import Real
import numpy as np
import xarray as xr
import dask.array as da

ArrayLike2D = Union[np.ndarray, da.Array]
OutputKind = Literal["match", "xarray", "numpy", "dask"]
ReturnType = Union[xr.DataArray, np.ndarray, da.Array]


def _is_dask_array(obj: Any) -> bool:
    """
    Return True if ``obj`` is a Dask array instance at runtime.
    """
    return isinstance(obj, da.Array)


def _coerce_to_xda(
    data: Union[xr.DataArray, ArrayLike2D],
    *,
    x_coord: str,
    y_coord: str,
    coords: Optional[Mapping[str, np.ndarray]] = None,
    dims: Optional[Tuple[str, str]] = None,
) -> xr.DataArray:
    """
    Coerce input into an ``xarray.DataArray`` with the requested x/y coordinates.

    This helper allows all generators to accept either an ``xarray.DataArray`` or
    a raw 2-D NumPy/Dask array. In the array case you must provide 1-D coordinate
    arrays for the horizontal and vertical axes via ``coords``. No data copies
    are made; the DataArray wraps the original array.

    Parameters
    ----------
    data
        Input grid. Either an ``xarray.DataArray`` or a 2-D NumPy/Dask array.
    x_coord, y_coord
        Names of the coordinate variables representing the horizontal and
        vertical axes in world units.
    coords
        Required when ``data`` is a NumPy/Dask array. A mapping that must include
        1-D arrays for ``x_coord`` and ``y_coord`` whose lengths match the array
        width and height, respectively.
    dims
        Optional dimension names to assign when wrapping a NumPy/Dask array.
        Defaults to ``(y_coord, x_coord)``.

    Returns
    -------
    xarray.DataArray
        A view on the input data with coordinates attached.

    Raises
    ------
    TypeError
        If ``data`` is not a supported type.
    ValueError
        If a NumPy/Dask input is not 2-D, coordinates are missing, or lengths
        are inconsistent with the array shape.
    """
    if isinstance(data, xr.DataArray):
        return data

    if not isinstance(data, (np.ndarray, da.Array)):
        raise TypeError(
            "data must be a DataArray, a 2-D NumPy ndarray, or a Dask array"
        )

    if data.ndim != 2:
        raise ValueError("NumPy/Dask array input must be 2-D")

    if coords is None:
        raise ValueError("coords must be provided for NumPy/Dask array input")

    if x_coord not in coords or y_coord not in coords:
        raise ValueError(
            f"coords must include 1-D arrays for {x_coord!r} and {y_coord!r}"
        )

    x_vals = np.asarray(coords[x_coord])
    y_vals = np.asarray(coords[y_coord])

    H, W = data.shape
    if y_vals.shape[0] != H or x_vals.shape[0] != W:
        raise ValueError(
            f"Coordinate lengths must match array shape: "
            f"{y_coord} len={y_vals.shape[0]} vs H={H}, "
            f"{x_coord} len={x_vals.shape[0]} vs W={W}"
        )

    if dims is None:
        dims = (y_coord, x_coord)

    return xr.DataArray(data, coords={y_coord: y_vals, x_coord: x_vals}, dims=dims)


def _rotated_coords(
    xda: xr.DataArray,
    *,
    x_coord: str,
    y_coord: str,
    x0: float,
    y0: float,
    theta: float,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Compute rotated, centered coordinates for an ellipse/Gaussian.

    Let ``X, Y`` be the broadcast 2-D coordinate grids in world units. This
    returns the rotated coordinates

    ``xp =  (X - x0) * cos(theta) + (Y - y0) * sin(theta)``
    ``yp = -(X - x0) * sin(theta) + (Y - y0) * cos(theta)``

    which align the x-axis with the ellipse/Gaussian semi-major axis.

    Parameters
    ----------
    xda
        DataArray that holds the coordinate variables.
    x_coord, y_coord
        Names of the coordinate variables to use.
    x0, y0
        Center in world coordinates.
    theta
        Rotation angle in radians measured from +x toward +y.

    Returns
    -------
    (xp, yp)
        Two DataArrays with the same broadcast shape as the input grids.
    """
    X, Y = xr.broadcast(xda[x_coord], xda[y_coord])
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    xp = (X - x0) * ct + (Y - y0) * st
    yp = -(X - x0) * st + (Y - y0) * ct
    return xp, yp


def _nearest_indices_1d(
    coord_vals: np.ndarray,
    targets: np.ndarray,
    *,
    out_of_range: Literal["ignore", "ignore_sloppy", "clip", "error"] = "ignore",
    return_valid_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Map world-coordinate ``targets`` to nearest integer indices along one axis.

    Behavior
    --------
    Accepts a strictly monotonic 1-D coordinate array ``coord_vals`` that may be
    increasing or decreasing. For each target, returns the index of the nearest
    coordinate in absolute distance. Midpoint ties between two neighbors choose
    the right-hand neighbor deterministically.

    Out-of-range handling is controlled by ``out_of_range``:
      - ``"ignore"`` (default). Strict ignore. A target is valid only if it lies
        within the closed coordinate range [min(coord_vals), max(coord_vals)].
        If ``return_valid_mask=True``, a boolean mask is returned so callers can
        skip OOR targets. If the mask is not requested, indices are still
        returned (clipped) to avoid breaking legacy callers.
      - ``"ignore_sloppy"``. Half-pixel tolerant ignore. A target is treated as
        valid if it lies within an expanded range that extends the lower edge by
        half the first pixel spacing and the upper edge by half the last pixel
        spacing. This is useful when sources land just outside the nominal
        coverage by less than half a pixel.
      - ``"clip"``. Clamp OOR targets to the nearest valid index.
      - ``"error"``. Raise ``ValueError`` if any target lies outside the
        coordinate range [min, max].

    Parameters
    ----------
    coord_vals
        1-D array of axis coordinates (in world units). Must be strictly
        monotonic (all increasing or all decreasing).
    targets
        1-D array-like of world coordinates to map to indices.
    out_of_range
        One of ``{"ignore", "ignore_sloppy", "clip", "error"}``. See Behavior.
    return_valid_mask
        If ``True``, also return a boolean array ``valid`` where ``True`` marks
        targets considered in-range under the chosen policy.

    Returns
    -------
    np.ndarray
        Integer indices into ``coord_vals`` for each target (same shape as
        ``targets``).
    (np.ndarray, np.ndarray)
        If ``return_valid_mask=True``, returns ``(indices, valid)`` where
        ``valid`` is a boolean mask array.

    Notes
    -----
    Internally, a monotone-increasing view of ``coord_vals`` is used for the
    binary search, and indices are mapped back if the original coords were
    decreasing. Half-pixel tolerance for ``"ignore_sloppy"`` uses the local edge
    spacings: ``0.5 * |vals[1]-vals[0]|`` at the lower edge and
    ``0.5 * |vals[-1]-vals[-2]|`` at the upper edge.
    """
    coord_vals = np.asarray(coord_vals)
    targets_arr = np.asarray(targets).ravel()

    if coord_vals.ndim != 1:
        raise ValueError("coord_vals must be 1-D")

    # Must check for empty before accessing [-1] or [0]
    if coord_vals.size < 1:
        raise ValueError("coord_vals must have length >= 1")

    ascending = bool(coord_vals[-1] >= coord_vals[0])
    vals = coord_vals if ascending else coord_vals[::-1]

    if vals.size >= 2:
        diffs = np.diff(vals)
        if not (diffs > 0).all():
            raise ValueError("coord_vals must be strictly monotonic")

    val_min = vals[0]
    val_max = vals[-1]

    if out_of_range == "error":
        below = targets_arr < val_min
        above = targets_arr > val_max
        if np.any(below | above):
            raise ValueError("One or more targets lie outside the coordinate range.")

    if out_of_range in ("ignore", "clip", "ignore_sloppy"):
        if out_of_range == "ignore":
            valid = (targets_arr >= val_min) & (targets_arr <= val_max)
        elif out_of_range == "ignore_sloppy" and vals.size >= 2:
            lower_half = 0.5 * abs(vals[1] - vals[0])
            upper_half = 0.5 * abs(vals[-1] - vals[-2])
            valid = (targets_arr >= val_min - lower_half) & (
                targets_arr <= val_max + upper_half
            )
        elif out_of_range == "ignore_sloppy" and vals.size == 1:
            valid = np.isclose(targets_arr, val_min)
        else:
            valid = np.ones_like(targets_arr, dtype=bool)
    else:
        raise ValueError(
            "out_of_range must be one of {'ignore', 'ignore_sloppy', 'clip', 'error'}"
        )

    idx_right = np.searchsorted(vals, targets_arr, side="left")
    idx_left = np.clip(idx_right - 1, 0, vals.size - 1)
    idx_right = np.clip(idx_right, 0, vals.size - 1)

    left = vals[idx_left]
    right = vals[idx_right]

    choose_right = np.abs(right - targets_arr) <= np.abs(targets_arr - left)
    out = np.where(choose_right, idx_right, idx_left)

    if not ascending:
        out = (vals.size - 1) - out

    out = out.astype(np.int64, copy=False)

    if out_of_range == "clip":
        if return_valid_mask:
            return out.reshape(targets.shape), np.ones_like(valid, dtype=bool).reshape(
                targets.shape
            )
        return out.reshape(targets.shape)

    if out_of_range in ("ignore", "ignore_sloppy"):
        if return_valid_mask:
            return out.reshape(targets.shape), valid.reshape(targets.shape)
        return out.reshape(targets.shape)


def _infer_handedness(
    x_vals: np.ndarray, y_vals: np.ndarray
) -> Literal["left", "right"]:
    """Infer grid handedness from 1-D rectilinear coordinates.

    Returns "left" if the grid is left-handed (dx * dy < 0), otherwise "right".
    Coordinates must be strictly monotonic along each axis.
    """
    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)

    dx = np.diff(x_vals)
    dy = np.diff(y_vals)

    if dx.size == 0 or dy.size == 0:
        raise ValueError("x and y coordinates must each have length >= 2.")

    if not ((dx > 0).all() or (dx < 0).all()):
        raise ValueError("x coordinates must be strictly monotonic for angle='auto'.")
    if not ((dy > 0).all() or (dy < 0).all()):
        raise ValueError("y coordinates must be strictly monotonic for angle='auto'.")

    return "left" if dx[0] * dy[0] < 0 else "right"


def _normalize_theta(
    angle_value: float, *, angle: Literal["pa", "math"], degrees: bool
) -> float:
    """Convert a user-provided angle to the internal math convention in radians.

    "math" is measured from +x toward +y (CCW).
    "pa"   is astronomical position angle, measured from +y toward +x (North→East).

    Relation: theta_math = (pi/2) - PA.
    """
    theta = float(np.deg2rad(angle_value) if degrees else angle_value)
    return (np.pi / 2.0) - theta if angle == "pa" else theta


def _validate_ab_theta_center(
    a: float,
    b: float,
    theta: float,
    x0: float,
    y0: float,
) -> None:
    """
    Validate that shape parameters are finite and positive where required.

    Parameters
    ----------
    a, b
        Semi-axis lengths or width-like parameters; must be positive finite numbers.
    theta
        Rotation angle in radians; must be finite.
    x0, y0
        Center coordinates in world units; must be finite.

    Raises
    ------
    ValueError
        If any check fails.
    """
    # Check finiteness
    for value, name in [
        (a, "'a'"),
        (b, "'b'"),
        (theta, "'theta'"),
        (x0, "'x0'"),
        (y0, "'y0'"),
    ]:
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

    # Check positivity for a and b
    for value, name in [
        (a, "'a'"),
        (b, "'b'"),
    ]:
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")


def _copy_meta(src: xr.DataArray, dest: xr.DataArray) -> xr.DataArray:
    """
    Copy name and attributes from one DataArray to another.

    Parameters
    ----------
    src :
        Source DataArray to copy metadata from.
    dest :
        Destination DataArray to copy metadata to.

    Returns
    -------
    xarray.DataArray
        The destination DataArray with updated metadata.
    """
    dest = dest.assign_attrs(src.attrs)
    dest.name = getattr(src, "name", None)
    return dest


def _apply_source_array(
    xda_in: xr.DataArray, source_array: xr.DataArray, *, add: bool
) -> xr.DataArray:
    """
    Apply a generated source array to an input DataArray.

    Parameters
    ----------
    xda_in :
        The existing DataArray to be modified.
    source_array :
        The generated values to insert/add. Must be same shape as xda_in.
    add :
        If True, add source_array to xda_in where non-zero.
        If False, replace values in xda_in where source_array is non-zero.

    Returns
    -------
    xarray.DataArray
        Modified DataArray with source array applied.
    """
    if add:
        return xda_in + source_array
    return xda_in.where(source_array == 0, other=source_array)


def _finalize_output(
    xda_out: xr.DataArray,
    input_obj: Union[xr.DataArray, ArrayLike2D],
    *,
    output: OutputKind,
) -> ReturnType:
    """
    Convert the internal xarray result to the requested output kind.

    If ``output='match'``, the output kind is chosen to match the input object
    type:

      - DataArray → ``'xarray'``
      - Dask array → ``'dask'``
      - NumPy array → ``'numpy'``

    Returns either the DataArray or a plain array view, computing Dask to NumPy
    if needed, or wrapping NumPy as Dask on request.
    """
    if output == "match":
        if isinstance(input_obj, xr.DataArray):
            target = "xarray"
        elif _is_dask_array(input_obj):
            target = "dask"
        else:
            target = "numpy"
    else:
        target = output

    if target == "xarray":
        return xda_out

    data = xda_out.data

    if target == "numpy":
        if _is_dask_array(data):
            return np.asarray(data.compute())
        return np.asarray(data)

    if target == "dask":
        if _is_dask_array(data):
            return data
        return da.from_array(np.asarray(data), chunks="auto")

    raise ValueError("output must be one of {'match', 'xarray', 'numpy', 'dask'}")


def make_disk(
    data: Union[xr.DataArray, ArrayLike2D],
    a: float,
    b: float,
    theta: float,
    x0: float,
    y0: float,
    height: Real,
    *,
    x_coord: str = "x",
    y_coord: str = "y",
    coords: Optional[Mapping[str, np.ndarray]] = None,
    dims: Optional[Tuple[str, str]] = None,
    add: bool = True,
    output: OutputKind = "match",
    angle: Literal["auto", "pa", "math"] = "auto",
    degrees: bool = False,
) -> ReturnType:
    """
    Fill a rotated ellipse (“disk”) with a constant value on a world-coordinate grid.

    ``make_disk`` writes a constant value ``height`` inside an ellipse defined in world
    coordinates by its semi-axes ``a`` and ``b``, rotation ``theta`` (radians,
    measured from +x toward +y), and center ``(x0, y0)``. The function accepts
    either an ``xarray.DataArray`` of any dimensionality that includes named
    ``x_coord`` and ``y_coord`` dims, or a 2-D NumPy/Dask array plus 1-D
    coordinate arrays via ``coords``. All coordinates are interpreted as world
    coordinates.

    Behavior controlled by ``add``:
      - ``add=True`` (default): **Additive** mode. ``height`` is added to the
        existing values **only** inside the ellipse; other locations are
        unchanged.
      - ``add=False``: **Replacement** mode. Values inside the ellipse are
        replaced with ``A``; other locations keep their original values.

    The ``output`` parameter controls the return type:
      - ``'match'`` returns the same kind as the input (DataArray, NumPy, or Dask).
      - ``'xarray'``, ``'numpy'``, or ``'dask'`` force that kind.

    Parameters
    ----------
    data
        Input field. If a DataArray, it must include dims named ``x_coord`` and
        ``y_coord``. If a NumPy/Dask array, it must be 2-D and you must pass
        ``coords``.
    a, b
        Semi-major and semi-minor axis lengths. Must be positive finite numbers.
    theta
        Rotation angle in radians measured from +x toward +y.
    x0, y0
        Ellipse center in world coordinates.
    height
        Value to write inside the ellipse, or the increment when ``add=True``.
    x_coord, y_coord
        Names of the horizontal and vertical coordinates or dims.
    coords
        Required when ``data`` is a NumPy/Dask array. Mapping containing 1-D
        arrays for ``x_coord`` and ``y_coord`` whose lengths match the array.
    dims
        Optional when ``data`` is a NumPy/Dask array. Defaults to
        ``(y_coord, x_coord)``.
    add
        Defaults to ``True``. If ``True``, add to values inside the ellipse;
        if ``False``, replace values inside the ellipse.
    output
        Output kind to return. One of ``'match'``, ``'xarray'``, ``'numpy'``,
        or ``'dask'``.
    angle
        Controls how ``theta`` is interpreted.

        - ``"auto"`` infers handedness from the 1-D coords. If left-handed
          (``dx * dy < 0``), interpret ``theta`` as position angle (PA, +y→+x).
          If right-handed, interpret as math angle (+x→+y, CCW).
        - ``"pa"`` forces position angle interpretation (North→East).
        - ``"math"`` forces the standard math convention (+x→+y, CCW).
    degrees
        If ``True``, ``theta`` is provided in degrees. If ``False``, radians.


    Returns
    -------
    xarray.DataArray | numpy.ndarray | dask.array.Array
        The field with the disk applied, in the requested output kind.

    Raises
    ------
    TypeError
        If ``data`` is not a supported type.
    ValueError
        If coordinates are missing or inconsistent, or parameters are invalid.

    Notes
    -----
    For DataArray inputs with extra dims (for example, ``("time", "y", "x")``),
    the 2-D mask broadcasts across the remaining dims. Dask inputs remain lazy.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> y = np.linspace(-5, 5, 101)
    >>> x = np.linspace(-5, 5, 121)
    >>> base = xr.DataArray(np.zeros((y.size, x.size)), coords={"y": y, "x": x}, dims=("y", "x"))
    >>> out = make_disk(base, a=3.0, b=1.5, theta=np.deg2rad(30), x0=0.0, y0=0.0, height=2.0)
    """
    _validate_ab_theta_center(a, b, theta, x0, y0)

    xda_in = _coerce_to_xda(
        data, x_coord=x_coord, y_coord=y_coord, coords=coords, dims=dims
    )
    x_vals = np.asarray(xda_in[x_coord].values)
    y_vals = np.asarray(xda_in[y_coord].values)

    if angle == "auto":
        handed = _infer_handedness(x_vals, y_vals)
        angle_mode: Literal["pa", "math"] = "pa" if handed == "left" else "math"
    else:
        angle_mode = angle

    theta_eff = _normalize_theta(theta, angle=angle_mode, degrees=degrees)

    xp, yp = _rotated_coords(
        xda_in, x_coord=x_coord, y_coord=y_coord, x0=x0, y0=y0, theta=theta_eff
    )
    mask = (xp / a) ** 2 + (yp / b) ** 2 <= 1.0
    source_array = xr.where(mask, height, 0)
    xda_out = _apply_source_array(xda_in, source_array, add=add)
    xda_out = _copy_meta(xda_in, xda_out)
    return _finalize_output(xda_out, data, output=output)


def make_gauss2d(
    data: Union[xr.DataArray, ArrayLike2D],
    a: float,
    b: float,
    theta: float,
    x0: float,
    y0: float,
    peak: Real,
    *,
    x_coord: str = "x",
    y_coord: str = "y",
    coords: Optional[Mapping[str, np.ndarray]] = None,
    dims: Optional[Tuple[str, str]] = None,
    add: bool = True,
    output: OutputKind = "match",
    angle: Literal["auto", "pa", "math"] = "auto",
    degrees: bool = False,
) -> ReturnType:
    """
    Generate or add a rotated elliptical 2-D Gaussian using **FWHM** parameters.

    ``make_gauss2d`` produces an elliptical Gaussian with peak amplitude ``peak`` at
    center ``(x0, y0)``. Inputs ``a`` and ``b`` are the **full width at half
    maximum (FWHM)** along the ellipse’s principal axes. The ellipse is rotated
    by ``theta`` radians measured from +x toward +y.

    Conversion from FWHM to standard deviation:

        ``sigma = FWHM / (2 * sqrt(2 * ln(2)))``

    Field definition:

        ``G(x, y) = peak * exp(-0.5 * [ (xp/σx)^2 + (yp/σy)^2 ])``

    where ``xp, yp`` are the rotated coordinates about ``(x0, y0)``.

    Behavior controlled by ``add``:
      - ``add=True`` (default): **Additive** mode. The Gaussian is added to the
        existing values.
      - ``add=False``: **Replacement** mode. The output equals the Gaussian
        everywhere and replaces any existing values.

    The ``output`` parameter controls the return type:
      - ``'match'`` returns the same kind as the input (DataArray, NumPy, or Dask).
      - ``'xarray'``, ``'numpy'``, or ``'dask'`` force that kind.

    Parameters
    ----------
    data
        Input field. If a DataArray, it must include dims named ``x_coord`` and
        ``y_coord``. If a NumPy/Dask array, it must be 2-D and you must pass
        ``coords``.
    a, b
        FWHM along the semi-major and semi-minor axes. Must be positive finite.
    theta
        Rotation angle in radians measured from +x toward +y.
    x0, y0
        Gaussian center in world coordinates.
    peak
        Peak amplitude at the center.
    x_coord, y_coord
        Names of the horizontal and vertical coordinates or dims.
    coords
        Required when ``data`` is a NumPy/Dask array. Mapping containing 1-D
        arrays for ``x_coord`` and ``y_coord`` whose lengths match the array.
    dims
        Optional when ``data`` is a NumPy/Dask array. Defaults to
        ``(y_coord, x_coord)``.
    add
        Defaults to ``True``. If ``True``, add the Gaussian to existing values;
        if ``False``, replace by the Gaussian everywhere.
    output
        Output kind to return. One of ``'match'``, ``'xarray'``, ``'numpy'``,
        or ``'dask'``.
    angle
        Controls how ``theta`` is interpreted.

        - ``"auto"`` infers handedness from the 1-D coords. If left-handed
          (``dx * dy < 0``), interpret ``theta`` as position angle (PA, +y→+x).
          If right-handed, interpret as math angle (+x→+y, CCW).
        - ``"pa"`` forces position angle interpretation (North→East).
        - ``"math"`` forces the standard math convention (+x→+y, CCW).
    degrees
        If ``True``, ``theta`` is provided in degrees. If ``False``, radians.


    Returns
    -------
    xarray.DataArray | numpy.ndarray | dask.array.Array
        The field with the Gaussian applied, in the requested output kind.

    Raises
    ------
    TypeError
        If ``data`` is not a supported type.
    ValueError
        If coordinates are missing or inconsistent, or parameters are invalid.

    Notes
    -----
    Works lazily with Dask arrays. For inputs with extra dims, the 2-D Gaussian
    broadcasts across the remaining dims.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> y = np.linspace(-4, 4, 200)
    >>> x = np.linspace(-5, 5, 300)
    >>> base = xr.DataArray(np.zeros((y.size, x.size)), coords={"y": y, "x": x}, dims=("y", "x"))
    >>> g = make_gauss2d(base, a=2.355, b=4.71, theta=np.deg2rad(30), x0=0.0, y0=0.0, peak=10.0)
    """
    _validate_ab_theta_center(a, b, theta, x0, y0)

    xda_in = _coerce_to_xda(
        data, x_coord=x_coord, y_coord=y_coord, coords=coords, dims=dims
    )
    x_vals = np.asarray(xda_in[x_coord].values)
    y_vals = np.asarray(xda_in[y_coord].values)

    if angle == "auto":
        handed = _infer_handedness(x_vals, y_vals)
        angle_mode: Literal["pa", "math"] = "pa" if handed == "left" else "math"
    else:
        angle_mode = angle

    theta_eff = _normalize_theta(theta, angle=angle_mode, degrees=degrees)

    xp, yp = _rotated_coords(
        xda_in, x_coord=x_coord, y_coord=y_coord, x0=x0, y0=y0, theta=theta_eff
    )

    denom = 2.0 * np.sqrt(2.0 * np.log(2.0))
    sigma_x = a / denom
    sigma_y = b / denom

    source_array = peak * np.exp(-0.5 * ((xp / sigma_x) ** 2 + (yp / sigma_y) ** 2))
    xda_out = _apply_source_array(xda_in, source_array, add=add)
    xda_out = _copy_meta(xda_in, xda_out)
    return _finalize_output(xda_out, data, output=output)


def make_pt_sources(
    data: Union[xr.DataArray, ArrayLike2D],
    amplitudes: Sequence[Real],
    xs: Union[np.ndarray, list, tuple],
    ys: Union[np.ndarray, list, tuple],
    *,
    x_coord: str = "x",
    y_coord: str = "y",
    coords: Optional[Mapping[str, np.ndarray]] = None,
    dims: Optional[Tuple[str, str]] = None,
    add: bool = True,
    output: OutputKind = "match",
    out_of_range: Literal["ignore", "ignore_sloppy", "clip", "error"] = "ignore",
) -> ReturnType:
    """
    Place a collection of point sources on a world-coordinate grid.

    Each source is defined by an amplitude and a position ``(x, y)`` expressed in
    the same world units as the grid coordinates. Sources are mapped to the nearest
    grid point along each axis in physical distance. If a target lies exactly midway
    between two coordinates along an axis, the right-hand coordinate is chosen
    deterministically.

    Duplicate hits are handled correctly and efficiently. If multiple sources map
    to the same grid point, their amplitudes are summed in one pass using a
    linear-index accumulation strategy.

    Out-of-range handling is controlled by ``out_of_range``:

      - ``"ignore"`` (default).
        Strict ignore. Sources whose x or y fall outside the closed coordinate
        range are skipped entirely. Use this to drop true out-of-coverage positions.

      - ``"ignore_sloppy"``.
        Half-pixel–tolerant ignore. Treat a source as in-range if it lands within
        a half pixel beyond either edge, where the half-pixel is computed from the
        local spacing at that edge. This is helpful for sources that are only just
        outside coverage because of rounding.

      - ``"clip"``.
        Clip to the nearest valid pixel. Sources outside the coordinate range are
        mapped to the closest edge pixel and included in the accumulation.

      - ``"error"``.
        Raise ``ValueError`` if any source coordinate is outside the coordinate range.

    Behavior controlled by ``add``:

      - ``add=True`` (default). Additive mode. Point-source amplitudes are added
        to the existing values at those pixels.

      - ``add=False``. Replacement mode. Point-source amplitudes replace the
        existing values at those pixels; other locations remain unchanged.

    The ``output`` parameter controls the return type:

      - ``'match'`` returns the same kind as the input (DataArray, NumPy, or Dask).
      - ``'xarray'``, ``'numpy'``, or ``'dask'`` force that kind.

    Parameters
    ----------
    data
        Input field. If a DataArray, it must include dims named ``x_coord`` and
        ``y_coord``. If a NumPy/Dask array, it must be 2-D and you must pass
        ``coords``.
    amplitudes
        Sequence of amplitudes, one per source.
    xs, ys
        Sequences of x and y world coordinates, one per source. Must be the same
        length as ``amplitudes``.
    x_coord, y_coord
        Names of the horizontal and vertical coordinates or dims.
    coords
        Required when ``data`` is a NumPy/Dask array. Mapping containing 1-D
        arrays for ``x_coord`` and ``y_coord`` whose lengths match the array.
    dims
        Optional when ``data`` is a NumPy/Dask array. Defaults to
        ``(y_coord, x_coord)``.
    add
        Defaults to ``True``. If ``True``, add to existing values at those
        pixels; if ``False``, replace values at those pixels.
    output
        Output kind to return. One of ``'match'``, ``'xarray'``, ``'numpy'``,
        or ``'dask'``.
    out_of_range
        One of ``{"ignore", "ignore_sloppy", "clip", "error"}`` controlling how
        sources outside the coordinate range are handled.

    Returns
    -------
    xarray.DataArray | numpy.ndarray | dask.array.Array
        The field with point sources applied, in the requested output kind.

    Raises
    ------
    ValueError
        If the lengths of ``amplitudes``, ``xs``, and ``ys`` are not equal, or
        if ``out_of_range='error'`` and any source lies outside the coordinate range.

    Notes
    -----
    For rectilinear yet irregular grids, the nearest 2-D pixel is the Cartesian
    pair of the nearest 1-D coordinates along x and y. This allows a fast,
    per-axis nearest search to be combined into a correct 2-D decision.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> y = np.linspace(-2, 2, 5)
    >>> x = np.linspace(-3, 3, 7)
    >>> base = xr.DataArray(np.zeros((y.size, x.size)),
    ...                     coords={"y": y, "x": x},
    ...                     dims=("y", "x"))
    >>> # Strict ignore (default): OOR sources are dropped
    >>> out1 = make_pt_sources(base,
    ...     amplitudes=[5.0, 3.0],
    ...     xs=[-0.1, 1.4],
    ...     ys=[0.2, -0.9],
    ...     out_of_range="ignore")
    >>> # Half-pixel–tolerant ignore: keep near-edge sources
    >>> out2 = make_pt_sources(base,
    ...     amplitudes=[5.0],
    ...     xs=[-3.0 - (x[1]-x[0])/2 + 1e-6],
    ...     ys=[0.0],
    ...     out_of_range="ignore_sloppy")

    """
    amps = np.asarray(amplitudes)
    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)

    if not (amps.size == xs_arr.size == ys_arr.size):
        raise ValueError("amplitudes, xs, and ys must have the same length")

    xda_in = _coerce_to_xda(
        data, x_coord=x_coord, y_coord=y_coord, coords=coords, dims=dims
    )

    x_vals = np.asarray(xda_in[x_coord].values)
    y_vals = np.asarray(xda_in[y_coord].values)

    # Request validity masks so we can truly ignore OOR targets when desired.
    xi, valid_x = _nearest_indices_1d(
        x_vals, xs_arr, out_of_range=out_of_range, return_valid_mask=True
    )
    yi, valid_y = _nearest_indices_1d(
        y_vals, ys_arr, out_of_range=out_of_range, return_valid_mask=True
    )

    # Combine masks.
    # For "clip", both valid_x and valid_y will be True everywhere.
    # For "ignore" and "ignore_sloppy", only sources within policy range are kept.
    valid = valid_x & valid_y

    # Short-circuit if nothing to add.
    if not np.any(valid):
        out_dtype = np.result_type(
            xda_in.values, amps.dtype if hasattr(amps, "dtype") else amps
        )
        source_array = xr.zeros_like(xda_in, dtype=out_dtype)
        xda_out = _apply_source_array(xda_in, source_array, add=add)
        xda_out = _copy_meta(xda_in, xda_out)
        return _finalize_output(xda_out, data, output=output)

    xi = np.asarray(xi)[valid]
    yi = np.asarray(yi)[valid]
    amps_kept = amps[valid]

    out_dtype = np.result_type(xda_in.values, amps_kept)

    # Dimensions
    height = xda_in.sizes[y_coord]
    width = xda_in.sizes[x_coord]

    # Accumulate amplitudes via bincount
    lin = yi * width + xi
    acc = (
        np.bincount(
            lin,
            weights=amps_kept.astype(out_dtype, copy=False),
            minlength=height * width,
        )
        .reshape(height, width)
        .astype(out_dtype, copy=False)
    )

    # Preserve Dask laziness if input was Dask-backed
    if _is_dask_array(xda_in.data):
        src_data = da.from_array(acc, chunks=xda_in.data.chunks)
    else:
        src_data = acc

    source_array = xr.DataArray(
        src_data,
        coords={y_coord: xda_in[y_coord], x_coord: xda_in[x_coord]},
        dims=(y_coord, x_coord),
    )

    xda_out = _apply_source_array(xda_in, source_array, add=add)
    xda_out = _copy_meta(xda_in, xda_out)
    return _finalize_output(xda_out, data, output=output)
