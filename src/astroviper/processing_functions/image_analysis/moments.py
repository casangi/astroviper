"""Image moments along a chosen axis (the *immoments* algorithm).

This is the science (processing-function) layer: a pure, stateless function
that collapses an in-memory image chunk along one axis into a set of moment
maps (integrated intensity, intensity-weighted coordinate, dispersion, median,
extrema, ...) matching the definitions of CASA's ``immoments`` task.  It
performs **no** I/O and **no** Dask/graph work (see ``AGENTS.md``).  Disk reads
and the parallel chunk writes are handled by the node-task layer
(:func:`astroviper.node_tasks.image_analysis.moments`) and the graph driver
(:func:`astroviper.distributed_applications.image_analysis.moments`),
mirroring the ``feather`` / ``image_cube_single_field`` layering.

Memory model (the critical design constraint)
---------------------------------------------
The moment axis can never be chunked (every moment needs the full axis), so
the arrays here can be large.  All single-pass moments are therefore computed
by **streaming plane-by-plane along the moment axis** with accumulators the
size of a single output map -- no full-cube temporaries are allocated.  The
only exceptions are the ``median``-family moments: ``median`` inherently needs
the full axis at once (``numpy`` partitions a copy), costing roughly one extra
copy of the chunk (two when pixel filtering / masking is active).  The
distributed application accounts for this in its chunk-size calculation.

AstroVIPER nomenclature is used throughout: the moment axis is one of the
image dimensions ``l`` (CASA *ra*), ``m`` (CASA *dec*), ``frequency`` (CASA
*spectral*), ``polarization`` (CASA *stokes*) or ``time``.
"""

import copy
import warnings

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import modify_data_groups_xds
from astroviper.utils.param_docs import shares_param_docs

# Ordered canonical moment names.  The list order fixes the order in which the
# output data variables are created.
MOMENT_NAMES = [
    "mean",
    "integrated",
    "weighted_coord",
    "weighted_dispersion_coord",
    "median",
    "median_coord",
    "standard_deviation",
    "rms",
    "abs_mean_dev",
    "maximum",
    "maximum_coord",
    "minimum",
    "minimum_coord",
]

# CASA ``immoments`` integer codes -> canonical AstroVIPER moment names.
CASA_MOMENT_CODES = {
    -1: "mean",
    0: "integrated",
    1: "weighted_coord",
    2: "weighted_dispersion_coord",
    3: "median",
    4: "median_coord",
    5: "standard_deviation",
    6: "rms",
    7: "abs_mean_dev",
    8: "maximum",
    9: "maximum_coord",
    10: "minimum",
    11: "minimum_coord",
}

# Moments whose result is a coordinate value (stored double precision) rather
# than an image value (stored at the input image precision).
COORDINATE_VALUED_MOMENTS = {
    "weighted_coord",
    "weighted_dispersion_coord",
    "median_coord",
    "maximum_coord",
    "minimum_coord",
}

# Axes a moment may be taken over (AstroVIPER names; CASA equivalents are
# ra->l, dec->m, spectral->frequency, stokes->polarization).
ALLOWED_MOMENT_AXES = ("l", "m", "frequency", "polarization", "time")


def normalize_moments(moments) -> list:
    """Normalize a moments selection to canonical AstroVIPER moment names.

    Parameters
    ----------
    moments : str, int or sequence of str/int
        Moments to compute, as canonical names (see :data:`MOMENT_NAMES`)
        and/or CASA ``immoments`` integer codes (see
        :data:`CASA_MOMENT_CODES`).  A single name/code is also accepted.

    Returns
    -------
    list of str
        De-duplicated canonical moment names, in first-occurrence order.

    Raises
    ------
    ValueError
        If a moment name/code is unknown or the selection is empty.
    """
    if isinstance(moments, str | int | np.integer):
        moments = [moments]
    normalized = []
    for moment in moments:
        if isinstance(moment, int | np.integer) and not isinstance(moment, bool):
            if int(moment) not in CASA_MOMENT_CODES:
                raise ValueError(
                    f"Unknown CASA moment code {moment}; allowed codes are "
                    f"{sorted(CASA_MOMENT_CODES)}."
                )
            name = CASA_MOMENT_CODES[int(moment)]
        elif isinstance(moment, str):
            if moment not in MOMENT_NAMES:
                raise ValueError(
                    f"Unknown moment '{moment}'; allowed names are {MOMENT_NAMES}."
                )
            name = moment
        else:
            raise ValueError(
                f"Moments must be names (str) or CASA integer codes, got {moment!r}."
            )
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("At least one moment must be requested.")
    return normalized


def normalize_pixel_range(pixel_range, parameter_name: str):
    """Normalize an include/exclude pixel range to a ``(low, high)`` tuple.

    Follows the CASA convention: a single value ``b`` means the symmetric
    range ``(-abs(b), abs(b))``; two values are ``(low, high)``.

    Parameters
    ----------
    pixel_range : None, float or sequence of float
        The range specification.  ``None`` disables the range.
    parameter_name : str
        Name used in error messages (``"include_pixel_range"`` or
        ``"exclude_pixel_range"``).

    Returns
    -------
    tuple of float or None
        ``(low, high)`` with ``low <= high``, or ``None``.

    Raises
    ------
    ValueError
        If more than two values are given or ``low > high``.
    """
    if pixel_range is None:
        return None
    if isinstance(pixel_range, int | float | np.floating | np.integer):
        pixel_range = [pixel_range]
    pixel_range = [float(value) for value in pixel_range]
    if len(pixel_range) == 1:
        low, high = -abs(pixel_range[0]), abs(pixel_range[0])
    elif len(pixel_range) == 2:
        low, high = pixel_range
    else:
        raise ValueError(
            f"{parameter_name} must contain one or two values, got {pixel_range}."
        )
    if low > high:
        raise ValueError(
            f"{parameter_name} low value {low} is greater than high value {high}."
        )
    return (low, high)


def moment_data_variable_key(moment_name: str) -> str:
    """Return the lowercase data-variable key for a moment (``sky_moment_<name>``)."""
    return "sky_moment_" + moment_name


def get_moments_data_variable_definitions(
    moment_names, dims, single_precision_image: bool
) -> dict:
    """Build the on-disk data-variable definitions for the requested moments.

    Used by the distributed application together with
    :func:`astroviper.utils.io.create_empty_data_variables_on_disk` to
    pre-allocate the output Zarr arrays.

    Parameters
    ----------
    moment_names : list of str
        Canonical moment names (see :func:`normalize_moments`).
    dims : sequence of str
        Dimension names of the sky image, in order (the moment axis is kept as
        a degenerate dimension of size 1 in the output).
    single_precision_image : bool
        If ``True`` image-valued moments are stored as ``float32``, otherwise
        ``float64``.  Coordinate-valued moments are always ``float64`` so
        coordinate values (e.g. frequencies in Hz) do not lose precision.

    Returns
    -------
    dict
        ``{variable_key: {"dims", "dtype", "name"}}`` in the style of
        ``astroviper.utils.io.imaging_data_variables_and_dims_*``.
    """
    value_dtype = "<f4" if single_precision_image else "<f8"
    definitions = {}
    for name in moment_names:
        dtype = "<f8" if name in COORDINATE_VALUED_MOMENTS else value_dtype
        key = moment_data_variable_key(name)
        definitions[key] = {
            "dims": list(dims),
            "dtype": dtype,
            "name": key.upper(),
        }
    return definitions


def resolve_moments_input_variables(
    img_xds: xr.Dataset, image_data_group_in_name: str, use_mask: bool
):
    """Resolve the sky (and optional mask) data-variable names from a data group.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image dataset (or a lazily opened view of one).
    image_data_group_in_name : str
        Key in ``img_xds.attrs["data_groups"]``.  If the dataset carries no
        data groups at all, the conventional ``"SKY"`` variable is used as a
        fallback.  A group without a ``"sky"`` role likewise falls back to
        ``"SKY"``.
    use_mask : bool
        If ``True``, also resolve the group's ``"mask"`` role (``None`` when
        the group defines no mask).

    Returns
    -------
    tuple of (str, str or None)
        ``(sky_variable_name, mask_variable_name)``.

    Raises
    ------
    AssertionError
        If the data group does not exist (when data groups are present) or the
        resolved sky variable is not in the dataset.
    """
    data_groups = img_xds.attrs.get("data_groups", {})
    if data_groups:
        assert image_data_group_in_name in data_groups, (
            "Data group "
            + image_data_group_in_name
            + " not found in image data_groups: "
            + str(list(data_groups.keys()))
        )
        data_group_in = data_groups[image_data_group_in_name]
    else:
        data_group_in = {}
    sky_name = data_group_in.get("sky", "SKY")
    assert sky_name in img_xds.data_vars, (
        "Sky data variable " + sky_name + " not found in image dataset."
    )
    mask_name = data_group_in.get("mask") if use_mask else None
    if mask_name is not None:
        assert mask_name in img_xds.data_vars, (
            "Mask data variable " + mask_name + " not found in image dataset."
        )
    return sky_name, mask_name


def _moment_axis_values(img_xds: xr.Dataset, moment_axis: str) -> np.ndarray:
    """Return the numeric coordinate values of the moment axis as ``float64``.

    Non-numeric coordinates (e.g. the ``polarization`` string labels) and a
    missing coordinate fall back to the plane index ``0..n-1``.
    """
    n = img_xds.sizes[moment_axis]
    if moment_axis in img_xds.coords and np.issubdtype(
        img_xds.coords[moment_axis].dtype, np.number
    ):
        return img_xds.coords[moment_axis].values.astype(np.float64)
    return np.arange(n, dtype=np.float64)


def moment_axis_units(img_xds: xr.Dataset, moment_axis: str) -> str:
    """Best-effort units string of the moment-axis coordinate (else ``"pixel"``)."""
    if moment_axis in ("l", "m"):
        return "rad"
    if moment_axis in img_xds.coords:
        units = img_xds.coords[moment_axis].attrs.get("units")
        if isinstance(units, str) and units:
            return units
    return "pixel"


def moment_units(moment_name: str, sky_units: str, axis_units: str) -> str:
    """Units string of one moment map given the sky and moment-axis units."""
    if moment_name == "integrated":
        return f"{sky_units}.{axis_units}" if sky_units else axis_units
    if moment_name in COORDINATE_VALUED_MOMENTS:
        return axis_units
    return sky_units


def collapsed_moment_axis_coords(img_xds: xr.Dataset, moment_axis: str) -> xr.Dataset:
    """Build the output coordinates with ``moment_axis`` collapsed to size 1.

    Numeric coordinates spanning the moment axis are replaced by their mean
    (the reference value of the collapsed axis, e.g. the mid frequency);
    non-numeric coordinates keep their first entry.
    """
    coords_xds = xr.Dataset(coords=img_xds.coords)
    collapsed = coords_xds.isel({moment_axis: slice(0, 1)})
    for name, coord in coords_xds.coords.items():
        if moment_axis in coord.dims and np.issubdtype(coord.dtype, np.number):
            axis_index = coord.dims.index(moment_axis)
            mean_values = np.nanmean(coord.values, axis=axis_index, keepdims=True)
            attrs = coord.attrs
            collapsed = collapsed.assign_coords({name: (coord.dims, mean_values)})
            collapsed[name].attrs = attrs
    return collapsed


@shares_param_docs
def moments(
    img_xds: xr.Dataset,
    moments=["integrated"],  # noqa: B006 - mirrors the distributed application signature; never mutated
    moment_axis: str = "frequency",
    image_data_group_in_name: str = "base",
    include_pixel_range=None,
    exclude_pixel_range=None,
    use_mask: bool = False,
) -> xr.Dataset:
    """Collapse an image along one axis into moment maps (CASA ``immoments``).

    All requested moments are computed in one streaming pass along the moment
    axis (a second cheap pass is used for ``abs_mean_dev`` and
    ``median_coord``), with accumulators the size of a single output map.
    Excluded pixels -- NaNs, pixels outside ``include_pixel_range`` / inside
    ``exclude_pixel_range``, and pixels where the mask is ``False`` -- do not
    contribute to any moment; output pixels with no contributing planes are
    NaN.

    Parameters
    ----------
    img_xds : xarray.Dataset
        In-memory (NumPy-backed) image dataset holding the sky data variable
        (and optionally a mask) referenced by the input data group.  Not
        modified.
    moments : list of str or int, default ["integrated"]
        The moments to compute, as canonical names and/or CASA ``immoments``
        integer codes:

        - ``"mean"`` (CASA ``-1``) : mean value of the profile.
        - ``"integrated"`` (``0``) : integrated value ``sum(I * delta_v)``
          with ``delta_v`` the per-plane moment-axis coordinate width.
        - ``"weighted_coord"`` (``1``) : intensity-weighted coordinate
          ``sum(I * v) / sum(I)`` (e.g. velocity field).  Use
          ``include_pixel_range`` to restrict to positive flux for sensible
          results.
        - ``"weighted_dispersion_coord"`` (``2``) : intensity-weighted
          coordinate dispersion ``sqrt(sum(I * v^2)/sum(I) - m1^2)``.
        - ``"median"`` (``3``) : median value of the profile.
        - ``"median_coord"`` (``4``) : coordinate at which the cumulative
          profile crosses 50% of its total (only meaningful for
          predominantly positive profiles).
        - ``"standard_deviation"`` (``5``) : standard deviation about the
          profile mean.
        - ``"rms"`` (``6``) : root mean square of the profile.
        - ``"abs_mean_dev"`` (``7``) : mean absolute deviation from the
          profile mean.
        - ``"maximum"`` (``8``) / ``"maximum_coord"`` (``9``) : maximum of
          the profile and the coordinate at which it occurs.
        - ``"minimum"`` (``10``) / ``"minimum_coord"`` (``11``) : minimum of
          the profile and the coordinate at which it occurs.
    moment_axis : str, default "frequency"
        Image dimension to collapse: ``"l"``, ``"m"``, ``"frequency"``,
        ``"polarization"`` or ``"time"`` (AstroVIPER names for CASA's *ra*,
        *dec*, *spectral*, *stokes*).  The moment axis is never used for
        parallelism.  Coordinate-valued moments are expressed in the native
        coordinate units of this axis (Hz for ``frequency``, rad for
        ``l``/``m``); a non-numeric axis (``polarization``) uses the plane
        index.
    image_data_group_in_name : str, default "base"
        Key in the image's ``data_groups`` whose ``"sky"`` (and, optionally,
        ``"mask"``) roles name the input data variables.  Datasets without
        data groups fall back to the conventional ``"SKY"`` variable.
    include_pixel_range : list of float, optional
        Only pixel values inside ``[low, high]`` contribute.  A single value
        ``b`` means ``[-abs(b), abs(b)]`` (CASA convention).  Mutually
        exclusive with ``exclude_pixel_range``.
    exclude_pixel_range : list of float, optional
        Pixel values inside ``[low, high]`` do not contribute.  Same
        conventions as ``include_pixel_range``.
    use_mask : bool, default False
        If ``True``, pixels where the input data group's mask variable is
        ``False`` are excluded (XRADIO convention: mask ``True`` = include).

    Returns
    -------
    xarray.Dataset
        New dataset with one ``SKY_MOMENT_<NAME>`` data variable per requested
        moment, the moment axis collapsed to size 1 (numeric coordinates
        replaced by their mean), and one data group ``moment_<name>`` (role
        ``"sky"``) registered per moment.

    Raises
    ------
    ValueError
        If a moment or the moment axis is unknown, both pixel ranges are
        given, or a pixel range is malformed.
    """
    moment_names = normalize_moments(moments)
    include_range = normalize_pixel_range(include_pixel_range, "include_pixel_range")
    exclude_range = normalize_pixel_range(exclude_pixel_range, "exclude_pixel_range")
    if include_range is not None and exclude_range is not None:
        raise ValueError(
            "Only one of include_pixel_range and exclude_pixel_range may be given."
        )
    if moment_axis not in ALLOWED_MOMENT_AXES:
        raise ValueError(
            f"moment_axis '{moment_axis}' not in allowed axes {ALLOWED_MOMENT_AXES}."
        )

    sky_name, mask_name = resolve_moments_input_variables(
        img_xds, image_data_group_in_name, use_mask
    )
    sky = img_xds[sky_name]
    if moment_axis not in sky.dims:
        raise ValueError(
            f"moment_axis '{moment_axis}' is not a dimension of {sky_name} "
            f"(dims: {sky.dims})."
        )

    data = sky.values
    axis = sky.dims.index(moment_axis)
    # Views with the moment axis first; no data is copied.
    data_planes = np.moveaxis(data, axis, 0)
    mask_planes = (
        np.moveaxis(img_xds[mask_name].values, axis, 0)
        if mask_name is not None
        else None
    )
    n_planes = data_planes.shape[0]
    map_shape = data_planes.shape[1:]

    coord_values = _moment_axis_values(img_xds, moment_axis)
    if n_planes > 1:
        coord_widths = np.abs(np.gradient(coord_values))
    else:
        coord_widths = np.ones(1, dtype=np.float64)
    # The coordinate-weighted sums accumulate in a shifted frame (v - v_ref) to
    # avoid catastrophic cancellation: e.g. frequencies ~1.4e9 Hz squared are
    # ~1e18 while their spread may be ~1e14, which would wipe out the variance.
    coord_reference = coord_values.mean()
    shifted_coord_values = coord_values - coord_reference

    filtering = (
        include_range is not None
        or exclude_range is not None
        or mask_planes is not None
    )

    def valid_plane(index):
        """Boolean map of pixels of plane ``index`` that contribute to the moments."""
        plane = data_planes[index]
        valid = np.isfinite(plane)
        if mask_planes is not None:
            valid &= mask_planes[index].astype(bool)
        if include_range is not None:
            valid &= (plane >= include_range[0]) & (plane <= include_range[1])
        if exclude_range is not None:
            valid &= (plane < exclude_range[0]) | (plane > exclude_range[1])
        return valid

    requested = set(moment_names)
    need_s1 = bool(
        requested
        & {
            "mean",
            "weighted_coord",
            "weighted_dispersion_coord",
            "standard_deviation",
            "abs_mean_dev",
            "median_coord",
        }
    )
    need_s2 = bool(requested & {"standard_deviation", "rms"})
    need_sv = bool(requested & {"weighted_coord", "weighted_dispersion_coord"})
    need_sv2 = "weighted_dispersion_coord" in requested
    need_integrated = "integrated" in requested
    need_max = bool(requested & {"maximum", "maximum_coord"})
    need_min = bool(requested & {"minimum", "minimum_coord"})

    count = np.zeros(map_shape, dtype=np.int64)
    s1 = np.zeros(map_shape, dtype=np.float64) if need_s1 else None
    s2 = np.zeros(map_shape, dtype=np.float64) if need_s2 else None
    sv = np.zeros(map_shape, dtype=np.float64) if need_sv else None
    sv2 = np.zeros(map_shape, dtype=np.float64) if need_sv2 else None
    integrated = np.zeros(map_shape, dtype=np.float64) if need_integrated else None
    if need_max:
        running_max = np.full(map_shape, -np.inf, dtype=np.float64)
        argmax = np.full(map_shape, -1, dtype=np.int64)
    if need_min:
        running_min = np.full(map_shape, np.inf, dtype=np.float64)
        argmin = np.full(map_shape, -1, dtype=np.int64)

    # ---- Pass 1: stream plane-by-plane along the moment axis -----------------
    # Only map-sized temporaries are allocated per plane.
    for i in range(n_planes):
        plane = data_planes[i]
        valid = valid_plane(i)
        count += valid
        values = np.where(valid, plane, 0.0).astype(np.float64, copy=False)
        if need_s1:
            s1 += values
        if need_s2:
            s2 += values * values
        if need_sv:
            sv += values * shifted_coord_values[i]
        if need_sv2:
            sv2 += values * (shifted_coord_values[i] ** 2)
        if need_integrated:
            integrated += values * coord_widths[i]
        if need_max:
            better = valid & (plane > running_max)
            running_max[better] = plane[better]
            argmax[better] = i
        if need_min:
            better = valid & (plane < running_min)
            running_min[better] = plane[better]
            argmin[better] = i

    empty = count == 0

    # ---- Pass 2 (cheap): moments that need a completed first pass ------------
    if "abs_mean_dev" in requested:
        with np.errstate(invalid="ignore", divide="ignore"):
            profile_mean = s1 / count
        sum_abs_dev = np.zeros(map_shape, dtype=np.float64)
        for i in range(n_planes):
            valid = valid_plane(i)
            deviation = np.where(valid, data_planes[i] - profile_mean, 0.0)
            sum_abs_dev += np.abs(deviation)

    if "median_coord" in requested:
        # Coordinate at which the cumulative profile crosses half its total,
        # streamed with map-sized accumulators (no full-cube cumsum).
        half_total = 0.5 * s1
        cumulative = np.zeros(map_shape, dtype=np.float64)
        median_coord_index = np.full(map_shape, -1, dtype=np.int64)
        for i in range(n_planes):
            valid = valid_plane(i)
            cumulative += np.where(valid, data_planes[i], 0.0)
            crossed = (median_coord_index < 0) & (cumulative >= half_total) & (s1 > 0)
            median_coord_index[crossed] = i

    if "median" in requested:
        if filtering:
            # One working copy of the chunk (at the input precision) with
            # excluded pixels set to NaN, applied plane-by-plane so no
            # full-size boolean cube is needed.
            if np.issubdtype(data_planes.dtype, np.floating):
                working = data_planes.copy()
            else:
                working = data_planes.astype(np.float64)
            for i in range(n_planes):
                working[i][~valid_plane(i)] = np.nan
        else:
            working = data_planes
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            median_map = np.nanmedian(working, axis=0)
        working = None

    # ---- Finalize the requested moments --------------------------------------
    value_dtype = data.dtype if data.dtype in (np.float32, np.float64) else np.float64
    results = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for name in moment_names:
            if name == "mean":
                result = s1 / count
            elif name == "integrated":
                result = integrated.copy()
            elif name == "weighted_coord":
                result = np.where(s1 != 0, coord_reference + sv / s1, np.nan)
            elif name == "weighted_dispersion_coord":
                # Shift-invariant: computed entirely in the (v - v_ref) frame.
                first_shifted = np.where(s1 != 0, sv / s1, np.nan)
                variance = (
                    np.where(s1 != 0, sv2 / s1, np.nan) - first_shifted * first_shifted
                )
                result = np.sqrt(np.where(variance >= 0, variance, np.nan))
            elif name == "median":
                result = median_map.astype(np.float64, copy=False)
            elif name == "median_coord":
                result = np.where(
                    median_coord_index >= 0,
                    coord_values[np.clip(median_coord_index, 0, n_planes - 1)],
                    np.nan,
                )
            elif name == "standard_deviation":
                variance = (s2 - count * (s1 / count) ** 2) / (count - 1)
                result = np.sqrt(np.where(variance >= 0, variance, 0.0))
                result = np.where(count > 1, result, np.nan)
            elif name == "rms":
                result = np.sqrt(s2 / count)
            elif name == "abs_mean_dev":
                result = sum_abs_dev / count
            elif name == "maximum":
                result = np.where(argmax >= 0, running_max, np.nan)
            elif name == "maximum_coord":
                result = np.where(
                    argmax >= 0, coord_values[np.clip(argmax, 0, n_planes - 1)], np.nan
                )
            elif name == "minimum":
                result = np.where(argmin >= 0, running_min, np.nan)
            elif name == "minimum_coord":
                result = np.where(
                    argmin >= 0, coord_values[np.clip(argmin, 0, n_planes - 1)], np.nan
                )
            result = np.where(empty, np.nan, result)
            if name in COORDINATE_VALUED_MOMENTS:
                results[name] = result.astype(np.float64, copy=False)
            else:
                results[name] = result.astype(value_dtype, copy=False)

    # ---- Assemble the output dataset -----------------------------------------
    moments_img_xds = xr.Dataset(
        coords=collapsed_moment_axis_coords(img_xds, moment_axis).coords
    )
    moments_img_xds.attrs = copy.deepcopy(img_xds.attrs)
    moments_img_xds.attrs["data_groups"] = {}

    sky_units = sky.attrs.get("units", "")
    axis_units = moment_axis_units(img_xds, moment_axis)
    for name in moment_names:
        variable_name = moment_data_variable_key(name).upper()
        moments_img_xds[variable_name] = xr.DataArray(
            np.expand_dims(results[name], axis), dims=sky.dims
        )
        units = moment_units(name, sky_units, axis_units)
        if units:
            moments_img_xds[variable_name].attrs["units"] = units
        modify_data_groups_xds(
            moments_img_xds,
            data_group_out_name="moment_" + name,
            data_group_out={"sky": variable_name},
            description=(
                f"Moment '{name}' of {sky_name} over the {moment_axis} axis "
                f"(immoments)."
            ),
        )
    return moments_img_xds
