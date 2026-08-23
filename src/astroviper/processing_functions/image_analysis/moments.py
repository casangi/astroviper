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
Every moment needs the full moment axis, so the arrays here can be large.
All moments except ``median`` are therefore computed by **streaming
plane-by-plane along the moment axis** with accumulators the size of a single
output map (:class:`MomentsAccumulator`): pass 1 accumulates the sums /
extrema, and ``abs_mean_dev`` / ``median_coord`` need a second, equally cheap
pass over the same planes.  The accumulator is fed either from an in-memory
cube (:func:`moments`) or -- the memory-efficient production path -- from
planes read on demand (:func:`moments_streamed`, driven by the node task),
so the per-task memory is O(output map) plus one read block, independent of
the length of the moment axis.  Pass-1 accumulators are also associative:
partial accumulators over disjoint moment-axis segments can be merged
(:meth:`MomentsAccumulator.merge`), which allows a map over moment-axis
chunks plus a reduce.  The only exception is ``median``, which inherently
needs the whole profile of every pixel at once (``numpy`` partitions a copy):
it costs roughly one extra copy of the chunk (two when pixel filtering /
masking is active) and forces the whole moment axis into memory.

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


# Moments whose accumulation needs a completed first pass (a second streaming
# pass over the same planes), and the one that needs the whole profile.
SECOND_PASS_MOMENTS = frozenset({"abs_mean_dev", "median_coord"})
FULL_PROFILE_MOMENTS = frozenset({"median"})


def moments_memory_model(moment_names) -> dict:
    """Describe how the requested moments can be computed (the planning API).

    Used by the node task and the distributed application to pick the
    streaming strategy and to size chunks.

    Parameters
    ----------
    moment_names : list of str
        Canonical moment names (see :func:`normalize_moments`).

    Returns
    -------
    dict
        ``n_passes`` (1 or 2 streaming passes over the moment axis),
        ``requires_full_profile`` (``True`` when ``median`` is requested: the
        whole moment axis of the chunk must be in memory at once) and
        ``mergeable`` (``True`` when every requested moment is a pass-1
        moment, so partial accumulators over moment-axis segments can be
        merged with :meth:`MomentsAccumulator.merge`).
    """
    requested = set(moment_names)
    second = bool(requested & SECOND_PASS_MOMENTS)
    full = bool(requested & FULL_PROFILE_MOMENTS)
    return {
        "n_passes": 2 if second else 1,
        "requires_full_profile": full,
        "mergeable": not second and not full,
    }


class MomentsAccumulator:
    """Streaming, map-sized accumulators for the moments along one axis.

    Planes are fed one at a time with :meth:`add_plane`; only output-map-sized
    state is held, never the cube. ``median`` is not supported here (it needs
    the whole profile; see :func:`moments`).

    Parameters
    ----------
    moment_names : list of str
        Canonical moment names (no ``"median"``).
    coord_values : numpy.ndarray
        Numeric moment-axis coordinate of every plane (float64).
    map_shape : tuple of int
        Shape of one output map (the sky shape with the moment axis removed).
    include_range, exclude_range : tuple of float, optional
        Normalised pixel-value ranges (see :func:`normalize_pixel_range`).

    Notes
    -----
    Pass 1 (``pass_index=0``) accumulates count / sums / extrema. If
    :func:`moments_memory_model` reports ``n_passes == 2`` the same planes must
    be fed again with ``pass_index=1`` (``abs_mean_dev`` needs the profile
    mean, ``median_coord`` the profile total). Pass-1 state of two accumulators
    over disjoint plane sets can be combined with :meth:`merge`.
    """

    def __init__(
        self,
        moment_names,
        coord_values,
        map_shape,
        include_range=None,
        exclude_range=None,
    ):
        requested = set(moment_names)
        if requested & FULL_PROFILE_MOMENTS:
            raise ValueError(
                "MomentsAccumulator cannot stream 'median' (needs the whole "
                "profile); use moments() on an in-memory chunk."
            )
        self.moment_names = list(moment_names)
        self.coord_values = np.asarray(coord_values, dtype=np.float64)
        self.n_planes = len(self.coord_values)
        if self.n_planes > 1:
            self.coord_widths = np.abs(np.gradient(self.coord_values))
        else:
            self.coord_widths = np.ones(1, dtype=np.float64)
        # The coordinate-weighted sums accumulate in a shifted frame
        # (v - v_ref) to avoid catastrophic cancellation: e.g. frequencies
        # ~1.4e9 Hz squared are ~1e18 while their spread may be ~1e14, which
        # would wipe out the variance.
        self.coord_reference = float(self.coord_values.mean())
        self.shifted_coord_values = self.coord_values - self.coord_reference
        self.include_range = include_range
        self.exclude_range = exclude_range
        self.map_shape = tuple(map_shape)

        self.need_s1 = bool(
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
        self.need_s2 = bool(requested & {"standard_deviation", "rms"})
        self.need_sv = bool(requested & {"weighted_coord", "weighted_dispersion_coord"})
        self.need_sv2 = "weighted_dispersion_coord" in requested
        self.need_integrated = "integrated" in requested
        self.need_max = bool(requested & {"maximum", "maximum_coord"})
        self.need_min = bool(requested & {"minimum", "minimum_coord"})
        self.need_abs_mean_dev = "abs_mean_dev" in requested
        self.need_median_coord = "median_coord" in requested

        shape = self.map_shape
        self.count = np.zeros(shape, dtype=np.int64)
        self.s1 = np.zeros(shape, dtype=np.float64) if self.need_s1 else None
        self.s2 = np.zeros(shape, dtype=np.float64) if self.need_s2 else None
        self.sv = np.zeros(shape, dtype=np.float64) if self.need_sv else None
        self.sv2 = np.zeros(shape, dtype=np.float64) if self.need_sv2 else None
        self.integrated = (
            np.zeros(shape, dtype=np.float64) if self.need_integrated else None
        )
        if self.need_max:
            self.running_max = np.full(shape, -np.inf, dtype=np.float64)
            self.argmax = np.full(shape, -1, dtype=np.int64)
        if self.need_min:
            self.running_min = np.full(shape, np.inf, dtype=np.float64)
            self.argmin = np.full(shape, -1, dtype=np.int64)
        # Pass-2 state (allocated lazily on the first pass-2 plane).
        self.sum_abs_dev = None
        self.profile_mean = None
        self.cumulative = None
        self.median_coord_index = None
        self.value_dtype = np.float64
        self._dtype_seen = False

    @property
    def n_passes(self) -> int:
        """Number of streaming passes the requested moments need (1 or 2)."""
        return 2 if (self.need_abs_mean_dev or self.need_median_coord) else 1

    def valid_plane(self, plane, mask_plane=None):
        """Boolean map of the pixels of ``plane`` that contribute to the moments."""
        valid = np.isfinite(plane)
        if mask_plane is not None:
            valid &= mask_plane.astype(bool)
        if self.include_range is not None:
            valid &= (plane >= self.include_range[0]) & (plane <= self.include_range[1])
        if self.exclude_range is not None:
            valid &= (plane < self.exclude_range[0]) | (plane > self.exclude_range[1])
        return valid

    def add_plane(self, index, plane, mask_plane=None, pass_index=0):
        """Accumulate one moment-axis plane (``plane.shape == map_shape``).

        Parameters
        ----------
        index : int
            Position of the plane along the moment axis (``0..n_planes-1``).
        plane : numpy.ndarray
            The plane's pixel values (any float/int dtype; NaN = excluded).
        mask_plane : numpy.ndarray, optional
            Boolean plane (``True`` = include).
        pass_index : int, default 0
            ``0`` for the first pass, ``1`` for the second pass of the
            two-pass moments.
        """
        plane = np.asarray(plane)
        if pass_index == 0 and not self._dtype_seen:
            # Output precision of image-valued moments follows the input
            # image precision (float32 stays float32, anything else float64).
            self.value_dtype = np.float32 if plane.dtype == np.float32 else np.float64
            self._dtype_seen = True
        valid = self.valid_plane(plane, mask_plane)
        if pass_index == 0:
            self._add_plane_pass1(index, plane, valid)
        else:
            self._add_plane_pass2(index, plane, valid)

    def _add_plane_pass1(self, i, plane, valid):
        self.count += valid
        values = np.where(valid, plane, 0.0).astype(np.float64, copy=False)
        if self.need_s1:
            self.s1 += values
        if self.need_s2:
            self.s2 += values * values
        if self.need_sv:
            self.sv += values * self.shifted_coord_values[i]
        if self.need_sv2:
            self.sv2 += values * (self.shifted_coord_values[i] ** 2)
        if self.need_integrated:
            self.integrated += values * self.coord_widths[i]
        if self.need_max:
            better = valid & (plane > self.running_max)
            self.running_max[better] = plane[better]
            self.argmax[better] = i
        if self.need_min:
            better = valid & (plane < self.running_min)
            self.running_min[better] = plane[better]
            self.argmin[better] = i

    def _add_plane_pass2(self, i, plane, valid):
        if self.need_abs_mean_dev:
            if self.sum_abs_dev is None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    self.profile_mean = self.s1 / self.count
                self.sum_abs_dev = np.zeros(self.map_shape, dtype=np.float64)
            deviation = np.where(valid, plane - self.profile_mean, 0.0)
            self.sum_abs_dev += np.abs(deviation)
        if self.need_median_coord:
            # Coordinate at which the cumulative profile crosses half its
            # total, streamed with map-sized accumulators (no full cumsum).
            if self.cumulative is None:
                self.cumulative = np.zeros(self.map_shape, dtype=np.float64)
                self.median_coord_index = np.full(self.map_shape, -1, dtype=np.int64)
            self.cumulative += np.where(valid, plane, 0.0)
            crossed = (
                (self.median_coord_index < 0)
                & (self.cumulative >= 0.5 * self.s1)
                & (self.s1 > 0)
            )
            self.median_coord_index[crossed] = i

    def merge(self, other):
        """Fold another accumulator's pass-1 state into this one (in place).

        Both must describe the same moments, map shape and moment-axis
        coordinate and have accumulated disjoint plane sets. Only valid for
        single-pass moments (``moments_memory_model(...)["mergeable"]``).
        """
        if self.n_passes != 1:
            raise ValueError(
                "merge() is only valid for single-pass moments (no abs_mean_dev / "
                "median_coord)."
            )
        if (
            other.moment_names != self.moment_names
            or other.map_shape != self.map_shape
            or not np.array_equal(other.coord_values, self.coord_values)
        ):
            raise ValueError("Cannot merge accumulators of different moments/shapes.")
        self.count += other.count
        for name in ("s1", "s2", "sv", "sv2", "integrated"):
            mine = getattr(self, name)
            if mine is not None:
                mine += getattr(other, name)
        if self.need_max:
            better = other.running_max > self.running_max
            self.running_max[better] = other.running_max[better]
            self.argmax[better] = other.argmax[better]
        if self.need_min:
            better = other.running_min < self.running_min
            self.running_min[better] = other.running_min[better]
            self.argmin[better] = other.argmin[better]
        if other.value_dtype == np.float64:
            self.value_dtype = np.float64
        return self

    def finalize(self) -> dict:
        """Return ``{moment_name: map}`` for the requested moments (NaN where
        no plane contributed)."""
        n_planes = self.n_planes
        coord_values = self.coord_values
        count = self.count
        empty = count == 0
        results = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for name in self.moment_names:
                if name == "mean":
                    result = self.s1 / count
                elif name == "integrated":
                    result = self.integrated.copy()
                elif name == "weighted_coord":
                    result = np.where(
                        self.s1 != 0, self.coord_reference + self.sv / self.s1, np.nan
                    )
                elif name == "weighted_dispersion_coord":
                    # Shift-invariant: computed entirely in the (v - v_ref) frame.
                    first_shifted = np.where(self.s1 != 0, self.sv / self.s1, np.nan)
                    variance = (
                        np.where(self.s1 != 0, self.sv2 / self.s1, np.nan)
                        - first_shifted * first_shifted
                    )
                    result = np.sqrt(np.where(variance >= 0, variance, np.nan))
                elif name == "median_coord":
                    index = self.median_coord_index
                    result = np.where(
                        index >= 0,
                        coord_values[np.clip(index, 0, n_planes - 1)],
                        np.nan,
                    )
                elif name == "standard_deviation":
                    variance = (self.s2 - count * (self.s1 / count) ** 2) / (count - 1)
                    result = np.sqrt(np.where(variance >= 0, variance, 0.0))
                    result = np.where(count > 1, result, np.nan)
                elif name == "rms":
                    result = np.sqrt(self.s2 / count)
                elif name == "abs_mean_dev":
                    result = self.sum_abs_dev / count
                elif name == "maximum":
                    result = np.where(self.argmax >= 0, self.running_max, np.nan)
                elif name == "maximum_coord":
                    result = np.where(
                        self.argmax >= 0,
                        coord_values[np.clip(self.argmax, 0, n_planes - 1)],
                        np.nan,
                    )
                elif name == "minimum":
                    result = np.where(self.argmin >= 0, self.running_min, np.nan)
                elif name == "minimum_coord":
                    result = np.where(
                        self.argmin >= 0,
                        coord_values[np.clip(self.argmin, 0, n_planes - 1)],
                        np.nan,
                    )
                result = np.where(empty, np.nan, result)
                if name in COORDINATE_VALUED_MOMENTS:
                    results[name] = result.astype(np.float64, copy=False)
                else:
                    results[name] = result.astype(self.value_dtype, copy=False)
        return results


def assemble_moments_dataset(
    results, coords_xds, attrs, sky_dims, sky_name, sky_attrs, moment_axis
) -> xr.Dataset:
    """Wrap finalized moment maps into the output image dataset.

    Parameters
    ----------
    results : dict
        ``{moment_name: map}`` from :meth:`MomentsAccumulator.finalize`.
    coords_xds : xarray.Dataset
        Dataset carrying the (uncollapsed) coordinates of the input chunk.
    attrs : dict
        Input dataset attributes (copied; ``data_groups`` is rebuilt).
    sky_dims : sequence of str
        Dimension names of the sky variable, in order.
    sky_name : str
        Name of the input sky variable (for the data-group description).
    sky_attrs : dict
        Attributes of the sky variable (``units`` is propagated).
    moment_axis : str
        The collapsed axis.
    """
    axis = list(sky_dims).index(moment_axis)
    moments_img_xds = xr.Dataset(
        coords=collapsed_moment_axis_coords(coords_xds, moment_axis).coords
    )
    moments_img_xds.attrs = copy.deepcopy(attrs)
    moments_img_xds.attrs["data_groups"] = {}

    sky_units = sky_attrs.get("units", "")
    axis_units = moment_axis_units(coords_xds, moment_axis)
    for name, result in results.items():
        variable_name = moment_data_variable_key(name).upper()
        moments_img_xds[variable_name] = xr.DataArray(
            np.expand_dims(result, axis), dims=list(sky_dims)
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


def _validate_moments_request(
    moments, moment_axis, include_pixel_range, exclude_pixel_range
):
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
    return moment_names, include_range, exclude_range


def moments_streamed(
    read_planes,
    coords_xds: xr.Dataset,
    sky_dims,
    sky_name: str,
    sky_attrs: dict,
    attrs: dict,
    moments=["integrated"],  # noqa: B006 - mirrors the distributed application signature; never mutated
    moment_axis: str = "frequency",
    include_pixel_range=None,
    exclude_pixel_range=None,
) -> xr.Dataset:
    """Compute the moments of a chunk whose planes are read on demand.

    The memory-efficient production path: the planes along the moment axis
    are supplied by ``read_planes`` (injected by the node task, which owns
    the I/O), so only output-map-sized accumulators plus one read block are
    ever in memory -- independent of the length of the moment axis. Not
    valid for ``median`` (:func:`moments_memory_model`).

    Parameters
    ----------
    read_planes : callable
        ``read_planes(start, stop) -> (planes, mask_planes)`` returning the
        moment-axis planes ``start:stop`` as an array with the moment axis
        FIRST (shape ``(stop - start, *map_shape)``) and the matching boolean
        mask planes (or ``None``). Called once per block per pass.
    coords_xds : xarray.Dataset
        The chunk's coordinates (metadata only; used for the moment-axis
        values and the collapsed output coordinates).
    sky_dims : sequence of str
        Dimension names of the sky variable, in order.
    sky_name, sky_attrs, attrs
        Name / attributes of the sky variable and attributes of the input
        dataset (see :func:`assemble_moments_dataset`).
    moments, moment_axis, include_pixel_range, exclude_pixel_range
        As for :func:`moments`.

    Returns
    -------
    xarray.Dataset
        Same layout as the return of :func:`moments`.
    """
    moment_names, include_range, exclude_range = _validate_moments_request(
        moments, moment_axis, include_pixel_range, exclude_pixel_range
    )
    if moments_memory_model(moment_names)["requires_full_profile"]:
        raise ValueError(
            "moments_streamed cannot compute 'median' (needs the whole profile); "
            "use moments() on an in-memory chunk."
        )
    sky_dims = list(sky_dims)
    if moment_axis not in sky_dims:
        raise ValueError(
            f"moment_axis '{moment_axis}' is not a dimension of {sky_name} "
            f"(dims: {sky_dims})."
        )
    map_shape = tuple(coords_xds.sizes[dim] for dim in sky_dims if dim != moment_axis)
    coord_values = _moment_axis_values(coords_xds, moment_axis)
    accumulator = MomentsAccumulator(
        moment_names, coord_values, map_shape, include_range, exclude_range
    )
    n_planes = len(coord_values)
    for pass_index in range(accumulator.n_passes):
        start = 0
        while start < n_planes:
            planes, mask_planes = read_planes(start, n_planes)
            planes = np.asarray(planes)
            for j in range(planes.shape[0]):
                accumulator.add_plane(
                    start + j,
                    planes[j],
                    None if mask_planes is None else mask_planes[j],
                    pass_index=pass_index,
                )
            start += planes.shape[0]
            planes = None
            mask_planes = None
    return assemble_moments_dataset(
        accumulator.finalize(),
        coords_xds,
        attrs,
        sky_dims,
        sky_name,
        sky_attrs,
        moment_axis,
    )


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
    moment_names, include_range, exclude_range = _validate_moments_request(
        moments, moment_axis, include_pixel_range, exclude_pixel_range
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
    filtering = (
        include_range is not None
        or exclude_range is not None
        or mask_planes is not None
    )

    # ---- Stream plane-by-plane along the moment axis (1 or 2 passes) ----------
    # Only map-sized accumulators are allocated; 'median' is handled apart.
    streamed_names = [name for name in moment_names if name not in FULL_PROFILE_MOMENTS]
    accumulator = MomentsAccumulator(
        streamed_names or ["mean"],
        coord_values,
        map_shape,
        include_range,
        exclude_range,
    )
    for pass_index in range(accumulator.n_passes):
        for i in range(n_planes):
            accumulator.add_plane(
                i,
                data_planes[i],
                None if mask_planes is None else mask_planes[i],
                pass_index=pass_index,
            )

    median_map = None
    if "median" in moment_names:
        if filtering:
            # One working copy of the chunk (at the input precision) with
            # excluded pixels set to NaN, applied plane-by-plane so no
            # full-size boolean cube is needed.
            if np.issubdtype(data_planes.dtype, np.floating):
                working = data_planes.copy()
            else:
                working = data_planes.astype(np.float64)
            for i in range(n_planes):
                mask_plane = None if mask_planes is None else mask_planes[i]
                working[i][~accumulator.valid_plane(data_planes[i], mask_plane)] = (
                    np.nan
                )
        else:
            working = data_planes
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            median_map = np.nanmedian(working, axis=0)
        working = None

    results = accumulator.finalize()
    if "median" in moment_names:
        empty = accumulator.count == 0
        results["median"] = np.where(empty, np.nan, median_map).astype(
            accumulator.value_dtype, copy=False
        )
    # Preserve the requested moment order.
    results = {name: results[name] for name in moment_names}

    return assemble_moments_dataset(
        results,
        img_xds,
        img_xds.attrs,
        sky.dims,
        sky_name,
        sky.attrs,
        moment_axis,
    )
