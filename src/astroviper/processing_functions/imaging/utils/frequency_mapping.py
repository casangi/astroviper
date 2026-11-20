"""Frequency-coordinate mapping helpers for partitioned imaging."""

import numpy as np

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def map_visibility_frequencies_to_image(
    visibility_frequencies,
    image_frequencies,
    *,
    full_image_frequencies=None,
    frequency_boundaries=None,
):
    """Map each visibility channel onto its nearest image-frequency channel.

    Nearest-channel assignment uses
    :func:`graphviper.graph_tools.coordinate_utils.map_to_nearest_coordinate`,
    the rule by which
    :func:`graphviper.graph_tools.coordinate_utils.interpolate_data_coords_onto_parallel_coords`
    selects the visibility channels of a task, so a task grids every channel it
    selects: every visibility channel is assigned to the image channel whose
    centre frequency is closest, provided it lies within the half channel of
    that image channel on its side (half the gap to the neighbouring image
    channel, or beyond either end of the image axis half the end gap). Image
    axes that are not copied verbatim from the processing set (regridded,
    shifted, or from a second measurement set with slightly different channel
    centres) therefore map without raising. Several visibility channels may
    map onto one image channel (channel averaging onto a coarser image axis).

    Parameters
    ----------
    visibility_frequencies : array-like
        One-dimensional visibility-channel centre frequencies (Hz).
    image_frequencies : array-like
        One-dimensional image-channel centre frequencies (Hz). The channel
        width is the spacing of this axis; when the image has a single
        channel the widest visibility spacing is used instead, and when both
        axes have one channel any frequency maps onto that plane.

    full_image_frequencies : array-like, optional
        Original, unpartitioned image-frequency axis (Hz), retained for legacy
        callers. Mapping and boundary checks use this axis, with observations
        required to belong to the local image-frequency subset. Precomputed
        ``frequency_boundaries`` take precedence when supplied.

    frequency_boundaries : dict, optional
        Precomputed image-channel acceptance intervals (Hz), keyed by channel
        centre. Each pair gives the lowest and highest accepted float64
        frequency, both inclusive. Only task-local channel records are needed.
        When omitted, use the supplied image axis to infer channel widths.

    Returns
    -------
    numpy.ndarray
        Integer (``int64``) image-plane index for each visibility channel.

    Raises
    ------
    ValueError
        If either coordinate is not one-dimensional and finite, or if a
        visibility frequency lies outside the accepted image-channel intervals.
        Also raised for duplicate full-axis centres, an inconsistent local
        subset, or malformed or overlapping precomputed boundaries.
    """
    visibility_frequencies = np.asarray(visibility_frequencies, dtype=np.float64)
    image_frequencies = np.asarray(image_frequencies, dtype=np.float64)

    if visibility_frequencies.ndim != 1 or image_frequencies.ndim != 1:
        raise ValueError("Visibility and image frequencies must be one-dimensional.")
    if visibility_frequencies.size == 0 or image_frequencies.size == 0:
        raise ValueError("Visibility and image frequencies must not be empty.")
    if not np.all(np.isfinite(visibility_frequencies)) or not np.all(
        np.isfinite(image_frequencies)
    ):
        raise ValueError("Visibility and image frequencies must be finite.")

    if frequency_boundaries is not None:
        return _map_with_frequency_boundaries(
            visibility_frequencies, image_frequencies, frequency_boundaries
        )

    from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

    mapping_frequencies = image_frequencies
    local_indices = None
    if full_image_frequencies is not None:
        mapping_frequencies = np.asarray(full_image_frequencies, dtype=np.float64)
        if (
            mapping_frequencies.ndim != 1
            or mapping_frequencies.size == 0
            or not np.all(np.isfinite(mapping_frequencies))
        ):
            raise ValueError(
                "Full image frequencies must be a nonempty finite one-dimensional axis."
            )
        if (
            np.unique(mapping_frequencies).size != mapping_frequencies.size
            or np.unique(image_frequencies).size != image_frequencies.size
        ):
            raise ValueError("Full and local image frequencies must be unique.")
        # Local centres are a subset of the original coordinates, not observed
        # frequencies to interpolate. Preserve global midpoint tie breaking.
        lookup = {value: index for index, value in enumerate(mapping_frequencies)}
        if any(value not in lookup for value in image_frequencies):
            raise ValueError(
                "Local image frequencies must be a subset of the full image axis."
            )
        local_indices = np.full(mapping_frequencies.size, -1, dtype=np.int64)
        for index, value in enumerate(image_frequencies):
            local_indices[lookup[value]] = index

    frequency_map = map_to_nearest_coordinate(
        visibility_frequencies, mapping_frequencies
    )
    too_far = frequency_map < 0
    if np.any(too_far):
        offending = np.flatnonzero(too_far)
        separations = np.abs(
            visibility_frequencies[offending, np.newaxis]
            - mapping_frequencies[np.newaxis, :]
        )
        nearest = np.argmin(separations, axis=1)
        raise ValueError(
            "Visibility channel(s) lie more than half an image channel width "
            "from the nearest image channel centre; visibility channel indices="
            f"{offending.tolist()}, visibility frequencies="
            f"{visibility_frequencies[offending].tolist()} Hz, nearest image "
            f"frequencies={mapping_frequencies[nearest].tolist()} Hz, "
            f"separations={separations[np.arange(offending.size), nearest].tolist()} Hz."
        )
    if local_indices is not None:
        frequency_map = local_indices[frequency_map]
        if np.any(frequency_map < 0):
            raise ValueError(
                "Visibility channels map outside this task's image-frequency subset; "
                f"visibility channel indices={np.flatnonzero(frequency_map < 0).tolist()}."
            )
    return frequency_map


def calculate_frequency_boundaries(image_frequencies, visibility_frequencies=None):
    """Calculate closed float64 acceptance intervals before partitioning.

    Parameters
    ----------
    image_frequencies : array-like
        Full image-channel centres in their original order (Hz), unique and
        finite. The lower original index owns an exact nearest-centre tie.
    visibility_frequencies : array-like, optional
        Full observation axis, required for a truly single-channel image.
        Its widest gap determines the tolerance, as in GraphVIPER.

    Returns
    -------
    dict
        Channel centre -> [lowest accepted frequency, highest accepted
        frequency]. Endpoints are representable float64 values, inclusive.
        Keys are strings so the metadata survive JSON/Zarr serialization.
        None for a single-channel image without observation context (direct
        callers retain GraphVIPER's existing single-channel rule).

    Raises
    ------
    ValueError
        If the image axis is empty, non-finite, not one-dimensional or has
        duplicate centres.
    """
    from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

    axis = np.asarray(image_frequencies, dtype=np.float64)
    if axis.ndim != 1 or not axis.size or not np.all(np.isfinite(axis)):
        raise ValueError(
            "Image frequencies must be a nonempty finite one-dimensional axis."
        )
    if np.unique(axis).size != axis.size:
        raise ValueError("Image frequencies must be unique.")
    if axis.size == 1:
        if visibility_frequencies is None:
            return None
        observed = np.asarray(visibility_frequencies, dtype=np.float64)
        # Validate with the same implementation used for graph selection.
        map_to_nearest_coordinate(observed, axis)
        half_width = (
            0.5 * np.diff(np.sort(observed)).max() if observed.size > 1 else np.inf
        )
        if np.isinf(half_width):
            limit = np.finfo(np.float64).max
            return {str(float(axis[0])): [-limit, limit]}
        order = np.array([0])
        # Compare distances with the fixed full-observation tolerance; rounded
        # centre +/- half_width alone can include or exclude an adjacent float.
        return {
            str(float(axis[0])): [
                float(
                    _bisect_frequency_boundaries(
                        axis, order, -np.inf, half_width=half_width
                    )[0]
                ),
                float(
                    _bisect_frequency_boundaries(
                        axis, order, np.inf, half_width=half_width
                    )[0]
                ),
            ]
        }

    order = np.argsort(axis, kind="stable")
    centres = axis[order]
    gaps = np.diff(centres)
    lower = centres - 0.5 * gaps[np.maximum(np.arange(axis.size) - 1, 0)]
    upper = centres + 0.5 * gaps[np.minimum(np.arange(axis.size), axis.size - 2)]

    # Rounded midpoints alone can disagree with GraphVIPER at a neighbouring
    # float. Resolve endpoints using its actual distance arithmetic, including
    # original-index tie breaking. This work happens once on the driver.
    for endpoints, direction in ((lower, -np.inf), (upper, np.inf)):
        for _ in range(16):
            owned = map_to_nearest_coordinate(endpoints, axis) == order
            outer = np.nextafter(endpoints, direction)
            outer_owned = map_to_nearest_coordinate(outer, axis) == order
            inward = ~owned
            outward = owned & outer_owned
            if not np.any(inward | outward):
                break
            endpoints[inward] = np.nextafter(endpoints[inward], centres[inward])
            endpoints[outward] = outer[outward]
        else:
            # Severe cancellation around zero can make a rounded-distance
            # tie span many representable values. Search float64 bit order
            # instead of walking that plateau one nextafter at a time.
            endpoints[:] = _bisect_frequency_boundaries(axis, order, direction)
    return {
        str(float(centre)): [float(lo), float(hi)]
        for centre, lo, hi in zip(centres, lower, upper, strict=True)
    }


def select_frequency_boundaries(frequency_boundaries, image_frequencies):
    """Return only the acceptance intervals of the requested image centres."""
    if frequency_boundaries is None:
        return None
    try:
        return {
            str(float(f)): frequency_boundaries[str(float(f))]
            for f in image_frequencies
        }
    except KeyError as exc:
        raise ValueError(
            "Local image frequencies must be a subset of the boundary axis."
        ) from exc


def task_frequency_boundaries(task_coords, image_params):
    """Read driver-supplied local intervals, or prepare them for direct calls."""
    coordinate = task_coords["frequency"]
    if "frequency_boundaries" in coordinate:
        return coordinate["frequency_boundaries"]
    return select_frequency_boundaries(
        calculate_frequency_boundaries(
            image_params.get("frequency_coords", coordinate["data"])
        ),
        coordinate["data"],
    )


def _map_with_frequency_boundaries(
    visibility_frequencies, image_frequencies, boundaries
):
    """Find local channel intervals without reconstructing the global axis."""
    local = select_frequency_boundaries(boundaries, image_frequencies)
    intervals = np.asarray(list(local.values()), dtype=np.float64)
    if (
        intervals.shape != (image_frequencies.size, 2)
        or np.any(np.isnan(intervals))
        or np.any(intervals[:, 0] > intervals[:, 1])
    ):
        raise ValueError("Frequency boundaries must contain ordered lower/upper pairs.")
    order = np.argsort(intervals[:, 0], kind="stable")
    sorted_intervals = intervals[order]
    if np.any(sorted_intervals[:-1, 1] >= sorted_intervals[1:, 0]):
        raise ValueError("Frequency boundaries must not overlap.")
    positions = (
        np.searchsorted(sorted_intervals[:, 0], visibility_frequencies, side="right")
        - 1
    )
    valid = (positions >= 0) & (
        visibility_frequencies <= sorted_intervals[np.maximum(positions, 0), 1]
    )
    if not np.all(valid):
        raise ValueError(
            "Visibility channels map outside this task's image-frequency boundaries "
            "(more than half an image channel width or owned by another task); "
            f"visibility channel indices={np.flatnonzero(~valid).tolist()}."
        )
    return order[positions].astype(np.int64)


def _bisect_frequency_boundaries(axis, order, direction, *, half_width=None):
    """Resolve long rounding plateaus in at most 64 float64-key bisections."""
    from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

    def map_values(values):
        if half_width is None:
            return map_to_nearest_coordinate(values, axis)
        return np.where(np.abs(values - axis[0]) <= half_width, 0, -1)

    sign = np.uint64(1 << 63)

    def keys(values):
        bits = np.asarray(values, dtype=np.float64).view(np.uint64)
        return np.where(bits & sign, ~bits, bits ^ sign)

    def values(ordered):
        return np.where(ordered & sign, ordered ^ sign, ~ordered).view(np.float64)

    centres = axis[order]
    accepted = keys(centres)
    limit = np.finfo(np.float64).max
    outer = (
        np.concatenate(([-limit], centres[:-1]))
        if direction < 0
        else np.concatenate((centres[1:], [limit]))
    )
    rejected = keys(outer)
    # At the finite extremes, the end interval can already include the limit.
    limit_owned = map_values(outer) == order
    accepted[limit_owned] = rejected[limit_owned]
    for _ in range(64):
        low = np.minimum(accepted, rejected)
        high = np.maximum(accepted, rejected)
        active = high - low > 1
        if not np.any(active):
            break
        middle = low[active] + (high[active] - low[active]) // np.uint64(2)
        owned = map_values(values(middle)) == order[active]
        indices = np.flatnonzero(active)
        accepted[indices[owned]] = middle[owned]
        rejected[indices[~owned]] = middle[~owned]
    return values(accepted)


def calculate_observation_frequency_boundaries(image_frequencies, ps_xdt):
    """Prepare single-image-channel intervals from each full observation axis.

    Only coordinate metadata are read. Multichannel images already have
    observation-independent boundaries and return an empty mapping.

    Parameters
    ----------
    image_frequencies : array-like
        Full image-channel centres in Hz.
    ps_xdt : mapping or xarray.DataTree
        Full observations before task-level frequency selection.

    Returns
    -------
    dict
        Observation names mapped to their single-channel acceptance intervals.
    """
    if len(image_frequencies) != 1:
        return {}
    return {
        name: calculate_frequency_boundaries(image_frequencies, child.frequency.values)
        for name, child in ps_xdt.items()
    }


def attach_observation_frequency_boundaries(ps_xdt, boundaries):
    """Retain each observation's driver-computed intervals across data slicing.

    The processing set may already be selected or preloaded. Never infer a
    replacement tolerance from its truncated coordinates.

    Parameters
    ----------
    ps_xdt : mapping or xarray.DataTree
        Selected observations whose frequency-coordinate attributes are updated.
    boundaries : dict or None
        Driver-computed acceptance intervals keyed by observation name.

    Returns
    -------
    mapping or xarray.DataTree
        The input processing set, with interval metadata attached in place.

    Raises
    ------
    ValueError
        A selected observation is missing its precomputed intervals.
    """
    if not boundaries:
        return ps_xdt
    for name, child in ps_xdt.items():
        if name not in boundaries:
            raise ValueError(f"Missing frequency boundaries for observation {name!r}.")
        child.frequency.attrs = {
            **child.frequency.attrs,
            "image_frequency_boundaries": boundaries[name],
        }
    return ps_xdt


def get_frequency_boundaries(img_xds, ms_xds):
    """Use common image intervals or this observation's single-channel interval.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image with task-local frequencies and optional common intervals.
    ms_xds : xarray.Dataset or xarray.DataTree
        Observation carrying its own intervals on the frequency coordinate.

    Returns
    -------
    dict or None
        Applicable precomputed intervals, or None for legacy direct calls.
    """
    common = img_xds.frequency.attrs.get("frequency_boundaries")
    if common is not None:
        return common
    return ms_xds.frequency.attrs.get("image_frequency_boundaries")


def without_frequency_mapping_metadata(img_xds):
    """Remove execution-only frequency metadata from a finished image.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Finished image, after all frequency mapping has completed.

    Returns
    -------
    xarray.Dataset
        Image with mapping context removed from the frequency attributes.
        Scientific metadata and array buffers are preserved. The input is
        unchanged; a shallow copy is made only when mapping context is present.
    """
    internal = {
        "full_image_frequencies",
        "frequency_boundaries",
        "image_frequency_boundaries",
        "observation_frequency_boundaries",
    }
    if "frequency" not in img_xds.coords or not internal.intersection(
        img_xds.frequency.attrs
    ):
        return img_xds
    result = img_xds.copy(deep=False)
    result.frequency.attrs = {
        name: value
        for name, value in img_xds.frequency.attrs.items()
        if name not in internal
    }
    return result
