"""Frequency-coordinate mapping helpers for partitioned imaging."""

import numpy as np


def _half_channel_widths(image_frequencies, visibility_frequencies):
    """Return the per-image-channel matching tolerance (half a channel width).

    The channel width is taken from the image axis when it has more than one
    channel, from the visibility axis when only that one does, and is
    unbounded when both axes hold a single channel (a single visibility
    channel imaged onto a single plane can only mean that plane).
    """
    if image_frequencies.size > 1:
        return 0.5 * np.abs(np.gradient(image_frequencies))
    if visibility_frequencies.size > 1:
        width = np.abs(np.gradient(visibility_frequencies)).max()
        return np.full(image_frequencies.shape, 0.5 * width)
    return np.full(image_frequencies.shape, np.inf)


def map_visibility_frequencies_to_image(
    visibility_frequencies,
    image_frequencies,
    *,
    matching="nearest",
    rtol=1e-12,
    atol=0.0,
):
    """Map visibility channels using an explicit cube or continuum policy.

    ``matching="exact"`` preserves the continuum one-to-one contract: each
    visibility frequency must match exactly one image frequency within
    ``rtol``/``atol``, and no two visibility channels may share an image plane.
    Sparse and reordered channel selections are allowed; frequency averaging
    and shifts beyond the numerical tolerance are rejected.

    The default ``matching="nearest"`` preserves cube behavior below.

    Every visibility channel is assigned to the image channel whose centre
    frequency is closest, provided that distance is at most half an image
    channel width. This mirrors the nearest-neighbour selection
    :func:`graphviper.graph_tools.coordinate_utils.interpolate_data_coords_onto_parallel_coords`
    uses to pick the visibility channels of a task, so image axes that are
    not copied verbatim from the processing set (regridded, shifted, or from
    a second measurement set with slightly different channel centres) still
    map without raising. Several visibility channels may map onto one image
    channel (channel averaging onto a coarser image axis).

    Parameters
    ----------
    visibility_frequencies : array-like
        One-dimensional visibility-channel centre frequencies (Hz).
    image_frequencies : array-like
        One-dimensional image-channel centre frequencies (Hz). The channel
        width is the spacing of this axis; when the image has a single
        channel the visibility spacing is used instead, and when both axes
        have one channel any frequency maps onto that plane.

    matching : {"nearest", "exact"}, default "nearest"
        Cube nearest-channel matching or continuum exact one-to-one matching.
    rtol : float, default 1e-12
        Relative numerical tolerance for exact matching only.
    atol : float, default 0.0
        Absolute numerical tolerance in Hz for exact matching only.

    Returns
    -------
    numpy.ndarray
        Integer (``int64``) image-plane index for each visibility channel.

    Raises
    ------
    ValueError
        If either coordinate is not one-dimensional and finite, or if a
        visibility frequency lies more than half an image channel width from
        every image channel centre in nearest mode. In exact mode, missing,
        ambiguous or many-to-one matches raise. Invalid policies or negative
        or nonfinite exact-match tolerances also raise.
    """
    if matching not in ("nearest", "exact"):
        raise ValueError("matching must be 'nearest' or 'exact'.")
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

    if matching == "exact":
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("Exact-match tolerances must be finite and nonnegative.")
        matches = np.isclose(
            visibility_frequencies[:, np.newaxis],
            image_frequencies[np.newaxis, :],
            rtol=rtol,
            atol=atol,
        )
        match_counts = matches.sum(axis=1)
        if np.any(match_counts != 1):
            raise ValueError(
                "Each visibility frequency must match exactly one image frequency; "
                f"visibility frequencies={visibility_frequencies}; "
                f"image frequencies={image_frequencies}."
            )

        frequency_map = np.argmax(matches, axis=1).astype(np.int64)
        if np.unique(frequency_map).size != visibility_frequencies.size:
            raise ValueError(
                "Visibility frequencies must map one-to-one onto image frequencies; "
                f"visibility frequencies={visibility_frequencies}; "
                f"image frequencies={image_frequencies}."
            )
        return frequency_map

    distances = np.abs(
        visibility_frequencies[:, np.newaxis] - image_frequencies[np.newaxis, :]
    )
    frequency_map = np.argmin(distances, axis=1).astype(np.int64)
    nearest_distance = distances[np.arange(visibility_frequencies.size), frequency_map]
    tolerance = _half_channel_widths(image_frequencies, visibility_frequencies)[
        frequency_map
    ]

    too_far = nearest_distance > tolerance
    if np.any(too_far):
        offending = np.flatnonzero(too_far)
        raise ValueError(
            "Visibility channel(s) lie more than half an image channel width "
            "from the nearest image channel centre; visibility channel indices="
            f"{offending.tolist()}, visibility frequencies="
            f"{visibility_frequencies[offending].tolist()} Hz, nearest image "
            f"frequencies={image_frequencies[frequency_map[offending]].tolist()} Hz, "
            f"separations={nearest_distance[offending].tolist()} Hz, allowed="
            f"{tolerance[offending].tolist()} Hz."
        )
    return frequency_map
