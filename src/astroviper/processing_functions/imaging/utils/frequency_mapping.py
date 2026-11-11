"""Frequency-coordinate mapping helpers for partitioned imaging."""

import numpy as np


def map_visibility_frequencies_to_image(
    visibility_frequencies,
    image_frequencies,
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

    Returns
    -------
    numpy.ndarray
        Integer (``int64``) image-plane index for each visibility channel.

    Raises
    ------
    ValueError
        If either coordinate is not one-dimensional and finite, or if a
        visibility frequency lies more than half an image channel width from
        every image channel centre.
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

    from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

    frequency_map = map_to_nearest_coordinate(visibility_frequencies, image_frequencies)
    too_far = frequency_map < 0
    if np.any(too_far):
        offending = np.flatnonzero(too_far)
        separations = np.abs(
            visibility_frequencies[offending, np.newaxis]
            - image_frequencies[np.newaxis, :]
        )
        nearest = np.argmin(separations, axis=1)
        raise ValueError(
            "Visibility channel(s) lie more than half an image channel width "
            "from the nearest image channel centre; visibility channel indices="
            f"{offending.tolist()}, visibility frequencies="
            f"{visibility_frequencies[offending].tolist()} Hz, nearest image "
            f"frequencies={image_frequencies[nearest].tolist()} Hz, "
            f"separations={separations[np.arange(offending.size), nearest].tolist()} Hz."
        )
    return frequency_map
