"""Frequency-coordinate mapping helpers for partitioned imaging."""

import numpy as np


def map_visibility_frequencies_to_image(
    visibility_frequencies,
    image_frequencies,
    *,
    rtol=1.0e-12,
    atol=0.0,
):
    """Map each visibility channel to exactly one image-frequency channel."""
    visibility_frequencies = np.asarray(visibility_frequencies, dtype=np.float64)
    image_frequencies = np.asarray(image_frequencies, dtype=np.float64)

    if visibility_frequencies.ndim != 1 or image_frequencies.ndim != 1:
        raise ValueError("Visibility and image frequencies must be one-dimensional.")
    if not np.all(np.isfinite(visibility_frequencies)) or not np.all(
        np.isfinite(image_frequencies)
    ):
        raise ValueError("Visibility and image frequencies must be finite.")

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

    channel_map = np.argmax(matches, axis=1).astype(np.int64)
    if np.unique(channel_map).size != visibility_frequencies.size:
        raise ValueError(
            "Visibility frequencies must map one-to-one onto image frequencies; "
            f"visibility frequencies={visibility_frequencies}; "
            f"image frequencies={image_frequencies}."
        )
    return channel_map
