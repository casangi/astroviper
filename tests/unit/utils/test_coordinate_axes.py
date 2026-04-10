"""Tests for public coordinate-axis interpolation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from astroviper.utils.coordinate_axes import (
    prepare_world_to_pixel_interp,
    representative_pixel_scale,
    world_value_to_pixel,
)


def test_prepare_world_to_pixel_interp_reverses_descending_axes():
    """Descending axes should be normalized into ascending interpolation inputs."""
    xp, fp = prepare_world_to_pixel_interp(np.array([3.0, 2.0, 1.0]))

    np.testing.assert_allclose(xp, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(fp, [2.0, 1.0, 0.0])


@pytest.mark.parametrize(
    ("axis", "match"),
    [
        (np.array([]), "finite 1-D arrays"),
        (np.array([1.0, 2.0, 2.0]), "strictly monotonic"),
    ],
)
def test_prepare_world_to_pixel_interp_rejects_invalid_axes(axis, match):
    """Invalid axes should raise clear validation errors before interpolation."""
    with pytest.raises(ValueError, match=match):
        prepare_world_to_pixel_interp(axis)


def test_world_value_to_pixel_interpolates_descending_axes():
    """World values should interpolate correctly even when the source axis descends."""
    pixel = world_value_to_pixel(2.5, np.array([3.0, 2.0, 1.0]), "l")

    np.testing.assert_allclose(pixel, 0.5, atol=1e-12)


def test_world_value_to_pixel_rejects_out_of_range_values():
    """Out-of-range world values should raise a clear error."""
    with pytest.raises(ValueError, match="outside the image coordinate range"):
        world_value_to_pixel(4.0, np.array([3.0, 2.0, 1.0]), "l")


def test_representative_pixel_scale_uses_absolute_spacing():
    """Representative pixel scale should be positive on descending axes."""
    scale = representative_pixel_scale(np.array([3.0, 2.5, 2.0, 1.5]), "l")

    np.testing.assert_allclose(scale, 0.5, atol=1e-12)


def test_representative_pixel_scale_rejects_single_point_axes():
    """A single coordinate does not define a usable pixel spacing."""
    # ``representative_pixel_scale`` validates after taking a median of ``np.diff``;
    # a single-point axis therefore emits NumPy runtime warnings before raising.
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="spacing must be positive and finite"):
            representative_pixel_scale(np.array([3.0]), "l")
