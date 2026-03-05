"""Tests for plotting helpers in ``astroviper.utils.plotting``."""

import matplotlib
import numpy as np
import pytest
from astropy.wcs import WCS

from astroviper.utils.plotting import generate_astro_plot

matplotlib.use("Agg", force=True)


def _make_mock_equatorial_sin_wcs() -> WCS:
    """Create a compact equatorial SIN WCS for plotting tests."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    # Negative RA increment gives astronomical handedness (RA increases to the left).
    wcs.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0])
    wcs.wcs.crval = [180.0, 30.0]
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def test_generate_astro_plot_places_source_at_requested_xy():
    """Verify the x/y source index is rendered at the same x/y plot location."""
    n_pix = 20
    x_target = 5
    y_target = 15
    data = np.zeros((n_pix, n_pix), dtype=float)

    # The package convention for this helper is data[x, y].
    data[x_target, y_target] = 1.0

    fig, ax = generate_astro_plot(
        data=data, wcs=_make_mock_equatorial_sin_wcs(), show_world_axes=False
    )

    # Inspect the exact array passed to imshow; argmax gives displayed (row, col).
    plotted_array = np.asarray(ax.images[0].get_array())
    peak_row, peak_col = np.unravel_index(np.argmax(plotted_array), plotted_array.shape)

    # Plot-space x is column, plot-space y is row for imshow image buffers.
    assert (peak_col, peak_row) == (x_target, y_target)
    assert ax.images[0].origin == "lower"

    matplotlib.pyplot.close(fig)


def test_generate_astro_plot_world_axes_have_ra_increasing_left():
    """Verify mocked RA/Dec WCS has increasing RA toward the left edge."""
    n_pix = 20
    data = np.zeros((n_pix, n_pix), dtype=float)
    wcs = _make_mock_equatorial_sin_wcs()

    fig, _ = generate_astro_plot(data=data, wcs=wcs, show_world_axes=True)

    # Compare world coordinates at left/right edges through the image midline.
    y_mid = (n_pix - 1) / 2.0
    ra_left_deg, _ = wcs.all_pix2world(0.0, y_mid, 0)
    ra_right_deg, _ = wcs.all_pix2world(n_pix - 1.0, y_mid, 0)

    # Astronomical convention for sky images: RA increases to the left.
    assert ra_left_deg > ra_right_deg

    matplotlib.pyplot.close(fig)


def test_generate_astro_plot_requires_wcs_for_world_axes():
    """Verify world-axis mode rejects missing WCS input."""
    data = np.zeros((20, 20), dtype=float)

    with pytest.raises(
        ValueError, match="wcs must be provided when show_world_axes=True"
    ):
        generate_astro_plot(data=data, wcs=None, show_world_axes=True)
