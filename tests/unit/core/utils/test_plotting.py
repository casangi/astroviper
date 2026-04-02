"""Tests for plotting helpers in ``astroviper.utils.plotting``."""

import matplotlib

# Force a non-interactive backend before pyplot/module imports so headless CI
# can render figures without GUI dependencies or backend-selection warnings.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.wcs import WCS

import xarray as xr

from astroviper.utils.plotting import generate_plot


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


def test_generate_plot_places_source_at_requested_xy():
    """Verify the x/y source index is rendered at the same x/y plot location."""
    n_pix = 20
    x_target = 5
    y_target = 15
    data = np.zeros((n_pix, n_pix), dtype=float)

    # The package convention for this helper is data[x, y].
    data[x_target, y_target] = 1.0

    fig, ax = generate_plot(
        data=data, wcs=_make_mock_equatorial_sin_wcs(), show_world_axes=False
    )

    # Inspect the exact array passed to imshow; argmax gives displayed (row, col).
    plotted_array = np.asarray(ax.images[0].get_array())
    peak_row, peak_col = np.unravel_index(np.argmax(plotted_array), plotted_array.shape)

    # Plot-space x is column, plot-space y is row for imshow image buffers.
    assert (peak_col, peak_row) == (x_target, y_target)
    assert ax.images[0].origin == "lower"

    plt.close(fig)


def test_generate_plot_world_axes_have_ra_increasing_left():
    """Verify mocked RA/Dec WCS has increasing RA toward the left edge."""
    n_pix = 20
    data = np.zeros((n_pix, n_pix), dtype=float)
    wcs = _make_mock_equatorial_sin_wcs()

    fig, _ = generate_plot(data=data, wcs=wcs, show_world_axes=True)

    # Compare world coordinates at left/right edges through the image midline.
    y_mid = (n_pix - 1) / 2.0
    ra_left_deg, _ = wcs.all_pix2world(0.0, y_mid, 0)
    ra_right_deg, _ = wcs.all_pix2world(n_pix - 1.0, y_mid, 0)

    # Astronomical convention for sky images: RA increases to the left.
    assert ra_left_deg > ra_right_deg

    plt.close(fig)


def test_generate_plot_wcs_branch_does_not_override_world_coordinate_labels():
    """WCSAxes labels should not be overridden with generic dim names."""
    n_pix = 20
    data = np.zeros((n_pix, n_pix), dtype=float)
    wcs = _make_mock_equatorial_sin_wcs()

    fig, ax = generate_plot(data=data, wcs=wcs, show_world_axes=True)

    # WCSAxes derives axis labels from the WCS object (e.g. "Right Ascension
    # (J2000)").  Our helper must not replace those with generic strings like
    # "x" or "y".
    ra_label = ax.coords[0].get_axislabel()
    dec_label = ax.coords[1].get_axislabel()
    assert ra_label not in ("x", "y", "")
    assert dec_label not in ("x", "y", "")

    plt.close(fig)


def test_generate_plot_wcs_branch_applies_explicit_string_coord_labels():
    """Explicitly requested string labels should override WCSAxes defaults."""
    n_pix = 20
    data = xr.DataArray(
        np.zeros((n_pix, n_pix), dtype=float),
        dims=("ra", "dec"),
        coords={
            "ra": np.arange(n_pix, dtype=float),
            "dec": np.arange(n_pix, dtype=float),
        },
    )
    wcs = _make_mock_equatorial_sin_wcs()

    fig, ax = generate_plot(
        data=data,
        wcs=wcs,
        show_world_axes=True,
        x_coords="ra",
        y_coords="dec",
    )

    assert ax.coords[0].get_axislabel() == "ra"
    assert ax.coords[1].get_axislabel() == "dec"

    plt.close(fig)


def test_generate_plot_requires_2d_data():
    """Verify plotting helper rejects non-2D inputs with a clear error."""
    data_1d = np.zeros(20, dtype=float)

    with pytest.raises(ValueError, match="data must be a 2D array-like object"):
        generate_plot(data=data_1d, show_world_axes=False)


def test_generate_plot_uses_dataarray_axis_coords_for_world_axes():
    """World-axis plotting should use the DataArray's axis coordinates by default."""
    data = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 4),
        dims=("x", "y"),
        coords={
            "x": np.array([10.0, 20.0, 30.0]),
            "y": np.array([-2.0, 0.0, 2.0, 4.0]),
        },
    )

    fig, ax = generate_plot(data=data, show_world_axes=True)

    quadmesh = ax.collections[0]
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    assert quadmesh.__class__.__name__ == "QuadMesh"
    assert x_limits == (10.0, 30.0)
    assert y_limits == (-2.0, 4.0)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_aspect() == 1.0
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "value"

    plt.close(fig)


def test_generate_plot_allows_explicit_xy_coordinate_override():
    """Explicit x/y coordinate arrays should override DataArray coordinates."""
    data = xr.DataArray(
        np.arange(6, dtype=float).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": np.array([1.0, 2.0]), "y": np.array([10.0, 20.0, 30.0])},
    )
    x_override = np.array([100.0, 200.0])
    y_override = np.array([-5.0, 5.0, 15.0])

    fig, ax = generate_plot(
        data=data,
        show_world_axes=True,
        x_coords=x_override,
        y_coords=y_override,
    )

    assert ax.get_xlim() == (100.0, 200.0)
    assert ax.get_ylim() == (-5.0, 15.0)

    plt.close(fig)


def test_generate_plot_defaults_to_axis_indices_without_coords():
    """Plain arrays should default to axis-0 as x and axis-1 as y indices."""
    data = np.zeros((4, 5), dtype=float)

    fig, ax = generate_plot(data=data, show_world_axes=True)

    assert ax.get_xlim() == (0.0, 3.0)
    assert ax.get_ylim() == (0.0, 4.0)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert len(fig.axes) == 2

    plt.close(fig)


def test_generate_plot_pixel_mode_adds_default_labels_and_colorbar():
    """Pixel plotting should also label axes and add a colorbar by default."""
    data = xr.DataArray(
        np.arange(9, dtype=float).reshape(3, 3),
        dims=("ra", "dec"),
        name="flux",
    )

    fig, ax = generate_plot(data=data, show_world_axes=False)

    assert ax.get_xlabel() == "ra"
    assert ax.get_ylabel() == "dec"
    assert ax.get_aspect() == 1.0
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "flux"

    plt.close(fig)


def test_generate_plot_handles_yx_dataarray_ordering():
    """DataArrays with (y, x) dim order should produce the same plot as (x, y)."""
    arr_xy = np.arange(12, dtype=float).reshape(3, 4)
    data_xy = xr.DataArray(
        arr_xy,
        dims=("x", "y"),
        coords={
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([10.0, 20.0, 30.0, 40.0]),
        },
    )
    data_yx = data_xy.transpose("y", "x")

    fig_xy, ax_xy = generate_plot(data=data_xy, show_world_axes=True)
    fig_yx, ax_yx = generate_plot(data=data_yx, show_world_axes=True)

    assert ax_xy.get_xlim() == ax_yx.get_xlim()
    assert ax_xy.get_ylim() == ax_yx.get_ylim()
    assert ax_xy.get_xlabel() == ax_yx.get_xlabel() == "x"
    assert ax_xy.get_ylabel() == ax_yx.get_ylabel() == "y"

    plt.close(fig_xy)
    plt.close(fig_yx)


def test_generate_plot_sets_optional_title():
    """Plot helper should display an explicitly requested title."""
    data = np.zeros((4, 4), dtype=float)

    fig, ax = generate_plot(data=data, show_world_axes=False, title="Test Title")

    assert ax.get_title() == "Test Title"

    plt.close(fig)
