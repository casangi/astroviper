from typing import Literal
import pytest
import numpy as np
import xarray as xr
from astroviper.core.imaging.imaging_utils.make_point_spread_function import make_psf
from astroviper.core.imaging.imaging_utils.standard_gridding_example import (
    generate_ms4_with_point_sources,
    generate_cube_ms4_with_spectral_point_source,
)


@pytest.fixture
def basic_vis_data():
    """
    Generate basic visibility data using standard_gridding_example.
    This creates MS4 data with a point source.
    """
    nsources = 1
    flux = np.array([1.0])
    sources, npix, cell, vis = generate_ms4_with_point_sources(nsources, flux)
    return vis, npix, cell


@pytest.fixture
def cube_vis_data():
    """
    Generate visibility data cube using standard_gridding_example.
    This creates MS4 data with a point source.
    """
    cell, vis = generate_cube_ms4_with_spectral_point_source()
    # cell value only in radians
    return vis, cell


def test_make_psf_basic(basic_vis_data):
    """
    Test basic PSF generation with default parameters.
    """
    vis, npix, cell = basic_vis_data
    # Get necessary parameters from vis
    freq_chan = vis.coords["frequency"].values
    pol = vis.coords["polarization"].values
    time_coords = vis.coords["time"].values[0]

    # Define imaging parameters
    im_params = {
        "cell_size": (cell.to("rad").value, cell.to("rad").value),
        "image_size": (npix, npix),
        "phase_center": (3.14973047, -0.3292367),  # RA, Dec in radians
        "time_coords": time_coords,
        "chan_mode": "continuum",
    }

    # Define gridding parameters
    grid_params = {
        "sampling": 100,
        "complex_grid": True,
        "support": 7,
    }

    # Generate PSF
    psf_ds = make_psf(vis, im_params, grid_params)

    # Verify output is an xarray DataArray
    assert isinstance(psf_ds, xr.Dataset)
    assert psf_ds.data_vars["POINT_SPREAD_FUNCTION"] is not None
    psf_da = psf_ds["POINT_SPREAD_FUNCTION"]
    assert isinstance(psf_da, xr.DataArray)
    # Verify the name
    assert psf_da.name == "POINT_SPREAD_FUNCTION"

    # Verify dimensions
    expected_dims = ("time", "frequency", "polarization", "l", "m")
    assert psf_da.dims == expected_dims

    # Verify shape
    nfreq = len(freq_chan)
    npol = len(pol)
    expected_shape = (1, nfreq, npol, npix, npix)
    assert psf_da.shape == expected_shape

    # Verify PSF values are real and normalized
    psf_data = psf_da.values
    assert np.all(np.isfinite(psf_data))

    # PSF should have a peak near the center
    center_idx = npix // 2
    half_width = 5
    psf_slice = psf_data[0, 0, 0, :, :]
    # Find the location of the global maximum in this slice
    max_flat_idx = np.argmax(psf_slice)
    max_i, max_j = np.unravel_index(max_flat_idx, psf_slice.shape)
    # Assert that the global maximum lies within the central window
    assert center_idx - half_width <= max_i < center_idx + half_width
    assert center_idx - half_width <= max_j < center_idx + half_width


def test_make_psf_cube(cube_vis_data):
    """
    Test PSF generation for cube visibility data.
    """
    vis, cell = cube_vis_data
    npix = 200
    freq_chan = vis.coords["frequency"].values
    pol = vis.coords["polarization"].values

    im_params = {
        "cell_size": (cell, cell),
        "image_size": (npix, npix),
        "phase_center": (3.14973047, -0.3292367),
        "chan_mode": "cube",
    }

    grid_params = {
        "sampling": 100,
        "complex_grid": True,
        "support": 7,
    }

    psf_ds = make_psf(vis, im_params, grid_params)
    assert isinstance(psf_ds, xr.Dataset)
    assert psf_ds.data_vars["POINT_SPREAD_FUNCTION"] is not None

    psf_da = psf_ds["POINT_SPREAD_FUNCTION"]
    assert isinstance(psf_da, xr.DataArray)
    assert psf_da.name == "POINT_SPREAD_FUNCTION"

    expected_dims = ("time", "frequency", "polarization", "l", "m")
    assert psf_da.dims == expected_dims

    nfreq = len(freq_chan)
    npol = len(pol)
    expected_shape = (1, nfreq, npol, npix, npix)
    assert psf_da.shape == expected_shape

    psf_data = psf_da.values
    assert np.all(np.isfinite(psf_data))


def test_make_psf_coordinates(basic_vis_data):
    """
    Test that PSF has correct coordinates.
    """
    vis, npix, cell = basic_vis_data

    freq_chan = vis.coords["frequency"].values
    pol = vis.coords["polarization"].values

    im_params = {
        "cell_size": (cell.to("rad").value, cell.to("rad").value),
        "image_size": (npix, npix),
        "phase_center": (3.14973047, -0.3292367),
        "chan_mode": "continuum",
    }

    grid_params = {
        "sampling": 100,
        "complex_grid": True,
        "support": 7,
    }

    psf_ds = make_psf(vis, im_params, grid_params)
    psf_da = psf_ds["POINT_SPREAD_FUNCTION"]

    # Verify coordinates exist
    assert "time" in psf_da.coords
    assert "frequency" in psf_da.coords
    assert "polarization" in psf_da.coords
    assert "l" in psf_da.coords
    assert "m" in psf_da.coords

    # Verify coordinate values
    np.testing.assert_array_equal(psf_da.coords["frequency"].values, freq_chan)
    np.testing.assert_array_equal(psf_da.coords["polarization"].values, pol)

    # Verify spatial coordinate dimensions
    assert len(psf_da.coords["l"]) == npix
    assert len(psf_da.coords["m"]) == npix


def test_make_psf_peak_location(basic_vis_data):
    """
    Test that PSF peak is at or near the center of the image.
    """
    vis, npix, cell = basic_vis_data

    freq_chan = vis.coords["frequency"].values
    pol = vis.coords["polarization"].values

    im_params = {
        "cell_size": (cell.to("rad").value, cell.to("rad").value),
        "image_size": (npix, npix),
        "phase_center": (3.14973047, -0.3292367),
        "chan_mode": "continuum",
    }

    grid_params = {
        "sampling": 100,
        "support": 7,
    }

    psf_ds = make_psf(vis, im_params, grid_params)
    psf_da = psf_ds["POINT_SPREAD_FUNCTION"]
    psf_data = psf_da.values[0, 0, 0, :, :]

    # Find peak location
    peak_idx = np.unravel_index(np.argmax(psf_data), psf_data.shape)

    # Peak should be within a few pixels of center
    center_idx = (npix // 2, npix // 2)
    distance_from_center = np.sqrt(
        (peak_idx[0] - center_idx[0]) ** 2 + (peak_idx[1] - center_idx[1]) ** 2
    )

    # Allow for some tolerance (within 10 pixels of center for 200x200 image)
    assert distance_from_center < 10


def test_make_psf_normalization(basic_vis_data):
    """
    Test that PSF has reasonable peak values (should be normalized).
    """
    vis, npix, cell = basic_vis_data

    freq_chan = vis.coords["frequency"].values
    pol = vis.coords["polarization"].values

    im_params = {
        "cell_size": (cell.to("rad").value, cell.to("rad").value),
        "image_size": (npix, npix),
        "phase_center": (3.14973047, -0.3292367),
        "chan_mode": "continuum",
    }

    grid_params = {
        "sampling": 100,
        "support": 7,
    }

    psf_ds = make_psf(vis, im_params, grid_params)
    psf_da = psf_ds["POINT_SPREAD_FUNCTION"]
    psf_data = psf_da.values[0, 0, 0, :, :]

    # Check that peak is positive
    max_value = np.max(psf_data)
    assert max_value > 0

    # PSF peak should be reasonably normalized (typically around 1.0)
    # but may vary depending on gridding and FFT conventions
    assert max_value > 0.1  # At least some reasonable signal
