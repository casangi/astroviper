from astroviper.core.imaging.imaging_utils.make_primary_beam import (
    cube_single_field_primary_beam,
)
from astroviper.core.imaging.imaging_utils.make_pb_symmetric import (
    airy_disk_rorder,
    casa_airy_disk_rorder,
)
from xradio.image import make_empty_sky_image
import xarray as xr
import numpy as np
import pytest


def test_cube_single_field_primary_beam():
    """
    Test cube_single_field_primary_beam function with ALMA telescope and default model.
    """

    # Define imaging parameters
    im_params = {
        "cell_size": (0.00001, 0.00001),  # in radians
        "image_size": (128, 128),
        "image_center": (64, 64),
        "frequency_coords": [1e11, 1.1e11],  # in Hz
        "polarization": ["I"],  # Stokes I
        "phase_center": (5.233697011339746, 0.7109380541842404),  # RA, Dec in radians
        "time_coords": [0.0],  # in seconds
    }

    pb_image = cube_single_field_primary_beam(im_params, telescope="ALMA")

    # Verify the output is an xarray Dataset
    assert isinstance(pb_image, xr.Dataset)

    # Verify PRIMARY_BEAM data variable exists
    assert "PRIMARY_BEAM" in pb_image.data_vars

    # Verify dimensions
    expected_dims = ("time", "frequency", "polarization", "l", "m")
    assert pb_image["PRIMARY_BEAM"].dims == expected_dims

    # Verify shape matches input parameters
    assert pb_image["PRIMARY_BEAM"].shape == (1, 2, 1, 128, 128)

    # Verify coordinate dimensions
    assert len(pb_image.frequency) == 2
    assert len(pb_image.polarization) == 1
    assert len(pb_image.time) == 1

    # Verify primary beam values are in expected range [0, 1]
    pb_data = pb_image["PRIMARY_BEAM"].values
    assert np.all(pb_data >= 0.0)
    assert np.all(pb_data <= 1.0)

    # Verify center pixel has maximum value (close to 1.0)
    center_l = 64
    center_m = 64
    center_value = pb_data[0, 0, 0, center_l, center_m]
    assert center_value > 0.9  # Center should be close to 1.0


def test_cube_single_field_primary_beam_aca():
    """
    Test cube_single_field_primary_beam function with ACA telescope.
    """

    im_params = {
        "cell_size": (-0.0001, 0.0001),
        "image_size": (64, 64),
        "image_center": (32, 32),
        "frequency_coords": [1e11, 1.1e11],
        "polarization": ["I"],
        "phase_center": (5.233697011339746, 0.7109380541842404),
        "time_coords": [0.0],
    }

    pb_image = cube_single_field_primary_beam(im_params, telescope="ACA")

    # Verify the output
    assert isinstance(pb_image, xr.Dataset)
    assert "PRIMARY_BEAM" in pb_image.data_vars
    assert pb_image["PRIMARY_BEAM"].shape == (1, 2, 1, 64, 64)

    # Verify primary beam values are valid
    pb_data = pb_image["PRIMARY_BEAM"].values
    assert np.all(pb_data >= 0.0)
    assert np.all(pb_data <= 1.0)


def test_cube_single_field_primary_beam_airy_disk_model():
    """
    Test cube_single_field_primary_beam with airy_disk model (non-CASA).
    """

    im_params = {
        "cell_size": (-0.0001, 0.0001),
        "image_size": (64, 64),
        "image_center": (32, 32),
        "frequency_coords": [1e11],
        "polarization": ["I"],
        "phase_center": (5.233697011339746, 0.7109380541842404),
        "time_coords": [0.0],
    }

    pb_image = cube_single_field_primary_beam(
        im_params, telescope="ALMA", model="airy_disk"
    )

    # Verify the output
    assert isinstance(pb_image, xr.Dataset)
    assert "PRIMARY_BEAM" in pb_image.data_vars
    assert pb_image["PRIMARY_BEAM"].shape == (1, 1, 1, 64, 64)

    # Verify primary beam values are valid
    pb_data = pb_image["PRIMARY_BEAM"].values
    assert np.all(pb_data >= 0.0)
    assert np.all(pb_data <= 1.0)


def test_cube_single_field_primary_beam_multiple_polarizations():
    """
    Test cube_single_field_primary_beam with multiple polarizations.
    """

    im_params = {
        "cell_size": (-0.00001, 0.00001),
        "image_size": (64, 64),
        "image_center": (32, 32),
        "frequency_coords": [1e11],
        "polarization": ["I", "Q", "U", "V"],  # Stokes I, Q, U, V
        "phase_center": (0.0, 0.0),
        "time_coords": [0.0],
    }

    pb_image = cube_single_field_primary_beam(im_params, telescope="ALMA")

    # Verify shape includes all polarizations
    assert pb_image["PRIMARY_BEAM"].shape == (1, 1, 4, 64, 64)
    assert len(pb_image.polarization) == 4

    # Verify all polarizations have valid values
    pb_data = pb_image["PRIMARY_BEAM"].values
    assert np.all(pb_data >= 0.0)
    assert np.all(pb_data <= 1.0)


def test_cube_single_field_primary_beam_multiple_channels():
    """
    Test cube_single_field_primary_beam with multiple frequency channels.
    """

    im_params = {
        "cell_size": (-0.00001, 0.00001),
        "image_size": (64, 64),
        "image_center": (32, 32),
        "frequency_coords": [1.00e11, 1.01e11, 1.02e11, 1.03e11],  # 4 channels
        "polarization": ["I"],
        "phase_center": (5.233697011339746, 0.7109380541842404),
        "time_coords": [0.0],
    }

    pb_image = cube_single_field_primary_beam(im_params, telescope="ALMA")

    # Verify shape includes all frequency channels
    assert pb_image["PRIMARY_BEAM"].shape == (1, 4, 1, 64, 64)
    assert len(pb_image.frequency) == 4

    # Verify frequency coordinates match input
    np.testing.assert_array_equal(
        pb_image.frequency.values, np.array([1.00e11, 1.01e11, 1.02e11, 1.03e11])
    )

    # Verify primary beam changes with frequency (lower frequency = wider beam)
    pb_data = pb_image["PRIMARY_BEAM"].values
    center_l = 32
    center_m = 32
    edge_l = 10
    edge_m = 32

    # At lower frequency, edge value should be higher (wider beam)
    edge_value_100ghz = pb_data[0, 0, 0, edge_l, edge_m]
    edge_value_103ghz = pb_data[0, 3, 0, edge_l, edge_m]
    assert edge_value_100ghz > edge_value_103ghz


def test_cube_single_field_primary_beam_different_image_sizes():
    """
    Test cube_single_field_primary_beam with different image sizes.
    """

    for size in [(32, 32), (64, 64), (128, 256)]:
        im_params = {
            "cell_size": (0.0001, 0.0001),
            "image_size": size,
            "image_center": (size[0] // 2, size[1] // 2),
            "frequency_coords": [100e9],
            "polarization": [0],
            "phase_center": (0.0, 0.0),
            "time_coords": [0.0],
        }

        pb_image = cube_single_field_primary_beam(im_params, telescope="ALMA")

        # Verify shape matches requested image size
        assert pb_image["PRIMARY_BEAM"].shape == (1, 1, 1, size[0], size[1])


def test_cube_single_field_primary_beam_different_cell_sizes():
    """
    Test cube_single_field_primary_beam with different cell sizes.
    """

    for cell_size in [(0.0001, 0.0001), (0.0005, 0.0005), (0.001, 0.001)]:
        im_params = {
            "cell_size": cell_size,
            "image_size": (64, 64),
            "image_center": (32, 32),
            "frequency_coords": [100e9],
            "polarization": [0],
            "phase_center": (0.0, 0.0),
            "time_coords": [0.0],
        }

        pb_image = cube_single_field_primary_beam(im_params, telescope="ALMA")

        # Verify output is created
        assert isinstance(pb_image, xr.Dataset)
        assert "PRIMARY_BEAM" in pb_image.data_vars

        # Verify primary beam values are valid
        pb_data = pb_image["PRIMARY_BEAM"].values
        assert np.all(pb_data >= 0.0)
        assert np.all(pb_data <= 1.0)


def test_cube_single_field_primary_beam_unsupported_telescope():
    """
    Test that unsupported telescope raises ValueError.
    """

    im_params = {
        "cell_size": (0.0001, 0.0001),
        "image_size": (64, 64),
        "image_center": (32, 32),
        "frequency_coords": [1.00e11],
        "polarization": ["I"],
        "phase_center": (5.233697011339746, 0.7109380541842404),
        "time_coords": [0.0],
    }

    with pytest.raises(ValueError, match="Unsupported telescope"):
        cube_single_field_primary_beam(im_params, telescope="VLA")


def test_cube_single_field_primary_beam_symmetry():
    """
    Test that primary beam is symmetric around the center.
    """

    im_params = {
        "cell_size": (0.0001, 0.0001),
        "image_size": (129, 129),
        "image_center": (64, 64),
        "frequency_coords": [1e11],
        "polarization": ["I"],
        "phase_center": (5.233697011339746, 0.7109380541842404),
        "time_coords": [0.0],
    }

    pb_image = cube_single_field_primary_beam(im_params, telescope="ALMA")
    pb_data = pb_image["PRIMARY_BEAM"].values[0, 0, 0, :, :]

    center = 64

    # Check symmetry along horizontal and vertical axes
    # Left vs Right
    np.testing.assert_array_almost_equal(
        pb_data[center, :center], pb_data[center, center + 1 :][::-1], decimal=5
    )

    # Top vs Bottom
    np.testing.assert_array_almost_equal(
        pb_data[:center, center], pb_data[center + 1 :, center][::-1], decimal=5
    )


def test_cube_single_field_primary_beam_coordinates():
    """
    Test that output coordinates match input parameters.
    """

    frequency_coords = [1.0e11, 1.05e11, 1.10e11]
    time_coords = [0.0, 100.0]

    im_params = {
        "cell_size": (0.0001, 0.0001),
        "image_size": (64, 64),
        "image_center": (32, 32),
        "frequency_coords": frequency_coords,
        "polarization": ["I", "Q", "U", "V"],
        "phase_center": (5.233697011339746, 0.7109380541842404),
        "time_coords": time_coords,
    }

    pb_image = cube_single_field_primary_beam(im_params, telescope="ALMA")

    # Verify frequency coordinates
    np.testing.assert_array_equal(pb_image.frequency.values, np.array(frequency_coords))

    # Verify time coordinates
    np.testing.assert_array_equal(pb_image.time.values, np.array(time_coords))

    # Verify polarization coordinates
    np.testing.assert_array_equal(
        pb_image.polarization.values, np.array(["I", "Q", "U", "V"])
    )
