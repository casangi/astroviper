"""
Unit tests for correlation product to Stokes parameter conversions.

Tests both corr_to_stokes and stokes_to_corr functions for:
- Linear polarization (XX, XY, YX, YY)
- Circular polarization (RR, RL, LR, LL)
- Custom transformation matrices
- Round-trip conversions
- Edge cases and error handling
- Compatibility with numpy arrays and xarray DataArrays
"""

import numpy as np
import xarray as xr
import unittest

from astroviper.core.imaging.imaging_utils.corr_to_stokes import (
    corr_to_stokes,
    stokes_to_corr,
    image_stokes_to_corr,
    image_corr_to_stokes,
)


class TestLinearPolarization(unittest.TestCase):
    """Test linear polarization conversions (XX, XY, YX, YY)."""

    def setUp(self):
        """Set up test data for linear polarization."""
        # Create test correlation data [XX, XY, YX, YY]
        self.corr_linear = np.array([1.0, 0.2 + 0.1j, 0.2 - 0.1j, 0.8])

        # Expected Stokes parameters [I, Q, U, V]
        # I = XX + YY = 1.0 + 0.8 = 1.8
        # Q = XX - YY = 1.0 - 0.8 = 0.2
        # U = XY + YX = (0.2+0.1j) + (0.2-0.1j) = 0.4
        # V = i(YX - XY) = i((0.2-0.1j) - (0.2+0.1j)) = i(-0.2j) = 0.2
        self.expected_stokes = np.array([1.8, 0.2, 0.4, 0.2])

    def test_corr_to_stokes_linear(self):
        """Test conversion from linear correlation to Stokes."""
        stokes = corr_to_stokes(self.corr_linear, corr_type="linear")

        np.testing.assert_allclose(
            stokes,
            self.expected_stokes,
            rtol=1e-10,
            err_msg="Linear correlation to Stokes conversion failed",
        )

    def test_stokes_to_corr_linear(self):
        """Test conversion from Stokes to linear correlation."""
        # First convert to Stokes
        stokes = corr_to_stokes(self.corr_linear, corr_type="linear")

        # Then convert back to correlation
        corr_reconstructed = stokes_to_corr(stokes, corr_type="linear")

        np.testing.assert_allclose(
            corr_reconstructed,
            self.corr_linear,
            rtol=1e-10,
            err_msg="Stokes to linear correlation conversion failed",
        )

    def test_round_trip_linear(self):
        """Test round-trip conversion corr -> stokes -> corr."""
        stokes = corr_to_stokes(self.corr_linear, corr_type="linear")
        corr_reconstructed = stokes_to_corr(stokes, corr_type="linear")

        np.testing.assert_allclose(
            corr_reconstructed,
            self.corr_linear,
            rtol=1e-10,
            err_msg="Linear round-trip conversion failed",
        )

    def test_multidimensional_linear(self):
        """Test conversion with multidimensional arrays."""
        # Create 3D array (time, baseline, correlation)
        ntime = 5
        nbaseline = 10
        corr_3d = np.random.rand(ntime, nbaseline, 4) + 1j * np.random.rand(
            ntime, nbaseline, 4
        )

        stokes_3d = corr_to_stokes(corr_3d, corr_type="linear")
        corr_reconstructed = stokes_to_corr(stokes_3d, corr_type="linear")

        self.assertEqual(
            stokes_3d.shape,
            (ntime, nbaseline, 4),
            "Output shape mismatch for 3D array",
        )
        np.testing.assert_allclose(
            corr_reconstructed,
            corr_3d,
            rtol=1e-10,
            err_msg="3D linear round-trip conversion failed",
        )

    def test_real_valued_correlations(self):
        """Test with real-valued correlation products (auto-correlations)."""
        corr_real = np.array([1.0, 0.0, 0.0, 0.8])
        stokes = corr_to_stokes(corr_real, corr_type="linear")

        # For real correlations: I=1.8, Q=0.2, U=0, V=0
        expected = np.array([1.8, 0.2, 0.0, 0.0])
        np.testing.assert_allclose(
            stokes, expected, rtol=1e-10, err_msg="Real correlation conversion failed"
        )


class TestCircularPolarization(unittest.TestCase):
    """Test circular polarization conversions (RR, RL, LR, LL)."""

    def setUp(self):
        """Set up test data for circular polarization."""
        # Create test correlation data [RR, RL, LR, LL]
        self.corr_circular = np.array([1.0, 0.1 + 0.05j, 0.1 - 0.05j, 0.8])

        # Expected Stokes parameters [I, Q, U, V]
        # I = RR + LL = 1.0 + 0.8 = 1.8
        # Q = RL + LR = (0.1+0.05j) + (0.1-0.05j) = 0.2
        # U = i(LR - RL) = i((0.1-0.05j) - (0.1+0.05j)) = i(-0.1j) = 0.1
        # V = RR - LL = 1.0 - 0.8 = 0.2
        self.expected_stokes = np.array([1.8, 0.2, 0.1, 0.2])

    def test_corr_to_stokes_circular(self):
        """Test conversion from circular correlation to Stokes."""
        stokes = corr_to_stokes(self.corr_circular, corr_type="circular")

        np.testing.assert_allclose(
            stokes,
            self.expected_stokes,
            rtol=1e-10,
            err_msg="Circular correlation to Stokes conversion failed",
        )

    def test_stokes_to_corr_circular(self):
        """Test conversion from Stokes to circular correlation."""
        stokes = corr_to_stokes(self.corr_circular, corr_type="circular")
        corr_reconstructed = stokes_to_corr(stokes, corr_type="circular")

        np.testing.assert_allclose(
            corr_reconstructed,
            self.corr_circular,
            rtol=1e-10,
            err_msg="Stokes to circular correlation conversion failed",
        )

    def test_round_trip_circular(self):
        """Test round-trip conversion corr -> stokes -> corr."""
        stokes = corr_to_stokes(self.corr_circular, corr_type="circular")
        corr_reconstructed = stokes_to_corr(stokes, corr_type="circular")

        np.testing.assert_allclose(
            corr_reconstructed,
            self.corr_circular,
            rtol=1e-10,
            err_msg="Circular round-trip conversion failed",
        )

    def test_multidimensional_circular(self):
        """Test conversion with multidimensional arrays."""
        # Create 3D array
        ntime = 5
        nbaseline = 10
        corr_3d = np.random.rand(ntime, nbaseline, 4) + 1j * np.random.rand(
            ntime, nbaseline, 4
        )

        stokes_3d = corr_to_stokes(corr_3d, corr_type="circular")
        corr_reconstructed = stokes_to_corr(stokes_3d, corr_type="circular")

        np.testing.assert_allclose(
            corr_reconstructed,
            corr_3d,
            rtol=1e-10,
            err_msg="3D circular round-trip conversion failed",
        )


class TestCustomTransformationMatrix(unittest.TestCase):
    """Test custom transformation matrices."""

    def test_custom_matrix_corr_to_stokes(self):
        """Test corr_to_stokes with custom transformation matrix."""
        # Use the linear matrix explicitly as custom
        custom_matrix = np.array(
            [
                [1, 0, 0, 1],  # I = XX + YY
                [1, 0, 0, -1],  # Q = XX - YY
                [0, 1, 1, 0],  # U = XY + YX
                [0, -1j, 1j, 0],  # V = i(YX - XY)
            ]
        )

        corr_data = np.array([1.0, 0.2 + 0.1j, 0.2 - 0.1j, 0.8])
        stokes_custom = corr_to_stokes(
            corr_data, corr_type="custom", transformation_matrix=custom_matrix
        )
        stokes_linear = corr_to_stokes(corr_data, corr_type="linear")

        np.testing.assert_allclose(
            stokes_custom,
            stokes_linear,
            rtol=1e-10,
            err_msg="Custom matrix should match linear conversion",
        )

    def test_custom_matrix_stokes_to_corr(self):
        """Test stokes_to_corr with custom transformation matrix."""
        # Use the linear inverse matrix explicitly as custom
        custom_matrix = np.array(
            [
                [0.5, 0.5, 0, 0],  # XX = (I + Q)/2
                [0, 0, 0.5, 0.5j],  # XY = (U + iV)/2
                [0, 0, 0.5, -0.5j],  # YX = (U - iV)/2
                [0.5, -0.5, 0, 0],  # YY = (I - Q)/2
            ]
        )

        stokes_data = np.array([1.8, 0.2, 0.4, 0.2])
        corr_custom = stokes_to_corr(
            stokes_data, corr_type="custom", transformation_matrix=custom_matrix
        )
        corr_linear = stokes_to_corr(stokes_data, corr_type="linear")

        np.testing.assert_allclose(
            corr_custom,
            corr_linear,
            rtol=1e-10,
            err_msg="Custom matrix should match linear inverse conversion",
        )

    def test_custom_matrix_required(self):
        """Test that custom corr_type requires transformation_matrix."""
        corr_data = np.array([1.0, 0.2, 0.2, 0.8])

        with self.assertRaises(ValueError) as context:
            corr_to_stokes(corr_data, corr_type="custom")

        self.assertIn("transformation_matrix must be provided", str(context.exception))

        with self.assertRaises(ValueError) as context:
            stokes_to_corr(corr_data, corr_type="custom")

        self.assertIn("transformation_matrix must be provided", str(context.exception))


class TestXarrayCompatibility(unittest.TestCase):
    """Test compatibility with xarray DataArrays."""

    def test_xarray_linear(self):
        """Test linear conversion with xarray DataArray."""
        # Create xarray DataArray
        corr_xr = xr.DataArray(
            np.random.rand(5, 4) + 1j * np.random.rand(5, 4),
            dims=["time", "corr"],
            coords={"corr": ["XX", "XY", "YX", "YY"]},
        )

        stokes = corr_to_stokes(corr_xr, corr_type="linear")
        corr_reconstructed = stokes_to_corr(stokes, corr_type="linear")

        # stokes and corr_reconstructed should be numpy arrays
        self.assertIsInstance(stokes, np.ndarray)
        self.assertIsInstance(corr_reconstructed, np.ndarray)

        np.testing.assert_allclose(
            corr_reconstructed,
            corr_xr.values,
            rtol=1e-10,
            err_msg="xarray round-trip conversion failed",
        )

    def test_xarray_circular(self):
        """Test circular conversion with xarray DataArray."""
        corr_xr = xr.DataArray(
            np.random.rand(5, 10, 4) + 1j * np.random.rand(5, 10, 4),
            dims=["time", "baseline", "corr"],
            coords={"corr": ["RR", "RL", "LR", "LL"]},
        )

        stokes = corr_to_stokes(corr_xr, corr_type="circular")
        corr_reconstructed = stokes_to_corr(stokes, corr_type="circular")

        np.testing.assert_allclose(
            corr_reconstructed,
            corr_xr.values,
            rtol=1e-10,
            err_msg="xarray circular round-trip failed",
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_zero_correlations(self):
        """Test with all-zero correlation products."""
        corr_zero = np.zeros(4, dtype=complex)
        stokes = corr_to_stokes(corr_zero, corr_type="linear")
        corr_reconstructed = stokes_to_corr(stokes, corr_type="linear")

        np.testing.assert_allclose(
            corr_reconstructed,
            corr_zero,
            rtol=1e-10,
            err_msg="Zero correlation conversion failed",
        )

    def test_unpolarized_source(self):
        """Test unpolarized source (Q=U=V=0, only Stokes I)."""
        # For unpolarized: XX = YY, XY = YX = 0
        corr_unpolarized = np.array([1.0, 0.0, 0.0, 1.0])
        stokes = corr_to_stokes(corr_unpolarized, corr_type="linear")

        # Should get I=2, Q=0, U=0, V=0
        expected = np.array([2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            stokes, expected, rtol=1e-10, err_msg="Unpolarized source failed"
        )

        # Round-trip should work
        corr_reconstructed = stokes_to_corr(stokes, corr_type="linear")
        np.testing.assert_allclose(corr_reconstructed, corr_unpolarized, rtol=1e-10)

    def test_fully_polarized_linear(self):
        """Test fully linearly polarized source (Q=I, U=V=0)."""
        # XX = I, YY = 0
        corr_linear_pol = np.array([2.0, 0.0, 0.0, 0.0])
        stokes = corr_to_stokes(corr_linear_pol, corr_type="linear")

        # Should get I=2, Q=2, U=0, V=0
        expected = np.array([2.0, 2.0, 0.0, 0.0])
        np.testing.assert_allclose(stokes, expected, rtol=1e-10)

        # Round-trip
        corr_reconstructed = stokes_to_corr(stokes, corr_type="linear")
        np.testing.assert_allclose(corr_reconstructed, corr_linear_pol, rtol=1e-10)

    def test_fully_polarized_circular(self):
        """Test fully circularly polarized source (V=I, Q=U=0)."""
        # RR = I, LL = 0
        corr_circular_pol = np.array([2.0, 0.0, 0.0, 0.0])
        stokes = corr_to_stokes(corr_circular_pol, corr_type="circular")

        # Should get I=2, Q=0, U=0, V=2
        expected = np.array([2.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(stokes, expected, rtol=1e-10)

        # Round-trip
        corr_reconstructed = stokes_to_corr(stokes, corr_type="circular")
        np.testing.assert_allclose(corr_reconstructed, corr_circular_pol, rtol=1e-10)

    def test_single_correlation_product(self):
        """Test with 1D array (single measurement)."""
        corr_single = np.array([1.0, 0.5 + 0.3j, 0.5 - 0.3j, 0.7])
        stokes = corr_to_stokes(corr_single, corr_type="linear")
        corr_reconstructed = stokes_to_corr(stokes, corr_type="linear")

        self.assertEqual(stokes.shape, (4,))
        np.testing.assert_allclose(corr_reconstructed, corr_single, rtol=1e-10)


class TestMatrixProperties(unittest.TestCase):
    """Test mathematical properties of transformation matrices."""

    def test_linear_matrices_are_inverses(self):
        """Test that linear transformation matrices are proper inverses."""
        corr_to_stokes_matrix = np.array(
            [
                [1, 0, 0, 1],  # I
                [1, 0, 0, -1],  # Q
                [0, 1, 1, 0],  # U
                [0, -1j, 1j, 0],  # V
            ]
        )

        stokes_to_corr_matrix = np.array(
            [
                [0.5, 0.5, 0, 0],  # XX
                [0, 0, 0.5, 0.5j],  # XY
                [0, 0, 0.5, -0.5j],  # YX
                [0.5, -0.5, 0, 0],  # YY
            ]
        )

        # Matrix product should be identity
        product = stokes_to_corr_matrix @ corr_to_stokes_matrix
        np.testing.assert_allclose(
            product, np.eye(4), rtol=1e-10, err_msg="Linear matrices are not inverses"
        )

    def test_circular_matrices_are_inverses(self):
        """Test that circular transformation matrices are proper inverses."""
        corr_to_stokes_matrix = np.array(
            [
                [1, 0, 0, 1],  # I
                [0, 1, 1, 0],  # Q
                [0, -1j, 1j, 0],  # U
                [1, 0, 0, -1],  # V
            ]
        )

        stokes_to_corr_matrix = np.array(
            [
                [0.5, 0, 0, 0.5],  # RR
                [0, 0.5, 0.5j, 0],  # RL
                [0, 0.5, -0.5j, 0],  # LR
                [0.5, 0, 0, -0.5],  # LL
            ]
        )

        # Matrix product should be identity
        product = stokes_to_corr_matrix @ corr_to_stokes_matrix
        np.testing.assert_allclose(
            product,
            np.eye(4),
            rtol=1e-10,
            err_msg="Circular matrices are not inverses",
        )


class TestImageConversions(unittest.TestCase):
    """Test high-level image conversion functions."""

    def test_image_corr_to_stokes_numpy_linear(self):
        """Test image_corr_to_stokes with numpy array (linear polarization)."""
        # Create synthetic image: (time, freq, pol, l, m) = (1, 64, 4, 128, 128)
        ntime, nfreq, npol, nl, nm = 1, 64, 4, 128, 128

        # Random correlation image
        corr_image = np.random.rand(ntime, nfreq, npol, nl, nm) + 1j * np.random.rand(
            ntime, nfreq, npol, nl, nm
        )

        # Convert to Stokes
        stokes_image = image_corr_to_stokes(corr_image, corr_type="linear")

        # Check shape preserved
        self.assertEqual(stokes_image.shape, corr_image.shape)

        # Round-trip conversion
        corr_reconstructed = image_stokes_to_corr(stokes_image, corr_type="linear")

        np.testing.assert_allclose(
            corr_reconstructed,
            corr_image,
            rtol=1e-10,
            err_msg="Image linear round-trip failed",
        )

    def test_image_corr_to_stokes_numpy_circular(self):
        """Test image_corr_to_stokes with numpy array (circular polarization)."""
        # Create synthetic image
        ntime, nfreq, npol, nl, nm = 1, 32, 4, 64, 64

        corr_image = np.random.rand(ntime, nfreq, npol, nl, nm) + 1j * np.random.rand(
            ntime, nfreq, npol, nl, nm
        )

        # Convert to Stokes and back
        stokes_image = image_corr_to_stokes(corr_image, corr_type="circular")
        corr_reconstructed = image_stokes_to_corr(stokes_image, corr_type="circular")

        np.testing.assert_allclose(
            corr_reconstructed,
            corr_image,
            rtol=1e-10,
            err_msg="Image circular round-trip failed",
        )

    def test_image_different_pol_axis(self):
        """Test image conversion with polarization at different axis positions."""
        # Shape: (pol, freq, l, m) = (4, 64, 128, 128)
        # Polarization at axis 0
        npol, nfreq, nl, nm = 4, 64, 128, 128

        corr_image = np.random.rand(npol, nfreq, nl, nm) + 1j * np.random.rand(
            npol, nfreq, nl, nm
        )

        # Convert with pol_axis=0
        stokes_image = image_corr_to_stokes(corr_image, corr_type="linear", pol_axis=0)
        corr_reconstructed = image_stokes_to_corr(
            stokes_image, corr_type="linear", pol_axis=0
        )

        self.assertEqual(stokes_image.shape, corr_image.shape)
        np.testing.assert_allclose(
            corr_reconstructed,
            corr_image,
            rtol=1e-10,
            err_msg="Image conversion with pol_axis=0 failed",
        )

    def test_image_negative_pol_axis(self):
        """Test image conversion with negative pol_axis indexing."""
        # Shape: (time, freq, l, m, pol) = (1, 32, 64, 64, 4)
        # Polarization at last axis (index -1)
        ntime, nfreq, nl, nm, npol = 1, 32, 64, 64, 4

        corr_image = np.random.rand(ntime, nfreq, nl, nm, npol) + 1j * np.random.rand(
            ntime, nfreq, nl, nm, npol
        )

        # Convert with pol_axis=-1
        stokes_image = image_corr_to_stokes(corr_image, corr_type="linear", pol_axis=-1)
        corr_reconstructed = image_stokes_to_corr(
            stokes_image, corr_type="linear", pol_axis=-1
        )

        np.testing.assert_allclose(
            corr_reconstructed,
            corr_image,
            rtol=1e-10,
            err_msg="Image conversion with pol_axis=-1 failed",
        )

    def test_image_large_realistic_shape(self):
        """Test with realistic large image shape."""
        # Realistic shape: (time, freq, pol, l, m) = (1, 128, 4, 512, 512)
        # This is ~500 MB for complex128
        ntime, nfreq, npol, nl, nm = 1, 128, 4, 512, 512

        # Use smaller dtype to save memory in tests
        corr_image = np.random.rand(ntime, nfreq, npol, nl, nm).astype(
            np.float32
        ) + 1j * np.random.rand(ntime, nfreq, npol, nl, nm).astype(np.float32)

        # Convert to Stokes
        stokes_image = image_corr_to_stokes(corr_image, corr_type="linear")

        # Verify shape
        self.assertEqual(stokes_image.shape, corr_image.shape)

        # Spot check: verify one pixel's conversion
        # Extract a single pixel across all dimensions
        pixel_corr = corr_image[0, 0, :, 0, 0]
        pixel_stokes_expected = corr_to_stokes(pixel_corr, corr_type="linear")
        pixel_stokes_actual = stokes_image[0, 0, :, 0, 0]

        np.testing.assert_allclose(
            pixel_stokes_actual,
            pixel_stokes_expected,
            rtol=1e-5,  # Slightly relaxed for float32
            err_msg="Large image pixel conversion mismatch",
        )


class TestImageConversionsXarray(unittest.TestCase):
    """Test high-level image conversion functions with xarray DataArrays."""

    def test_image_xarray_linear_preserves_structure(self):
        """Test that xarray input preserves dims, coords, and attrs."""
        # Create xarray image DataArray
        ntime, nfreq, npol, nl, nm = 1, 64, 4, 128, 128

        data = np.random.rand(ntime, nfreq, npol, nl, nm) + 1j * np.random.rand(
            ntime, nfreq, npol, nl, nm
        )

        image_da = xr.DataArray(
            data,
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "time": np.arange(ntime),
                "frequency": np.linspace(1e9, 2e9, nfreq),
                "polarization": ["XX", "XY", "YX", "YY"],
                "l": np.linspace(-0.1, 0.1, nl),
                "m": np.linspace(-0.1, 0.1, nm),
            },
            attrs={"telescope": "VLA", "field": "test_field"},
        )

        # Convert to Stokes
        stokes_da = image_corr_to_stokes(image_da, corr_type="linear")

        # Check type
        self.assertIsInstance(stokes_da, xr.DataArray)

        # Check dims preserved
        self.assertEqual(stokes_da.dims, image_da.dims)

        # Check shape preserved
        self.assertEqual(stokes_da.shape, image_da.shape)

        # Check polarization coordinates updated
        np.testing.assert_array_equal(
            stokes_da.coords["polarization"].values, ["I", "Q", "U", "V"]
        )

        # Check other coordinates preserved
        np.testing.assert_allclose(
            stokes_da.coords["frequency"].values, image_da.coords["frequency"].values
        )
        np.testing.assert_allclose(
            stokes_da.coords["l"].values, image_da.coords["l"].values
        )
        np.testing.assert_allclose(
            stokes_da.coords["m"].values, image_da.coords["m"].values
        )

        # Check attributes preserved
        self.assertEqual(stokes_da.attrs["telescope"], "VLA")
        self.assertEqual(stokes_da.attrs["field"], "test_field")

    def test_image_xarray_round_trip(self):
        """Test round-trip conversion with xarray preserves data."""
        ntime, nfreq, npol, nl, nm = 1, 32, 4, 64, 64

        data = np.random.rand(ntime, nfreq, npol, nl, nm) + 1j * np.random.rand(
            ntime, nfreq, npol, nl, nm
        )

        corr_da = xr.DataArray(
            data,
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "time": [0],
                "frequency": np.linspace(1e9, 1.5e9, nfreq),
                "polarization": ["XX", "XY", "YX", "YY"],
                "l": np.linspace(-0.05, 0.05, nl),
                "m": np.linspace(-0.05, 0.05, nm),
            },
        )

        # Round-trip: corr -> stokes -> corr
        stokes_da = image_corr_to_stokes(corr_da, corr_type="linear")
        corr_reconstructed = image_stokes_to_corr(stokes_da, corr_type="linear")

        # Check data matches
        np.testing.assert_allclose(
            corr_reconstructed.values,
            corr_da.values,
            rtol=1e-10,
            err_msg="xarray round-trip data mismatch",
        )

        # Check polarization coords restored
        np.testing.assert_array_equal(
            corr_reconstructed.coords["polarization"].values, ["XX", "XY", "YX", "YY"]
        )

    def test_image_xarray_circular(self):
        """Test xarray image conversion with circular polarization."""
        ntime, nfreq, npol, nl, nm = 1, 16, 4, 32, 32

        data = np.random.rand(ntime, nfreq, npol, nl, nm) + 1j * np.random.rand(
            ntime, nfreq, npol, nl, nm
        )

        corr_da = xr.DataArray(
            data,
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "time": [0],
                "frequency": np.linspace(1e9, 1.2e9, nfreq),
                "polarization": ["RR", "RL", "LR", "LL"],
                "l": np.linspace(-0.02, 0.02, nl),
                "m": np.linspace(-0.02, 0.02, nm),
            },
        )

        # Convert circular -> Stokes
        stokes_da = image_corr_to_stokes(corr_da, corr_type="circular")

        # Check polarization coords
        np.testing.assert_array_equal(
            stokes_da.coords["polarization"].values, ["I", "Q", "U", "V"]
        )

        # Convert back: Stokes -> circular
        corr_reconstructed = image_stokes_to_corr(stokes_da, corr_type="circular")

        # Check polarization coords
        np.testing.assert_array_equal(
            corr_reconstructed.coords["polarization"].values, ["RR", "RL", "LR", "LL"]
        )

        # Check data
        np.testing.assert_allclose(
            corr_reconstructed.values,
            corr_da.values,
            rtol=1e-10,
            err_msg="xarray circular round-trip failed",
        )

    def test_image_xarray_different_pol_axis(self):
        """Test xarray conversion with polarization at different axis."""
        # Shape: (pol, freq, l, m) with pol_axis=0
        npol, nfreq, nl, nm = 4, 32, 64, 64

        data = np.random.rand(npol, nfreq, nl, nm) + 1j * np.random.rand(
            npol, nfreq, nl, nm
        )

        corr_da = xr.DataArray(
            data,
            dims=["polarization", "frequency", "l", "m"],
            coords={
                "polarization": ["XX", "XY", "YX", "YY"],
                "frequency": np.linspace(1e9, 1.5e9, nfreq),
                "l": np.linspace(-0.05, 0.05, nl),
                "m": np.linspace(-0.05, 0.05, nm),
            },
        )

        # Convert with pol_axis=0
        stokes_da = image_corr_to_stokes(corr_da, corr_type="linear", pol_axis=0)
        corr_reconstructed = image_stokes_to_corr(
            stokes_da, corr_type="linear", pol_axis=0
        )

        # Verify
        self.assertEqual(stokes_da.dims[0], "polarization")
        np.testing.assert_array_equal(
            stokes_da.coords["polarization"].values, ["I", "Q", "U", "V"]
        )
        np.testing.assert_allclose(
            corr_reconstructed.values, corr_da.values, rtol=1e-10
        )

    def test_image_xarray_custom_matrix(self):
        """Test image conversion with custom transformation matrix."""
        ntime, nfreq, npol, nl, nm = 1, 16, 4, 32, 32

        data = np.random.rand(ntime, nfreq, npol, nl, nm) + 1j * np.random.rand(
            ntime, nfreq, npol, nl, nm
        )

        corr_da = xr.DataArray(
            data,
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "polarization": ["XX", "XY", "YX", "YY"],
            },
        )

        # Use linear matrix as custom
        custom_matrix = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 1, 1, 0],
                [0, -1j, 1j, 0],
            ]
        )

        # Convert with custom matrix
        stokes_custom = image_corr_to_stokes(
            corr_da, corr_type="custom", transformation_matrix=custom_matrix
        )

        # Convert with linear (should match)
        stokes_linear = image_corr_to_stokes(corr_da, corr_type="linear")

        np.testing.assert_allclose(
            stokes_custom.values,
            stokes_linear.values,
            rtol=1e-10,
            err_msg="Custom matrix result should match linear",
        )


class TestImageConversionCorrectness(unittest.TestCase):
    """Test correctness of image conversions against low-level functions."""

    def test_image_matches_lowlevel_per_pixel(self):
        """Verify image conversion matches low-level function pixel-by-pixel."""
        # Small image for detailed checking
        ntime, nfreq, npol, nl, nm = 2, 3, 4, 5, 5

        corr_image = np.random.rand(ntime, nfreq, npol, nl, nm) + 1j * np.random.rand(
            ntime, nfreq, npol, nl, nm
        )

        # Convert using image function
        stokes_image = image_corr_to_stokes(corr_image, corr_type="linear")

        # Verify each pixel independently using low-level function
        for t in range(ntime):
            for f in range(nfreq):
                for l in range(nl):
                    for m in range(nm):
                        # Extract polarization vector for this pixel
                        pixel_corr = corr_image[t, f, :, l, m]

                        # Convert using low-level function
                        pixel_stokes_expected = corr_to_stokes(
                            pixel_corr, corr_type="linear"
                        )

                        # Get from image conversion
                        pixel_stokes_actual = stokes_image[t, f, :, l, m]

                        np.testing.assert_allclose(
                            pixel_stokes_actual,
                            pixel_stokes_expected,
                            rtol=1e-10,
                            err_msg=f"Pixel mismatch at ({t}, {f}, {l}, {m})",
                        )

    def test_image_physical_meaning(self):
        """Test that image conversion preserves physical meaning of Stokes parameters."""
        # Create image with known polarization properties
        ntime, nfreq, npol, nl, nm = 1, 1, 4, 10, 10

        # Unpolarized source: XX = YY = 1, XY = YX = 0
        corr_unpolarized = np.zeros((ntime, nfreq, npol, nl, nm), dtype=complex)
        corr_unpolarized[:, :, 0, :, :] = 1.0  # XX
        corr_unpolarized[:, :, 3, :, :] = 1.0  # YY

        stokes = image_corr_to_stokes(corr_unpolarized, corr_type="linear")

        # Should get I=2, Q=0, U=0, V=0 everywhere
        np.testing.assert_allclose(stokes[:, :, 0, :, :], 2.0, rtol=1e-10)  # I
        np.testing.assert_allclose(stokes[:, :, 1, :, :], 0.0, atol=1e-10)  # Q
        np.testing.assert_allclose(stokes[:, :, 2, :, :], 0.0, atol=1e-10)  # U
        np.testing.assert_allclose(stokes[:, :, 3, :, :], 0.0, atol=1e-10)  # V


if __name__ == "__main__":
    unittest.main()
