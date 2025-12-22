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


if __name__ == "__main__":
    unittest.main()
