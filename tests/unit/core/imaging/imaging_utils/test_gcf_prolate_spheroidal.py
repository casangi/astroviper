# -*- coding: utf-8 -*-
import numpy as np
import unittest
from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel,
    create_prolate_spheroidal_kernel_1D,
    prolate_spheroidal_function,
    coordinates,
    coordinates2,
)


class TestGcfProlateSpheroidal(unittest.TestCase):
    """Test suite for gcf_prolate_spheroidal module"""

    def test_create_prolate_spheroidal_kernel_even_oversampling(self):
        """Test create_prolate_spheroidal_kernel with even oversampling"""
        oversampling = 100
        support = 7
        n_uv = np.array([200, 200])

        kernel, kernel_image = create_prolate_spheroidal_kernel(
            oversampling, support, n_uv
        )

        # Check kernel shape for even oversampling
        expected_shape = (oversampling + 1, oversampling + 1, support, support)
        self.assertEqual(kernel.shape, expected_shape)

        # Check kernel image shape
        self.assertEqual(kernel_image.shape, (n_uv[0], n_uv[1]))

        # Kernel values should be real and non-negative for PSWF
        self.assertTrue(np.all(kernel >= 0))

    def test_create_prolate_spheroidal_kernel_odd_oversampling(self):
        """Test create_prolate_spheroidal_kernel with odd oversampling"""
        oversampling = 101
        support = 7
        n_uv = np.array([200, 200])

        kernel, kernel_image = create_prolate_spheroidal_kernel(
            oversampling, support, n_uv
        )

        # Check kernel shape for odd oversampling
        expected_shape = (oversampling, oversampling, support, support)
        self.assertEqual(kernel.shape, expected_shape)

        # Check kernel image shape
        self.assertEqual(kernel_image.shape, (n_uv[0], n_uv[1]))

        # Kernel values should be real and non-negative for PSWF
        self.assertTrue(np.all(kernel >= 0))

    def test_create_prolate_spheroidal_kernel_1D(self):
        """Test create_prolate_spheroidal_kernel_1D"""
        oversampling = 100
        support = 7

        kernel_1D = create_prolate_spheroidal_kernel_1D(oversampling, support)

        # Check output shape
        support_center = support // 2
        expected_length = oversampling * (support_center + 1)
        self.assertEqual(len(kernel_1D), expected_length)

        # Kernel should be non-negative at center
        self.assertGreaterEqual(kernel_1D[0], 0.0)

    def test_prolate_spheroidal_function_scalar(self):
        """Test prolate_spheroidal_function with scalar input"""
        u = 0.5

        grdsf, griddata = prolate_spheroidal_function(u)

        # grdsf should be positive within valid range
        self.assertGreater(float(grdsf), 0.0)

        # griddata should equal (1-u^2)*grdsf
        expected_griddata = (1 - u**2) * grdsf
        np.testing.assert_almost_equal(float(griddata), float(expected_griddata))

    def test_prolate_spheroidal_function_array(self):
        """Test prolate_spheroidal_function with array input"""
        u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        grdsf, griddata = prolate_spheroidal_function(u)

        # Check output shapes match input
        self.assertEqual(grdsf.shape, u.shape)
        self.assertEqual(griddata.shape, u.shape)

        # At u=0, grdsf should be positive
        self.assertGreater(grdsf[0], 0.0)

        # At u=1, both should approach 0
        self.assertAlmostEqual(griddata[4], 0.0, places=10)

    def test_prolate_spheroidal_function_out_of_range(self):
        """Test prolate_spheroidal_function with values outside [0,1]"""
        u = np.array([1.5, 2.0, -0.5])

        grdsf, griddata = prolate_spheroidal_function(u)

        # Values outside [0,1] should be zero
        self.assertEqual(grdsf[0], 0.0)
        self.assertEqual(grdsf[1], 0.0)
        # Negative values are made positive by np.abs
        self.assertGreater(grdsf[2], 0.0)

    def test_coordinates(self):
        """Test coordinates function"""
        npixel = 100

        coords = coordinates(npixel)

        # Check shape
        self.assertEqual(len(coords), npixel)

        # Check range: should span [-.5, .5)
        min_val = float(coords[0])
        max_val = float(coords[-1])
        self.assertAlmostEqual(min_val, -0.5)
        self.assertLess(max_val, 0.5)

        # Check that 0 is at position npixel//2
        center_val = float(coords[npixel // 2])
        self.assertAlmostEqual(center_val, 0.0)

    def test_coordinates2(self):
        """Test coordinates2 function"""
        npixel = 100

        coords = coordinates2(npixel)

        # Check shape: should be (2, npixel, npixel)
        self.assertEqual(coords.shape, (2, npixel, npixel))

        # Check that (0,0) is at pixel (floor(n/2), floor(n/2))
        center = npixel // 2
        center_x = float(coords[0, center, center])
        center_y = float(coords[1, center, center])
        self.assertAlmostEqual(center_x, 0.0)
        self.assertAlmostEqual(center_y, 0.0)

        # Check range: values should be in range based on step size 2/npixel
        # Maximum absolute value should be close to 1
        # Check a corner value instead of using .max()
        corner_val = abs(float(coords[0, 0, 0]))
        self.assertLess(corner_val, 1.0)

    def test_prolate_spheroidal_function_boundary_conditions(self):
        """Test prolate_spheroidal_function at boundary points"""
        # Test at exactly u=0.75 (boundary between regions)
        u_boundary = 0.75
        grdsf, griddata = prolate_spheroidal_function(u_boundary)

        # Should still produce valid output
        self.assertGreater(grdsf, 0.0)
        self.assertGreater(griddata, 0.0)

    def test_kernel_symmetry(self):
        """Test that kernel is symmetric"""
        oversampling = 50
        support = 7
        n_uv = np.array([100, 100])

        kernel, _ = create_prolate_spheroidal_kernel(oversampling, support, n_uv)

        # For a given oversampling level, the kernel should be symmetric
        # in u and v directions (kernel[x,y] should equal kernel[y,x])
        center_u = oversampling // 2
        center_v = oversampling // 2
        kernel_slice = kernel[center_u, center_v, :, :]

        # Check that the kernel slice is symmetric
        self.assertTrue(np.allclose(kernel_slice, kernel_slice.T))


if __name__ == "__main__":
    unittest.main()
