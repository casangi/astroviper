# run using eg
# python -m pytest -s ../astroviper/tests/image/model/test_disk.py

from astroviper.image.model import generate_disk

import numpy as np
from numpy.testing import assert_allclose
import unittest

def first_moments(img: np.ndarray, X: np.ndarray, Y: np.ndarray, dx: float, dy: float):
    """Compute centroid (x̄,ȳ) using first moments."""
    A = img.sum() * dx * dy
    if A == 0:
        return np.nan, np.nan
    xbar = (img * X).sum() * dx * dy / A
    ybar = (img * Y).sum() * dx * dy / A
    return xbar, ybar

class DiskTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a moderately fine grid with square pixels
        nx, ny = 401, 401
        x = np.linspace(-20.0, 20.0, nx)
        y = np.linspace(-20.0, 20.0, ny)
        cls.X, cls.Y = np.meshgrid(x, y)
        cls.dx = float(np.mean(np.diff(x)))
        cls.dy = float(np.mean(np.diff(y)))

    def test_basic_shape_dtype(self):
        img = generate_disk(self.X, self.Y, 0.0, 0.0, 5.0, 3.0, 0.0, 2.5)
        self.assertEqual(img.shape, self.X.shape)
        self.assertEqual(img.dtype, float)

    def test_amplitude_inside_outside(self):
        A_d = 3.7
        img = generate_disk(self.X, self.Y, 0.0, 0.0, 5.0, 3.0, 0.0, A_d)
        cy, cx = np.array(self.X.shape) // 2
        self.assertAlmostEqual(img[cy, cx], A_d)
        self.assertEqual(img[0, 0], 0.0)

    def test_flux_matches_area(self):
        x0, y0, a, b, A = 1.0, -2.0, 6.0, 4.0, 5.0
        theta = 30.0
        img = generate_disk(self.X, self.Y, x0, y0, a, b, theta, A)
        flux_discrete = img.sum() * self.dx * self.dy
        flux_continuous = A * np.pi * a * b
        assert_allclose(flux_discrete, flux_continuous, rtol=5e-3, atol=5e-2)

    def test_centroid_matches_parameters(self):
        x0, y0, a, b, A = 3.2, -4.8, 7.5, 2.5, 2.0
        theta = 45.0
        img = generate_disk(self.X, self.Y, x0, y0, a, b, theta, A)
        xbar, ybar = first_moments(img, self.X, self.Y, self.dx, self.dy)
        assert_allclose([xbar, ybar], [x0, y0], atol=0.05)

    def test_axis_aligned_symmetry(self):
        img0 = generate_disk(self.X, self.Y, 0.0, 0.0, 8.0, 3.0, 0.0, 1.0)
        img180 = generate_disk(self.X, self.Y, 0.0, 0.0, 8.0, 3.0, 180.0, 1.0)
        assert_allclose(img0, img180)

    def test_rotation_equivalence(self):
        x0, y0, a, b, A = 1.0, 2.0, 5.0, 2.0, 3.0
        theta = 25.0
        img_rot = generate_disk(self.X, self.Y, x0, y0, a, b, theta, A)

        # Rotate grid about (x0, y0) by -theta
        t = np.deg2rad(-theta)
        Xc = self.X - x0
        Yc = self.Y - y0
        Xr = Xc * np.cos(t) + Yc * np.sin(t) + x0
        Yr = -Xc * np.sin(t) + Yc * np.cos(t) + y0
        img_unrot = generate_disk(Xr, Yr, x0, y0, a, b, 0.0, A)

        assert_allclose(img_rot, img_unrot, atol=1e-7)

    def test_degenerate_axes_edge_cases(self):
        x0, y0, A = 0.0, 0.0, 1.0
        img_thin_a = generate_disk(self.X, self.Y, x0, y0, 1e-6, 3.0, 0.0, A)
        img_thin_b = generate_disk(self.X, self.Y, x0, y0, 3.0, 1e-6, 0.0, A)
        area_pix_a = img_thin_a.sum() * self.dx * self.dy
        area_pix_b = img_thin_b.sum() * self.dx * self.dy
        self.assertLess(area_pix_a, 1e-3)
        self.assertLess(area_pix_b, 1e-3)


if __name__ == "__main__":
    unittest.main()

