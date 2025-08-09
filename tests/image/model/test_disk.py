# run using eg
# python -m pytest -s ../astroviper/tests/image/model/test_disk.py

from astroviper.image.model import generate_disk

import numpy as np
from numpy.testing import assert_allclose
from scipy.ndimage import binary_dilation
import unittest

def first_moments(
    img: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float
) -> tuple[float, float]:
    """
    Compute the first spatial moments (centroid coordinates) of a 2D image.

    Parameters
    ----------
    img : np.ndarray
        2D array of image values (e.g., intensity or mask).
    X : np.ndarray
        2D array of the same shape as `img` giving the x-coordinate of each pixel center.
    Y : np.ndarray
        2D array of the same shape as `img` giving the y-coordinate of each pixel center.
    dx : float
        Pixel size along the x-axis in the same units as `X`.
    dy : float
        Pixel size along the y-axis in the same units as `Y`.

    Returns
    -------
    xbar : float
        Flux-weighted centroid x-coordinate.
    ybar : float
        Flux-weighted centroid y-coordinate.

    Notes
    -----
    - If the total flux is zero, both `xbar` and `ybar` will be returned as `np.nan`.
    - The integration assumes a uniform rectangular pixel grid with spacings `dx` and `dy`.
    """
    A = img.sum() * dx * dy
    if A == 0.0:
        return np.nan, np.nan
    xbar = (img * X).sum() * dx * dy / A
    ybar = (img * Y).sum() * dx * dy / A
    return xbar, ybar

def second_moments(
    img: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float
) -> np.ndarray:
    """
    Compute the 2×2 covariance (second central moments) of a 2D image.

    Parameters
    ----------
    img : np.ndarray
        Image values.
    X, Y : np.ndarray
        Coordinate grids matching `img`.
    dx, dy : float
        Pixel sizes along x and y.

    Returns
    -------
    cov : np.ndarray
        2×2 covariance matrix:
        [[mu_xx, mu_xy],
         [mu_xy, mu_yy]]
    """
    A = img.sum() * dx * dy
    if A == 0.0:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    xbar = (img * X).sum() * dx * dy / A
    ybar = (img * Y).sum() * dx * dy / A
    Xc = X - xbar
    Yc = Y - ybar
    mu_xx = (img * (Xc * Xc)).sum() * dx * dy / A
    mu_yy = (img * (Yc * Yc)).sum() * dx * dy / A
    mu_xy = (img * (Xc * Yc)).sum() * dx * dy / A
    return np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]])

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

        # Allow a couple of boundary pixels to flip; require identical flux and IoU ~ 1
        assert_allclose(
            img0.sum() * self.dx * self.dy,
            img180.sum() * self.dx * self.dy,
            rtol=0, atol=1.25 * self.dx * self.dy  # allow a hair over 1 pixel area
        )
        m0 = img0 > 0
        m1 = img180 > 0
        diff_px = np.count_nonzero(m0 ^ m1)  # symmetric difference
        self.assertLessEqual(diff_px, 1)     # at most one boundary pixel differs

    def test_rotation_equivalence(self):
        """
        Comparing masks after rotation should use flux/centroid and overlap, not exact equality.
        """
        x0, y0, a, b, A = 1.0, 2.0, 5.0, 2.0, 3.0
        theta = 25.0
        img_rot = generate_disk(self.X, self.Y, x0, y0, a, b, theta, A)

        # Counter-rotate grid by -theta and draw unrotated ellipse
        t = np.deg2rad(-theta)
        Xc = self.X - x0
        Yc = self.Y - y0
        Xr = Xc * np.cos(t) + Yc * np.sin(t) + x0
        Yr = -Xc * np.sin(t) + Yc * np.cos(t) + y0
        img_unrot = generate_disk(Xr, Yr, x0, y0, a, b, 0.0, A)

        # Flux equal within discretization
        assert_allclose(img_rot.sum(), img_unrot.sum(), rtol=2e-3, atol=1e-2)

        # Centroids match
        xbar_r, ybar_r = first_moments(img_rot, self.X, self.Y, self.dx, self.dy)
        xbar_u, ybar_u = first_moments(img_unrot, self.X, self.Y, self.dx, self.dy)
        assert_allclose([xbar_r, ybar_r], [xbar_u, ybar_u], atol=0.05)

        # Overlap via second moments (rotation-invariant check)
        cov_r = second_moments(img_rot, self.X, self.Y, self.dx, self.dy)
        cov_u = second_moments(img_unrot, self.X, self.Y, self.dx, self.dy)

        # 1) Eigenvalues match up to pixelization
        evals_r = np.sort(np.linalg.eigvalsh(cov_r))
        evals_u = np.sort(np.linalg.eigvalsh(cov_u))
        assert_allclose(evals_r, evals_u, atol=0.05, rtol=0)

        # 2) |mu_xy| should match even if the sign flips
        assert_allclose(abs(cov_r[0, 1]), abs(cov_u[0, 1]), atol=0.05, rtol=0)

    def test_degenerate_axes_edge_cases(self):
        """When an axis ~ 0, the rasterized area is ~ one-pixel wide strip."""
        x0, y0, A = 0.0, 0.0, 1.0

        # a -> 0: expect area ~ A * (2*b) * dx
        b = 3.0
        img_thin_a = generate_disk(self.X, self.Y, x0, y0, 1e-9, b, 0.0, A)
        area_pix_a = img_thin_a.sum() * self.dx * self.dy
        area_expected_a = A * (2.0 * b) * self.dx
        assert_allclose(area_pix_a, area_expected_a, rtol=0.1, atol=1e-3)

        # b -> 0: expect area ~ A * (2*a) * dy
        a = 3.0
        img_thin_b = generate_disk(self.X, self.Y, x0, y0, a, 1e-9, 0.0, A)
        area_pix_b = img_thin_b.sum() * self.dx * self.dy
        area_expected_b = A * (2.0 * a) * self.dy
        assert_allclose(area_pix_b, area_expected_b, rtol=0.1, atol=1e-3)

if __name__ == "__main__":
    unittest.main()

