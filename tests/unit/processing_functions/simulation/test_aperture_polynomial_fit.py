"""Zernike aperture coefficients fitted to holography samples."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.simulation.antenna_beams import SPEED_OF_LIGHT
from astroviper.processing_functions.simulation.aperture_polynomial_fit import (
    _basis_transform,
    fit_aperture_polynomial_coefficients,
)
from astroviper.processing_functions.simulation.zernike_polynomials import (
    zernike_basis,
)

DISH_DIAMETER = 12.0
FREQUENCY = np.array([100.0e9])
POLARIZATION = ["XX", "XY", "YX", "YY"]
CELL = 0.25
EVEN_ORDERS = np.concatenate(
    [np.arange(n * (n + 1) // 2, (n + 1) * (n + 2) // 2) for n in range(0, 11, 2)]
)


def _true_coefficients():
    """Tapered illumination in the parallel hands, weak cross hands. The
    parallel hands have even radial orders only, so that their beams are
    symmetric about the centre and the main-lobe fit does not move them."""
    rng = np.random.default_rng(7)
    zpc = 0.01 * (rng.normal(size=(4, 66)) + 1j * rng.normal(size=(4, 66)))
    odd = np.setdiff1d(np.arange(66), EVEN_ORDERS)
    zpc[np.ix_([0, 3], odd)] = 0.0
    zpc[[0, 3], 0] += 1.0
    zpc[[0, 3], 4] -= 0.3
    return zpc


def _make_map(zpc, offset, gain, frequency=FREQUENCY):
    """Raster of +-6 lambda/D in steps of lambda/(4 D) around a pointing offset."""
    lambda_over_d = SPEED_OF_LIGHT / frequency.mean() / DISH_DIAMETER
    steps = np.arange(-24, 25) * lambda_over_d / 4
    l_grid, m_grid = np.meshgrid(steps, steps, indexing="ij")
    lm = np.stack([l_grid.ravel(), m_grid.ravel()], axis=1)
    radius = DISH_DIAMETER / 2
    axis = (
        np.arange(-(int(np.ceil(radius / CELL)) + 1), int(np.ceil(radius / CELL)) + 2)
        * CELL
    )
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    basis = zernike_basis(x_grid / radius, y_grid / radius) * (
        np.hypot(x_grid, y_grid) <= radius
    )
    vis = np.empty((lm.shape[0], frequency.size, len(POLARIZATION)), dtype=complex)
    for i_chan, freq in enumerate(frequency):
        design = _basis_transform(
            basis, axis, lm - np.asarray(offset), SPEED_OF_LIGHT / freq
        )
        vis[:, i_chan] = gain * (design @ zpc.T) * CELL**2
    return xr.Dataset(
        {
            "DIRECTIONAL_COSINES": (("time", "lm"), lm),
            "VIS": (("time", "frequency", "polarization"), vis),
            "WEIGHT": (("time", "frequency", "polarization"), np.ones(vis.shape)),
        },
        coords={"frequency": frequency, "polarization": POLARIZATION},
    )


def _relative(zpc):
    """Coefficients relative to the piston of the parallel-hand mean."""
    return zpc / zpc[..., [0, 3], 0].mean(axis=-1)[..., None, None]


class TestFitAperturePolynomialCoefficients:
    def test_recovers_coefficients(self):
        """Test that the coefficients of a single centred map are recovered up to one complex scale"""
        zpc = _true_coefficients()
        fit = fit_aperture_polynomial_coefficients(
            [_make_map(zpc, (0.0, 0.0), 1.0)], DISH_DIAMETER, aperture_cell=CELL
        )
        assert fit.ZPC.shape == (1, 4, 66)
        assert np.allclose(_relative(fit.ZPC.values[0]), _relative(zpc), atol=1e-6)

    def test_maps_with_offsets_and_gains(self):
        """Test that maps with different pointing offsets and complex gains are aligned before the fit, to within the 1e-3 lambda/D of the main-lobe fit"""
        zpc = _true_coefficients()
        lambda_over_d = SPEED_OF_LIGHT / FREQUENCY.mean() / DISH_DIAMETER
        maps = [
            _make_map(
                zpc, (0.1 * lambda_over_d, -0.05 * lambda_over_d), 0.8 * np.exp(0.5j)
            ),
            _make_map(
                zpc, (-0.05 * lambda_over_d, 0.08 * lambda_over_d), 1.2 * np.exp(-1.0j)
            ),
        ]
        fit = fit_aperture_polynomial_coefficients(
            maps, DISH_DIAMETER, aperture_cell=CELL
        )
        assert np.allclose(_relative(fit.ZPC.values[0]), _relative(zpc), atol=5e-3)

    def test_frequency_groups_and_attributes(self):
        """Test the channel groups, the legacy dimension names and the attributes"""
        holog_map = _make_map(
            _true_coefficients(),
            (0.0, 0.0),
            1.0,
            frequency=np.array([99.0e9, 100.0e9, 101.0e9, 102.0e9]),
        ).rename({"frequency": "chan", "polarization": "pol"})
        fit = fit_aperture_polynomial_coefficients(
            [holog_map], DISH_DIAMETER, eta=1.0, n_frequency=2, aperture_cell=CELL
        )
        assert np.allclose(fit.frequency.values, [99.5e9, 101.5e9])
        assert fit.frequency.attrs["units"] == "Hz"
        assert list(fit.polarization.values) == POLARIZATION
        assert np.all(fit.ETA.values == 1.0)
        low, high = np.percentile(holog_map.DIRECTIONAL_COSINES.values, [1, 99], axis=0)
        expected_radius = min(-low[0], high[0], -low[1], high[1])
        assert fit.attrs["max_rad_1GHz"] == pytest.approx(expected_radius * 99.5)
        assert fit.attrs["dish_diameter"] == DISH_DIAMETER

    def test_missing_variable(self):
        """Test that a map without visibilities is refused"""
        holog_map = _make_map(_true_coefficients(), (0.0, 0.0), 1.0).drop_vars("VIS")
        with pytest.raises(ValueError, match="VIS"):
            fit_aperture_polynomial_coefficients([holog_map], DISH_DIAMETER)
