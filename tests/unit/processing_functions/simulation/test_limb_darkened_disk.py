"""Limb-darkened disk component: analytic visibility, image-plane profile and simulator wiring."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, j1

from astroviper.processing_functions.simulation import (
    limb_darkened_disk_image,
    limb_darkened_disk_uv_response,
    simulate_processing_set,
)
from astroviper.processing_functions.simulation.calculate_visibilities_cpp import (
    cpp_kernel_available,
)
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180 * 3600)
DIAMETER = 1e-3  # radians


def hankel_oracle(x, alpha):
    """V(x) = (alpha + 2) int_0^1 (1 - r^2)^(alpha/2) J0(x r) r dr (numerical)."""
    value, _ = quad(
        lambda r: (1 - r * r) ** (alpha / 2) * j0(x * r) * r, 0, 1, limit=200
    )
    return (alpha + 2) * value


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.5, -0.5, -1.0, -1.5])
def test_uv_response_matches_numerical_hankel_transform(alpha):
    x = np.array([0.0, 1e-6, 0.3, 1.0, 3.8, 7.0, 12.0, 25.0])
    q = x / (np.pi * DIAMETER)  # circular disk: x = pi D q
    response = limb_darkened_disk_uv_response(q, 0.0, DIAMETER, DIAMETER, 0.0, alpha)
    expected = np.array([hankel_oracle(xi, alpha) for xi in x])
    np.testing.assert_allclose(response, expected, atol=1e-9, rtol=0)


def test_uv_response_special_cases():
    x = np.linspace(0.01, 40, 800)
    q = x / (np.pi * DIAMETER)
    args = (q, 0.0, DIAMETER, DIAMETER, 0.0)
    # uniform disk, optically thin shell, infinitely thin ring
    np.testing.assert_allclose(
        limb_darkened_disk_uv_response(*args, 0.0), 2 * j1(x) / x, atol=1e-14
    )
    np.testing.assert_allclose(
        limb_darkened_disk_uv_response(*args, -1.0), np.sin(x) / x, atol=1e-14
    )
    np.testing.assert_allclose(
        limb_darkened_disk_uv_response(*args, -2.0), j0(x), atol=1e-14
    )
    # unit response at the origin, for a zero-diameter disk, and the small-x series joins smoothly
    assert limb_darkened_disk_uv_response(0.0, 0.0, DIAMETER, DIAMETER, 0.3, 1.0) == 1.0
    assert limb_darkened_disk_uv_response(1e5, 3e4, 0.0, 0.0, 0.3, 1.0) == 1.0
    x_join = np.array([0.5e-4, 0.99e-4, 1.01e-4, 2e-4])
    r = limb_darkened_disk_uv_response(x_join / (np.pi * DIAMETER), 0.0, *args[2:], 2.0)
    np.testing.assert_allclose(r, 1 - x_join**2 / 12, rtol=0, atol=1e-14)
    # the response is symmetric (real, even) and bounded by 1
    u = np.linspace(-3e5, 3e5, 51)[:, None]
    v = np.linspace(-2e5, 2e5, 41)[None, :]
    r = limb_darkened_disk_uv_response(u, v, 2 * DIAMETER, DIAMETER, 0.4, 0.7)
    assert r.shape == (51, 41)
    np.testing.assert_allclose(r, r[::-1, ::-1], atol=1e-15)
    assert np.abs(r).max() <= 1.0
    with pytest.raises(ValueError):
        limb_darkened_disk_uv_response(u, v, DIAMETER, DIAMETER, 0.0, -2.5)
    with pytest.raises(ValueError):
        limb_darkened_disk_uv_response(u, v, -DIAMETER, DIAMETER, 0.0, 0.0)


@pytest.mark.parametrize("alpha", [0.0, 1.5, -0.5])
def test_image_is_unit_flux_and_its_fft_is_the_uv_response(alpha):
    """The image-plane and uv forms are Fourier pairs with the same [major, minor, pa] convention."""
    n = 1024
    major, minor, pa = 3 * DIAMETER, 1.6 * DIAMETER, 0.9
    cell = major / 200
    axis = (np.arange(n) - n // 2) * cell
    l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
    image = limb_darkened_disk_image(l_grid, m_grid, major, minor, pa, alpha)
    assert image.shape == (n, n)
    np.testing.assert_allclose(image.sum() * cell * cell, 1.0, atol=3e-3)
    assert image[n // 2, n // 2] > 0
    # orientation: the disk extends further along the major axis (sin pa, cos pa)
    e = np.array([np.sin(pa), np.cos(pa)])
    p = np.array([np.cos(pa), -np.sin(pa)])
    assert image[tuple(np.round(n // 2 + 0.45 * major * e / cell).astype(int))] > 0
    assert image[tuple(np.round(n // 2 + 0.45 * major * p / cell).astype(int))] == 0

    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))) * cell * cell
    frequency = np.fft.fftshift(np.fft.fftfreq(n, cell))
    u_grid, v_grid = np.meshgrid(frequency, frequency, indexing="ij")
    response = limb_darkened_disk_uv_response(u_grid, v_grid, major, minor, pa, alpha)
    # compare where the pixelised rim does not dominate (low spatial frequencies)
    q = np.sqrt(u_grid**2 + v_grid**2)
    select = q < 2.5 / major
    np.testing.assert_allclose(spectrum.real[select], response[select], atol=6e-3)
    np.testing.assert_allclose(spectrum.imag[select], 0.0, atol=6e-3)
    with pytest.raises(ValueError):
        limb_darkened_disk_image(l_grid, m_grid, major, minor, pa, -2.0)
    with pytest.raises(ValueError):
        limb_darkened_disk_image(l_grid, m_grid, 0.0, minor, pa, 0.0)


class TestDiskSources:
    """Disk sources in the simulator: point-source kernel times the analytic response."""

    def _pf_kwargs(self):
        antenna_position = read_telescope_layout("vla.d").ANTENNA_POSITION.values[:8]
        return dict(
            time=np.array(["2019-10-03T19:00:00.000"]),
            frequency=np.array([3.0e9, 3.2e9]),
            polarization=["RR", "LL"],
            antenna_position=antenna_position,
            site_position=antenna_position.mean(axis=0),
            phase_center_ra_dec=np.array([[5.2337, 0.7109]]),
            beam_models=[
                {
                    "func": "none",
                    "dish_diameter": 25.0,
                    "blockage_diameter": 0.0,
                    "max_rad_1GHz": 0.014946999714,
                }
            ],
            beam_model_map=np.zeros(len(antenna_position), dtype=int),
        )

    def test_zero_size_disk_equals_point_source(self):
        kwargs = self._pf_kwargs()
        source = kwargs["phase_center_ra_dec"][:, None, :] + 1e-4
        flux = np.array([[[[2.0, 0.0, 0.0, 2.0]]]])
        point_xds, _ = simulate_processing_set(
            point_source_flux=flux, point_source_ra_dec=source, **kwargs
        )
        disk_xds, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=source,
            disk_source_flux=flux,
            disk_source_ra_dec=source,
            disk_source_shape=np.zeros((1, 3)),
            **kwargs,
        )
        np.testing.assert_allclose(
            disk_xds.VISIBILITY.values, point_xds.VISIBILITY.values, rtol=1e-13
        )

    @pytest.mark.parametrize(
        "implementation",
        ["numpy"] + (["cpp"] if cpp_kernel_available() else []),
    )
    def test_response_is_applied_per_disk_and_defaults_to_uniform(self, implementation):
        kwargs = self._pf_kwargs()
        pc = kwargs["phase_center_ra_dec"]
        sources = np.stack([pc[0] + 2e-4, pc[0] - 3e-4])[None]  # [1, 2, 2]
        fluxes = np.array([[[[3.0, 0, 0, 3.0]]], [[[1.0, 0, 0, 1.0]]]])
        shapes = np.array(
            [[400 * ARCSEC, 250 * ARCSEC, 0.7], [150 * ARCSEC, 150 * ARCSEC, 0.0]]
        )
        alphas = np.array([1.5, -1.0])
        disks, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=sources[:, :1],
            disk_source_flux=fluxes,
            disk_source_ra_dec=sources,
            disk_source_shape=shapes,
            disk_source_limb_darkening=alphas,
            implementation=implementation,
            **kwargs,
        )
        from astroviper.processing_functions.simulation.calculate_visibilities import (
            source_frame_uvw,
        )

        expected = np.zeros_like(disks.VISIBILITY.values)
        for i in range(2):
            point, _ = simulate_processing_set(
                point_source_flux=fluxes[i : i + 1],
                point_source_ra_dec=sources[:, i : i + 1],
                implementation=implementation,
                **kwargs,
            )
            # the response is evaluated in the frame of each disk, with the w term
            u, v, w = source_frame_uvw(
                disks.UVW.values, kwargs["frequency"], kwargs["phase_center_ra_dec"],
                sources[:, i],
            )  # fmt: skip
            response = limb_darkened_disk_uv_response(u, v, *shapes[i], alphas[i], w=w)
            assert response.real.min() < 0.9  # the disks are resolved
            expected += point.VISIBILITY.values * response[..., None]
        np.testing.assert_allclose(
            disks.VISIBILITY.values, expected, rtol=1e-12, atol=1e-14
        )

        # omitting the exponents simulates uniform disks
        uniform, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=sources[:, :1],
            disk_source_flux=fluxes,
            disk_source_ra_dec=sources,
            disk_source_shape=shapes,
            implementation=implementation,
            **kwargs,
        )
        explicit, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=sources[:, :1],
            disk_source_flux=fluxes,
            disk_source_ra_dec=sources,
            disk_source_shape=shapes,
            disk_source_limb_darkening=np.zeros(2),
            implementation=implementation,
            **kwargs,
        )
        np.testing.assert_array_equal(
            uniform.VISIBILITY.values, explicit.VISIBILITY.values
        )
        assert not np.allclose(uniform.VISIBILITY.values, disks.VISIBILITY.values)

    @pytest.mark.skipif(not cpp_kernel_available(), reason="C++ kernel not built")
    def test_cpp_matches_numpy(self):
        kwargs = self._pf_kwargs()
        source = kwargs["phase_center_ra_dec"][:, None, :] + 2e-4
        disk = dict(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=source,
            disk_source_flux=np.array([[[[3.0, 0.5, 0.5, 3.0]]]]),
            disk_source_ra_dec=source,
            disk_source_shape=np.array([[30 * ARCSEC, 20 * ARCSEC, 1.1]]),
            disk_source_limb_darkening=np.array([0.8]),
        )
        cpp_xds, _ = simulate_processing_set(implementation="cpp", **disk, **kwargs)
        numpy_xds, _ = simulate_processing_set(implementation="numpy", **disk, **kwargs)
        np.testing.assert_allclose(
            cpp_xds.VISIBILITY.values, numpy_xds.VISIBILITY.values, rtol=1e-12
        )
