"""The w term of the extended components and the Gaussian-broadened ring.

The analytic responses solve the measurement equation in the frame of the
component to second order in the direction cosines (paraxial expansion
``w (n - 1) ~ -w r^2 / 2``, phase convention ``exp(+2 pi i (u l + v m + w (n - 1)))``
of ``calculate_visibilities``).  They are pinned here to brute-force 2-D
integration of the brightness with the quadratic phase, to each other in
their common limits, and, through the simulator itself, to a dense grid of
point sources (which the kernel treats exactly, including the w term and the
rotation to the source frame).
"""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.restore import elliptical_gaussian_uv_taper
from astroviper.processing_functions.simulation import (
    elliptical_gaussian_uv_response,
    gaussian_ring_image,
    gaussian_ring_uv_response,
    limb_darkened_disk_image,
    limb_darkened_disk_uv_response,
    simulate_processing_set,
)
from astroviper.processing_functions.simulation.calculate_visibilities import (
    calculate_visibilities,
    source_frame_uvw,
)
from astroviper.processing_functions.simulation.calculate_visibilities_cpp import (
    cpp_kernel_available,
)
from astroviper.processing_functions.simulation.gaussian_source import FWHM_TO_SIGMA
from astroviper.utils.coordinate_transforms import inverse_sin_project
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180 * 3600)


def paraxial_oracle(image, extent, n, u, v, w):
    """sum b(l, m) exp(-i pi w (l^2 + m^2)) exp(2 pi i (u l + v m)) dl dm on a fine grid."""
    axis = (np.arange(n) - n // 2) * (2 * extent / n)
    l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
    brightness = image(l_grid, m_grid)
    cell = axis[1] - axis[0]
    out = []
    for uu, vv, ww in zip(u, v, w, strict=True):
        phase = np.exp(
            -1j * np.pi * ww * (l_grid**2 + m_grid**2)
            + 2j * np.pi * (uu * l_grid + vv * m_grid)
        )
        out.append(np.sum(brightness * phase) * cell * cell)
    return np.array(out)


# Baselines up to 100 wavelengths, w up to 3000 wavelengths, with the origin
# and a pure tangent-plane point included; the components are ~1 degree so that
# the Fresnel parameter pi w theta^2 reaches ~1-4 and the w term is a large effect.
RNG = np.random.default_rng(3)
U = RNG.uniform(-100, 100, 10)
V = RNG.uniform(-100, 100, 10)
W = RNG.uniform(-3000, 3000, 10)
U[0] = V[0] = W[0] = 0.0
U[1], V[1], W[1] = 60.0, -20.0, 0.0


class TestAnalyticResponses:
    def test_elliptical_gaussian(self):
        major, minor, pa = 0.024, 0.012, 0.7
        s_major, s_minor = major * FWHM_TO_SIGMA, minor * FWHM_TO_SIGMA

        def image(l_grid, m_grid):
            l_maj = l_grid * np.sin(pa) + m_grid * np.cos(pa)
            l_min = l_grid * np.cos(pa) - m_grid * np.sin(pa)
            return np.exp(-0.5 * (l_maj**2 / s_major**2 + l_min**2 / s_minor**2)) / (
                2 * np.pi * s_major * s_minor
            )

        response = elliptical_gaussian_uv_response(U, V, major, minor, pa, w=W)
        np.testing.assert_allclose(
            response, paraxial_oracle(image, 0.08, 1024, U, V, W), atol=1e-12
        )
        assert response[0] == 1.0
        # the w term is a large effect here, and vanishes exactly at w = 0
        tangent = elliptical_gaussian_uv_taper(U, V, major, minor, pa)
        assert np.abs(response - tangent).max() > 0.05
        np.testing.assert_array_equal(
            elliptical_gaussian_uv_response(U, V, major, minor, pa), tangent
        )
        assert (
            elliptical_gaussian_uv_response(U, V, major, minor, pa).dtype == np.float64
        )

    @pytest.mark.parametrize("inclination_deg", [0.0, 55.0])
    def test_gaussian_ring(self, inclination_deg):
        radius, fwhm, inclination, pa = 0.02, 0.01, np.deg2rad(inclination_deg), 0.9
        response = gaussian_ring_uv_response(U, V, radius, fwhm, inclination, pa, w=W)
        oracle = paraxial_oracle(
            lambda l_grid, m_grid: gaussian_ring_image(l_grid, m_grid, radius, fwhm, inclination, pa),
            0.08, 1024, U, V, W,
        )  # fmt: skip
        np.testing.assert_allclose(response, oracle, atol=1e-12)
        assert response[0] == 1.0
        assert (
            np.abs(
                response
                - gaussian_ring_uv_response(U, V, radius, fwhm, inclination, pa)
            ).max()
            > 0.1
        )
        # the memo's tangent-plane closed form at w = 0
        from scipy.special import j0

        q = np.cos(inclination)
        u_maj = U * np.sin(pa) + V * np.cos(pa)
        u_min = U * np.cos(pa) - V * np.sin(pa)
        rho = np.sqrt(u_maj**2 + (q * u_min) ** 2)
        sigma = fwhm * FWHM_TO_SIGMA
        np.testing.assert_allclose(
            gaussian_ring_uv_response(U, V, radius, fwhm, inclination, pa),
            j0(2 * np.pi * radius * rho) * np.exp(-2 * np.pi**2 * sigma**2 * rho**2),
            atol=1e-14,
        )

    def test_ring_of_zero_radius_is_a_circular_gaussian(self):
        ring = gaussian_ring_uv_response(U, V, 0.0, 0.01, 0.0, 0.3, w=W)
        gaussian = elliptical_gaussian_uv_response(U, V, 0.01, 0.01, 0.3, w=W)
        np.testing.assert_allclose(ring, gaussian, atol=1e-14)

    def test_ring_image_has_unit_flux_and_is_inclined(self):
        n = 1024
        axis = (np.arange(n) - n // 2) * 1.6e-4
        l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
        pa = 0.4
        image = gaussian_ring_image(l_grid, m_grid, 0.02, 0.008, np.deg2rad(60), pa)
        np.testing.assert_allclose(image.sum() * 1.6e-4**2, 1.0, atol=1e-6)
        # brightest along the major axis (sin pa, cos pa) at the ring radius, fainter along the minor axis
        e_major = np.array([np.sin(pa), np.cos(pa)])
        e_minor = np.array([np.cos(pa), -np.sin(pa)])

        def at(vec):
            return image[tuple(np.round(n // 2 + 0.02 * vec / 1.6e-4).astype(int))]

        assert at(e_major) > 10 * at(e_minor)
        with pytest.raises(ValueError):
            gaussian_ring_uv_response(U, V, 0.02, 0.01, np.pi / 2, 0.0)
        with pytest.raises(ValueError):
            gaussian_ring_image(l_grid, m_grid, 0.02, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        ("alpha", "axis_ratio"), [(0.0, 1.0), (1.5, 1.0), (1.5, 0.6), (-0.5, 0.6)]
    )
    def test_limb_darkened_disk(self, alpha, axis_ratio):
        major, minor, pa = 0.04, 0.04 * axis_ratio, 0.5
        response = limb_darkened_disk_uv_response(U, V, major, minor, pa, alpha, w=W)
        oracle = paraxial_oracle(
            lambda l_grid, m_grid: limb_darkened_disk_image(l_grid, m_grid, major, minor, pa, alpha),
            0.08, 2048, U, V, W,
        )  # fmt: skip
        # the oracle's pixelised rim limits the agreement for the sharp-edged profiles
        np.testing.assert_allclose(
            response, oracle, atol=1e-5 if alpha == 1.5 else 1e-3
        )
        assert response[0] == 1.0
        tangent = limb_darkened_disk_uv_response(U, V, major, minor, pa, alpha)
        assert tangent.dtype == np.float64  # the w = 0 path is unchanged
        assert np.abs(response - tangent).max() > 0.05

    def test_paraxial_expansion_error_is_small(self):
        """The neglected terms (w r^4 / 8 phase, 1/n weighting) are ~1e-4 even for a 1.4 deg Gaussian."""
        major = minor = 0.024
        sigma = major * FWHM_TO_SIGMA

        def image(l_grid, m_grid):
            return np.exp(-0.5 * (l_grid**2 + m_grid**2) / sigma**2) / (
                2 * np.pi * sigma**2
            )

        n = 1024
        axis = (np.arange(n) - n // 2) * (0.16 / n)
        l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
        n_cos = np.sqrt(1 - l_grid**2 - m_grid**2)
        cell = axis[1] - axis[0]
        spherical = np.array(
            [
                np.sum(image(l_grid, m_grid) / n_cos * np.exp(2j * np.pi * (uu * l_grid + vv * m_grid + ww * (n_cos - 1))))
                * cell**2
                for uu, vv, ww in zip(U, V, W, strict=True)
            ]
        )  # fmt: skip
        response = elliptical_gaussian_uv_response(U, V, major, minor, 0.0, w=W)
        assert np.abs(response - spherical).max() < 2e-4


class TestSimulatorWiring:
    """Extended components through calculate_visibilities: rotation to the source frame plus the w term."""

    @staticmethod
    def _setup():
        phase_center = np.array([[5.2337, 0.7109]])
        # a Gaussian 34' FWHM offset by ~2.9 deg from the phase centre, and baselines
        # with |w| up to 3000 wavelengths: both the rotation of the uvw (~ w * offset ~ 150
        # wavelengths shift of the effective u) and the Fresnel term (pi w sigma^2 ~ 0.9) are
        # first-order effects here.
        source = phase_center + np.array([[0.05, 0.02]])
        fwhm = 0.01 / FWHM_TO_SIGMA
        uvw_wavelengths = np.array(
            [
                [
                    [30.0, 10.0, 3000.0],
                    [-60.0, 40.0, -2000.0],
                    [80.0, -30.0, 1000.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )
        frequency = np.array([1.0e9])
        uvw = uvw_wavelengths * 299792458.0 / frequency[0]
        kwargs = dict(
            antenna1=np.array([0, 0, 0, 1]),
            antenna2=np.array([1, 2, 3, 2]),
            frequency=frequency,
            polarization_index=np.array([0]),
            phase_center_ra_dec=phase_center,
            pointing_ra_dec=None,
            beam_model_map=np.zeros(4, dtype=int),
            packed_beam_models=[
                {
                    "kind": "analytic",
                    "func": "none",
                    "dish_diameter": 25.0,
                    "blockage_diameter": 0.0,
                    "max_rad_1GHz": 1.0,
                }
            ],  # fmt: skip
            parallactic_angle=np.zeros(1),
            mueller_selection=np.array([0, 5, 10, 15]),
        )
        return uvw, source, fwhm, kwargs

    @pytest.mark.parametrize(
        "implementation", ["numpy"] + (["cpp"] if cpp_kernel_available() else [])
    )
    def test_offset_gaussian_matches_dense_point_source_grid(self, implementation):
        uvw, source, fwhm, kwargs = self._setup()
        sigma = fwhm * FWHM_TO_SIGMA
        flux = np.array([[[[1.0, 0, 0, 1.0]]]])
        gaussian = calculate_visibilities(
            uvw,
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=source[None],
            gaussian_source_flux=flux,
            gaussian_source_ra_dec=source[None],
            gaussian_source_shape=np.array([[fwhm, fwhm, 0.0]]),
            implementation=implementation,
            **kwargs,
        )
        # the same Gaussian as ~10^4 point sources on the tangent plane of its centre
        n = 101
        axis = np.linspace(-5 * sigma, 5 * sigma, n)
        l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
        cell = axis[1] - axis[0]
        weights = (
            np.exp(-0.5 * (l_grid**2 + m_grid**2) / sigma**2)
            / (2 * np.pi * sigma**2)
            * cell**2
        )
        weights /= weights.sum()
        positions = inverse_sin_project(
            source[0], np.stack([l_grid.ravel(), m_grid.ravel()], axis=-1)
        )
        point_flux = np.zeros((n * n, 1, 1, 4))
        point_flux[:, 0, 0, 0] = point_flux[:, 0, 0, 3] = weights.ravel()
        grid = calculate_visibilities(
            uvw,
            point_source_flux=point_flux,
            point_source_ra_dec=positions[None],
            implementation=implementation,
            **kwargs,
        )
        # agreement at the level of the neglected higher-order terms (~1e-4)
        np.testing.assert_allclose(gaussian, grid, atol=3e-4)
        assert np.abs(gaussian[0, :3, 0, 0]).min() < 0.9  # resolved
        # and the tangent-plane treatment (phase-centre uv, no w) would be wrong by a lot
        u = uvw[:, :, 0, None] * kwargs["frequency"] / 299792458.0
        v = uvw[:, :, 1, None] * kwargs["frequency"] / 299792458.0
        point = calculate_visibilities(
            uvw, point_source_flux=flux, point_source_ra_dec=source[None],
            implementation=implementation, **kwargs,
        )  # fmt: skip
        tangent_plane = (
            point * elliptical_gaussian_uv_taper(u, v, fwhm, fwhm, 0.0)[..., None]
        )
        assert np.abs(tangent_plane - grid).max() > 0.05

    def test_source_frame_uvw(self):
        uvw, source, _, kwargs = self._setup()
        u, v, w = source_frame_uvw(
            uvw, kwargs["frequency"], kwargs["phase_center_ra_dec"], source
        )
        assert u.shape == v.shape == w.shape == (1, 4, 1)
        # a rotation: baseline lengths are preserved
        inverse_wavelength = kwargs["frequency"][0] / 299792458.0
        np.testing.assert_allclose(
            np.sqrt(u**2 + v**2 + w**2)[..., 0],
            np.linalg.norm(uvw, axis=-1) * inverse_wavelength,
        )
        # identity for a source at the phase centre
        u0, v0, w0 = source_frame_uvw(
            uvw,
            kwargs["frequency"],
            kwargs["phase_center_ra_dec"],
            kwargs["phase_center_ra_dec"],
        )
        np.testing.assert_allclose(
            np.stack([u0, v0, w0], -1)[..., 0, :], uvw * inverse_wavelength, atol=1e-9
        )


class TestGaussianRingSources:
    """Rings through the processing function: the response is applied per ring in its frame."""

    def _pf_kwargs(self):
        antenna_position = read_telescope_layout("vla.d").ANTENNA_POSITION.values[:8]
        return dict(
            time=np.array(["2019-10-03T19:00:00.000"]),
            frequency=np.array([1.5e9, 3.0e9]),
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
            ],  # fmt: skip
            beam_model_map=np.zeros(len(antenna_position), dtype=int),
        )

    def test_rings_apply_their_response_and_zero_ring_is_a_gaussian(self):
        kwargs = self._pf_kwargs()
        pc = kwargs["phase_center_ra_dec"]
        sources = np.stack([pc[0] + 3e-3, pc[0] - 2e-3])[None]
        fluxes = np.array([[[[2.0, 0, 0, 2.0]]], [[[0.7, 0, 0, 0.7]]]])
        shapes = np.array(
            [[600 * ARCSEC, 200 * ARCSEC, 0.8, 0.4], [0.0, 400 * ARCSEC, 0.0, 0.0]]
        )
        rings, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=sources[:, :1],
            gaussian_ring_source_flux=fluxes,
            gaussian_ring_source_ra_dec=sources,
            gaussian_ring_source_shape=shapes,
            **kwargs,
        )
        expected = np.zeros_like(rings.VISIBILITY.values)
        for i in range(2):
            point, _ = simulate_processing_set(
                point_source_flux=fluxes[i : i + 1],
                point_source_ra_dec=sources[:, i : i + 1],
                **kwargs,
            )
            u, v, w = source_frame_uvw(
                rings.UVW.values, kwargs["frequency"], pc, sources[:, i]
            )
            response = gaussian_ring_uv_response(u, v, *shapes[i], w=w)
            assert np.abs(response).min() < 0.9  # resolved
            expected += point.VISIBILITY.values * response[..., None]
        np.testing.assert_allclose(
            rings.VISIBILITY.values, expected, rtol=1e-12, atol=1e-14
        )

        # a zero-radius ring is the circular Gaussian component
        gaussian, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=sources[:, 1:],
            gaussian_source_flux=fluxes[1:],
            gaussian_source_ra_dec=sources[:, 1:],
            gaussian_source_shape=np.array([[400 * ARCSEC, 400 * ARCSEC, 0.0]]),
            **kwargs,
        )
        ring_only, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=sources[:, 1:],
            gaussian_ring_source_flux=fluxes[1:],
            gaussian_ring_source_ra_dec=sources[:, 1:],
            gaussian_ring_source_shape=shapes[1:],
            **kwargs,
        )
        np.testing.assert_allclose(
            ring_only.VISIBILITY.values, gaussian.VISIBILITY.values, rtol=1e-12
        )
