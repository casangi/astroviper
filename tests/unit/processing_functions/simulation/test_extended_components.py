"""The six analytic components added on top of Gaussian / disk / Gaussian ring, and the blurred disk.

Every response ``T(u, v, w)`` is pinned to brute-force integration of its
image-plane twin with the paraxial quadratic phase (a 2-D grid oracle, or a
polar quadrature for the smooth circular profiles), to its tangent-plane
closed form at ``w = 0``, and to the neighbouring components in their common
limits.  A dense point-source grid pushed through the simulator checks the
wiring (rotation into the source frame plus the w term) for an m-ring.
"""

import numpy as np
import pytest
from scipy.special import hyp1f1, j0, j1, jv

from astroviper.processing_functions.simulation import (
    annulus_image,
    annulus_uv_response,
    calculate_visibilities,
    crescent_image,
    crescent_uv_response,
    elliptical_gaussian_uv_response,
    exponential_disk_image,
    exponential_disk_uv_response,
    gaussian_ring_uv_response,
    limb_darkened_disk_image,
    limb_darkened_disk_uv_response,
    m_ring_image,
    m_ring_uv_response,
    shapelet_image,
    shapelet_uv_response,
    tapered_power_law_image,
    tapered_power_law_uv_response,
)
from astroviper.processing_functions.simulation.calculate_visibilities_cpp import (
    cpp_kernel_available,
)
from astroviper.processing_functions.simulation.gaussian_source import FWHM_TO_SIGMA
from astroviper.utils.coordinate_transforms import inverse_sin_project

# Baselines up to 100 wavelengths, w up to 3000 wavelengths; the components are
# ~1 degree so that the Fresnel parameter pi w theta^2 is of order unity.
RNG = np.random.default_rng(7)
U = RNG.uniform(-100, 100, 8)
V = RNG.uniform(-100, 100, 8)
W = RNG.uniform(-3000, 3000, 8)
U[0] = V[0] = W[0] = 0.0
U[1], V[1], W[1] = 60.0, -20.0, 0.0


def grid_oracle(image, extent, n, u, v, w):
    """sum b(l, m) exp(-i pi w (l^2 + m^2)) exp(2 pi i (u l + v m)) dl dm on a fine grid."""
    axis = (np.arange(n) - n // 2) * (2 * extent / n)
    l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
    brightness = image(l_grid, m_grid)
    cell = axis[1] - axis[0]
    return np.array(
        [
            np.sum(brightness * np.exp(-1j * np.pi * ww * (l_grid**2 + m_grid**2) + 2j * np.pi * (uu * l_grid + vv * m_grid))) * cell**2
            for uu, vv, ww in zip(u, v, w, strict=True)
        ]
    )  # fmt: skip


def polar_oracle(radial, r_max, k1, k2, w1, w2, n_r=4000, n_phi=512, power=2.0):
    """The same integral for a circular profile in its face-on frame, with the anisotropic phase."""
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_r)
    t_max = r_max ** (1.0 / power)
    t = t_max * (t_nodes + 1) / 2
    r = t**power
    weight_r = radial(r) * r * power * t ** (power - 1) * t_weights * t_max / 2
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    x = r[:, None] * np.cos(phi)[None, :]
    y = r[:, None] * np.sin(phi)[None, :]
    weight = weight_r[:, None] * (2 * np.pi / n_phi)
    return np.array(
        [
            np.sum(weight * np.exp(-1j * np.pi * (b1 * x**2 + b2 * y**2) + 2j * np.pi * (a1 * x + a2 * y)))
            for a1, a2, b1, b2 in zip(k1, k2, w1, w2, strict=True)
        ]
    )  # fmt: skip


def deproject(u, v, pa, q):
    return u * np.sin(pa) + v * np.cos(pa), q * (u * np.cos(pa) - v * np.sin(pa))


def image_flux(image, extent=0.08, n=1024):
    axis = (np.arange(n) - n // 2) * (2 * extent / n)
    l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
    cell = axis[1] - axis[0]
    return image(l_grid, m_grid).sum() * cell**2


class TestMRing:
    def test_thick_stretched_m_ring_matches_oracle_and_limits(self):
        radius, fwhm, inclination, pa = 0.02, 0.008, np.deg2rad(50), 0.7
        beta = [0.3 * np.exp(1j * 2.0), 0.1 * np.exp(-0.5j)]
        response = m_ring_uv_response(U, V, radius, beta, fwhm, inclination, pa, w=W)
        oracle = grid_oracle(
            lambda l_grid, m_grid: m_ring_image(l_grid, m_grid, radius, beta, fwhm, inclination, pa),
            0.08, 1024, U, V, W,
        )  # fmt: skip
        np.testing.assert_allclose(response, oracle, atol=1e-11)
        assert response[0] == 1.0
        assert (
            np.abs(
                response - m_ring_uv_response(U, V, radius, beta, fwhm, inclination, pa)
            ).max()
            > 0.05
        )
        # no modes: the Gaussian ring
        np.testing.assert_allclose(
            m_ring_uv_response(U, V, radius, [], fwhm, inclination, pa, w=W),
            gaussian_ring_uv_response(U, V, radius, fwhm, inclination, pa, w=W),
            atol=1e-13,
        )
        # face on, tangent plane: the classic sum beta_m i^m J_m(2 pi R rho) e^{i m phi}
        thin = m_ring_uv_response(U, V, radius, beta)
        rho = np.hypot(U, V)
        phi = np.arctan2(U, V)  # from +v (north) towards +u (east)
        expected = j0(2 * np.pi * radius * rho).astype(complex)
        for order, coefficient in enumerate(beta, start=1):
            term = (
                coefficient
                * 1j**order
                * jv(order, 2 * np.pi * radius * rho)
                * np.exp(1j * order * phi)
            )
            expected += term + (-1) ** order * np.conj(
                term
            )  # beta_{-m} = conj(beta_m), J_{-m} = (-1)^m J_m
        np.testing.assert_allclose(thin, expected, atol=1e-14)
        assert np.abs(thin).max() > 1.0 - 1e-12 and np.abs(thin[2:]).min() < 0.9

    def test_m_ring_image_flux_and_asymmetry(self):
        radius, fwhm = 0.02, 0.006
        beta = [0.4]  # bright side at phi = 0, i.e. towards +m
        image = lambda l_grid, m_grid: m_ring_image(l_grid, m_grid, radius, beta, fwhm)  # noqa: E731
        np.testing.assert_allclose(image_flux(image), 1.0, atol=1e-6)
        north = m_ring_image(0.0, radius, radius, beta, fwhm)
        south = m_ring_image(0.0, -radius, radius, beta, fwhm)
        # the thin ring has I(0) / I(pi) = 1.8 / 0.2; the blur mixes in ive(1, z) / ive(0, z)
        from scipy.special import ive

        z = radius**2 / (fwhm * FWHM_TO_SIGMA) ** 2
        ratio = (ive(0, z) + 0.8 * ive(1, z)) / (ive(0, z) - 0.8 * ive(1, z))
        np.testing.assert_allclose(north / south, ratio, rtol=1e-10)
        assert 8 < ratio < 9
        # 2 Re(beta e^{i phi}) = 2 |beta| cos(phi + arg beta): a phase of +90 degrees puts the
        # bright side at position angle -90 degrees (west), the eht-imaging convention
        west = m_ring_image(-radius, 0.0, radius, [0.4j], fwhm)
        east = m_ring_image(radius, 0.0, radius, [0.4j], fwhm)
        np.testing.assert_allclose(west / east, ratio, rtol=1e-10)
        with pytest.raises(ValueError):
            m_ring_uv_response(U, V, radius, beta, fwhm, np.pi / 2, 0.0)
        with pytest.raises(ValueError):
            m_ring_image(0.0, 0.0, radius, beta, 0.0)


class TestCrescent:
    def test_blurred_crescent_matches_oracle(self):
        radius, inner, offset, pa, floor, fwhm = 0.02, 0.012, 0.005, 1.1, 0.1, 0.005
        response = crescent_uv_response(
            U, V, radius, inner, offset, pa, floor, fwhm, w=W
        )
        oracle = grid_oracle(
            lambda l_grid, m_grid: crescent_image(l_grid, m_grid, radius, inner, offset, pa, floor, fwhm),
            0.08, 1024, U, V, W,
        )  # fmt: skip
        np.testing.assert_allclose(response, oracle, atol=1e-8)
        assert response[0] == 1.0
        assert (
            np.abs(
                response
                - crescent_uv_response(U, V, radius, inner, offset, pa, floor, fwhm)
            ).max()
            > 0.05
        )

    def test_sharp_crescent_closed_form_and_limits(self):
        radius, inner, offset, pa, floor = 0.02, 0.012, 0.006, 0.4, 0.2
        response = crescent_uv_response(U, V, radius, inner, offset, pa, floor)
        # two Airy patterns; the hole is displaced away from the bright side
        rho = np.hypot(U, V)

        def airy(r):
            return np.where(
                rho > 0,
                2 * j1(2 * np.pi * r * rho) / np.where(rho > 0, 2 * np.pi * r * rho, 1),
                1.0,
            )  # noqa: E731

        a_l, a_m = -offset * np.sin(pa), -offset * np.cos(pa)
        weight = (1 - floor) * inner**2
        expected = (
            radius**2 * airy(radius)
            - weight * np.exp(2j * np.pi * (U * a_l + V * a_m)) * airy(inner)
        ) / (radius**2 - weight)
        np.testing.assert_allclose(response, expected, atol=1e-14)
        # oracle of the sharp image (rim-limited accuracy)
        oracle = grid_oracle(
            lambda l_grid, m_grid: crescent_image(l_grid, m_grid, radius, inner, offset, pa, floor),
            0.08, 2048, U, V, np.zeros_like(W),
        )  # fmt: skip
        np.testing.assert_allclose(response, oracle, atol=2e-3)
        # floor = 1 is the uniform disk; offset = 0 and floor = 0 the annulus
        np.testing.assert_allclose(
            crescent_uv_response(U, V, radius, inner, offset, pa, 1.0, w=W),
            limb_darkened_disk_uv_response(U, V, 2 * radius, 2 * radius, 0.0, 0.0, w=W),
            atol=1e-13,
        )
        np.testing.assert_allclose(
            crescent_uv_response(U, V, radius, inner, 0.0, pa, 0.0, fwhm=0.004, w=W),
            annulus_uv_response(U, V, radius, inner, fwhm=0.004, w=W),
            atol=1e-13,
        )
        # the bright side is where the crescent is thick; opposite to it, inside the
        # displaced hole (which reaches offset + inner = 0.018 from the centre), only the floor is left
        bright = crescent_image(
            np.sin(pa) * 0.0175, np.cos(pa) * 0.0175, radius, inner, offset, pa, floor
        )
        dark = crescent_image(
            -np.sin(pa) * 0.0175, -np.cos(pa) * 0.0175, radius, inner, offset, pa, floor
        )
        assert bright > 0
        np.testing.assert_allclose(dark, floor * bright)
        with pytest.raises(ValueError):
            crescent_uv_response(
                U, V, radius, inner, radius, pa
            )  # hole outside the disk


class TestAnnulus:
    def test_blurred_inclined_annulus_matches_oracle(self):
        radius, inner, inclination, pa, fwhm = 0.02, 0.014, np.deg2rad(45), 0.3, 0.004
        response = annulus_uv_response(U, V, radius, inner, inclination, pa, fwhm, w=W)
        oracle = grid_oracle(
            lambda l_grid, m_grid: annulus_image(l_grid, m_grid, radius, inner, inclination, pa, fwhm),
            0.08, 1024, U, V, W,
        )  # fmt: skip
        np.testing.assert_allclose(response, oracle, atol=1e-8)
        assert response[0] == 1.0
        assert (
            np.abs(
                response
                - annulus_uv_response(U, V, radius, inner, inclination, pa, fwhm)
            ).max()
            > 0.05
        )

    def test_sharp_annulus_closed_form_and_disk_limit(self):
        radius, inner, inclination, pa = 0.02, 0.014, np.deg2rad(45), 0.3
        k1, k2 = deproject(U, V, pa, np.cos(inclination))
        rho = np.hypot(k1, k2)

        def airy(r):
            return np.where(
                rho > 0,
                2 * j1(2 * np.pi * r * rho) / np.where(rho > 0, 2 * np.pi * r * rho, 1),
                1.0,
            )  # noqa: E731

        expected = (radius**2 * airy(radius) - inner**2 * airy(inner)) / (
            radius**2 - inner**2
        )
        np.testing.assert_allclose(
            annulus_uv_response(U, V, radius, inner, inclination, pa),
            expected,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            annulus_uv_response(U, V, radius, 0.0, inclination, pa, w=W),
            limb_darkened_disk_uv_response(
                U, V, 2 * radius, 2 * radius * np.cos(inclination), pa, 0.0, w=W
            ),
            atol=1e-13,
        )

        def image(l_grid, m_grid):
            return annulus_image(l_grid, m_grid, radius, inner, inclination, pa)  # noqa: E731

        np.testing.assert_allclose(image_flux(image), 1.0, atol=3e-3)
        with pytest.raises(ValueError):
            annulus_uv_response(U, V, radius, radius, inclination, pa)


class TestExponentialDisk:
    @pytest.mark.parametrize("inclination_deg", [0.0, 60.0])
    def test_matches_polar_oracle_and_closed_form(self, inclination_deg):
        scale, inclination, pa = 0.006, np.deg2rad(inclination_deg), 0.9
        q = np.cos(inclination)
        k1, k2 = deproject(U, V, pa, q)
        response = exponential_disk_uv_response(U, V, scale, inclination, pa, w=W)
        oracle = polar_oracle(
            lambda r: np.exp(-r / scale) / (2 * np.pi * scale**2),
            40 * scale,
            k1,
            k2,
            W,
            q * q * W,
        )
        np.testing.assert_allclose(response, oracle, atol=1e-10)
        tangent = exponential_disk_uv_response(U, V, scale, inclination, pa)
        np.testing.assert_allclose(
            tangent, (1 + 4 * np.pi**2 * scale**2 * (k1**2 + k2**2)) ** -1.5, atol=1e-14
        )
        assert tangent.dtype == np.float64 and abs(response[0] - 1.0) < 1e-14
        assert np.abs(response - tangent).max() > 0.02

        # the sky image integrates to one and is stretched along the major axis
        def image(l_grid, m_grid):
            return exponential_disk_image(l_grid, m_grid, scale, inclination, pa)  # noqa: E731

        np.testing.assert_allclose(image_flux(image, 0.16, 2048), 1.0, atol=1e-5)

    def test_strongly_resolved_baselines(self):
        scale = 0.006
        rho = np.array([2e3, 1e4, 5e4])
        response = exponential_disk_uv_response(
            rho, 0 * rho, scale, w=np.full(3, 2000.0)
        )
        tangent = (1 + 4 * np.pi**2 * scale**2 * rho**2) ** -1.5
        assert np.all(np.abs(response) < 3 * tangent) and np.all(np.isfinite(response))


class TestTaperedPowerLaw:
    @pytest.mark.parametrize(
        ("index", "inclination_deg"), [(0.5, 0.0), (1.0, 55.0), (1.5, 30.0)]
    )
    def test_matches_polar_oracle_and_kummer(self, index, inclination_deg):
        from scipy.special import gamma

        cutoff, inclination, pa = 0.008, np.deg2rad(inclination_deg), 0.4
        q = np.cos(inclination)
        k1, k2 = deproject(U, V, pa, q)
        norm = 1 / (np.pi * (2 * cutoff**2) ** (1 - index / 2) * gamma(1 - index / 2))
        response = tapered_power_law_uv_response(
            U, V, cutoff, index, inclination, pa, w=W
        )
        oracle = polar_oracle(
            lambda r: norm * r ** (-index) * np.exp(-0.5 * r**2 / cutoff**2),
            12 * cutoff,
            k1,
            k2,
            W,
            q * q * W,
            power=2.5,
        )
        np.testing.assert_allclose(response, oracle, atol=1e-10)
        tangent = tapered_power_law_uv_response(U, V, cutoff, index, inclination, pa)
        np.testing.assert_allclose(
            tangent,
            hyp1f1(1 - index / 2, 1.0, -2 * np.pi**2 * cutoff**2 * (k1**2 + k2**2)),
            atol=1e-10,
        )
        assert (
            abs(response[0] - 1.0) < 1e-14 and np.abs(response - tangent).max() > 0.01
        )

    def test_index_zero_is_a_gaussian_and_image_is_normalised(self):
        cutoff = 0.008
        fwhm = cutoff / FWHM_TO_SIGMA
        np.testing.assert_allclose(
            tapered_power_law_uv_response(U, V, cutoff, 0.0, 0.5, 0.2, w=W),
            elliptical_gaussian_uv_response(U, V, fwhm, fwhm * np.cos(0.5), 0.2, w=W),
            atol=1e-14,
        )
        n, extent = 2048, 0.1
        axis = (np.arange(n) - n // 2) * (2 * extent / n)
        l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
        cell = axis[1] - axis[0]
        image = tapered_power_law_image(
            l_grid, m_grid, cutoff, 1.2, np.deg2rad(40), 0.3, pixel_area=cell**2
        )
        assert np.isfinite(image).all()
        np.testing.assert_allclose(image.sum() * cell**2, 1.0, atol=2e-3)
        with pytest.raises(ValueError):
            tapered_power_law_uv_response(U, V, cutoff, 2.0)


class TestShapelet:
    def test_matches_oracle_and_gaussian_limit(self):
        scale = [0.006, 0.004]
        coefficients = np.array(
            [[1.0, 0.3, 0.0, 0.1], [0.2, 0.0, -0.1, 0.0], [0.0, 0.15, 0.0, 0.0]]
        )
        pa = 0.8
        response = shapelet_uv_response(U, V, scale, coefficients, pa, w=W)
        oracle = grid_oracle(
            lambda l_grid, m_grid: shapelet_image(
                l_grid, m_grid, scale, coefficients, pa
            ),
            0.06,
            1024,
            U,
            V,
            W,
        )
        np.testing.assert_allclose(response, oracle, atol=1e-11)
        assert response[0] == 1.0
        assert (
            np.abs(response - shapelet_uv_response(U, V, scale, coefficients, pa)).max()
            > 0.02
        )
        # the zeroth-order shapelet is a Gaussian of sigma = beta along each axis
        beta = 0.005
        np.testing.assert_allclose(
            shapelet_uv_response(U, V, beta, [[1.0]], 0.3, w=W),
            elliptical_gaussian_uv_response(
                U, V, beta / FWHM_TO_SIGMA, beta / FWHM_TO_SIGMA, 0.3, w=W
            ),
            atol=1e-14,
        )

        def image(l_grid, m_grid):
            return shapelet_image(l_grid, m_grid, scale, coefficients, pa)  # noqa: E731

        np.testing.assert_allclose(image_flux(image, 0.06, 1024), 1.0, atol=1e-10)
        with pytest.raises(ValueError):
            shapelet_uv_response(U, V, scale, [[0.0, 1.0]])  # odd order only: zero flux


class TestBlurredDisk:
    @pytest.mark.parametrize(
        ("alpha", "axis_ratio"), [(0.0, 1.0), (1.0, 0.6), (-0.5, 0.7)]
    )
    def test_blurred_limb_darkened_disk_matches_oracle(self, alpha, axis_ratio):
        major, minor, pa, fwhm = 0.04, 0.04 * axis_ratio, 0.5, 0.006
        response = limb_darkened_disk_uv_response(
            U, V, major, minor, pa, alpha, w=W, fwhm=fwhm
        )
        oracle = grid_oracle(
            lambda l_grid, m_grid: limb_darkened_disk_image(l_grid, m_grid, major, minor, pa, alpha, fwhm),
            0.08, 1024, U, V, W,
        )  # fmt: skip
        np.testing.assert_allclose(response, oracle, atol=1e-8)
        assert response[0] == 1.0
        # blur at w = 0 is the sharp response times the Gaussian taper (circular disk)
        if axis_ratio == 1.0:
            sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
            np.testing.assert_allclose(
                limb_darkened_disk_uv_response(
                    U, V, major, minor, pa, alpha, fwhm=fwhm
                ),
                limb_darkened_disk_uv_response(U, V, major, minor, pa, alpha)
                * np.exp(-2 * np.pi**2 * sigma2 * (U**2 + V**2)),
                atol=1e-13,
            )


class TestSimulatorWiring:
    @pytest.mark.parametrize(
        "implementation", ["numpy"] + (["cpp"] if cpp_kernel_available() else [])
    )
    def test_offset_m_ring_matches_dense_point_source_grid(self, implementation):
        phase_center = np.array([[5.2337, 0.7109]])
        source = phase_center + np.array([[0.05, 0.02]])
        radius, fwhm, inclination, pa = 0.008, 0.004, 0.6, 0.4
        beta = [0.3 * np.exp(1j * 1.2)]
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
                {"kind": "analytic", "func": "none", "dish_diameter": 25.0, "blockage_diameter": 0.0, "max_rad_1GHz": 1.0}
            ],
            parallactic_angle=np.zeros(1),
            mueller_selection=np.array([0, 5, 10, 15]),
            implementation=implementation,
        )  # fmt: skip
        ring = calculate_visibilities(
            uvw,
            point_source_flux=None,
            point_source_ra_dec=None,
            sky_components=[
                {"kind": "m_ring", "flux": 1.0, "ra_dec": source[0], "radius": radius, "beta": beta,
                 "fwhm": fwhm, "inclination": inclination, "pa": pa}
            ],
            **kwargs,
        )  # fmt: skip
        n = 121
        axis = np.linspace(-radius - 4 * fwhm, radius + 4 * fwhm, n)
        l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
        cell = axis[1] - axis[0]
        weights = (
            m_ring_image(l_grid, m_grid, radius, beta, fwhm, inclination, pa) * cell**2
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
            **kwargs,
        )
        np.testing.assert_allclose(ring, grid, atol=3e-4)
        assert np.abs(ring[0, :3, 0, 0]).min() < 0.9  # resolved
