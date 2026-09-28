"""The unified sky-component interface: normalisation, registry, rasteriser and the simulator layers."""

import astropy.units as units
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from astroviper.processing_functions.simulation import (
    COMPONENT_KINDS,
    calculate_visibilities,
    component_image,
    component_uv_response,
    normalize_sky_components,
    simulate_processing_set,
    sky_components_from_arrays,
    sky_model_image,
)
from astroviper.processing_functions.simulation.sky_components import (
    describe_sky_components,
    normalize_sky_component,
    slice_sky_components,
)
from astroviper.utils.coordinate_transforms import inverse_sin_project, sin_project
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180 * 3600)
PC = np.array([[5.2337, 0.7109]])


def pf_kwargs(n_time=1, n_frequency=2):
    antenna_position = read_telescope_layout("vla.d").ANTENNA_POSITION.values[:6]
    times = np.array(
        [
            "2019-10-03T19:00:00.000",
            "2019-10-03T19:10:00.000",
            "2019-10-03T19:20:00.000",
        ]
    )[:n_time]
    return dict(
        time=times,
        frequency=np.linspace(1.5e9, 3.0e9, n_frequency),
        polarization=["RR", "LL"],
        antenna_position=antenna_position,
        site_position=antenna_position.mean(axis=0),
        phase_center_ra_dec=PC,
        beam_models=[{"func": "none", "dish_diameter": 25.0, "blockage_diameter": 0.0, "max_rad_1GHz": 0.0149}],
        beam_model_map=np.zeros(len(antenna_position), dtype=int),
    )  # fmt: skip


class TestNormalisation:
    def test_flux_and_position_forms(self):
        base = {"kind": "gaussian", "ra_dec": [5.2, 0.7], "major": 1e-4, "minor": 5e-5}
        scalar = normalize_sky_component({**base, "flux": 2.0})
        np.testing.assert_array_equal(scalar["flux"], [[[2.0, 0.0, 0.0, 2.0]]])
        assert scalar["ra_dec"].shape == (1, 2) and scalar["pa"] == 0.0
        four = normalize_sky_component({**base, "flux": [1, 0.1, 0.1, 0.5]})
        assert four["flux"].shape == (1, 1, 4)
        spectrum = normalize_sky_component(
            {**base, "flux": np.ones((3, 4))}, n_frequency=3
        )
        assert spectrum["flux"].shape == (1, 3, 4)
        cube = normalize_sky_component(
            {**base, "flux": np.ones((2, 3, 4))}, n_time=2, n_frequency=3
        )
        assert cube["flux"].shape == (2, 3, 4)
        with pytest.raises(ValueError, match="frequency axis"):
            normalize_sky_component({**base, "flux": np.ones((3, 4))}, n_frequency=2)
        moving = normalize_sky_component(
            {**base, "flux": 1.0, "ra_dec": np.zeros((2, 2))}, n_time=2
        )
        assert moving["ra_dec"].shape == (2, 2)
        coord = SkyCoord(ra="04h31m38.4s", dec="+18d13m57.7s", frame="icrs")
        sky = normalize_sky_component({**base, "flux": 1.0, "ra_dec": coord})
        np.testing.assert_allclose(sky["ra_dec"], [[coord.ra.rad, coord.dec.rad]])
        fk5 = normalize_sky_component(
            {**base, "flux": 1.0, "ra_dec": coord}, direction_frame="fk5"
        )
        assert (
            abs(fk5["ra_dec"][0, 0] - coord.ra.rad) < 1e-6
            and fk5["ra_dec"][0, 0] != coord.ra.rad
        )

    def test_angles_defaults_and_errors(self):
        component = normalize_sky_component(
            {"kind": "gaussian_ring", "flux": 1.0, "ra_dec": [1.0, 0.5], "radius": 2 * units.arcsec,
             "fwhm": 0.5 * units.arcsec, "inclination": 30 * units.deg}
        )  # fmt: skip
        np.testing.assert_allclose(component["radius"], 2 * ARCSEC)
        np.testing.assert_allclose(component["inclination"], np.deg2rad(30))
        assert component["pa"] == 0.0
        with pytest.raises(ValueError, match="unknown sky component kind"):
            normalize_sky_component({"kind": "blob", "flux": 1.0, "ra_dec": [1, 0.5]})
        with pytest.raises(ValueError, match="missing"):
            normalize_sky_component(
                {"kind": "disk", "flux": 1.0, "ra_dec": [1, 0.5], "major": 1e-4}
            )
        with pytest.raises(ValueError, match="unknown parameter"):
            normalize_sky_component(
                {"kind": "point", "flux": 1.0, "ra_dec": [1, 0.5], "major": 1e-4}
            )
        with pytest.raises(ValueError, match="inclination"):
            normalize_sky_component(
                {
                    "kind": "annulus",
                    "flux": 1.0,
                    "ra_dec": [1, 0.5],
                    "radius": 1e-4,
                    "inner_radius": 5e-5,
                    "inclination": 2.0,
                }
            )
        with pytest.raises(ValueError, match="expected an angle"):
            normalize_sky_component(
                {
                    "kind": "gaussian",
                    "flux": 1.0,
                    "ra_dec": [1, 0.5],
                    "major": 1 * units.Jy,
                    "minor": 1e-5,
                }
            )
        with pytest.raises(TypeError):
            normalize_sky_component(["point"])
        assert normalize_sky_components(None) == []
        assert (
            len(
                normalize_sky_components(
                    {"kind": "point", "flux": 1.0, "ra_dec": [1, 0.5]}
                )
            )
            == 1
        )

    def test_registry_covers_every_kind(self):
        assert set(COMPONENT_KINDS) == {
            "point", "gaussian", "disk", "gaussian_ring", "m_ring", "crescent", "annulus",
            "exponential_disk", "tapered_power_law", "shapelet",
        }  # fmt: skip
        examples = {
            "point": {},
            "gaussian": {"major": 4e-5, "minor": 2e-5, "pa": 0.3},
            "disk": {"major": 4e-5, "minor": 3e-5, "limb_darkening": 1.0, "fwhm": 5e-6},
            "gaussian_ring": {"radius": 3e-5, "fwhm": 1e-5, "inclination": 0.4},
            "m_ring": {"radius": 3e-5, "beta": [0.2j], "fwhm": 1e-5},
            "crescent": {
                "radius": 3e-5,
                "inner_radius": 2e-5,
                "offset": 5e-6,
                "fwhm": 4e-6,
            },
            "annulus": {"radius": 3e-5, "inner_radius": 2e-5, "fwhm": 4e-6},
            "exponential_disk": {"scale_radius": 1e-5, "inclination": 0.5},
            "tapered_power_law": {"cutoff_radius": 2e-5, "index": 1.0},
            "shapelet": {"scale": 1e-5, "coefficients": [[1.0, 0.2], [0.1, 0.0]]},
        }
        u = np.array([0.0, 3e4, -2e4])
        v = np.array([0.0, 1e4, 5e4])
        w = np.array([0.0, 2e5, -1e5])
        axis = (np.arange(256) - 128) * 1e-6
        l_grid, m_grid = np.meshgrid(axis, axis, indexing="ij")
        for kind, shape in examples.items():
            component = normalize_sky_component(
                {"kind": kind, "flux": 1.0, "ra_dec": [1, 0.5], **shape}
            )
            response = component_uv_response(component, u, v, w=w)
            assert response.shape == (3,)
            np.testing.assert_allclose(response[0], 1.0, atol=1e-13)
            if kind == "point":
                np.testing.assert_array_equal(response, 1.0)
                with pytest.raises(ValueError):
                    component_image(component, l_grid, m_grid)
                continue
            assert np.abs(response[1:]).max() < 1.0
            image = component_image(component, l_grid, m_grid, pixel_area=1e-12)
            assert image.shape == l_grid.shape and np.isfinite(image).all()
            np.testing.assert_allclose(image.sum() * 1e-12, 1.0, atol=5e-3)


class TestLegacyArraysAndPoints:
    def test_bulk_arrays_equal_component_list(self):
        kwargs = pf_kwargs(n_time=2, n_frequency=2)
        sources = np.stack([PC[0] + 3e-3, PC[0] - 2e-3])[None]  # [1, 2, 2]
        fluxes = np.array([[[[2.0, 0, 0, 2.0]]], [[[0.7, 0, 0, 0.7]]]])
        shapes = np.array(
            [[600 * ARCSEC, 200 * ARCSEC, 0.8, 0.4], [0.0, 400 * ARCSEC, 0.0, 0.0]]
        )
        disk_shapes = np.array(
            [[300 * ARCSEC, 200 * ARCSEC, 0.4], [120 * ARCSEC, 120 * ARCSEC, 0.0]]
        )
        point_flux = np.array([[[[1.0, 0, 0, 1.0]]]])
        legacy, _ = simulate_processing_set(
            point_source_flux=point_flux, point_source_ra_dec=sources[:, :1],
            gaussian_source_flux=fluxes, gaussian_source_ra_dec=sources, gaussian_source_shape=disk_shapes,
            disk_source_flux=fluxes, disk_source_ra_dec=sources, disk_source_shape=disk_shapes,
            disk_source_limb_darkening=np.array([1.0, -0.5]),
            gaussian_ring_source_flux=fluxes, gaussian_ring_source_ra_dec=sources, gaussian_ring_source_shape=shapes,
            **kwargs,
        )  # fmt: skip
        components = (
            sky_components_from_arrays("point", point_flux, sources[:, :1])
            + sky_components_from_arrays("gaussian", fluxes, sources, disk_shapes)
            + sky_components_from_arrays(
                "disk", fluxes, sources, disk_shapes, [1.0, -0.5]
            )
            + sky_components_from_arrays("gaussian_ring", fluxes, sources, shapes)
        )
        assert [c["kind"] for c in components] == ["point"] + ["gaussian"] * 2 + [
            "disk"
        ] * 2 + ["gaussian_ring"] * 2
        assert (
            components[3]["limb_darkening"] == 1.0
            and components[4]["limb_darkening"] == -0.5
        )
        listed, _ = simulate_processing_set(
            point_source_flux=None,
            point_source_ra_dec=None,
            sky_components=components,
            **kwargs,
        )
        np.testing.assert_allclose(
            listed.VISIBILITY.values, legacy.VISIBILITY.values, rtol=1e-13, atol=1e-14
        )
        assert describe_sky_components(normalize_sky_components(components)) == (
            "1 point source(s), 2 Gaussian source(s), 2 limb-darkened disk source(s), 2 Gaussian ring source(s)"
        )
        with pytest.raises(ValueError, match="shape"):
            sky_components_from_arrays("gaussian", fluxes, sources, shapes[:1])

    def test_point_components_are_grouped_by_layout(self):
        kwargs = pf_kwargs(n_time=2, n_frequency=2)
        positions = PC[0] + np.array([[1e-3, 0.0], [0.0, -2e-3], [2e-3, 1e-3]])
        # one time-dependent, one frequency-dependent and one constant point source
        fluxes = [
            np.array([[[1, 0, 0, 1]], [[2, 0, 0, 2]]]),
            np.array([[[1, 0, 0, 1], [3, 0, 0, 3]]]),
            0.5,
        ]
        components = [
            {"kind": "point", "flux": flux, "ra_dec": pos}
            for flux, pos in zip(fluxes, positions, strict=True)
        ]
        listed, _ = simulate_processing_set(
            point_source_flux=None,
            point_source_ra_dec=None,
            sky_components=components,
            **kwargs,
        )
        expected = np.zeros_like(listed.VISIBILITY.values)
        for component in components:
            flux = normalize_sky_component(component)["flux"]
            one, _ = simulate_processing_set(
                point_source_flux=flux[None],
                point_source_ra_dec=np.asarray(component["ra_dec"])[None, None],
                **kwargs,
            )
            expected += one.VISIBILITY.values
        np.testing.assert_allclose(
            listed.VISIBILITY.values, expected, rtol=1e-13, atol=1e-14
        )
        # no sources at all: zero visibilities
        empty, _ = simulate_processing_set(
            point_source_flux=None, point_source_ra_dec=None, **kwargs
        )
        assert not np.any(empty.VISIBILITY.values)

    def test_slice_sky_components(self):
        components = normalize_sky_components(
            [{"kind": "point", "flux": np.arange(4 * 6 * 4).reshape(4, 6, 4), "ra_dec": np.zeros((4, 2))},
             {"kind": "point", "flux": 1.0, "ra_dec": [0.0, 0.0]}],
            n_time=4, n_frequency=6,
        )  # fmt: skip
        chunk = slice_sky_components(components, slice(1, 3), slice(4, 6))
        assert chunk[0]["flux"].shape == (2, 2, 4) and chunk[0]["ra_dec"].shape == (
            2,
            2,
        )
        np.testing.assert_array_equal(chunk[0]["flux"], components[0]["flux"][1:3, 4:6])
        assert chunk[1]["flux"].shape == (1, 1, 4) and chunk[1]["ra_dec"].shape == (
            1,
            2,
        )


class TestSkyModelImage:
    def test_rasterises_points_and_extended_components(self):
        n = 200
        cell = 0.5 * ARCSEC
        l_axis = (
            -(np.arange(n) - n // 2) * cell
        )  # l decreasing with pixel index (east to the left)
        m_axis = (np.arange(n) - n // 2) * cell
        ring_position = inverse_sin_project(
            PC[0], np.array([[-10 * ARCSEC, 5 * ARCSEC]])
        )[0]
        point_position = inverse_sin_project(
            PC[0], np.array([[20 * ARCSEC, -15 * ARCSEC]])
        )[0]
        components = [
            {"kind": "point", "flux": [1.0, 0, 0, 3.0], "ra_dec": point_position},
            {"kind": "gaussian_ring", "flux": [[[0.5, 0, 0, 0.5]], [[0.25, 0, 0, 0.25]]], "ra_dec": ring_position,
             "radius": 8 * ARCSEC, "fwhm": 3 * ARCSEC, "inclination": 0.5, "pa": 0.3},
            {"kind": "point", "flux": 7.0, "ra_dec": inverse_sin_project(PC[0], np.array([[500 * ARCSEC, 0.0]]))[0]},
        ]  # fmt: skip
        image = sky_model_image(components, l_axis, m_axis, PC[0])
        assert image.shape == (n, n)
        i_l, i_m = n // 2 - 40, n // 2 - 30  # (20'', -15'') -> pixel offsets (-40, -30)
        np.testing.assert_allclose(
            image[i_l, i_m], 2.0
        )  # Stokes I of the first point source
        assert np.isclose(
            image.sum() - 2.0, 0.5, atol=1e-3
        )  # the ring (t=0); the far point is off the grid
        second = sky_model_image(
            components, l_axis, m_axis, PC[0], time_index=1, correlation=3
        )
        np.testing.assert_allclose(second[i_l, i_m], 3.0)
        assert np.isclose(second.sum() - 3.0, 0.25, atol=1e-3)
        # the ring peaks around its own centre
        l0, m0 = sin_project(PC[0], ring_position)
        ring_only = sky_model_image(components[1:2], l_axis, m_axis, PC[0])
        i_peak = np.unravel_index(np.argmax(ring_only), ring_only.shape)
        assert (
            abs(l_axis[i_peak[0]] - l0) < 12 * ARCSEC
            and abs(m_axis[i_peak[1]] - m0) < 12 * ARCSEC
        )


class TestKernelInterface:
    def test_calculate_visibilities_accepts_components_only(self):
        uvw = np.array([[[300.0, 100.0, 30.0], [-600.0, 400.0, -20.0]]])
        kwargs = dict(
            antenna1=np.array([0, 0]), antenna2=np.array([1, 2]), frequency=np.array([1.0e9, 2.0e9]),
            polarization_index=np.array([0, 3]), phase_center_ra_dec=PC, pointing_ra_dec=None,
            beam_model_map=np.zeros(3, dtype=int),
            packed_beam_models=[{"kind": "analytic", "func": "none", "dish_diameter": 25.0, "blockage_diameter": 0.0, "max_rad_1GHz": 1.0}],
            parallactic_angle=np.zeros(1), mueller_selection=np.array([0, 5, 10, 15]), implementation="numpy",
        )  # fmt: skip
        component = {
            "kind": "exponential_disk",
            "flux": [2.0, 0, 0, 1.0],
            "ra_dec": PC[0] + 1e-3,
            "scale_radius": 3e-4,
        }
        visibility = calculate_visibilities(
            uvw,
            point_source_flux=None,
            point_source_ra_dec=None,
            sky_components=[component],
            **kwargs,
        )
        assert visibility.shape == (1, 2, 2, 2)
        point = calculate_visibilities(
            uvw,
            point_source_flux=np.array([[[[2.0, 0, 0, 1.0]]]]),
            point_source_ra_dec=(PC + 1e-3)[None],
            **kwargs,
        )
        from astroviper.processing_functions.simulation.calculate_visibilities import (
            source_frame_uvw,
        )

        u, v, w = source_frame_uvw(uvw, kwargs["frequency"], PC, PC + 1e-3)
        response = component_uv_response(normalize_sky_component(component), u, v, w=w)
        np.testing.assert_allclose(visibility, point * response[..., None], rtol=1e-12)
        assert np.abs(response).min() < 0.99
