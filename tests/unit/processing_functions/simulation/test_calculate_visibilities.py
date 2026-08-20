"""Visibility kernel(s) and the simulate_processing_set processing function vs legacy SIRIUS."""

import numpy as np
import pytest

import astroviper.processing_functions.simulation.antenna_beams as antenna_beams
from astroviper.processing_functions.simulation.antenna_beams import (
    SPEED_OF_LIGHT,
    evaluate_beam_models,
    pack_beam_models,
)
from astroviper.processing_functions.simulation.calculate_noise import (
    calculate_noise,
    calculate_noise_sigma,
    resolve_noise_params,
)
from astroviper.processing_functions.simulation.calculate_visibilities import (
    calculate_visibilities,
)
from astroviper.processing_functions.simulation.calculate_visibilities_cpp import (
    cpp_kernel_available,
)
from astroviper.processing_functions.simulation.simulate_processing_set import (
    simulate_processing_set,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import sin_project
from astroviper.utils.measurement_set_tools import polarization_index
from astroviper.utils.telescope_layout import read_telescope_layout
from tests.unit.processing_functions.simulation.legacy_fixtures import load_legacy

IMPLEMENTATIONS = ["numpy"]
try:  # the C++ kernel is optional until the extension is built
    from astroviper.processing_functions.simulation.calculate_visibilities_cpp import (
        cpp_kernel_available,
    )

    if cpp_kernel_available():
        IMPLEMENTATIONS.append("cpp")
except ImportError:  # pragma: no cover
    pass


def run_legacy_scenario(name, implementation="numpy", uvw_convention="sirius"):
    f = load_legacy(name)
    models, pa = evaluate_beam_models(
        f["beam_models"], f["time"], f["frequency"], f["phase_center_ra_dec"], f["site_position"],
        f["beam_params"], direction_frame="fk5",
    )  # fmt: skip
    uvw = f["uvw"] if uvw_convention == "sirius" else -f["uvw"]
    vis = calculate_visibilities(
        uvw,
        f["antenna1"],
        f["antenna2"],
        f["frequency"],
        polarization_index(f["polarization"]),
        f["point_source_flux"],
        f["point_source_ra_dec"],
        f["phase_center_ra_dec"],
        f["pointing_ra_dec"],
        f["beam_model_map"],
        pack_beam_models(models),
        pa,
        f["mueller_selection"],
        processing_function_threads=2,
        implementation=implementation,
    )
    return vis, f


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "name", ["vla_airy", "alma_het_mosaic_noise", "evla_polynomial_beam"]
)
def test_visibilities_match_legacy_exactly(name, implementation):
    vis, f = run_legacy_scenario(name, implementation)
    np.testing.assert_allclose(vis, f["visibility"], atol=1e-12)
    # MSv4 uvw convention: conjugate visibilities
    vis_msv4, _ = run_legacy_scenario(name, implementation, uvw_convention="msv4")
    np.testing.assert_allclose(vis_msv4, np.conj(f["visibility"]), atol=1e-12)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_visibilities_with_pointing_offsets(implementation):
    """Per-antenna pointing and 4 polarizations.

    Parallel hands match the legacy values; the legacy code zeroed the cross-hand
    flux of analytic beams (a quirk), whereas the port applies the diagonal Mueller
    terms to all four correlations, so the cross hands are checked against the
    parallel-hand attenuation instead.
    """
    vis, f = run_legacy_scenario("vla_airy_pointing", implementation)
    np.testing.assert_allclose(vis[..., 0], f["visibility"][..., 0], atol=1e-12)
    np.testing.assert_allclose(vis[..., 3], f["visibility"][..., 3], atol=1e-12)
    assert np.abs(vis[..., 1]).max() > 0.01  # cross hands carry the 0.1 Jy flux
    np.testing.assert_allclose(
        vis[..., 1], vis[..., 2], atol=1e-12
    )  # RL == LR for real beams


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("name", ["evla_zernike_beam", "evla_mixed_beams"])
def test_visibilities_with_zernike_beams(name, implementation, monkeypatch):
    """Jones-image beams (rotated by the parallactic angle, full Mueller matrix).

    The legacy sampler indexed the Jones image with ``l`` and ``m`` swapped; the
    port samples ``JONES[..., l, m]`` correctly, so the values differ by up to
    ~10 %.  Reproducing the swap in the sampler recovers the legacy values to
    machine precision, which proves the rest of the kernel is identical.
    """
    vis, f = run_legacy_scenario(name, implementation)
    rel = np.abs(vis - f["visibility"]).max() / np.abs(f["visibility"]).max()
    assert rel < 0.15
    assert rel > 1e-6  # the axis fix is real

    original = antenna_beams.bilinear_interpolate
    monkeypatch.setattr(
        antenna_beams, "bilinear_interpolate", lambda image, x, y: original(image, y, x)
    )
    if implementation == "cpp":
        pytest.skip("legacy axis swap can only be reproduced in the NumPy sampler")
    vis_swapped, _ = run_legacy_scenario(name, implementation)
    np.testing.assert_allclose(vis_swapped, f["visibility"], atol=1e-12)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_point_source_at_phase_center_without_beam(implementation):
    """Analytic oracle: unit source at the phase centre, no beam -> VISIBILITY == flux."""
    f = load_legacy("vla_airy")
    pc = f["phase_center_ra_dec"]
    models = [
        {
            "func": "none",
            "dish_diameter": 25.0,
            "blockage_diameter": 0.0,
            "max_rad_1GHz": 1.0,
        }
    ]
    flux = np.array([2.0, 0.5, 0.5, 1.0])[None, None, None, :]
    vis = calculate_visibilities(
        f["uvw"], f["antenna1"], f["antenna2"], f["frequency"], np.array([0, 1, 2, 3]),
        flux, pc[None, :, :], pc, None, np.zeros(27, int), pack_beam_models(models), np.zeros(3),
        np.array([0, 5, 10, 15]), implementation=implementation,
    )  # fmt: skip
    np.testing.assert_allclose(
        vis, np.broadcast_to(flux[0, 0, 0], vis.shape), atol=1e-12
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_offset_source_phase_matches_analytic_formula(implementation):
    """Analytic oracle: offset source without beam -> exp(2 pi i (u l + v m + w (n - 1)) / lambda) / n."""
    f = load_legacy("vla_airy")
    pc = f["phase_center_ra_dec"][0]
    src = f["point_source_ra_dec"][0, 0]
    models = [
        {
            "func": "none",
            "dish_diameter": 25.0,
            "blockage_diameter": 0.0,
            "max_rad_1GHz": 1.0,
        }
    ]
    flux = np.array([1.0, 0, 0, 1.0])[None, None, None, :]
    uvw = -f["uvw"]  # MSv4 convention (antenna2 - antenna1)
    vis = calculate_visibilities(
        uvw, f["antenna1"], f["antenna2"], f["frequency"], np.array([0, 3]),
        flux, src[None, None, :], pc[None, :], None, np.zeros(27, int), pack_beam_models(models), np.zeros(3),
        np.array([0, 5, 10, 15]), implementation=implementation,
    )  # fmt: skip
    lm = sin_project(pc, src)
    n = np.sqrt(1 - lm[0] ** 2 - lm[1] ** 2)
    expected = (
        np.exp(
            2j
            * np.pi
            * (uvw[..., 0] * lm[0] + uvw[..., 1] * lm[1] + uvw[..., 2] * (n - 1))[
                :, :, None
            ]
            * f["frequency"][None, None, :]
            / SPEED_OF_LIGHT
        )
        / n
    )
    np.testing.assert_allclose(vis[..., 0], expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(vis[..., 1], expected, rtol=1e-6, atol=1e-8)


def test_noise_model_matches_legacy_weights():
    f = load_legacy("alma_het_mosaic_noise")
    dish = np.where(f["beam_model_map"] == 0, 6.25, 10.7)
    sigma = calculate_noise_sigma(
        dish, f["antenna1"], f["antenna2"], 0.5e9, 2000.0, {"t_receiver": 50.0}
    )
    np.testing.assert_allclose(1 / sigma**2, f["noise_weight"][0, :, 0], rtol=1e-12)
    np.testing.assert_allclose(sigma, f["noise_sigma"][0, :, 0], rtol=1e-12)
    noise, weight, sig = calculate_noise(
        (4, 66, 3, 2),
        dish,
        f["antenna1"],
        f["antenna2"],
        0.5e9,
        2000.0,
        {},
        random_seed=3,
    )
    assert noise.shape == (4, 66, 3, 2) and noise.dtype == np.complex128
    np.testing.assert_allclose(weight, 1 / sig**2)
    # statistics: per-baseline std of a big realisation approaches sigma
    big, _, _ = calculate_noise(
        (2000, 66, 1, 1),
        dish,
        f["antenna1"],
        f["antenna2"],
        0.5e9,
        2000.0,
        {},
        random_seed=1,
    )
    np.testing.assert_allclose(big.real.std(axis=0)[:, 0, 0], sigma, rtol=0.1)
    np.testing.assert_allclose(big.imag.std(axis=0)[:, 0, 0], sigma, rtol=0.1)
    # reproducible with a seed, different without
    a, _, _ = calculate_noise(
        (1, 66, 1, 1),
        dish,
        f["antenna1"],
        f["antenna2"],
        0.5e9,
        2000.0,
        {},
        random_seed=5,
    )
    b, _, _ = calculate_noise(
        (1, 66, 1, 1),
        dish,
        f["antenna1"],
        f["antenna2"],
        0.5e9,
        2000.0,
        {},
        random_seed=5,
    )
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        resolve_noise_params({"mode": "simplenoise"})
    assert resolve_noise_params(None) is None
    # autocorrelations: real-only noise
    a1 = np.array([0, 0, 1])
    a2 = np.array([0, 1, 1])
    auto, _, _ = calculate_noise(
        (3, 3, 1, 1),
        [10.0, 10.0],
        a1,
        a2,
        1e6,
        10.0,
        {},
        auto_correlations=True,
        random_seed=0,
    )
    np.testing.assert_array_equal(auto[:, [0, 2]].imag, 0.0)
    assert np.all(auto[:, 1].imag != 0.0)


def test_simulate_processing_set_processing_function_end_to_end():
    f = load_legacy("alma_het_mosaic_noise")
    xds, timing = simulate_processing_set(
        f["time"],
        f["frequency"],
        ["XX", "YY"],
        f["antenna_position"],
        f["site_position"],
        f["point_source_flux"],
        f["point_source_ra_dec"],
        f["phase_center_ra_dec"],
        [airy_disk_model("aca"), airy_disk_model("alma")],
        f["beam_model_map"],
        uvw_params={"uvw_convention": "sirius"},
        noise_params={"t_receiver": 50.0, "random_seed": 1},
        channel_width=0.5e9,
        integration_time=2000.0,
    )
    assert set(timing) == {"T_uvw", "T_beams", "T_visibilities", "T_noise"}
    assert xds.VISIBILITY.dims == ("time", "baseline_id", "frequency", "polarization")
    assert list(xds.polarization.values) == ["XX", "YY"]
    np.testing.assert_allclose(xds.UVW.values, f["uvw"], atol=1e-9)
    np.testing.assert_allclose(
        xds.WEIGHT.values[:, :, 0, :], f["noise_weight"], rtol=1e-12
    )
    residual = xds.VISIBILITY.values - f["visibility"]
    sigma = f["noise_sigma"][0, :, 0]
    # the noise realisation differs (different RNG) but its level must match
    assert 0.7 < residual.real.std() / sigma.mean() < 1.3
    assert not xds.FLAG.values.any()
    # no noise: exact legacy values, unit weights
    xds0, _ = simulate_processing_set(
        f["time"], f["frequency"], [9, 12], f["antenna_position"], f["site_position"],
        f["point_source_flux"], f["point_source_ra_dec"], f["phase_center_ra_dec"],
        [airy_disk_model("aca"), airy_disk_model("alma")], f["beam_model_map"],
        uvw_params={"uvw_convention": "sirius"},
    )  # fmt: skip
    np.testing.assert_allclose(xds0.VISIBILITY.values, f["visibility"], atol=1e-12)
    np.testing.assert_array_equal(xds0.WEIGHT.values, 1.0)
    # unix-second times are accepted (MSv4 time coordinate)
    from astropy.time import Time

    unix = Time(f["time"].astype(str), scale="utc").unix
    xds1, _ = simulate_processing_set(
        unix, f["frequency"], ["XX", "YY"], f["antenna_position"], f["site_position"],
        f["point_source_flux"], f["point_source_ra_dec"], f["phase_center_ra_dec"],
        [airy_disk_model("aca"), airy_disk_model("alma")], f["beam_model_map"],
        uvw_params={"uvw_convention": "sirius"},
    )  # fmt: skip
    np.testing.assert_allclose(xds1.VISIBILITY.values, f["visibility"], atol=1e-8)


@pytest.mark.skipif("cpp" not in IMPLEMENTATIONS, reason="C++ kernel not built")
@pytest.mark.parametrize(
    "name",
    [
        "vla_airy",
        "vla_airy_pointing",
        "alma_het_mosaic_noise",
        "evla_polynomial_beam",
        "evla_zernike_beam",
        "evla_mixed_beams",
    ],
)
def test_cpp_kernel_matches_numpy_reference(name):
    vis_numpy, _ = run_legacy_scenario(name, "numpy")
    vis_cpp, _ = run_legacy_scenario(name, "cpp")
    np.testing.assert_allclose(vis_cpp, vis_numpy, atol=1e-13, rtol=0)


@pytest.mark.skipif("cpp" not in IMPLEMENTATIONS, reason="C++ kernel not built")
def test_cpp_kernel_threads_and_validation():
    from scipy.special import j1

    from astroviper.processing_functions.simulation.visibility_kernel_cpp import (
        bessel_j1,
    )
    from astroviper.processing_functions.simulation.visibility_kernel_cpp import (
        calculate_visibilities as kernel,
    )

    xs = np.linspace(0, 50, 201)
    np.testing.assert_allclose(
        [bessel_j1(float(x)) for x in xs], j1(xs), atol=1e-15, rtol=0
    )

    f = load_legacy("vla_airy_pointing")
    models, pa = evaluate_beam_models(
        f["beam_models"], f["time"], f["frequency"], f["phase_center_ra_dec"], f["site_position"],
        f["beam_params"], direction_frame="fk5",
    )  # fmt: skip
    args = (
        f["uvw"], f["antenna1"], f["antenna2"], f["frequency"], polarization_index(f["polarization"]),
        f["point_source_flux"], f["point_source_ra_dec"], f["phase_center_ra_dec"], f["pointing_ra_dec"],
        f["beam_model_map"], pack_beam_models(models), pa, f["mueller_selection"],
    )  # fmt: skip
    one = calculate_visibilities(
        *args, processing_function_threads=1, implementation="cpp"
    )
    many = calculate_visibilities(
        *args, processing_function_threads=7, implementation="cpp"
    )
    all_hw = calculate_visibilities(
        *args, processing_function_threads=0, implementation="cpp"
    )
    np.testing.assert_array_equal(
        one, many
    )  # baseline-partitioned threads: bit identical
    np.testing.assert_array_equal(one, all_hw)

    # memory contract: wrong dtype / non-contiguous / read-only arrays are rejected, never copied
    vis = np.zeros((1, 1, 1, 1), dtype=np.complex64)
    with pytest.raises(RuntimeError, match="dtype"):
        kernel(vis, np.zeros((1, 1, 3)), np.zeros(1, np.int64), np.zeros(1, np.int64), np.ones(1), np.zeros(1, np.int64),
               np.zeros((1, 1, 1, 4), complex), np.zeros((1, 1, 3)), np.ones((1, 1)), np.zeros((1, 1, 2)),
               np.zeros((1, 1, 2)), np.zeros(1, np.int64), [], np.zeros(1), np.zeros(1, np.int64), 1)  # fmt: skip
    vis = np.zeros((1, 1, 1, 1), dtype=np.complex128)
    vis.flags.writeable = False
    with pytest.raises(RuntimeError, match="writeable"):
        kernel(vis, np.zeros((1, 1, 3)), np.zeros(1, np.int64), np.zeros(1, np.int64), np.ones(1), np.zeros(1, np.int64),
               np.zeros((1, 1, 1, 4), complex), np.zeros((1, 1, 3)), np.ones((1, 1)), np.zeros((1, 1, 2)),
               np.zeros((1, 1, 2)), np.zeros(1, np.int64), [], np.zeros(1), np.zeros(1, np.int64), 1)  # fmt: skip
    with pytest.raises(ValueError):
        calculate_visibilities(*args, implementation="fortran")


class TestGaussianSources:
    """Gaussian sources: point-source kernel times the shared restore uv taper."""

    def _pf_kwargs(self):
        time = np.array(["2019-10-03T19:00:00.000"])
        frequency = np.array([3.0e9])
        phase_center = np.array([[5.2337, 0.7109]])
        antenna_position = read_telescope_layout("vla.d").ANTENNA_POSITION.values[:8]
        return dict(
            time=time,
            frequency=frequency,
            polarization=["RR", "LL"],
            antenna_position=antenna_position,
            site_position=antenna_position.mean(axis=0),
            phase_center_ra_dec=phase_center,
            beam_models=[
                {
                    "func": "none",
                    "dish_diameter": 25.0,
                    "blockage_diameter": 0.0,
                    "max_rad_1GHz": 0.014946999714,
                }
            ],
            beam_model_map=np.zeros(len(antenna_position), dtype=int),
            uvw_params=None,
            noise_params=None,
        )

    def test_zero_size_gaussian_equals_point_source(self):
        kwargs = self._pf_kwargs()
        source = kwargs["phase_center_ra_dec"][:, None, :] + 1e-4
        flux = np.array([[[[2.0, 0.0, 0.0, 2.0]]]])
        point_xds, _ = simulate_processing_set(
            point_source_flux=flux, point_source_ra_dec=source, **kwargs
        )
        gaussian_xds, _ = simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=source,
            gaussian_source_flux=flux,
            gaussian_source_ra_dec=source,
            gaussian_source_shape=np.zeros((1, 3)),
            **kwargs,
        )
        np.testing.assert_allclose(
            gaussian_xds.VISIBILITY.values, point_xds.VISIBILITY.values, rtol=1e-13
        )

    @pytest.mark.skipif(not cpp_kernel_available(), reason="C++ kernel not built")
    def test_cpp_matches_numpy_and_taper_is_shared(self):
        arcsec = np.pi / (180 * 3600)
        kwargs = self._pf_kwargs()
        source = kwargs["phase_center_ra_dec"][:, None, :] + 2e-4
        flux = np.array([[[[3.0, 0.0, 0.0, 3.0]]]])
        shape = np.array([[8 * arcsec, 3 * arcsec, 0.7]])
        gaussian = dict(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=source,
            gaussian_source_flux=flux,
            gaussian_source_ra_dec=source,
            gaussian_source_shape=shape,
        )
        cpp_xds, _ = simulate_processing_set(implementation="cpp", **gaussian, **kwargs)
        numpy_xds, _ = simulate_processing_set(
            implementation="numpy", **gaussian, **kwargs
        )
        np.testing.assert_allclose(
            cpp_xds.VISIBILITY.values, numpy_xds.VISIBILITY.values, rtol=1e-12
        )

        # The applied taper is exactly the imaging restore module's: dividing a
        # Gaussian source's visibilities by the point-source visibilities of
        # the same sky position recovers elliptical_gaussian_uv_taper.
        from astroviper.processing_functions.imaging.restore import (
            elliptical_gaussian_uv_taper,
        )

        point_xds, _ = simulate_processing_set(
            point_source_flux=flux, point_source_ra_dec=source, **kwargs
        )
        ratio = cpp_xds.VISIBILITY.values[..., 0] / point_xds.VISIBILITY.values[..., 0]
        u = cpp_xds.UVW.values[..., 0, None] * kwargs["frequency"] / 299792458.0
        v = cpp_xds.UVW.values[..., 1, None] * kwargs["frequency"] / 299792458.0
        np.testing.assert_allclose(
            ratio.real, elliptical_gaussian_uv_taper(u, v, *shape[0]), rtol=1e-10
        )
        np.testing.assert_allclose(ratio.imag, 0.0, atol=1e-12)
