"""Visibilities of a sky image: exact DFT and anti-aliasing degridding against a point source."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.simulation.simulate_processing_set import (
    simulate_processing_set,
)
from astroviper.processing_functions.simulation.sky_image_visibilities import (
    sky_image_visibilities,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import inverse_sin_project
from astroviper.utils.telescope_layout import read_telescope_layout

SPEED_OF_LIGHT = 299792458.0
ARCSEC = np.pi / (180 * 3600)
N = 128
CELL = 0.05 * ARCSEC
PIXEL = (N // 2 + 37, N // 2 - 21)


def _image(stokes=("I",), flux=1.0):
    l = -(np.arange(N) - N // 2) * CELL  # noqa: E741
    m = (np.arange(N) - N // 2) * CELL
    sky = np.zeros((1, 1, len(stokes), N, N))
    sky[0, 0, 0][PIXEL] = flux
    return xr.Dataset(
        {"SKY": (("time", "frequency", "polarization", "l", "m"), sky)},
        coords={
            "time": [0],
            "frequency": [8e9],
            "polarization": list(stokes),
            "l": l,
            "m": m,
        },
    )


def _point_visibilities(image, uvw, frequency):
    """``exp(+2 pi i (u l + v m + w (n - 1))) / n`` of the pixel: [time, baseline, frequency]."""
    l0, m0 = image.l.values[PIXEL[0]], image.m.values[PIXEL[1]]
    n0 = np.sqrt(1 - l0**2 - m0**2)
    k = frequency / SPEED_OF_LIGHT
    phase = (uvw[..., 0:1] * l0 + uvw[..., 1:2] * m0 + uvw[..., 2:3] * (n0 - 1)) * k
    return np.exp(2j * np.pi * phase) / n0


@pytest.fixture
def baselines():
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, 40, 3)) * np.array([3000.0, 3000.0, 300.0]), np.array(
        [7.9e9, 8.1e9]
    )


@pytest.mark.parametrize("w_term", [True, False])
@pytest.mark.parametrize(
    ("stokes", "polarization"),
    [
        (("I",), ["RR", "LL"]),
        (("I",), ["XX", "YY"]),
        (("I", "Q", "U", "V"), ["XX", "XY", "YX", "YY"]),
    ],
)
def test_dft_is_the_exact_point_source(baselines, w_term, stokes, polarization):
    uvw, frequency = baselines
    if not w_term:
        uvw = uvw * np.array([1.0, 1.0, 0.0])
    image = _image(stokes)
    vis = sky_image_visibilities(image, uvw, frequency, polarization, predict="dft")
    exact = _point_visibilities(image, uvw, frequency)
    expected = np.zeros_like(vis)
    expected[..., 0] = expected[..., -1] = exact  # Stokes I: both parallel hands
    np.testing.assert_allclose(vis, expected, rtol=0, atol=1e-10)


def test_gridded_is_the_point_source_to_the_kernel_error(baselines):
    uvw, frequency = baselines
    uvw = uvw * np.array([1.0, 1.0, 0.0])  # the gridded predict has no w term
    image = _image()
    vis = sky_image_visibilities(
        image,
        uvw,
        frequency,
        ["RR", "LL"],
        predict="gridded",
        gridding_kernel_params={
            "type": "prolate_spheroidal",
            "support": 13,
            "oversampling": 10000,
            "interpolation": "linear",
        },
        fft_padding=1.2,
    )
    exact = _point_visibilities(image, uvw, frequency)
    assert np.abs(vis[..., 0] - exact).max() < 1e-8


def test_primary_beam_scales_every_channel(baselines):
    uvw, frequency = baselines
    image = _image()
    beam = np.stack([np.full((N, N), 0.5), np.full((N, N), 0.25)])
    vis = sky_image_visibilities(image, uvw, frequency, ["RR", "LL"], primary_beam=beam)
    exact = _point_visibilities(image, uvw, frequency)
    np.testing.assert_allclose(vis[..., 0], exact * np.array([0.5, 0.25]), atol=1e-10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"predict": "fft"}, "predict must be one of"),
        ({"primary_beam": np.ones((1, N, N))}, "primary_beam must have shape"),
        ({"polarization": ["RR", "RL", "LR", "LL"]}, "not the simulated"),
    ],
)
def test_refusals(baselines, kwargs, message):
    uvw, frequency = baselines
    kwargs = {"polarization": ["RR", "LL"], **kwargs}
    with pytest.raises(ValueError, match=message):
        sky_image_visibilities(_image(), uvw, frequency, **kwargs)


def _pf_kwargs():
    antenna_position = read_telescope_layout("vla.d").ANTENNA_POSITION.values[:8]
    return dict(
        time=np.array(["2019-10-03T19:00:00.000"]),
        frequency=np.array([7.9e9, 8.1e9]),
        polarization=["RR", "LL"],
        antenna_position=antenna_position,
        site_position=antenna_position.mean(axis=0),
        phase_center_ra_dec=np.array([[5.2337, 0.7109]]),
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(len(antenna_position), dtype=int),
    )


def test_simulated_image_pixel_equals_the_point_source():
    """A one-pixel image through the processing function equals a point source
    at the pixel's direction: the same beam, w term and phase."""
    kwargs = _pf_kwargs()
    image = _image(flux=2.0)
    lm = np.array([[image.l.values[PIXEL[0]], image.m.values[PIXEL[1]]]])
    ra_dec = inverse_sin_project(kwargs["phase_center_ra_dec"][0], lm)
    point, _ = simulate_processing_set(
        point_source_flux=np.array([[[[2.0, 0, 0, 2.0]]]]),
        point_source_ra_dec=ra_dec[None],
        **kwargs,
    )
    pixel, timing = simulate_processing_set(
        point_source_flux=np.zeros((1, 1, 1, 4)),
        point_source_ra_dec=ra_dec[None],
        sky_model_image=image,
        **kwargs,
    )
    np.testing.assert_allclose(
        pixel.VISIBILITY.values, point.VISIBILITY.values, rtol=0, atol=1e-9
    )
    assert timing["T_sky_image"] > 0


def test_gridded_with_the_w_term_is_refused():
    with pytest.raises(ValueError, match="no w term"):
        simulate_processing_set(
            point_source_flux=np.zeros((1, 1, 1, 4)),
            point_source_ra_dec=np.array([[[5.2337, 0.7109]]]),
            sky_model_image=_image(),
            sky_model_image_predict={"predict": "gridded"},
            **_pf_kwargs(),
        )
