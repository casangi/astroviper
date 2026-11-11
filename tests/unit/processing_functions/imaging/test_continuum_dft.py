"""Unit tests for the direct Fourier transform option of continuum (MFS) imaging.

``gridding_kernel_params={"type": "dft"}`` makes the Taylor residuals, the Taylor
PSFs and the model visibilities of a continuum map task by the exact transforms of
:mod:`astroviper.processing_functions.imaging.direct_fourier_transform`, whose
convention is the measurement equation of ``simulate_processing_set``. A polarized
point source simulated on an off-centre pixel (four correlations, the ALMA 12 m
primary beam, one channel or two channels 4 percent apart), with and without the w
term, must:

* image to its apparent flux at its pixel: Taylor term ``t`` there is the
  weighted mean of ``x**t`` times the apparent flux of each channel, and order
  ``k`` of the PSF at the phase centre the weighted mean of ``x**k``;
* leave a residual at rounding for the true Taylor model (with two channels the
  two Taylor terms that pass through the two apparent fluxes);
* give the same products in repeated runs and for every thread count, bit for bit.

All data are small and simulated (no downloads).
"""

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.image import make_empty_sky_image
from xradio.measurement_set import load_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
    LINEAR_CORR_TO_STOKES,
)
from astroviper.processing_functions.imaging.direct_fourier_transform import (
    normalize_dft_image,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    prepare_model_uv_continuum_single_field,
    residual_update_continuum_single_field,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import inverse_sin_project
from astroviper.utils.telescope_layout import read_telescope_layout

DFT = {"type": "dft"}
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
CELL = 0.7 * np.pi / (180 * 3600)
SIZE = 64
PIXEL = (SIZE // 2 + 13, SIZE // 2 - 9)
STOKES = np.array([1.0, 0.2, -0.1, 0.05])  # I, Q, U, V of the source, Jy
# linear feeds: XX = I + Q, XY = U + iV, YX = U - iV, YY = I - Q
CORRELATION_FLUX = np.array(
    [
        STOKES[0] + STOKES[1],
        STOKES[2] + 1j * STOKES[3],
        STOKES[2] - 1j * STOKES[3],
        STOKES[0] - STOKES[1],
    ]
)
CORRELATIONS = ["XX", "XY", "YX", "YY"]
FREQUENCY = 100e9
FREQUENCY_DELTA = 4e9
C = 299792458.0


def _empty_image(frequencies, polarization=CORRELATIONS):
    return make_empty_sky_image(
        phase_center=PHASE_CENTER,
        image_size=[SIZE, SIZE],
        cell_size=[-CELL, CELL],
        frequency_coords=frequencies,
        pol_coords=polarization,
        time_coords=[0],
        do_sky_coords=False,
    )


def _simulate(root, w_term, n_channels):
    img_xds = _empty_image([FREQUENCY])
    lm = np.array([img_xds.l.values[PIXEL[0]], img_xds.m.values[PIXEL[1]]])
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(8))
    )
    store = str(root / f"point_w{int(w_term)}_{n_channels}chan.ps.zarr")
    with dask.config.set(scheduler="synchronous"):
        distributed_applications.simulation.simulate_processing_set(
            ps_store=store,
            antenna_xds=antenna_xds,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 600.0,
                "n_samples": 6,
            },
            frequency_params={
                "freq_start": FREQUENCY,
                "freq_delta": FREQUENCY_DELTA,
                "n_channels": n_channels,
                "channel_width": 2e6,
            },
            polarization=CORRELATIONS,
            sky_components=[
                {
                    "kind": "point",
                    "flux": CORRELATION_FLUX,
                    "ra_dec": inverse_sin_project(PHASE_CENTER, lm),
                }
            ],
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[airy_disk_model("alma")],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            uvw_params={"w_term": w_term},
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    return store


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """``simulated(w_term, n_channels)`` -> store, simulated once each."""
    root = tmp_path_factory.mktemp("continuum_dft_point_source")
    cache = {}

    def get(w_term, n_channels):
        if (w_term, n_channels) not in cache:
            cache[(w_term, n_channels)] = _simulate(root, w_term, n_channels)
        return cache[(w_term, n_channels)]

    return get


def _image_params(frequencies):
    return {
        "image_size": [SIZE, SIZE],
        "cell_size": [-CELL, CELL],
        "phase_direction": PHASE_CENTER,
        "time_coords": [0],
        "polarization_coords": ["I", "Q", "U", "V"],
        "fft_padding": 1.2,
        "frequency_coords": frequencies,
        "n_taylor_terms": len(frequencies),
        "reference_frequency": float(np.mean(frequencies)),
    }


def _residual_update(
    ps_xdt, image_params, threads, model_uv_xds=None, single_precision=False
):
    """One map task's residual update, normalised as the append node does."""
    img_xds, _ = residual_update_continuum_single_field(
        ps_xdt,
        _empty_image(image_params["frequency_coords"]),
        image_params,
        {"weighting": "natural"},
        continuum_mode="mfs",
        processing_set_data_group_name="base",
        single_precision_image=single_precision,
        processing_function_threads=threads,
        fft_backend="scipy",
        is_first_cycle=model_uv_xds is None,
        model_uv_xds=model_uv_xds,
        gridding_kernel_params=DFT,
    )
    normalize_dft_image(img_xds, "residual", uv_sampling=False)
    if model_uv_xds is None:
        normalize_dft_image(img_xds, "residual", uv_sampling=True)
    return img_xds


def _apparent_fluxes(ms_xdt):
    """The correlation fluxes of the source per channel, from the visibilities.

    At the source's pixel ``V exp(-i phase) n`` is the apparent flux of every
    sample; returns its mean over the samples and their largest deviation from it.
    """
    img_xds = _empty_image([FREQUENCY])
    l, m = img_xds.l.values[PIXEL[0]], img_xds.m.values[PIXEL[1]]  # noqa: E741
    n = np.sqrt(1.0 - l**2 - m**2)
    uvw = ms_xdt["UVW"].values[:, :, None, :] * ms_xdt.frequency.values[:, None] / C
    phase = uvw[..., 0] * l + uvw[..., 1] * m + uvw[..., 2] * (n - 1.0)
    samples = ms_xdt["VISIBILITY"].values * np.exp(-2j * np.pi * phase)[..., None] * n
    samples = samples.reshape(-1, *samples.shape[2:])  # (rows, channel, correlation)
    flux = samples.mean(axis=0)
    return flux, np.abs(samples - flux).max()


def _true_model(apparent, frequencies, reference_frequency):
    """The Stokes Taylor model whose terms pass through the apparent fluxes."""
    x = (np.asarray(frequencies) - reference_frequency) / reference_frequency
    # sum over t of x_k**t M_t = apparent_k at every channel k
    taylor = np.linalg.solve(np.vander(x, len(x), increasing=True), apparent)
    stokes = (LINEAR_CORR_TO_STOKES @ taylor.T).T.real
    template = _empty_image([reference_frequency], ["I", "Q", "U", "V"])
    model = np.zeros((1, len(x), 4, SIZE, SIZE))
    model[0, :, :, PIXEL[0], PIXEL[1]] = stokes
    sky = xr.DataArray(
        model,
        dims=("time", "taylor_term", "polarization", "l", "m"),
        coords={
            "time": template.time,
            "taylor_term": np.arange(len(x), dtype=np.int32),
            "polarization": template.polarization,
            "l": template.l,
            "m": template.m,
        },
    )
    return xr.Dataset(
        {"SKY_MODEL": sky}, attrs={"data_groups": {"model": {"sky": "SKY_MODEL"}}}
    )


def _image(store, threads=1, true_model=False, single_precision=False):
    """Products of the first residual update and, optionally, of the true model's."""
    ps_xdt = load_processing_set(store)
    ms_xdt = next(iter(ps_xdt.values()))
    frequencies = ms_xdt.frequency.values
    image_params = _image_params(frequencies)
    img_xds = _residual_update(ps_xdt, image_params, threads, None, single_precision)
    out = {
        "frequencies": frequencies,
        "weight_sums": np.nansum(ms_xdt["WEIGHT_IMAGING"].values, axis=(0, 1)),
        "dirty": img_xds["SKY_RESIDUAL"].values[0].copy(),
        "psf": img_xds["POINT_SPREAD_FUNCTION"].values[0].copy(),
        "primary_beam": img_xds["PRIMARY_BEAM_REFERENCE"].values[0, 0],
        "img_xds": img_xds,
        "ms_xdt": ms_xdt,
    }
    out["apparent"], out["apparent_spread"] = _apparent_fluxes(ms_xdt)
    if true_model:
        model_uv_xds = prepare_model_uv_continuum_single_field(
            _true_model(
                out["apparent"], frequencies, image_params["reference_frequency"]
            ),
            image_params,
            single_precision_image=single_precision,
            gridding_kernel_params=DFT,
        )
        img_xds = _residual_update(
            ps_xdt, image_params, threads, model_uv_xds, single_precision
        )
        out["residual"] = img_xds["SKY_RESIDUAL"].values[0].copy()
        out["model_visibilities"] = ms_xdt["VISIBILITY_MODEL"].values.copy()
        out["residual_visibilities"] = ms_xdt["VISIBILITY_RESIDUAL"].values.copy()
    return out


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("w_term", [False, True])
def test_taylor_terms_of_the_dirty_image_are_the_apparent_flux(
    simulated, w_term, n_channels
):
    out = _image(simulated(w_term, n_channels))
    frequencies = out["frequencies"]
    x = (frequencies - frequencies.mean()) / frequencies.mean()
    weight_sums = out["weight_sums"]  # (channel, correlation)
    # the simulated visibilities follow the convention to the simulator's
    # rounding (a wrong sign or a missing n would spread them by percents)
    assert out["apparent_spread"] < 1e-9 * np.abs(out["apparent"]).max()
    assert out["dirty"].shape == (n_channels, 4, SIZE, SIZE)
    for t in range(n_channels):
        expected = np.sum(weight_sums * x[:, None] ** t * out["apparent"], axis=0)
        expected /= weight_sums.sum(axis=0)
        np.testing.assert_allclose(
            out["dirty"][t, :, PIXEL[0], PIXEL[1]], expected, rtol=0, atol=1e-12
        )
    if n_channels == 1:
        # one channel at the reference frequency: the flux times the imager's
        # beam, which differs from the simulator's by 1.1e-10 here (measured)
        primary_beam = out["primary_beam"][PIXEL]
        assert primary_beam < 0.95
        np.testing.assert_allclose(
            out["dirty"][0, :, PIXEL[0], PIXEL[1]],
            CORRELATION_FLUX * primary_beam,
            rtol=0,
            atol=1e-9,
        )
    # the PSF orders at the phase centre: the weighted means of x**k
    assert out["psf"].shape == (2 * n_channels - 1, 4, SIZE, SIZE)
    for k in range(2 * n_channels - 1):
        expected = np.sum(weight_sums * x[:, None] ** k, axis=0)
        expected /= weight_sums.sum(axis=0)
        np.testing.assert_allclose(
            out["psf"][k, :, SIZE // 2, SIZE // 2], expected, rtol=0, atol=1e-15
        )
    # the normalization of every Taylor plane is the sum of the imaging weights
    img_xds = out["img_xds"]
    for name in ("VISIBILITY_NORMALIZATION", "UV_SAMPLING_NORMALIZATION"):
        np.testing.assert_allclose(
            img_xds[name].values[0],
            np.broadcast_to(weight_sums.sum(axis=0), img_xds[name].shape[1:]),
            rtol=1e-14,
            atol=0,
        )


@pytest.mark.parametrize("n_channels", [1, 2])
@pytest.mark.parametrize("w_term", [False, True])
def test_residual_of_the_true_taylor_model_is_rounding(simulated, w_term, n_channels):
    """Measured: residual visibilities up to about 1e-10 of the largest visibility.

    The floor is that of the simulated input (the rounding of the simulator's
    source direction, see test_direct_fourier_transform.py); hence the 1e-9 bound.
    """
    out = _image(simulated(w_term, n_channels), true_model=True)
    visibility_scale = np.abs(out["ms_xdt"]["VISIBILITY"].values).max()
    assert np.abs(out["residual_visibilities"]).max() < 1e-9 * visibility_scale
    assert np.abs(out["residual"]).max() < 1e-9 * np.abs(out["apparent"]).max()
    # the residual is no smaller than the dirty image by accident
    assert np.abs(out["dirty"]).max() > 0.5 * np.abs(out["apparent"]).max()


def test_deterministic_across_runs_and_thread_counts(simulated):
    store = simulated(True, 2)
    runs = [_image(store, threads=t, true_model=True) for t in (1, 1, 3, 0)]
    for run in runs[1:]:
        for name in ("dirty", "psf", "residual", "model_visibilities"):
            np.testing.assert_array_equal(run[name], runs[0][name])


def test_single_precision_products(simulated):
    store = simulated(True, 2)
    double = _image(store, true_model=True)
    single = _image(store, true_model=True, single_precision=True)
    # complex in the four-correlation basis, as the gridder path makes them
    assert single["dirty"].dtype == np.complex64
    assert single["psf"].dtype == np.float32
    assert single["residual"].dtype == np.complex64
    for name in ("dirty", "psf", "residual"):
        np.testing.assert_allclose(single[name], double[name], rtol=0, atol=1e-6)
    # the model visibilities stay double precision
    assert single["model_visibilities"].dtype == np.complex128


def test_a_read_only_sum_is_normalised_into_a_copy():
    """A sum received from another process may be read-only."""
    image = np.arange(8.0).reshape(1, 2, 1, 2, 2)
    image.flags.writeable = False
    img_xds = xr.Dataset(
        {
            "SKY_RESIDUAL": (("time", "taylor_term", "polarization", "l", "m"), image),
            "VISIBILITY_NORMALIZATION": (
                ("time", "taylor_term", "polarization"),
                np.full((1, 2, 1), 4.0),
            ),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "visibility_normalization": "VISIBILITY_NORMALIZATION",
                }
            }
        },
    )
    normalize_dft_image(img_xds, "residual")
    np.testing.assert_array_equal(img_xds["SKY_RESIDUAL"].values, image / 4.0)
    np.testing.assert_array_equal(image, np.arange(8.0).reshape(1, 2, 1, 2, 2))


def test_the_model_for_the_dft_is_its_taylor_images_in_the_correlation_basis():
    """No FFT: the map tasks predict from the model images, as named."""
    model_xds = _true_model(CORRELATION_FLUX[None, :], [FREQUENCY], FREQUENCY)
    model_uv_xds = prepare_model_uv_continuum_single_field(
        model_xds,
        _image_params(np.array([FREQUENCY])),
        single_precision_image=False,
        image_data_group_out_name="model_dft",
        gridding_kernel_params=DFT,
    )
    assert "VISIBILITY_MODEL" not in model_uv_xds
    for group in ("model", "model_dft"):
        assert model_uv_xds.attrs["data_groups"][group]["sky"] == "SKY_MODEL"
    model = model_uv_xds["SKY_MODEL"].transpose(
        "time", "taylor_term", "polarization", "l", "m"
    )
    assert list(model.polarization.values) == CORRELATIONS
    np.testing.assert_allclose(
        model.values[0, 0, :, PIXEL[0], PIXEL[1]],
        CORRELATION_FLUX,
        rtol=0,
        atol=1e-15,
    )
    # the Stokes model of the next model update is not changed
    assert list(model_xds.polarization.values) == ["I", "Q", "U", "V"]
