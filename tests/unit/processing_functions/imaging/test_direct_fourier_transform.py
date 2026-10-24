"""Unit tests for the direct Fourier transform option of the single-field imager.

``gridding_kernel_params={"type": "dft"}`` replaces gridding and FFT by the exact
transforms of :mod:`astroviper.processing_functions.imaging.direct_fourier_transform`,
whose convention is the measurement equation of ``simulate_processing_set``. Point
sources simulated by ``simulate_processing_set`` (four correlations, polarized, the
ALMA 12 m primary beam), with and without the w term, at an off-centre pixel of a small
field (100 GHz) and of a wide one (100 MHz, where ``n - 1 = -1.9e-3`` at the source and
the w term turns the phase by up to a tenth of a turn) must:

* image to their apparent flux at their pixel, and the PSF be 1 at the centre;
* leave a residual at rounding for the true model;
* agree with the gridder on w-off data to the gridder's kernel error;
* give the same images in repeated runs and for every thread count, bit for bit.

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
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    make_gridding_kernel,
)
from astroviper.processing_functions.imaging.make_point_spread_function import (
    make_subtraction_point_spread_function_single_field,
)
from astroviper.processing_functions.imaging.residual_update import (
    imaging_setup_single_field,
    residual_update_cube_single_field,
)
from astroviper.processing_functions.imaging.utils.fft_sizing import padded_grid_size
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import inverse_sin_project
from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.telescope_layout import read_telescope_layout

DFT = {"type": "dft"}
# the exponential of a semicircle of support 7 with linear lookup in a table of 300
# samples per cell: a gridder kernel with a floor far below the sky's 1 / n here
ES7 = {
    "type": "exponential_semicircle",
    "support": 7,
    "oversampling": 300,
    "interpolation": "linear",
}
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
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
FIELDS = {
    # frequency (Hz), cell (rad), image size, source pixel offset from the centre
    "small_100GHz": (100e9, 0.7 * np.pi / (180 * 3600), 64, (13, -9)),
    "wide_100MHz": (100e6, 2e-3, 64, (25, -18)),
}
FFT_PADDING = 1.2


def _empty_image(field):
    frequency, cell, size, _ = FIELDS[field]
    return make_empty_sky_image(
        phase_center=PHASE_CENTER,
        image_size=[size, size],
        cell_size=[-cell, cell],
        frequency_coords=[frequency],
        pol_coords=["XX", "XY", "YX", "YY"],
        time_coords=[0],
        do_sky_coords=False,
    )


def _image_params(field):
    frequency, cell, size, _ = FIELDS[field]
    return {
        "image_size": [size, size],
        "cell_size": [-cell, cell],
        "phase_direction": PHASE_CENTER,
        "time_coords": [0],
        "polarization_coords": ["I", "Q", "U", "V"],
        "fft_padding": FFT_PADDING,
    }


def _simulate(root, field, w_term):
    """A point source on the centre of an off-centre pixel; returns (store, pixel)."""
    frequency, _, size, offset = FIELDS[field]
    img_xds = _empty_image(field)
    pixel = (size // 2 + offset[0], size // 2 + offset[1])
    lm = np.array([img_xds.l.values[pixel[0]], img_xds.m.values[pixel[1]]])
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(8))
    )
    store = str(root / f"{field}_w{int(w_term)}.ps.zarr")
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
                "freq_start": frequency,
                "freq_delta": 1e-3 * frequency,
                "n_channels": 1,
                "channel_width": 1e-5 * frequency,
            },
            polarization=["XX", "XY", "YX", "YY"],
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
    return store, pixel


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """``simulated(field, w_term)`` -> (store, source pixel), simulated once each."""
    root = tmp_path_factory.mktemp("dft_point_sources")
    cache = {}

    def get(field, w_term):
        if (field, w_term) not in cache:
            cache[(field, w_term)] = _simulate(root, field, w_term)
        return cache[(field, w_term)]

    return get


def _add_model(img_xds, pixel, stokes):
    """The ``model`` image group holding ``stokes`` (Jy, apparent) at ``pixel``."""
    _, data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name="residual",
        data_group_out_name="model",
        data_group_out_modified={"sky": "SKY_MODEL"},
        overwrite=True,
    )
    model = xr.zeros_like(img_xds["SKY_RESIDUAL"])
    model.values[0, 0, :, pixel[0], pixel[1]] = stokes
    img_xds[data_group_out["sky"]] = model
    modify_data_groups_xds(
        img_xds, "model", data_group_out, description="Point-source test model."
    )
    return img_xds


def _image(store, field, kernel, pixel=None, threads=1, single_precision=False):
    """Setup, dirty residual update and, with ``pixel``, the true model's residual update."""
    ps_xdt = load_processing_set(store)
    common = dict(
        processing_set_data_group_name="base",
        single_precision_image=single_precision,
        fft_backend="scipy",
        gridding_kernel_params=kernel,
        processing_function_threads=threads,
    )
    image_params = _image_params(field)
    img_xds, _ = imaging_setup_single_field(
        ps_xdt, _empty_image(field), image_params, {"weighting": "natural"}, **common
    )
    img_xds, _ = residual_update_cube_single_field(
        ps_xdt, img_xds, image_params, model_exists=False, **common
    )
    out = {
        "ps_xdt": ps_xdt,
        "dirty": img_xds["SKY_RESIDUAL"].values[0, 0].copy(),
        "psf": img_xds["POINT_SPREAD_FUNCTION"].values[0, 0].copy(),
        "primary_beam": img_xds["PRIMARY_BEAM"].values[0, 0, 0].astype(float),
    }
    if pixel is not None:
        apparent = STOKES * out["primary_beam"][pixel]
        img_xds = _add_model(img_xds, pixel, apparent)
        img_xds, _ = residual_update_cube_single_field(
            ps_xdt, img_xds, image_params, model_exists=True, **common
        )
        out["residual"] = img_xds["SKY_RESIDUAL"].values[0, 0].copy()
        ms_xdt = next(iter(ps_xdt.values()))
        out["model_visibilities"] = ms_xdt["VISIBILITY_MODEL"].values.copy()
        out["residual_visibilities"] = ms_xdt["VISIBILITY_RESIDUAL"].values.copy()
        out["visibilities"] = ms_xdt["VISIBILITY"].values.copy()
    out["img_xds"] = img_xds
    return out


def _n(field, pixel):
    img_xds = _empty_image(field)
    l, m = img_xds.l.values[pixel[0]], img_xds.m.values[pixel[1]]  # noqa: E741
    return np.sqrt(1.0 - l**2 - m**2)


@pytest.mark.parametrize("w_term", [False, True])
@pytest.mark.parametrize("field", list(FIELDS))
def test_dirty_image_is_the_apparent_flux(simulated, field, w_term):
    store, pixel = simulated(field, w_term)
    out = _image(store, field, DFT)
    apparent = STOKES * out["primary_beam"][pixel]
    assert out["primary_beam"][pixel] < 0.95  # the primary beam is part of the test
    np.testing.assert_allclose(
        out["dirty"][:, pixel[0], pixel[1]], apparent, rtol=0, atol=1e-12
    )
    assert np.unravel_index(np.argmax(out["dirty"][0]), out["dirty"][0].shape) == pixel
    centre = tuple(size // 2 for size in out["psf"].shape[1:])
    np.testing.assert_allclose(
        out["psf"][(slice(None), *centre)], 1.0, rtol=0, atol=1e-15
    )


@pytest.mark.parametrize("w_term", [False, True])
@pytest.mark.parametrize("field", list(FIELDS))
def test_residual_of_the_true_model_is_rounding(simulated, field, w_term):
    """Measured: residual visibilities up to 7e-11 (small field) and 2e-14 (wide field).

    The floor is that of the simulated input, not of the transforms: a source
    simulated at the phase centre, whose prediction is the model itself, leaves
    a residual of the same order (the rounding of the simulator's source
    direction, a phase error that grows with ``|u|`` and so with frequency);
    hence the 1e-9 bound.
    """
    store, pixel = simulated(field, w_term)
    out = _image(store, field, DFT, pixel=pixel)
    visibility_scale = np.abs(out["visibilities"]).max()
    assert np.abs(out["residual_visibilities"]).max() < 1e-9 * visibility_scale
    apparent_i = STOKES[0] * out["primary_beam"][pixel]
    assert np.abs(out["residual"]).max() < 1e-9 * apparent_i


def test_the_image_carries_the_n_the_two_dimensional_gridder_does_not(simulated):
    """The simulator divides by n: the DFT's imaging multiplies by it, the 2-D gridder does not."""
    store, pixel = simulated("wide_100MHz", False)
    n = _n("wide_100MHz", pixel)
    assert 1.0 - n > 1e-3  # the test resolves the factor
    dft = _image(store, "wide_100MHz", DFT)
    gridder = _image(store, "wide_100MHz", ES7)
    apparent = STOKES * dft["primary_beam"][pixel]
    np.testing.assert_allclose(
        dft["dirty"][:, pixel[0], pixel[1]], apparent, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        gridder["dirty"][:, pixel[0], pixel[1]], apparent / n, rtol=0, atol=2e-5
    )


def test_agreement_with_the_gridder_to_its_kernel_error(simulated):
    """On w-off data the DFT and the gridder differ by the kernel's map error only."""
    store, pixel = simulated("small_100GHz", False)
    dft = _image(store, "small_100GHz", DFT, pixel=pixel)
    gridder = _image(store, "small_100GHz", ES7, pixel=pixel)
    size = dft["dirty"].shape[-1]
    x = (np.arange(size) - size // 2) / padded_grid_size([size, size], FFT_PADDING)[0]
    error = make_gridding_kernel(ES7, FFT_PADDING).map_error(x)
    floor = error[:, None] + error[None, :]  # per pixel, the two axes' errors add
    # measured: 0.27 (dirty), 0.24 (psf) and 0.03 (residual) of the largest map error
    for name, scale in (("dirty", STOKES[0]), ("psf", 1.0), ("residual", STOKES[0])):
        difference = np.abs(dft[name] - gridder[name]).max()
        assert difference <= scale * floor.max(), name
    # at the source the dirty images differ by 0.08 of the map error there
    difference = np.abs(dft["dirty"] - gridder["dirty"])[:, pixel[0], pixel[1]].max()
    assert difference <= STOKES[0] * floor[pixel]
    # the residual of the true model: rounding for the DFT, the kernel floor for the gridder
    assert np.abs(dft["residual"]).max() < 1e-4 * np.abs(gridder["residual"]).max()


def test_deterministic_across_runs_and_thread_counts(simulated):
    store, pixel = simulated("small_100GHz", True)
    runs = [
        _image(store, "small_100GHz", DFT, pixel=pixel, threads=t) for t in (1, 1, 3, 0)
    ]
    for run in runs[1:]:
        for name in ("dirty", "psf", "residual", "model_visibilities"):
            np.testing.assert_array_equal(run[name], runs[0][name])


def test_single_precision_image(simulated):
    store, pixel = simulated("small_100GHz", True)
    double = _image(store, "small_100GHz", DFT, pixel=pixel)
    single = _image(store, "small_100GHz", DFT, pixel=pixel, single_precision=True)
    assert single["dirty"].dtype == np.float32 and single["psf"].dtype == np.float32
    np.testing.assert_allclose(single["dirty"], double["dirty"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(single["psf"], double["psf"], rtol=0, atol=1e-6)
    # the model visibilities stay double precision
    assert single["model_visibilities"].dtype == np.complex128


def test_flagged_samples_and_the_normalization(simulated):
    """Flagged samples are left out of the sums and of the sum of the weights."""
    store, pixel = simulated("small_100GHz", True)
    ps_xdt = load_processing_set(store)
    ms_xdt = next(iter(ps_xdt.values()))
    flag = ms_xdt["FLAG"].values
    flag[np.random.default_rng(0).random(flag.shape[:2]) < 0.2] = True
    common = dict(
        processing_set_data_group_name="base",
        single_precision_image=False,
        fft_backend="scipy",
        gridding_kernel_params=DFT,
    )
    image_params = _image_params("small_100GHz")
    img_xds, _ = imaging_setup_single_field(
        ps_xdt,
        _empty_image("small_100GHz"),
        image_params,
        {"weighting": "natural"},
        **common,
    )
    img_xds, _ = residual_update_cube_single_field(
        ps_xdt, img_xds, image_params, model_exists=False, **common
    )
    assert img_xds.attrs["data_groups"]["residual"]["sky"] == "SKY_RESIDUAL"
    assert "VISIBILITY_NORMALIZATION" in img_xds
    # the transform of the unflagged samples, evaluated here at three pixels
    vis = ms_xdt["VISIBILITY"].values[:, :, 0]
    weight = ms_xdt["WEIGHT_IMAGING"].values[:, :, 0]
    uvw = ms_xdt["UVW"].values * ms_xdt.frequency.values[0] / 299792458.0
    valid = np.isfinite(weight)
    assert 0 < valid.sum() < valid.size
    for i, j in (pixel, (32, 32), (5, 50)):
        l, m = img_xds.l.values[i], img_xds.m.values[j]  # noqa: E741
        n = np.sqrt(1 - l**2 - m**2)
        fringe = np.exp(
            -2j * np.pi * (uvw[..., 0] * l + uvw[..., 1] * m + uvw[..., 2] * (n - 1))
        )
        correlations = np.array(
            [
                n
                * np.sum(
                    np.where(valid[..., p], weight[..., p] * vis[..., p] * fringe, 0)
                )
                / np.sum(np.where(valid[..., p], weight[..., p], 0))
                for p in range(4)
            ]
        )
        expected = (LINEAR_CORR_TO_STOKES @ correlations).real
        np.testing.assert_allclose(
            img_xds["SKY_RESIDUAL"].values[0, 0, :, i, j], expected, rtol=0, atol=1e-12
        )


def test_subtraction_psf_of_twice_the_size_holds_the_image_psf(simulated):
    store, _ = simulated("small_100GHz", True)
    out = _image(store, "small_100GHz", DFT)
    img_xds = out["img_xds"]
    psf_2x, _ = make_subtraction_point_spread_function_single_field(
        out["ps_xdt"],
        img_xds,
        _image_params("small_100GHz"),
        ["XX", "XY", "YX", "YY"],
        ms_data_group_in_name="base",
        gridding_kernel_params=DFT,
    )
    n_l, n_m = img_xds.sizes["l"], img_xds.sizes["m"]
    rows = slice(n_l - n_l // 2, n_l - n_l // 2 + n_l)
    columns = slice(n_m - n_m // 2, n_m - n_m // 2 + n_m)
    np.testing.assert_allclose(
        psf_2x[0, 0, :, rows, columns], out["psf"], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(psf_2x[0, 0, :, n_l, n_m], 1.0, rtol=0, atol=1e-15)


def test_refused_kernel_keys_with_the_dft(simulated):
    store, _ = simulated("small_100GHz", False)
    for extra in ({"support": 7}, {"oversampling": 100}, {"interpolation": "linear"}):
        with pytest.raises(ValueError, match="do not apply to the type 'dft'"):
            _image(store, "small_100GHz", {**DFT, **extra})
