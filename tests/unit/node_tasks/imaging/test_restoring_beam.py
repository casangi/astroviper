"""``restoring_beam`` in the node tasks: the cube task hands it to the science
function, the continuum append preparation refuses a target that does not
contain the PSF beam right after the PSF fit, and finalization hands it to the
continuum restore."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import astroviper.node_tasks.imaging.image_continuum_single_field as continuum_node
import astroviper.processing_functions.imaging.image_continuum_single_field as continuum_processing
from tests.unit.node_tasks.imaging.test_channel_loop import (
    _fake_processing_set,
    _FakeScience,
    _task_inputs,
)

# The package exports the function common_beam under the module's name.
common_beam = importlib.import_module(
    "astroviper.processing_functions.image_analysis.common_beam"
)
VIS_FREQUENCIES = np.array([1.0e9, 1.1e9, 1.2e9])
CELL = 1e-5
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
PSF_SIGMA_PIXELS = (2.0, 1.5)
PSF_PARAMS = {
    "continuum_mode": "mfs",
    "gridding_kernel_params": {"type": "dft"},
    "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
}


class _Reached(Exception):
    """Raised by a stand-in to show that a step was reached."""


@pytest.mark.parametrize("restoring_beam", [None, [1.2e-5, 0.8e-5, 0.25]])
def test_cube_task_hands_the_target_to_every_channel(
    tmp_path, monkeypatch, restoring_beam
):
    import astroviper.processing_functions.imaging as pf_imaging
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    science = _FakeScience(cycles_per_channel=[1, 1, 1])
    monkeypatch.setattr(pf_imaging, "image_cube_single_field", science)
    image_cube_single_field(
        **_task_inputs(tmp_path, restore=True, restoring_beam=restoring_beam),
        input_data=_fake_processing_set({"ms_a": VIS_FREQUENCIES}),
    )
    assert [call["kwargs"]["restoring_beam"] for call in science.calls] == [
        restoring_beam
    ] * 3


def test_cube_task_checks_a_direct_target(tmp_path):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    with pytest.raises(ValueError, match="requires restore=True"):
        image_cube_single_field(
            **_task_inputs(tmp_path, restoring_beam=[1.2e-5, 0.8e-5, 0.0]),
            input_data=_fake_processing_set({"ms_a": VIS_FREQUENCIES}),
        )


def _reduced_dft_image():
    """A reduced direct Fourier transform MFS image of one Taylor term in two
    correlations, whose PSF is a Gaussian of widths ``PSF_SIGMA_PIXELS``
    along l and m on cells of ``CELL`` radians."""
    n = 32
    offset = np.arange(n) - n // 2
    sigma_l, sigma_m = PSF_SIGMA_PIXELS
    psf = np.exp(
        -0.5 * ((offset[:, None] / sigma_l) ** 2 + (offset[None, :] / sigma_m) ** 2)
    )
    return xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.zeros((1, 1, 2, n, n)),
            ),
            "VISIBILITY_NORMALIZATION": (
                ("time", "taylor_term", "polarization"),
                np.ones((1, 1, 2)),
            ),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                np.tile(psf, (1, 1, 2, 1, 1)),
            ),
            "UV_SAMPLING_NORMALIZATION": (
                ("time", "psf_taylor_order", "polarization"),
                np.ones((1, 1, 2)),
            ),
        },
        coords={
            "time": [0.0],
            "taylor_term": [0],
            "psf_taylor_order": [0],
            "polarization": ["XX", "YY"],
            "l": -CELL * offset,
            "m": CELL * offset,
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "visibility_normalization": "VISIBILITY_NORMALIZATION",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
                }
            }
        },
    )


def _prepare(restoring_beam):
    params = dict(PSF_PARAMS)
    if restoring_beam is not None:
        params["restoring_beam"] = restoring_beam
    return continuum_node._prepare_continuum_image(
        _reduced_dft_image(),
        params,
        initialize_static_products=True,
        input_checked=True,
    )


@pytest.fixture
def stop_after_the_psf_checks(monkeypatch):
    """The primary beam is installed right after the restoring-beam check."""

    def reached(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(continuum_node, "_finalize_reference_primary_beam", reached)


def test_continuum_preparation_checks_the_fitted_beam_of_the_first_polarization(
    monkeypatch, stop_after_the_psf_checks
):
    calls = []

    def recording_correcting_beam(beam_fit_params, restoring_beam):
        calls.append((beam_fit_params, restoring_beam))

    monkeypatch.setattr(common_beam, "correcting_beam", recording_correcting_beam)
    target = [3e-4, 2e-4, 0.0]
    with pytest.raises(_Reached):
        _prepare(target)
    [(beam, restoring_beam)] = calls
    assert restoring_beam is target
    assert beam.sizes["polarization"] == 1
    assert str(beam.polarization.values[0]) == "XX"
    # The fitted PSF beam: FWHM 2 sqrt(2 ln 2) sigma along l (major) and m.
    np.testing.assert_allclose(
        beam.values.reshape(3)[:2],
        FWHM * CELL * np.array(PSF_SIGMA_PIXELS),
        rtol=1e-2,
    )


def test_continuum_preparation_without_a_target_checks_nothing(
    monkeypatch, stop_after_the_psf_checks
):
    def unexpected(*args, **kwargs):
        pytest.fail("correcting_beam called without a restoring_beam")

    monkeypatch.setattr(common_beam, "correcting_beam", unexpected)
    with pytest.raises(_Reached):
        _prepare(None)


def test_continuum_preparation_refuses_a_target_that_does_not_contain_the_psf_beam(
    monkeypatch, stop_after_the_psf_checks
):
    refusals = []
    correcting_beam = common_beam.correcting_beam

    def recording_correcting_beam(beam_fit_params, restoring_beam):
        try:
            return correcting_beam(beam_fit_params, restoring_beam)
        except ValueError as error:
            refusals.append(error)
            raise

    monkeypatch.setattr(common_beam, "correcting_beam", recording_correcting_beam)
    psf_major = FWHM * CELL * PSF_SIGMA_PIXELS[0]  # along l
    # pa = 0 puts the major axis along m: long enough there, too narrow along l.
    with pytest.raises(ValueError) as refusal:
        _prepare([2.0 * psf_major, 0.5 * psf_major, 0.0])
    assert refusals == [refusal.value]
    with pytest.raises(_Reached):
        _prepare([2.0 * psf_major, 2.0 * psf_major, 0.0])


@pytest.mark.parametrize("restoring_beam", [None, [3e-5, 2e-5, 0.25]])
def test_finalization_hands_the_target_to_the_continuum_restore(
    monkeypatch, restoring_beam
):
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": xr.DataArray(
                np.zeros((1, 1, 1, 2, 2)),
                dims=("time", "taylor_term", "polarization", "l", "m"),
            )
        },
        attrs={"data_groups": {"residual": {"sky": "SKY_RESIDUAL"}}},
    )
    model = image.rename({"SKY_RESIDUAL": "SKY_MODEL"})
    model.attrs = {"data_groups": {"model": {"sky": "SKY_MODEL"}}}
    received = {}

    def fake_restore(image, **kwargs):
        received.update(kwargs)
        return image, pd.DataFrame({"T_restore": [0.0]})

    monkeypatch.setattr(continuum_processing, "restore_continuum_image", fake_restore)
    params = {
        "prepared_continuum_image": True,
        "static_xds": xr.Dataset(),
        "model_xds": model,
        "restore": True,
    }
    if restoring_beam is not None:
        params["restoring_beam"] = restoring_beam
    continuum_node.continuum_finalize_node(
        {"image": image, "residual_converged": True}, params, input_checked=True
    )
    assert received["restoring_beam"] is restoring_beam
