"""Each deconvolver hands every argument of deconvolve to the function that runs it."""

import pytest

from astroviper.processing_functions.imaging import image_continuum_single_field
from astroviper.processing_functions.imaging.model_update import deconvolve
from astroviper.processing_functions.imaging.model_update.deconvolvers.deconvolver_registry import (
    DECONVOLVERS,
)


@pytest.mark.parametrize(
    "name, function",
    [
        ("hogbom", "hogbom_clean"),
        ("hogbom_many_threads", "hogbom_clean_many_threads"),
        ("asp", "asp_clean"),
    ],
)
@pytest.mark.parametrize("input_checked", [False, True])
def test_pixel_deconvolvers_pass_every_argument_by_keyword(
    monkeypatch, name, function, input_checked
):
    calls = []
    monkeypatch.setattr(
        deconvolve, function, lambda **kwargs: calls.append(kwargs) or "results"
    )
    arguments = {
        "residual_cube": object(),
        "psf_cube": object(),
        "model_cube": object(),
        "deconvolve_params": object(),
        "processing_function_threads": 7,
        "mask_cube": object(),
    }
    results = DECONVOLVERS[name].deconvolve(**arguments, input_checked=input_checked)
    assert results == "results"
    assert calls == [{**arguments, "input_checked": input_checked}]


@pytest.mark.parametrize("input_checked", [False, True])
def test_mtmfs_passes_every_argument_to_the_continuum_adapter(
    monkeypatch, input_checked
):
    calls = []
    monkeypatch.setattr(
        image_continuum_single_field,
        "_model_update_native_mtmfs",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "results",
    )
    arguments = tuple(object() for _ in range(8))
    results = DECONVOLVERS["mtmfs"].deconvolve(*arguments, input_checked=input_checked)
    assert results == "results"
    assert calls == [(arguments, {"input_checked": input_checked})]
