"""Runtime Hessian reuse is content verified and never skips changing products."""

import pickle
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import CONTINUUM_SINGLE_FIELD
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_model_products,
)
from astroviper.processing_functions.imaging.deconvolvers import mtmfs
from tests.utils.continuum_images import make_runaway_taylor_image


def problem():
    image = make_runaway_taylor_image()
    image.POINT_SPREAD_FUNCTION.values[:] = 0
    image.POINT_SPREAD_FUNCTION.values[0, :, 0, 16, 16] = [1, 0, 0.25]
    return image, {"deconvolver": "mtmfs", "cycle_control_params": {"scales": [0.0]}}


def check(image, config, cache=None):
    return check_model_products(
        image, config, mode=CONTINUUM_SINGLE_FIELD, hessian_cache=cache
    )


def count_hessians(monkeypatch):
    calls = []
    original = mtmfs.hessian

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mtmfs, "hessian", record)
    return calls


def test_reuse_survives_worker_transfer_and_does_not_alias_metadata(monkeypatch):
    image, config = problem()
    cache = {}
    calls = count_hessians(monkeypatch)
    expected = check(image, config, cache)
    image, cache = pickle.loads(pickle.dumps((image, cache)))
    actual = check(image, config, cache)
    assert actual == expected
    assert len(calls) == 1
    actual["taylor_hessian_spatial_blocks"].clear()
    assert check(image, config, cache) == expected
    assert len(calls) == 1
    assert "hessian_cache" not in image.attrs


@pytest.mark.parametrize(
    "change",
    [
        "psf",
        "replacement",
        "precision",
        "scales",
        "bias",
        "cell_size",
        "coordinates",
        "reference",
    ],
)
def test_changed_static_inputs_recompute_hessians(monkeypatch, change):
    image, config = problem()
    cache = {}
    calls = count_hessians(monkeypatch)
    check(image, config, cache)
    if change == "psf":
        image.POINT_SPREAD_FUNCTION.values[0, 2, 0, 16, 16] *= 2
    elif change == "replacement":
        image["POINT_SPREAD_FUNCTION"] = image.POINT_SPREAD_FUNCTION * 2
    elif change == "precision":
        image = image.astype(np.float32)
    elif change == "scales":
        config["cycle_control_params"]["scales"] = [0.0, 3.0]
    elif change == "bias":
        config["cycle_control_params"]["small_scale_bias"] = 0.2
    elif change == "cell_size":
        config["image_params"] = {"cell_size": [-1e-5, 1e-5]}
    elif change == "coordinates":
        image = image.assign_coords(time=[1.0])
    else:
        config["image_params"] = {"reference_frequency": 1e9}
        image.attrs["reference_frequency"] = 1e9
    actual = check(image, config, cache)
    assert len(calls) == 2
    assert actual == check(image, config)


@pytest.mark.parametrize("fault", ["mask", "model", "threshold", "basis", "psf"])
def test_cached_hessians_cannot_hide_invalid_runtime_products(fault):
    image, config = problem()
    cache = {}
    check(image, config, cache)
    old_cache = deepcopy(cache)
    if fault == "mask":
        image["MASK"] = xr.full_like(image.SKY_RESIDUAL, np.nan)
        image.attrs["data_groups"]["residual"]["mask"] = "MASK"
    elif fault == "model":
        image["SKY_MODEL"] = image.SKY_RESIDUAL.astype(np.float32)
    elif fault == "threshold":
        config["cycle_control_params"]["threshold_per_cycle"] = np.nan
    elif fault == "basis":
        config["image_params"] = {"n_taylor_terms": 3}
    else:
        image.POINT_SPREAD_FUNCTION.values[0, 2, 0, 16, 16] = -0.25
    with pytest.raises((ValueError, TypeError)):
        check(image, config, cache)
    assert cache == old_cache

    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        model_update_mtmfs_single_field,
    )

    snapshot = image.copy(deep=True)
    with pytest.raises((ValueError, TypeError)):
        model_update_mtmfs_single_field(
            image,
            "mtmfs",
            config["cycle_control_params"],
            input_checked=True,
            product_config=config,
            hessian_cache=cache,
        )
    xr.testing.assert_identical(image, snapshot)
    assert cache == old_cache


def test_changing_residuals_and_controls_refresh_diagnostics_without_recomputing(
    monkeypatch,
):
    image, config = problem()
    cache = {}
    calls = count_hessians(monkeypatch)
    check(image, config, cache)
    image.SKY_RESIDUAL.values *= 0.5
    config["cycle_control_params"]["gain"] = 0.2
    actual = check(image, config, cache)
    assert len(calls) == 1
    assert actual["validated_deconvolution_inputs"]["gain"] == 0.2
    assert actual == check(image, config)


def test_no_cache_always_checks_and_failed_hessians_are_not_cached(monkeypatch):
    image, config = problem()
    calls = count_hessians(monkeypatch)
    check(image, config)
    check(image, config)
    assert len(calls) == 2
    image.POINT_SPREAD_FUNCTION.values[0, 2, 0, 16, 16] = -0.25
    cache = {}
    for _ in range(2):
        with pytest.raises(ValueError, match="Taylor-Hessian"):
            check(image, config, cache)
        assert cache == {}
    assert len(calls) == 4
