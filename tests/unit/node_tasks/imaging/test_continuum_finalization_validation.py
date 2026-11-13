"""Direct finalization validates configuration without bypassing product checks."""

import importlib
from unittest.mock import Mock

import pytest
import xarray as xr

from astroviper.processing_functions.imaging import config_check
from tests.utils.continuum_images import make_continuum_image

node = importlib.import_module(
    "astroviper.node_tasks.imaging.image_continuum_single_field"
)
processing = importlib.import_module(
    "astroviper.processing_functions.imaging.image_continuum_single_field"
)


@pytest.mark.parametrize("mode", ["mfs", "mvc"])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize(
    "invalid,match",
    [
        ({"deconvolver": "bogus"}, "deconvolver"),
        ({"primary_beam_limit": 1.1}, "primary_beam_limit"),
        ({"restoration_output": "bogus"}, "restoration_output"),
        ({"prepared_continuum_image": "False"}, "prepared_continuum_image"),
    ],
)
def test_unchecked_finalization_rejects_configuration_before_runtime(
    monkeypatch, mode, prepared, restore, invalid, match
):
    image = make_continuum_image(continuum_mode=mode)
    params = {
        "continuum_mode": mode,
        "deconvolver": "mtmfs",
        "prepared_continuum_image": prepared,
        "restore": restore,
        "static_xds": xr.Dataset(),
        "model_xds": image[["SKY_MODEL"]].copy(deep=True),
        **invalid,
    }
    snapshot = image.copy(deep=True)
    model_snapshot = params["model_xds"].copy(deep=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid configuration reached runtime processing")

    monkeypatch.setattr(node, "_prepare_continuum_image", forbidden)
    monkeypatch.setattr(node, "_prepare_cached_mfs_residual_grid", forbidden)
    monkeypatch.setattr(processing, "restore_continuum_image", forbidden)
    error = TypeError if "prepared_continuum_image" in invalid else ValueError
    with pytest.raises(error, match=match):
        node.continuum_finalize_node({"image": image}, params)
    xr.testing.assert_identical(image, snapshot)
    xr.testing.assert_identical(params["model_xds"], model_snapshot)
    assert "input_checked" not in params


class RuntimeReached(Exception):
    pass


@pytest.mark.parametrize("mode", ["mfs", "mvc"])
@pytest.mark.parametrize(
    "keyword,dictionary", [(False, False), (True, False), (False, True), (True, True)]
)
def test_finalization_real_preparation_reuses_only_checked_configuration(
    monkeypatch, mode, keyword, dictionary
):
    checks = {}
    for name in (
        "check_model_update",
        "check_driver_workflow",
        "check_residual_update",
    ):
        checks[name] = Mock(wraps=getattr(config_check, name))
        monkeypatch.setattr(config_check, name, checks[name])
    params = {
        "continuum_mode": mode,
        "deconvolver": "mtmfs",
        "input_checked": dictionary,
        "image_params": {"n_taylor_terms": 2, "reference_frequency": 100.0},
        "static_xds": make_continuum_image(continuum_mode=mode),
        "model_xds": xr.Dataset(),
        "restore": False,
    }
    if mode == "mfs":
        params["gridding_kernel_params"] = {"type": "dft"}
        runtime_name = "astroviper.processing_functions.imaging.direct_fourier_transform.normalize_dft_image"
    else:
        params["static_xds"]["PRIMARY_BEAM"] = params[
            "static_xds"
        ].PRIMARY_BEAM.expand_dims(frequency=[100.0])
        params["mvc_model_update_approximation"] = "channel_psf"
        runtime_name = "astroviper.processing_functions.imaging.image_continuum_single_field.finalize_mvc_taylor_normal_equations"
    before = dict(params)
    runtime = Mock(side_effect=RuntimeReached)
    monkeypatch.setattr(runtime_name, runtime)
    with pytest.raises(RuntimeReached):
        node.continuum_finalize_node(
            {"image": xr.Dataset()}, params, input_checked=keyword
        )
    for check in checks.values():
        assert check.call_count == int(not (keyword or dictionary))
    runtime.assert_called_once()
    assert params.keys() == before.keys()
    for name in params:
        assert params[name] is before[name]


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("missing", ["static_xds", "model_xds", "SKY_MODEL"])
def test_checked_finalization_still_requires_runtime_products(dictionary, missing):
    params = {
        "prepared_continuum_image": True,
        "static_xds": xr.Dataset(),
        "model_xds": xr.Dataset(),
        "input_checked": dictionary,
    }
    if missing != "SKY_MODEL":
        params.pop(missing)
    with pytest.raises(KeyError, match=missing):
        node.continuum_finalize_node(
            {"image": xr.Dataset()}, params, input_checked=not dictionary
        )


@pytest.mark.parametrize("mode", ["mfs", "mvc"])
@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("missing", ["POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"])
def test_checked_restoration_still_checks_actual_products(mode, dictionary, missing):
    image = make_continuum_image(continuum_mode=mode)
    psf_zero = image.POINT_SPREAD_FUNCTION.isel(psf_taylor_order=0, drop=True)
    image = image.drop_dims("psf_taylor_order")
    image["POINT_SPREAD_FUNCTION"] = xr.concat(
        [psf_zero, 0 * psf_zero, 0.25 * psf_zero],
        dim=xr.IndexVariable("psf_taylor_order", [0, 1, 2]),
    ).transpose("time", "psf_taylor_order", "polarization", "l", "m")
    if missing == "PRIMARY_BEAM" and mode == "mfs":
        missing = "PRIMARY_BEAM_TAYLOR"
    image = image.drop_vars(missing)
    model = image[["SKY_MODEL"]].copy(deep=True)
    snapshot = model.copy(deep=True)
    params = {
        "continuum_mode": mode,
        "deconvolver": "mtmfs",
        "prepared_continuum_image": True,
        "restore": True,
        "restoration_output": "primary_beam_corrected",
        "static_xds": xr.Dataset(),
        "model_xds": model,
        "input_checked": dictionary,
    }
    with pytest.raises(KeyError, match=missing):
        node.continuum_finalize_node(
            {"image": image}, params, input_checked=not dictionary
        )
    xr.testing.assert_identical(model, snapshot)
