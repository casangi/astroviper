"""Continuum node boundary checks and best-effort failure diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.node_tasks.imaging.image_continuum_single_field import (
    _attach_imaging_weights_continuum,
    _extract_mvc_primary_beam_xds,
    _resolve_continuum_append_configuration,
    _unwrap_continuum_reduce_result,
    _write_task_kill_switch_log,
)


@pytest.mark.parametrize("wrapper", [None, list, tuple])
def test_append_accepts_direct_or_single_wrapped_reduction(wrapper):
    result = {"image": xr.Dataset()}
    data = result if wrapper is None else wrapper([result])
    assert _unwrap_continuum_reduce_result(data, "append") is result


@pytest.mark.parametrize(
    "data,error,message",
    [
        ([], ValueError, "exactly one"),
        ([{}, {}], ValueError, "exactly one"),
        (None, TypeError, "dictionary"),
        ({}, KeyError, "image"),
    ],
)
def test_append_rejects_invalid_reduction_envelopes(data, error, message):
    with pytest.raises(error, match=message):
        _unwrap_continuum_reduce_result(data, "append")


@pytest.mark.parametrize(
    "params,error,message",
    [
        ({}, KeyError, "image_params"),
        ({"image_params": {}}, KeyError, "nterms"),
        ({"image_params": {"nterms": 0}}, ValueError, "positive"),
        ({"image_params": {"nterms": 2}}, KeyError, "reference_frequency"),
        (
            {"image_params": {"nterms": 2, "reference_frequency": np.nan}},
            ValueError,
            "finite",
        ),
        (
            {"image_params": {"nterms": 2, "reference_frequency": -1}},
            ValueError,
            "positive",
        ),
    ],
)
def test_append_rejects_incomplete_or_invalid_configuration(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_append_configuration(params)


@pytest.mark.parametrize("single_precision", [False, True])
def test_append_supports_reference_frequency_alias_and_explicit_controls(
    single_precision,
):
    params = {
        "image_params": {"nterms": 3, "reference_frequency_hz": 100.0},
        "single_precision_image": single_precision,
        "processing_function_threads": 2,
        "fft_backend": "numpy",
        "image_data_variables_keep": ["sky_model"],
        "image_data_group_in_name": "custom",
    }
    actual = _resolve_continuum_append_configuration(params)
    assert actual["reference_frequency"] == 100.0
    assert actual["n_psf_taylor_terms"] == 5
    assert actual["complex_dtype"] == (
        np.complex64 if single_precision else np.complex128
    )
    for name in (
        "processing_function_threads",
        "fft_backend",
        "image_data_variables_keep",
        "image_data_group_in_name",
    ):
        assert actual[name] == params[name]


@pytest.mark.parametrize(
    "fault", ["group", "role", "variable", "dimension", "frequency", "none"]
)
def test_extract_task_beam_validates_channel_ownership(fault):
    image = xr.Dataset(
        {"PB": ("frequency", [0.7, 0.8])},
        coords={"frequency": [100.0, 110.0]},
        attrs={"data_groups": {"residual": {"primary_beam": "PB"}}},
    )
    coords = {"frequency": {"data": np.array([100.0, 110.0])}}
    if fault == "group":
        image.attrs["data_groups"] = {}
    elif fault == "role":
        image.attrs["data_groups"]["residual"] = {}
    elif fault == "variable":
        image = image.drop_vars("PB")
    elif fault == "dimension":
        image = image.rename({"frequency": "other"})
    elif fault == "frequency":
        coords["frequency"]["data"] = np.array([110.0, 100.0])
    if fault != "none":
        with pytest.raises(
            KeyError if fault in ("group", "role", "variable") else ValueError,
            match="beam|frequency|frequencies|data group",
        ):
            _extract_mvc_primary_beam_xds(image, coords, 7)
    else:
        actual = _extract_mvc_primary_beam_xds(image, coords, 7)
        xr.testing.assert_equal(actual.PB, image.PB)
        assert actual.attrs["task_id"] == 7
        assert actual.attrs["primary_beam_name"] == "PB"


@pytest.mark.parametrize(
    "fault", ["empty", "child", "variable", "dimensions", "shape", "none"]
)
def test_attach_cached_weights_checks_each_chunk_layout(fault):
    child = xr.Dataset(
        {"WEIGHT": ("frequency", [1.0, 1.0])},
        coords={"frequency": [100.0, 110.0]},
        attrs={"data_groups": {"base": {"weight": "WEIGHT"}}},
    )
    cached = xr.Dataset(
        {"WEIGHT_IMAGING": ("frequency", [3.0, 4.0])},
        coords={"frequency": [110.0, 100.0]},
    )
    inputs, weights = {"child": child}, {"child": cached}
    if fault == "empty":
        inputs = {}
    elif fault == "child":
        weights = {}
    elif fault == "variable":
        weights["child"] = cached.drop_vars("WEIGHT_IMAGING")
    elif fault == "dimensions":
        weights["child"] = cached.rename({"frequency": "other"})
    elif fault == "shape":
        weights["child"] = cached.isel(frequency=[0])
    if fault != "none":
        error = (
            RuntimeError
            if fault == "empty"
            else KeyError
            if fault in ("child", "variable")
            else ValueError
        )
        with pytest.raises(
            error, match="weights|WEIGHT_IMAGING|dimensions|size|datasets"
        ):
            _attach_imaging_weights_continuum(inputs, weights, "base")
    else:
        result = _attach_imaging_weights_continuum(inputs, weights, "base")
        np.testing.assert_array_equal(result["child"].frequency, [100.0, 110.0])
        np.testing.assert_array_equal(result["child"].WEIGHT_IMAGING, [3.0, 4.0])
        assert (
            result["child"].attrs["data_groups"]["base"]["weight_imaging"]
            == "WEIGHT_IMAGING"
        )


@pytest.mark.parametrize("writable", [True, False])
def test_watchdog_diagnostic_never_masks_original_failure(tmp_path, writable):
    output = (
        tmp_path / "image.zarr" if writable else tmp_path / "missing" / "image.zarr"
    )
    result = _write_task_kill_switch_log(
        pd.DataFrame({"T_grid": [2.0]}), 3.0, 1.0, str(output), 7, "worker"
    )
    if writable:
        text = Path(result).read_text()
        assert "task_id: 7" in text
        assert "T_grid" in text
        assert "threshold: 1.0" in text
    else:
        assert result.startswith("(failed to write kill-switch log:")
