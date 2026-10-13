"""Boundary cases for continuum setup and MVC grid accumulation."""

import importlib

import numpy as np
import pytest
import xarray as xr
import xradio.image.image_xds  # noqa: F401

from astroviper.processing_functions.imaging.add_visibility_grid_continuum_mvc import (
    add_visibility_grid_mvc_single_field,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    prepare_model_uv_continuum_single_field,
)
from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _get_reference_frequency_hz,
    _imaging_weights_are_available,
    _validate_continuum_parameters,
)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, [1.0, 2.0], []])
def test_reference_frequency_rejects_invalid_scalars(value):
    with pytest.raises(ValueError, match="scalar|finite and positive"):
        _get_reference_frequency_hz({"reference_frequency": value}, xr.Dataset())


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, 100.0),
        ({"reference_frequency_hz": 120.0}, 120.0),
        (
            {"reference_frequency": np.array([130.0]), "reference_frequency_hz": 120.0},
            130.0,
        ),
    ],
)
def test_reference_frequency_fallback_and_precedence(params, expected):
    assert (
        _get_reference_frequency_hz(
            params, xr.Dataset(coords={"frequency": [80.0, 120.0]})
        )
        == expected
    )


@pytest.mark.parametrize("image", [xr.Dataset(), xr.Dataset(coords={"frequency": []})])
def test_reference_frequency_cannot_be_inferred_without_channels(image):
    with pytest.raises(ValueError, match="frequency"):
        _get_reference_frequency_hz({}, image)


def test_invalid_taylor_order_is_rejected_before_setup():
    with pytest.raises(ValueError, match="nterms"):
        _validate_continuum_parameters({"nterms": 0}, xr.Dataset())


@pytest.mark.parametrize("fault", ["empty", "group", "role", "variable", "none"])
def test_setup_checks_weights_in_every_child(fault):
    child = xr.Dataset(
        {"W": ("frequency", [1.0])},
        attrs={"data_groups": {"base": {"weight_imaging": "W"}}},
    )
    children = {"first": child.copy(deep=True), "second": child.copy(deep=True)}
    if fault == "empty":
        with pytest.raises(RuntimeError, match="No processing-set"):
            _imaging_weights_are_available({}, "base")
        return
    if fault == "group":
        children["second"].attrs["data_groups"] = {}
        with pytest.raises(KeyError, match="missing"):
            _imaging_weights_are_available(children, "base")
        return
    if fault == "role":
        children["second"].attrs["data_groups"]["base"] = {}
    elif fault == "variable":
        children["second"] = children["second"].drop_vars("W")
    assert _imaging_weights_are_available(children, "base") is (fault == "none")


@pytest.fixture
def mvc_grid_inputs(monkeypatch):
    ms = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline", "frequency", "polarization"),
                np.ones((1, 1, 2, 1), dtype=complex),
            ),
            "WEIGHT_IMAGING": (
                ("time", "baseline", "frequency", "polarization"),
                np.ones((1, 1, 2, 1)),
            ),
            "UVW": (("time", "baseline", "uvw_label"), np.zeros((1, 1, 3))),
        },
        coords={"frequency": [1e9, 1.1e9]},
        attrs={
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "weight_imaging": "WEIGHT_IMAGING",
                    "uvw": "UVW",
                }
            }
        },
    )
    image = xr.Dataset(
        coords={
            "time": [0.0],
            "frequency": [1e9, 1.1e9],
            "polarization": ["I"],
            "l": np.arange(4) * 1e-5,
            "m": np.arange(4) * 1e-5,
        },
        attrs={"type": "image_dataset", "data_groups": {"residual": {}}},
    )

    def kernel(grid, normalization, *args, **kwargs):
        grid += 1 + 2j
        normalization += 3

    module = importlib.import_module(
        "astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp"
    )
    monkeypatch.setattr(module, "prolate_spheroidal_grid", kernel)
    return ms, image


@pytest.mark.parametrize(
    "keyword,value,message",
    [
        ("complex_dtype", np.float64, "complex"),
        ("fft_padding", 0.9, "fft_padding"),
        ("fft_padding", np.nan, "fft_padding"),
        ("processing_function_threads", 0, "threads"),
    ],
)
def test_mvc_grid_rejects_invalid_execution_parameters(
    mvc_grid_inputs, keyword, value, message
):
    ms, image = mvc_grid_inputs
    with pytest.raises(
        TypeError if keyword == "complex_dtype" else ValueError, match=message
    ):
        add_visibility_grid_mvc_single_field(ms, np.ones(8), image, **{keyword: value})
    assert not image.data_vars


@pytest.mark.parametrize(
    "fault,message",
    [
        ("group", "data group"),
        ("role", "missing roles"),
        ("variable", "absent"),
        ("frequency", "frequency coordinate"),
        ("nonfinite", "non-finite"),
        ("weight_dims", "imaging-weight array"),
        ("visibility_shape", "identical shapes"),
    ],
)
def test_mvc_grid_rejects_malformed_measurement_sets(mvc_grid_inputs, fault, message):
    ms, image = mvc_grid_inputs
    error = KeyError
    if fault == "group":
        ms.attrs["data_groups"] = {}
    elif fault == "role":
        del ms.attrs["data_groups"]["base"]["uvw"]
    elif fault == "variable":
        ms = ms.drop_vars("UVW")
    elif fault == "frequency":
        ms = ms.drop_vars("frequency")
    elif fault == "nonfinite":
        ms = ms.assign_coords(frequency=[1e9, np.nan])
        error = ValueError
    elif fault == "weight_dims":
        ms["WEIGHT_IMAGING"] = xr.DataArray([1.0], dims="invalid")
        error = ValueError
    else:
        ms["VISIBILITY"] = xr.DataArray([1 + 0j], dims="invalid")
        error = ValueError
    with pytest.raises(error, match=message):
        add_visibility_grid_mvc_single_field(ms, np.ones(8), image)
    assert not image.data_vars


@pytest.mark.parametrize(
    "fault,message",
    [
        ("frequency", "frequency coordinate"),
        ("polarization", "polarization coordinate"),
        ("polarization_size", "different lengths"),
        ("time", "time coordinate"),
        ("time_size", "one image time"),
    ],
)
def test_mvc_grid_rejects_malformed_image_axes(mvc_grid_inputs, fault, message):
    ms, image = mvc_grid_inputs
    error = KeyError
    if fault.endswith("_size"):
        image = image.reindex({fault.removesuffix("_size"): [0, 1]})
        error = ValueError
    else:
        image = image.drop_vars(fault)
    with pytest.raises(error, match=message):
        add_visibility_grid_mvc_single_field(ms, np.ones(8), image)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_mvc_grid_reuses_existing_buffers_across_children(mvc_grid_inputs, dtype):
    ms, image = mvc_grid_inputs
    for _ in range(2):
        add_visibility_grid_mvc_single_field(ms, np.ones(8), image, complex_dtype=dtype)
        if _ == 0:
            grid = image.VISIBILITY.data
            normalization = image.VISIBILITY_NORMALIZATION.data
    assert image.VISIBILITY.data is grid
    assert image.VISIBILITY_NORMALIZATION.data is normalization
    assert grid.dtype == dtype
    np.testing.assert_array_equal(grid, 2 + 4j)
    np.testing.assert_array_equal(normalization, 6)


@pytest.mark.parametrize(
    "fault,message",
    [
        ("missing_normalization", "missing"),
        ("grid_shape", "shape"),
        ("normalization_shape", "shape"),
        ("grid_dimensions", "dimensions"),
        ("normalization_dimensions", "dimensions"),
    ],
)
def test_mvc_grid_rejects_incompatible_preallocated_buffers(
    mvc_grid_inputs, fault, message
):
    ms, image = mvc_grid_inputs
    add_visibility_grid_mvc_single_field(ms, np.ones(8), image)
    if fault == "missing_normalization":
        image = image.drop_vars("VISIBILITY_NORMALIZATION")
    elif fault == "grid_shape":
        image["VISIBILITY"] = image.VISIBILITY.isel(u=0, drop=True)
    elif fault == "normalization_shape":
        image["VISIBILITY_NORMALIZATION"] = image.VISIBILITY_NORMALIZATION.isel(
            time=0, drop=True
        )
    elif fault == "grid_dimensions":
        image["VISIBILITY"] = image.VISIBILITY.rename({"u": "other_u"})
    else:
        image["VISIBILITY_NORMALIZATION"] = image.VISIBILITY_NORMALIZATION.rename(
            {"time": "other_time"}
        )
    with pytest.raises(
        KeyError if fault == "missing_normalization" else ValueError, match=message
    ):
        add_visibility_grid_mvc_single_field(ms, np.ones(8), image)


@pytest.mark.parametrize(
    "fault,message",
    [
        ("type", "Dataset"),
        ("basis", "polarization_basis"),
        ("nterms", "nterms"),
        ("group", "data group"),
        ("role", "sky"),
        ("variable", "not present"),
        ("dimensions", "taylor_term"),
        ("terms", "Taylor terms"),
    ],
)
def test_model_fft_rejects_incompatible_models_before_transform(fault, message):
    model = xr.Dataset(
        {"SKY_MODEL": ("taylor_term", [1.0, 2.0])},
        attrs={"data_groups": {"model": {"sky": "SKY_MODEL"}}},
    )
    params = {"nterms": 2}
    basis = "linear"
    error = ValueError
    if fault == "type":
        model = None
        error = TypeError
    elif fault == "basis":
        basis = "unknown"
    elif fault == "nterms":
        params["nterms"] = 0
    elif fault == "group":
        model.attrs["data_groups"] = {}
        error = KeyError
    elif fault == "role":
        model.attrs["data_groups"]["model"] = {}
        error = KeyError
    elif fault == "variable":
        model = model.drop_vars("SKY_MODEL")
        error = KeyError
    elif fault == "dimensions":
        model = model.rename({"taylor_term": "frequency"})
    else:
        params["nterms"] = 3
    with pytest.raises(error, match=message):
        prepare_model_uv_continuum_single_field(
            model, params, instrument_polarization_basis=basis
        )
