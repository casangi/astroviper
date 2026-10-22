"""Invalid and degenerate distributed continuum reducer inputs."""

import numpy as np
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _resolve_continuum_deconvolver,
    _resolve_continuum_taylor_expansion,
    combine_continuum_imaging_weight_chunks,
    combine_continuum_weight_density_chunks,
    image_continuum_single_field,
)


def test_public_api_resolves_single_term_mtmfs_to_hogbom():
    with pytest.warns(UserWarning, match="n_taylor_terms=1.*falling back.*hogbom"):
        resolved = _resolve_continuum_deconvolver("mtmfs", {"n_taylor_terms": 1})

    assert resolved == "hogbom"


@pytest.mark.parametrize("n_taylor_terms", [2, 3, 4])
def test_public_api_preserves_supported_mtmfs_taylor_orders(n_taylor_terms):
    assert (
        _resolve_continuum_deconvolver("mtmfs", {"n_taylor_terms": n_taylor_terms})
        == "mtmfs"
    )


@pytest.mark.parametrize(
    "params,error,message",
    [
        ({}, KeyError, "must contain 'n_taylor_terms'"),
        ({"n_taylor_terms": 1.0}, TypeError, "must be an integer"),
        ({"n_taylor_terms": 0}, ValueError, "between 2 and 4"),
        ({"n_taylor_terms": 5}, ValueError, "between 2 and 4"),
    ],
)
def test_public_api_rejects_invalid_mtmfs_taylor_orders(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_deconvolver("mtmfs", params)


def test_public_api_rejects_msv2_before_creating_output(monkeypatch, tmp_path):
    """Continuum users get a conversion hint without an empty image store."""
    monkeypatch.setattr(
        "astroviper.utils.measurement_set_v2.is_measurement_set_v2",
        lambda path: True,
    )
    output = tmp_path / "must_not_exist.zarr"
    with pytest.raises(ValueError, match="convert_msv2_to_processing_set"):
        image_continuum_single_field(
            "input.ms",
            str(output),
            {
                "n_taylor_terms": 1,
                "reference_frequency": 1.0e9,
                "image_size": [8, 8],
            },
            {"weighting": "natural"},
            {},
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "deconvolver", ["hogbom_many_threads", "asp", "aspclean", "asp_clean"]
)
def test_public_api_rejects_deconvolvers_the_continuum_model_update_lacks(
    deconvolver,
):
    with pytest.raises(ValueError, match="must be 'hogbom' or 'mtmfs'"):
        _resolve_continuum_deconvolver(deconvolver, {"n_taylor_terms": 2})


def test_non_mtmfs_deconvolver_does_not_require_nterms():
    assert _resolve_continuum_deconvolver("hogbom", {}) == "hogbom"


@pytest.mark.parametrize("n_taylor_terms", [2, np.int64(2)])
@pytest.mark.parametrize(
    "keys",
    [
        {"reference_frequency": 100},
        {"reference_frequency": np.float32(100.0)},
        {"reference_frequency": xr.DataArray(100.0)},
        {"reference_frequency": np.array([100.0])},
        {"reference_frequency_hz": 100.0},
        {"reference_frequency": 100.0, "reference_frequency_hz": np.float32(100.0)},
    ],
)
def test_taylor_expansion_is_resolved_to_python_scalars(n_taylor_terms, keys):
    """NumPy and xarray scalars are accepted; ``n_taylor_terms`` becomes an int and the
    deprecated alias is replaced by ``reference_frequency``, a float in Hz."""
    params = {"n_taylor_terms": n_taylor_terms, **keys}
    if "reference_frequency_hz" in params:
        with pytest.warns(FutureWarning, match="reference_frequency_hz"):
            _resolve_continuum_taylor_expansion(params)
    else:
        _resolve_continuum_taylor_expansion(params)
    assert params == {"n_taylor_terms": 2, "reference_frequency": 100.0}
    assert type(params["n_taylor_terms"]) is int
    assert type(params["reference_frequency"]) is float


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
    "params,error,message",
    [
        (
            {"n_taylor_terms": 2.0, "reference_frequency": 100.0},
            TypeError,
            "an integer",
        ),
        ({"n_taylor_terms": 2}, KeyError, "must contain 'reference_frequency'"),
        (
            {
                "n_taylor_terms": 2,
                "reference_frequency": 100.0,
                "reference_frequency_hz": 101.0,
            },
            ValueError,
            "two different reference frequencies",
        ),
        (
            {"n_taylor_terms": 2, "reference_frequency": 0.0},
            ValueError,
            "finite and positive",
        ),
        (
            {"n_taylor_terms": 2, "reference_frequency": np.nan},
            ValueError,
            "finite and positive",
        ),
        (
            {"n_taylor_terms": 2, "reference_frequency_hz": -1.0},
            ValueError,
            "finite and positive",
        ),
        (
            {"n_taylor_terms": 2, "reference_frequency": [100.0, 200.0]},
            ValueError,
            "scalar",
        ),
    ],
)
def test_taylor_expansion_rejects_invalid_values(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_taylor_expansion(params)


@pytest.mark.parametrize(
    "inputs,error,message",
    [
        ([], ValueError, "no inputs"),
        ([None], TypeError, "dictionary"),
        ([{}], KeyError, "neither"),
        ([{"task_id": 0}], KeyError, "weight_datasets"),
        (
            [
                {"task_id": 0, "weight_datasets": {}},
                {"task_id": 0, "weight_datasets": {}},
            ],
            ValueError,
            "Duplicate",
        ),
        (
            [{"weight_cache_mapping": {0: {}}}, {"weight_cache_mapping": {0: {}}}],
            ValueError,
            "Duplicate",
        ),
    ],
)
def test_weight_cache_reducer_rejects_missing_or_duplicate_task_results(
    inputs, error, message
):
    with pytest.raises(error, match=message):
        combine_continuum_imaging_weight_chunks(inputs, {})


def _density():
    return xr.Dataset(
        {
            "WEIGHT_DENSITY_GRID": (
                ("frequency", "weight_polarization", "u", "v"),
                np.ones((2, 1, 2, 2)),
            ),
            "SUM_WEIGHT": (("frequency", "weight_polarization"), np.ones((2, 1))),
        },
        coords={
            "frequency": [100.0, 110.0],
            "weight_polarization": [0],
            "u": [0, 1],
            "v": [0, 1],
        },
    )


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("empty", ValueError, "no inputs"),
        ("result_type", TypeError, "dictionary"),
        ("missing_density", KeyError, "weight_density"),
        ("dataset_type", TypeError, "Dataset"),
        ("missing_variable", KeyError, "missing"),
        ("missing_frequency", KeyError, "frequency coordinate"),
        ("empty_frequency", ValueError, "no frequency"),
        ("nan_frequency", ValueError, "non-finite"),
        ("duplicate_frequency", ValueError, "duplicate"),
        ("density_dimensions", ValueError, "dimensions"),
        ("sum_dimensions", ValueError, "dimensions"),
    ],
)
def test_density_reducer_rejects_invalid_channels_and_arrays(fault, error, message):
    density = _density()
    inputs = [{"weight_density": density}]
    if fault == "empty":
        inputs = []
    elif fault == "result_type":
        inputs = [None]
    elif fault == "missing_density":
        inputs = [{}]
    elif fault == "dataset_type":
        inputs = [{"weight_density": None}]
    else:
        if fault == "missing_variable":
            density = density.drop_vars("SUM_WEIGHT")
        elif fault == "missing_frequency":
            density = density.drop_vars("frequency")
        elif fault == "empty_frequency":
            density = density.isel(frequency=slice(0, 0))
        elif fault == "nan_frequency":
            density = density.assign_coords(frequency=[100.0, np.nan])
        elif fault == "duplicate_frequency":
            density = density.assign_coords(frequency=[100.0, 100.0])
        elif fault == "density_dimensions":
            density["WEIGHT_DENSITY_GRID"] = density.WEIGHT_DENSITY_GRID.transpose(
                "weight_polarization", "frequency", "u", "v"
            )
        else:
            density["SUM_WEIGHT"] = density.SUM_WEIGHT.transpose(
                "weight_polarization", "frequency"
            )
        inputs = [{"weight_density": density}]
    with pytest.raises(error, match=message):
        combine_continuum_weight_density_chunks(inputs, {})
