"""Invalid and degenerate distributed continuum reducer inputs."""

import numpy as np
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _resolve_continuum_deconvolver,
    _validate_cycle_acceleration,
    _warn_on_disabled_momentum,
    combine_continuum_imaging_weight_chunks,
    combine_continuum_weight_density_chunks,
)


def test_public_api_resolves_single_term_mtmfs_to_hogbom():
    with pytest.warns(UserWarning, match="nterms=1.*falling back.*hogbom"):
        resolved = _resolve_continuum_deconvolver("mtmfs", {"nterms": 1})

    assert resolved == "hogbom"


@pytest.mark.parametrize("nterms", [2, 3, 4])
def test_public_api_preserves_supported_mtmfs_taylor_orders(nterms):
    assert _resolve_continuum_deconvolver("mtmfs", {"nterms": nterms}) == "mtmfs"


@pytest.mark.parametrize(
    "params,error,message",
    [
        ({}, KeyError, "must contain 'nterms'"),
        ({"nterms": 1.0}, TypeError, "must be an integer"),
        ({"nterms": 0}, ValueError, "between 2 and 4"),
        ({"nterms": 5}, ValueError, "between 2 and 4"),
    ],
)
def test_public_api_rejects_invalid_mtmfs_taylor_orders(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_deconvolver("mtmfs", params)


def test_non_mtmfs_deconvolver_does_not_require_nterms():
    assert _resolve_continuum_deconvolver("hogbom", {}) == "hogbom"


@pytest.mark.parametrize(
    "cycle_acceleration", ["none", "conjugate_gradient", "nesterov_momentum"]
)
@pytest.mark.parametrize("momentum_coefficient", [0, 0.5, 0.99])
def test_public_api_accepts_cycle_accelerations(
    cycle_acceleration, momentum_coefficient
):
    _validate_cycle_acceleration(cycle_acceleration, momentum_coefficient)


@pytest.mark.parametrize("cycle_acceleration", ["cg", "CG", None, ""])
def test_public_api_rejects_unknown_cycle_accelerations(cycle_acceleration):
    with pytest.raises(ValueError, match="cycle_acceleration must be"):
        _validate_cycle_acceleration(cycle_acceleration, 0.5)


@pytest.mark.parametrize(
    "momentum_coefficient", [-0.1, 1, 1.5, np.nan, np.inf, True, "0.5"]
)
def test_public_api_rejects_invalid_momentum_coefficients(momentum_coefficient):
    with pytest.raises(ValueError, match=r"momentum_coefficient must be .* \[0, 1\)"):
        _validate_cycle_acceleration("nesterov_momentum", momentum_coefficient)


def test_driver_warns_when_momentum_switches_off_on_further_planes(monkeypatch):
    import toolviper.utils.logger as logger

    messages = []
    monkeypatch.setattr(logger, "warning", messages.append)

    def state(disabled):
        return xr.Dataset(
            {"MOMENTUM_DISABLED": (("time", "polarization"), [disabled])},
            coords={"polarization": ["I", "Q"]},
        )

    conjugate_gradient_state = xr.Dataset(
        {"SKY_MODEL_DIRECTION": (("time", "polarization"), [[1.0, 1.0]])}
    )
    for previous, current in [
        (None, None),
        (None, conjugate_gradient_state),
        (None, state([False, False])),
        (state([False, False]), state([False, True])),
        (state([False, True]), state([False, True])),
    ]:
        _warn_on_disabled_momentum(previous, current)
    assert len(messages) == 1
    assert "[(0, 'Q')]" in messages[0]
    assert messages[0].endswith(
        "Any later model increment of these planes is added without momentum."
    )


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"cycle_acceleration": "bogus"}, AssertionError, None),
        (
            {"cycle_acceleration": "nesterov_momentum", "momentum_coefficient": 1.0},
            ValueError,
            "momentum_coefficient must be",
        ),
        (
            {"cycle_acceleration": "nesterov_momentum", "momentum_coefficient": np.nan},
            ValueError,
            "momentum_coefficient must be",
        ),
    ],
)
def test_public_api_rejects_invalid_cycle_acceleration_options(
    tmp_path, options, error, message
):
    """The schema rejects unknown values; only the driver's own validation
    rejects coefficients of 1 and NaN."""
    from astroviper.distributed_applications.imaging import (
        image_continuum_single_field,
    )

    with pytest.raises(error, match=message):
        image_continuum_single_field(
            ps_store=str(tmp_path / "input.ps.zarr"),
            image_store=str(tmp_path / "output.img.zarr"),
            image_params={},
            imaging_weights_params={},
            iteration_control_params={},
            **options,
        )
    assert not any(tmp_path.iterdir())


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
