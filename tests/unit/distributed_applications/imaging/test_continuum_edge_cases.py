"""Invalid and degenerate distributed continuum reducer inputs."""

import numpy as np
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _refuse_unsupported_continuum_cycle_options,
    _resolve_continuum_deconvolver,
    _resolve_continuum_taylor_expansion,
    _validate_cycle_acceleration,
    _warn_on_disabled_momentum,
    combine_continuum_imaging_weight_chunks,
    combine_continuum_weight_density_chunks,
    image_continuum_single_field,
)


def _image_params(n_taylor_terms=1):
    """A valid image configuration for tests targeting a different parameter."""
    return {
        "n_taylor_terms": n_taylor_terms,
        "reference_frequency": 1e9,
        "image_size": [8, 8],
        "cell_size": [-1e-5, 1e-5],
        "phase_direction": [0.0, 0.0],
        "frequency_coords": [1e9],
        "time_coords": [0.0],
        "polarization_coords": ["I", "Q"],
        "fft_padding": 1.2,
    }


def _cycle_controls():
    """Valid controls so an unrelated missing key cannot mask the target error."""
    return {
        "max_iter": 1,
        "max_cycles": 1,
        "max_iter_per_cycle": 1,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.0,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }


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
        ({"n_taylor_terms": True}, TypeError, "must be an integer"),
        ({"n_taylor_terms": "2"}, TypeError, "must be an integer"),
        ({"n_taylor_terms": 0}, ValueError, "must be positive"),
        ({"n_taylor_terms": 1}, ValueError, "between 2 and 4"),
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
                "cell_size": [1.0e-5, 1.0e-5],
                "phase_direction": [0.0, 0.0],
                "frequency_coords": [1.0e9],
                "time_coords": [0.0],
                "polarization_coords": ["I", "Q"],
                "fft_padding": 1.2,
            },
            {"weighting": "natural"},
            dict(
                max_iter=0,
                max_cycles=0,
                threshold=0.0,
                gain=0.1,
                max_iter_per_cycle=1,
                psf_sidelobe_factor=1.0,
                min_psf_fraction=0.05,
                max_psf_fraction=0.8,
            ),
        )
    assert not output.exists()


@pytest.mark.parametrize("deconvolver", ["hogbom_many_threads", "clark", "", "MTMFS"])
def test_public_api_rejects_deconvolvers_the_continuum_model_update_lacks(
    deconvolver,
):
    with pytest.raises(ValueError, match="deconvolver must be one of"):
        _resolve_continuum_deconvolver(deconvolver, {"n_taylor_terms": 2})


def test_non_mtmfs_deconvolver_does_not_require_nterms():
    assert _resolve_continuum_deconvolver("hogbom", {}) == "hogbom"


def test_public_api_resolves_asp_with_one_taylor_term():
    assert _resolve_continuum_deconvolver("asp", {"n_taylor_terms": 1}) == "asp"


@pytest.mark.parametrize(
    "params,error,message",
    [
        ({}, KeyError, "must contain 'n_taylor_terms' when deconvolver='asp'"),
        ({"n_taylor_terms": 1.0}, TypeError, "must be an integer"),
        ({"n_taylor_terms": 0}, ValueError, "must be positive"),
        (
            {"n_taylor_terms": 2},
            ValueError,
            "requires n_taylor_terms=1",
        ),
        (
            {"n_taylor_terms": 4},
            ValueError,
            "requires n_taylor_terms=1",
        ),
    ],
)
def test_public_api_refuses_asp_without_exactly_one_taylor_term(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_deconvolver("asp", params)


def test_public_api_refuses_multi_term_asp_before_writing(tmp_path):
    """The graph layer refuses Asp with two Taylor terms before it opens the
    input or creates the image store."""
    ps_store = tmp_path / "input.ps.zarr"
    ps_store.mkdir()
    image_store = tmp_path / "output.img.zarr"
    with pytest.raises(ValueError, match="Continuum asp requires n_taylor_terms=1"):
        image_continuum_single_field(
            ps_store=str(ps_store),
            image_store=str(image_store),
            image_params=_image_params(2),
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=_cycle_controls(),
            deconvolver="asp",
        )
    assert not image_store.exists()
    assert list(ps_store.iterdir()) == []


def _assert_refused_before_writing(tmp_path, match, **kwargs):
    """Call the public continuum API on an empty input directory and check that
    it raises ValueError before the input is opened or the image store is
    created."""
    ps_store = tmp_path / "input.ps.zarr"
    ps_store.mkdir()
    image_store = tmp_path / "output.img.zarr"
    image_params = {**_image_params(), **kwargs.pop("image_params", {})}
    cycle_controls = {**_cycle_controls(), **kwargs.pop("cycle_control_params", {})}
    arguments = {
        "ps_store": str(ps_store),
        "image_store": str(image_store),
        "image_params": image_params,
        "imaging_weights_params": {"weighting": "natural"},
        "cycle_control_params": cycle_controls,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        image_continuum_single_field(**arguments)
    assert not image_store.exists()
    assert list(ps_store.iterdir()) == []


def test_a_clean_mask_with_auto_mask_params_is_refused_before_writing(tmp_path):
    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, np.ones((8, 8), dtype=bool))
    _assert_refused_before_writing(
        tmp_path,
        "exclude each other",
        image_params={
            "n_taylor_terms": 1,
            "image_size": [8, 8],
            "reference_frequency": 1.0e9,
        },
        clean_mask=str(mask_file),
        auto_mask_params={},
    )


def test_an_unknown_auto_mask_parameter_is_refused_before_writing(tmp_path):
    _assert_refused_before_writing(
        tmp_path,
        "Unknown auto_mask_params keys",
        image_params={
            "n_taylor_terms": 1,
            "image_size": [8, 8],
            "reference_frequency": 1.0e9,
        },
        auto_mask_params={"noise": 5.0},
    )


@pytest.mark.parametrize("deconvolver", ["hogbom", "asp", "mtmfs"])
@pytest.mark.parametrize(
    "options",
    [{}, {"record_iteration_history": False, "exact_psf_subtraction": False}],
    ids=["unset", "false"],
)
def test_continuum_accepts_the_cycle_options_off(deconvolver, options):
    _refuse_unsupported_continuum_cycle_options(deconvolver, options)


@pytest.mark.parametrize("deconvolver", ["hogbom", "asp", "mtmfs"])
@pytest.mark.parametrize("value", [True, np.True_])
def test_continuum_deconvolvers_may_record_the_iteration_history(deconvolver, value):
    _refuse_unsupported_continuum_cycle_options(
        deconvolver, {"record_iteration_history": value}
    )


@pytest.mark.parametrize("deconvolver", ["hogbom", "asp", "mtmfs"])
def test_continuum_refuses_exact_psf_subtraction(deconvolver):
    with pytest.raises(
        ValueError,
        match="exact_psf_subtraction is not supported by continuum imaging "
        f"\\(deconvolver='{deconvolver}'\\): the continuum graph makes no PSF",
    ):
        _refuse_unsupported_continuum_cycle_options(
            deconvolver, {"exact_psf_subtraction": True}
        )


@pytest.mark.parametrize("name", ["record_iteration_history", "exact_psf_subtraction"])
@pytest.mark.parametrize("value", [1, "yes", None])
def test_continuum_refuses_a_cycle_option_that_is_not_a_bool(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a bool"):
        _refuse_unsupported_continuum_cycle_options("hogbom", {name: value})


@pytest.mark.parametrize(
    "deconvolver,n_taylor_terms,option,message",
    [
        ("asp", 1, "exact_psf_subtraction", "not supported by continuum imaging"),
        ("mtmfs", 2, "exact_psf_subtraction", "not supported by continuum imaging"),
        ("hogbom", 1, "exact_psf_subtraction", "not supported by continuum imaging"),
    ],
)
def test_public_api_refuses_unsupported_cycle_options_before_writing(
    tmp_path, deconvolver, n_taylor_terms, option, message
):
    """Refused at the graph layer, not after the first residual update."""
    _assert_refused_before_writing(
        tmp_path,
        message,
        deconvolver=deconvolver,
        image_params={"n_taylor_terms": n_taylor_terms, "image_size": [8, 8]},
        cycle_control_params={"max_iter": 1, option: True},
    )


@pytest.mark.parametrize("n_taylor_terms", [2, np.int64(2)])
@pytest.mark.parametrize("reference", [100, np.float32(100.0), xr.DataArray(100.0)])
def test_taylor_expansion_preserves_supplied_scalars(n_taylor_terms, reference):
    """Validation preserves caller objects rather than rewriting scalar types."""
    params = {"n_taylor_terms": n_taylor_terms, "reference_frequency": reference}
    _resolve_continuum_taylor_expansion(params)
    assert params["n_taylor_terms"] is n_taylor_terms
    assert params["reference_frequency"] is reference


@pytest.mark.parametrize(
    "params,error,message",
    [
        (
            {"n_taylor_terms": 2.0, "reference_frequency": 100.0},
            TypeError,
            "an integer",
        ),
        (
            {"n_taylor_terms": True, "reference_frequency": 100.0},
            TypeError,
            "an integer",
        ),
        ({"n_taylor_terms": 2}, KeyError, "must contain 'reference_frequency'"),
        (
            {"n_taylor_terms": 2, "reference_frequency": 0.0},
            ValueError,
            "positive",
        ),
        (
            {"n_taylor_terms": 2, "reference_frequency": np.nan},
            ValueError,
            "finite",
        ),
        (
            {"n_taylor_terms": 2, "reference_frequency": [100.0, 200.0]},
            TypeError,
            "scalar",
        ),
    ],
)
def test_taylor_expansion_rejects_invalid_values(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_taylor_expansion(params)


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
    with pytest.raises(ValueError, match="momentum_coefficient"):
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
            "momentum_coefficient",
        ),
        (
            {"cycle_acceleration": "nesterov_momentum", "momentum_coefficient": np.nan},
            ValueError,
            "momentum_coefficient",
        ),
    ],
)
def test_public_api_rejects_invalid_cycle_acceleration_options(
    tmp_path, options, error, message
):
    """The schema rejects unknown values; only the driver's own validation
    rejects coefficients of 1 and NaN."""
    with pytest.raises(error, match=message):
        image_continuum_single_field(
            ps_store=str(tmp_path / "input.ps.zarr"),
            image_store=str(tmp_path / "output.img.zarr"),
            image_params=_image_params(),
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=_cycle_controls(),
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
