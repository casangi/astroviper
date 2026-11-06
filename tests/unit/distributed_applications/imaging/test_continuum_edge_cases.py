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
    _warn_on_mvc_asp_significance_without_clean_mask,
    combine_continuum_imaging_weight_chunks,
    combine_continuum_weight_density_chunks,
    image_continuum_single_field,
)


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
        ({"n_taylor_terms": 0}, ValueError, "at least 1"),
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
            },
            {"weighting": "natural"},
            {},
        )
    assert not output.exists()


@pytest.mark.parametrize("deconvolver", ["hogbom_many_threads", "clark", "", "MTMFS"])
def test_public_api_rejects_deconvolvers_the_continuum_model_update_lacks(
    deconvolver,
):
    with pytest.raises(ValueError, match="must be 'hogbom', 'asp' or 'mtmfs'"):
        _resolve_continuum_deconvolver(deconvolver, {"n_taylor_terms": 1})


def test_non_mtmfs_deconvolver_does_not_require_nterms():
    assert _resolve_continuum_deconvolver("hogbom", {}) == "hogbom"


def test_public_api_resolves_asp_with_one_taylor_term():
    assert _resolve_continuum_deconvolver("asp", {"n_taylor_terms": 1}) == "asp"


@pytest.mark.parametrize(
    "params,error,message",
    [
        ({}, KeyError, "must contain 'n_taylor_terms' when deconvolver='asp'"),
        ({"n_taylor_terms": 1.0}, TypeError, "must be an integer"),
        ({"n_taylor_terms": 0}, ValueError, "at least 1"),
        (
            {"n_taylor_terms": 2},
            ValueError,
            r"requires image_params\['n_taylor_terms'\] = 1",
        ),
        (
            {"n_taylor_terms": 4},
            ValueError,
            r"requires image_params\['n_taylor_terms'\] = 1",
        ),
    ],
)
def test_public_api_refuses_asp_without_exactly_one_taylor_term(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_deconvolver("asp", params)


@pytest.mark.parametrize(
    "continuum_mode,deconvolver,cycle_control_params,clean_mask,warns",
    [
        ("mvc", "asp", {}, None, True),
        ("mvc", "asp", {"asp_scale_significance": True}, None, True),
        ("mvc", "asp", {"asp_scale_significance": False}, None, False),
        ("mvc", "asp", {}, np.ones((8, 8), dtype=bool), False),
        ("mfs", "asp", {}, None, False),
        ("mvc", "hogbom", {}, None, False),
        ("mvc", "mtmfs", {}, None, False),
    ],
)
def test_mvc_asp_significance_without_a_clean_mask_warns(
    monkeypatch, continuum_mode, deconvolver, cycle_control_params, clean_mask, warns
):
    """MVC sets the residual to 0 outside the primary-beam support, where Asp
    measures the noise of its significance test when there is no clean mask."""
    import toolviper.utils.logger

    messages = []
    monkeypatch.setattr(
        toolviper.utils.logger,
        "warning",
        lambda message, *args, **kwargs: messages.append(message),
    )
    _warn_on_mvc_asp_significance_without_clean_mask(
        continuum_mode, deconvolver, cycle_control_params, clean_mask
    )
    assert len(messages) == int(warns)
    if warns:
        assert "Give a clean_mask" in messages[0]


def test_mvc_asp_significance_with_an_auto_mask_does_not_warn(monkeypatch):
    """No warning: an automatic mask does not exist yet when this is tested."""
    import toolviper.utils.logger

    messages = []
    monkeypatch.setattr(
        toolviper.utils.logger,
        "warning",
        lambda message, *args, **kwargs: messages.append(message),
    )
    _warn_on_mvc_asp_significance_without_clean_mask("mvc", "asp", {}, None, {})
    assert messages == []


def test_public_api_refuses_multi_term_asp_before_writing(tmp_path):
    """The graph layer refuses Asp with two Taylor terms before it opens the
    input or creates the image store."""
    ps_store = tmp_path / "input.ps.zarr"
    ps_store.mkdir()
    image_store = tmp_path / "output.img.zarr"
    with pytest.raises(ValueError, match="deconvolver='asp' cleans Taylor term zero"):
        image_continuum_single_field(
            ps_store=str(ps_store),
            image_store=str(image_store),
            image_params={"n_taylor_terms": 2, "image_size": [8, 8]},
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params={"max_iter": 1},
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
    arguments = {
        "ps_store": str(ps_store),
        "image_store": str(image_store),
        "image_params": {"n_taylor_terms": 1, "image_size": [8, 8]},
        "imaging_weights_params": {"weighting": "natural"},
        "cycle_control_params": {"max_iter": 1},
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
    with pytest.raises(ValueError, match=f"{name} must be a bool; got {value!r}."):
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
@pytest.mark.parametrize(
    "keys",
    [
        {"reference_frequency": 100},
        {"reference_frequency": np.float32(100.0)},
        {"reference_frequency": xr.DataArray(100.0)},
        {"reference_frequency": np.array([100.0])},
    ],
)
def test_taylor_expansion_is_resolved_to_python_scalars(n_taylor_terms, keys):
    """NumPy and xarray scalars are accepted; ``n_taylor_terms`` becomes an int and
    ``reference_frequency`` a float in Hz."""
    params = {"n_taylor_terms": n_taylor_terms, **keys}
    _resolve_continuum_taylor_expansion(params)
    assert params == {"n_taylor_terms": 2, "reference_frequency": 100.0}
    assert type(params["n_taylor_terms"]) is int
    assert type(params["reference_frequency"]) is float


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
            "finite and positive",
        ),
        (
            {"n_taylor_terms": 2, "reference_frequency": np.nan},
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
    with pytest.raises(error, match=message):
        image_continuum_single_field(
            ps_store=str(tmp_path / "input.ps.zarr"),
            image_store=str(tmp_path / "output.img.zarr"),
            image_params={"n_taylor_terms": 1},
            imaging_weights_params={},
            cycle_control_params={},
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
