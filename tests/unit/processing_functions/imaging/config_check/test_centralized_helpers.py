"""Direct helpers share configuration predicates while retaining runtime guards."""

import importlib
from unittest.mock import Mock

import pytest

from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_max_cycles,
)


def test_kernel_resolver_delegates_without_rewriting_input(monkeypatch):
    checks = importlib.import_module(
        "astroviper.processing_functions.imaging.config_check._gridding"
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        resolve_gridding_kernel_params,
    )

    recipe = {"oversampling": 50}
    check = Mock(wraps=checks.check_gridding_kernel_parameters)
    monkeypatch.setattr(checks, "check_gridding_kernel_parameters", check)
    result = resolve_gridding_kernel_params(recipe)
    check.assert_called_once_with(recipe)
    assert result["oversampling"] == 50
    assert recipe == {"oversampling": 50}


def test_beam_resolver_uses_the_preflight_predicate(monkeypatch):
    checks = importlib.import_module(
        "astroviper.processing_functions.imaging.config_check._primary_beam"
    )
    from astroviper.processing_functions.imaging.primary_beam.make_primary_beam import (
        resolve_primary_beam_params,
    )

    params = {"telescope": "alma"}
    check = Mock(wraps=checks.check_cube_primary_beam)
    monkeypatch.setattr(checks, "check_cube_primary_beam", check)
    assert resolve_primary_beam_params(params)["beam_model"]["dish_diameter"] > 0
    check.assert_called_once_with(params)
    assert params == {"telescope": "alma"}


def test_cycle_count_legacy_policy_is_explicit():
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        validate_max_cycles,
    )

    assert validate_max_cycles(5.0) == 5
    with pytest.raises(TypeError, match="integer"):
        check_max_cycles(5.0)
    with pytest.raises(ValueError, match="integer"):
        validate_max_cycles(5.5)


@pytest.mark.parametrize("mode", ["mfs", "mvc"])
def test_direct_psf_builder_rejects_config_before_accessing_data(mode):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field"
    )
    builder = (
        module.make_point_spread_function_continuum_single_field
        if mode == "mfs"
        else module.make_point_spread_function_mvc_single_field
    )
    with pytest.raises(ValueError, match="n_taylor_terms"):
        builder(None, None, {"n_taylor_terms": 0})


def test_direct_accumulator_rejects_mode_before_accessing_products():
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        accumulate_continuum_model,
    )

    with pytest.raises(ValueError, match="continuum_mode"):
        accumulate_continuum_model(None, continuum_mode="typo")


def test_direct_restoration_rejects_cutoff_before_accessing_products():
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        restore_continuum_image,
    )

    with pytest.raises(ValueError, match="primary_beam_limit"):
        restore_continuum_image(None, primary_beam_limit=1)


def test_direct_prediction_rejects_order_before_accessing_products():
    from astroviper.processing_functions.imaging.residual_cycle_continuum_single_field import (
        make_visibility_model_continuum_single_field,
    )

    with pytest.raises(ValueError, match="n_taylor_terms"):
        make_visibility_model_continuum_single_field(None, None, None, 0, 1e9)


@pytest.mark.parametrize("reference", [0.0, float("nan"), [1e9, 2e9]])
def test_legacy_reference_helper_preserves_its_value_error_api(reference):
    from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
        _get_reference_frequency_hz,
    )

    with pytest.raises(ValueError, match="scalar, finite and positive"):
        _get_reference_frequency_hz({"reference_frequency": reference})


def test_source_units_use_the_shared_metadata_check():
    import xarray as xr

    from astroviper.processing_functions.imaging.config_check import CUBE_SINGLE_FIELD
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_input_metadata,
    )

    source = xr.Dataset(coords={"frequency": [1.0]})
    source.frequency.attrs["units"] = ["GHz"]
    with pytest.raises(ValueError, match="expressed in Hz"):
        check_input_metadata(source, {}, mode=CUBE_SINGLE_FIELD)


def test_append_validates_legacy_nested_mode_before_product_access():
    module = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    params = {
        "image_params": {
            "n_taylor_terms": 1,
            "reference_frequency": 1e9,
            "continuum_mode": "typo",
        },
    }
    with pytest.raises(ValueError, match="continuum_mode"):
        module._resolve_continuum_append_configuration(params)
