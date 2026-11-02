"""PSF entry points retain validation and dispatch DFT without building a kernel."""

import importlib
from unittest.mock import Mock

import pytest

from astroviper.processing_functions.imaging import config_check


@pytest.mark.parametrize("kind", ["continuum", "mvc"])
def test_psf_rejects_invalid_configuration_before_reading_data(kind):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field"
    )
    function = getattr(module, f"make_point_spread_function_{kind}_single_field")
    with pytest.raises(ValueError, match="frequency_coords"):
        function(None, None, {"frequency_coords": [0.0]})


@pytest.mark.parametrize("checked", [False, True])
def test_mfs_dft_dispatch_preserves_configuration_and_skips_kernel(
    monkeypatch, checked
):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field"
    )
    dft = importlib.import_module(
        "astroviper.processing_functions.imaging.direct_fourier_transform"
    )
    kernel = importlib.import_module(
        "astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel"
    )
    check = Mock(wraps=config_check.check_residual_update)
    monkeypatch.setattr(config_check, "check_residual_update", check)
    build = Mock(side_effect=AssertionError("DFT needs no kernel table"))
    monkeypatch.setattr(kernel, "make_gridding_kernel", build)
    result = object()
    transform = Mock(return_value=result)
    monkeypatch.setattr(
        dft, "make_dft_point_spread_function_continuum_single_field", transform
    )
    params = {"n_taylor_terms": 2, "reference_frequency": 1e9}
    recipe = {"type": "dft"}
    assert (
        module.make_point_spread_function_continuum_single_field(
            None, None, params, gridding_kernel_params=recipe, input_checked=checked
        )
        is result
    )
    assert check.call_count == (0 if checked else 1)
    assert transform.call_args.args == (None, None, 3, 1e9)
    build.assert_not_called()
    assert params == {"n_taylor_terms": 2, "reference_frequency": 1e9}
    assert recipe == {"type": "dft"}
