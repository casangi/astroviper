"""PSF entry points retain validation and dispatch DFT without building a kernel."""

import importlib
from unittest.mock import Mock

import pytest
import xarray as xr

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
    kernel = importlib.import_module(
        "astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term"
    )
    check = Mock(wraps=config_check.check_residual_update)
    monkeypatch.setattr(config_check, "check_residual_update", check)
    build = Mock(side_effect=AssertionError("DFT needs no kernel table"))
    monkeypatch.setattr(kernel, "make_gridding_kernel", build)
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.utils.drop_auto_correlations", Mock()
    )
    transform = Mock()
    monkeypatch.setattr(
        module, "add_uv_sampling_grid_continuum_single_field", transform
    )
    ms_xdt = object()
    img_xds = xr.Dataset(attrs={"data_groups": {"residual": {}}})
    params = {"n_taylor_terms": 2, "reference_frequency": 1e9}
    result, _ = module.make_point_spread_function_continuum_single_field(
        {"ms": ms_xdt},
        img_xds,
        params,
        input_checked=checked,
        gridder="direct_fourier_transform",
    )
    assert result is img_xds
    assert check.call_count == (0 if checked else 1)
    assert transform.call_args.args == (ms_xdt, None, img_xds)
    assert transform.call_args.kwargs["n_psf_taylor_terms"] == 3
    assert transform.call_args.kwargs["reference_frequency"] == 1e9
    assert transform.call_args.kwargs["fft_padding"] is None
    assert transform.call_args.kwargs["gridder"] == "direct_fourier_transform"
    build.assert_not_called()
    assert params == {"n_taylor_terms": 2, "reference_frequency": 1e9}
