"""Gaussian fitting validates supplied sampling before accessing image pixels."""

import pytest
import xarray as xr


def test_checked_fitter_still_requires_runtime_data_group(monkeypatch):
    import astroviper.processing_functions.imaging.config_check as checks
    from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
        point_spread_function_gaussian_fit,
    )

    def unexpected_check(*args, **kwargs):
        raise AssertionError("Repeated the shared configuration check")

    monkeypatch.setattr(checks, "check_model_update", unexpected_check)

    with pytest.raises(AssertionError, match="Data group image not found"):
        point_spread_function_gaussian_fit(xr.Dataset(), input_checked=True)


@pytest.mark.parametrize("sampling", [(1, 3), (2.5, 3), (True, True), (3,), "3"])
def test_psf_fit_core_rejects_sampling_before_accessing_image(sampling):
    from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
        psf_gaussian_fit_core,
    )

    with pytest.raises((TypeError, ValueError), match="sampling"):
        psf_gaussian_fit_core(None, None, None, sampling, 0.1, None)
