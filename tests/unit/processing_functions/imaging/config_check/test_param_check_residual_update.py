"""Residual-update configuration, geometry, and weighting contracts."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_residual_update,
)


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
@pytest.mark.parametrize(
    "options,message",
    [
        ({"fft_backend": "missing"}, "fft_backend"),
        ({"gridder": "mosaic"}, "gridder"),
        ({"complex_dtype": "float64"}, "complex_dtype"),
        ({"image_params": {"image_size": [1, 8]}}, "at least 2"),
    ],
)
def test_shared_residual_options_fail_before_gridding(mode, options, message):
    with pytest.raises((ValueError, TypeError), match=message):
        check_residual_update(options, mode=mode)


def test_missing_optional_fft_backend_fails_early(monkeypatch):
    import importlib

    original = importlib.import_module

    def unavailable(name, *args, **kwargs):
        if name == "pyfftw.interfaces.scipy_fft":
            raise ImportError("pyfftw unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="pyfftw"):
        check_residual_update({"fft_backend": "pyfftw"}, mode=CUBE_SINGLE_FIELD)


def test_mvc_time_restriction_is_mode_specific():
    config = {"continuum_mode": "mvc", "image_params": {"time_coords": [0.0, 1.0]}}
    with pytest.raises(ValueError, match="one image time plane"):
        check_residual_update(config, mode=CONTINUUM_SINGLE_FIELD)
    check_residual_update(config, mode=CUBE_SINGLE_FIELD)
    config["continuum_mode"] = "mfs"
    check_residual_update(config, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize(
    "image,message",
    [
        ({"frequency_coords": [[1e9, 1.1e9]]}, "frequency_coords"),
        ({"nterms": 2}, "nterms"),
        ({"cell_size": [0.0, 1e-5]}, "cell_size"),
        ({"cell_size": [1e-5]}, "cell_size"),
        ({"phase_direction": [0.0, np.pi]}, "phase_direction"),
        ({"phase_direction": [0.0]}, "phase_direction"),
        ({"fft_padding": 11.0}, "fft_padding"),
        ({"fft_padding": 0.9}, "fft_padding"),
    ],
)
def test_invalid_geometry_rejected_without_changing_supplied_values(image, message):
    original = deepcopy(image)
    with pytest.raises(ValueError, match=message):
        check_residual_update(
            {"image_params": image},
            mode=CONTINUUM_SINGLE_FIELD if "nterms" in image else CUBE_SINGLE_FIELD,
        )
    assert image == original


def test_complete_cube_rejects_multiple_image_times():
    from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
        configuration,
    )

    config = configuration()
    config["image_params"].update(n_taylor_terms=1, time_coords=[0.0, 1.0])
    config["deconvolver"] = "hogbom"
    with pytest.raises(ValueError, match="one image time plane"):
        check_residual_update(config, mode=CUBE_SINGLE_FIELD, require_complete=True)


def test_unknown_weight_parameter_does_not_select_a_default():
    params = {"weighting": "natural", "robusst": 0.5}
    with pytest.raises(ValueError, match="Unknown imaging-weight parameters.*robusst"):
        check_residual_update(
            {"imaging_weights_params": params}, mode=CUBE_SINGLE_FIELD
        )
    assert params == {"weighting": "natural", "robusst": 0.5}


def test_selected_sample_list_requires_at_least_one_source_channel():
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_selected_samples,
    )

    with pytest.raises(ValueError, match="no source channels"):
        check_selected_samples([])
    check_selected_samples([("ms", 0, 1e9)])


def test_delphi_rejects_unsupported_weighting_before_deriving_parameters():
    from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
        check_delphi_parameters,
    )

    with pytest.raises(ValueError, match="Delphi weighting"):
        check_delphi_parameters({"weighting": "briggs_abs"})
