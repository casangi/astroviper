"""The shared workflow checker owns the MVC Asp significance diagnostic."""

import importlib
from unittest.mock import Mock

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_driver_workflow,
)


@pytest.mark.parametrize(
    "continuum_mode,deconvolver,controls,mask,warns",
    [
        ("mvc", "asp", {}, None, True),
        ("mvc", "asp", {"asp_scale_significance": True}, None, True),
        ("mvc", "asp", {"asp_scale_significance": np.True_}, None, True),
        ("mvc", "asp", {"asp_scale_significance": False}, None, False),
        ("mvc", "asp", {}, np.ones((8, 8), dtype=bool), False),
        ("mvc", "asp", {}, "mask.npy", False),
        ("mfs", "asp", {}, None, False),
        ("mvc", "hogbom", {}, None, False),
        ("mvc", "mtmfs", {}, None, False),
    ],
)
@pytest.mark.parametrize("controls_key", ["cycle_control_params", "deconvolve_params"])
def test_workflow_warns_for_the_mvc_asp_combination_only(
    monkeypatch, continuum_mode, deconvolver, controls, mask, warns, controls_key
):
    warning = Mock()
    monkeypatch.setattr("toolviper.utils.logger.warning", warning)
    config = {
        "continuum_mode": continuum_mode,
        "deconvolver": deconvolver,
        controls_key: controls,
        "clean_mask": mask,
    }
    check_driver_workflow(config, mode=CONTINUUM_SINGLE_FIELD)
    assert warning.call_count == int(warns)
    assert config[controls_key] is controls
    assert config["clean_mask"] is mask
    if warns:
        assert "Give a clean_mask" in warning.call_args.args[0]


@pytest.mark.parametrize("value", ["false", 1, None])
def test_standalone_workflow_checker_validates_flag_before_warning(monkeypatch, value):
    warning = Mock()
    monkeypatch.setattr("toolviper.utils.logger.warning", warning)
    with pytest.raises(TypeError, match="asp_scale_significance must be a bool"):
        check_driver_workflow(
            {
                "continuum_mode": "mvc",
                "deconvolver": "asp",
                "cycle_control_params": {"asp_scale_significance": value},
            },
            mode=CONTINUUM_SINGLE_FIELD,
        )
    warning.assert_not_called()


def test_cube_asp_does_not_receive_an_mvc_warning(monkeypatch):
    warning = Mock()
    monkeypatch.setattr("toolviper.utils.logger.warning", warning)
    check_driver_workflow({"deconvolver": "asp"}, mode=CUBE_SINGLE_FIELD)
    warning.assert_not_called()


def test_workflow_reads_mode_from_image_configuration(monkeypatch):
    warning = Mock()
    monkeypatch.setattr("toolviper.utils.logger.warning", warning)
    check_driver_workflow(
        {"deconvolver": "asp", "image_params": {"continuum_mode": "mvc"}},
        mode=CONTINUUM_SINGLE_FIELD,
    )
    warning.assert_called_once()


def test_public_driver_warns_once_before_opening_input(monkeypatch, tmp_path):
    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.image_continuum_single_field"
    )
    warning = Mock()
    monkeypatch.setattr("toolviper.utils.logger.warning", warning)
    open_input = Mock(side_effect=RuntimeError("input opening reached"))
    monkeypatch.setattr("xradio.measurement_set.open_processing_set", open_input)
    image_params = {
        "n_taylor_terms": 1,
        "reference_frequency": 1e9,
        "image_size": [8, 8],
        "cell_size": [-1e-5, 1e-5],
        "phase_direction": [0.0, 0.0],
        "frequency_coords": [1e9],
        "time_coords": [0.0],
        "polarization_coords": ["I", "Q"],
        "fft_padding": 1.2,
    }
    controls = {
        "max_iter": 1,
        "max_cycles": 1,
        "max_iter_per_cycle": 1,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.0,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    output = tmp_path / "image.zarr"
    with pytest.raises(RuntimeError, match="input opening reached"):
        driver.image_continuum_single_field(
            str(tmp_path / "input.ps"),
            str(output),
            image_params,
            {"weighting": "natural"},
            controls,
            continuum_mode="mvc",
            deconvolver="asp",
        )
    open_input.assert_called_once()
    warning.assert_called_once()
    assert not output.exists()


@pytest.mark.parametrize("checked", [False, True])
def test_direct_append_warns_only_when_configuration_is_unchecked(monkeypatch, checked):
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    warning = Mock()
    monkeypatch.setattr("toolviper.utils.logger.warning", warning)
    monkeypatch.setattr(
        node,
        "_unwrap_continuum_reduce_result",
        Mock(side_effect=RuntimeError("runtime products reached")),
    )
    params = {
        "continuum_mode": "mvc",
        "deconvolver": "asp",
        "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
    }
    with pytest.raises(RuntimeError, match="runtime products reached"):
        node.continuum_cycle_append_node(None, params, input_checked=checked)
    assert warning.call_count == (0 if checked else 1)
