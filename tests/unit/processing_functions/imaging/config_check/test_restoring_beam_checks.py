"""The restoring beam is an explicit beam in radians, used only by a restore;
the drivers refuse anything else before reading their input."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_config,
    check_driver_workflow,
)
from astroviper.processing_functions.imaging.config_check._output import (
    check_kept_variables,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_restoring_beam,
)
from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
    configuration,
)

ARCSEC = np.pi / (180 * 3600)
BEAM = [3.0 * ARCSEC, 2.0 * ARCSEC, 0.5]
MODES = [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD]
DRIVERS = {"cube": image_cube_single_field, "continuum": image_continuum_single_field}


@pytest.mark.parametrize(
    "value",
    [
        BEAM,
        tuple(BEAM),
        np.array(BEAM),
        np.array(BEAM, dtype=np.float32),
        [2.0 * ARCSEC, 2.0 * ARCSEC, -1.5],
    ],
)
def test_explicit_beams_are_accepted_unchanged(value):
    before = deepcopy(value)
    check_restoring_beam(value)
    np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("value", ["common", "", "1e-5, 1e-5, 0"])
def test_a_string_points_to_common_beam(value):
    with pytest.raises(ValueError, match="common_beam"):
        check_restoring_beam(value)


@pytest.mark.parametrize(
    "value",
    [
        BEAM[:2],
        BEAM + [0.0],
        [BEAM],
        3.0 * ARCSEC,
    ],
)
def test_three_values_are_required(value):
    with pytest.raises(ValueError, match=r"\[major, minor, pa\]"):
        check_restoring_beam(value)


@pytest.mark.parametrize(
    "value",
    [
        [np.nan, BEAM[1], BEAM[2]],
        [BEAM[0], BEAM[1], np.inf],
        [],
    ],
)
def test_values_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        check_restoring_beam(value)


@pytest.mark.parametrize(
    "value",
    [
        [BEAM[1], BEAM[0], 0.0],
        [BEAM[0], 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-BEAM[1], -BEAM[0], 0.0],
    ],
)
def test_axes_must_be_positive_with_major_not_below_minor(value):
    with pytest.raises(ValueError, match="major >= minor > 0"):
        check_restoring_beam(value)


@pytest.mark.parametrize("pa", [45.0, -4.0, np.nextafter(np.pi, 4.0)])
def test_the_position_angle_must_be_in_radians(pa):
    """tclean's ['1.5arcsec', '1.2arcsec', '45deg'] with only the axes in radians."""
    for limit in (np.pi, -np.pi):
        check_restoring_beam([BEAM[0], BEAM[1], limit])
    with pytest.raises(ValueError, match=r"not in \[-pi, pi\]"):
        check_restoring_beam([BEAM[0], BEAM[1], pa])


@pytest.mark.parametrize(
    "value", [[True, True, False], ["1", "1", "0"], {"major": 1.0}]
)
def test_values_must_be_real_numbers(value):
    with pytest.raises(TypeError, match="restoring_beam"):
        check_restoring_beam(value)


@pytest.mark.parametrize("mode", MODES)
def test_workflow_check_runs_the_predicate(mode):
    with pytest.raises(ValueError, match="common_beam"):
        check_driver_workflow({"restore": True, "restoring_beam": "common"}, mode=mode)


@pytest.mark.parametrize("mode", MODES)
def test_a_target_requires_restore(mode):
    check_driver_workflow({"restore": False, "restoring_beam": None}, mode=mode)
    check_driver_workflow({"restore": True, "restoring_beam": BEAM}, mode=mode)
    with pytest.raises(ValueError, match="requires restore=True"):
        check_driver_workflow({"restore": False, "restoring_beam": BEAM}, mode=mode)


@pytest.mark.parametrize("mode", MODES)
def test_the_major_axis_must_be_smaller_than_the_image(mode):
    # 40 x 32 pixels of 1 by 2 arcsec: the smaller extent is 32 arcsec.
    image_params = {"image_size": [40, 32], "cell_size": [-2 * ARCSEC, ARCSEC]}
    below = [np.nextafter(32 * ARCSEC, 0.0), ARCSEC, 0.0]
    config = {"restore": True, "restoring_beam": below, "image_params": image_params}
    before = deepcopy(config)
    check_driver_workflow(config, mode=mode)
    assert config == before
    for major in (32 * ARCSEC, 3.0, np.deg2rad(1.0)):
        with pytest.raises(ValueError, match="not smaller than the image"):
            check_driver_workflow(
                {
                    "restore": True,
                    "restoring_beam": [major, ARCSEC, 0.0],
                    "image_params": image_params,
                },
                mode=mode,
            )


@pytest.mark.parametrize("mode", MODES)
def test_a_complete_configuration_with_a_target_is_read_only(mode):
    config = configuration()
    config.update(restore=True, restoring_beam=np.array(BEAM))
    if mode is CUBE_SINGLE_FIELD:
        config["deconvolver"] = "hogbom"
        config["image_params"]["n_taylor_terms"] = 1
    snapshot = deepcopy(config)
    check_config(config, mode=mode, require_complete=True)
    np.testing.assert_array_equal(config.pop("restoring_beam"), BEAM)
    snapshot.pop("restoring_beam")
    assert config == snapshot


@pytest.mark.parametrize("mode", MODES)
def test_the_restored_beam_is_kept_only_with_a_restore_to_a_target(mode):
    kept = ["beam_fit_params_sky_restored"]
    check_driver_workflow(
        {"image_data_variables_keep": kept, "restore": True, "restoring_beam": BEAM},
        mode=mode,
    )
    for config in (
        {"restore": False, "restoring_beam": None},
        {"restore": True, "restoring_beam": None},
    ):
        with pytest.raises(ValueError, match="requires restore=True and a restoring"):
            check_driver_workflow(
                {"image_data_variables_keep": kept, **config}, mode=mode
            )


def test_a_direct_writer_call_keeps_the_restored_beam():
    check_kept_variables(
        {"image_data_variables_keep": ["beam_fit_params_sky_restored"]},
        fits=True,
        require_image=False,
    )


def _configuration(kind):
    config = configuration()
    if kind == "cube":
        config["deconvolver"] = "hogbom"
        config["image_params"]["n_taylor_terms"] = 1
    return config


@pytest.fixture
def unopened_input(monkeypatch):
    import xradio.measurement_set

    def unexpected_open(*args, **kwargs):
        pytest.fail("Input opened before the restoring_beam check")

    monkeypatch.setattr(xradio.measurement_set, "open_processing_set", unexpected_open)


@pytest.mark.parametrize("kind", ["cube", "continuum"])
def test_the_schema_refuses_a_string(unopened_input, tmp_path, kind):
    output = tmp_path / "refused.img.zarr"
    with pytest.raises(AssertionError):  # the param.json schema
        DRIVERS[kind](
            ps_store="unopened.ps.zarr",
            image_store=str(output),
            restore=True,
            restoring_beam="common",
            **_configuration(kind),
        )
    assert not output.exists()


@pytest.mark.parametrize("kind", ["cube", "continuum"])
@pytest.mark.parametrize(
    "restore,beam,match",
    [
        (False, [3 * ARCSEC, 2 * ARCSEC, 0.0], "requires restore=True"),
        (True, [3.0, 2.0, 0.0], "not smaller than the image"),
        (True, [2 * ARCSEC, 3 * ARCSEC, 0.0], "major >= minor > 0"),
        (True, [3 * ARCSEC, 2 * ARCSEC], r"\[major, minor, pa\]"),
    ],
)
def test_the_drivers_refuse_before_reading_input(
    unopened_input, tmp_path, kind, restore, beam, match
):
    output = tmp_path / "refused.img.zarr"
    with pytest.raises(ValueError, match=match):
        DRIVERS[kind](
            ps_store="unopened.ps.zarr",
            image_store=str(output),
            restore=restore,
            restoring_beam=beam,
            **_configuration(kind),
        )
    assert not output.exists()
