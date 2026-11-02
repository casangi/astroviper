"""Validate acceleration and monitoring policies without changing configuration."""

from types import MappingProxyType

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    check_driver_workflow,
)


@pytest.mark.parametrize(
    "acceleration", ["none", "conjugate_gradient", "nesterov_momentum"]
)
@pytest.mark.parametrize("coefficient", [0, 0.5, np.float32(0.5), np.nextafter(1, 0)])
def test_acceleration_accepts_supported_values_without_changing_config(
    acceleration, coefficient
):
    config = MappingProxyType(
        {
            "cycle_acceleration": acceleration,
            "momentum_coefficient": coefficient,
        }
    )

    check_driver_workflow(config, mode=CONTINUUM_SINGLE_FIELD)

    assert config["cycle_acceleration"] is acceleration
    assert config["momentum_coefficient"] is coefficient


@pytest.mark.parametrize("acceleration", ["NONE", "momentum", "", None, False, 1, []])
def test_acceleration_rejects_unknown_or_non_string_methods(acceleration):
    with pytest.raises(ValueError, match="cycle_acceleration"):
        check_driver_workflow(
            {"cycle_acceleration": acceleration}, mode=CONTINUUM_SINGLE_FIELD
        )


@pytest.mark.parametrize(
    "coefficient",
    [
        -0.01,
        1,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        "0.5",
        [0.5],
        None,
    ],
)
def test_momentum_rejects_invalid_coefficients_even_when_acceleration_is_none(
    coefficient,
):
    with pytest.raises((ValueError, TypeError), match="momentum_coefficient"):
        check_driver_workflow(
            {"cycle_acceleration": "none", "momentum_coefficient": coefficient},
            mode=CONTINUUM_SINGLE_FIELD,
        )


def test_workflow_does_not_insert_omitted_acceleration_defaults():
    config = {}

    check_driver_workflow(config, mode=CONTINUUM_SINGLE_FIELD)

    assert config == {}


@pytest.mark.parametrize(
    "name", ["task_time_kill_switch_seconds", "monitor_resources_seconds"]
)
@pytest.mark.parametrize("value", [None, 0, 0.5])
def test_monitoring_accepts_omitted_limit_or_non_negative_scalar(name, value):
    check_driver_workflow({name: value}, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize(
    "name", ["task_time_kill_switch_seconds", "monitor_resources_seconds"]
)
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True])
def test_monitoring_rejects_invalid_supplied_limit(name, value):
    with pytest.raises((ValueError, TypeError), match=name):
        check_driver_workflow({name: value}, mode=CONTINUUM_SINGLE_FIELD)
