"""CLEAN boxes use nonempty half-open bounds on the (l, m) image geometry."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_model_update,
)


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize(
    "image_size,box",
    [
        ([64, 64], (60, 64, 0, 64)),
        ([48, 96], (0, 90, 0, 40)),
        ([96, 48], (0, 40, 0, 90)),
        ([48, 96], (0, 96, 0, 48)),
        ([48, 96], (95, 96, 47, 48)),
        ([96, 48], (47, 48, 95, 96)),
        ([48, 96], None),
        ([48, 96], (-1, -1, -1, -1)),
        ([48, 96], tuple(np.int64(value) for value in (0, 96, 0, 48))),
    ],
)
def test_clean_box_accepts_edges_and_preserves_the_configuration(mode, image_size, box):
    config = {
        "image_params": {"image_size": image_size},
        "cycle_control_params": {"clean_box": box},
    }
    original = deepcopy(config)
    check_model_update(config, mode=mode)
    assert config == original


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize(
    "box,error",
    [
        ((0, 97, 0, 48), ValueError),
        ((0, 96, 0, 49), ValueError),
        ((96, 97, 0, 1), ValueError),
        ((0, 1, 48, 49), ValueError),
        ((0, 0, 0, 1), ValueError),
        ((0, 1, 0, 0), ValueError),
        ((2, 1, 0, 1), ValueError),
        ((0, 1, 2, 1), ValueError),
        ((-1, 96, 0, 48), ValueError),
        ((-2, 96, 0, 48), ValueError),
        ((0, 96.0, 0, 48), TypeError),
        ((False, 96, 0, 48), TypeError),
        ((-1.0, -1.0, -1.0, -1.0), TypeError),
        ([0, 96, 0, 48], ValueError),
    ],
)
def test_clean_box_rejects_empty_out_of_bounds_and_noninteger_boxes(mode, box, error):
    with pytest.raises(error, match="clean_box"):
        check_model_update(
            {
                "image_params": {"image_size": [48, 96]},
                "cycle_control_params": {"clean_box": box},
            },
            mode=mode,
        )
