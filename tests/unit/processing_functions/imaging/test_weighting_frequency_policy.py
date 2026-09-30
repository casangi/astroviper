"""Frequency assignment and UV-cell selection remain independent policies."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import xradio.image.image_xds  # noqa: F401 -- register the image accessor

from astroviper.processing_functions.imaging.calculate_imaging_weights import (
    calculate_imaging_weights,
)


def _inputs(frequencies):
    dims = ("time", "baseline", "frequency", "polarization")
    ms = xr.Dataset(
        {
            "WEIGHT": (dims, np.ones((1, 1, 2, 2))),
            "FLAG": (dims, np.zeros((1, 1, 2, 2), dtype=np.int8)),
            "UVW": (("time", "baseline", "uvw_label"), np.zeros((1, 1, 3))),
        },
        coords={"frequency": frequencies},
        attrs={
            "data_groups": {"base": {"weight": "WEIGHT", "flag": "FLAG", "uvw": "UVW"}}
        },
    )
    ps = xr.DataTree.from_dict({"ms": ms})
    image = xr.Dataset(
        coords={
            "frequency": [1.0e9, 1.1e9, 1.2e9],
            "l": np.arange(8) * 1e-5,
            "m": np.arange(8) * 1e-5,
        },
        attrs={"type": "image_dataset"},
    )
    return ps, image


@pytest.mark.parametrize("matching", ["nearest", "exact"])
@pytest.mark.parametrize("truncate", [False, True])
def test_weighting_preserves_reordered_subset_and_uv_policy(matching, truncate):
    ps, image = _inputs([1.2e9, 1.0e9])
    module = "astroviper.processing_functions.imaging.imaging_weighting"
    with (
        patch(module + ".grid_imaging_weights.grid_imaging_weights") as grid,
        patch(
            module + ".grid_imaging_weights.degrid_imaging_weights",
            return_value=np.ones((1, 1, 2, 1)),
        ) as degrid,
        patch(
            module + ".briggs_weighting.calculate_briggs_params",
            return_value=np.zeros((2, 3, 1)),
        ),
    ):
        calculate_imaging_weights(
            ps,
            image,
            {"weighting": "briggs", "robust": 0.5},
            frequency_matching=matching,
            truncate_uv_cells=truncate,
        )
    for operation in (grid, degrid):
        np.testing.assert_array_equal(
            operation.call_args.kwargs["frequency_map"], [2, 0]
        )
        assert operation.call_args.kwargs["truncate_uv_cells"] is truncate


def test_exact_weighting_rejects_off_grid_frequency():
    ps, image = _inputs([1.21e9, 1.0e9])
    with pytest.raises(ValueError):
        calculate_imaging_weights(
            ps,
            image,
            {"weighting": "briggs", "robust": 0.5},
            frequency_matching="exact",
        )


def test_mvc_psf_rejects_shifted_frequency_before_gridding():
    from astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field import (
        make_point_spread_function_mvc_single_field,
    )

    ps, image = _inputs([1.21e9, 1.0e9])
    with pytest.raises(ValueError, match="exactly one image frequency"):
        make_point_spread_function_mvc_single_field(ps, image, {})
