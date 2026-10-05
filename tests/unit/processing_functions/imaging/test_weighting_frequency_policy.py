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


def test_mvc_psf_accepts_shifted_frequency_with_cube_gridder():
    from astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field import (
        make_point_spread_function_mvc_single_field,
    )

    ps, image = _inputs([1.21e9, 1.0e9])
    module = "astroviper.processing_functions.imaging."
    with (
        patch(module + "utils.drop_auto_correlations"),
        patch(
            module + "make_point_spread_function.add_uv_sampling_grid_single_field"
        ) as grid,
    ):
        make_point_spread_function_mvc_single_field(ps, image, {"fft_padding": 1.2})
    grid.assert_called_once()
    assert grid.call_args.kwargs["chan_mode"] == "cube"


def test_global_weight_lookup_uses_nearest_channel_for_shifted_frequencies():
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        degrid_imaging_weights_continuum,
    )

    ps, image = _inputs([1.21e9, 1.22e9])
    density = np.broadcast_to(np.arange(1, 4)[:, None, None, None], (3, 1, 8, 8)).copy()
    global_weights = xr.Dataset(
        {
            "WEIGHT_DENSITY_GRID": (
                ("frequency", "weight_polarization", "u", "v"),
                density,
            ),
            "SUM_WEIGHT": (("frequency", "weight_polarization"), np.ones((3, 1))),
            "BRIGGS_FACTORS": (
                ("briggs_parameter", "frequency", "weight_polarization"),
                np.ones((2, 3, 1)),
            ),
        },
        coords={"frequency": image.frequency.values},
    )
    module = (
        "astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights"
    )
    with patch(
        module + ".degrid_imaging_weights", return_value=np.ones((1, 1, 2, 1))
    ) as degrid:
        degrid_imaging_weights_continuum(
            ps, image, global_weights, {"weighting": "briggs", "robust": 0.5}
        )
    # Both shifted observations select the third global density plane.
    np.testing.assert_array_equal(degrid.call_args.args[0], density[[2, 2]])
