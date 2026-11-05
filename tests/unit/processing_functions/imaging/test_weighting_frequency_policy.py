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


def test_collapsed_global_weight_lookup_maps_channels_without_repeating_grid():
    """Continuum weighting shares one density plane through a zero channel map."""
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        degrid_imaging_weights_continuum,
    )

    ps, image = _inputs([1.0e9, 1.2e9])
    density = np.ones((1, 1, 8, 8))
    briggs_factors = np.arange(2, dtype=np.float64).reshape(2, 1, 1)
    global_weights = xr.Dataset(
        {
            "WEIGHT_DENSITY_GRID": (
                ("frequency", "weight_polarization", "u", "v"),
                density,
            ),
            "SUM_WEIGHT": (("frequency", "weight_polarization"), np.ones((1, 1))),
            "BRIGGS_FACTORS": (
                ("briggs_parameter", "frequency", "weight_polarization"),
                briggs_factors,
            ),
        },
        coords={"frequency": [1.1e9]},
        attrs={"continuum_frequency_collapsed": True},
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

    np.testing.assert_array_equal(degrid.call_args.args[0], density)
    np.testing.assert_array_equal(degrid.call_args.args[3], briggs_factors)
    np.testing.assert_array_equal(
        degrid.call_args.kwargs["frequency_map"], np.zeros(2, dtype=np.int64)
    )


class _WeightsWritten(Exception):
    """Raised in place of an imaging setup's PSF, after its weights are written."""


def _one_channel_inputs():
    """One channel; about three samples per density cell, flagged ones too."""
    rng = np.random.default_rng(3)
    dims = ("time", "baseline", "frequency", "polarization")
    shape = (20, 30, 1, 2)
    flag = np.zeros(shape, dtype=np.int8)
    flag[0, :3] = 1
    weight = rng.uniform(0.5, 2.0, shape)
    # Up to +-10.7 cells from the centre of the 32 x 32 density grid.
    uvw = rng.uniform(-1.0e3, 1.0e3, (20, 30, 3))
    # Two time steps reach the edge in u or in v, 14.9 to 17.1 cells out, where
    # both, one or neither of the two Hermitian cells is on the grid.
    for time, axis in ((1, 0), (2, 1)):
        uvw[time, :, axis] = rng.choice([-1.0, 1.0], 30) * rng.uniform(1.4e3, 1.6e3, 30)
    ms = xr.Dataset(
        {
            "WEIGHT": (dims, weight),
            "FLAG": (dims, flag),
            "UVW": (("time", "baseline", "uvw_label"), uvw),
        },
        coords={"frequency": [1.0e9]},
        attrs={
            "data_groups": {"base": {"weight": "WEIGHT", "flag": "FLAG", "uvw": "UVW"}}
        },
    )
    image = xr.Dataset(
        coords={
            "frequency": [1.0e9],
            "l": np.arange(32) * 1e-4,
            "m": np.arange(32) * 1e-4,
        },
        attrs={"type": "image_dataset", "data_groups": {"base": {}}},
    )
    return xr.DataTree.from_dict({"ms": ms}), image


@pytest.mark.parametrize(
    "weighting",
    [{"weighting": "briggs", "robust": 0.5}, {"weighting": "uniform"}],
    ids=["briggs", "uniform"],
)
@pytest.mark.parametrize("casa", [False, True])
def test_one_channel_continuum_weights_equal_cube_weights(monkeypatch, weighting, casa):
    """Both continuum scopes use the cube's nearest density cell, bit for bit."""
    import importlib

    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        degrid_imaging_weights_continuum,
        grid_imaging_weight_density_continuum,
        normalize_imaging_weight_params,
    )
    from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
        imaging_setup_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.imaging_weighting.briggs_weighting import (
        calculate_briggs_params,
    )
    from astroviper.processing_functions.imaging.residual_update import (
        imaging_setup_single_field,
    )

    def stop(*args, **kwargs):
        raise _WeightsWritten

    module = "astroviper.processing_functions.imaging."
    for name in (
        "make_point_spread_function.make_point_spread_function_single_field",
        "make_point_spread_function_continuum_single_field."
        "make_point_spread_function_continuum_single_field",
    ):
        module_name, function_name = name.rsplit(".", 1)
        monkeypatch.setattr(
            importlib.import_module(module + module_name), function_name, stop
        )

    weighting = {**weighting, "casa_weighting_implementation": casa}
    local = {**weighting, "weighting_scope": "local"}
    weights = {}
    for name, setup in (
        ("cube", imaging_setup_single_field),
        ("continuum local", imaging_setup_continuum_single_field),
    ):
        ps, image = _one_channel_inputs()
        with pytest.raises(_WeightsWritten):
            setup(
                ps,
                image,
                {"n_taylor_terms": 1, "reference_frequency": 1.0e9},
                local,
                processing_set_data_group_name="base",
            )
        weights[name] = ps["ms"]["WEIGHT_IMAGING"].values

    # Global scope: the map, reduce (one partition) and degrid stages.
    glob = {**weighting, "weighting_scope": "global"}
    ps, image = _one_channel_inputs()
    density = grid_imaging_weight_density_continuum(ps, image, glob)
    density["BRIGGS_FACTORS"] = (
        ("briggs_parameter", "frequency", "weight_polarization"),
        calculate_briggs_params(
            density.WEIGHT_DENSITY_GRID.values,
            density.SUM_WEIGHT.values,
            normalize_imaging_weight_params(glob),
        ),
    )
    degrid_imaging_weights_continuum(ps, image, density, glob)
    weights["continuum global"] = ps["ms"]["WEIGHT_IMAGING"].values

    for name in ("continuum local", "continuum global"):
        np.testing.assert_array_equal(weights[name], weights["cube"], err_msg=name)

    # CASA's continuum truncation moves samples to other cells and weights.
    ps, image = _one_channel_inputs()
    calculate_imaging_weights(ps, image, local, truncate_uv_cells=True)
    truncated = ps["ms"]["WEIGHT_IMAGING"].values
    assert not np.allclose(truncated, weights["cube"], equal_nan=True)
