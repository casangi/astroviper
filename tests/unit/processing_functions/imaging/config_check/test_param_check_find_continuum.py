"""Tests of the find_continuum configuration and input checks."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    check_find_continuum_channel_range,
    check_find_continuum_config,
    check_find_continuum_first_round,
    check_find_continuum_first_round_store,
    check_find_continuum_inputs,
    check_find_continuum_ultra_task_memory,
)
from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
    FIND_CONTINUUM_FIRST_ROUND_VERSION,
)
from astroviper.utils.data_group_tools import modify_data_groups_xds
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    make_find_continuum_image_xds,
)

N_CHANNELS = 6
WINDOW = 230.0e9 + 1.0e6 * np.arange(6)
# Two spectral windows of 6 channels, with a gap and another channel width.
TWO_WINDOWS = np.concatenate([WINDOW, 232.0e9 + 2.0e6 * np.arange(6)])
BINDING = {
    "image_store_zarr_json_mtime_ns": 1763058000123456789,
    "frequency_sha1": "a" * 40,
    "l_sha1": "b" * 40,
    "m_sha1": "c" * 40,
    "shape": (1, 15360, 2, 11250, 11250),
}


@pytest.mark.parametrize(
    "n_bin, primary_beam_limit, n_mapping_parallelism",
    [(1, None, None), (3, 0.2, {"frequency": 4}), (1, 0, None)],
)
def test_valid_config(n_bin, primary_beam_limit, n_mapping_parallelism):
    check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism)


@pytest.mark.parametrize(
    "n_bin, primary_beam_limit, n_mapping_parallelism, error",
    [
        (0, None, None, ValueError),
        (1.5, None, None, TypeError),
        (1, 1.0, None, ValueError),
        (1, -0.1, None, ValueError),
        (1, "0.2", None, TypeError),
        (1, None, {"time": 2}, ValueError),
        (1, None, {"frequency": 0}, ValueError),
        (1, None, {"frequency": True}, TypeError),
        (1, None, [("frequency", 2)], TypeError),
    ],
)
def test_invalid_config(n_bin, primary_beam_limit, n_mapping_parallelism, error):
    with pytest.raises(error):
        check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism)


def test_first_round_of_the_imager():
    first_round = {
        "store": "cube_find_continuum_first_round.zarr",
        "primary_beam_limit": 0.2,
        "complete": True,
        "moments_complete": True,
    }
    check_find_continuum_config(1, 0.2, None, first_round)
    with pytest.raises(ValueError, match="made with primary_beam_limit 0.2, not 0.3"):
        check_find_continuum_config(1, 0.3, None, first_round)
    in_memory = {"primary_beam_limit": 0.2, "reductions": [{}, {}, {}, {}]}
    with pytest.raises(ValueError, match="path of its store as 'store'"):
        check_find_continuum_config(1, 0.2, None, in_memory)
    with pytest.raises(TypeError):
        check_find_continuum_config(1, 0.2, None, [{}, {}, {}, {}])


@pytest.mark.parametrize(
    "thread_info, memory_budget_gb",
    [
        ({"n_threads": 24, "memory_per_thread": 21.33}, None),
        ({"n_threads": 1, "memory_per_thread": 2}, 1.5),
        (None, 4),
    ],
)
def test_task_memory_settings(thread_info, memory_budget_gb):
    check_find_continuum_config(
        1, 0.2, None, thread_info=thread_info, memory_budget_gb=memory_budget_gb
    )


@pytest.mark.parametrize(
    "thread_info, memory_budget_gb, error",
    [
        ({"n_threads": 0, "memory_per_thread": 4.0}, None, ValueError),
        ({"n_threads": 2.0, "memory_per_thread": 4.0}, None, TypeError),
        ({"memory_per_thread": 4.0}, None, TypeError),
        ({"n_threads": 2, "memory_per_thread": 0.0}, None, ValueError),
        ({"n_threads": 2, "memory_per_thread": np.inf}, None, ValueError),
        ({"n_threads": 2, "memory_per_thread": "4"}, None, TypeError),
        ([2, 4.0], None, TypeError),
        (None, 0, ValueError),
        (None, -1.0, ValueError),
        (None, np.nan, ValueError),
        (None, "1", TypeError),
    ],
)
def test_invalid_task_memory_settings(thread_info, memory_budget_gb, error):
    with pytest.raises(error):
        check_find_continuum_config(
            1, 0.2, None, thread_info=thread_info, memory_budget_gb=memory_budget_gb
        )


def test_first_round_needs_stokes_i_in_zarr():
    check_find_continuum_first_round({"polarization_coords": ["I", "Q"]}, "zarr")
    with pytest.raises(ValueError, match="Stokes I"):
        check_find_continuum_first_round({"polarization_coords": ["Q"]}, "zarr")
    with pytest.raises(ValueError, match="output_image_format 'zarr', not 'fits'"):
        check_find_continuum_first_round({"polarization_coords": ["I"]}, "fits")


def _store_attrs(**changes):
    first_round = {
        "version": FIND_CONTINUUM_FIRST_ROUND_VERSION,
        "written_by": "image_cube_single_field",
        "complete": True,
        "primary_beam_limit": 0.2,
        "polarization_index": 0,
        "tile_shape": [625, 625],
        "binding": dict(BINDING, shape=list(BINDING["shape"])),
    }
    return {"find_continuum_first_round": dict(first_round, **changes)}


def test_first_round_store_of_the_cube():
    check_find_continuum_first_round_store(_store_attrs(), BINDING, 0.2, 0, (625, 625))
    check_find_continuum_first_round_store(
        _store_attrs(primary_beam_limit=None), BINDING, None, 0, (625, 625)
    )


@pytest.mark.parametrize(
    "attrs, match",
    [
        (None, "missing"),
        ({"other": 1}, "missing"),
        (_store_attrs(version=2), "version 2, not 1"),
        (_store_attrs(complete=False), "incomplete"),
        (_store_attrs(primary_beam_limit=0.3), "primary_beam_limit 0.3, not 0.2"),
        (_store_attrs(polarization_index=1), "polarization_index 1, not 0"),
        (_store_attrs(tile_shape=[1250, 625]), r"tile_shape \[1250, 625\], not"),
        *[
            (_store_attrs(binding=dict(BINDING, **{name: value})), f"{name} differs")
            for name, value in (
                (
                    "image_store_zarr_json_mtime_ns",
                    BINDING["image_store_zarr_json_mtime_ns"] + 1,
                ),
                ("frequency_sha1", "d" * 40),
                ("l_sha1", "e" * 40),
                ("m_sha1", "f" * 40),
                ("shape", [1, 15360, 1, 11250, 11250]),
            )
        ],
        (_store_attrs(binding={"shape": BINDING["shape"]}), "frequency_sha1 differs"),
    ],
)
def test_first_round_store_refused(attrs, match):
    with pytest.raises(ValueError, match=match):
        check_find_continuum_first_round_store(attrs, BINDING, 0.2, 0, (625, 625))


def test_ultra_task_memory():
    check_find_continuum_ultra_task_memory(2**30, 2**30)
    with pytest.raises(ValueError, match="needs 2.00 GiB .* budget of 1.00 GiB"):
        check_find_continuum_ultra_task_memory(2 * 2**30, 2**30)


def test_channel_range_of_one_window():
    assert check_find_continuum_channel_range(None, WINDOW) == [0, 6]
    assert check_find_continuum_channel_range([1, 4], WINDOW) == [1, 4]
    assert check_find_continuum_channel_range([6, 12], TWO_WINDOWS) == [6, 12]
    channel_range = check_find_continuum_channel_range(np.array([0, 6]), TWO_WINDOWS)
    assert channel_range == [0, 6] and all(type(c) is int for c in channel_range)
    assert check_find_continuum_channel_range(None, WINDOW[:1]) == [0, 1]


def test_cube_of_two_windows_needs_a_channel_range():
    with pytest.raises(
        ValueError,
        match=r"ranges are \[\[0, 6\], \[6, 12\]\]: pass one of them as channel_range",
    ):
        check_find_continuum_channel_range(None, TWO_WINDOWS)
    with pytest.raises(ValueError, match=r"channels \[4, 8\) of the cube are not"):
        check_find_continuum_channel_range([4, 8], TWO_WINDOWS)


@pytest.mark.parametrize(
    "channel_range, error",
    [
        ([3, 3], ValueError),
        ([4, 2], ValueError),
        ([0, 7], ValueError),
        ([-1, 2], ValueError),
        ([0], ValueError),
        ([0.0, 2], TypeError),
        ([True, 2], TypeError),
    ],
)
def test_invalid_channel_range(channel_range, error):
    with pytest.raises(error):
        check_find_continuum_channel_range(channel_range, WINDOW)


@pytest.fixture
def img_xds():
    return make_find_continuum_image_xds(n_pixels=8, n_channels=N_CHANNELS)


def test_valid_inputs(img_xds):
    check_find_continuum_inputs(
        img_xds, "residual", np.ones(N_CHANNELS), [20.0] * N_CHANNELS
    )


def test_data_group_must_hold_every_cube(img_xds):
    with pytest.raises(ValueError, match="no data group"):
        check_find_continuum_inputs(
            img_xds, "model", np.ones(N_CHANNELS), np.ones(N_CHANNELS)
        )
    modify_data_groups_xds(img_xds, "sky_only", {"sky": "SKY_RESIDUAL"}, "Sky only")
    with pytest.raises(ValueError, match="lacks the roles"):
        check_find_continuum_inputs(
            img_xds, "sky_only", np.ones(N_CHANNELS), np.ones(N_CHANNELS)
        )


def test_point_spread_function_cube_is_not_needed(img_xds):
    residual = img_xds.attrs["data_groups"]["residual"]
    roles = ("sky", "primary_beam", "beam_fit_params_point_spread_function")
    modify_data_groups_xds(
        img_xds, "beams", {role: residual[role] for role in roles}, "PSF beams only"
    )
    check_find_continuum_inputs(
        img_xds, "beams", np.ones(N_CHANNELS), np.ones(N_CHANNELS)
    )


def test_one_time_and_stokes_i(img_xds):
    two_times = xr.concat([img_xds, img_xds.assign_coords(time=[1.0])], dim="time")
    with pytest.raises(ValueError, match="one time"):
        check_find_continuum_inputs(
            two_times, "residual", np.ones(N_CHANNELS), np.ones(N_CHANNELS)
        )
    with pytest.raises(ValueError, match="Stokes I"):
        check_find_continuum_inputs(
            img_xds.assign_coords(polarization=["Q"]),
            "residual",
            np.ones(N_CHANNELS),
            np.ones(N_CHANNELS),
        )


@pytest.mark.parametrize("which", ["transmission", "sky_temperature"])
def test_one_atmospheric_value_per_channel(img_xds, which):
    spectra = {
        "transmission": np.ones(N_CHANNELS),
        "sky_temperature": np.ones(N_CHANNELS),
    }
    spectra[which] = np.ones(N_CHANNELS + 1)
    with pytest.raises(
        ValueError, match=f"atmospheric_{which} must have one value per channel"
    ):
        check_find_continuum_inputs(
            img_xds, "residual", spectra["transmission"], spectra["sky_temperature"]
        )
