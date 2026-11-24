"""Tests of the find_continuum configuration and input checks."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    check_find_continuum_channel_range,
    check_find_continuum_cluster_threads,
    check_find_continuum_config,
    check_find_continuum_first_round,
    check_find_continuum_first_round_store,
    check_find_continuum_inputs,
    check_find_continuum_ultra_config,
    check_find_continuum_ultra_inputs,
    check_find_continuum_ultra_task_config,
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
VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}


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


def test_first_round_needs_stokes_i_in_zarr():
    check_find_continuum_first_round({"polarization_coords": ["I", "Q"]}, "zarr")
    with pytest.raises(ValueError, match="Stokes I"):
        check_find_continuum_first_round({"polarization_coords": ["Q"]}, "zarr")
    with pytest.raises(ValueError, match="output_image_format 'zarr', not 'fits'"):
        check_find_continuum_first_round({"polarization_coords": ["I"]}, "fits")


def _store_attrs(binding=BINDING, **changes):
    first_round = {
        "version": FIND_CONTINUUM_FIRST_ROUND_VERSION,
        "written_by": "image_cube_single_field",
        "complete": True,
        "primary_beam_limit": 0.2,
        "polarization_index": 0,
        "tile_shape": [625, 625],
        "variables": VARIABLES,
        "windows": [
            {"channel_range": [0, 3840]},
            {"channel_range": [3840, 7680]},
        ],
    }
    return {
        "find_continuum_first_round": dict(first_round, **changes),
        "image_binding": dict(binding, shape=list(binding["shape"])),
    }


@pytest.mark.parametrize("channel_range", [[0, 3840], [3840, 7680], [4000, 5000]])
def test_first_round_store_of_the_cube(channel_range):
    assert FIND_CONTINUUM_FIRST_ROUND_VERSION == 3
    check_find_continuum_first_round_store(
        _store_attrs(), BINDING, 0.2, 0, (625, 625), VARIABLES, channel_range
    )


@pytest.mark.parametrize(
    "attrs, channel_range, match",
    [
        (None, [0, 3840], "missing"),
        ({"other": 1}, [0, 3840], "missing"),
        (
            _store_attrs(version=1),
            [0, 3840],
            "version 1, not 3: make the first round again",
        ),
        (_store_attrs(version=2), [0, 3840], "version 2, not 3"),
        (_store_attrs(complete=False), [0, 3840], "incomplete"),
        (
            _store_attrs(primary_beam_limit=0.3),
            [0, 3840],
            "primary_beam_limit 0.3, not 0.2",
        ),
        (
            _store_attrs(polarization_index=1),
            [0, 3840],
            "polarization_index 1, not 0",
        ),
        (
            _store_attrs(tile_shape=[1250, 625]),
            [0, 3840],
            r"tile_shape \[1250, 625\], not",
        ),
        (
            _store_attrs(variables=dict(VARIABLES, sky="SKY_RESTORED")),
            [0, 3840],
            "made with variables .*SKY_RESTORED.*, not .*SKY_RESIDUAL",
        ),
        *[
            (
                _store_attrs(),
                channel_range,
                r"holds channels \[\[0, 3840\], \[3840, 7680\]\] of the cube, "
                rf"not \[{channel_range[0]}, {channel_range[1]}\)",
            )
            for channel_range in ([3000, 4000], [7680, 11520], [0, 7680])
        ],
        *[
            (
                _store_attrs(binding=dict(BINDING, **{name: value})),
                [0, 3840],
                f"{name} differs",
            )
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
        (
            _store_attrs(binding={"shape": BINDING["shape"]}),
            [0, 3840],
            "frequency_sha1 differs",
        ),
        (
            {
                "find_continuum_first_round": _store_attrs()[
                    "find_continuum_first_round"
                ]
            },
            [0, 3840],
            "frequency_sha1 differs",
        ),
    ],
)
def test_first_round_store_refused(attrs, channel_range, match):
    with pytest.raises(ValueError, match=match):
        check_find_continuum_first_round_store(
            attrs, BINDING, 0.2, 0, (625, 625), VARIABLES, channel_range
        )


def test_ultra_task_memory():
    check_find_continuum_ultra_task_memory(2**30, 2**30)
    with pytest.raises(
        ValueError,
        match=r"needs 2.00 GiB .* than the 1.00 GiB of memory per thread: .*"
        r"thread_info\['memory_per_thread'\]",
    ):
        check_find_continuum_ultra_task_memory(2 * 2**30, 2**30)


@pytest.mark.parametrize(
    "thread_info",
    [
        {"n_threads": 0, "memory_per_thread": -1},
        {"n_threads": 4, "memory_per_thread": 0.0},
        {"n_threads": 4, "memory_per_thread": float("inf")},
    ],
)
def test_cluster_threads_without_workers_or_memory_limit(thread_info):
    check_find_continuum_cluster_threads({"n_threads": 24, "memory_per_thread": 20.0})
    with pytest.raises(ValueError, match="pass thread_info"):
        check_find_continuum_cluster_threads(thread_info)


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


ULTRA_CONFIG = dict(
    n_bin=1,
    spectral_dynamic_range_bandwidth=None,
    telescope_name="ALMA",
    single_continuum=False,
    primary_beam_limit=0.2,
    first_round_store=None,
    n_mapping_parallelism=None,
    thread_info=None,
    memory_budget_gb=None,
    monitor_resources_seconds=None,
)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        dict(
            n_bin=3,
            spectral_dynamic_range_bandwidth=1.875e9,
            telescope_name="VLA",
            single_continuum=True,
            primary_beam_limit=0.5,
            first_round_store="cube_find_continuum_first_round.zarr",
            n_mapping_parallelism={"frequency": 4},
            thread_info={"n_threads": 24, "memory_per_thread": 20.0, "other": 1},
            memory_budget_gb=12,
            monitor_resources_seconds=0.5,
        ),
    ],
)
def test_valid_ultra_config(changes):
    check_find_continuum_ultra_config(**{**ULTRA_CONFIG, **changes})


@pytest.mark.parametrize(
    "changes, error, match",
    [
        (dict(n_bin=0), ValueError, "n_bin"),
        (dict(n_bin=1.0), TypeError, "n_bin"),
        (dict(spectral_dynamic_range_bandwidth=0.0), ValueError, "positive"),
        (dict(spectral_dynamic_range_bandwidth=np.inf), ValueError, "finite"),
        (dict(spectral_dynamic_range_bandwidth="1GHz"), TypeError, "real"),
        (dict(telescope_name="alma"), ValueError, r"\('ALMA', 'VLA'\), not 'alma'"),
        (dict(single_continuum=1), TypeError, "bool"),
        (dict(primary_beam_limit=0.0), ValueError, r"\(0, 1\)"),
        (dict(primary_beam_limit=1.0), ValueError, r"\(0, 1\)"),
        (dict(primary_beam_limit=None), TypeError, "real"),
        (dict(first_round_store=1), TypeError, "first_round_store"),
        (dict(n_mapping_parallelism={"time": 2}), ValueError, "frequency"),
        (dict(n_mapping_parallelism={"frequency": 0}), ValueError, ">= 1"),
        (dict(thread_info=[24, 20.0]), TypeError, "mapping"),
        (dict(thread_info={"memory_per_thread": 20.0}), TypeError, "n_threads"),
        (
            dict(thread_info={"n_threads": 0, "memory_per_thread": 20.0}),
            ValueError,
            ">= 1",
        ),
        (
            dict(thread_info={"n_threads": 4, "memory_per_thread": 0.0}),
            ValueError,
            "positive",
        ),
        (dict(memory_budget_gb=-1.0), ValueError, "memory_budget_gb"),
        (dict(monitor_resources_seconds=np.nan), ValueError, "finite"),
    ],
)
def test_invalid_ultra_config(changes, error, match):
    with pytest.raises(error, match=match):
        check_find_continuum_ultra_config(**{**ULTRA_CONFIG, **changes})


def test_ultra_task_config():
    check_find_continuum_ultra_task_config(4, 0, [3, 45])
    check_find_continuum_ultra_task_config(1, 1, (0, 1))
    for arguments, error in (
        ((0, 0, [3, 45]), ValueError),
        ((4.0, 0, [3, 45]), TypeError),
        ((4, -1, [3, 45]), ValueError),
        ((4, 0, [3]), ValueError),
        ((4, 0, [3, 3]), ValueError),
        ((4, 0, [3.0, 45]), TypeError),
    ):
        with pytest.raises(error):
            check_find_continuum_ultra_task_config(*arguments)


@pytest.fixture
def window_xds():
    return make_find_continuum_image_xds(n_pixels=8, n_channels=12)


def _ultra_inputs(img_xds, channel_range=None, n_channels=12, **spectra):
    return check_find_continuum_ultra_inputs(
        img_xds,
        "residual",
        channel_range,
        spectra.get("transmission", np.full(n_channels, 0.9)),
        spectra.get("sky_temperature", [20.0] * n_channels),
    )


def test_valid_ultra_inputs(window_xds):
    assert _ultra_inputs(window_xds) == [0, 12]
    assert _ultra_inputs(window_xds, [2, 10], n_channels=8) == [2, 10]
    residual = window_xds.attrs["data_groups"]["residual"]
    roles = ("sky", "primary_beam", "beam_fit_params_point_spread_function")
    modify_data_groups_xds(
        window_xds, "beams", {role: residual[role] for role in roles}, "PSF beams"
    )
    check_find_continuum_ultra_inputs(
        window_xds, "beams", None, np.ones(12), np.ones(12)
    )


def _without_rest_frequency(img_xds):
    frequency = img_xds.frequency.copy()
    frequency.attrs = {
        key: value for key, value in frequency.attrs.items() if key != "rest_frequency"
    }
    return img_xds.assign_coords(frequency=frequency)


def _with_beam(img_xds, value):
    beams = img_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].values.copy()
    beams[0, 3, 0, 1] = value
    return img_xds.assign(
        BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION=img_xds[
            "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
        ].copy(data=beams)
    )


@pytest.mark.parametrize(
    "change, arguments, error, match",
    [
        (
            lambda xds: xds,
            dict(channel_range=[0, 7], n_channels=7),
            ValueError,
            "at least 8",
        ),
        (lambda xds: xds, dict(channel_range=[0, 13]), ValueError, "outside"),
        (
            lambda xds: xds.isel(frequency=slice(None, None, -1)),
            {},
            ValueError,
            "must increase",
        ),
        (
            lambda xds: xds.assign(
                SKY_RESIDUAL=xds.SKY_RESIDUAL.transpose(
                    "time", "frequency", "polarization", "m", "l"
                )
            ),
            {},
            ValueError,
            "dimensions",
        ),
        (lambda xds: xds, dict(n_channels=11), ValueError, "one value per channel"),
        (
            lambda xds: xds,
            dict(transmission=np.r_[np.nan, np.ones(11)]),
            ValueError,
            "finite",
        ),
        (lambda xds: xds, dict(sky_temperature=np.zeros(12)), ValueError, "no zero"),
        (
            lambda xds: xds,
            dict(transmission=np.array(["a"] * 12)),
            TypeError,
            "real",
        ),
        (lambda xds: _with_beam(xds, 0.0), {}, ValueError, "beam axes"),
        (lambda xds: _with_beam(xds, np.nan), {}, ValueError, "beam axes"),
        (_without_rest_frequency, {}, ValueError, "rest_frequency"),
        (lambda xds: xds.isel(l=slice(0, 1)), {}, ValueError, "two pixels"),
        (
            lambda xds: xds.assign_coords(polarization=["Q"]),
            {},
            ValueError,
            "Stokes I",
        ),
    ],
)
def test_ultra_inputs_refused(window_xds, change, arguments, error, match):
    with pytest.raises(error, match=match):
        _ultra_inputs(change(window_xds), **arguments)


def test_ultra_inputs_need_the_roles_and_one_time(window_xds):
    with pytest.raises(ValueError, match="no data group"):
        check_find_continuum_ultra_inputs(
            window_xds, "model", None, np.ones(12), np.ones(12)
        )
    modify_data_groups_xds(window_xds, "sky_only", {"sky": "SKY_RESIDUAL"}, "Sky only")
    with pytest.raises(ValueError, match="lacks the roles"):
        check_find_continuum_ultra_inputs(
            window_xds, "sky_only", None, np.ones(12), np.ones(12)
        )
    two_times = xr.concat(
        [window_xds, window_xds.assign_coords(time=[1.0])], dim="time"
    )
    with pytest.raises(ValueError, match="one time"):
        _ultra_inputs(two_times)


def test_first_round_requires_an_explicitly_retained_primary_beam():
    with pytest.raises(ValueError, match="primary_beam in image_data_variables_keep"):
        check_find_continuum_first_round(
            {"polarization_coords": ["I"]},
            "zarr",
            image_data_variables_keep=["sky_residual"],
        )
    check_find_continuum_first_round(
        {"polarization_coords": ["I"]},
        "zarr",
        image_data_variables_keep=["primary_beam"],
    )
