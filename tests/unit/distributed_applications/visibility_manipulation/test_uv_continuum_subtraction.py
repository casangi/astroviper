"""Full-stack tests of the uv_continuum_subtraction distributed application.

A small simulated processing set has two spectral windows with a flat
continuum and a line at the phase centre; the Antennae TOPO test data
(``Antennae_fld1_topo.ps.zarr``) has continuum ranges in LSRK.
"""

import functools
import importlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from toolviper.utils.data import download
from xradio.measurement_set import load_processing_set, open_processing_set
from xradio.schema.check import check_datatree

from astroviper.distributed_applications.visibility_manipulation import (
    uv_continuum_subtraction,
)
from astroviper.distributed_applications.visibility_manipulation.uv_continuum_subtraction import (
    DISTRIBUTED_APPLICATION_TIMING_PHASES,
    OUTPUT_CHUNK_BYTES,
    _output_chunking,
    _time_blocks,
    combine_uv_continuum_subtraction_results,
)
from astroviper.processing_functions.visibility_manipulation import (
    uv_continuum_subtraction as uv_continuum_subtraction_pf,
)
from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
    measurement_set_doppler_factors,
)

N_ANTENNA, N_CHANNELS = 6, 32
CHANNEL_WIDTH = 1e6
PHASE_CENTER = np.array([1.2, -0.4])
LINE_CHANNELS = slice(12, 20)
FIRST_FREQUENCY = {"spw_0": 230.0e9, "spw_1": 231.0e9}
TOPO_PS = "Antennae_fld1_topo.ps.zarr"


def _correlations(spectrum):
    zeros = np.zeros_like(spectrum)
    return np.stack([spectrum, zeros, zeros, spectrum], axis=-1)


def _line():
    line = np.zeros(N_CHANNELS)
    line[LINE_CHANNELS] = np.hanning(LINE_CHANNELS.stop - LINE_CHANNELS.start + 2)[1:-1]
    return line


@pytest.fixture(scope="module")
def simulated_store(tmp_path_factory):
    from astroviper.distributed_applications.simulation import (
        simulate_processing_set,
    )
    from astroviper.utils.beam_models import airy_disk_model
    from astroviper.utils.telescope_layout import read_telescope_layout

    store = str(tmp_path_factory.mktemp("uv_continuum") / "simulated.ps.zarr")
    for name, first_frequency in FIRST_FREQUENCY.items():
        simulate_processing_set(
            ps_store=store,
            antenna_xds=read_telescope_layout(
                "alma.cycle8.1", antenna_selection=list(range(N_ANTENNA))
            ),
            time_params={
                "time_start": "2026-03-01T22:12:00.000",
                "time_delta": 60.0,
                "n_samples": 5,
            },
            frequency_params={
                "freq_start": first_frequency,
                "freq_delta": CHANNEL_WIDTH,
                "n_channels": N_CHANNELS,
                "channel_width": CHANNEL_WIDTH,
                "spectral_window_name": name,
            },
            polarization=["XX", "YY"],
            sky_components=[
                {
                    "kind": "point",
                    "flux": _correlations(np.full(N_CHANNELS, 0.5)),
                    "ra_dec": PHASE_CENTER,
                },
                {
                    "kind": "point",
                    "flux": _correlations(_line()),
                    "ra_dec": PHASE_CENTER,
                },
            ],
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[airy_disk_model("alma")],
            beam_model_map=np.zeros(N_ANTENNA, int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    return store


@pytest.fixture
def store(simulated_store, tmp_path):
    copy = str(tmp_path / "simulated.ps.zarr")
    shutil.copytree(simulated_store, copy)
    return copy


def _continuum_ranges(name="spw_0"):
    frequency = FIRST_FREQUENCY[name] + CHANNEL_WIDTH * np.arange(N_CHANNELS)
    half = CHANNEL_WIDTH / 2
    return [
        [frequency[0] - half, frequency[LINE_CHANNELS.start - 1] + half],
        [frequency[LINE_CHANNELS.stop] - half, frequency[-1] + half],
    ]


def test_full_stack_matches_processing_function(store):
    result = uv_continuum_subtraction(
        ps_store=store,
        spectral_window_name="spw_0",
        continuum_frequency_ranges=_continuum_ranges(),
        continuum_frequency_frame="lsrk",
        fit_order=1,
        n_mapping_parallelism={"time": 2},
    )
    assert result["ms_names"] == ["ALMA_spw_0"]
    ps_xdt = open_processing_set(store)
    ms_xdt = ps_xdt["ALMA_spw_0"]
    group = ms_xdt.attrs["data_groups"]["continuum_subtracted"]
    assert group["correlated_data"] == "VISIBILITY_CONTINUUM_SUBTRACTED"
    assert group["flag"] == "FLAG_CONTINUUM_SUBTRACTED"
    assert group["weight"] == "WEIGHT"
    assert "uv_continuum_subtraction: polynomial of order 1" in group["description"]
    assert list(ps_xdt["ALMA_spw_1"].attrs["data_groups"]) == ["base"]
    assert "VISIBILITY_CONTINUUM_SUBTRACTED" not in ps_xdt["ALMA_spw_1"].data_vars
    assert str(check_datatree(ps_xdt)) == "No schema issues found"

    in_memory = load_processing_set(store, data_group_name="base")["ALMA_spw_0"]
    uv_continuum_subtraction_pf(in_memory, _continuum_ranges(), "lsrk", 1)
    written = ms_xdt.VISIBILITY_CONTINUUM_SUBTRACTED.values
    np.testing.assert_array_equal(
        written, in_memory.VISIBILITY_CONTINUUM_SUBTRACTED.values
    )
    np.testing.assert_allclose(
        written,
        np.broadcast_to(_line()[:, np.newaxis], written.shape),
        atol=1e-12,
    )
    assert not ms_xdt.FLAG_CONTINUUM_SUBTRACTED.values.any()
    # The on-disk chunks are the tasks' blocks of times.
    tasks = result["timing_node_tasks"]
    assert tasks["n_time"].sum() == ms_xdt.sizes["time"]
    chunks = zarr.open_array(
        os.path.join(store, "ALMA_spw_0", "VISIBILITY_CONTINUUM_SUBTRACTED")
    ).chunks
    assert chunks[0] in (tasks["n_time"].max(), 1) and chunks[2:] == (N_CHANNELS, 2)

    statistics = result["fit_statistics"]
    assert list(statistics["polarization"]) == ["XX", "YY"]
    n_baseline = ms_xdt.sizes["baseline_id"]
    assert (statistics["n_fitted"] == 5 * n_baseline).all()
    assert (statistics["chi_squared_max_real"] < 1e-20).all()
    channels = result["continuum_channels"].iloc[0]
    assert channels["n_continuum_channels_every_time"] == N_CHANNELS - 8
    assert channels["continuum_channel_fraction"] == pytest.approx(24 / 32)
    spectra = result["mean_spectra"]["ALMA_spw_0"]
    np.testing.assert_allclose(
        spectra["VISIBILITY_MEAN_INPUT"].values[0], 0.5, rtol=1e-12
    )
    np.testing.assert_allclose(
        spectra["VISIBILITY_MEAN_OUTPUT"].values[:, 1], _line(), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        spectra["CONTINUUM_TIME_FRACTION"].values[LINE_CHANNELS], 0.0
    )
    timing = result["timing_distributed_application"]
    for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES:
        for _, key in steps:
            assert timing[key] >= 0.0
    assert timing["T_total"] > 0.0
    assert result["report"] is None

    with pytest.raises(ValueError, match="Data group 'continuum_subtracted' exists"):
        uv_continuum_subtraction(
            ps_store=store,
            spectral_window_name="spw_0",
            continuum_frequency_ranges=_continuum_ranges(),
            continuum_frequency_frame="lsrk",
        )
    result = uv_continuum_subtraction(
        ps_store=store,
        spectral_window_name="spw_0",
        continuum_frequency_ranges=[_continuum_ranges()[0]],
        continuum_frequency_frame="lsrk",
        fit_order=0,
        n_mapping_parallelism={"time": 1},
        overwrite=True,
    )
    ms_xdt = open_processing_set(store)["ALMA_spw_0"]
    assert (
        ms_xdt.attrs["data_groups"]["continuum_subtracted"]["description"].count(
            "uv_continuum_subtraction"
        )
        == 1
    )
    line = np.broadcast_to(
        _line(), (ms_xdt.sizes["time"], ms_xdt.sizes["baseline_id"], N_CHANNELS)
    )
    np.testing.assert_allclose(
        ms_xdt.VISIBILITY_CONTINUUM_SUBTRACTED.values[..., 0], line, atol=1e-12
    )


def test_report_and_resource_monitor(store):
    result = uv_continuum_subtraction(
        ps_store=store,
        spectral_window_name="spw_1",
        continuum_frequency_ranges=_continuum_ranges("spw_1"),
        continuum_frequency_frame="lsrk",
        ms_data_group_out_name="line",
        ms_data_group_out_modified={
            "correlated_data": "VISIBILITY_LINE",
            "flag": "FLAG_LINE",
        },
        thread_info={"n_threads": 2, "memory_per_thread": 1.0},
        monitor_resources_seconds=0.01,
        report="html",
    )
    assert result["report"].endswith("simulated_spw_1_report.html")
    assert os.path.isfile(result["report"])
    assert "memory_rss_bytes" in result["timing_node_tasks"]
    ms_xdt = open_processing_set(store)["ALMA_spw_1"]
    assert ms_xdt.attrs["data_groups"]["line"]["correlated_data"] == "VISIBILITY_LINE"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"spectral_window_name": "spw_9"}, "no MSv4 of the spectral window"),
        ({"ms_data_group_in_name": "corrected"}, "no data group 'corrected'"),
        ({"continuum_frequency_frame": "TOPO"}, "cannot be converted"),
        ({"n_mapping_parallelism": {"frequency": 2}}, 'must be {"time": n}'),
        ({"n_mapping_parallelism": {"time": 0}}, "positive integer"),
        ({"thread_info": {"n_threads": 2}}, "memory_per_thread"),
        ({"monitor_resources_seconds": 0.0}, "must be positive"),
        ({"continuum_frequency_ranges": [[2.0, 1.0]]}, "low <= high"),
        (
            {
                "ms_data_group_out_modified": {
                    "correlated_data": "VISIBILITY",
                    "flag": "FLAG_LINE",
                }
            },
            "is a variable of the data group 'base'",
        ),
    ],
)
def test_refusals_before_anything_is_written(store, overrides, message):
    kwargs = {
        "ps_store": store,
        "spectral_window_name": "spw_0",
        "continuum_frequency_ranges": _continuum_ranges(),
        "continuum_frequency_frame": "lsrk",
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        uv_continuum_subtraction(**kwargs)
    ms_xdt = open_processing_set(store)["ALMA_spw_0"]
    assert list(ms_xdt.attrs["data_groups"]) == ["base"]
    assert "VISIBILITY_CONTINUUM_SUBTRACTED" not in ms_xdt.data_vars


def test_refusals_of_the_measurement_set(store):
    group = zarr.open_group(os.path.join(store, "ALMA_spw_0"), mode="r+")
    data_groups = dict(group.attrs["data_groups"])
    data_groups["no_weight"] = {
        key: value for key, value in data_groups["base"].items() if key != "weight"
    }
    group.attrs["data_groups"] = data_groups
    zarr.consolidate_metadata(store)
    with pytest.raises(ValueError, match="has no 'weight' role"):
        uv_continuum_subtraction(
            ps_store=store,
            spectral_window_name="spw_0",
            continuum_frequency_ranges=_continuum_ranges(),
            continuum_frequency_frame="lsrk",
            ms_data_group_in_name="no_weight",
        )
    group.attrs["data_groups"] = {"base": data_groups["base"]}
    group.create_array(
        "VISIBILITY_CONTINUUM_SUBTRACTED",
        shape=(1,),
        dtype="f8",
        dimension_names=["uvw_label_unused"],
    )
    zarr.consolidate_metadata(store)
    with pytest.raises(ValueError, match="exists in the measurement set"):
        uv_continuum_subtraction(
            ps_store=store,
            spectral_window_name="spw_0",
            continuum_frequency_ranges=_continuum_ranges(),
            continuum_frequency_frame="lsrk",
        )
    with pytest.raises(ValueError, match="not those of the visibilities"):
        uv_continuum_subtraction(
            ps_store=store,
            spectral_window_name="spw_0",
            continuum_frequency_ranges=_continuum_ranges(),
            continuum_frequency_frame="lsrk",
            overwrite=True,
        )


@pytest.fixture(scope="module")
def topo_store(tmp_path_factory):
    folder = tmp_path_factory.mktemp("uv_continuum_topo")
    download(TOPO_PS, folder=str(folder))
    return str(folder / TOPO_PS)


def test_topo_measurement_set_with_lsrk_ranges(topo_store, tmp_path):
    store = str(tmp_path / TOPO_PS)
    shutil.copytree(topo_store, store)
    ps_xdt = open_processing_set(store)
    ms_name = next(iter(ps_xdt.children))
    lsrk = ps_xdt[ms_name].frequency.values * np.median(
        measurement_set_doppler_factors(ps_xdt[ms_name], "lsrk")
    )
    ranges = [[lsrk[0], lsrk[60]], [lsrk[90], lsrk[-1]]]
    result = uv_continuum_subtraction(
        ps_store=store,
        spectral_window_name="spw_0",
        continuum_frequency_ranges=ranges,
        continuum_frequency_frame="lsrk",
        n_mapping_parallelism={"time": 3},
    )
    channels = result["continuum_channels"].iloc[0]
    assert (
        channels["n_continuum_channels_some_time"]
        > channels["n_continuum_channels_every_time"]
    )
    full = xr.open_datatree(store)[ms_name].load()
    expected = full.copy()
    expected.dataset = full.to_dataset().drop_vars(
        ["VISIBILITY_CONTINUUM_SUBTRACTED", "FLAG_CONTINUUM_SUBTRACTED"]
    )
    data_groups = dict(expected.attrs["data_groups"])
    del data_groups["continuum_subtracted"]
    expected.attrs["data_groups"] = data_groups
    uv_continuum_subtraction_pf(expected, ranges, "lsrk", 1)
    np.testing.assert_array_equal(
        full.VISIBILITY_CONTINUUM_SUBTRACTED.values,
        expected.VISIBILITY_CONTINUUM_SUBTRACTED.values,
    )
    np.testing.assert_array_equal(
        full.FLAG_CONTINUUM_SUBTRACTED.values, expected.FLAG_CONTINUUM_SUBTRACTED.values
    )
    statistics = result["fit_statistics"]
    assert (statistics["n_flagged_without_continuum"] > 0).all()


def test_automatic_time_chunks(store):
    result = uv_continuum_subtraction(
        ps_store=store,
        spectral_window_name="spw_0",
        continuum_frequency_ranges=_continuum_ranges(),
        continuum_frequency_frame="lsrk",
        thread_info={"n_threads": 2, "memory_per_thread": 1.0},
    )
    # The simulated visibilities are one on-disk time chunk: every block would
    # read all of it, so one block.
    assert len(result["timing_node_tasks"]) == 1


@pytest.mark.parametrize(
    "n_time, n_target, max_task_length, input_time_chunk, expected",
    [
        (60, 2, 60, 10, 2),  # two blocks of three chunks
        (100, 8, 100, 10, 5),  # the most blocks of whole chunks up to 8
        (60, 16, 5, 10, 12),  # a chunk does not fit a task: two blocks per chunk
        (61, 4, 61, 1, 4),  # chunks of one time: the target
        (70, 8, 70, 70, 1),  # one chunk
        (60, 20, 3, 10, 20),  # the memory of a task sets the blocks
    ],
)
def test_default_time_blocks(
    n_time, n_target, max_task_length, input_time_chunk, expected
):
    assert _time_blocks(n_time, n_target, max_task_length, input_time_chunk) == expected


def test_output_chunking():
    sizes = {"baseline_id": 1225, "frequency": 3840, "polarization": 2}
    chunking = _output_chunking(10, sizes, 8)
    bytes_per_baseline = 3840 * 2 * 8
    assert chunking["baseline_id"] == OUTPUT_CHUNK_BYTES // bytes_per_baseline
    assert chunking["time"] == 1
    small = _output_chunking(
        12, {"baseline_id": 10, "frequency": 4, "polarization": 2}, 8
    )
    assert small == {"time": 12, "baseline_id": 10}
    huge = _output_chunking(
        4, {"baseline_id": 3, "frequency": 2**24, "polarization": 4}, 16
    )
    assert huge == {"time": 1, "baseline_id": 1}


def test_combine_results():
    def node(task_id, usage=None):
        result = {
            "timing_node_tasks": pd.DataFrame([{"task_id": task_id}]),
            "fit_statistics": pd.DataFrame([{"ms_name": "a", "n_fitted": 1}]),
            "spectrum_sums": {"a": {"sum_weight_in": np.ones(2)}},
        }
        if usage is not None:
            result["resource_usage"] = usage
        return result

    partial = combine_uv_continuum_subtraction_results(
        [node(0, {"cpu_percent": [1.0, 2.0], "sample_interval_seconds": 0.1}), None],
        {},
    )
    combined = combine_uv_continuum_subtraction_results([partial, node(1)], {})
    assert list(combined["timing_node_tasks"]["task_id"]) == [0, 1]
    assert combined["timing_node_tasks"]["cpu_percent"].iloc[0] == [1.0, 2.0]
    assert len(combined["fit_statistics"]) == 2
    np.testing.assert_array_equal(combined["spectrum_sums"]["a"]["sum_weight_in"], 2.0)
    empty = combine_uv_continuum_subtraction_results([None], {})
    assert empty["timing_node_tasks"].empty and empty["spectrum_sums"] == {}


def _failing_node_task(monkeypatch, failing_task_id):
    """Make the node task of ``failing_task_id`` raise, as a lost worker would."""
    module = importlib.import_module(
        "astroviper.node_tasks.visibility_manipulation.uv_continuum_subtraction"
    )
    node_task = module.uv_continuum_subtraction

    @functools.wraps(node_task)
    def failing(*args, **kwargs):
        if kwargs.get("task_id") == failing_task_id:
            raise OSError("injected failure")
        return node_task(*args, **kwargs)

    monkeypatch.setattr(module, "uv_continuum_subtraction", failing)


def _run(store, **overrides):
    return uv_continuum_subtraction(
        **{
            "ps_store": store,
            "spectral_window_name": "spw_0",
            "continuum_frequency_ranges": _continuum_ranges(),
            "continuum_frequency_frame": "lsrk",
            **overrides,
        }
    )


def test_a_failed_run_leaves_no_data_group_and_a_retry_needs_overwrite(
    store, monkeypatch
):
    with monkeypatch.context() as patch:
        _failing_node_task(patch, 1)
        with pytest.raises(OSError, match="injected failure"):
            _run(store, n_mapping_parallelism={"time": 5})
    ms_xdt = open_processing_set(store)["ALMA_spw_0"]
    assert "continuum_subtracted" not in ms_xdt.attrs["data_groups"]
    # The processing set's metadata lists the arrays the failed run made.
    assert "VISIBILITY_CONTINUUM_SUBTRACTED" in ms_xdt.data_vars
    with pytest.raises(ValueError, match="exists in the measurement set"):
        _run(store, n_mapping_parallelism={"time": 2})

    result = _run(store, n_mapping_parallelism={"time": 2}, overwrite=True)
    assert len(result["timing_node_tasks"]) == 2
    ms_xdt = open_processing_set(store)["ALMA_spw_0"]
    assert "continuum_subtracted" in ms_xdt.attrs["data_groups"]
    chunks = zarr.open_array(
        os.path.join(store, "ALMA_spw_0", "VISIBILITY_CONTINUUM_SUBTRACTED")
    ).chunks
    assert 3 % chunks[0] == 0  # divides the first block of 3 times
    written = ms_xdt.VISIBILITY_CONTINUUM_SUBTRACTED.values
    np.testing.assert_allclose(
        written, np.broadcast_to(_line()[:, np.newaxis], written.shape), atol=1e-12
    )


def test_a_failed_overwrite_unregisters_the_earlier_data_group(store, monkeypatch):
    _run(store, n_mapping_parallelism={"time": 2})
    with monkeypatch.context() as patch:
        _failing_node_task(patch, 0)
        with pytest.raises(OSError, match="injected failure"):
            _run(store, n_mapping_parallelism={"time": 5}, overwrite=True)
    ms_xdt = open_processing_set(store)["ALMA_spw_0"]
    assert "continuum_subtracted" not in ms_xdt.attrs["data_groups"]
    # The consolidated metadata matches the recreated arrays: they can be read.
    assert ms_xdt.VISIBILITY_CONTINUUM_SUBTRACTED.values.shape[0] == 5


def test_an_output_variable_of_another_role_is_refused(store):
    with pytest.raises(ValueError, match="is a variable of the data group 'base'"):
        _run(
            store,
            ms_data_group_out_modified={
                "correlated_data": "VISIBILITY_LINE",
                "flag": "WEIGHT",
            },
            overwrite=True,
        )
    ms_xdt = open_processing_set(store)["ALMA_spw_0"]
    assert ms_xdt.WEIGHT.dtype.kind == "f"
    assert "VISIBILITY_LINE" not in ms_xdt.data_vars


def test_several_measurement_sets_of_one_spectral_window(store):
    shutil.copytree(
        os.path.join(store, "ALMA_spw_0"), os.path.join(store, "ALMA_spw_0_b")
    )
    visibility = zarr.open_array(
        os.path.join(store, "ALMA_spw_0_b", "VISIBILITY"), mode="r+"
    )
    visibility[...] = 2.0 * visibility[...]
    zarr.consolidate_metadata(store)
    result = _run(store, n_mapping_parallelism={"time": 2})
    assert result["ms_names"] == ["ALMA_spw_0", "ALMA_spw_0_b"]
    assert sorted(set(result["fit_statistics"]["ms_name"])) == result["ms_names"]
    assert len(result["timing_node_tasks"]) == 4
    spectra = result["mean_spectra"]
    np.testing.assert_allclose(
        spectra["ALMA_spw_0_b"]["VISIBILITY_MEAN_INPUT"].values,
        2.0 * spectra["ALMA_spw_0"]["VISIBILITY_MEAN_INPUT"].values,
        rtol=1e-12,
    )
    ps_xdt = open_processing_set(store)
    for name, scale in (("ALMA_spw_0", 1.0), ("ALMA_spw_0_b", 2.0)):
        ms_xdt = ps_xdt[name]
        assert "continuum_subtracted" in ms_xdt.attrs["data_groups"]
        written = ms_xdt.VISIBILITY_CONTINUUM_SUBTRACTED.values
        np.testing.assert_allclose(
            written,
            scale * np.broadcast_to(_line()[:, np.newaxis], written.shape),
            atol=1e-12,
        )
    assert list(ps_xdt["ALMA_spw_1"].attrs["data_groups"]) == ["base"]


def test_an_extension_dataset_is_skipped(store):
    extension = zarr.open_group(os.path.join(store, "gains"), mode="w")
    extension.attrs["type"] = "extension:gains.quartical"
    zarr.consolidate_metadata(store)
    result = _run(store, n_mapping_parallelism={"time": 1})
    assert result["ms_names"] == ["ALMA_spw_0"]


def test_mean_spectra_of_data_with_nan_samples(topo_store, tmp_path):
    store = str(tmp_path / TOPO_PS)
    shutil.copytree(topo_store, store)
    ps_xdt = open_processing_set(store)
    ms_name = next(iter(ps_xdt.children))
    frequency = ps_xdt[ms_name].frequency.values
    result = uv_continuum_subtraction(
        ps_store=store,
        spectral_window_name="spw_0",
        continuum_frequency_ranges=[[frequency.min(), frequency.max()]],
        continuum_frequency_frame="TOPO",
        n_mapping_parallelism={"time": 1},
    )
    spectra = result["mean_spectra"][ms_name]
    # 35 % of the samples are NaN and unflagged; a channel without a valid
    # sample (unflagged, finite) has no mean.
    ms_xdt = xr.open_datatree(store)[ms_name]
    valid = (
        ~ms_xdt.FLAG.values
        & np.isfinite(ms_xdt.VISIBILITY.values)
        & np.isfinite(ms_xdt.WEIGHT.values)
    ).any(axis=(0, 1))
    assert valid.mean() > 0.9
    for name in ("VISIBILITY_MEAN_INPUT", "VISIBILITY_MEAN_OUTPUT"):
        np.testing.assert_array_equal(np.isfinite(spectra[name].values), valid)


def _two_measurement_sets(store):
    shutil.copytree(
        os.path.join(store, "ALMA_spw_0"), os.path.join(store, "ALMA_spw_0_b")
    )
    zarr.consolidate_metadata(store)


def test_a_failure_while_creating_the_outputs_leaves_consistent_metadata(
    store, monkeypatch
):
    _two_measurement_sets(store)
    _run(store, n_mapping_parallelism={"time": 2})
    io = importlib.import_module("astroviper.utils.io")
    create = io.create_empty_data_variables_on_disk

    def failing(zarr_store, *args, **kwargs):
        if zarr_store.endswith("ALMA_spw_0_b"):
            raise OSError("injected failure")
        return create(zarr_store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(io, "create_empty_data_variables_on_disk", failing)
        with pytest.raises(OSError, match="injected failure"):
            _run(store, n_mapping_parallelism={"time": 5}, overwrite=True)
    ps_xdt = open_processing_set(store)
    for name in ("ALMA_spw_0", "ALMA_spw_0_b"):
        assert "continuum_subtracted" not in ps_xdt[name].attrs["data_groups"]
    # The processing set's metadata matches the arrays: recreated in the first
    # MSv4, deleted in the second.
    assert ps_xdt["ALMA_spw_0"].VISIBILITY_CONTINUUM_SUBTRACTED.values.shape[0] == 5
    assert "VISIBILITY_CONTINUUM_SUBTRACTED" not in ps_xdt["ALMA_spw_0_b"].data_vars
    result = _run(store, n_mapping_parallelism={"time": 1}, overwrite=True)
    assert result["ms_names"] == ["ALMA_spw_0", "ALMA_spw_0_b"]


def test_every_measurement_set_keeps_its_consolidated_metadata(store):
    _run(store, n_mapping_parallelism={"time": 1})
    ms_path = os.path.join(store, "ALMA_spw_0")
    with open(os.path.join(ms_path, "zarr.json")) as metadata:
        consolidated = json.load(metadata)["consolidated_metadata"]["metadata"]
    assert "VISIBILITY_CONTINUUM_SUBTRACTED" in consolidated
    group = zarr.open_group(ms_path, mode="r")
    assert "continuum_subtracted" in group.attrs["data_groups"]
