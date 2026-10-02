"""The imaging node task images one frequency channel at a time: every channel
gets its own imaging cycle loop (so ``max_cycles`` counts per channel and a
converged channel stops cycling), the science function only ever sees the
visibility channels that map onto the image channel it is given, and the
finished channels are gathered into on-disk frequency chunks
(``image_chunking["frequency"]``) that are written -- and freed -- as soon as
they are complete, with the timing frames and imaging dicts combined."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.node_tasks.imaging.image_cube_single_field import (
    _chunk_task_coords,
    _combine_channel_timing_frames,
    _ImageChunkAccumulator,
    _select_processing_set_channel,
    _shift_imaging_dict_channels,
    _visibility_to_image_frequency_maps,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    ImagingDict,
    Key,
)

VIS_FREQUENCIES = np.array([1.0e9, 1.1e9, 1.2e9])


def _fake_measurement_set(frequencies, seed=0):
    rng = np.random.default_rng(seed)
    n_time, n_baseline, n_pol = 2, 3, 2
    ds = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                rng.normal(size=(n_time, n_baseline, len(frequencies), n_pol)) + 0j,
            ),
            "WEIGHT": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.ones((n_time, n_baseline, len(frequencies), n_pol)),
            ),
        },
        coords={
            "time": np.arange(n_time, dtype=float),
            "baseline_id": np.arange(n_baseline),
            "frequency": np.asarray(frequencies, dtype=float),
            "polarization": ["XX", "YY"],
        },
        attrs={
            "data_groups": {
                "base": {"correlated_data": "VISIBILITY", "weight": "WEIGHT"}
            }
        },
    )
    return ds


def _fake_processing_set(ms_frequencies):
    """A processing-set DataTree (what ``load_processing_set`` returns)."""
    return xr.DataTree.from_dict(
        {
            name: _fake_measurement_set(freqs, seed=i)
            for i, (name, freqs) in enumerate(ms_frequencies.items())
        }
    )


def _task_inputs(tmp_path, n_chan=3, data_selection=None, **overrides):
    image_params = {
        "phase_direction": np.array([1.0, 0.5]),
        "image_size": [4, 4],
        "cell_size": np.array([-1.0, 1.0]) * 4.85e-6,
        "time_coords": [0],
        "polarization_coords": ["I", "Q"],
        "fft_padding": 1.2,
    }
    inputs = dict(
        image_params=image_params,
        imaging_weights_params={"weighting": "natural"},
        iteration_control_params={"max_iter": 10, "max_cycles": 3},
        task_coords={"frequency": {"data": VIS_FREQUENCIES[:n_chan]}},
        data_selection=data_selection or {"ms_a": {"frequency": slice(0, n_chan)}},
        image_store=str(tmp_path / "img.zarr"),
        input_data_store=str(tmp_path / "unused.ps.zarr"),
        image_data_variables_keep=["sky_residual"],
        graph_mode=False,  # whole task written with to_zarr
        task_id=3,
    )
    inputs.update(overrides)
    return inputs


class _FakeScience:
    """Stand-in for the science function: records what it is handed and
    returns a filled one-channel image, a timing row and an imaging dict."""

    def __init__(self, cycles_per_channel):
        self.cycles_per_channel = cycles_per_channel
        self.calls = []

    def __call__(
        self,
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        iteration_control_params,
        **kwargs,
    ):
        vis_frequencies = {
            name: np.array(ms_xdt.frequency.values) for name, ms_xdt in ps_xdt.items()
        }
        self.calls.append(
            {
                "n_chan": img_xds.sizes["frequency"],
                "frequency": float(img_xds.frequency.values[0]),
                "vis_frequencies": vis_frequencies,
                "kwargs": kwargs,
            }
        )
        # The science function registers data groups on the objects it is
        # given -- these must not leak between channels or into the chunk.
        for ms_xdt in ps_xdt.values():
            ms_xdt.attrs["data_groups"]["model"] = {
                "correlated_data": "VISIBILITY_MODEL"
            }
        img_xds.attrs["data_groups"]["residual"] = {"sky": "SKY_RESIDUAL"}

        chan = len(self.calls) - 1
        # the science function returns the Stokes planes of the two parallel hands
        shape = (img_xds.sizes["time"], 1, 2, 4, 4)
        img_xds = img_xds.assign_coords(polarization=["I", "Q"])
        img_xds["SKY_RESIDUAL"] = (
            ("time", "frequency", "polarization", "l", "m"),
            np.full(shape, float(chan), dtype=np.float32),
        )
        img_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = (
            ("time", "frequency", "polarization", "beam_params_label"),
            np.full((shape[0], 1, 2, 3), 10.0 * chan),
        )
        img_xds["STATIC"] = (("l", "m"), np.arange(16.0).reshape(4, 4))
        n_cycles = self.cycles_per_channel[chan]
        timing = pd.DataFrame(
            {
                "T_prep": [1.0],
                "T_residual_update": [float(n_cycles)],
                "task_id": [kwargs["task_id"]],
                "n_channels": [1],
                "n_cycles": [n_cycles],
            }
        )
        imaging_dict = ImagingDict()
        imaging_dict.add(
            {"iter_done": 5 * (chan + 1), "peakres": 0.1}, time=0, pol=0, chan=0
        )
        return img_xds, timing, imaging_dict


@pytest.fixture
def fake_science(monkeypatch):
    import astroviper.processing_functions.imaging as pf_imaging

    fake = _FakeScience(cycles_per_channel=[1, 3, 2])
    monkeypatch.setattr(pf_imaging, "image_cube_single_field", fake)
    return fake


def test_node_task_images_one_channel_at_a_time(tmp_path, fake_science):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    ps_xdt = _fake_processing_set({"ms_a": VIS_FREQUENCIES})
    # The node task releases the task-owned tree (severs its children) before
    # returning, so keep a handle on the measurement-set node itself.
    ms_a = ps_xdt["ms_a"]
    original_groups = copy.deepcopy(ms_a.attrs["data_groups"])

    result = image_cube_single_field(**_task_inputs(tmp_path), input_data=ps_xdt)

    # One science call per image channel, in order, each with exactly the
    # visibility channel that maps onto it.
    assert [c["n_chan"] for c in fake_science.calls] == [1, 1, 1]
    assert [c["frequency"] for c in fake_science.calls] == list(VIS_FREQUENCIES)
    for call, freq in zip(fake_science.calls, VIS_FREQUENCIES, strict=True):
        assert call["vis_frequencies"] == {"ms_a": pytest.approx([freq])}
        assert call["kwargs"]["task_id"] == 3
    # Data groups registered by the science function never reach the chunk.
    assert ms_a.attrs["data_groups"] == original_groups

    # The written image is the chunk cube with the channels in place.
    img = xr.open_zarr(str(tmp_path / "img.zarr")).load()
    assert img.sizes["frequency"] == 3
    np.testing.assert_array_equal(img.frequency.values, VIS_FREQUENCIES)
    assert img["SKY_RESIDUAL"].dtype == np.float32
    for chan in range(3):
        assert (img["SKY_RESIDUAL"].isel(frequency=chan).values == chan).all()
        assert (
            img["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].isel(frequency=chan).values
            == 10.0 * chan
        ).all()
    np.testing.assert_array_equal(img["STATIC"].values, np.arange(16.0).reshape(4, 4))
    assert list(img.polarization.values) == ["I", "Q"]

    # Timing: T_* summed, n_cycles = max over channels, n_cycles_total = sum.
    timing = result["timing_node_tasks"]
    assert len(timing) == 1
    assert timing["n_channels"].iloc[0] == 3
    assert timing["n_cycles"].iloc[0] == 3
    assert timing["n_cycles_total"].iloc[0] == 6
    assert timing["T_prep"].iloc[0] == pytest.approx(3.0)
    assert timing["T_residual_update"].iloc[0] == pytest.approx(6.0)
    assert timing["task_id"].iloc[0] == 3
    assert "T_channel_bookkeeping" in timing.columns
    assert "task_failed_phase" not in timing.columns

    # Imaging dict: one entry per channel, on chunk-local (here global) channels.
    deconv = result["deconvolution"]
    assert sorted(deconv.data) == [Key(time=0, pol=0, chan=c) for c in range(3)]
    assert deconv.data[Key(0, 0, 1)]["iter_done"] == [10]

    # Per-plane image statistics cover the whole chunk.
    stats = result["image_statistics"]["sky_residual"]
    assert stats.sizes["frequency"] == 3


def test_imaging_dict_channels_shift_onto_global_numbers(tmp_path, fake_science):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    ps_xdt = _fake_processing_set({"ms_a": VIS_FREQUENCIES})
    result = image_cube_single_field(
        **_task_inputs(tmp_path, data_selection={"ms_a": {"frequency": slice(5, 8)}}),
        input_data=ps_xdt,
    )
    assert sorted(key.chan for key in result["deconvolution"].data) == [5, 6, 7]


@pytest.mark.parametrize("image_chunking", [{"frequency": 1}, None])
def test_channel_measurement_sets_die_before_the_statistics(
    tmp_path, monkeypatch, fake_science, image_chunking
):
    """The measurement sets the science function was handed for a channel
    (which then hold its model and residual visibilities and imaging
    weights) are freed by reference counting as soon as the channel is
    imaged, before the chunk's statistics and write."""
    import gc
    import weakref

    import astroviper.processing_functions.image_analysis.plane_statistics as ps_mod
    import astroviper.processing_functions.imaging as pf_imaging
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    handed = []

    def science_keeping_weakrefs(ps_xdt, *args, **kwargs):
        handed.extend(weakref.ref(ms_xdt) for ms_xdt in ps_xdt.values())
        return fake_science(ps_xdt, *args, **kwargs)

    monkeypatch.setattr(pf_imaging, "image_cube_single_field", science_keeping_weakrefs)
    statistics = ps_mod.calculate_plane_statistics
    alive_at_statistics = []

    def statistics_checking(*args, **kwargs):
        alive_at_statistics.append(sum(ref() is not None for ref in handed))
        return statistics(*args, **kwargs)

    monkeypatch.setattr(ps_mod, "calculate_plane_statistics", statistics_checking)
    overrides = {}
    if image_chunking is not None:
        overrides = dict(
            image_store=_make_store(tmp_path, 3, [[0, 1, 2]], image_chunking),
            graph_mode=True,
            image_chunking=image_chunking,
        )
    inputs = _task_inputs(tmp_path, **overrides)
    ps_xdt = _fake_processing_set({"ms_a": VIS_FREQUENCIES})
    gc_was_enabled = gc.isenabled()
    gc.disable()  # reference counting alone
    try:
        image_cube_single_field(**inputs, input_data=ps_xdt)
    finally:
        if gc_was_enabled:
            gc.enable()

    assert len(handed) == 3
    assert alive_at_statistics == [0] * len(alive_at_statistics)
    assert len(alive_at_statistics) == (3 if image_chunking else 1)


def test_channel_without_visibilities_leaves_the_loaded_chunk_untouched(
    tmp_path, monkeypatch, fake_science
):
    """An image channel no visibility channel maps onto is imaged from the
    whole loaded chunk, handed over as views with their own attrs: the
    variables and data groups the science function registers stay off the
    loaded tree, and die by reference counting before the statistics."""
    import gc
    import weakref

    import astroviper.processing_functions.image_analysis.plane_statistics as ps_mod
    import astroviper.processing_functions.imaging as pf_imaging
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    handed = []

    def science_registering(ps_xdt, *args, **kwargs):
        for ms_xdt in ps_xdt.values():
            # what the residual update does: register model visibilities
            ms_xdt["VISIBILITY_MODEL"] = ms_xdt["VISIBILITY"] * 0
            handed.append(weakref.ref(ms_xdt))
        return fake_science(ps_xdt, *args, **kwargs)

    monkeypatch.setattr(pf_imaging, "image_cube_single_field", science_registering)
    statistics = ps_mod.calculate_plane_statistics
    alive_at_statistics = []

    def statistics_checking(*args, **kwargs):
        alive_at_statistics.append(sum(ref() is not None for ref in handed))
        return statistics(*args, **kwargs)

    monkeypatch.setattr(ps_mod, "calculate_plane_statistics", statistics_checking)
    # visibility channels on the first and the last image channel only
    ps_xdt = _fake_processing_set({"ms_a": VIS_FREQUENCIES[[0, 2]]})
    ms_a = ps_xdt["ms_a"]
    original_groups = copy.deepcopy(ms_a.attrs["data_groups"])
    gc_was_enabled = gc.isenabled()
    gc.disable()  # reference counting alone
    try:
        image_cube_single_field(**_task_inputs(tmp_path), input_data=ps_xdt)
    finally:
        if gc_was_enabled:
            gc.enable()

    # channel 1 was imaged from the whole chunk, the others from their own
    assert [c["vis_frequencies"]["ms_a"].tolist() for c in fake_science.calls] == [
        [VIS_FREQUENCIES[0]],
        VIS_FREQUENCIES[[0, 2]].tolist(),
        [VIS_FREQUENCIES[2]],
    ]
    assert "VISIBILITY_MODEL" not in ms_a
    assert ms_a.attrs["data_groups"] == original_groups
    assert len(handed) == 3
    assert alive_at_statistics == [0]


# --------------------------------------------------------------------------- #
# Chunk-wise writes: a chunk is written as soon as its channels are imaged
# --------------------------------------------------------------------------- #
def _make_store(tmp_path, n_total, task_chunks, image_chunking):
    """A pre-created image store (as the driver makes it) holding
    ``sky_residual`` for ``n_total`` channels, 4x4 pixels, Stokes I and Q."""
    import zarr

    from astroviper.utils.io import create_empty_data_variables_on_disk

    store = str(tmp_path / "chunked.img.zarr")
    zarr.open_group(store, mode="w")
    create_empty_data_variables_on_disk(
        store,
        ["sky_residual"],
        shape_dict={"time": 1, "frequency": n_total, "polarization": 2, "l": 4, "m": 4},
        parallel_coords={"frequency": {"data_chunks": task_chunks}},
        compressor=None,
        double_precision=False,
        data_variable_definitions="imaging",
        image_chunking=image_chunking,
    )
    return store


@pytest.fixture
def write_spy(monkeypatch):
    """Record every chunk write's global frequency slice; optionally fail one."""
    import astroviper.utils.io as io_module

    original = io_module.write_result_chunk_to_disk_using_zarr
    spy = {"slices": [], "fail_on_call": None}

    def wrapper(image_store, image_data_variables_keep, task_coords, img_xds):
        call_index = len(spy["slices"])
        spy["slices"].append(task_coords["frequency"]["slice"])
        if spy["fail_on_call"] == call_index:
            raise OSError("simulated Lustre write failure")
        original(image_store, image_data_variables_keep, task_coords, img_xds)

    monkeypatch.setattr(io_module, "write_result_chunk_to_disk_using_zarr", wrapper)
    return spy


@pytest.mark.parametrize(
    "image_chunking, expected_slices",
    [
        ({"frequency": 1}, [slice(0, 1), slice(1, 2), slice(2, 3)]),
        ({"frequency": 2}, [slice(0, 2), slice(2, 3)]),
        (None, [slice(0, 3)]),
    ],
    ids=["one_channel_chunks", "two_channel_chunks", "whole_task"],
)
def test_chunks_are_written_as_soon_as_complete(
    tmp_path, fake_science, write_spy, image_chunking, expected_slices
):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    store = _make_store(tmp_path, 3, [[0, 1, 2]], image_chunking)
    ps_xdt = _fake_processing_set({"ms_a": VIS_FREQUENCIES})
    result = image_cube_single_field(
        **_task_inputs(
            tmp_path, image_store=store, graph_mode=True, image_chunking=image_chunking
        ),
        input_data=ps_xdt,
    )
    assert write_spy["slices"] == expected_slices
    img = xr.open_zarr(store).load()
    for chan in range(3):
        assert (img["SKY_RESIDUAL"].isel(frequency=chan).values == chan).all()
    timing = result["timing_node_tasks"]
    assert "task_failed_phase" not in timing.columns
    assert timing["T_write"].iloc[0] >= 0.0
    # Statistics are taken per chunk and concatenated along frequency.
    stats = result["image_statistics"]["sky_residual"]
    assert stats.sizes["frequency"] == 3
    # one value per channel and Stokes plane (I, Q)
    assert stats.sizes["polarization"] == 2
    for plane in range(2):
        np.testing.assert_allclose(
            stats["max"].isel(polarization=plane).values.ravel(), [0.0, 1.0, 2.0]
        )


def test_failed_chunk_write_is_skipped_and_logged(tmp_path, fake_science, write_spy):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    store = _make_store(tmp_path, 3, [[0, 1, 2]], {"frequency": 1})
    write_spy["fail_on_call"] = 1  # the second chunk (channel 1) fails
    ps_xdt = _fake_processing_set({"ms_a": VIS_FREQUENCIES})
    result = image_cube_single_field(
        **_task_inputs(
            tmp_path,
            image_store=store,
            graph_mode=True,
            image_chunking={"frequency": 1},
        ),
        input_data=ps_xdt,
    )
    # All three chunks were attempted; the other two landed on disk.
    assert write_spy["slices"] == [slice(0, 1), slice(1, 2), slice(2, 3)]
    img = xr.open_zarr(store).load()
    assert (img["SKY_RESIDUAL"].isel(frequency=0).values == 0).all()
    assert np.isnan(img["SKY_RESIDUAL"].isel(frequency=1).values).all()
    assert (img["SKY_RESIDUAL"].isel(frequency=2).values == 2).all()
    row = result["timing_node_tasks"].iloc[0]
    assert row["task_failed_phase"] == "write"
    assert "simulated Lustre write failure" in row["task_error"]
    assert row["failed_channel_start"] == 1
    assert row["failed_n_channels"] == 1
    assert row["n_failed_chunks"] == 1
    assert row["n_channels"] == 3  # the task's channel count is untouched
    # Statistics describe every channel, written or not.
    assert result["image_statistics"]["sky_residual"].sizes["frequency"] == 3


def test_chunks_land_at_the_task_global_channels(tmp_path, fake_science, write_spy):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    # An 8-channel store; this task owns global channels [5, 8).
    store = _make_store(tmp_path, 8, [[0, 1, 2, 3, 4], [5, 6, 7]], {"frequency": 1})
    ps_xdt = _fake_processing_set({"ms_a": VIS_FREQUENCIES})
    result = image_cube_single_field(
        **_task_inputs(
            tmp_path,
            data_selection={"ms_a": {"frequency": slice(5, 8)}},
            image_store=store,
            graph_mode=True,
            image_chunking={"frequency": 1},
        ),
        input_data=ps_xdt,
    )
    assert write_spy["slices"] == [slice(5, 6), slice(6, 7), slice(7, 8)]
    img = xr.open_zarr(store).load()
    for local, chan in enumerate(range(5, 8)):
        assert (img["SKY_RESIDUAL"].isel(frequency=chan).values == local).all()
    assert np.isnan(img["SKY_RESIDUAL"].isel(frequency=slice(0, 5)).values).all()
    assert sorted(key.chan for key in result["deconvolution"].data) == [5, 6, 7]


def test_chunk_task_coords():
    task_coords = {
        "frequency": {"data": np.array([1.0, 2.0, 3.0, 4.0]), "slice": slice(10, 14)},
        "other": {"data": [0]},
    }
    chunk = _chunk_task_coords(task_coords, None, 1, 3)
    assert chunk["frequency"]["slice"] == slice(11, 13)
    np.testing.assert_array_equal(chunk["frequency"]["data"], [2.0, 3.0])
    assert chunk["other"] is task_coords["other"]
    assert task_coords["frequency"]["slice"] == slice(10, 14)  # input untouched
    # Without a task slice the data_selection offset places the chunk.
    chunk = _chunk_task_coords(
        {"frequency": {"data": np.array([1.0, 2.0])}},
        {"ms": {"frequency": slice(7, 9)}},
        0,
        2,
    )
    assert chunk["frequency"]["slice"] == slice(7, 9)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_select_processing_set_channel_maps_visibility_channels():
    # Two visibility channels per image channel in ms_a; ms_b only covers the
    # first two image channels.
    image_frequencies = np.array([1.0e9, 1.1e9, 1.2e9])
    ps_xdt = _fake_processing_set(
        {
            "ms_a": np.array([0.98e9, 1.02e9, 1.08e9, 1.12e9, 1.18e9, 1.22e9]),
            "ms_b": np.array([1.0e9, 1.1e9]),
        }
    )
    img_xds = xr.Dataset(coords={"frequency": image_frequencies})
    frequency_maps = _visibility_to_image_frequency_maps(ps_xdt, img_xds)
    np.testing.assert_array_equal(frequency_maps["ms_a"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(frequency_maps["ms_b"], [0, 1])

    chan1 = _select_processing_set_channel(ps_xdt, frequency_maps, 1)
    assert sorted(chan1) == ["ms_a", "ms_b"]
    np.testing.assert_array_equal(chan1["ms_a"].frequency.values, [1.08e9, 1.12e9])
    np.testing.assert_array_equal(chan1["ms_b"].frequency.values, [1.1e9])
    # Zero-copy: the sliced visibilities are views of the loaded arrays.
    assert np.shares_memory(
        chan1["ms_a"]["VISIBILITY"].values, ps_xdt["ms_a"]["VISIBILITY"].values
    )
    # ... but the attrs are the slice's own.
    chan1["ms_a"].attrs["data_groups"]["residual"] = {}
    assert "residual" not in ps_xdt["ms_a"].attrs["data_groups"]

    chan2 = _select_processing_set_channel(ps_xdt, frequency_maps, 2)
    assert sorted(chan2) == ["ms_a"]
    # An image channel no measurement set maps onto selects nothing.
    ps_b_only = _fake_processing_set({"ms_b": np.array([1.0e9, 1.1e9])})
    maps_b_only = _visibility_to_image_frequency_maps(ps_b_only, img_xds)
    assert _select_processing_set_channel(ps_b_only, maps_b_only, 2) is None


def _channel_result(chunk, chan, value=None):
    value = chan if value is None else value
    return xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "l", "m"),
                np.full((1, 1, 2, 2, 2), value, dtype=np.float32),
            ),
            "STATIC": (("l", "m"), np.ones((2, 2))),
        },
        coords={
            "time": [0.0],
            "frequency": chunk.frequency.values[chan : chan + 1],
            "velocity": ("frequency", chunk.velocity.values[chan : chan + 1]),
            "polarization": ["I", "Q"],
            "l": [0, 1],
            "m": [0, 1],
        },
        attrs={"data_groups": {"residual": {"sky": "SKY"}}, "n": chan},
    )


def _chunk_image():
    return xr.Dataset(
        coords={
            "frequency": np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9]),
            "velocity": ("frequency", np.array([40.0, 30.0, 20.0, 10.0])),
        }
    )


def test_chunk_accumulator_places_channels_and_keeps_static_parts():
    chunk = _chunk_image()
    accumulator = _ImageChunkAccumulator(chunk, start=1, n_channels=3)
    assert (accumulator.start, accumulator.stop) == (1, 4)
    for chan in (1, 2, 3):
        assert not accumulator.complete
        accumulator.insert(_channel_result(chunk, chan), chan)
    assert accumulator.complete
    out = accumulator.assemble()
    assert dict(out.sizes) == {
        "time": 1,
        "frequency": 3,
        "polarization": 2,
        "l": 2,
        "m": 2,
    }
    np.testing.assert_array_equal(out.frequency.values, chunk.frequency.values[1:4])
    np.testing.assert_array_equal(out.velocity.values, chunk.velocity.values[1:4])
    assert out["SKY"].dtype == np.float32
    for local, chan in enumerate((1, 2, 3)):
        assert (out["SKY"].isel(frequency=local).values == chan).all()
    np.testing.assert_array_equal(out["STATIC"].values, np.ones((2, 2)))
    assert out.attrs["data_groups"] == {"residual": {"sky": "SKY"}}
    assert out.attrs["n"] == 1  # attrs come from the chunk's first channel


def test_chunk_accumulator_single_channel_is_the_result_itself():
    chunk = _chunk_image()
    accumulator = _ImageChunkAccumulator(chunk, start=2, n_channels=1)
    result = _channel_result(chunk, 2)
    accumulator.insert(result, 2)
    assert accumulator.complete
    assert accumulator.assemble() is result  # no copy, no allocation


def test_chunk_accumulator_rejects_inconsistent_channels():
    chunk = _chunk_image()
    accumulator = _ImageChunkAccumulator(chunk, start=0, n_channels=2)
    accumulator.insert(_channel_result(chunk, 0), 0)
    with pytest.raises(RuntimeError, match="has 1 of 2 channels"):
        accumulator.assemble()
    with pytest.raises(ValueError, match="out of order"):
        accumulator.insert(_channel_result(chunk, 1), 3)
    with pytest.raises(RuntimeError, match="missing variable"):
        accumulator.insert(_channel_result(chunk, 1).drop_vars("SKY"), 1)
    with pytest.raises(ValueError, match="one-channel"):
        accumulator.insert(
            xr.Dataset(
                {"SKY": (("frequency", "l"), np.zeros((2, 2)))},
                coords={"frequency": chunk.frequency.values[:2], "l": [0, 1]},
            ),
            1,
        )


def test_combine_channel_timing_frames():
    frames = [
        pd.DataFrame(
            {"T_prep": [1.0], "task_id": [4], "n_channels": [1], "n_cycles": [2]}
        ),
        pd.DataFrame(
            {"T_prep": [2.0], "task_id": [4], "n_channels": [1], "n_cycles": [5]}
        ),
        pd.DataFrame(
            {
                "T_prep": [3.0],
                "T_restore": [0.5],
                "task_id": [4],
                "n_channels": [1],
                "n_cycles": [1],
            }
        ),
    ]
    combined = _combine_channel_timing_frames(frames)
    assert len(combined) == 1
    row = combined.iloc[0]
    assert row["T_prep"] == pytest.approx(6.0)
    assert row["T_restore"] == pytest.approx(0.5)
    assert row["task_id"] == 4
    assert row["n_channels"] == 3
    assert row["n_cycles"] == 5
    assert row["n_cycles_total"] == 8


def test_shift_imaging_dict_channels():
    imaging_dict = ImagingDict()
    imaging_dict.add({"iter_done": 3}, time=0, pol=1, chan=0)
    assert _shift_imaging_dict_channels(imaging_dict, 0) is imaging_dict
    shifted = _shift_imaging_dict_channels(imaging_dict, 4)
    assert list(shifted.data) == [Key(time=0, pol=1, chan=4)]
    assert shifted.data[Key(0, 1, 4)]["iter_done"] == [3]
