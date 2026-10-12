"""The imaging node task must skip-and-log a failed read/write instead of
raising -- one bad chunk previously tore down entire multi-node runs (the
2019.1.01463.S 80000-channel benchmark died on a single unreadable shard).

The one exception is a Measurement Set v2 that changed after the distributed
application opened it (``xradio.measurement_set.MSv2ChangedError``): the graph
no longer describes it, so the node task raises and the run aborts. The tests
of the ``lazy_input_data`` input (a Measurement Set v2 read through XRADIO's
``xradio_msv2`` engine) are here too: what it loads, a failed read, a changed
Measurement Set (simulated, and gated by the engine on the generated
``imageable_msv2`` of ``tests/unit/conftest.py``).
"""

from __future__ import annotations

import os
import pickle
import shutil

import numpy as np
import pytest
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from astroviper.node_tasks.imaging.utils import (
    add_lazy_input_data,
    msv2_engine_available,
)

_ENGINE_AVAILABLE, _ENGINE_REASON = msv2_engine_available()
requires_msv2_engine = pytest.mark.skipif(
    not _ENGINE_AVAILABLE, reason=_ENGINE_REASON or "needs XRADIO with open_msv2"
)


def _minimal_task_inputs(tmp_path):
    """Just enough to reach (and fail) the load phase: the input store does not
    exist, so the read raises after the empty per-chunk image is built."""
    image_params = {
        "phase_direction": np.array([1.0, 0.5]),
        "image_size": [4, 4],
        "cell_size": np.array([-1.0, 1.0]) * 4.85e-6,
        "time_coords": [0],
        "polarization_coords": ["I", "Q"],
        "fft_padding": 1.2,
    }
    task_coords = {"frequency": {"data": np.array([1.0e9, 1.1e9])}}
    data_selection = {"ms_a": {"frequency": slice(5, 7)}}
    return dict(
        image_params=image_params,
        imaging_weights_params={"weighting": "natural"},
        iteration_control_params={"max_iter": 0},
        task_coords=task_coords,
        data_selection=data_selection,
        image_store=str(tmp_path / "img.zarr"),
        input_data_store=str(tmp_path / "does_not_exist.ps.zarr"),
        skunk_works=True,
        task_id=7,
    )


def test_load_failure_returns_marked_row_instead_of_raising(tmp_path):
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    result = image_cube_single_field(**_minimal_task_inputs(tmp_path))
    df = result["timing_node_tasks"]
    assert list(df["task_failed_phase"]) == ["load"]
    assert df["failed_channel_start"].iloc[0] == 5  # from the frequency slice
    assert df["n_channels"].iloc[0] == 2
    assert df["task_id"].iloc[0] == 7
    assert df["task_error"].iloc[0]  # carries the original exception repr
    assert "T_image_cube_task" in df.columns
    assert result["deconvolution"].data == {}  # empty ImagingDict merges cleanly


def test_failed_rows_merge_through_the_standard_reduce(tmp_path):
    """Failed-task results must flow through the production reducer unchanged
    (the shape a mixed failed/successful run reduces through)."""
    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        combine_return_data_frames,
    )
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    r1 = image_cube_single_field(**_minimal_task_inputs(tmp_path))
    r2 = image_cube_single_field(**_minimal_task_inputs(tmp_path))
    combined = combine_return_data_frames([r1, r2], {})
    assert len(combined["timing_node_tasks"]) == 2
    assert list(combined["timing_node_tasks"]["task_failed_phase"]) == ["load", "load"]
    assert combined["deconvolution"].data == {}


def test_reduce_records_timing_provenance(tmp_path):
    """Every reduce call appends one timing record (start/end/host/n_inputs)
    and pools its children's records, so the final result carries one record
    per reduce node of the whole tree -- what the task-stream analysis draws."""
    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        combine_return_data_frames,
    )
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    leaves = [
        image_cube_single_field(**_minimal_task_inputs(tmp_path)) for _ in range(4)
    ]
    assert all("timing_reduce_nodes" not in r for r in leaves)

    partial_a = combine_return_data_frames(leaves[:2], {})
    partial_b = combine_return_data_frames(leaves[2:], {})
    assert len(partial_a["timing_reduce_nodes"]) == 1
    rec = partial_a["timing_reduce_nodes"][0]
    assert rec["n_inputs"] == 2 and rec["n_rows_out"] == 2
    assert rec["end_unixtime"] >= rec["start_unixtime"]
    assert rec["hostname"]

    final = combine_return_data_frames([partial_a, partial_b], {})
    # 2 child records + this call's own = one record per reduce node.
    assert len(final["timing_reduce_nodes"]) == 3
    assert len(final["timing_node_tasks"]) == 4


# --------------------------------------------------------------------------- #
# lazy_input_data: a Measurement Set v2 read through XRADIO's engine
# --------------------------------------------------------------------------- #
VISIBILITY_DIMS = ("time", "baseline_id", "frequency", "polarization")
# 8 channels; the task of _minimal_task_inputs selects channels 5 and 6, whose
# frequencies are the task's (1.0 and 1.1 GHz).
MS_FREQUENCIES = 0.5e9 + 0.1e9 * np.arange(8)


class _LazyArray(BackendArray):
    """A lazily indexed array (xarray's backend API, as the engine's arrays):
    a read returns the selected ``values``, or raises ``error``."""

    def __init__(self, values, error=None):
        self.shape = values.shape
        self.dtype = values.dtype
        self.values = values
        self.error = error

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._read
        )

    def _read(self, key):
        if self.error is not None:
            raise self.error
        return self.values[key]


def _lazy_measurement_set(error=None):
    """An MSv4-like node with lazily indexed variables (``error``: every read
    raises it) and the ``base`` data group."""
    rng = np.random.default_rng(1)
    antenna = np.array(["ea01", "ea02", "ea03"])
    antenna1, antenna2 = np.triu_indices(antenna.size, 1)
    shape = (2, antenna1.size, MS_FREQUENCIES.size, 2)

    def lazy(dims, values):
        return xr.Variable(dims, indexing.LazilyIndexedArray(_LazyArray(values, error)))

    return xr.Dataset(
        {
            "VISIBILITY": lazy(
                VISIBILITY_DIMS, rng.standard_normal(shape) + 1j * rng.random(shape)
            ),
            "FLAG": lazy(VISIBILITY_DIMS, rng.random(shape) < 0.1),
            "WEIGHT": lazy(VISIBILITY_DIMS, rng.uniform(0.5, 2, shape)),
            "UVW": lazy(
                ("time", "baseline_id", "uvw_label"),
                rng.uniform(-100, 100, (shape[0], shape[1], 3)),
            ),
        },
        coords={
            "time": [0.0, 1.0],
            "baseline_id": np.arange(antenna1.size),
            "frequency": MS_FREQUENCIES,
            "polarization": ["XX", "YY"],
            "uvw_label": ["u", "v", "w"],
            "baseline_antenna1_name": ("baseline_id", antenna[antenna1]),
            "baseline_antenna2_name": ("baseline_id", antenna[antenna2]),
        },
        attrs={
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                }
            }
        },
    )


def _lazy_input_data(ms_xds, data_selection):
    """One task's ``lazy_input_data``, made as the distributed application makes
    it, after a pickle round trip (as a worker receives it)."""
    mapping = {0: {"data_selection": data_selection}}
    add_lazy_input_data(xr.DataTree.from_dict({"ms_a": ms_xds}), mapping, "base")
    return pickle.loads(pickle.dumps(mapping[0]["lazy_input_data"]))


class _ChangedError(RuntimeError):
    """A stand-in for ``xradio.measurement_set.MSv2ChangedError``."""


class _Imaged(Exception):
    """Raised by a stand-in science function, carrying what it was given."""

    def __init__(self, ps_chan):
        super().__init__("imaged")
        self.ps_chan = ps_chan


def test_lazy_input_data_is_read_for_the_science(tmp_path, monkeypatch):
    """The node task reads ``lazy_input_data`` (neither the missing input store
    nor ``skunk_works`` is consulted) and hands the science function the
    loaded values: NumPy, writeable, with the baseline names rebuilt."""
    import astroviper.processing_functions.imaging as pf_imaging
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    def science(ps_chan, *args, **kwargs):
        raise _Imaged(ps_chan)

    monkeypatch.setattr(pf_imaging, "image_cube_single_field", science)
    inputs = _minimal_task_inputs(tmp_path)
    inputs["skunk_works"] = False
    ms_xds = _lazy_measurement_set()
    lazy_input_data = _lazy_input_data(ms_xds, inputs["data_selection"])

    with pytest.raises(_Imaged) as imaged:
        image_cube_single_field(**inputs, lazy_input_data=lazy_input_data)

    # The first image channel (1.0 GHz) is the MS's channel 5.
    ms_chan = imaged.value.ps_chan["ms_a"].to_dataset()
    expected = ms_xds.isel(frequency=slice(5, 6)).load()
    assert sorted(ms_chan.data_vars) == ["FLAG", "UVW", "VISIBILITY", "WEIGHT"]
    for name in ms_chan.data_vars:
        values = ms_chan[name].values
        assert isinstance(values, np.ndarray)
        np.testing.assert_array_equal(values, expected[name].values, err_msg=name)
    assert ms_chan["WEIGHT"].values.flags.writeable
    for name in ("frequency", "polarization", "baseline_antenna1_name"):
        xr.testing.assert_identical(ms_chan[name].variable, expected[name].variable)
    xr.testing.assert_identical(
        ms_chan["baseline_antenna2_name"].variable,
        expected["baseline_antenna2_name"].variable,
    )
    assert list(ms_chan.attrs["data_groups"]) == ["base"]
    # The task's lazy selection itself was not read into memory.
    assert isinstance(
        lazy_input_data["ms_a"]["VISIBILITY"].variable._data,
        indexing.LazilyIndexedArray,
    )


def test_lazy_input_data_read_failure_is_skipped(tmp_path):
    """A read error of ``lazy_input_data`` is skipped and logged like any other
    load failure."""
    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    inputs = _minimal_task_inputs(tmp_path)
    lazy_input_data = _lazy_input_data(
        _lazy_measurement_set(error=OSError("unreadable tile")),
        inputs["data_selection"],
    )
    result = image_cube_single_field(**inputs, lazy_input_data=lazy_input_data)
    df = result["timing_node_tasks"]
    assert list(df["task_failed_phase"]) == ["load"]
    assert "OSError" in df["task_error"].iloc[0]
    assert "unreadable tile" in df["task_error"].iloc[0]
    assert df["failed_channel_start"].iloc[0] == 5
    assert result["deconvolution"].data == {}


def test_changed_measurement_set_aborts_the_task(tmp_path, monkeypatch):
    """``MSv2ChangedError`` (here a stand-in, so that the test runs without the
    engine) is raised instead of skipping the chunk; any other load error of
    the same task is skipped."""
    import xradio.measurement_set

    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    monkeypatch.setattr(
        xradio.measurement_set, "MSv2ChangedError", _ChangedError, raising=False
    )
    inputs = _minimal_task_inputs(tmp_path)
    changed = _lazy_input_data(
        _lazy_measurement_set(error=_ChangedError("MAIN rows added")),
        inputs["data_selection"],
    )
    with pytest.raises(_ChangedError, match="MAIN rows added"):
        image_cube_single_field(**inputs, lazy_input_data=changed)

    unreadable = _lazy_input_data(
        _lazy_measurement_set(error=RuntimeError("other")), inputs["data_selection"]
    )
    result = image_cube_single_field(**inputs, lazy_input_data=unreadable)
    assert list(result["timing_node_tasks"]["task_failed_phase"]) == ["load"]


@requires_msv2_engine
def test_changed_msv2_aborts_the_task(imageable_msv2, tmp_path, monkeypatch):
    """A MAIN row added to the Measurement Set after the distributed
    application's step (open, graph mapping, ``add_lazy_input_data``) makes the
    node task raise XRADIO's ``MSv2ChangedError`` (on a copy of the generated
    Measurement Set)."""
    from casacore import tables
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from xradio.measurement_set import MSv2ChangedError

    from astroviper.node_tasks.imaging.image_cube_single_field import (
        image_cube_single_field,
    )
    from astroviper.node_tasks.imaging.utils import (
        open_processing_set_skunk_works_msv2,
    )

    monkeypatch.delenv("XRADIO_MSV2_PARTITION_CACHE", raising=False)
    ms_path = str(tmp_path / os.path.basename(imageable_msv2))
    shutil.copytree(imageable_msv2, ms_path)
    ps_xdt = open_processing_set_skunk_works_msv2(ms_path)
    parallel_coords = {
        "frequency": make_parallel_coord(coord=ps_xdt.xr_ps.get_freq_axis(), n_chunks=3)
    }
    mapping = interpolate_data_coords_onto_parallel_coords(parallel_coords, ps_xdt)
    add_lazy_input_data(ps_xdt, mapping, "base")
    task = pickle.loads(pickle.dumps(mapping[0]))

    with tables.table(ms_path, readonly=False, ack=False) as main:
        main.addrows(1)

    inputs = _minimal_task_inputs(tmp_path)
    inputs.update(
        task_coords=task["task_coords"],
        data_selection=task["data_selection"],
        input_data_store=ms_path,
        processing_set_data_group_name="base",
    )
    with pytest.raises(MSv2ChangedError):
        image_cube_single_field(**inputs, lazy_input_data=task["lazy_input_data"])
