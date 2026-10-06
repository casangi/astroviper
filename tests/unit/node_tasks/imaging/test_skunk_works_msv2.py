"""Unit tests for the Measurement Set v2 input of the skunk-works imaging path.

Two kinds of tests:

* Not gated (every CI job): the helpers on a synthetic lazy processing set whose
  arrays record their reads (xarray's lazy backend-array API), and the engine
  checks with XRADIO's ``open_msv2`` monkeypatched in or away.
* Gated by ``requires_msv2_engine`` (XRADIO's ``xradio_msv2`` engine, with
  python-casacore for the fixture): on the generated ``imageable_msv2``
  Measurement Set (``tests/unit/conftest.py``), the loader equals XRADIO's
  production loader on the converted processing set for every selection, the
  graph mapping equals the converted one, a spawned process reads a pickled
  task payload, and an open plus every task's load leaves the Measurement Set
  unwritten.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

from astroviper.node_tasks.imaging.utils import (
    add_lazy_input_data,
    check_data_group_skunk_works_msv2,
    is_fatal_load_error,
    load_processing_set_skunk_works_msv2,
    msv2_engine_available,
    open_processing_set_skunk_works_msv2,
    require_msv2_engine,
    skunk_works_msv2,
)

_ENGINE_AVAILABLE, _ENGINE_REASON = msv2_engine_available()
requires_msv2_engine = pytest.mark.skipif(
    not _ENGINE_AVAILABLE, reason=_ENGINE_REASON or "needs XRADIO with open_msv2"
)

ROLES = ("correlated_data", "flag", "weight", "uvw")
NAME_COORDINATES = ("baseline_antenna1_name", "baseline_antenna2_name")
CODE_COORDINATES = ("baseline_antenna1_code", "baseline_antenna2_code")
NAME_TABLE = "baseline_antenna_name_table"
# The coordinates of a loaded task, and of a task's lazy selection (the
# baseline antenna names shipped as codes).
KEPT_COORDINATES = {"frequency", "polarization", *NAME_COORDINATES}
SHIPPED_COORDINATES = {"frequency", "polarization", *CODE_COORDINATES}
VISIBILITY_DIMS = ("time", "baseline_id", "frequency", "polarization")


# --------------------------------------------------------------------------- #
# A synthetic lazy processing set
# --------------------------------------------------------------------------- #
class RecordingArray(BackendArray):
    """A lazily indexed array that records its reads (xarray's backend API).

    ``negative_stride`` returns views with a negative stride along the first
    axis and ``read_only`` returns read-only arrays, as a backend may.
    """

    def __init__(self, values, reads, negative_stride=False, read_only=False):
        self.shape = values.shape
        self.dtype = values.dtype
        self.values = values
        self.reads = reads
        self.negative_stride = negative_stride
        self.read_only = read_only

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._read
        )

    def _read(self, key):
        self.reads.append(key)
        values = self.values
        if self.negative_stride:
            values = np.ascontiguousarray(values[::-1])[::-1]
        out = values[key]
        if self.read_only:
            out = out.view()
            out.flags.writeable = False
        return out


def make_lazy_ps(
    reads,
    ms_names=("ms_0", "ms_1"),
    n_antenna=4,
    antenna_prefix="ANT",
    arrays=None,
    **array_options,
):
    """A small lazy processing set: MSv4-like nodes with recording arrays.

    Each node has the ``base`` and ``corrected`` data groups, an extra data
    variable and the coordinates of an MSv4. ``arrays`` (optional) collects the
    recording arrays by ``(ms_name, variable)``.
    """
    rng = np.random.default_rng(0)
    antenna = np.array([f"{antenna_prefix}{k}" for k in range(n_antenna)])
    antenna1, antenna2 = np.triu_indices(n_antenna, 1)
    n_time, n_baseline, n_frequency = 3, antenna1.size, 5
    shape = (n_time, n_baseline, n_frequency, 2)

    nodes = {}
    for k, ms_name in enumerate(ms_names):

        def lazy(dims, values, name, ms_name=ms_name):
            array = RecordingArray(values, reads, **array_options)
            if arrays is not None:
                arrays[(ms_name, name)] = array
            return xr.Variable(dims, indexing.LazilyIndexedArray(array))

        def visibilities():
            return (
                rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            ).astype(np.complex64)

        data_vars = {
            "VISIBILITY": lazy(VISIBILITY_DIMS, visibilities(), "VISIBILITY"),
            "VISIBILITY_CORRECTED": lazy(
                VISIBILITY_DIMS, visibilities(), "VISIBILITY_CORRECTED"
            ),
            "FLAG": lazy(VISIBILITY_DIMS, rng.random(shape) < 0.1, "FLAG"),
            "WEIGHT": lazy(
                VISIBILITY_DIMS, rng.uniform(0.5, 2, shape).astype(np.float32), "WEIGHT"
            ),
            "UVW": lazy(
                ("time", "baseline_id", "uvw_label"),
                rng.uniform(-100, 100, (n_time, n_baseline, 3)),
                "UVW",
            ),
            "TIME_CENTROID": lazy(
                ("time", "baseline_id"), np.zeros((n_time, n_baseline)), "TIME_CENTROID"
            ),
        }
        coords = {
            "time": ("time", np.arange(n_time, dtype=np.float64)),
            "baseline_id": ("baseline_id", np.arange(n_baseline)),
            "baseline_antenna1_name": ("baseline_id", antenna[antenna1]),
            "baseline_antenna2_name": ("baseline_id", antenna[antenna2]),
            "frequency": (
                "frequency",
                1e9 + 1e6 * (n_frequency * k + np.arange(n_frequency)),
                {"units": "Hz"},
            ),
            "polarization": ("polarization", ["XX", "YY"]),
            "uvw_label": ("uvw_label", ["u", "v", "w"]),
            "scan_name": ("time", ["1", "1", "2"]),
            "field_name": ("time", ["F"] * n_time),
        }
        data_groups = {
            "base": {
                "correlated_data": "VISIBILITY",
                "flag": "FLAG",
                "weight": "WEIGHT",
                "uvw": "UVW",
                "description": "base",
            },
            "corrected": {
                "correlated_data": "VISIBILITY_CORRECTED",
                "flag": "FLAG",
                "weight": "WEIGHT",
                "uvw": "UVW",
            },
        }
        nodes[ms_name] = xr.Dataset(
            data_vars, coords, {"type": "visibility", "data_groups": data_groups}
        )
    return xr.DataTree.from_dict(nodes)


def make_mapping():
    """A graph mapping of two tasks; the second also selects correlations."""
    return {
        0: {
            "chunk_indices": (0,),
            "data_selection": {"ms_0": {"frequency": slice(0, 2)}},
        },
        1: {
            "chunk_indices": (1,),
            "data_selection": {
                "ms_0": {"frequency": slice(2, 5), "polarization": [1, 0]},
                "ms_1": {"frequency": slice(0, 1), "polarization": [1, 0]},
            },
        },
    }


# --------------------------------------------------------------------------- #
# add_lazy_input_data (not gated)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group", ["base", "corrected"])
def test_add_lazy_input_data_selections(group):
    reads = []
    ps_xdt = make_lazy_ps(reads)
    mapping = make_mapping()
    expected = copy.deepcopy(mapping)

    add_lazy_input_data(ps_xdt, mapping, group)

    assert reads == [], "the mapping step must not read data"
    for task_id, task in mapping.items():
        # The other keys of the task are left alone.
        assert {k: v for k, v in task.items() if k != "lazy_input_data"} == (
            expected[task_id]
        )
        assert set(task["lazy_input_data"]) == set(task["data_selection"])
        for ms_name, dataset in task["lazy_input_data"].items():
            source = ps_xdt[ms_name].to_dataset()
            data_group = source.attrs["data_groups"][group]
            assert list(dataset.data_vars) == [data_group[role] for role in ROLES]
            assert set(dataset.coords) == SHIPPED_COORDINATES
            assert set(dataset.attrs) == {"data_groups", NAME_TABLE}
            assert dataset.attrs["data_groups"] == {group: data_group}
            want = source.isel(task["data_selection"][ms_name])
            assert dict(dataset.sizes) == {
                "time": 3,
                "baseline_id": 6,
                "frequency": want.sizes["frequency"],
                "polarization": 2,
                "uvw_label": 3,
            }
            np.testing.assert_array_equal(dataset.frequency, want.frequency)
            np.testing.assert_array_equal(dataset.polarization, want.polarization)
    assert list(mapping[1]["lazy_input_data"]["ms_1"].polarization.values) == [
        "YY",
        "XX",
    ]
    assert reads == []


def test_add_lazy_input_data_loads_the_selected_values():
    """Loading a task's selection gives the selected values of the source."""
    reads = []
    ps_xdt = make_lazy_ps(reads)
    mapping = make_mapping()
    add_lazy_input_data(ps_xdt, mapping, "corrected")

    loaded = load_processing_set_skunk_works_msv2(mapping[1]["lazy_input_data"])

    assert reads, "the load reads the data"
    for ms_name, selection in mapping[1]["data_selection"].items():
        want = ps_xdt[ms_name].to_dataset().isel(selection).compute()
        got = loaded[ms_name].to_dataset()
        for name in ("VISIBILITY_CORRECTED", "FLAG", "WEIGHT", "UVW"):
            assert got[name].dims == want[name].dims
            assert got[name].dtype == want[name].dtype
            np.testing.assert_array_equal(got[name].values, want[name].values)
        for name in KEPT_COORDINATES:
            np.testing.assert_array_equal(got[name].values, want[name].values)


def test_add_lazy_input_data_whole_ms_without_selection():
    """An MSv4 whose selection is None is taken whole (as by the production
    loader)."""
    ps_xdt = make_lazy_ps([])
    mapping = {0: {"data_selection": {"ms_0": None, "ms_1": {}}}}
    add_lazy_input_data(ps_xdt, mapping, "base")
    for dataset in mapping[0]["lazy_input_data"].values():
        assert dataset.sizes["frequency"] == 5
        assert dataset.sizes["polarization"] == 2


def test_add_lazy_input_data_on_a_task_that_selects_nothing():
    """A task that selects no MSv4 gets an empty ``lazy_input_data``.
    GraphVIPER's map reuses one parameter dict across the tasks (it updates
    it with each task, then deep-copies it), so without the key such a task
    would inherit the previous task's selections."""
    ps_xdt = make_lazy_ps([])
    mapping = {
        0: {"data_selection": {"ms_0": {"frequency": slice(0, 2)}}},
        1: {"data_selection": {}},
    }
    add_lazy_input_data(ps_xdt, mapping, "base")
    assert mapping[1]["lazy_input_data"] == {}

    input_params, node_task_parameters = {}, []  # as GraphVIPER's map
    for task in mapping.values():
        input_params.update(task)
        node_task_parameters.append(copy.deepcopy(input_params))
    assert list(node_task_parameters[0]["lazy_input_data"]) == ["ms_0"]
    assert node_task_parameters[1]["lazy_input_data"] == {}


def test_add_lazy_input_data_leaves_the_input_attrs_alone():
    """The per-task datasets get attrs of their own; the MSv4's data groups
    are not modified."""
    ps_xdt = make_lazy_ps([])
    before = copy.deepcopy(dict(ps_xdt["ms_0"].attrs))
    mapping = make_mapping()
    add_lazy_input_data(ps_xdt, mapping, "base")
    mapping[0]["lazy_input_data"]["ms_0"].attrs["data_groups"]["base"]["flag"] = "X"
    assert dict(ps_xdt["ms_0"].attrs) == before


def test_add_lazy_input_data_shares_one_subset_per_ms():
    """The per-MSv4 subset is built once; each task indexes it."""
    ps_xdt = make_lazy_ps([])
    mapping = make_mapping()
    add_lazy_input_data(ps_xdt, mapping, "base")
    first = mapping[0]["lazy_input_data"]["ms_0"]
    second = mapping[1]["lazy_input_data"]["ms_0"]
    assert first is not second
    assert first.sizes["frequency"] == 2 and second.sizes["frequency"] == 3


def _add_lazy_input_data(ps_xdt, group):
    add_lazy_input_data(ps_xdt, make_mapping(), group)


#: The data-group checks: the mapping step, and the driver's check right after
#: the open (before anything is written).
data_group_checks = pytest.mark.parametrize(
    "check",
    [_add_lazy_input_data, check_data_group_skunk_works_msv2],
    ids=["add_lazy_input_data", "check_data_group"],
)


@data_group_checks
def test_data_group_missing(check):
    ps_xdt = make_lazy_ps([])
    with pytest.raises(ValueError, match=r"ms_0 has no data group 'imaging'"):
        check(ps_xdt, "imaging")


@data_group_checks
def test_data_group_single_dish(check):
    """A data group without uvw (a single-dish MSv4) is refused."""
    ps_xdt = make_lazy_ps([])
    dataset = ps_xdt["ms_0"].to_dataset()
    dataset.attrs["data_groups"] = {
        "base": {"correlated_data": "VISIBILITY", "flag": "FLAG", "weight": "WEIGHT"}
    }
    ps_xdt["ms_0"] = xr.DataTree(dataset)
    with pytest.raises(ValueError, match=r"has no \['uvw'\].*single-dish"):
        check(ps_xdt, "base")


@data_group_checks
def test_data_group_missing_variable(check):
    ps_xdt = make_lazy_ps([])
    ps_xdt["ms_1"] = xr.DataTree(ps_xdt["ms_1"].to_dataset().drop_vars("WEIGHT"))
    with pytest.raises(ValueError, match=r"ms_1 lacks the variables \['WEIGHT'\]"):
        check(ps_xdt, "base")


@pytest.mark.parametrize("group", ["base", "corrected"])
def test_check_data_group_reads_nothing(group):
    reads = []
    check_data_group_skunk_works_msv2(make_lazy_ps(reads), group)
    assert reads == []


def test_check_data_group_checks_every_measurement_set():
    """The driver's check covers every MSv4, also one that no task selects
    (which the mapping step never looks at)."""
    ps_xdt = make_lazy_ps([])
    ps_xdt["ms_1"] = xr.DataTree(ps_xdt["ms_1"].to_dataset().drop_vars("WEIGHT"))
    mapping = {0: {"data_selection": {"ms_0": {"frequency": slice(0, 2)}}}}
    add_lazy_input_data(ps_xdt, mapping, "base")
    with pytest.raises(ValueError, match=r"ms_1 lacks the variables"):
        check_data_group_skunk_works_msv2(ps_xdt, "base")


# --------------------------------------------------------------------------- #
# load_processing_set_skunk_works_msv2 (not gated)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "array_options",
    [{}, {"negative_stride": True}, {"read_only": True}],
    ids=["contiguous", "negative_stride", "read_only"],
)
def test_load_gives_contiguous_writeable_numpy(array_options):
    """The data variables are NumPy, C-contiguous and writeable (WEIGHT is
    modified in place by the imaging weights)."""
    arrays = {}
    ps_xdt = make_lazy_ps([], arrays=arrays, **array_options)
    mapping = {0: {"data_selection": {"ms_0": {"frequency": slice(1, 4)}}}}
    add_lazy_input_data(ps_xdt, mapping, "base")

    loaded = load_processing_set_skunk_works_msv2(mapping[0]["lazy_input_data"])

    assert isinstance(loaded, xr.DataTree)
    assert list(loaded.children) == ["ms_0"]
    dataset = loaded["ms_0"].to_dataset()
    for name in ("VISIBILITY", "FLAG", "WEIGHT", "UVW"):
        values = dataset[name].values
        assert type(values) is np.ndarray
        assert values.flags.c_contiguous and values.flags.writeable, name
        source = arrays[("ms_0", name)].values
        want = source[:, :, 1:4] if source.ndim == 4 else source
        np.testing.assert_array_equal(values, want)
    dataset["WEIGHT"].values[...] = 0.0  # in place, as calculate_imaging_weights


def test_load_copies_only_when_needed():
    """An array that is already C-contiguous and writeable is not copied."""
    arrays = {}
    ps_xdt = make_lazy_ps([], arrays=arrays)
    mapping = {
        0: {"data_selection": {"ms_0": {"frequency": slice(None)}}},
        1: {"data_selection": {"ms_0": {"frequency": slice(1, 4)}}},
    }
    add_lazy_input_data(ps_xdt, mapping, "base")

    whole = load_processing_set_skunk_works_msv2(mapping[0]["lazy_input_data"])
    part = load_processing_set_skunk_works_msv2(mapping[1]["lazy_input_data"])

    for name in ("VISIBILITY", "FLAG", "WEIGHT", "UVW"):
        source = arrays[("ms_0", name)].values
        assert np.shares_memory(whole["ms_0"][name].values, source), name
        # A channel range of a C-ordered array is a strided view: copied
        # (UVW has no frequency axis and is not).
        not_copied = name == "UVW"
        assert np.shares_memory(part["ms_0"][name].values, source) == not_copied


def test_load_leaves_the_task_selection_lazy():
    """The lazy datasets in the task's parameters are not loaded in place:
    every load reads again, and nothing is held between loads."""
    reads = []
    ps_xdt = make_lazy_ps(reads)
    mapping = make_mapping()
    add_lazy_input_data(ps_xdt, mapping, "base")
    lazy_input_data = mapping[1]["lazy_input_data"]

    first = load_processing_set_skunk_works_msv2(lazy_input_data)
    n_reads = len(reads)
    second = load_processing_set_skunk_works_msv2(lazy_input_data)

    assert n_reads > 0 and len(reads) == 2 * n_reads
    xr.testing.assert_identical(first, second)


def test_load_with_no_measurement_set():
    assert len(load_processing_set_skunk_works_msv2({}).children) == 0


# --------------------------------------------------------------------------- #
# Baseline antenna names as integer codes (not gated)
# --------------------------------------------------------------------------- #
def test_baseline_codes_rebuild_the_names_exactly():
    """The selections carry small integer codes and one name table; the load
    rebuilds the name coordinates exactly (values, dtype, attributes)."""
    ps_xdt = make_lazy_ps([], antenna_prefix="DV")
    source = ps_xdt["ms_0"].to_dataset()
    source = source.assign_coords(
        baseline_antenna1_name=source.baseline_antenna1_name.assign_attrs(
            description="first antenna"
        )
    )
    ps_xdt["ms_0"] = xr.DataTree(source)
    mapping = make_mapping()
    add_lazy_input_data(ps_xdt, mapping, "base")

    shipped = mapping[1]["lazy_input_data"]["ms_0"]
    table = shipped.attrs[NAME_TABLE]
    np.testing.assert_array_equal(table, ["DV0", "DV1", "DV2", "DV3"])
    for name, code_name in zip(NAME_COORDINATES, CODE_COORDINATES, strict=True):
        assert shipped[code_name].dtype == np.uint8
        assert shipped[code_name].dims == ("baseline_id",)
        np.testing.assert_array_equal(table[shipped[code_name].values], source[name])

    loaded = load_processing_set_skunk_works_msv2(mapping[1]["lazy_input_data"])

    for ms_name in ("ms_0", "ms_1"):
        got = loaded[ms_name].to_dataset()
        want = ps_xdt[ms_name].to_dataset()
        assert set(got.coords) == KEPT_COORDINATES
        assert NAME_TABLE not in got.attrs
        for name in NAME_COORDINATES:
            xr.testing.assert_identical(got[name].variable, want[name].variable)
    assert loaded["ms_0"]["baseline_antenna1_name"].attrs == {
        "description": "first antenna"
    }


@pytest.mark.parametrize(
    ("n_names", "dtype"), [(1, np.uint8), (256, np.uint8), (257, np.uint16)]
)
def test_baseline_codes_use_the_smallest_unsigned_type(n_names, dtype):
    names = np.array([f"A{k:04d}" for k in range(n_names)])
    dataset = xr.Dataset(
        coords={
            "baseline_antenna1_name": ("baseline_id", names),
            "baseline_antenna2_name": ("baseline_id", names[::-1]),
        }
    )
    encoded = skunk_works_msv2._encode_baseline_antenna_names(dataset)
    assert encoded["baseline_antenna1_code"].dtype == dtype
    assert encoded.attrs[NAME_TABLE].size == n_names
    xr.testing.assert_identical(
        skunk_works_msv2._decode_baseline_antenna_names(encoded), dataset
    )


def test_baseline_codes_without_names():
    """A dataset without name coordinates, or without a name table, passes
    through unchanged."""
    dataset = xr.Dataset(coords={"frequency": [1.0, 2.0]})
    assert skunk_works_msv2._encode_baseline_antenna_names(dataset) is dataset
    named = dataset.assign_coords(baseline_antenna1_name=("baseline_id", ["A", "B"]))
    assert skunk_works_msv2._decode_baseline_antenna_names(named) is named


def test_baseline_codes_with_one_name_coordinate():
    """A dataset with only one of the two name coordinates round-trips."""
    named = xr.Dataset(coords={"baseline_antenna1_name": ("baseline_id", ["B", "A"])})
    encoded = skunk_works_msv2._encode_baseline_antenna_names(named)
    assert set(encoded.coords) == {"baseline_antenna1_code"}
    xr.testing.assert_identical(
        skunk_works_msv2._decode_baseline_antenna_names(encoded), named
    )


def test_baseline_codes_shrink_the_payload():
    """For an array with many antennas the names, repeated per baseline,
    dominate a task's coordinates; the codes and the table are far smaller."""
    ps_xdt = make_lazy_ps([], ms_names=("ms_0",), n_antenna=40, antenna_prefix="PAD")
    mapping = {0: {"data_selection": {"ms_0": {"frequency": slice(0, 1)}}}}
    add_lazy_input_data(ps_xdt, mapping, "base")
    shipped = mapping[0]["lazy_input_data"]["ms_0"]

    names = pickle.dumps([ps_xdt["ms_0"][name].variable for name in NAME_COORDINATES])
    codes = pickle.dumps(
        [shipped[name].variable for name in CODE_COORDINATES]
        + [shipped.attrs[NAME_TABLE]]
    )
    assert shipped.sizes["baseline_id"] == 780
    assert len(codes) < len(names) / 4


# --------------------------------------------------------------------------- #
# Engine checks and the open (not gated: open_msv2 monkeypatched)
# --------------------------------------------------------------------------- #
def test_msv2_engine_available_without_open_msv2(monkeypatch):
    import xradio.measurement_set

    monkeypatch.delattr(xradio.measurement_set, "open_msv2", raising=False)
    available, reason = msv2_engine_available()
    assert not available
    assert "open_msv2" in reason and "casacore" in reason


def test_msv2_engine_available_with_open_msv2(monkeypatch):
    import xradio.measurement_set

    monkeypatch.setattr(
        xradio.measurement_set, "open_msv2", lambda *a, **k: None, raising=False
    )
    assert msv2_engine_available() == (True, "")


def test_msv2_engine_available_without_xradio_measurement_set(monkeypatch):
    monkeypatch.setitem(sys.modules, "xradio.measurement_set", None)
    available, reason = msv2_engine_available()
    assert not available
    assert "cannot be imported" in reason


def test_require_msv2_engine_message(monkeypatch):
    import xradio.measurement_set

    monkeypatch.delattr(xradio.measurement_set, "open_msv2", raising=False)
    with pytest.raises(ImportError) as error:
        require_msv2_engine()
    message = str(error.value)
    for text in (
        "xradio_msv2",
        "python_casacore",
        "convert_msv2_to_processing_set",
        "open_msv2",
    ):
        assert text in message
    # No release that does not exist yet is named.
    assert "1.2.5" not in message


def test_require_msv2_engine_returns_open_msv2(monkeypatch):
    import xradio.measurement_set

    def fake_open_msv2(*args, **kwargs):
        return None

    monkeypatch.setattr(
        xradio.measurement_set, "open_msv2", fake_open_msv2, raising=False
    )
    assert require_msv2_engine() is fake_open_msv2


class OpenRecorder:
    """Stands in for ``open_msv2``: records the call, returns ``result``."""

    def __init__(self, result=None):
        self.calls = []
        self.result = (
            xr.DataTree.from_dict({"ms_0": xr.Dataset()}) if result is None else result
        )

    def __call__(self, ms_path, **kwargs):
        self.calls.append((ms_path, kwargs))
        return self.result


@pytest.fixture
def open_recorder(monkeypatch):
    recorder = OpenRecorder()
    monkeypatch.setattr(skunk_works_msv2, "require_msv2_engine", lambda: recorder)
    monkeypatch.delenv("XRADIO_MSV2_PARTITION_CACHE", raising=False)
    return recorder


def test_open_defaults(open_recorder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ps_xdt = open_processing_set_skunk_works_msv2("data.ms", ["OBSERVE_TARGET#*"])
    assert ps_xdt is open_recorder.result
    ((ms_path, kwargs),) = open_recorder.calls
    assert ms_path == os.path.join(os.getcwd(), "data.ms")  # absolute
    assert kwargs == {
        "scan_intents": ["OBSERVE_TARGET#*"],
        "array_backend": "xarray",
        "with_pointing": False,
        "partition_cache": "read",
    }


def test_open_expands_user(open_recorder, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    open_processing_set_skunk_works_msv2("~/data.ms")
    assert open_recorder.calls[0][0] == str(tmp_path / "data.ms")


def test_open_partition_cache_from_environment(open_recorder, monkeypatch):
    """With XRADIO_MSV2_PARTITION_CACHE set, XRADIO's own default applies."""
    monkeypatch.setenv("XRADIO_MSV2_PARTITION_CACHE", "auto")
    open_processing_set_skunk_works_msv2("/data.ms")
    assert "partition_cache" not in open_recorder.calls[0][1]


def test_open_user_options_win(open_recorder):
    options = {
        "partition_cache": "auto",
        "with_pointing": True,
        "partition_scheme": ["FIELD_ID"],
        "skip_columns": ["WEIGHT_SPECTRUM"],
    }
    given = copy.deepcopy(options)
    open_processing_set_skunk_works_msv2("/data.ms", msv2_open_options=options)
    kwargs = open_recorder.calls[0][1]
    assert {key: kwargs[key] for key in options} == options
    assert kwargs["array_backend"] == "xarray"
    assert options == given  # not modified


@pytest.mark.parametrize("key", ["array_backend", "scan_intents"])
def test_open_reserved_options(open_recorder, key):
    with pytest.raises(ValueError, match=rf"may not set \['{key}'\]"):
        open_processing_set_skunk_works_msv2(
            "/data.ms", msv2_open_options={key: "dask"}
        )
    assert open_recorder.calls == []  # refused before the open


def test_open_reserved_options_before_the_engine_check(monkeypatch):
    """A reserved option is reported even without the engine."""

    def no_engine():
        raise ImportError("no engine")

    monkeypatch.setattr(skunk_works_msv2, "require_msv2_engine", no_engine)
    with pytest.raises(ValueError, match="array_backend"):
        open_processing_set_skunk_works_msv2(
            "/data.ms", msv2_open_options={"array_backend": "dask"}
        )
    with pytest.raises(ImportError, match="no engine"):
        open_processing_set_skunk_works_msv2("/data.ms")


def test_open_empty_after_scan_intents(open_recorder):
    open_recorder.result = xr.DataTree()
    with pytest.raises(ValueError, match=r"scan_intents=None"):
        open_processing_set_skunk_works_msv2(
            "/data.ms", scan_intents=["OBSERVE_TARGET#ON_SOURCE"]
        )


def test_open_empty_after_a_scan_intent_string(open_recorder):
    """The driver also accepts one scan intent as a string; the message names
    it whole."""
    open_recorder.result = xr.DataTree()
    with pytest.raises(ValueError, match=r"\['OBSERVE_TARGET#ON_SOURCE'\]"):
        open_processing_set_skunk_works_msv2(
            "/data.ms", scan_intents="OBSERVE_TARGET#ON_SOURCE"
        )
    assert open_recorder.calls[0][1]["scan_intents"] == "OBSERVE_TARGET#ON_SOURCE"


def test_open_empty_measurement_set(open_recorder):
    open_recorder.result = xr.DataTree()
    with pytest.raises(ValueError, match=r"no visibilities"):
        open_processing_set_skunk_works_msv2("/data.ms")


def test_is_fatal_load_error(monkeypatch):
    import xradio.measurement_set

    class FakeChangedError(RuntimeError):
        pass

    monkeypatch.setattr(
        xradio.measurement_set, "MSv2ChangedError", FakeChangedError, raising=False
    )
    assert is_fatal_load_error(FakeChangedError("MAIN rows added"))
    assert not is_fatal_load_error(OSError("disk"))
    assert not is_fatal_load_error(ValueError("conflicting sizes"))


def test_is_fatal_load_error_without_engine(monkeypatch):
    import xradio.measurement_set

    monkeypatch.delattr(xradio.measurement_set, "MSv2ChangedError", raising=False)
    assert not is_fatal_load_error(RuntimeError("anything"))


# --------------------------------------------------------------------------- #
# Through XRADIO's engine, on the generated Measurement Set (gated)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def opened_msv2(imageable_msv2):
    """The generated Measurement Set, opened once for the module (read only)."""
    return open_processing_set_skunk_works_msv2(
        imageable_msv2, msv2_open_options={"partition_cache": "read"}
    )


def load_production(ps_store, data_selection, group):
    """The production node task's load of a converted processing set."""
    from xradio.measurement_set import load_processing_set

    return load_processing_set(
        ps_store,
        sel_parms=copy.deepcopy(data_selection),
        data_group_name=group,
        load_sub_datasets=False,
    )


def _group_roles(data_group):
    return {k: v for k, v in data_group.items() if k not in ("date", "description")}


def assert_equals_production(loaded, production, group):
    """The MSv2 load equals the production load of the converted processing
    set: the data group's variables (values, dims, dtypes), frequency,
    polarization and baseline names, and the data group."""
    assert sorted(loaded.children) == sorted(production.children)
    for ms_name in production.children:
        got = loaded[ms_name].to_dataset()
        want = production[ms_name].to_dataset()
        data_group = want.attrs["data_groups"][group]
        names = [data_group[role] for role in ROLES]
        assert sorted(got.data_vars) == sorted(names)
        for name in names:
            assert got[name].dims == want[name].dims, (ms_name, name)
            assert got[name].dtype == want[name].dtype, (ms_name, name)
            np.testing.assert_array_equal(
                got[name].values, want[name].values, err_msg=f"{ms_name} {name}"
            )
            assert got[name].values.flags.c_contiguous
            assert got[name].values.flags.writeable
        assert set(got.coords) == KEPT_COORDINATES
        for name in KEPT_COORDINATES:
            assert got[name].dtype == want[name].dtype, (ms_name, name)
            np.testing.assert_array_equal(
                got[name].values, want[name].values, err_msg=f"{ms_name} {name}"
            )
        assert list(got.attrs["data_groups"]) == [group]
        assert _group_roles(got.attrs["data_groups"][group]) == _group_roles(data_group)


@requires_msv2_engine
@pytest.mark.parametrize("polarization", [None, [1, 0]], ids=["all_pol", "pol_1_0"])
@pytest.mark.parametrize(
    "frequency",
    [slice(0, 1), slice(3, 7), slice(2, 13), slice(None)],
    ids=["chan_0", "chan_3_7", "chan_2_13", "all_chan"],
)
@pytest.mark.parametrize("group", ["base", "corrected"])
def test_loader_equals_production(
    opened_msv2, imageable_msv2_ps, group, frequency, polarization
):
    """Every selection of every MSv4 equals the production load of the converted
    processing set: whole-cell tiles (``base``) and 4-channel tiles
    (``corrected``; the slices start inside a tile and cross tiles), the
    decreasing SPW, the duplicated row and the padded grid."""
    selection = {"frequency": frequency}
    if polarization is not None:
        selection["polarization"] = polarization
    data_selection = {ms_name: dict(selection) for ms_name in opened_msv2.children}
    mapping = {0: {"data_selection": data_selection}}

    add_lazy_input_data(opened_msv2, mapping, group)
    loaded = load_processing_set_skunk_works_msv2(mapping[0]["lazy_input_data"])

    assert_equals_production(
        loaded, load_production(imageable_msv2_ps, data_selection, group), group
    )


def _graph_mapping(ps_xdt, n_chunks):
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )

    parallel_coords = {
        "frequency": make_parallel_coord(
            coord=ps_xdt.xr_ps.get_freq_axis(), n_chunks=n_chunks
        )
    }
    return interpolate_data_coords_onto_parallel_coords(parallel_coords, ps_xdt)


@requires_msv2_engine
@pytest.mark.parametrize("group", ["base", "corrected"])
def test_mapping_and_tasks_equal_production(opened_msv2, imageable_msv2_ps, group):
    """The MSv4 names, frequency axis and graph mapping equal those of the
    converted processing set, and so does every task's load, including the
    task that straddles both spectral windows."""
    from xradio.measurement_set import open_processing_set

    converted = open_processing_set(imageable_msv2_ps)
    assert sorted(opened_msv2.children) == sorted(converted.children)
    np.testing.assert_array_equal(
        opened_msv2.xr_ps.get_freq_axis().values,
        converted.xr_ps.get_freq_axis().values,
    )
    mapping = _graph_mapping(opened_msv2, n_chunks=3)
    converted_mapping = _graph_mapping(converted, n_chunks=3)
    assert len(mapping) == len(converted_mapping) == 3
    for task_id, task in mapping.items():
        assert task["data_selection"] == converted_mapping[task_id]["data_selection"]
        np.testing.assert_array_equal(
            task["task_coords"]["frequency"]["data"],
            converted_mapping[task_id]["task_coords"]["frequency"]["data"],
        )
    assert max(len(task["data_selection"]) for task in mapping.values()) == 4
    for task in mapping.values():  # the imaging's correlation selection
        for selection in task["data_selection"].values():
            selection["polarization"] = [1, 0]

    add_lazy_input_data(opened_msv2, mapping, group)

    for task in mapping.values():
        assert_equals_production(
            load_processing_set_skunk_works_msv2(task["lazy_input_data"]),
            load_production(imageable_msv2_ps, task["data_selection"], group),
            group,
        )


@requires_msv2_engine
def test_open_scan_intents(imageable_msv2):
    """The generated MSv4s have the intent ``scan_intent#subscan_intent``; the
    imaging's usual default intent leaves none."""
    options = {"partition_cache": "read"}
    ps_xdt = open_processing_set_skunk_works_msv2(
        imageable_msv2, ["scan_intent#subscan_intent"], msv2_open_options=options
    )
    assert len(ps_xdt.children) == 4
    with pytest.raises(ValueError, match="scan_intents=None"):
        open_processing_set_skunk_works_msv2(
            imageable_msv2, ["OBSERVE_TARGET#ON_SOURCE"], msv2_open_options=options
        )


@requires_msv2_engine
def test_spawned_process_reads_a_pickled_task(opened_msv2):
    """A task's payload pickles without an open table, stays small (the
    baseline names travel as codes), and a spawned process (no state shared
    with this one) reads the same values and rebuilds the same names."""
    mapping = _graph_mapping(opened_msv2, n_chunks=3)
    for task in mapping.values():
        for selection in task["data_selection"].values():
            selection["polarization"] = [1, 0]
    add_lazy_input_data(opened_msv2, mapping, "corrected")
    lazy_input_data = max(
        (task["lazy_input_data"] for task in mapping.values()), key=len
    )
    assert len(lazy_input_data) == 4
    for ms_name, dataset in lazy_input_data.items():
        assert set(dataset.coords) == SHIPPED_COORDINATES
        payload = len(pickle.dumps(dataset))
        assert payload < 16_000
        # The same selection with the name strings instead of the codes.
        with_names = dataset.drop_vars(list(CODE_COORDINATES)).assign_coords(
            {name: opened_msv2[ms_name][name].variable for name in NAME_COORDINATES}
        )
        with_names.attrs = {
            key: value for key, value in dataset.attrs.items() if key != NAME_TABLE
        }
        assert payload < len(pickle.dumps(with_names))

    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        spawned = executor.submit(
            load_processing_set_skunk_works_msv2, lazy_input_data
        ).result(timeout=300)

    loaded = load_processing_set_skunk_works_msv2(lazy_input_data)
    xr.testing.assert_identical(spawned, loaded)
    for ms_name in lazy_input_data:
        for name in NAME_COORDINATES:
            xr.testing.assert_identical(
                spawned[ms_name][name].variable, opened_msv2[ms_name][name].variable
            )


def _file_state(path):
    """Every file under ``path``: its relative path, size and mtime."""
    state = {}
    for directory, _, files in os.walk(path):
        for name in files:
            full = os.path.join(directory, name)
            stat = os.stat(full)
            state[os.path.relpath(full, path)] = (stat.st_size, stat.st_mtime_ns)
    return state


@requires_msv2_engine
def test_open_and_load_do_not_write(imageable_msv2, tmp_path, monkeypatch):
    """With the default open options, opening the Measurement Set, mapping the
    graph and loading every task leave it as it was: no partition cache and no
    other file written or touched."""
    monkeypatch.delenv("XRADIO_MSV2_PARTITION_CACHE", raising=False)
    ms_path = str(tmp_path / os.path.basename(imageable_msv2))
    shutil.copytree(imageable_msv2, ms_path)
    before = _file_state(ms_path)

    ps_xdt = open_processing_set_skunk_works_msv2(ms_path)
    mapping = _graph_mapping(ps_xdt, n_chunks=3)
    add_lazy_input_data(ps_xdt, mapping, "corrected")
    for task in mapping.values():
        load_processing_set_skunk_works_msv2(task["lazy_input_data"])

    assert _file_state(ms_path) == before
    assert not os.path.exists(os.path.join(ms_path, "XRADIO_PARTITIONS"))
