"""Tests for astroviper.utils.measurement_set_tools (empty MSv4 creation)."""

import numpy as np
import pytest
import xarray as xr
import zarr
from graphviper.graph_tools.coordinate_utils import make_parallel_coord
from xradio.measurement_set import load_processing_set, open_processing_set
from xradio.schema.check import check_datatree

from astroviper.utils.measurement_set_tools import (
    baseline_antenna_pairs,
    create_empty_measurement_set_v4_on_disk,
    make_empty_visibility_xds,
    make_field_and_source_xds,
    make_frequency_coordinate,
    make_time_coordinate,
    normalize_polarization,
    number_of_baselines,
    polarization_basis,
    polarization_index,
    resolve_fields,
    write_visibility_chunk_to_disk,
)
from astroviper.utils.telescope_layout import read_telescope_layout


def test_polarization_helpers():
    assert normalize_polarization([5, 8]) == ["RR", "LL"]
    assert normalize_polarization(["xx", "XY", "YX", "YY"]) == ["XX", "XY", "YX", "YY"]
    np.testing.assert_array_equal(polarization_index(["RR", "LL"]), [0, 3])
    np.testing.assert_array_equal(polarization_index([9, 10, 11, 12]), [0, 1, 2, 3])
    assert polarization_basis(["RL"]) == ("R", "L")
    assert polarization_basis(["YY"]) == ("X", "Y")
    with pytest.raises(ValueError):
        normalize_polarization(["RR", "XX"])
    with pytest.raises(ValueError):
        normalize_polarization([42])


def test_baseline_antenna_pairs():
    a1, a2 = baseline_antenna_pairs(4)
    np.testing.assert_array_equal(a1, [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(a2, [1, 2, 3, 2, 3, 3])
    assert number_of_baselines(4) == 6
    a1, a2 = baseline_antenna_pairs(3, auto_correlations=True)
    np.testing.assert_array_equal(a1, [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(a2, [0, 1, 2, 1, 2, 2])
    assert number_of_baselines(3, True) == 6
    assert number_of_baselines(27) == 351


def test_time_and_frequency_coordinates():
    tc = make_time_coordinate(
        {"time_start": "2019-10-03T19:00:00.000", "time_delta": 10.0, "n_samples": 3}
    )
    assert tc["dims"] == "time"
    np.testing.assert_allclose(np.diff(tc["data"]), 10.0)
    # MS convention: stamps are integration midpoints (start + delta / 2).
    from astropy.time import Time

    np.testing.assert_allclose(
        tc["data"][0], Time("2019-10-03T19:00:00.000", scale="utc").unix + 5.0
    )
    assert tc["attrs"]["format"] == "unix" and tc["attrs"]["scale"] == "utc"
    assert tc["attrs"]["integration_time"]["data"] == 10.0
    fc = make_frequency_coordinate(
        {
            "freq_start": 1e9,
            "freq_delta": 1e6,
            "n_channels": 5,
            "spectral_window_name": "L",
        }
    )
    np.testing.assert_allclose(fc["data"], 1e9 + 1e6 * np.arange(5))
    assert fc["attrs"]["spectral_window_name"] == "L"
    assert fc["attrs"]["channel_width"]["data"] == 1e6
    assert fc["attrs"]["reference_frequency"]["data"] == fc["data"][2]


def test_resolve_fields_single_and_mosaic():
    names, uniq, upc = resolve_fields(np.array([[1.0, 0.5]]), None, 4)
    assert list(names) == ["field_0"] * 4 and list(uniq) == ["field_0"]
    pcs = np.array([[1.0, 0.5], [1.0, 0.5], [1.1, 0.5], [1.1, 0.5]])
    names, uniq, upc = resolve_fields(pcs, None, 4)
    assert list(names) == ["field_0", "field_0", "field_1", "field_1"]
    np.testing.assert_allclose(upc, [[1.0, 0.5], [1.1, 0.5]])
    names, uniq, upc = resolve_fields(pcs, ["A", "A", "B", "B"], 4)
    assert list(uniq) == ["A", "B"]
    with pytest.raises(ValueError):
        resolve_fields(pcs, ["A", "A", "A", "B"], 4)  # 'A' used with two centres
    with pytest.raises(ValueError):
        resolve_fields(pcs[:3], None, 4)


@pytest.fixture()
def empty_store(tmp_path):
    ant = read_telescope_layout("vla.d", antenna_selection=[0, 1, 2, 3])
    tc = make_time_coordinate(
        {"time_start": "2019-10-03T19:00:00.000", "time_delta": 60.0, "n_samples": 6}
    )
    fc = make_frequency_coordinate(
        {"freq_start": 3e9, "freq_delta": 1e6, "n_channels": 4}
    )
    names, uniq, upc = resolve_fields(np.array([[1.0, 0.5]]), ["f1"], 6)
    fs = make_field_and_source_xds(uniq, upc)
    ms = make_empty_visibility_xds(tc, fc, [5, 8], ant, names)
    parallel_coords = {
        "time": make_parallel_coord(coord=tc, n_chunks=2),
        "frequency": make_parallel_coord(coord=fc, n_chunks=2),
    }
    ps_store = str(tmp_path / "sim.ps.zarr")
    ms_path = create_empty_measurement_set_v4_on_disk(
        ps_store, "vla_sim", ms, ant, fs, parallel_coords, overwrite=True
    )
    zarr.consolidate_metadata(ps_store)
    return ps_store, ms_path, parallel_coords


def test_empty_measurement_set_is_schema_valid(empty_store):
    ps_store, ms_path, _ = empty_store
    ps_xdt = open_processing_set(ps_store)
    assert str(check_datatree(ps_xdt)) == "No schema issues found"
    ms = ps_xdt["vla_sim"].ds
    assert ms.sizes == {
        "time": 6,
        "baseline_id": 6,
        "frequency": 4,
        "polarization": 2,
        "uvw_label": 3,
    }
    assert list(ms.polarization.values) == ["RR", "LL"]
    assert ms.VISIBILITY.dtype == np.complex128
    assert ms.FLAG.dtype == bool
    assert ms.attrs["data_groups"]["base"]["correlated_data"] == "VISIBILITY"
    assert (
        ms.attrs["data_groups"]["base"]["field_and_source"]
        == "field_and_source_base_xds"
    )
    assert ms.scan_name.values[0] == "scan_1"
    assert ms.field_name.values[-1] == "f1"
    assert "antenna_xds" in ps_xdt["vla_sim"].children
    assert "field_and_source_base_xds" in ps_xdt["vla_sim"].children
    # unwritten chunks read back as NaN
    assert np.isnan(ms.VISIBILITY.values).all()


def test_write_visibility_chunk(empty_store):
    ps_store, ms_path, pc = empty_store
    task_coords = {
        "time": {"slice": pc["time"]["data_chunk_slices"][1]},
        "frequency": {"slice": pc["frequency"]["data_chunk_slices"][0]},
    }
    chunk = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.full((3, 6, 2, 2), 1 + 2j),
            ),
            "UVW": (("time", "baseline_id", "uvw_label"), np.full((3, 6, 3), 7.0)),
            "WEIGHT": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.full((3, 6, 2, 2), 0.5),
            ),
            "FLAG": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.ones((3, 6, 2, 2), bool),
            ),
        }
    )
    write_visibility_chunk_to_disk(ms_path, task_coords, chunk)
    ms = load_processing_set(ps_store)["vla_sim"].ds
    np.testing.assert_array_equal(ms.VISIBILITY.values[3:, :, :2], 1 + 2j)
    assert np.isnan(ms.VISIBILITY.values[:3]).all()
    assert np.isnan(ms.VISIBILITY.values[3:, :, 2:]).all()
    np.testing.assert_array_equal(ms.UVW.values[3:], 7.0)
    assert ms.FLAG.values[3:, :, :2].all() and not ms.FLAG.values[:3].any()
    np.testing.assert_array_equal(ms.WEIGHT.values[3:, :, :2], 0.5)


def test_overwrite_guard(empty_store):
    ps_store, _, pc = empty_store
    ant = read_telescope_layout("vla.d", antenna_selection=[0, 1])
    tc = make_time_coordinate(
        {"time_start": "2019-10-03T19:00:00.000", "time_delta": 60.0, "n_samples": 2}
    )
    fc = make_frequency_coordinate(
        {"freq_start": 3e9, "freq_delta": 1e6, "n_channels": 2}
    )
    names, uniq, upc = resolve_fields(np.array([[1.0, 0.5]]), None, 2)
    ms = make_empty_visibility_xds(tc, fc, ["XX"], ant, names)
    fs = make_field_and_source_xds(uniq, upc)
    with pytest.raises(FileExistsError):
        create_empty_measurement_set_v4_on_disk(
            ps_store, "x", ms, ant, fs, {}, overwrite=False
        )
