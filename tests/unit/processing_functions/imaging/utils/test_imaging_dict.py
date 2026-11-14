"""Unit tests for the ImagingDict -> DataFrame flattening, its inverse and the
per-channel arrays of the deconvolution summary."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.utils.cycle_control import StopCode
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    ImagingDict,
    Key,
    deconvolution_per_channel,
    imaging_dict_from_dataframe,
    imaging_dict_to_dataframe,
)


def _make_imaging_dict():
    rd = ImagingDict()
    # channel 0: deconvolved over 2 cycles
    for cycle in range(2):
        rd.add(
            {
                "iter_done": [40, 25][cycle],
                "peakres": [0.05, 0.002][cycle],
                "start_peakres": [0.11, 0.05][cycle],
                "model_flux": [0.3, 0.42][cycle],
                "masksum": 100,
                "max_iter": 5000,
                "gain": 0.1,
                "max_psf_sidelobe": 0.35,
            },
            time=0,
            pol=0,
            chan=0,
        )
    rd.add(
        {
            "stop_code": StopCode(imaging=2, model_update=0),
            "stop_description": "threshold",
        },
        time=0,
        pol=0,
        chan=0,
    )
    # channel 1: never deconvolved (no iterations recorded)
    rd.add(
        {
            "iter_done": 0,
            "peakres": 0.001,
            "start_peakres": 0.001,
            "model_flux": 0.0,
            "stop_code": StopCode(imaging=2, model_update=0),
            "stop_description": "threshold",
            "max_iter": 5000,
        },
        time=0,
        pol=1,
        chan=1,
    )
    return rd


def test_flattens_history_and_stop_code():
    df = imaging_dict_to_dataframe(_make_imaging_dict())
    assert len(df) == 2
    row0 = df[(df.pol == 0) & (df.chan == 0)].iloc[0]
    assert row0["n_cycles"] == 2
    assert row0["iter_total"] == 65
    assert row0["peakres_start"] == 0.11
    assert row0["peakres_final"] == 0.002
    assert row0["model_flux_final"] == 0.42
    assert row0["stop_code_imaging"] == 2
    assert row0["iter_done"] == [40, 25]  # history preserved as a list
    row1 = df[(df.pol == 1) & (df.chan == 1)].iloc[0]
    assert row1["iter_total"] == 0  # the "never deconvolved" marker
    assert row1["stop_description"] == "threshold"


def test_empty_dict_gives_empty_frame():
    df = imaging_dict_to_dataframe(ImagingDict())
    assert df.empty
    assert "iter_total" in df.columns


def _assert_frames_equal(rebuilt, frame):
    """Equal up to column order; list cells compared element by element."""
    assert sorted(rebuilt.columns) == sorted(frame.columns)
    for column in frame.columns:
        for got, expected in zip(rebuilt[column], frame[column], strict=True):
            if isinstance(expected, list | np.ndarray):
                assert len(got) == len(expected), column
                for a, b in zip(got, expected, strict=True):
                    np.testing.assert_array_equal(a, b)
            elif pd.isna(expected):
                assert pd.isna(got), column
            else:
                assert got == expected, column


def test_from_dataframe_inverts_to_dataframe():
    imaging_dict = _make_imaging_dict()
    history = [np.array([0.3, 0.2]), np.array([0.1])]
    imaging_dict.data[Key(0, 0, 0)]["iteration_peak_residual"] = history
    frame = imaging_dict_to_dataframe(imaging_dict)
    rebuilt = imaging_dict_from_dataframe(frame)
    assert list(rebuilt.data) == [Key(0, 0, 0), Key(0, 1, 1)]
    fields = rebuilt.data[Key(0, 0, 0)]
    assert fields["stop_code"] == StopCode(imaging=2, model_update=0)
    assert isinstance(fields["stop_code"], StopCode)
    assert fields["iter_done"] == [40, 25]
    assert isinstance(fields["iteration_peak_residual"][0], np.ndarray)
    for derived in ("n_cycles", "iter_total", "peakres_final", "stop_code_imaging"):
        assert derived not in fields
    # the plane without these fields does not get them back as NaN
    assert "gain" not in rebuilt.data[Key(0, 1, 1)]
    assert "iteration_peak_residual" not in rebuilt.data[Key(0, 1, 1)]
    _assert_frames_equal(imaging_dict_to_dataframe(rebuilt), frame)


def test_from_dataframe_of_feather_cells_and_empty_frames():
    """Feather gives arrays for list cells and None in object columns."""
    frame = pd.DataFrame(
        {
            "time_index": [0],
            "pol": [1],
            "chan": [3],
            "iter_done": [np.array([0, 0])],
            "entropy": [np.array([None, None], dtype=object)],
            "clean_box": [(1, 2)],
            "stop_code_imaging": [np.nan],
            "stop_code_model_update": [np.nan],
            "peakres_start": [None],
        }
    )
    fields = imaging_dict_from_dataframe(frame).data[Key(0, 1, 3)]
    assert fields == {"iter_done": [0, 0], "entropy": [None, None], "clean_box": [1, 2]}
    assert (
        imaging_dict_from_dataframe(imaging_dict_to_dataframe(ImagingDict())).data == {}
    )


def _deconvolution_return_dict(entropy=True):
    """Residual statistics of 3 channels (Stokes I, Q) and the record of
    channels 0 and 2, placed by their frequency fields (out of channel
    order); channel 1 has no record."""
    frequency = np.array([1.0e9, 1.1e9, 1.2e9])
    dims = ("time", "frequency", "polarization")
    data = {"rms": (dims, np.arange(6.0).reshape(1, 3, 2))}
    if entropy:
        data["entropy"] = (dims, np.full((1, 3, 2), 0.7))
    statistics = xr.Dataset(
        data,
        coords={"time": [0.0], "frequency": frequency, "polarization": ["I", "Q"]},
    )
    record = ImagingDict()
    for pol in range(2):
        # recorded as channel 0 but at 1.2 GHz: two model updates
        record.data[Key(0, pol, 0)] = {
            "frequency": 1.2e9,
            "iter_done": [10, 5, 0],
            "start_peakres": [-0.4, 0.1],
            "peakres": [0.1, 0.02, 0.01],
            "model_update_stop_code": [1, 2],
            "threshold": 0.005,
            "stop_code": StopCode(imaging=2, model_update=2),
        }
        # recorded as channel 5 but at 1.0 GHz: no model update ran
        record.data[Key(0, pol, 5)] = {
            "frequency": 1.0e9,
            "iter_done": [0, 0],
            "start_peakres": [0.3],
            "peakres": [0.3, 0.3],
            "model_update_stop_code": [0],
            "stop_code": StopCode(imaging=4, model_update=0),
        }
    record.data[Key(1, 0, 0)] = {"iter_done": [7]}  # another time plane
    return {"deconvolution": record, "image_statistics": {"sky_residual": statistics}}


def test_deconvolution_per_channel():
    result = deconvolution_per_channel(_deconvolution_return_dict())
    assert result["stokes"] == ["I", "Q"]
    np.testing.assert_array_equal(result["frequency"], [1.0e9, 1.1e9, 1.2e9])
    nan = np.nan
    np.testing.assert_array_equal(
        result["imaging_stop_code"], [[4, 4], [nan, nan], [2, 2]]
    )
    np.testing.assert_array_equal(result["iterations"], [[0, 0], [nan, nan], [15, 15]])
    np.testing.assert_array_equal(result["imaging_cycles"], [1, nan, 2])
    np.testing.assert_array_equal(
        result["peakres_start"], [[0.3, 0.3], [nan, nan], [-0.4, -0.4]]
    )
    np.testing.assert_array_equal(
        result["peakres_final"], [[0.3, 0.3], [nan, nan], [0.01, 0.01]]
    )
    np.testing.assert_array_equal(
        result["threshold"], [[nan, nan], [nan, nan], [0.005, 0.005]]
    )
    # the model update that did not run (code 0) is not counted
    assert list(result["model_update_stop_code_counts"]) == [1, 2]
    np.testing.assert_array_equal(
        result["model_update_stop_code_counts"][1], [0, nan, 2]
    )
    np.testing.assert_array_equal(result["entropy"], np.full((3, 2), 0.7))
    np.testing.assert_array_equal(result["residual_rms"], np.arange(6.0).reshape(3, 2))

    # the second time plane: one plane of channel 0 (no frequency field)
    return_dict = _deconvolution_return_dict()
    statistics = return_dict["image_statistics"]["sky_residual"]
    return_dict["image_statistics"]["sky_residual"] = xr.concat(
        [statistics, statistics.assign_coords(time=[1.0])], dim="time"
    )
    result = deconvolution_per_channel(return_dict, time_index=1)
    np.testing.assert_array_equal(
        result["iterations"], [[7, nan], [nan, nan], [nan, nan]]
    )
    return_dict["deconvolution"].data.pop(Key(1, 0, 0))
    assert deconvolution_per_channel(return_dict, time_index=1) is None


def test_deconvolution_per_channel_by_channel_number_and_refusals():
    return_dict = _deconvolution_return_dict(entropy=False)
    for fields in return_dict["deconvolution"].data.values():
        fields.pop("frequency", None)
        fields.pop("model_update_stop_code", None)
    return_dict["deconvolution"].data.pop(Key(1, 0, 0))
    with pytest.raises(ValueError, match="outside the 3 channels"):
        deconvolution_per_channel(return_dict)  # channel 5
    for pol in range(2):
        return_dict["deconvolution"].data.pop(Key(0, pol, 5))
    result = deconvolution_per_channel(return_dict)
    np.testing.assert_array_equal(result["iterations"][0], [15, 15])
    assert np.all(np.isnan(result["entropy"]))
    assert result["model_update_stop_code_counts"] == {}

    assert deconvolution_per_channel({"deconvolution": ImagingDict()}) is None
    assert (
        deconvolution_per_channel(
            {"deconvolution": ImagingDict(), "image_statistics": {}}
        )
        is None
    )
    empty = {**return_dict, "deconvolution": ImagingDict()}
    assert deconvolution_per_channel(empty) is None
    assert deconvolution_per_channel({**return_dict, "deconvolution": None}) is None
