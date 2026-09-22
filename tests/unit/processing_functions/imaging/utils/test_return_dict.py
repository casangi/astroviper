"""Unit tests for the ReturnDict -> DataFrame flattening."""

from astroviper.processing_functions.imaging.utils.iteration_control import StopCode
from astroviper.processing_functions.imaging.utils.return_dict import (
    ReturnDict,
    return_dict_to_dataframe,
)


def _make_return_dict():
    rd = ReturnDict()
    # channel 0: deconvolved over 2 cycles
    for cycle in range(2):
        rd.add(
            {
                "iter_done": [40, 25][cycle],
                "peakres": [0.05, 0.002][cycle],
                "start_peakres": [0.11, 0.05][cycle],
                "model_flux": [0.3, 0.42][cycle],
                "masksum": 100,
                "niter": 5000,
                "loop_gain": 0.1,
                "max_psf_sidelobe": 0.35,
            },
            time=0,
            pol=0,
            chan=0,
        )
    rd.add(
        {"stop_code": StopCode(major=2, minor=0), "stop_description": "threshold"},
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
            "stop_code": StopCode(major=2, minor=0),
            "stop_description": "threshold",
            "niter": 5000,
        },
        time=0,
        pol=1,
        chan=1,
    )
    return rd


def test_flattens_history_and_stop_code():
    df = return_dict_to_dataframe(_make_return_dict())
    assert len(df) == 2
    row0 = df[(df.pol == 0) & (df.chan == 0)].iloc[0]
    assert row0["n_cycles"] == 2
    assert row0["iter_total"] == 65
    assert row0["peakres_start"] == 0.11
    assert row0["peakres_final"] == 0.002
    assert row0["model_flux_final"] == 0.42
    assert row0["stop_code_major"] == 2
    assert row0["iter_done"] == [40, 25]  # history preserved as a list
    row1 = df[(df.pol == 1) & (df.chan == 1)].iloc[0]
    assert row1["iter_total"] == 0  # the "never deconvolved" marker
    assert row1["stop_description"] == "threshold"


def test_empty_dict_gives_empty_frame():
    df = return_dict_to_dataframe(ReturnDict())
    assert df.empty
    assert "iter_total" in df.columns
