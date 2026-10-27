"""The ``report`` of the image_cube_single_field distributed application, on a
small simulated processing set."""

import os

import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.processing_functions.imaging.utils import imaging_dict_to_dataframe
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
N_ANTENNA = 8
ARCSEC = np.pi / (180 * 3600)

CYCLE_CONTROL_PARAMS = {
    "max_iter": 40,
    "max_cycles": 3,
    "threshold": 0.0,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.5,
    "max_iter_per_cycle": 15,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
    "primary_beam_limit": 0.2,
}


@pytest.fixture(scope="module")
def processing_set(tmp_path_factory):
    """1 Jy unpolarised point source at the phase centre, 8 VLA antennas,
    2 channels, linear feeds (a FITS Stokes axis holds I and Q, not I and V)."""
    work_dir = tmp_path_factory.mktemp("report")
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(work_dir / "point.ps.zarr"),
        antenna_xds=read_telescope_layout(
            "vla.d", antenna_selection=list(range(N_ANTENNA))
        ),
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1800.0,
            "n_samples": 4,
        },
        frequency_params={
            "freq_start": 3e9,
            "freq_delta": 0.4e9,
            "n_channels": 2,
            "channel_width": 1e7,
        },
        polarization=["XX", "YY"],
        point_source_flux=np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        point_source_ra_dec=PC[None, None, :],
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(N_ANTENNA, int),
        n_time_chunks=1,
        n_frequency_chunks=2,
        overwrite=True,
    )
    return result["ps_store"]


def _image(processing_set, image_store, **kwargs):
    ps_xdt = open_processing_set(processing_set)
    arguments = {
        "ps_store": processing_set,
        "image_store": image_store,
        "image_params": {
            "image_size": [64, 64],
            "cell_size": np.array([-8.0, 8.0]) * ARCSEC,
            "phase_direction": PC,
            "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
            "polarization_coords": ["I", "Q"],
            "time_coords": [0],
            "fft_padding": 1.2,
        },
        "instrument_polarization_basis": "linear",
        "imaging_weights_params": {"weighting": "natural"},
        "cycle_control_params": CYCLE_CONTROL_PARAMS,
        "image_data_variables_keep": [
            "sky_residual",
            "sky_model",
            "point_spread_function",
            "primary_beam",
        ],
        "processing_set_data_group_name": "base",
        "single_precision_image": False,
        "restore": True,
        "primary_beam_correction": True,
        "n_mapping_parallelism": {"frequency": 2},
        "overwrite": True,
        **kwargs,
    }
    return distributed_applications.imaging.image_cube_single_field(**arguments)


def _text(path):
    with open(path, encoding="utf-8") as file:
        return file.read()


def test_html_report_of_a_monitored_single_task_run(processing_set, tmp_path):
    """One map task: the monitor's series reach the timing frame without a
    reduce node; the report has every section, the moment maps are kept and
    the residual statistics hold the final entropy."""
    pytest.importorskip("psutil")
    image_store = str(tmp_path / "cube.img.zarr")
    result = _image(
        processing_set,
        image_store,
        n_mapping_parallelism={"frequency": 1},
        monitor_resources_seconds=0.02,
        report="html",
    )
    assert result["report"] == str(tmp_path / "cube_report.html")
    text = _text(result["report"])
    for heading in (
        "Processing",
        "Results",
        "Point spread function",
        "Maximum over frequency",
        "Spectrum",
        "Deconvolution",
    ):
        assert f">{heading}</h" in text, heading
    assert "Cluster CPU" in text
    assert text.count("data:image/png;base64,") >= 8
    files = tmp_path / "cube_report_files"
    for key in ("sky_residual", "sky_model", "sky_restored_primary_beam_corrected"):
        assert (files / f"moment_maximum_{key}.img.zarr").exists(), key

    assert "cpu_percent" in result["timing_node_tasks"]
    assert len(result["timing_node_tasks"]) == 1
    timing = result["timing_distributed_application"]
    assert timing["T_total"] == pytest.approx(
        timing["application_end_unixtime"] - timing["application_start_unixtime"]
    )
    entropy = result["image_statistics"]["sky_residual"]["entropy"]
    assert entropy.dims == ("time", "frequency", "polarization")
    # the report keeps the beam fit with the point spread function
    assert "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION" in xr.open_zarr(image_store)
    record = imaging_dict_to_dataframe(result["deconvolution"])
    for codes, n_cycles in zip(
        record["model_update_stop_code"], record["n_cycles"], strict=True
    ):
        assert len(codes) == n_cycles  # one code per model update
    # the moment map of the residual is the maximum over the channels
    maximum = xr.open_zarr(files / "moment_maximum_sky_residual.img.zarr")
    np.testing.assert_allclose(
        maximum["SKY_MOMENT_MAXIMUM"].values[:, 0],
        xr.open_zarr(image_store)["SKY_RESIDUAL"].max(dim="frequency").values,
        rtol=1e-6,
    )


def test_no_report(processing_set, tmp_path):
    result = _image(processing_set, str(tmp_path / "cube.img.zarr"))
    assert result["report"] is None
    assert "entropy" not in result["image_statistics"]["sky_residual"]
    assert not (tmp_path / "cube_report_files").exists()


def test_notebook_report_and_a_failing_report(processing_set, tmp_path, monkeypatch):
    import IPython.display

    import astroviper.distributed_applications.imaging.image_cube_single_field_report as module

    shown = []
    monkeypatch.setattr(IPython.display, "display", shown.append)
    result = _image(processing_set, str(tmp_path / "cube.img.zarr"), report="notebook")
    assert result["report"] is None
    assert len(shown) == 1 and "Deconvolution" in shown[0].data

    def failing(*args, **kwargs):
        raise RuntimeError("no report")

    monkeypatch.setattr(module, "image_cube_single_field_report", failing)
    result = _image(processing_set, str(tmp_path / "again.img.zarr"), report="html")
    assert result["report"] is None  # logged; the run's results are intact
    assert not result["deconvolution"].data == {}
    assert not os.path.exists(tmp_path / "again_report.html")


def test_pdf_report_of_fits_output(processing_set, tmp_path):
    """FITS output: the moment maps read the FITS files of the run."""
    result = _image(
        processing_set,
        str(tmp_path / "cube.img.fits"),
        skunk_works=True,
        output_image_format="fits",
        report="pdf",
    )
    assert result["report"] == str(tmp_path / "cube_report.pdf")
    with open(result["report"], "rb") as file:
        assert file.read(4) == b"%PDF"
    files = tmp_path / "cube_report_files"
    for key in ("sky_residual", "sky_model", "sky_restored_primary_beam_corrected"):
        assert (files / f"moment_maximum_{key}.img.zarr").exists(), key


def test_report_after_the_run(processing_set, tmp_path):
    """The report function makes the report of a finished run; what the run
    did not write is left out with a note."""
    from astroviper.distributed_applications.imaging.image_cube_single_field_report import (
        image_cube_single_field_report,
    )

    image_store = str(tmp_path / "dirty.img.zarr")
    result = _image(
        processing_set,
        image_store,
        cycle_control_params={**CYCLE_CONTROL_PARAMS, "max_iter": 0, "max_cycles": 0},
        image_data_variables_keep=["sky_residual"],
        restore=False,
        primary_beam_correction=False,
    )
    path = image_cube_single_field_report(result, image_store, report="html")
    text = _text(path)
    assert "The point spread function was not kept" in text
    assert "No model: sky_model was not written." in text
    assert "it needs restore=True and primary_beam_correction=True" in text
    # the stop test of the dirty image is recorded; the entropy only with a report
    assert "Per channel: the cycle control stop code" in text
    assert "The final residual entropy is not a number for every plane" in text


def test_unknown_report_format_is_refused(processing_set, tmp_path):
    with pytest.raises(AssertionError):  # the param.json schema
        _image(processing_set, str(tmp_path / "cube.img.zarr"), report="docx")
    assert not (tmp_path / "cube.img.zarr").exists()
