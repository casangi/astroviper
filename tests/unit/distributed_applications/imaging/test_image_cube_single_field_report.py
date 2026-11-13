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
        "restoration_output": "both",
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
        "Restoring beam",
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


def test_report_of_a_run_with_a_restoring_beam(processing_set, tmp_path):
    """A 250" circle contains the PSF beam of channel 1 (213" x 189") but not
    that of channel 0 (290" x 106"): the restored image is labelled with it in
    every plane, and the report names channel 0 with the growth the circle
    would need, the ratio of the major axes."""
    import html

    image_store = str(tmp_path / "cube.img.zarr")
    restoring_beam = [250 * ARCSEC, 250 * ARCSEC, 0.0]
    result = _image(
        processing_set, image_store, restoring_beam=restoring_beam, report="html"
    )
    stored = xr.open_zarr(image_store)
    assert (
        stored.attrs["data_groups"]["restored"]["beam_fit_params_sky"]
        == "BEAM_FIT_PARAMS_SKY_RESTORED"
    )
    restored_beam = stored["BEAM_FIT_PARAMS_SKY_RESTORED"].values
    np.testing.assert_array_equal(
        restored_beam, np.broadcast_to(restoring_beam, restored_beam.shape)
    )
    psf_major = stored["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].values[0, :, 0, 0]
    assert psf_major[0] > restoring_beam[0] > psf_major[1]
    text = html.unescape(_text(result["report"]))
    assert 'Every plane was restored to the beam 250" x 250"' in text
    assert text.count("<td>not contained</td>") == 1
    assert f"<td>{psf_major[0] / restoring_beam[0]:.10g}</td>" in text


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
        restoration_output="apparent",
    )
    path = image_cube_single_field_report(result, image_store, report="html")
    text = _text(path)
    assert "The point spread function was not kept" in text
    assert "No restoring beam: beam_fit_params_sky_restored was not written" in text
    assert "No model: sky_model was not written." in text
    assert "it needs restore=True and restoration_output=" in text
    # the stop test of the dirty image is recorded; the entropy only with a report
    assert "Per channel: the cycle control stop code" in text
    assert "The final residual entropy is not a number for every plane" in text


def test_unknown_report_format_is_refused(processing_set, tmp_path):
    with pytest.raises(AssertionError):  # the param.json schema
        _image(processing_set, str(tmp_path / "cube.img.zarr"), report="docx")
    assert not (tmp_path / "cube.img.zarr").exists()


def _beam_store(tmp_path, output_image_format, psf_beams, restoring_beam):
    """An image store with the PSF fits ``psf_beams`` (one per channel, equal
    in Stokes I and Q) and ``restoring_beam`` in every plane, as a run writes
    them; returns the store and its kept variables."""
    from xradio.image import make_empty_sky_image

    from astroviper.node_tasks.imaging.utils import (
        create_empty_fits_images,
        write_result_chunk_to_fits_skunk_works,
    )
    from astroviper.utils.io import write_zarr_image_store

    n_chan = len(psf_beams)
    img_xds = make_empty_sky_image(
        phase_center=PC,
        image_size=[8, 8],
        cell_size=np.array([-1.0, 1.0]) * ARCSEC,
        frequency_coords=1e11 + 1e7 * np.arange(n_chan),
        pol_coords=["I", "Q"],
        time_coords=[60000.0],
        do_sky_coords=False,
    )
    image_dims = ["time", "frequency", "polarization", "l", "m"]
    beam_dims = ["time", "frequency", "polarization", "beam_params_label"]
    psf = np.repeat(np.asarray(psf_beams)[None, :, None], 2, axis=2)
    img_xds = img_xds.assign(
        SKY_RESIDUAL=(image_dims, np.zeros((1, n_chan, 2, 8, 8))),
        SKY_RESTORED=(image_dims, np.zeros((1, n_chan, 2, 8, 8))),
        BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION=(beam_dims, psf),
        BEAM_FIT_PARAMS_SKY_RESTORED=(
            beam_dims,
            np.broadcast_to(restoring_beam, psf.shape).copy(),
        ),
    )
    keep = [
        "sky_residual",
        "sky_restored",
        "beam_fit_params_point_spread_function",
        "beam_fit_params_sky_restored",
    ]
    if output_image_format == "zarr":
        return write_zarr_image_store(img_xds, str(tmp_path / "cube.img.zarr")), keep
    store = str(tmp_path / "cube.img.fits")
    create_empty_fits_images(store, img_xds, keep)
    write_result_chunk_to_fits_skunk_works(
        store, keep, {"frequency": {"slice": slice(0, n_chan)}}, img_xds
    )
    return store, keep


def _restoring_beam_section(tmp_path, store, output_image_format, available):
    import html

    from astroviper.distributed_applications.imaging.image_cube_single_field_report import (
        _add_restoring_beam,
    )
    from astroviper.utils.report import Report

    path = str(tmp_path / "section.html")
    with Report("restoring beam", "html", path) as document:
        _add_restoring_beam(document, store, output_image_format, available)
    return html.unescape(_text(path))


@pytest.mark.parametrize("output_image_format", ["zarr", "fits"])
def test_restoring_beam_section(tmp_path, output_image_format):
    """The restoring beam, the ratios of its axes to those of the PSF fits, and
    the channels whose residual was left uncorrected: one without a fit, and
    one whose 2" x 1" beam the 1.5" circle does not contain (it would have to
    grow by 2 / 1.5)."""
    psf_beams = np.array(
        [[1.2, 1.0, 0.3], [np.nan, np.nan, np.nan], [2.0, 1.0, 0.0], [1.4, 1.1, 1.0]]
    )
    psf_beams[:, :2] *= ARCSEC
    store, keep = _beam_store(
        tmp_path, output_image_format, psf_beams, [1.5 * ARCSEC, 1.5 * ARCSEC, 0.0]
    )
    text = _restoring_beam_section(tmp_path, store, output_image_format, set(keep))
    assert ">Restoring beam</h3>" in text
    assert 'restored to the beam 1.5" x 1.5", position angle 0.0 deg' in text
    assert "0.7500 to 1.2500 times the major axes" in text
    assert "1.3636 to 1.5000 times the minor axes" in text
    assert text.count("data:image/png;base64,") == 1
    assert text.count("<td>no fit</td>") == 1
    assert text.count("<td>not contained</td>") == 1
    assert "<td>1.333333333</td>" in text


def test_restoring_beam_section_without_uncorrected_channels(tmp_path):
    psf_beams = np.array([[1.2, 1.0, 0.3], [1.4, 1.1, 1.0]])
    psf_beams[:, :2] *= ARCSEC
    store, keep = _beam_store(tmp_path, "zarr", psf_beams, [2 * ARCSEC, 2 * ARCSEC, 0])
    text = _restoring_beam_section(tmp_path, store, "zarr", set(keep))
    assert "The restoring beam contains the PSF beam of every channel." in text
    assert "<table" not in text
    # without the PSF fits only the restoring beam is shown
    available = set(keep) - {"beam_fit_params_point_spread_function"}
    text = _restoring_beam_section(tmp_path, store, "zarr", available)
    assert 'restored to the beam 2" x 2"' in text
    assert "The beam fit of the point spread function was not kept" in text
