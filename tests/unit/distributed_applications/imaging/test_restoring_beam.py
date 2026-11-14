"""``restoring_beam`` of the imaging drivers: the beam variables kept, three
floats in every task's parameters, a target equal to the PSF beam restoring bit
for bit, and the continuum refusal of a target that does not contain its PSF
beam before any model update."""

import functools
import importlib

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

import astroviper.node_tasks as node_tasks
from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from astroviper.distributed_applications.simulation import simulate_processing_set
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180 * 3600)
PHASE_CENTER = np.array([5.233686, 0.71094])
N_ANTENNA = 8
CYCLE_CONTROL_PARAMS = {
    "max_iter": 40,
    "max_cycles": 2,
    "threshold": 0.0,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.5,
    "max_iter_per_cycle": 15,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}


@pytest.fixture(scope="module")
def processing_set(tmp_path_factory):
    """1 Jy unpolarised point source at the phase centre, 8 VLA antennas in D
    configuration, 4 times, 2 channels at 3.0 and 3.4 GHz, linear feeds."""
    store = tmp_path_factory.mktemp("restoring_beam") / "point.ps.zarr"
    with dask.config.set(scheduler="synchronous"):
        simulate_processing_set(
            ps_store=str(store),
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
            sky_components=[
                {
                    "kind": "point",
                    "flux": np.array([1.0, 0.0, 0.0, 1.0]),
                    "ra_dec": PHASE_CENTER,
                }
            ],
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[airy_disk_model("vla_nvss")],
            beam_model_map=np.zeros(N_ANTENNA, int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    return str(store)


def _image_params(processing_set):
    frequencies = open_processing_set(processing_set).xr_ps.get_freq_axis().values
    return {
        "image_size": [64, 64],
        "cell_size": np.array([-8.0, 8.0]) * ARCSEC,
        "phase_direction": PHASE_CENTER,
        "frequency_coords": frequencies,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }


def _image_cube(processing_set, image_store, image_data_variables_keep, **kwargs):
    with dask.config.set(scheduler="synchronous"):
        image_cube_single_field(
            ps_store=processing_set,
            image_store=str(image_store),
            image_params=_image_params(processing_set),
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=CYCLE_CONTROL_PARAMS,
            image_data_variables_keep=image_data_variables_keep,
            processing_set_data_group_name="base",
            single_precision_image=False,
            n_mapping_parallelism={"frequency": 1},
            restore=True,
            restoration_output="both",
            **kwargs,
        )
    return xr.open_zarr(str(image_store)).load()


def _image_continuum(processing_set, image_store, **kwargs):
    image_params = _image_params(processing_set)
    image_params["n_taylor_terms"] = 1
    image_params["reference_frequency"] = float(
        np.mean(image_params["frequency_coords"])
    )
    with dask.config.set(scheduler="synchronous"):
        image_continuum_single_field(
            ps_store=processing_set,
            image_store=str(image_store),
            image_params=image_params,
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=CYCLE_CONTROL_PARAMS,
            processing_set_data_group_name="base",
            single_precision_image=False,
            n_mapping_parallelism={"frequency": 1},
            restore=True,
            restoration_output="both",
            **kwargs,
        )
    return xr.open_zarr(str(image_store)).load()


def test_a_cube_target_equal_to_the_psf_beam_restores_bit_for_bit(
    processing_set, tmp_path, monkeypatch
):
    """Channel 0 is restored to its own PSF beam as without a target; the
    target is kept although the caller's list (left unchanged) does not have
    it, the PSF beams not (only a report keeps them), and every task gets the
    target as three floats. Without a target the restored beam is the PSF
    beam of Stokes I, which the restore applies to Stokes Q too."""
    reference = _image_cube(
        processing_set,
        tmp_path / "none.img.zarr",
        ["sky_residual", "sky_model", "beam_fit_params_point_spread_function"],
    )
    psf_beam = reference.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values
    np.testing.assert_array_equal(
        reference.BEAM_FIT_PARAMS_SKY_RESTORED.values,
        np.repeat(psf_beam[:, :, :1], psf_beam.shape[2], axis=2),
    )
    target = psf_beam[0, 0, 0]

    received = []
    node_task = node_tasks.imaging.image_cube_single_field

    @functools.wraps(node_task)
    def recording_node_task(*args, **kwargs):
        received.append(kwargs["restoring_beam"])
        return node_task(*args, **kwargs)

    monkeypatch.setattr(
        node_tasks.imaging, "image_cube_single_field", recording_node_task
    )
    keep = ["sky_residual", "sky_model"]
    image = _image_cube(
        processing_set, tmp_path / "target.img.zarr", keep, restoring_beam=target
    )

    assert keep == ["sky_residual", "sky_model"]
    assert received == [list(target)]
    assert all(type(value) is float for value in received[0])
    channel = {"frequency": 0}
    for name in ("SKY_RESTORED", "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"):
        np.testing.assert_array_equal(
            image[name].isel(channel).values, reference[name].isel(channel).values
        )
    for name in ("SKY_RESIDUAL", "SKY_MODEL"):
        np.testing.assert_array_equal(image[name].values, reference[name].values)
    assert "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION" not in image
    restored_beam = image.BEAM_FIT_PARAMS_SKY_RESTORED.values
    assert restored_beam.shape == reference.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.shape
    np.testing.assert_array_equal(
        restored_beam.reshape(-1, 3), np.tile(target, (restored_beam.size // 3, 1))
    )


@pytest.fixture(scope="module")
def flagged_processing_set(processing_set, tmp_path_factory):
    """The processing set with channel 1 fully flagged."""
    import shutil

    import zarr

    store = tmp_path_factory.mktemp("flagged") / "point.ps.zarr"
    shutil.copytree(processing_set, store)
    for ms_name in open_processing_set(str(store)):
        zarr.open_array(f"{store}/{ms_name}/FLAG", mode="r+")[:, :, 1] = True
    return str(store)


def test_without_a_target_a_flagged_channel_holds_the_tclean_beam(
    flagged_processing_set, tmp_path
):
    """The task restores the flagged channel with no beam and records its null
    fit; once every task has written its beams the driver gives it the beam
    tclean gives it, that of largest area, channel 0's. The restored group
    records the beams."""
    image = _image_cube(
        flagged_processing_set,
        tmp_path / "none.img.zarr",
        ["beam_fit_params_point_spread_function"],
    )
    psf_beam = image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values
    np.testing.assert_array_equal(psf_beam[0, 1], 0.0)
    restored_beam = image.BEAM_FIT_PARAMS_SKY_RESTORED.values
    np.testing.assert_array_equal(
        restored_beam, np.broadcast_to(psf_beam[0, 0, 0], restored_beam.shape)
    )
    assert (
        image.attrs["data_groups"]["restored"]["beam_fit_params_sky"]
        == "BEAM_FIT_PARAMS_SKY_RESTORED"
    )


def test_fits_rows_without_a_fit_hold_the_tclean_beam(flagged_processing_set, tmp_path):
    """FITS output: after the run the BEAMS rows of the flagged channel hold the
    beam of largest area, channel 0's, in the restored image (the beams the
    restore applied, those of Stokes I) as in the residual image (the PSF
    beams); the primary beam corrected image has no beam."""
    from astropy.io import fits

    image_store = tmp_path / "none_fits"
    with dask.config.set(scheduler="synchronous"):
        image_cube_single_field(
            ps_store=flagged_processing_set,
            image_store=str(image_store),
            image_params=_image_params(flagged_processing_set),
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=CYCLE_CONTROL_PARAMS,
            image_data_variables_keep=[
                "sky_residual",
                "beam_fit_params_point_spread_function",
            ],
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 2},
            restore=True,
            restoration_output="both",
            skunk_works=True,
            output_image_format="fits",
        )
    for name in ("SKY_RESTORED", "SKY_RESIDUAL"):
        with fits.open(image_store / f"{name}.fits") as hdulist:
            beams = hdulist["BEAMS"].data
            rows = np.stack([beams[column] for column in ("BMAJ", "BMIN", "BPA")], -1)
        rows = rows.reshape(2, 2, 3)  # (channel, Stokes I and Q)
        assert np.all(rows[:, :, :2] > 0), name
        np.testing.assert_array_equal(rows[1], rows[0], err_msg=name)
        if name == "SKY_RESTORED":
            np.testing.assert_array_equal(rows[:, 1], rows[:, 0])
    with fits.open(image_store / "SKY_RESTORED_PRIMARY_BEAM_CORRECTED.fits") as hdulist:
        assert len(hdulist) == 1
        assert "CASAMBM" not in hdulist[0].header
        assert "BMAJ" not in hdulist[0].header


def _record_at_report(monkeypatch, read):
    """What ``read()`` returns when the driver starts its report."""
    import astroviper.distributed_applications.imaging.image_cube_single_field_report as report_module

    recorded = []
    report_function = report_module.image_cube_single_field_report

    def recording_report(*args, **kwargs):
        recorded.append(read())
        return report_function(*args, **kwargs)

    monkeypatch.setattr(
        report_module, "image_cube_single_field_report", recording_report
    )
    return recorded


def test_zarr_restored_beams_are_filled_before_the_report(
    flagged_processing_set, tmp_path, monkeypatch
):
    """The report, which an exit of the process can cut short, starts once the
    flagged channel, imaged by a task of its own, holds the tclean beam; told
    that the run had no restoring_beam, the report does not take the now equal
    restored beams for one."""
    import html

    import zarr

    image_store = str(tmp_path / "none.img.zarr")
    recorded = _record_at_report(
        monkeypatch,
        lambda: zarr.open_group(image_store, mode="r")["BEAM_FIT_PARAMS_SKY_RESTORED"][
            ...
        ],
    )
    with dask.config.set(scheduler="synchronous"):
        result = image_cube_single_field(
            ps_store=flagged_processing_set,
            image_store=image_store,
            image_params=_image_params(flagged_processing_set),
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=CYCLE_CONTROL_PARAMS,
            image_data_variables_keep=["sky_residual"],
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 2},
            restore=True,
            report="html",
        )
    assert len(recorded) == 1
    np.testing.assert_array_equal(recorded[0][0, 1], recorded[0][0, 0])
    assert np.all(recorded[0][..., :2] > 0)
    with open(result["report"], encoding="utf-8") as file:
        text = html.unescape(file.read())
    assert "No restoring beam: every plane was restored with the PSF fit" in text


def test_fits_rows_are_filled_before_the_report(
    flagged_processing_set, tmp_path, monkeypatch
):
    """With a restoring beam: the report starts once the rows of the flagged
    channel are filled, and names that channel as without a PSF fit in the
    section of the point spread function (its centre channel) and in that of
    the restoring beam."""
    import html

    from astropy.io import fits

    image_store = tmp_path / "target_fits"

    def read_rows():
        with fits.open(image_store / "SKY_RESIDUAL.fits") as hdulist:
            beams = hdulist["BEAMS"].data
            return np.stack([beams[column] for column in ("BMAJ", "BMIN")], -1)

    recorded = _record_at_report(monkeypatch, read_rows)
    with dask.config.set(scheduler="synchronous"):
        result = image_cube_single_field(
            ps_store=flagged_processing_set,
            image_store=str(image_store),
            image_params=_image_params(flagged_processing_set),
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params=CYCLE_CONTROL_PARAMS,
            image_data_variables_keep=["sky_residual", "point_spread_function"],
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 2},
            restore=True,
            restoring_beam=[300 * ARCSEC, 300 * ARCSEC, 0.0],
            skunk_works=True,
            output_image_format="fits",
            report="html",
        )
    assert len(recorded) == 1 and np.all(recorded[0] > 0)
    with open(result["report"], encoding="utf-8") as file:
        text = html.unescape(file.read())
    assert "The beam fit of the centre channel failed" in text
    assert text.count("<td>no fit</td>") == 1


def test_a_continuum_target_equal_to_the_psf_beam_restores_bit_for_bit(
    processing_set, tmp_path, monkeypatch
):
    """The restored Taylor stacks equal those without a target; the beam
    variables are kept although the caller's list (left unchanged) has
    neither, and the restore gets the target as three floats."""
    import astroviper.processing_functions.imaging.image_continuum_single_field as continuum_processing

    reference = _image_continuum(processing_set, tmp_path / "none.img.zarr")
    assert "BEAM_FIT_PARAMS_SKY_RESTORED" not in reference
    target = reference.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values.reshape(-1, 3)[0]

    received = []
    restore = continuum_processing.restore_continuum_image

    def recording_restore(*args, **kwargs):
        received.append(kwargs["restoring_beam"])
        return restore(*args, **kwargs)

    monkeypatch.setattr(
        continuum_processing, "restore_continuum_image", recording_restore
    )
    keep = ["sky_residual"]
    image = _image_continuum(
        processing_set,
        tmp_path / "target.img.zarr",
        image_data_variables_keep=keep,
        restoring_beam=target,
    )

    assert keep == ["sky_residual"]
    assert received == [list(target)]
    assert all(type(value) is float for value in received[0])
    for name in ("SKY_RESTORED_FLAT_NOISE", "SKY_RESTORED_FLAT_SKY", "SKY_RESIDUAL"):
        np.testing.assert_array_equal(image[name].values, reference[name].values)
    np.testing.assert_array_equal(
        image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
        reference.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
    )
    restored_beam = image.BEAM_FIT_PARAMS_SKY_RESTORED.values.reshape(-1, 3)
    np.testing.assert_array_equal(
        restored_beam, np.tile(target, (len(restored_beam), 1))
    )


def test_a_continuum_target_that_does_not_contain_the_psf_beam_is_refused_before_any_model_update(
    processing_set, tmp_path, monkeypatch
):
    import astroviper.node_tasks.imaging.image_continuum_single_field as continuum_node

    # The package exports the function common_beam under the module's name.
    common_beam = importlib.import_module(
        "astroviper.processing_functions.image_analysis.common_beam"
    )

    def unexpected_model_update(*args, **kwargs):
        pytest.fail("A model update ran before the restoring_beam was refused")

    monkeypatch.setattr(
        continuum_node, "model_update_continuum_single_field", unexpected_model_update
    )
    refusals = []
    correcting_beam = common_beam.correcting_beam

    def recording_correcting_beam(beam_fit_params, restoring_beam):
        try:
            return correcting_beam(beam_fit_params, restoring_beam)
        except ValueError as error:
            refusals.append(error)
            raise

    monkeypatch.setattr(common_beam, "correcting_beam", recording_correcting_beam)
    with pytest.raises(ValueError) as refusal:
        _image_continuum(
            processing_set,
            tmp_path / "refused.img.zarr",
            restoring_beam=[ARCSEC, ARCSEC, 0.0],
        )
    assert [str(error) for error in refusals] == [str(refusal.value)]
