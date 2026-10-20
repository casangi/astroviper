"""Component test: continuum imaging with the Asp deconvolver on simulated data.

No downloads: the simulator writes the input processing set. A 6 x 30 min
VLA D configuration observation with circular feeds (``RR``, ``LL`` in that
order), two channels at 3.0 and 3.4 GHz and thermal noise sees a 2 Jy point
source 4.2 arcmin from the phase centre and a 1 Jy Gaussian of
60 x 40 arcsec. The simulator also writes the true sky on the imaging grid.

The public continuum application images it with ``deconvolver="asp"`` and one
Taylor term (128 x 128 pixels of 8 arcsec), in MFS and in MVC, with the Asp
options of the imager on (``asp_residual_components`` and
``asp_scale_significance``, the default) and off (Asp as in CASA), and with
restoration and primary-beam correction. Every run images its own copy of the
processing set, with ``weight_memory_mode="in_memory"`` so that the input is
not written. Without a clean mask the significance test of
``asp_scale_significance`` cannot act in MVC: Asp measures the noise outside
its mask, the pixels outside the primary-beam support, where MVC sets the
residual to 0. The application warns about it in MVC with the options on, and
only there.

The CLEAN model of each source (a box around it) must hold the true flux
times the effective primary beam of the image (the model of continuum
imaging is the sky seen through that beam, in MFS and MVC alike) within
5 percent (measured: within 2.2 percent), the Stokes I residual must stay
below 3 percent of the point source's apparent flux (measured: at most
1.3 percent), and the cycle controller must have booked every model update.
A second test gives the application an explicit ``clean_mask`` around the
point source only.
"""

import shutil

import dask
import numpy as np
import pytest
import toolviper.utils.logger
import xarray as xr
from astropy.coordinates import SkyCoord
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging import image_continuum_single_field
from astroviper.processing_functions.imaging.utils.cycle_control import (
    MODEL_UPDATE_STOP_DESCRIPTIONS,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import celestial_coord_to_sin_pixel
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
POINT = SkyCoord(ra="19h59m50.51793355s", dec="+40d48m11.3694551s", frame="fk5")
GAUSSIAN = SkyCoord(ra="19h59m20.0s", dec="+40d42m30.0s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
SOURCES = {
    "point": (np.array([POINT.ra.rad, POINT.dec.rad]), 2.0, 4),
    "gaussian": (np.array([GAUSSIAN.ra.rad, GAUSSIAN.dec.rad]), 1.0, 12),
}
IMAGE_SIZE = [128, 128]
CELL_SIZE = np.array([-8.0, 8.0]) * np.pi / (180 * 3600)
ARCSEC = np.pi / (180 * 3600)
OPTIONS = {
    "asp_options_on": {"asp_residual_components": True, "asp_scale_significance": True},
    "asp_options_off": {
        "asp_residual_components": False,
        "asp_scale_significance": False,
    },
}


def simulate_input(directory):
    """Simulate the processing set and the true sky on the imaging grid."""
    point, point_flux, _ = SOURCES["point"]
    gaussian, gaussian_flux, _ = SOURCES["gaussian"]
    truth_store = str(directory / "truth.img.zarr")
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(directory / "input.ps.zarr"),
        antenna_xds=read_telescope_layout("vla.d"),
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1800.0,
            "n_samples": 6,
        },
        frequency_params={
            "freq_start": 3e9,
            "freq_delta": 0.4e9,
            "n_channels": 2,
            "channel_width": 1e7,
        },
        polarization=["RR", "LL"],
        point_source_flux=np.array([point_flux, 0, 0, point_flux])[None, None, None],
        point_source_ra_dec=point[None, None, :],
        gaussian_source_flux=np.array([gaussian_flux, 0, 0, gaussian_flux])[
            None, None, None
        ],
        gaussian_source_ra_dec=gaussian[None, None, :],
        gaussian_source_shape=np.array([[60.0 * ARCSEC, 40.0 * ARCSEC, 0.5]]),
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(27, int),
        noise_params={"t_receiver": 50.0, "random_seed": 5},
        sky_image_params={
            "image_store": truth_store,
            "image_size": IMAGE_SIZE,
            "cell_size": CELL_SIZE,
            "polarization_coords": ["I"],
        },
        n_time_chunks=2,
        n_frequency_chunks=2,
        overwrite=True,
    )
    truth = load_image(truth_store).SKY.values[0, :, 0]  # [frequency, l, m]
    return result["ps_store"], truth.mean(axis=0)


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """The simulated processing set and the true sky on the imaging grid."""
    return simulate_input(tmp_path_factory.mktemp("continuum_asp"))


def run_continuum_asp(simulated, directory, continuum_mode, options, clean_mask=None):
    """Image a copy of the simulated processing set with continuum Asp
    (``clean_mask``: the path of an explicit clean mask, or ``None``)."""
    source_store, _ = simulated
    ps_store = str(directory / "input.ps.zarr")
    shutil.copytree(source_store, ps_store)
    processing_set = open_processing_set(ps_store)
    fields = processing_set.xr_ps.get_combined_field_and_source_xds()
    frequency = processing_set.xr_ps.get_freq_axis().values
    reference_frequency = float(np.mean(frequency))
    image_store = str(directory / "asp.img.zarr")
    with dask.config.set(scheduler="synchronous"):
        result = image_continuum_single_field(
            ps_store=ps_store,
            image_store=image_store,
            image_params={
                "image_size": IMAGE_SIZE,
                "cell_size": CELL_SIZE,
                "phase_direction": fields.FIELD_PHASE_CENTER_DIRECTION.sel(
                    field_name=fields.attrs["center_field_name"]
                ).values,
                "frequency_coords": frequency,
                "polarization_coords": ["I", "V"],
                "time_coords": [0],
                "fft_padding": 1.2,
                "cpp_gridder": True,
                "n_taylor_terms": 1,
                "reference_frequency": reference_frequency,
            },
            imaging_weights_params={
                "weighting": "natural",
                "robust": 0.5,
                "casa_weighting_implementation": True,
            },
            cycle_control_params={
                "max_iter": 300,
                "max_cycles": 5,
                "max_iter_per_cycle": 100,
                "threshold": 0.0,
                "gain": 0.1,
                "psf_sidelobe_factor": 1.5,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
                **options,
            },
            gridder="anti_alias",
            deconvolver="asp",
            clean_mask=clean_mask,
            continuum_mode=continuum_mode,
            instrument_polarization_basis="circular",
            restore=True,
            primary_beam_correction=True,
            primary_beam_limit=0.2,
            scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
            processing_set_data_group_name="base",
            image_data_variables_keep=[
                "sky_model",
                "sky_residual",
                "point_spread_function",
                "primary_beam",
                "beam_fit_params_point_spread_function",
            ],
            single_precision_image=False,
            processing_function_threads=1,
            n_mapping_parallelism={"frequency": 2},
            memory_mode="in_memory",
            weight_memory_mode="in_memory",
            visibility_memory_mode="recompute",
            wideband_primary_beam_memory_mode="in_memory",
            overwrite=True,
            compute_backend="dask",
        )
    with xr.open_zarr(image_store) as stored:
        image = stored.load()
    return result, image


def source_boxes():
    """Pixel box (slices) around every source."""
    boxes = {}
    for name, (position, _, half_width) in SOURCES.items():
        i, j = np.rint(
            celestial_coord_to_sin_pixel(PC, IMAGE_SIZE, CELL_SIZE, position)
        ).astype(int)
        boxes[name] = (
            slice(i - half_width, i + half_width + 1),
            slice(j - half_width, j + half_width + 1),
        )
    return boxes


def significance_warnings(monkeypatch):
    """Collect the application's warnings about ``asp_scale_significance``."""
    messages = []

    def warning(message, *args, **kwargs):
        if "asp_scale_significance" in message:
            messages.append(message)

    monkeypatch.setattr(toolviper.utils.logger, "warning", warning)
    return messages


@pytest.mark.parametrize("options", list(OPTIONS), ids=list(OPTIONS))
@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_continuum_asp_recovers_simulated_sources(
    tmp_path, simulated, monkeypatch, continuum_mode, options
):
    warnings = significance_warnings(monkeypatch)
    result, image = run_continuum_asp(
        simulated, tmp_path, continuum_mode, OPTIONS[options]
    )
    _, truth = simulated
    controller = result["controller"]

    # MVC without a clean mask leaves the significance test no noise to measure.
    assert len(warnings) == int(continuum_mode == "mvc" and options == "asp_options_on")

    assert image.sizes["taylor_term"] == 1
    assert image.sizes["frequency"] == 1
    assert list(image.polarization.values) == ["I", "V"]
    assert controller.cycles_done >= 2
    # Every model update is booked: the iterations of the per-plane records
    # add up to the controller's total, and every plane has a known stop code.
    planes = result["deconvolution"].data.values()
    assert sum(int(np.sum(fields["iter_done"])) for fields in planes) == (
        controller.total_iter_done
    )
    assert 0 < controller.total_iter_done <= 300 * image.sizes["polarization"]
    assert set(np.unique(controller.stop_code_model_update)) <= set(
        MODEL_UPDATE_STOP_DESCRIPTIONS
    )

    model = image.SKY_MODEL.values[0, 0, 0]  # Stokes I, Taylor term 0
    residual = image.SKY_RESIDUAL.values[0, 0, 0]
    primary_beam = image.PRIMARY_BEAM.isel(frequency=0).values[0, 0]
    apparent = {}
    for name, box in source_boxes().items():
        apparent[name] = float(np.sum(truth[box] * primary_beam[box]))
        np.testing.assert_allclose(
            model[box].sum(),
            apparent[name],
            rtol=0.05,
            err_msg=f"{continuum_mode} {name}",
        )
    assert np.isfinite(residual).all() and np.isfinite(model).all()
    assert np.abs(residual).max() < 0.03 * apparent["point"]

    # Restore and primary-beam correction as for Hogbom.
    beam = image.PRIMARY_BEAM.isel(frequency=0, drop=True)
    for name in ("SKY_RESTORED", "SKY_RESTORED_PRIMARY_BEAM_CORRECTED"):
        assert name in image
    np.testing.assert_array_equal(
        np.isfinite(image.SKY_RESTORED_PRIMARY_BEAM_CORRECTED),
        np.isfinite(beam) & (beam >= 0.2),
    )


@pytest.mark.parametrize("options", list(OPTIONS), ids=list(OPTIONS))
@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_continuum_asp_with_an_explicit_clean_mask(
    tmp_path, simulated, monkeypatch, continuum_mode, options
):
    """An explicit ``clean_mask`` (a ``.npy`` file) of 17 x 17 pixels around
    the point source only: Asp cleans the point source and leaves the Gaussian
    in the residual. The model of the point source holds its apparent flux
    within 8 percent (measured: within 5.2 percent), the model of the Gaussian
    less than 3 percent of its own (measured: at most 1.3 percent; the wings
    of Asp components reach past the mask), the residual in the mask stays
    below 3 percent of the point source's apparent flux (measured: at most
    1.9 percent) and the Gaussian's residual peak is more than 10 times the
    largest residual in the mask (measured: at least 13 times). With the Asp
    options on, the noise of the significance test per scale is measured on
    the pixels outside the mask (in MVC these include the pixels outside the
    primary-beam support, where the residual is 0), and the application logs
    no warning about it."""
    warnings = significance_warnings(monkeypatch)
    point_box = source_boxes()["point"]
    i, j = (box.start + 4 for box in point_box)  # the point source's pixel
    mask = np.zeros(IMAGE_SIZE, dtype=bool)
    mask[i - 8 : i + 9, j - 8 : j + 9] = True
    mask_file = tmp_path / "point_source_mask.npy"
    np.save(mask_file, mask)

    result, image = run_continuum_asp(
        simulated,
        tmp_path,
        continuum_mode,
        OPTIONS[options],
        clean_mask=str(mask_file),
    )
    _, truth = simulated
    controller = result["controller"]

    assert warnings == []
    assert controller.cycles_done >= 2
    planes = result["deconvolution"].data.values()
    assert sum(int(np.sum(fields["iter_done"])) for fields in planes) == (
        controller.total_iter_done
    )
    model = image.SKY_MODEL.values[0, 0, 0]  # Stokes I, Taylor term 0
    residual = image.SKY_RESIDUAL.values[0, 0, 0]
    primary_beam = image.PRIMARY_BEAM.isel(frequency=0).values[0, 0]
    apparent = {
        name: float(np.sum(truth[box] * primary_beam[box]))
        for name, box in source_boxes().items()
    }
    gaussian_box = source_boxes()["gaussian"]
    assert mask[np.unravel_index(np.argmax(model), model.shape)]
    np.testing.assert_allclose(model[point_box].sum(), apparent["point"], rtol=0.08)
    assert abs(model[gaussian_box].sum()) < 0.03 * apparent["gaussian"]
    assert np.abs(residual[mask]).max() < 0.03 * apparent["point"]
    assert residual[gaussian_box].max() > 10 * np.abs(residual[mask]).max()
