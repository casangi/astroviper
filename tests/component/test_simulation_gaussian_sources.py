"""Component test: Gaussian + point sources, ASP CLEAN, primary-beam-corrected fluxes.

Simulates two point sources and two elliptical-Gaussian sources with the
simulator (Gaussians via the imaging restore module's
``elliptical_gaussian_uv_taper``), images the result with the Adaptive Scale
Pixel deconvolver, and checks that after primary-beam correction the input
fluxes are recovered: point-source peaks equal their fluxes, Gaussian
aperture-integrated fluxes equal their integrated fluxes, and Gaussian peaks
match the analytic clean-beam convolution.
"""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

# OSError: an editable install with the simulation sources removed raises
# FileNotFoundError instead of ImportError.
try:
    import astroviper.distributed_applications.simulation  # noqa: F401
except (ImportError, OSError):
    pytest.skip(
        "requires the SIRIUS simulation port (branch 265-port-sirius)",
        allow_module_level=True,
    )

import astroviper.distributed_applications as distributed_applications  # noqa: E402
from astroviper.utils.beam_models import airy_disk_model  # noqa: E402
from astroviper.utils.coordinate_transforms import (  # noqa: E402
    sin_pixel_to_celestial_coord,
)
from astroviper.utils.telescope_layout import read_telescope_layout  # noqa: E402

ARC = np.pi / (180 * 3600)
PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="-40d44m01.5s", frame="icrs")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])[None, :]
IMAGE_SIZE = np.array([256, 256])
CELL_SIZE = np.array([-0.5, 0.5]) * ARC

PIXELS_POINT = np.array([[128, 128], [88, 168]])
FLUX_POINT = [1.0, 1.5]
PIXELS_GAUSSIAN = np.array([[168, 168], [128, 68]])
FLUX_GAUSSIAN = [2.0, 1.2]
SHAPE_GAUSSIAN = np.array([[4.0 * ARC, 4.0 * ARC, 0.0], [6.0 * ARC, 2.5 * ARC, 0.5]])


def _covariance(major, minor, pa):
    """Sky covariance matrix of a Gaussian given FWHM axes and position angle."""
    sigma_major = (major / (2 * np.sqrt(2 * np.log(2)))) ** 2
    sigma_minor = (minor / (2 * np.sqrt(2 * np.log(2)))) ** 2
    e = np.array([np.sin(pa), np.cos(pa)])
    p = np.array([np.cos(pa), -np.sin(pa)])
    return sigma_major * np.outer(e, e) + sigma_minor * np.outer(p, p)


def test_gaussian_and_point_source_fluxes_recovered(tmp_path):
    ps_store = str(tmp_path / "gaussian_sim.ps.zarr")
    image_store = str(tmp_path / "gaussian_sim.img.zarr")

    point_pos = sin_pixel_to_celestial_coord(PC[0], IMAGE_SIZE, CELL_SIZE, PIXELS_POINT)
    gaussian_pos = sin_pixel_to_celestial_coord(
        PC[0], IMAGE_SIZE, CELL_SIZE, PIXELS_GAUSSIAN
    )
    ant = read_telescope_layout("alma.all")
    ant = ant.isel(
        antenna_name=np.where(ant.ANTENNA_DISH_DIAMETER.values == 12.0)[0][:30]
    )
    distributed_applications.simulation.simulate_processing_set(
        ps_store=ps_store,
        antenna_xds=ant,
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1600.0,
            "n_samples": 6,
        },
        frequency_params={
            "freq_start": 90e9,
            "freq_delta": 0.5e9,
            "n_channels": 1,
            "channel_width": 0.5e9,
        },
        polarization=["XX", "YY"],
        point_source_flux=np.stack([np.array([f, 0, 0, f]) for f in FLUX_POINT])[
            :, None, None, :
        ],
        point_source_ra_dec=point_pos[None],
        gaussian_source_flux=np.stack([np.array([f, 0, 0, f]) for f in FLUX_GAUSSIAN])[
            :, None, None, :
        ],
        gaussian_source_ra_dec=gaussian_pos[None],
        gaussian_source_shape=SHAPE_GAUSSIAN,
        phase_center_ra_dec=PC,
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(30, int),
        n_time_chunks=2,
        n_frequency_chunks=1,
        overwrite=True,
    )

    ps_xdt = open_processing_set(ps_store)
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    phase_direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values
    distributed_applications.imaging.image_cube_single_field(
        ps_store=ps_store,
        image_store=image_store,
        image_params={
            "image_size": list(IMAGE_SIZE),
            "cell_size": CELL_SIZE,
            "phase_direction": phase_direction,
            "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
            "polarization_coords": ["I"],
            "time_coords": [0],
            "fft_padding": 1.2,
            "cpp_gridder": True,
        },
        imaging_weights_params={
            "weighting": "natural",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "niter": 3000,
            "nmajor": -1,
            "threshold": 2e-4,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
            "primary_beam_limit": 0.2,
        },
        gridder="prolate_spheroidal",
        deconvolver="asp",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
            "sky_model",
            "mask",
        ],
        processing_set_data_group_name="base",
        single_precision_image=False,
        processing_function_threads=1,
        n_chunks=1,
        overwrite=True,
        restore=True,
        primary_beam_correction=True,
    )

    img = load_image(image_store)
    corrected = img.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values[0, 0, 0]
    primary_beam = img.PRIMARY_BEAM.values[0, 0, 0]
    beam = img.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0, 0, 0]
    beam_area = np.pi / (4 * np.log(2)) * beam[0] * beam[1]
    pixel_area = float(np.abs(CELL_SIZE[0] * CELL_SIZE[1]))

    # Point sources: the PB-corrected restored peak equals the sky flux.
    for pix, flux in zip(PIXELS_POINT, FLUX_POINT, strict=True):
        assert primary_beam[pix[0], pix[1]] < 1.01  # attenuated away from centre
        np.testing.assert_allclose(corrected[pix[0], pix[1]], flux, rtol=0.02)

    beam_covariance = _covariance(*beam)
    for pix, flux, shape in zip(
        PIXELS_GAUSSIAN, FLUX_GAUSSIAN, SHAPE_GAUSSIAN, strict=True
    ):
        # Aperture-integrated PB-corrected flux recovers the integrated flux.
        box = corrected[pix[0] - 25 : pix[0] + 25, pix[1] - 25 : pix[1] + 25]
        integrated = np.nansum(box) * pixel_area / beam_area
        np.testing.assert_allclose(integrated, flux, rtol=0.05)
        # The peak matches the analytic convolution with the fitted clean beam:
        # sky and clean-beam covariances add under convolution.
        source_covariance = _covariance(*shape)
        expected_peak = flux * np.sqrt(
            np.linalg.det(beam_covariance)
            / np.linalg.det(beam_covariance + source_covariance)
        )
        np.testing.assert_allclose(corrected[pix[0], pix[1]], expected_peak, rtol=0.03)
