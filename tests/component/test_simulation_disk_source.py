"""Component test: an inclined limb-darkened disk (protoplanetary disk) observed by ALMA.

Simulates an HL Tau-like disk (2.4'' outer diameter, 40 degree inclination,
limb-darkening exponent 1.5) plus a point-source star with the ALMA C-4
configuration at 230 GHz, checks the simulated visibilities against the
analytic Hestroffer disk model, images the data with Hogbom CLEAN and with the
Adaptive Scale Pixel deconvolver, and verifies that both primary-beam-corrected
restored images reproduce the model convolved with the clean beam (integrated
flux, peak and pixel-by-pixel shape).
"""

import numpy as np
import pytest
import scipy.fft
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
from astroviper.processing_functions.imaging.restore import (  # noqa: E402
    _elliptical_gaussian_kernel,
)
from astroviper.processing_functions.simulation import (  # noqa: E402
    limb_darkened_disk_image,
    limb_darkened_disk_uv_response,
)
from astroviper.utils.beam_models import airy_disk_model  # noqa: E402
from astroviper.utils.telescope_layout import read_telescope_layout  # noqa: E402

ARC = np.pi / (180 * 3600)
PHASE_CENTER = SkyCoord(ra="04h31m38.4s", dec="+18d13m57.7s", frame="icrs")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])[None, :]
FREQUENCY = 230e9
FLUX_DISK, FLUX_STAR = 0.5, 0.003
MAJOR = 2.4 * ARC
MINOR = MAJOR * np.cos(np.deg2rad(40.0))
POSITION_ANGLE = np.deg2rad(120.0)
LIMB_DARKENING = 1.5
IMAGE_SIZE = np.array([256, 256])
CELL_ARCSEC = 0.05
CELL_SIZE = np.array([-CELL_ARCSEC, CELL_ARCSEC]) * ARC


@pytest.mark.parametrize(
    ("deconvolver", "max_iter"), [("hogbom_many_threads", 5000), ("asp", 3000)]
)
def test_protoplanetary_disk_is_simulated_and_recovered(
    tmp_path, deconvolver, max_iter
):
    ps_store = str(tmp_path / "disk_sim.ps.zarr")
    image_store = str(tmp_path / "disk_sim.img.zarr")
    antenna_xds = read_telescope_layout("alma.cycle8.4")
    n_antenna = antenna_xds.sizes["antenna_name"]

    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=ps_store,
        antenna_xds=antenna_xds,
        time_params={
            "time_start": "2021-10-03T07:15:00.000",
            "time_delta": 900.0,
            "n_samples": 8,
        },
        frequency_params={
            "freq_start": FREQUENCY,
            "freq_delta": 2e9,
            "n_channels": 1,
            "channel_width": 2e9,
            "spectral_window_name": "Band6",
        },
        polarization=["XX", "YY"],
        point_source_flux=np.array([FLUX_STAR, 0, 0, FLUX_STAR])[None, None, None, :],
        point_source_ra_dec=PC[None],
        disk_source_flux=np.array([FLUX_DISK, 0, 0, FLUX_DISK])[None, None, None, :],
        disk_source_ra_dec=PC[None],
        disk_source_shape=np.array([[MAJOR, MINOR, POSITION_ANGLE]]),
        disk_source_limb_darkening=[LIMB_DARKENING],
        phase_center_ra_dec=PC,
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(n_antenna, int),
        n_time_chunks=2,
        n_frequency_chunks=1,
        overwrite=True,
    )

    # Visibilities: the analytic inclined-disk response plus the star (both at
    # the phase centre, so the source frame is the phase-centre frame) and
    # resolved beyond the first null.
    ps_xdt = open_processing_set(ps_store)
    ms_xds = ps_xdt[result["ms_name"]].ds
    uvw = ms_xds.UVW.values
    u = uvw[:, :, 0, None] * ms_xds.frequency.values / 299792458.0
    v = uvw[:, :, 1, None] * ms_xds.frequency.values / 299792458.0
    w = uvw[:, :, 2, None] * ms_xds.frequency.values / 299792458.0
    expected = (
        FLUX_DISK
        * limb_darkened_disk_uv_response(
            u, v, MAJOR, MINOR, POSITION_ANGLE, LIMB_DARKENING, w=w
        )
        + FLUX_STAR
    )
    # (the response includes the w term of the extended disk; at these
    # baselines and frequencies it changes the visibilities by ~1e-5 only)
    for polarization in ["XX", "YY"]:
        visibility = ms_xds.VISIBILITY.sel(polarization=polarization).values
        np.testing.assert_allclose(visibility, expected, atol=1e-9)
    assert (
        expected.real.min() < 0.0
    )  # the negative lobe beyond the first null is sampled

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
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "max_iter": max_iter,
            "max_cycles": -1,
            "threshold": 1e-4,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "primary_beam_limit": 0.2,
        },
        gridder="prolate_spheroidal",
        deconvolver=deconvolver,
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
        n_mapping_parallelism={"frequency": 1},
        overwrite=True,
        restore=True,
        primary_beam_correction=True,
    )

    img = load_image(image_store)
    restored = img.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values[0, 0, 0]
    beam = img.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0, 0, 0]
    assert beam[0] < 0.6 * ARC  # the disk (2.4'') is well resolved

    # Expected restored image: the input sky convolved with the fitted clean beam.
    l_grid, m_grid = np.meshgrid(img.l.values, img.m.values, indexing="ij")
    pixel_area = float(np.abs(CELL_SIZE[0] * CELL_SIZE[1]))
    model = (
        FLUX_DISK
        * limb_darkened_disk_image(
            l_grid, m_grid, MAJOR, MINOR, POSITION_ANGLE, LIMB_DARKENING
        )
        * pixel_area
    )
    model[tuple(IMAGE_SIZE // 2)] += FLUX_STAR
    kernel = _elliptical_gaussian_kernel(
        *IMAGE_SIZE, beam[0] / (CELL_ARCSEC * ARC), beam[1] / (CELL_ARCSEC * ARC), beam[2], np.float64
    )  # fmt: skip
    expected_restored = scipy.fft.irfft2(
        scipy.fft.rfft2(model) * scipy.fft.rfft2(scipy.fft.ifftshift(kernel)),
        s=tuple(IMAGE_SIZE),
    )
    beam_area = np.pi / (4 * np.log(2)) * beam[0] * beam[1]
    integrated = np.nansum(restored) * pixel_area / beam_area
    np.testing.assert_allclose(integrated, FLUX_DISK + FLUX_STAR, rtol=0.02)
    np.testing.assert_allclose(np.nanmax(restored), expected_restored.max(), rtol=0.02)
    assert (
        np.nanmax(np.abs(restored - expected_restored)) < 0.02 * expected_restored.max()
    )
    # the inclination is imaged: the disk is longer along the major axis
    e_major = np.array([np.sin(POSITION_ANGLE), np.cos(POSITION_ANGLE)])
    e_minor = np.array([np.cos(POSITION_ANGLE), -np.sin(POSITION_ANGLE)])
    half_light = 0.5 * np.nanmax(restored)
    centre = IMAGE_SIZE // 2

    def extent_along(direction):
        # pixel offsets from the centre along a sky direction (l axis has negative cell)
        steps = np.arange(0, 60)
        pixels = centre[:, None] + np.outer(direction / np.sign(CELL_SIZE), steps)
        values = restored[
            np.round(pixels[0]).astype(int), np.round(pixels[1]).astype(int)
        ]
        return steps[values > half_light].max()

    assert extent_along(e_major) > 1.15 * extent_along(e_minor)
