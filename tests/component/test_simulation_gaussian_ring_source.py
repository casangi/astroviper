"""Component test: a protoplanetary disk of nested Gaussian rings observed by ALMA.

Simulates three nested Gaussian-broadened rings (the "Gaussian disk" of the
simulation memo, version 2) plus a point-source star with the ALMA C-4
configuration at 230 GHz, checks the simulated visibilities against the
analytic ring responses including the w term, images the data and verifies
that the restored image reproduces the rings convolved with the clean beam
(integrated flux and pixel-by-pixel shape).
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
    gaussian_ring_image,
    gaussian_ring_uv_response,
)
from astroviper.utils.beam_models import airy_disk_model  # noqa: E402
from astroviper.utils.telescope_layout import read_telescope_layout  # noqa: E402

ARC = np.pi / (180 * 3600)
PHASE_CENTER = SkyCoord(ra="04h31m38.4s", dec="+18d13m57.7s", frame="icrs")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])[None, :]
FREQUENCY = 230e9
RING_RADIUS = np.array([0.0, 0.5, 1.0]) * ARC
RING_FWHM = np.array([0.35, 0.2, 0.25]) * ARC
RING_FLUX = np.array([0.1, 0.15, 0.2])  # Jy
FLUX_STAR = 0.01
INCLINATION = np.deg2rad(40.0)
POSITION_ANGLE = np.deg2rad(60.0)
IMAGE_SIZE = np.array([256, 256])
CELL_ARCSEC = 0.05
CELL_SIZE = np.array([-CELL_ARCSEC, CELL_ARCSEC]) * ARC


def test_gaussian_disk_is_simulated_and_recovered(tmp_path):
    ps_store = str(tmp_path / "ring_sim.ps.zarr")
    image_store = str(tmp_path / "ring_sim.img.zarr")
    antenna_xds = read_telescope_layout("alma.cycle8.4")
    n_antenna = antenna_xds.sizes["antenna_name"]
    shape = np.stack(
        [RING_RADIUS, RING_FWHM, np.full(3, INCLINATION), np.full(3, POSITION_ANGLE)],
        axis=1,
    )

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
        gaussian_ring_source_flux=np.stack([np.array([f, 0, 0, f]) for f in RING_FLUX])[
            :, None, None, :
        ],
        gaussian_ring_source_ra_dec=np.repeat(PC[None], 3, axis=1),
        gaussian_ring_source_shape=shape,
        phase_center_ra_dec=PC,
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(n_antenna, int),
        n_time_chunks=2,
        n_frequency_chunks=1,
        overwrite=True,
    )

    # Visibilities: the analytic ring responses (with the w term) plus the star.
    ps_xdt = open_processing_set(ps_store)
    ms_xds = ps_xdt[result["ms_name"]].ds
    uvw = ms_xds.UVW.values
    u, v, w = (
        uvw[:, :, i, None] * ms_xds.frequency.values / 299792458.0 for i in range(3)
    )
    expected = np.full(u.shape, FLUX_STAR, dtype=complex)
    for flux, ring in zip(RING_FLUX, shape, strict=True):
        expected += flux * gaussian_ring_uv_response(u, v, *ring, w=w)
    for polarization in ["XX", "YY"]:
        visibility = ms_xds.VISIBILITY.sel(polarization=polarization).values
        np.testing.assert_allclose(visibility, expected, atol=1e-9)
    assert (
        np.abs(expected).min() < 0.3 * expected[0, 0, 0].real
    )  # the rings are resolved

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
            "niter": 5000,
            "nmajor": -1,
            "threshold": 1e-4,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
            "primary_beam_limit": 0.2,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom_many_threads",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
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
        primary_beam_correction=False,
    )

    img = load_image(image_store)
    restored = img.SKY_RESTORED.values[0, 0, 0]
    beam = img.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0, 0, 0]
    assert beam[0] < 0.6 * ARC  # the 2'' disk is resolved

    # Expected restored image: the input rings and star convolved with the fitted clean beam.
    l_grid, m_grid = np.meshgrid(img.l.values, img.m.values, indexing="ij")
    pixel_area = float(np.abs(CELL_SIZE[0] * CELL_SIZE[1]))
    sky = np.zeros(tuple(IMAGE_SIZE))
    for flux, ring in zip(RING_FLUX, shape, strict=True):
        sky += flux * gaussian_ring_image(l_grid, m_grid, *ring) * pixel_area
    sky[tuple(IMAGE_SIZE // 2)] += FLUX_STAR
    kernel = _elliptical_gaussian_kernel(
        *IMAGE_SIZE, beam[0] / (CELL_ARCSEC * ARC), beam[1] / (CELL_ARCSEC * ARC), beam[2], np.float64
    )  # fmt: skip
    expected_restored = scipy.fft.irfft2(
        scipy.fft.rfft2(sky) * scipy.fft.rfft2(scipy.fft.ifftshift(kernel)),
        s=tuple(IMAGE_SIZE),
    )
    beam_area = np.pi / (4 * np.log(2)) * beam[0] * beam[1]
    integrated = np.nansum(restored) * pixel_area / beam_area
    np.testing.assert_allclose(integrated, RING_FLUX.sum() + FLUX_STAR, rtol=0.02)
    assert (
        np.nanmax(np.abs(restored - expected_restored)) < 0.02 * expected_restored.max()
    )
