"""Known-truth continuum checks adapted from the external validation harness.

The harness's ``phase_center`` case is a point source with spectral index -1;
its ``off_axis`` case is a one-Jy, flat-spectrum point source away from the
pointing centre.  These compact generated equivalents exercise the same
scientific oracles without adding a stored Measurement Set or a CASA runtime
to the repository test suite.
"""

from __future__ import annotations

import dask
import numpy as np
import pytest

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180.0 * 3600.0)
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
FREQUENCIES = np.linspace(80.0e9, 120.0e9, 5)
REFERENCE_FREQUENCY = 100.0e9
IMAGE_SIZE = [64, 64]
CELL_SIZE = np.array([-2.0, 2.0]) * ARCSEC
SOURCE_OFFSET_ARCSEC = 36.0


@pytest.fixture(scope="module")
def known_truth_processing_sets(tmp_path_factory):
    """Generate the harness phase-centre and off-axis point-source cases."""
    work_dir = tmp_path_factory.mktemp("continuum_known_truth")
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(6))
    )
    cases = {
        "phase_center": {
            "position": PHASE_CENTER,
            "spectral_index": -1.0,
        },
        "off_axis": {
            "position": PHASE_CENTER
            + np.array(
                [
                    SOURCE_OFFSET_ARCSEC * ARCSEC / np.cos(PHASE_CENTER[1]),
                    0.0,
                ]
            ),
            "spectral_index": 0.0,
        },
    }
    for name, case in cases.items():
        spectrum = (FREQUENCIES / REFERENCE_FREQUENCY) ** case["spectral_index"]
        correlation_flux = np.stack(
            [spectrum, np.zeros_like(spectrum), np.zeros_like(spectrum), spectrum],
            axis=1,
        )
        case["store"] = str(work_dir / f"{name}.ps.zarr")
        with dask.config.set(scheduler="synchronous"):
            distributed_applications.simulation.simulate_processing_set(
                ps_store=case["store"],
                antenna_xds=antenna_xds,
                time_params={
                    "time_start": "2019-10-03T19:00:00.000",
                    "time_delta": 600.0,
                    "n_samples": 4,
                },
                frequency_params={
                    "freq_start": FREQUENCIES[0],
                    "freq_delta": FREQUENCIES[1] - FREQUENCIES[0],
                    "n_channels": len(FREQUENCIES),
                    "channel_width": 2.0e6,
                },
                polarization=["XX", "YY"],
                sky_components=[
                    {
                        "kind": "point",
                        "flux": correlation_flux,
                        "ra_dec": case["position"],
                    }
                ],
                phase_center_ra_dec=PHASE_CENTER[None, :],
                beam_models=[airy_disk_model("alma")],
                beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
                n_time_chunks=1,
                n_frequency_chunks=1,
                overwrite=True,
            )
    return cases


@pytest.fixture(scope="module")
def sparse_wideband_off_axis_processing_set(tmp_path_factory):
    """Five-channel, sparse-uv off-axis case for the MVC runaway safeguard."""
    work_dir = tmp_path_factory.mktemp("continuum_mvc_runaway")
    store = str(work_dir / "sparse_wideband_off_axis.ps.zarr")
    frequencies = np.linspace(70.0e9, 130.0e9, 5)
    antenna_xds = read_telescope_layout(
        "alma.cycle8.1", antenna_selection=list(range(12))
    )
    source_offset_arcsec = 50.0
    source = PHASE_CENTER + np.array(
        [source_offset_arcsec * ARCSEC / np.cos(PHASE_CENTER[1]), 0.0]
    )
    correlation_flux = np.ones((len(frequencies), 4))
    correlation_flux[:, 1:3] = 0.0
    with dask.config.set(scheduler="synchronous"):
        distributed_applications.simulation.simulate_processing_set(
            ps_store=store,
            antenna_xds=antenna_xds,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 600.0,
                "n_samples": 4,
            },
            frequency_params={
                "freq_start": frequencies[0],
                "freq_delta": frequencies[1] - frequencies[0],
                "n_channels": len(frequencies),
                "channel_width": 2.0e6,
            },
            polarization=["XX", "YY"],
            sky_components=[
                {"kind": "point", "flux": correlation_flux, "ra_dec": source}
            ],
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[airy_disk_model("alma")],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    return {"store": store, "frequencies": frequencies}


def _dirty_continuum_image(
    tmp_path,
    case,
    continuum_mode,
    mvc_model_update_approximation=None,
):
    """Run the harness case through the public two-term continuum application."""
    with dask.config.set(scheduler="synchronous"):
        result = image_continuum_single_field(
            ps_store=case["store"],
            image_store=str(tmp_path / f"{continuum_mode}.img.zarr"),
            image_params={
                "image_size": IMAGE_SIZE,
                "cell_size": CELL_SIZE,
                "phase_direction": PHASE_CENTER,
                "frequency_coords": FREQUENCIES,
                "polarization_coords": ["I", "Q"],
                "time_coords": [0],
                "fft_padding": 1.2,
                "n_taylor_terms": 2,
                "reference_frequency": REFERENCE_FREQUENCY,
            },
            imaging_weights_params={
                "weighting": "natural",
                "weighting_scope": "local",
            },
            cycle_control_params={
                "max_iter": 0,
                "max_cycles": 0,
                "threshold": 0.0,
                "gain": 0.1,
                "psf_sidelobe_factor": 1.0,
                "max_iter_per_cycle": 1,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
            },
            continuum_mode=continuum_mode,
            **(
                {}
                if mvc_model_update_approximation is None
                else {
                    "mvc_model_update_approximation": mvc_model_update_approximation,
                }
            ),
            primary_beam_limit=0.1,
            processing_set_data_group_name="base",
            image_data_variables_keep=[
                "sky_residual",
                "point_spread_function",
                "primary_beam",
            ],
            n_mapping_parallelism={"frequency": 1},
            overwrite=True,
        )
    return result["image"]


def _source_taylor_coefficients(image):
    """Solve the two-term dirty normal equations at their source peak."""
    residual = image.SKY_RESIDUAL.isel(time=0, polarization=0)
    psf = image.POINT_SPREAD_FUNCTION.isel(time=0, polarization=0)
    source_pixel = np.unravel_index(
        np.nanargmax(np.abs(residual.isel(taylor_term=0).values)),
        tuple(residual.sizes[axis] for axis in ("l", "m")),
    )
    psf_peak = np.unravel_index(
        np.nanargmax(np.abs(psf.isel(psf_taylor_order=0).values)),
        tuple(psf.sizes[axis] for axis in ("l", "m")),
    )
    hessian = np.array(
        [
            [psf.values[0, *psf_peak], psf.values[1, *psf_peak]],
            [psf.values[1, *psf_peak], psf.values[2, *psf_peak]],
        ]
    )
    dirty_terms = residual.values[:, *source_pixel]
    return np.linalg.solve(hessian, dirty_terms), source_pixel


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_phase_center_spectral_index_matches_known_truth(
    tmp_path, known_truth_processing_sets, continuum_mode
):
    """MFS and MVC recover the harness source's flux and spectral index."""
    case = known_truth_processing_sets["phase_center"]
    image = _dirty_continuum_image(tmp_path, case, continuum_mode)
    coefficients, source_pixel = _source_taylor_coefficients(image)
    spectral_index = coefficients[1] / coefficients[0]
    primary_beam = image.PRIMARY_BEAM.isel(time=0, frequency=0, polarization=0).values[
        source_pixel
    ]

    assert coefficients[0] / primary_beam == pytest.approx(1.0, rel=0.03)
    assert spectral_index == pytest.approx(case["spectral_index"], abs=0.06)


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_off_axis_primary_beam_corrected_flux_matches_known_truth(
    tmp_path, known_truth_processing_sets, continuum_mode
):
    """The harness's single-field off-axis source recovers one Jy after PB correction."""
    image = _dirty_continuum_image(
        tmp_path, known_truth_processing_sets["off_axis"], continuum_mode
    )
    coefficients, source_pixel = _source_taylor_coefficients(image)
    primary_beam = image.PRIMARY_BEAM.isel(time=0, frequency=0, polarization=0).values[
        source_pixel
    ]

    assert 0.1 < primary_beam < 0.95
    assert coefficients[0] / primary_beam == pytest.approx(1.0, rel=0.05)


def test_off_axis_mvc_default_local_beam_moments_recovers_known_spectral_index(
    tmp_path, known_truth_processing_sets
):
    """The default MVC local-beam operator recovers the off-axis flat spectrum."""
    case = known_truth_processing_sets["off_axis"]
    image = _dirty_continuum_image(tmp_path, case, "mvc")
    coefficients, source_pixel = _source_taylor_coefficients(image)
    primary_beam = image.PRIMARY_BEAM.isel(time=0, frequency=0, polarization=0).values[
        source_pixel
    ]

    assert 0.1 < primary_beam < 0.95
    assert coefficients[0] / primary_beam == pytest.approx(1.0, rel=0.05)
    assert coefficients[1] / coefficients[0] == pytest.approx(
        case["spectral_index"], abs=0.03
    )


def test_channel_psf_runaway_stops_on_refreshed_taylor_zero_residual(
    tmp_path, sparse_wideband_off_axis_processing_set
):
    """The public MVC application stops a sparse, wideband off-axis runaway.

    This exercises the review case's controlling dimensions: five channels
    spanning +/-30 percent, twelve antennas with only four time samples, three
    Taylor terms, and a source near the primary-beam edge. Stop code 5 must
    come from consecutive exact Taylor-zero residuals, before the model can
    grow without bound.
    """
    from astroviper.processing_functions.imaging.utils.cycle_control import (
        IMAGING_DIVERGENCE,
    )

    case = sparse_wideband_off_axis_processing_set
    with dask.config.set(scheduler="synchronous"):
        result = image_continuum_single_field(
            ps_store=case["store"],
            image_store=str(tmp_path / "channel_psf_runaway.img.zarr"),
            image_params={
                "image_size": IMAGE_SIZE,
                "cell_size": CELL_SIZE,
                "phase_direction": PHASE_CENTER,
                "frequency_coords": case["frequencies"],
                "polarization_coords": ["I", "Q"],
                "time_coords": [0],
                "fft_padding": 1.2,
                "n_taylor_terms": 3,
                "reference_frequency": REFERENCE_FREQUENCY,
            },
            imaging_weights_params={"weighting": "natural", "weighting_scope": "local"},
            cycle_control_params={
                "max_iter": 3000,
                "max_cycles": 12,
                "threshold": 0.0,
                "gain": 1.0,
                "psf_sidelobe_factor": 1.0,
                "max_iter_per_cycle": 100,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
            },
            continuum_mode="mvc",
            deconvolver="mtmfs",
            mvc_model_update_approximation="channel_psf",
            primary_beam_limit=0.01,
            processing_set_data_group_name="base",
            image_data_variables_keep=["sky_residual", "point_spread_function"],
            n_mapping_parallelism={"frequency": 1},
            overwrite=True,
        )

    assert result["stopcode"].imaging == IMAGING_DIVERGENCE
    assert result["controller"].refreshed_residual_divergence_stopped.any()
