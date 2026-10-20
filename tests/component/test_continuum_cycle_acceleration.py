"""Imaging-cycle accelerations of the continuum imager on simulated data.

Ten VLA D-configuration antennas, six times and four channels over 20 % of
the band observe three point sources with spectral indices and one Gaussian.
Every ``"none"`` run is the baseline of both accelerations. Comparisons are
asserted only where the measured margin is large; the numbers in the test
docstrings were measured on this data. The whole file took 76 s with a peak
RSS of 0.38 GiB.
"""

import importlib

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
import astroviper.node_tasks.imaging.image_continuum_single_field as continuum_node
from astroviper.distributed_applications.imaging import image_continuum_single_field
from astroviper.processing_functions.imaging.utils.iteration_control import (
    IMAGING_MAX_CYCLES,
    IMAGING_THRESHOLD,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import sin_pixel_to_celestial_coord
from astroviper.utils.telescope_layout import read_telescope_layout

ARC = np.pi / (180 * 3600)
IMAGE_SIZE = np.array([64, 64])
CELL_SIZE = np.array([-6.0, 6.0]) * ARC
PHASE_CENTER = np.array([[np.deg2rad(299.87), np.deg2rad(40.73)]])
FREQUENCIES = 2.7e9 + 0.2e9 * np.arange(4)
REFERENCE_FREQUENCY = FREQUENCIES.mean()
ACCELERATIONS = ("conjugate_gradient", "nesterov_momentum")
DECONVOLVERS = [("hogbom", 1, {}), ("mtmfs", 2, {"scales": [0]})]


@pytest.fixture(scope="module")
def simulated_store(tmp_path_factory):
    n_antennas = 10
    antennas = read_telescope_layout("vla.d").isel(antenna_name=np.arange(n_antennas))
    spectrum = FREQUENCIES / REFERENCE_FREQUENCY
    point_flux = (
        np.array([1.0, 0.6, 0.4])[:, None]
        * spectrum ** np.array([-0.7, 0.0, -1.5])[:, None]
    )
    gaussian_flux = 0.8 * spectrum[None] ** -0.5

    def correlations(flux):
        return np.stack([flux, 0 * flux, 0 * flux, flux], axis=-1)[:, None]

    def directions(pixels):
        return sin_pixel_to_celestial_coord(
            PHASE_CENTER[0], IMAGE_SIZE, CELL_SIZE, np.asarray(pixels)
        )[None]

    store = str(tmp_path_factory.mktemp("acceleration") / "simulated.ps.zarr")
    distributed_applications.simulation.simulate_processing_set(
        ps_store=store,
        antenna_xds=antennas,
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1800.0,
            "n_samples": 6,
        },
        frequency_params={
            "freq_start": FREQUENCIES[0],
            "freq_delta": 0.2e9,
            "n_channels": 4,
            "channel_width": 1e7,
        },
        polarization=["XX", "YY"],
        point_source_flux=correlations(point_flux),
        point_source_ra_dec=directions([[32, 32], [24, 40], [42, 26]]),
        gaussian_source_flux=correlations(gaussian_flux),
        gaussian_source_ra_dec=directions([[38, 38]]),
        gaussian_source_shape=np.array([[60.0 * ARC, 40.0 * ARC, 0.3]]),
        phase_center_ra_dec=PHASE_CENTER,
        beam_models=[airy_disk_model("vla")],
        beam_model_map=np.zeros(n_antennas, int),
        n_time_chunks=1,
        n_frequency_chunks=2,
        overwrite=True,
    )
    return store


@pytest.fixture(scope="module")
def run(simulated_store, tmp_path_factory):
    """Image once per configuration and acceleration; return cycles, residuals
    and the acceleration state and momentum coefficient each model update
    received."""
    results = {}
    output_directory = tmp_path_factory.mktemp("acceleration_images")
    accelerate = continuum_node._accelerate_continuum_model_increment

    def run(
        deconvolver,
        nterms,
        specmode,
        controls,
        cycle_acceleration,
        momentum_coefficient=0.5,
    ):
        key = (
            deconvolver,
            specmode,
            repr(sorted(controls.items())),
            cycle_acceleration,
            momentum_coefficient,
        )
        if key in results:
            return results[key]
        processing_set = open_processing_set(simulated_store)
        fields = processing_set.xr_ps.get_combined_field_and_source_xds()
        output = str(output_directory / f"image_{len(results)}.img.zarr")
        received_states = []
        received_coefficients = []

        def recording(*args):
            state_xds = args[-1].get("cycle_acceleration_state_xds")
            received_states.append(
                None if state_xds is None else state_xds.attrs["cycle_acceleration"]
            )
            received_coefficients.append(args[-1]["momentum_coefficient"])
            return accelerate(*args)

        with (
            pytest.MonkeyPatch.context() as patch,
            dask.config.set(scheduler="synchronous"),
        ):
            # The synchronous scheduler runs the append node in this process.
            patch.setattr(
                continuum_node, "_accelerate_continuum_model_increment", recording
            )
            result = image_continuum_single_field(
                ps_store=simulated_store,
                image_store=output,
                image_params={
                    "image_size": list(IMAGE_SIZE),
                    "cell_size": CELL_SIZE,
                    "phase_direction": fields.FIELD_PHASE_CENTER_DIRECTION.sel(
                        field_name=fields.attrs["center_field_name"]
                    ).values,
                    "frequency_coords": processing_set.xr_ps.get_freq_axis().values,
                    "polarization_coords": ["I", "Q"],
                    "time_coords": [0],
                    "fft_padding": 1.2,
                    "cpp_gridder": True,
                    "nterms": nterms,
                    "reference_frequency": REFERENCE_FREQUENCY,
                    "reference_frequency_hz": REFERENCE_FREQUENCY,
                },
                imaging_weights_params={"weighting": "natural"},
                iteration_control_params={
                    "max_iter": 1000,
                    "max_iter_per_cycle": 10,
                    "gain": 0.1,
                    "psf_sidelobe_factor": 1.5,
                    "min_psf_fraction": 0.05,
                    "max_psf_fraction": 0.8,
                    **controls,
                },
                deconvolver=deconvolver,
                specmode=specmode,
                image_data_variables_keep=["sky_residual", "sky_model", "primary_beam"],
                processing_set_data_group_name="base",
                single_precision_image=False,
                processing_function_threads=1,
                n_chunks=2,
                overwrite=True,
                cycle_acceleration=cycle_acceleration,
                momentum_coefficient=momentum_coefficient,
            )
        with xr.open_zarr(output) as stored:
            image = stored[["SKY_RESIDUAL", "SKY_MODEL", "PRIMARY_BEAM"]].load()
        primary_beam = image.PRIMARY_BEAM.isel(time=0, frequency=0, polarization=0)
        residual = image.SKY_RESIDUAL.isel(time=0, taylor_term=0, polarization=0)
        results[key] = {
            "cycles": result["n_major_cycles"],
            "stop_codes": result["controller"].stop_code_imaging.ravel(),
            "rms": float(
                np.sqrt(np.mean(residual.values[primary_beam.values >= 0.2] ** 2))
            ),
            "finite": bool(
                np.isfinite(image.SKY_RESIDUAL).all()
                and np.isfinite(image.SKY_MODEL).all()
            ),
            "received_states": received_states,
            "received_coefficients": received_coefficients,
        }
        return results[key]

    return run


def _assert_state_forwarded(result, cycle_acceleration):
    """The first model update starts without a state; every later one
    receives the state of its predecessor."""
    states = result["received_states"]
    assert len(states) >= 2
    assert states == [None] + (len(states) - 1) * [cycle_acceleration]


@pytest.mark.parametrize("deconvolver,nterms,options", DECONVOLVERS)
def test_shallow_threshold_runs_converge_and_conjugate_gradient_saves_cycles(
    run, deconvolver, nterms, options
):
    """Measured: none 11, conjugate gradient 8 and momentum 9 imaging cycles
    with both deconvolvers; neither the momentum cycles nor the final
    residual RMS are asserted."""
    controls = {"max_cycles": 12, "threshold": 0.02, **options}
    plain = run(deconvolver, nterms, "mfs", controls, "none")
    cycles = {}
    for cycle_acceleration in ACCELERATIONS:
        accelerated = run(deconvolver, nterms, "mfs", controls, cycle_acceleration)
        assert accelerated["finite"]
        np.testing.assert_array_equal(accelerated["stop_codes"], IMAGING_THRESHOLD)
        _assert_state_forwarded(accelerated, cycle_acceleration)
        cycles[cycle_acceleration] = accelerated["cycles"]
    assert plain["received_states"] == []
    assert cycles["conjugate_gradient"] < plain["cycles"]


@pytest.mark.parametrize("deconvolver,nterms,options", DECONVOLVERS)
def test_fixed_cycle_budget_leaves_a_smaller_residual(
    run, deconvolver, nterms, options
):
    """Four imaging cycles; measured residual RMS ratios (accelerated / none)
    0.32 and 0.30 (conjugate gradient) and 0.42 and 0.41 (momentum) for
    Högbom and MT-MFS."""
    controls = {"max_cycles": 4, "threshold": 0.0, **options}
    plain = run(deconvolver, nterms, "mfs", controls, "none")
    for cycle_acceleration in ACCELERATIONS:
        accelerated = run(deconvolver, nterms, "mfs", controls, cycle_acceleration)
        assert accelerated["finite"]
        np.testing.assert_array_equal(accelerated["stop_codes"], IMAGING_MAX_CYCLES)
        _assert_state_forwarded(accelerated, cycle_acceleration)
        assert accelerated["rms"] <= 0.5 * plain["rms"]


@pytest.mark.parametrize("cycle_acceleration", ("none",) + ACCELERATIONS)
def test_adaptive_depth_hogbom_runs_reach_the_threshold(run, cycle_acceleration):
    """``max_iter_per_cycle=-1``; measured: none 7, conjugate gradient 8 and
    momentum 7 imaging cycles. The momentum's residual norm grows once, after
    its third step, which restarts it. With ``psf_sidelobe_factor=1.0``
    instead of this file's 1.5, the conjugate gradient ran out of
    ``max_iter`` at a residual peak of 13 Jy (measured, not asserted)."""
    result = run(
        "hogbom",
        1,
        "mfs",
        {
            "max_iter": 10000,
            "max_cycles": 12,
            "max_iter_per_cycle": -1,
            "threshold": 0.005,
        },
        cycle_acceleration,
    )
    assert result["finite"]
    np.testing.assert_array_equal(result["stop_codes"], IMAGING_THRESHOLD)


@pytest.mark.parametrize("cycle_acceleration", ACCELERATIONS)
def test_mvc_runs_reach_the_threshold(run, cycle_acceleration):
    result = run(
        "mtmfs",
        2,
        "mvc",
        {"max_cycles": 12, "threshold": 0.02, "scales": [0]},
        cycle_acceleration,
    )
    assert result["finite"]
    np.testing.assert_array_equal(result["stop_codes"], IMAGING_THRESHOLD)
    _assert_state_forwarded(result, cycle_acceleration)


def test_momentum_coefficient_reaches_every_model_update(run):
    result = run(
        "hogbom",
        1,
        "mfs",
        {"max_cycles": 4, "threshold": 0.0},
        "nesterov_momentum",
        momentum_coefficient=0.3,
    )
    assert result["finite"]
    _assert_state_forwarded(result, "nesterov_momentum")
    assert result["received_coefficients"] == len(result["received_states"]) * [0.3]


def test_only_momentum_runs_check_for_switched_off_planes(run, monkeypatch):
    """The driver calls the switch-off warning in every imaging cycle with
    momentum, and never with ``"none"`` or the conjugate gradient."""
    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.image_continuum_single_field"
    )
    original = driver._warn_on_disabled_momentum
    calls = []

    def counting(previous_state_xds, state_xds):
        calls.append(state_xds)
        return original(previous_state_xds, state_xds)

    monkeypatch.setattr(driver, "_warn_on_disabled_momentum", counting)
    for cycle_acceleration in ("none",) + ACCELERATIONS:
        calls.clear()
        result = run(
            "hogbom", 1, "mfs", {"max_cycles": 2, "threshold": 0.0}, cycle_acceleration
        )
        if cycle_acceleration == "nesterov_momentum":
            assert len(calls) >= result["cycles"] > 0
        else:
            assert calls == []


def test_driver_warns_once_when_momentum_switches_off_on_a_plane(run, monkeypatch):
    """Momentum is forced off on Stokes I from the second model update on; the
    driver warns once, for that plane only."""
    import toolviper.utils.logger as logger

    import astroviper.processing_functions.imaging.cycle_acceleration as acceleration

    step_function = acceleration.nesterov_momentum_model_step
    calls = []

    def switching_off(*args, **kwargs):
        step, state_xds = step_function(*args, **kwargs)
        calls.append(len(calls))
        if len(calls) >= 2:
            state_xds = state_xds.assign(
                MOMENTUM_DISABLED=state_xds.MOMENTUM_DISABLED
                | (state_xds.polarization == "I")
            )
        return step, state_xds

    messages = []
    monkeypatch.setattr(acceleration, "nesterov_momentum_model_step", switching_off)
    monkeypatch.setattr(
        logger, "warning", lambda message, *args, **kwargs: messages.append(message)
    )
    result = run(
        "hogbom", 1, "mfs", {"max_cycles": 3, "threshold": 0.0}, "nesterov_momentum"
    )
    warnings = [m for m in map(str, messages) if m.startswith("Nesterov momentum")]
    assert result["finite"]
    assert len(calls) == 3
    assert len(warnings) == 1
    assert "'I'" in warnings[0]
    assert "'Q'" not in warnings[0]
