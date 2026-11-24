"""The fringe search of astroviper.calibration.fringefit in each FFT library, on
simulated data (no download).

A noiseless 1 Jy point source at the phase centre (6 ALMA antennas, 24 times of 10 s,
32 channels of 2 MHz, RR RL LR LL) whose baselines to the reference antenna get a
delay and rate ramp on the search grid (3 and -2 padded bins) and whose visibilities
all get a phase noise of 1 rad (fixed seed), so that the injected bins are the peak of
every padded spectrum of those baselines by a margin far above rounding.
"""

import dask
import numpy as np
import pytest

from astroviper.calibration import fringefit

NPAD = 8  # the padding of _fringe_node_task
DELAY_BINS, RATE_BINS = 3, -2


@pytest.fixture(scope="module")
def search(tmp_path_factory):
    from xradio.measurement_set import open_processing_set

    from astroviper.distributed_applications.simulation import (
        simulate_processing_set,
    )
    from astroviper.utils.beam_models import airy_disk_model
    from astroviper.utils.telescope_layout import read_telescope_layout

    store = str(tmp_path_factory.mktemp("fringe") / "fringe.ps.zarr")
    phase_center = np.array([4.267, np.deg2rad(-23.0)])
    antennas = read_telescope_layout("alma.cycle8.1", antenna_selection=list(range(6)))
    with dask.config.set(scheduler="synchronous"):
        simulate_processing_set(
            ps_store=store,
            antenna_xds=antennas,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 10.0,
                "n_samples": 24,
            },
            frequency_params={
                "freq_start": 100e9,
                "freq_delta": 2e6,
                "n_channels": 32,
                "channel_width": 2e6,
            },
            polarization=["RR", "RL", "LR", "LL"],
            sky_components=[
                {
                    "kind": "point",
                    "flux": np.array([1.0, 0, 0, 0]),
                    "ra_dec": phase_center,
                }
            ],
            phase_center_ra_dec=phase_center[None, :],
            beam_models=[airy_disk_model("alma")],
            beam_model_map=np.zeros(antennas.sizes["antenna_name"], int),
            noise_params=None,
            n_time_chunks=1,
            n_frequency_chunks=1,
            processing_function_threads=1,
            overwrite=True,
        )
    ps = open_processing_set(store)
    name = next(iter(ps))
    xds = ps[name]
    reference = str(xds.baseline_antenna1_name.values[0])
    ddelay, drate = fringefit.getFourierSpacings(xds, NPAD)
    time = xds.time.values
    frequency = xds.frequency.values
    t = (time - time[0])[:, None, None, None]
    f = (frequency - frequency[0])[None, None, :, None]
    to_reference = (
        (xds.baseline_antenna1_name.values == reference)
        | (xds.baseline_antenna2_name.values == reference)
    )[None, :, None, None]
    ramp = (
        np.exp(2j * np.pi * (DELAY_BINS * ddelay * f + RATE_BINS * drate * t))
        * to_reference
        + ~to_reference
    )
    noise = np.exp(
        1j * np.random.default_rng(19).normal(0.0, 1.0, xds.VISIBILITY.shape)
    )
    ps[name]["VISIBILITY"] = xds.VISIBILITY * ramp * noise
    return {
        "ps": ps,
        "name": name,
        "reference": reference,
        "ddelay": ddelay,
        "drate": drate,
        "reference_frequency": float(xds.frequency.reference_frequency["data"]),
        "start": float(time[0]),
        "interval": float(time[-1] - time[0]) + 1.0,
        "to_reference": np.flatnonzero(to_reference[0, :, 0, 0]),
    }


def _solve(search, fft_backend):
    tree = fringefit.fringefit_ps(
        search["ps"],
        search["reference"],
        search["start"],
        search["interval"],
        fft_backend=fft_backend,
    )
    part = next(iter(tree.children.values())).dataset
    dims = ("antenna_name", "polarization", "cal_parameter")
    return (
        part.CALIBRATION_PARAMETER.isel(time=0, frequency=0).transpose(*dims),
        part.SNR.isel(time=0, frequency=0).transpose(*dims).values,
    )


def test_the_injected_bins_dominate_the_padded_spectra(search):
    """On every baseline to the reference, in the two polarizations the search
    takes, the top bin of the padded spectrum exceeds the next by more than 1e-6
    of it, so that no library's rounding can move the peak."""
    xds = search["ps"][search["name"]]
    vis = xds.VISIBILITY.values[:, :, :, ::3]
    normed = np.exp(1j * np.angle(vis)) * xds.WEIGHT.values[:, :, :, ::3]
    shape = (NPAD * vis.shape[0], NPAD * vis.shape[2])
    spectra = np.abs(np.fft.fft2(normed, axes=(0, 2), s=shape))
    for baseline in search["to_reference"]:
        for polarization in range(2):
            top, second = np.sort(spectra[:, baseline, :, polarization].ravel())[:-3:-1]
            assert top > (1 + 1e-6) * second


def test_numpy_finds_the_injected_delay_and_rate(search):
    """Polarization R of every antenna whose baseline to the reference has a
    nonzero id (the node skips baseline id 0) gets the injected delay and rate."""
    parameters, _ = _solve(search, "numpy")
    xds = search["ps"][search["name"]]
    found = []
    for baseline in search["to_reference"]:
        if baseline == 0:
            continue
        antenna = str(xds.baseline_antenna2_name.values[baseline])
        if antenna == search["reference"]:
            antenna = str(xds.baseline_antenna1_name.values[baseline])
        found.append(antenna)
        _, delay, rate = parameters.sel(antenna_name=antenna, polarization="R").values
        assert delay == DELAY_BINS * search["ddelay"]
        assert rate == RATE_BINS * search["drate"] / search["reference_frequency"]
    assert len(found) == 4


@pytest.mark.parametrize("fft_backend", ["scipy", "pyfftw"])
def test_each_library_finds_numpys_solutions(search, fft_backend):
    """The same delay and rate as under "numpy", phi0 within 8 eps pi and the SNR
    within 16 eps of numpy's."""
    expected, expected_snr = _solve(search, "numpy")
    got, snr = _solve(search, fft_backend)
    np.testing.assert_array_equal(got.values[..., 1:], expected.values[..., 1:])
    eps = np.finfo(np.float64).eps
    finite = np.isfinite(expected.values[..., 0])
    np.testing.assert_array_equal(np.isfinite(got.values[..., 0]), finite)
    phi0 = np.abs(got.values[..., 0][finite] - expected.values[..., 0][finite])
    assert phi0.max() <= 8 * eps * np.pi
    measured = np.isfinite(expected_snr)
    np.testing.assert_array_equal(np.isfinite(snr), measured)
    relative = np.abs(snr[measured] - expected_snr[measured]) / expected_snr[measured]
    assert relative.max() <= 16 * eps


@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_the_value_reaches_the_transform(search, fft_backend, monkeypatch):
    """fringefit_ps hands its fft_backend to the node's transform, which runs one
    thread."""
    from astroviper.processing_functions import fft

    seen = []
    real = fft.fft2

    def spy(*args, **kwargs):
        seen.append((kwargs["fft_backend"], kwargs["processing_function_threads"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(fft, "fft2", spy)
    _solve(search, fft_backend)
    assert seen == [(fft_backend, 1)]
