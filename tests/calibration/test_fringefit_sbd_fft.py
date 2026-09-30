"""Fringe fitting (single-band delay by FFT) of a global VLBI observation.

A 90 s interval of the data set ``global_vlbi_gg084b_reduced.ms`` is fringe
fitted with Mauna Kea (``MK``) as the reference antenna and the solutions are
applied.  The phased VLA (``YY``) is by far the most sensitive antenna: on the
``MK-YY`` baseline the fit finds a delay of about -65 ns, and removing it turns
visibilities that cancel across the band into a coherent signal.

Everything that touches data happens in fixtures, inside a temporary
directory: importing this module must not download, convert or write
anything (pytest imports every test module when it collects), and the casacore
tables of the Measurement Set must not live in a synchronised folder.
"""

import astropy.time
import numpy as np
import pytest
import toolviper
from xradio.measurement_set import convert_msv2_to_processing_set, open_processing_set

from astroviper.calibration.fringefit import apply_cal_ps, fringefit_ps

MSV2_NAME = "global_vlbi_gg084b_reduced.ms"
PARTITION_SCHEME = ["FIELD_ID", "SCAN_NUMBER", "SPW"]
FRINGE_FIT_START = astropy.time.Time("2018-05-27 07:15:00", format="iso").unix
FRINGE_FIT_INTERVAL = 90  # seconds
REFERENCE_ANTENNA = "MK"
TARGET_ANTENNA = "YY"  # the phased VLA
PARALLEL_HANDS = ["RR", "LL"]


@pytest.fixture(scope="module")
def fringe_fit(tmp_path_factory):
    """Download, convert, fringe fit and apply once; every test reads the result."""
    folder = tmp_path_factory.mktemp("fringefit")
    toolviper.utils.data.download(file=MSV2_NAME, folder=str(folder))
    convert_msv2_to_processing_set(
        in_file=str(folder / MSV2_NAME),
        out_file=str(folder / "gg084b"),  # the conversion appends ".ps.zarr"
        partition_scheme=PARTITION_SCHEME,
        persistence_mode="w",
        parallel_mode="none",
    )
    ps_xdt = open_processing_set(str(folder / "gg084b.ps.zarr"))
    calibration_tree = fringefit_ps(
        ps_xdt, REFERENCE_ANTENNA, FRINGE_FIT_START, FRINGE_FIT_INTERVAL
    )
    corrected_ps_xdt = apply_cal_ps(
        ps_xdt, calibration_tree, FRINGE_FIT_START, FRINGE_FIT_INTERVAL
    )
    # the selection keeps every partition; only those of the interval hold data
    partitions = {
        name: ms_xdt
        for name, ms_xdt in corrected_ps_xdt.items()
        if ms_xdt.time.size > 0
    }
    return {
        "ps_xdt": ps_xdt,
        "calibration_tree": calibration_tree,
        "partitions": partitions,
    }


def baselines_to_reference(ms_xdt):
    """``{other antenna: baseline index}`` of the baselines to the reference antenna."""
    antenna1 = ms_xdt.baseline_antenna1_name.values
    antenna2 = ms_xdt.baseline_antenna2_name.values
    baselines = {}
    for index, (name1, name2) in enumerate(zip(antenna1, antenna2, strict=True)):
        if name1 != name2 and REFERENCE_ANTENNA in (name1, name2):
            baselines[str(name2 if name1 == REFERENCE_ANTENNA else name1)] = index
    return baselines


def unflagged(ms_xdt, variable, baseline, polarization):
    """Visibilities ``[time, frequency]`` of one baseline and correlation, flagged samples as NaN."""
    visibility = (
        ms_xdt[variable].isel(baseline_id=baseline).sel(polarization=polarization)
    )
    flag = ms_xdt["FLAG"].isel(baseline_id=baseline).sel(polarization=polarization)
    return np.where(flag.values.astype(bool), np.nan, visibility.values)


def coherence(visibility):
    """``|mean V| / mean |V|``: 1 for a constant phase, near 0 when the phases wind."""
    return np.abs(np.nanmean(visibility)) / np.nanmean(np.abs(visibility))


def phase_scatter_over_frequency(visibility):
    """Standard deviation (radians) of the unwrapped phase of the time-averaged spectrum."""
    spectrum = np.nanmean(visibility, axis=0)
    spectrum = spectrum[np.isfinite(spectrum)]
    return np.std(np.unwrap(np.angle(spectrum)))


def test_the_interval_holds_one_partition_per_spectral_window(fringe_fit):
    ps_xdt, partitions = fringe_fit["ps_xdt"], fringe_fit["partitions"]
    assert len(ps_xdt) == 10
    spectral_windows = sorted(
        ms_xdt.frequency.attrs["spectral_window_name"] for ms_xdt in partitions.values()
    )
    assert spectral_windows == ["spw_0", "spw_1"]
    for ms_xdt in partitions.values():
        assert list(ms_xdt.polarization.values) == ["RR", "RL", "LR", "LL"]
        assert "VISIBILITY_CORRECTED" in ms_xdt.data_vars
        assert TARGET_ANTENNA in baselines_to_reference(ms_xdt)
        assert len(baselines_to_reference(ms_xdt)) == 6  # seven antennas


def test_calibration_solutions(fringe_fit):
    calibration_tree = fringe_fit["calibration_tree"]
    assert len(calibration_tree) == 2  # one solution set per spectral window
    for solution in calibration_tree.values():
        assert list(solution.cal_parameter.values) == ["phase", "delay", "rate"]
        assert list(solution.polarization.values) == ["R", "L"]
        assert list(solution.ref_antenna.values) == [REFERENCE_ANTENNA]
        assert solution.sizes["antenna_name"] == 7
        parameters = solution.CALIBRATION_PARAMETER
        assert np.isfinite(parameters.values).all()
        assert np.isfinite(solution.SNR.values).all()
        # the reference antenna carries no correction (a phase of order 1e-10 rad is left)
        reference = parameters.sel(antenna_name=REFERENCE_ANTENNA).values
        assert np.abs(reference).max() < 1e-6, reference  # numerically zero
        # the phased VLA: a strong detection with a delay of -62.5 ns (R) and
        # -70.3 ns (L); the FFT delay search has a resolution of 7.8 ns
        target = parameters.sel(antenna_name=TARGET_ANTENNA)
        delay = target.sel(cal_parameter="delay").values.ravel()
        assert np.all((delay > -80e-9) & (delay < -55e-9)), delay
        snr = solution.SNR.sel(antenna_name=TARGET_ANTENNA).values
        assert snr.min() > 50.0, snr
        # every other antenna is detected too, with a delay of at most two resolution elements
        others = [
            str(name)
            for name in solution.antenna_name.values
            if name not in (REFERENCE_ANTENNA, TARGET_ANTENNA)
        ]
        assert solution.SNR.sel(antenna_name=others).values.min() > 8.0
        other_delays = parameters.sel(antenna_name=others, cal_parameter="delay")
        assert np.abs(other_delays.values).max() < 20e-9


def test_correction_makes_the_target_baseline_coherent(fringe_fit):
    """Before the correction the phases wind across the band and the visibilities cancel."""
    for name, ms_xdt in fringe_fit["partitions"].items():
        baseline = baselines_to_reference(ms_xdt)[TARGET_ANTENNA]
        for polarization in PARALLEL_HANDS:
            raw = unflagged(ms_xdt, "VISIBILITY", baseline, polarization)
            corrected = unflagged(
                ms_xdt, "VISIBILITY_CORRECTED", baseline, polarization
            )
            label = f"{name} {polarization}"
            assert coherence(raw) < 0.2, label
            assert coherence(corrected) > 0.6, label
            assert phase_scatter_over_frequency(raw) > 1.0, label
            assert phase_scatter_over_frequency(corrected) < 0.3, label
            # a phase correction: the amplitudes are untouched
            np.testing.assert_allclose(
                np.abs(corrected), np.abs(raw), rtol=1e-6, equal_nan=True, err_msg=label
            )


def test_correction_does_not_degrade_the_other_baselines(fringe_fit):
    for name, ms_xdt in fringe_fit["partitions"].items():
        for antenna, baseline in baselines_to_reference(ms_xdt).items():
            for polarization in PARALLEL_HANDS:
                raw = unflagged(ms_xdt, "VISIBILITY", baseline, polarization)
                corrected = unflagged(
                    ms_xdt, "VISIBILITY_CORRECTED", baseline, polarization
                )
                assert coherence(corrected) > coherence(raw) - 0.02, (
                    f"{name} {REFERENCE_ANTENNA}-{antenna} {polarization}"
                )
