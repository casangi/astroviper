"""Tests of find_continuum_ultra's channel selection against the vendored
findContinuum: the inputs of the port's rounds on the synthetic cubes of the
find_continuum tests, seeded random spectra and chosen channel thresholds."""

import copy
import inspect
import os
import random
import sys
import warnings

import numpy as np
import pytest
from xradio.image import write_image

from astroviper.distributed_applications.imaging import find_continuum
from astroviper.processing_functions.imaging.find_continuum_ultra import (
    channel_selection,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.channel_selection import (
    all_continuum_selected,
    continuum_channel_threshold,
    reject_narrow_inner_runs,
    select_continuum_channels,
    trim_runs,
    widest_run_in_other_half,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    channels_of_ranges,
    range_count,
    ranges_of_channels,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    ChannelThreshold,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    CASA_RESULTS,
    make_find_continuum_image_xds,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    findContinuum,
    pick,
    quiet,
    random_entry,
    random_ranges,
    vendored_block,
)

# Random spectra per test; the manual run sets 4000.
N_RANDOM = int(os.environ.get("FIND_CONTINUUM_ULTRA_RANDOM_SPECTRA", "400"))
ROUND_KINDS = {
    "": "single",
    ".original": "original",
    ".amendedMask": "amended_mask",
    ".onlyExtraMask": "only_extra_mask",
    ".extraMask": "extra_mask",
    ".autoLower": "auto_lower",
    ".autoLower2": "auto_lower_2",
}
THRESHOLD_SIGMA_MODES = {"auto": "auto", "autolower": "auto_lower"}
# Locals of findContinuumChannels (segment_4.md 15.1) and their names here.
THRESHOLD_INTERNALS = {
    "sigma_effective": "sigmaEffective",
    "percentile": "percentile",
    "peak_feature_sigma": "peakFeatureSigma",
    "mad_original": "madOriginal",
}


def _same(value, expected):
    """Equal bit for bit: NaN equals NaN, arrays and lists element by element
    in their order, None only None."""
    if value is None or expected is None:
        return value is None and expected is None
    if isinstance(value, str) or isinstance(expected, str):
        return value == expected
    value, expected = np.asarray(value), np.asarray(expected)
    if value.shape != expected.shape:
        return False
    if value.dtype.kind in "fc" or expected.dtype.kind in "fc":
        return bool(np.array_equal(value, expected, equal_nan=True))
    return bool(np.array_equal(value, expected))


def _same_calls(calls, expected):
    return len(calls) == len(expected) and all(
        len(call) == len(other) and all(map(_same, call, other))
        for call, other in zip(calls, expected, strict=True)
    )


def _ranges(selection):
    """``[start, stop)`` pairs of findContinuum's selection string, in its order."""
    if not selection:
        return []
    return [
        [int(first), int(last) + 1]
        for first, last in (token.split("~") for token in selection.split(";"))
    ]


def _selection(ranges):
    """findContinuum's selection string of ``[start, stop)`` pairs."""
    return ";".join(f"{start}~{stop - 1}" for start, stop in ranges)


def _positions(positions):
    """findContinuum's non-zero MAD positions, a tuple holding one array."""
    return None if positions is None else np.asarray(positions[0])


def _threshold_of_returned(returned):
    """findContinuumChannels' 23 returned values (fC 7074-7079) as a
    ChannelThreshold."""
    return ChannelThreshold(
        continuum_ranges=_ranges(returned[1]),
        group_count=returned[4],
        positive_threshold=returned[2],
        negative_threshold=returned[9],
        median=returned[3],
        corrected_median=returned[6],
        mad=returned[7],
        correction_factor=returned[5],
        signal_ratio=returned[20],
        line_strength_factor=returned[10],
        groups_above_threshold=returned[12],
        single_channel_peaks_above_threshold=returned[11],
        spectral_difference=returned[13][0],
        spectral_difference_2=returned[13][1],
        trim=returned[14],
        use_low_baseline=returned[15],
        narrow=returned[16],
        baseline_channels=returned[17][0],
        baseline_values=returned[17][1],
        mad_ratio=returned[18],
        use_middle_baseline=returned[19],
        ranges_dropped=returned[21],
        nonzero_mad_positions=_positions(returned[22]),
    )


def _threshold_differences(threshold, returned):
    expected = _threshold_of_returned(returned)
    differences = [
        name
        for name in ChannelThreshold._fields
        if not _same(getattr(threshold, name), getattr(expected, name))
    ]
    if not _same(
        channels_of_ranges(threshold.continuum_ranges),
        np.asarray(returned[0], dtype=np.int64).ravel(),
    ):
        differences.append("channels")
    return differences


def _threshold_arguments(arguments):
    """The arguments of continuum_channel_threshold from findContinuumChannels'."""
    return (
        arguments["spectrum"],
        arguments["nBaselineChannels"],
        arguments["sigmaFindContinuum"],
        arguments["trimChannels"],
        arguments["maxTrim"],
        arguments["narrow"],
        arguments["peakOverMad"],
        arguments["signalRatioTier1"],
        THRESHOLD_SIGMA_MODES[arguments["sigmaFindContinuumMode"]],
        ROUND_KINDS[arguments["amendMaskIterationName"]],
        _positions(arguments["idxNonZeroMAD"]),
    )


def _selection_arguments(entry):
    """The arguments of select_continuum_channels from runFindContinuum's
    locals at fC 5228."""
    threshold_sigma = entry["sigmaFindContinuum"]
    return dict(
        spectrum=entry["avgSpectrumNansReplaced"].copy(),
        n_baseline_channels=entry["nBaselineChannels"],
        channel_width=entry["channelWidth"],
        first_frequency=entry["firstFreq"],
        last_frequency=entry["lastFreq"],
        threshold_sigma=(
            "auto" if threshold_sigma in ("auto", "autolower", -1) else threshold_sigma
        ),
        threshold_sigma_mode=THRESHOLD_SIGMA_MODES[entry["sigmaFindContinuumMode"]],
        signal_ratio_tier_1=entry["signalRatioTier1"],
        round_kind=ROUND_KINDS[entry["amendMaskIterationName"]],
        moment_difference_snr=entry["momDiffSNR"],
        n_pixels_maximum_mask=entry["numberPixelsInMom8Mask"],
        integrated_peak=entry.get("mom0peak"),
        maximum_peak=entry.get("mom8peak"),
        single_continuum=entry["singleContinuum"],
        nonzero_mad_positions=_positions(entry["idxNonZeroMAD"]),
        rounds_enabled=entry["amendMask"],
    )


def _selection_differences(selection, block_locals):
    """Differences of a ChannelSelection from runFindContinuum's locals at
    fC 5862."""
    added = block_locals["secondSelectionAdded"]
    expected = {
        "continuum_ranges": _ranges(block_locals["selection"]),
        "ranges_before_blue_pruning": _ranges(block_locals["selectionPreBluePruning"]),
        "final_threshold_sigma": block_locals["sigmaFindContinuum"],
        "positive_threshold": block_locals["threshold"],
        "negative_threshold": block_locals["negativeThreshold"],
        "corrected_median": block_locals["medianTrue"],
        "median": block_locals["median"],
        "mad": block_locals["mad"],
        "baseline_channels": block_locals["allBaselineChannelsXY"][0],
        "baseline_values": block_locals["allBaselineChannelsXY"][1],
        "nonzero_mad_positions": _positions(block_locals["idxNonZeroMAD"]),
        "peak_over_mad": block_locals["peakOverMad"],
    }
    figure_values = {
        "peak_minus_median_over_mad": block_locals["peakMinusMedianOverMad"],
        "signal_ratio": block_locals["signalRatio"],
        "mad_ratio": block_locals["madRatio"],
        "use_middle_baseline": block_locals["useMiddleChannels"],
        "correction_factor": block_locals["correctionFactor"],
        "trim": block_locals["trimChannels"],
        "max_trim": block_locals["maxTrim"],
        "narrow": block_locals["narrow"],
        "added_range": _ranges(added)[0] if added else None,
    }
    assert selection.figure_values.keys() == figure_values.keys()
    return [
        name
        for name, value in expected.items()
        if not _same(getattr(selection, name), value)
    ] + [
        f"figure_values[{name!r}]"
        for name, value in figure_values.items()
        if not _same(selection.figure_values[name], value)
    ]


def _outcome(failures, function, *args, **kwargs):
    """``"failure"`` for an exception of ``failures``, ``"TypeError"`` for a
    TypeError, else ``"ok"`` and the result."""
    try:
        with quiet(), np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", function(*args, **kwargs)
    except failures:
        return "failure", None
    except TypeError:
        return "TypeError", None


def _native_outcome(function, *args, **kwargs):
    return _outcome(ValueError, function, *args, **kwargs)


def _vendored_outcome(function, *args, **kwargs):
    """findContinuum fails with ValueError or ZeroDivisionError where the
    native functions raise ValueError."""
    return _outcome((ValueError, ZeroDivisionError), function, *args, **kwargs)


def _block_outcome(block, entry):
    """The outcome of runFindContinuum's lines fC 5228-5862: no result
    (fC 5683) and an empty selection, on which its figure fails (fC 5942), are
    failures."""
    status, block_locals = _vendored_outcome(block, **entry)
    if status == "ok" and (block_locals is None or block_locals["selection"] == ""):
        return "failure", None
    return status, block_locals


def _selection_block(parameter_names):
    return vendored_block(
        5228,
        "if narrow == 'auto':",
        5862,
        "groups = selection.count(';')+1",
        sorted(parameter_names),
    )


def _record_threshold_calls(monkeypatch, block, results=None):
    """Record the arguments of every channel threshold of the selection block
    and of the native selection, as arguments of continuum_channel_threshold;
    the thresholds return ``results`` in order when given."""
    signature = inspect.signature(findContinuum.findContinuumChannels)
    vendored_calls, native_calls = [], []

    def vendored(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        vendored_calls.append(_threshold_arguments(arguments.arguments))
        if results is None:
            return findContinuum.findContinuumChannels(*args, **kwargs)
        return copy.deepcopy(results[len(vendored_calls) - 1])

    def native(*args):
        native_calls.append(args)
        if results is None:
            return continuum_channel_threshold(*args)
        return _threshold_of_returned(copy.deepcopy(results[len(native_calls) - 1]))

    block.__globals__["findContinuumChannels"] = vendored
    monkeypatch.setattr(channel_selection, "continuum_channel_threshold", native)
    return vendored_calls, native_calls


def _with_locals(function, *args, **kwargs):
    """``function(*args, **kwargs)`` and its local variables at its return."""
    captured = {}

    def profile(frame, event, arg):
        if event == "return" and frame.f_code is function.__code__:
            captured.update(frame.f_locals)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        return function(*args, **kwargs), captured
    finally:
        sys.setprofile(previous)


@pytest.fixture(scope="module")
def port_rounds(tmp_path_factory):
    """The rounds of the port's runs on the synthetic cubes of CASA_RESULTS:
    runFindContinuum's locals when it first calls findContinuumChannels, with
    its arguments ``narrow`` and ``sigmaFindContinuum`` (its locals at
    fC 5228), and the arguments of every findContinuumChannels call."""
    run_find_continuum = findContinuum.runFindContinuum
    find_continuum_channels = findContinuum.findContinuumChannels
    run_signature = inspect.signature(run_find_continuum)
    channels_signature = inspect.signature(find_continuum_channels)
    rounds = []
    case = None

    def run(*args, **kwargs):
        arguments = run_signature.bind(*args, **kwargs).arguments
        rounds.append(
            {
                "case": case,
                "entry": None,
                "calls": [],
                "arguments": {
                    "narrow": arguments["narrow"],
                    "sigmaFindContinuum": arguments["sigmaFindContinuum"],
                },
            }
        )
        return run_find_continuum(*args, **kwargs)

    def channels(*args, **kwargs):
        record = rounds[-1]
        if record["entry"] is None:
            record["entry"] = {}
            for name, value in sys._getframe(1).f_locals.items():
                try:
                    record["entry"][name] = copy.deepcopy(value)
                except TypeError:  # an open file; fC 5228-5862 do not read it
                    pass
            record["entry"].update(record["arguments"])
        arguments = channels_signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        record["calls"].append(copy.deepcopy(dict(arguments.arguments)))
        return find_continuum_channels(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(findContinuum, "runFindContinuum", run)
        patch.setattr(findContinuum, "findContinuumChannels", channels)
        for case, (cube, *_) in CASA_RESULTS.items():
            directory = tmp_path_factory.mktemp(case)
            store = str(directory / f"{case}.img.zarr")
            write_image(
                make_find_continuum_image_xds(**cube),
                imagename=store,
                out_format="zarr",
                overwrite=True,
            )
            patch.chdir(directory)
            find_continuum(
                input_image_store=store,
                work_directory=str(directory / "work"),
                atmospheric_transmission=np.full(480, 0.92),
                atmospheric_sky_temperature=np.full(480, 27.0),
            )
    return rounds


def test_threshold_equals_the_ports_on_the_synthetic_cubes(port_rounds):
    """Every channel threshold of the port's rounds, with its internals."""
    calls = [arguments for record in port_rounds for arguments in record["calls"]]
    assert len(calls) > len(port_rounds)
    for arguments in calls:
        with quiet():
            returned, vendored_locals = _with_locals(
                findContinuum.findContinuumChannels, **arguments
            )
        threshold, native_locals = _with_locals(
            continuum_channel_threshold, *_threshold_arguments(arguments)
        )
        assert not _threshold_differences(threshold, returned)
        for name, vendored_name in THRESHOLD_INTERNALS.items():
            assert _same(native_locals[name], vendored_locals[vendored_name]), name


def test_selection_equals_the_ports_on_the_synthetic_cubes(port_rounds, monkeypatch):
    """Every round of the port, its channel thresholds called with the same
    arguments; each cube's last round selects CASA's channels."""
    rounds = [record["entry"]["amendMaskIterationName"] for record in port_rounds]
    assert rounds.count(".original") == len(CASA_RESULTS) and ".amendedMask" in rounds
    for record in port_rounds:
        entry = record["entry"]
        block = _selection_block(entry)
        vendored_calls, native_calls = _record_threshold_calls(monkeypatch, block)
        with quiet():
            block_locals = block(**entry)
        selection = select_continuum_channels(**_selection_arguments(entry))
        assert not _selection_differences(selection, block_locals)
        assert _same_calls(native_calls, vendored_calls)
        assert _same_calls(
            native_calls, [_threshold_arguments(call) for call in record["calls"]]
        )
    for case, (_, casa_selection, *_) in CASA_RESULTS.items():
        last_round = [record for record in port_rounds if record["case"] == case][-1]
        selection = select_continuum_channels(
            **_selection_arguments(last_round["entry"])
        )
        assert _selection(selection.continuum_ranges) == casa_selection, case


def _random_threshold_arguments(rng):
    entry = random_entry(rng, {})
    spectrum = entry["avgSpectrumNansReplaced"]
    positions = None
    if rng.random() < 0.1:
        positions = (
            np.sort(
                rng.choice(
                    np.arange(max(1, spectrum.size - 2)),
                    size=max(1, spectrum.size // 2),
                    replace=False,
                )
            ),
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_over_mad = np.max(spectrum) / findContinuum.MAD(spectrum)
    return dict(
        spectrum=spectrum,
        nBaselineChannels=entry["nBaselineChannels"],
        sigmaFindContinuum=pick(rng, [2.0, 2.5, 3.6, 4.2, 4.5, 6.3, 7.2]),
        nanmin=entry["nanmin"],
        baselineMode="min",
        trimChannels=pick(rng, ["auto", "auto", "auto", 2, 6, 13]),
        narrow=int(np.ceil(np.log10(spectrum.size))),
        verbose=False,
        maxTrim=pick(rng, [20, 20, 75.0, 1.25, 300.0]),
        maxTrimFraction=1.0,
        separator=";",
        peakOverMad=peak_over_mad,
        negativeThresholdFactor=1.15,
        dropBaselineChannels=2.0,
        madRatioUpperLimit=1.5,
        madRatioLowerLimit=1.15,
        signalRatioTier1=entry["signalRatioTier1"],
        signalRatioTier2=0.94,
        sigmaFindContinuumMode=entry["sigmaFindContinuumMode"],
        enableRejectNarrowInnerWindows=True,
        useUncorrectedMedian=False,
        returnBluePoints=False,
        removeOutlierBaselineChannels="",
        amendMaskIterationName=entry["amendMaskIterationName"],
        useBaseline="auto",
        idxNonZeroMAD=positions,
    )


def test_threshold_equals_findcontinuum_on_random_spectra():
    """Including the internals of segment_4.md 15.1 and the same failures; a
    float maximum trim of at least 1 (off the pipeline path) makes both fail
    with a TypeError when the automatic trim picks it."""
    rng = np.random.default_rng(20261114)
    statuses = []
    for trial in range(N_RANDOM):
        arguments = _random_threshold_arguments(rng)
        status, vendored = _vendored_outcome(
            _with_locals, findContinuum.findContinuumChannels, **arguments
        )
        native_status, native = _native_outcome(
            _with_locals, continuum_channel_threshold, *_threshold_arguments(arguments)
        )
        assert native_status == status, (trial, status, native_status)
        statuses.append(status)
        if status != "ok":
            continue
        (returned, vendored_locals), (threshold, native_locals) = vendored, native
        assert not _threshold_differences(threshold, returned), trial
        for name, vendored_name in THRESHOLD_INTERNALS.items():
            assert _same(native_locals[name], vendored_locals[vendored_name]), (
                trial,
                name,
            )
    assert statuses.count("ok") > N_RANDOM / 2 and "failure" in statuses


def test_selection_equals_findcontinuum_on_random_spectra(port_rounds, monkeypatch):
    template = port_rounds[0]["entry"]
    block = _selection_block(template)
    rng = np.random.default_rng(20261115)
    statuses = []
    for trial in range(N_RANDOM):
        entry = random_entry(rng, template)
        vendored_calls, native_calls = _record_threshold_calls(monkeypatch, block)
        status, block_locals = _block_outcome(block, entry)
        native_status, selection = _native_outcome(
            select_continuum_channels, **_selection_arguments(entry)
        )
        assert native_status == status, (trial, status, native_status)
        assert _same_calls(native_calls, vendored_calls), trial
        statuses.append(status)
        if status == "ok":
            assert not _selection_differences(selection, block_locals), trial
    assert statuses.count("ok") > N_RANDOM / 2 and "failure" in statuses


def _random_thresholds(rng, entry):
    """Three findContinuumChannels results, as it returns them, chosen to reach
    the branches of fC 5321-5712 (segment_4_check.py's stubs)."""
    spectrum = entry["avgSpectrumNansReplaced"]
    n_channels = spectrum.size
    median, spread = float(np.median(spectrum)), float(np.std(spectrum))
    results = []
    for _ in range(3):
        n_ranges = pick(rng, [0, 1, 1, 2, 3, 7, 8, 9, 15, 16, 33, 40])
        cuts = np.sort(
            rng.choice(n_channels, size=min(n_channels, 2 * n_ranges), replace=False)
        )
        channels = [
            channel
            for first, last in zip(cuts[0::2], cuts[1::2], strict=False)
            for channel in range(int(first), int(last) + 1)
        ]
        if rng.random() < 0.15 and n_channels > 20:
            first = int(rng.integers(0, n_channels - 3))
            channels = list(
                range(first, first + int(rng.integers(1, max(2, n_channels // 25))))
            )
        channels = np.array(sorted(set(channels)), dtype=np.int64)
        n_peaks = int(rng.integers(0, 6))
        single_peaks = (
            n_peaks if rng.random() < 0.3 else n_peaks - int(rng.integers(0, 20))
        )
        n_blue = int(rng.integers(0, max(1, n_channels // 3)))
        if rng.random() < 0.5:
            blue = np.sort(rng.choice(n_channels, size=n_blue, replace=False))
        else:
            first = int(rng.integers(0, max(1, n_channels - n_blue)))
            blue = np.arange(first, min(n_channels, first + n_blue))
            if rng.random() < 0.5 and n_blue > 3:
                blue = np.concatenate([blue[: n_blue // 3], blue[2 * n_blue // 3 :]])
        blue = blue.astype(np.int64)
        ranges = ranges_of_channels(channels)
        results.append(
            (
                channels,
                _selection(ranges),
                float(median + rng.uniform(0, 5) * spread),
                median,
                range_count(ranges),
                2.6,
                median,
                spread,
                2.9,
                median - spread,
                1.1,
                single_peaks,
                n_peaks,
                [1.0, 2.0],
                pick(rng, ["auto", "auto", "auto", "auto", 6, 13, 2]),
                bool(rng.random() < 0.8),
                int(np.ceil(np.log10(n_channels))),
                [blue, spectrum[blue]],
                None if rng.random() < 0.3 else float(rng.uniform(0.9, 1.6)),
                bool(rng.random() < 0.2),
                float(rng.uniform(0.5, 1.0)),
                int(rng.integers(0, 3)),
                None,
            )
        )
    return results


def test_selection_with_chosen_thresholds_equals_findcontinuum(
    port_rounds, monkeypatch
):
    """The branches the computed thresholds rarely reach: noise spikes, many
    ranges, the maximum mask, gaps between baseline channels, the range added
    to a single narrow range, and the failures."""
    template = port_rounds[0]["entry"]
    block = _selection_block(template)
    rng = np.random.default_rng(20261116)
    statuses = []
    for trial in range(N_RANDOM):
        entry = random_entry(rng, template)
        vendored_calls, native_calls = _record_threshold_calls(
            monkeypatch, block, _random_thresholds(rng, entry)
        )
        status, block_locals = _block_outcome(block, entry)
        native_status, selection = _native_outcome(
            select_continuum_channels, **_selection_arguments(entry)
        )
        assert native_status == status, (trial, status, native_status)
        assert _same_calls(native_calls, vendored_calls), trial
        statuses.append(status)
        if status == "ok":
            assert not _selection_differences(selection, block_locals), trial
    assert statuses.count("ok") > N_RANDOM / 2 and "failure" in statuses


@pytest.mark.parametrize(
    "spectrum, n_baseline_channels, match, error",
    [
        (
            np.random.default_rng(1).standard_normal(16),
            3,
            "between the thresholds",
            ValueError,
        ),
        (np.arange(12.0) ** 2, 2, "twice the threshold", ZeroDivisionError),
    ],
)
def test_threshold_failures(spectrum, n_baseline_channels, match, error):
    """A threshold sigma of 0 leaves no channel strictly between the
    thresholds or, when no channel holds the baseline median, puts every
    channel beyond twice the threshold (a signal ratio of 0)."""
    narrow = int(np.ceil(np.log10(spectrum.size)))
    with pytest.raises(ValueError, match=match):
        continuum_channel_threshold(
            spectrum,
            n_baseline_channels,
            0.0,
            "auto",
            20,
            narrow,
            5.0,
            0.967,
            "auto",
            "original",
            None,
        )
    with quiet(), pytest.raises(error):
        findContinuum.findContinuumChannels(
            spectrum,
            n_baseline_channels,
            0.0,
            peakOverMad=5.0,
            signalRatioTier1=0.967,
            amendMaskIterationName=".original",
        )


def test_corrected_median_below_the_spectrum_is_the_median():
    """Flagged channels at zero make the highest values the baseline; their
    zero deviation sends the baseline to the lowest values, which are then
    corrected downwards below the spectrum (fC 6915-6917)."""
    spectrum = -0.01 - 0.001 * (7 * np.arange(32.0) % 32)
    spectrum[10:20] = 0.0
    spectrum[5], spectrum[25] = 0.003, 0.002
    threshold = continuum_channel_threshold(
        spectrum, 6, 4.2, "auto", 20, 2, 5.0, 0.967, "auto", "original", None
    )
    with quiet():
        returned = findContinuum.findContinuumChannels(
            spectrum,
            6,
            4.2,
            peakOverMad=5.0,
            signalRatioTier1=0.967,
            amendMaskIterationName=".original",
        )
    assert not _threshold_differences(threshold, returned)
    assert not threshold.use_low_baseline and not threshold.use_middle_baseline
    assert threshold.corrected_median == np.median(spectrum)


def _chosen_threshold(continuum_ranges, baseline_channels, spectrum, **values):
    """A ChannelThreshold with the given ranges and baseline channels."""
    defaults = dict(
        group_count=range_count(continuum_ranges),
        positive_threshold=float(np.median(spectrum) + np.std(spectrum)),
        negative_threshold=float(np.median(spectrum) - np.std(spectrum)),
        median=float(np.median(spectrum)),
        corrected_median=float(np.median(spectrum)),
        mad=float(np.std(spectrum)),
        correction_factor=2.6,
        signal_ratio=0.9,
        line_strength_factor=1.1,
        groups_above_threshold=0,
        single_channel_peaks_above_threshold=0,
        spectral_difference=1.0,
        spectral_difference_2=2.0,
        trim="auto",
        use_low_baseline=True,
        narrow=int(np.ceil(np.log10(spectrum.size))),
        baseline_values=spectrum[baseline_channels],
        mad_ratio=None,
        use_middle_baseline=False,
        ranges_dropped=0,
        nonzero_mad_positions=None,
    )
    return ChannelThreshold(
        continuum_ranges=continuum_ranges,
        baseline_channels=np.asarray(baseline_channels, dtype=np.int64),
        **{**defaults, **values},
    )


def _selection_of(spectrum, channel_width=976562.5, **arguments):
    return select_continuum_channels(
        **{
            "spectrum": spectrum,
            "n_baseline_channels": 6,
            "channel_width": channel_width,
            "first_frequency": 86e9,
            "last_frequency": 86e9 + (spectrum.size - 1) * channel_width,
            "threshold_sigma": "auto",
            "threshold_sigma_mode": "auto",
            "signal_ratio_tier_1": 0.967,
            "round_kind": "original",
            "moment_difference_snr": -1.0,
            "n_pixels_maximum_mask": 0,
            "integrated_peak": 1.0,
            "maximum_peak": 0.1,
            "single_continuum": False,
            "nonzero_mad_positions": None,
            "rounds_enabled": True,
            **arguments,
        }
    )


def test_selection_failures(monkeypatch):
    """No channel below the corrected median (fC 6472), a gap of a
    time-division-mode spectrum without a channel (fC 5619), and no range
    without rounds (fC 5683) or at all (fC 5942)."""
    spectrum = np.cos(np.arange(32.0)) + 10.0

    def selection_with(threshold, **arguments):
        monkeypatch.setattr(
            channel_selection, "continuum_channel_threshold", lambda *args: threshold
        )
        return _selection_of(spectrum, **arguments)

    threshold = _chosen_threshold([[2, 30]], [5, 6, 20], spectrum)
    with pytest.raises(ValueError, match="below the corrected median"):
        selection_with(threshold._replace(corrected_median=float(spectrum.min())))
    # A range of 31 of the 32 channels and a peak in the maximum mask lower the
    # threshold sigma; the minimum gap is then negative, so a step of 1 between
    # baseline channels is a gap.
    with pytest.raises(ValueError, match="holds no channel"):
        selection_with(
            _chosen_threshold([[0, 31]], [10, 11, 12], spectrum),
            channel_width=15.625e6,
            n_pixels_maximum_mask=10,
            maximum_peak=1.0,
        )
    empty = _chosen_threshold([], [3, 4, 5], spectrum)
    with pytest.raises(ValueError, match="has no continuum channel range"):
        selection_with(empty, rounds_enabled=False)
    with pytest.raises(ValueError, match="no baseline channels in the other half"):
        selection_with(empty)
    selection = selection_with(_chosen_threshold([], [3, 4, 20, 21, 22], spectrum))
    assert selection.continuum_ranges == [[20, 23]]
    assert selection.figure_values["added_range"] == [20, 23]


def test_single_narrow_range_gets_a_range_of_the_other_half(monkeypatch):
    """The widest baseline run of the other half is appended after a single
    range of less than 5 % of the channels, or put before it when it starts
    at or below the range's last channel, even where they overlap."""
    spectrum = np.cos(np.arange(100.0)) + 3.0  # peak over MAD below 6: no pruning
    cases = [
        ([10, 13], [60, 61, 62, 70, 71], [60, 63], True),
        ([80, 83], [5, 6, 7, 20, 21], [5, 8], False),
        ([48, 52], [50, 51, 70], [50, 52], False),
    ]
    for single_range, baseline_channels, added_range, appended in cases:
        threshold = _chosen_threshold([single_range], baseline_channels, spectrum)
        monkeypatch.setattr(
            channel_selection,
            "continuum_channel_threshold",
            lambda *args, threshold=threshold: threshold,
        )
        selection = _selection_of(spectrum)
        assert selection.continuum_ranges == (
            [single_range, added_range] if appended else [added_range, single_range]
        )
        assert selection.ranges_before_blue_pruning == [single_range]
        assert selection.figure_values["added_range"] == added_range


def test_trim_runs_equals_findcontinuum():
    rng = np.random.default_rng(2)
    for _ in range(600):
        n_channels = pick(rng, [8, 40, 74, 75, 257, 480, 3840])
        channels = np.flatnonzero(rng.random(n_channels) < rng.uniform(0.2, 1.0))
        if channels.size == 0:
            continue
        trim = pick(rng, ["auto", "auto", 0.1, 0.5, 2, 6, 13, 0])
        max_trim = pick(rng, [20, 20, 3, 0.1, 75.0])
        status, expected = _vendored_outcome(
            findContinuum.splitListIntoContiguousListsAndTrim,
            n_channels,
            channels,
            trim,
            max_trim,
        )
        native_status, kept = _native_outcome(
            trim_runs, n_channels, channels, trim, max_trim
        )
        assert native_status == status
        if status == "ok":
            np.testing.assert_array_equal(kept, np.asarray(expected, dtype=np.int64))
    assert trim_runs(480, np.arange(40, 100), -1, 20).tolist() == list(range(40, 100))


def test_reject_narrow_inner_runs_equals_findcontinuum():
    rng = np.random.default_rng(3)
    for _ in range(400):
        channels = np.flatnonzero(rng.random(240) < rng.uniform(0.05, 0.95))
        expected, _, n_dropped = findContinuum.rejectNarrowInnerWindowsChannels(
            channels
        )
        kept, n_dropped_native = reject_narrow_inner_runs(channels)
        np.testing.assert_array_equal(kept, np.asarray(expected, dtype=np.int64))
        assert n_dropped_native == n_dropped
    channels = np.array([0, 1, 2, 5, 9, 10, 11])
    assert reject_narrow_inner_runs(channels)[0].tolist() == [0, 1, 2, 9, 10, 11]
    assert reject_narrow_inner_runs(channels)[1] == 1


def test_widest_run_in_other_half_equals_findcontinuum():
    rng = np.random.default_rng(4)
    for _ in range(400):
        n_channels = pick(rng, [8, 9, 120, 480])
        channels = rng.permutation(
            np.flatnonzero(rng.random(n_channels) < rng.uniform(0.0, 0.6))
        )
        for lower_half in (True, False):
            expected = findContinuum.findWidestContiguousListInOtherHalf(
                channels, lower_half, n_channels
            )
            run = widest_run_in_other_half(channels, lower_half, n_channels)
            assert (run is None) == (expected is None)
            if run is not None:
                assert run.tolist() == expected
    channels = np.array([2, 3, 7, 8, 9, 12, 13, 14, 15])
    assert widest_run_in_other_half(channels, True, 20).tolist() == [12, 13, 14, 15]
    assert widest_run_in_other_half(channels, False, 20).tolist() == [7, 8, 9]


def test_all_continuum_selected_equals_findcontinuum():
    rng = random.Random(5)
    cases = [(ranges, n) for n in (8, 74, 75, 480) for ranges in ([], [[0, n]])]
    cases += [([[0, stop]], 40) for stop in range(30, 41)]
    cases += [([[1, stop]], 75) for stop in range(65, 76)]
    cases += [([[start, 200]], 200) for start in range(10, 20)]
    for _ in range(300):
        n_channels = rng.choice([8, 40, 74, 75, 480, 3840])
        cases.append((random_ranges(rng, n_channels, max_ranges=2), n_channels))
    for ranges, n_channels in cases:
        with quiet():
            expected = findContinuum.allContinuumSelected(
                _selection(ranges), n_channels
            )
        assert all_continuum_selected(ranges, n_channels) is expected, (
            ranges,
            n_channels,
        )
    assert all_continuum_selected([[0, 37]], 40)  # 0.925 of 40 channels >= 0.91
    assert not all_continuum_selected([[0, 36]], 40)
    assert all_continuum_selected([[15, 200]], 200)  # exactly 0.925
    assert not all_continuum_selected([[16, 200]], 200)
    assert not all_continuum_selected([[0, 100], [100, 200]], 200)
