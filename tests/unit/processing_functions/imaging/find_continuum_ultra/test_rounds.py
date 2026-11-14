"""Tests of find_continuum_ultra's rounds against the vendored findContinuum,
the simple port's lattice expressions and the rounds the port recorded."""

import math
import random
import warnings

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum import (
    evaluate_lattice_expression,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    channels_of_ranges,
    ranges_of_channels,
    round_to_decimals,
    scaled_median_absolute_deviation,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.products import (
    bandwidth_warnings,
    selection_bandwidth,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.rounds import (
    EXTRA_ROUND_KINDS,
    after_amended_intersection,
    after_amended_round,
    after_extra_round,
    after_low_spread_fix,
    after_original_round,
    after_rounds,
    amend_mask_decision,
    amended_joint_mask,
    change_code,
    extra_mask_decision,
    low_spread_ranges,
    only_extra_mask_decision,
    path_code,
    round_maps_ranges,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    BandwidthWarnings,
    CubeNoiseStatistics,
    Decision,
    MapsRanges,
    MomentDifferenceCounts,
    SelectionStatistics,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    findContinuum,
    quiet,
    random_ranges,
    vendored_block,
)

N_CASES = 2000  # random cases per test; 20000 in the manual run (design 14.2)
CHANNEL_WIDTH = 976562.5
FIRST_FREQUENCY = 230e9
NAN = math.nan
NATIVE_DECISION = {
    False: "not_evaluated",
    "No": "no",
    "YesMom": "yes_moment",
    "YesCube": "yes_cube",
    "NoImprovement": "no_improvement",
}
FINDCONTINUUM_DECISION = {native: fc for fc, native in NATIVE_DECISION.items()}
FINDCONTINUUM_ROUND = {
    "only_extra_mask": ".onlyExtraMask",
    "extra_mask": ".extraMask",
    "auto_lower": ".autoLower",
    "auto_lower_2": ".autoLower2",
}
NATIVE_ROUND = {fc: native for native, fc in FINDCONTINUUM_ROUND.items()}

# findContinuum's inline blocks (fC lines) as functions of their locals.
INTERSECTION = vendored_block(
    2548,
    "channelList0 = convertSelectionIntoChannelList(selection0)",
    2550,
    "channelList = np.intersect1d(channelList0,channelList1)",
    ["selection0", "selection"],
)
REFERENCE_RANGES = vendored_block(
    2559,
    "channelList = channelList0  # reverting",
    2560,
    "selection = convertChannelListIntoSelection(channelList) # this line was added "
    "to fix PIPE-1211",
    ["channelList0"],
)
INTERSECTED = vendored_block(
    2594,
    "selection = convertChannelListIntoSelection(channelList)",
    2612,
    "selectionPreBluePruning = selection  # PIPE-2417",
    [
        "channelList",
        "channelList0",
        "amendMaskIterationName",
        "nchan",
        "extraMaskDecision",
        "autoLowerDecision",
        "autoLower2Decision",
    ],
)
DECISION_LOCALS = [
    "amendMaskIterationName",
    "amendMaskIterations",
    "amendMaskIteration",
    "amendMaskIterationNames",
    "badAtmosphere",
    "momDiffMedian",
    "momDiffSNR",
    "momDiffSNRCube",
    "NpixMomDiff",
    "NpixMomDiffBadAtm",
    "NpixCubeDiffMedian",
    "TenEventSigma",
    "momDiffMAD",
    "MADCubeOutside",
    "momDiffLevelForProceeding",
    "momDiffLevelBadAtm",
    "cubeLevel",
    "keepIntermediatePngs",
    "png",
    "signalRatioTier1",
    "sigmaFindContinuumMode",
    "sigmaFindContinuum",
    "finalSigmaFindContinuum",
    "selection",
    "selection0",
]
AUTO_LOWER = vendored_block(
    2683,
    "if (amendMaskIterationName == '.extraMask' and amendMaskIterations > 2) or "
    "(amendMaskIterationName == '.onlyExtraMask' and amendMaskIterations > 1):   "
    "# PL2022",
    2710,
    "casalogPost(\"Using sigmaFindContinuumMode='autolower' with "
    'sigmaFindContinuum=%f" % (sigmaFindContinuum))',
    [
        *DECISION_LOCALS,
        "chanInfo",
        "smallBandwidthFraction",
        "smallSpreadFraction",
        "momDiffApproachLevel",
        "autoLowerDecision",
    ],
)
AUTO_LOWER_2 = vendored_block(
    2717,
    "if amendMaskIterationName != '.autoLower':",
    2735,
    "casalogPost(\"Using sigmaFindContinuumMode='autolower' with "
    'sigmaFindContinuum=%f" % (sigmaFindContinuum))',
    [*DECISION_LOCALS, "autoLower2Decision"],
)
REVERSION = vendored_block(
    2834,
    "if amendMask:",
    2854,
    'casalogPost("Will try to fix the LowSpread condition")    # PL2024',
    [
        "amendMask",
        "momDiffCode",
        "mom8code",
        "momDiffSNR",
        "momDiffLevel",
        "selection",
        "chanInfo",
        "smallBandwidthFraction",
        "smallSpreadFraction",
    ],
)
LOW_SPREAD = vendored_block(
    2856,
    "continuumChannels = convertSelectionIntoChannelList(selection)",
    2920,
    "casalogPost('  %s ****** Added another selection in wider side of the "
    "spectrum: %s' % (projectCode,secondSelectionAdded))",
    [
        "selection",
        "nchan",
        "smallSpreadFraction",
        "allBaselineChannelsXY",
        "avgspectrumAboveThreshold",
        "negativeThreshold",
        "positiveThreshold",
        "separator",
        "projectCode",
    ],
)
LOW_SPREAD_CODE_REVERSION = vendored_block(
    3040,
    "if momDiffCode is None:",
    3050,
    "casalogPost('This momDiff code with momDiffSNR>%.1f triggers reversion to "
    "original result.'%(momDiffLevel))",
    [
        "momDiffCode",
        "mom8code",
        "reversionCodes",
        "revert",
        "momDiffSNR",
        "momDiffLevel",
    ],
)
WARNINGS_REVERSION = vendored_block(
    3051,
    "if len(warnings[0]) > 0 and len(warnings[1]) > 0:",
    3053,
    "casalogPost('This combination of bandwidth warnings triggers reversion to "
    "original result, which had only %d warning(s).' % (len(originalWarnings)))",
    ["warnings", "revert", "originalWarnings"],
)
DECISION_CODE = vendored_block(
    3424,
    "decisionCode = ''",
    3482,
    "decisionCode += '+' # added a selection of channels at the end due to "
    "LowBW/LowSpread",
    [
        "amendMaskDecision",
        "extraMaskDecision",
        "extraMaskDecision2",
        "autoLowerDecision",
        "intersectionOfSelections",
        "useMomentDiff",
        "sigmaUsedToAmendMask",
        "addedSelection",
        "autoLower2Decision",
        "amendMaskIterations",
    ],
)

# The rounds of seven cubes as the simple port ran them (traced at the decision
# functions, the change codes and fC 1859, 1965, 1985, 2113, 2683 and 2834):
# statistics (snr, peak, median, scaled MAD, snr outside, peak outside, sum,
# cube peak) of the moment difference, the cube noise and the counts in the
# order of their named tuples, decisions as (kind, level, sigma used), and the
# final ranges, aggregate bandwidth (GHz) and path code.
PORT_ROUNDS = {
    "line_set": {
        "bad_atmosphere": False,
        "ranges": ("6~85;154~367;433~472", "3~88;153~388;432~476"),
        "final_threshold_sigma": 4.2,
        "statistics": (
            6.250675052139181,
            0.005333462219561835,
            0.002928920764526769,
            0.00038468508360743446,
            4.920249983145049,
            0.0048216675406624,
            12.4360314327894,
            0.007904806174337864,
        ),
        "cube_noise": (
            0.0009892459014610832,
            -7.533929965575226e-05,
            4.413473317906214,
            1.913322838477377,
        ),
        "counts": (0, 0, 0, 0, 0, 0, 0, 0),
        "negative_fraction": 0.0,
        "masked_pixels": None,
        "amend_mask": ("no", 0.0, 0.0),
        "only_extra_mask": ("no", 0.0, 5.5),
        "change_code": "SSSS",
        "result": ("6~85;154~367;433~472", 0.326171875, "S"),
    },
    "line_set_badatm": {
        "bad_atmosphere": True,
        "ranges": ("6~85;154~367;433~472", "3~88;153~388;432~476"),
        "final_threshold_sigma": 4.2,
        "statistics": (
            6.250675052139181,
            0.005333462219561835,
            0.002928920764526769,
            0.00038468508360743446,
            4.920249983145049,
            0.0048216675406624,
            12.4360314327894,
            0.007904806174337864,
        ),
        "cube_noise": (
            0.0009892459014610832,
            -7.533929965575226e-05,
            4.413473317906214,
            1.913322838477377,
        ),
        "counts": (0, 0, 0, 0, 0, 0, 0, 0),
        "negative_fraction": 0.0,
        "masked_pixels": None,
        "amend_mask": ("no", 0.0, 0.0),
        "only_extra_mask": ("no", 0.0, 8.0),
        "change_code": "SSSS",
        "result": ("6~85;154~367;433~472", 0.326171875, "S"),
    },
    "eight_channels": {
        "bad_atmosphere": False,
        "ranges": ("1~1", "1~1"),
        "final_threshold_sigma": 4.2,
        "statistics": (
            3.588337115333422,
            7.270232164555135e-09,
            1.833870372520822e-10,
            1.9749663700826937e-09,
            2.6334711550836576,
            5.384404005125132e-09,
            1.857167811636856e-07,
            NAN,
        ),
        "cube_noise": (
            0.001119895245509845,
            9.757520820130594e-05,
            3.426972195444488,
            4.6441995255594e-06,
        ),
        "counts": (2017.0, 2017.0, 2017.0, 2017.0, 0, 0, 2017.0, 2017.0),
        "negative_fraction": 0.468505859375,
        "masked_pixels": None,
        "amend_mask": ("no", 0.0, -2.0),
        "only_extra_mask": ("not_evaluated", 0.0, 0.0),
        "change_code": "SSSS",
        "low_spread": {
            "baseline_channels": [7, 6],
            "spectrum": [
                0.00035315465507324006,
                0.0004503587386352592,
                0.0005422079011509595,
                0.0056657225428807575,
                0.01706290395813994,
                0.006096894125948893,
                -0.00017096830254672568,
                -0.0004858025438579716,
            ],
            "positive_threshold": 0.0016769564415398414,
            "negative_threshold": -0.002082293409966274,
            "values": (
                0.002889855969686387,
                0.002889855969686387,
                3.674425480545301,
                0.0004944847317711765,
            ),
            "change_code": "HHHH",
            "moment_difference_snr": 4.122212360874777,
        },
        "result": ("1~1;6~7", 0.0029296875, "P#+"),
    },
    "sidelobes_1_6": {
        "bad_atmosphere": False,
        "ranges": ("7~95;172~473", "3~97;168~476"),
        "final_threshold_sigma": 4.2,
        "statistics": (
            10.43522113722134,
            0.0065183016770891476,
            0.0028323483748144314,
            0.0003532223470691299,
            9.37910899904034,
            0.006145259268872658,
            27.69650962976297,
            0.0075113228522241116,
        ),
        "cube_noise": (
            0.000959999577658012,
            0.00011808650378952734,
            4.585864264594227,
            3.4509503661869423,
        ),
        "counts": (52.0, 22.0, 0, 0, 0, 0, 57.0, 0),
        "negative_fraction": 0.0,
        "masked_pixels": None,
        "amend_mask": ("no", 0.0, -1.0),
        "only_extra_mask": ("yes_moment", 0.0047750712836946456, 5.5),
        "second": {
            "kind": "only_extra_mask",
            "threshold_sigma": "auto",
            "threshold_sigma_mode": "auto",
            "signal_ratio_tier_1": 1.0,
            "final_threshold_sigma": 4.2,
            "decision": ("no", 0.0, 5.5),
            "ranges": ("4~260;325~476", "3~260;325~476"),
            "statistics": (
                4.889022540531544,
                0.004476895425920276,
                0.00278370530741106,
                0.0003463248746497146,
                NAN,
                0.004476895425920276,
                26.2010630416968,
                NAN,
            ),
        },
        "change_code": "LLLS",
        "result": ("7~95;172~260;325~473", 0.3193359375, "ED"),
    },
    "imaging": {
        "bad_atmosphere": False,
        "ranges": ("4~82;152~230;442~471", "3~89;151~243;433~476"),
        "final_threshold_sigma": 4.2,
        "statistics": (
            10.687880639813397,
            0.014179529368484585,
            0.005767057361101946,
            0.0007871038506965881,
            5.197931043770518,
            0.009858368901309056,
            67.65960950280821,
            0.017589136958122253,
        ),
        "cube_noise": (
            0.002089466768806404,
            -0.00016056316962931305,
            4.646742243759769,
            1.9580649002349289,
        ),
        "counts": (0, 0, 0, 0, 0, 0, 14.0, 0),
        "negative_fraction": 0.0,
        "masked_pixels": None,
        "amend_mask": ("no", 0.0, 0.0),
        "only_extra_mask": ("yes_moment", 0.01009612853993318, 5.5),
        "second": {
            "kind": "only_extra_mask",
            "threshold_sigma": "auto",
            "threshold_sigma_mode": "auto",
            "signal_ratio_tier_1": 1.0,
            "final_threshold_sigma": 4.2,
            "decision": ("no", 0.0, 5.5),
            "ranges": ("3~145;196~473", "3~145;196~476"),
            "statistics": (
                5.346099981050852,
                0.009871396619539315,
                0.00552161200042336,
                0.000813636975465046,
                NAN,
                0.009871396619539315,
                64.92837703157056,
                NAN,
            ),
        },
        "change_code": "LSLH",
        "result": ("4~82;196~230;442~471", 0.140625, "ED"),
    },
    "sidelobes_2_6": {
        "bad_atmosphere": False,
        "ranges": ("10~95;172~473", "3~96;169~476"),
        "final_threshold_sigma": 4.2,
        "statistics": (
            10.398616106229763,
            0.006506596271370719,
            0.002831769905893972,
            0.00035339571419269684,
            10.398616106229763,
            0.006506596271370719,
            27.680538819105504,
            NAN,
        ),
        "cube_noise": (
            0.0009578317198405295,
            -6.716617644997314e-05,
            4.585864264594227,
            3.8366095936858025,
        ),
        "counts": (55.0, 26.0, 0, 0, 0, 0, 0, 0),
        "negative_fraction": 0.0,
        "masked_pixels": (190, 447),
        "amend_mask": ("yes_moment", 0.004775446333953805, 5.5),
        "second": {
            "kind": "amended_mask",
            "threshold_sigma": "auto",
            "threshold_sigma_mode": "auto",
            "signal_ratio_tier_1": 0.967,
            "final_threshold_sigma": 4.2,
            "decision": ("no", 0.0, 5.5),
            "ranges": ("3~86;154~255;339~472", "3~88;152~269;334~476"),
            "statistics": (
                5.006945984912221,
                0.004521961400353714,
                0.0027673161017360655,
                0.00035044222644003823,
                5.006945984912221,
                0.004521961400353714,
                26.049668814953957,
                NAN,
            ),
        },
        "change_code": "LLLS",
        "result": ("3~86;154~255;339~472", 0.3125, "AD"),
    },
    "sidelobes_3_6": {
        "bad_atmosphere": False,
        "ranges": ("3~92;174~473", "3~94;170~476"),
        "final_threshold_sigma": 4.2,
        "statistics": (
            10.45677318972014,
            0.006511555494751836,
            0.0028322663018478847,
            0.0003518570333457163,
            10.45677318972014,
            0.006511555494751836,
            27.64965936739048,
            NAN,
        ),
        "cube_noise": (
            0.00095866095247762,
            -0.00010501849465072155,
            4.585864264594227,
            3.837946234688061,
        ),
        "counts": (42.0, 22.0, 0, 0, 0, 0, 0, 0),
        "negative_fraction": 0.0,
        "masked_pixels": (1019, 1232),
        "amend_mask": ("yes_moment", 0.004767479985249325, 5.5),
        "second": {
            "kind": "amended_mask",
            "threshold_sigma": 4.2,
            "threshold_sigma_mode": "auto_lower",
            "signal_ratio_tier_1": 0.967,
            "final_threshold_sigma": 4.2,
            "decision": ("no", 0.0, 5.5),
            "ranges": ("3~83;158~265;339~473", "3~85;155~267;334~476"),
            "statistics": (
                4.937679973437442,
                0.004509788374111581,
                0.0027729092942370324,
                0.00035176015643342584,
                4.937679973437442,
                0.004509788374111581,
                26.123474870367897,
                NAN,
            ),
        },
        "change_code": "LLLS",
        "result": ("3~83;158~265;339~473", 0.31640625, "AD"),
    },
}


def _same(value, expected):
    """Equal bit for bit, NaN equal to NaN."""
    return np.array_equal(np.float64(value), np.float64(expected), equal_nan=True)


def _assert_decision(decision, expected):
    """``decision`` equals findContinuum's ``(decision, level, sigma used)``."""
    kind, level, sigma_used = expected
    assert decision.kind == NATIVE_DECISION.get(kind, kind), (decision, expected)
    assert _same(decision.level, level) and _same(decision.sigma_used, sigma_used), (
        decision,
        expected,
    )


def _selection(ranges):
    """findContinuum's selection string of ``[start, stop)`` ranges."""
    return ";".join(f"{start}~{stop - 1}" for start, stop in ranges)


def _ranges(selection):
    """The ``[start, stop)`` ranges of a findContinuum selection string."""
    return [
        [int(first), int(last) + 1]
        for first, last in (pair.split("~") for pair in selection.split(";"))
    ]


def _channel_info(n_channels):
    return [
        n_channels,
        FIRST_FREQUENCY,
        FIRST_FREQUENCY + n_channels * CHANNEL_WIDTH,
        CHANNEL_WIDTH,
    ]


def _warnings(ranges, n_channels):
    return bandwidth_warnings(ranges, n_channels, CHANNEL_WIDTH, CHANNEL_WIDTH)


def _statistics(values):
    """Selection statistics with the given moment-difference values and cube
    peak; the maximum map's statistics, which the rounds do not read, NaN."""
    snr, peak, median, mad, snr_outside, peak_outside, total, cube_peak = values
    return SelectionStatistics(
        *([NAN] * 7),
        moment_difference_snr=snr,
        moment_difference_peak=peak,
        moment_difference_median=median,
        moment_difference_scaled_mad=mad,
        moment_difference_snr_outside=snr_outside,
        moment_difference_peak_outside=peak_outside,
        moment_difference_sum=total,
        cube_peak=cube_peak,
    )


def _change_values(statistics):
    return (
        statistics.moment_difference_peak,
        statistics.moment_difference_peak_outside,
        statistics.moment_difference_sum,
        statistics.moment_difference_scaled_mad,
    )


def _findcontinuum_change_code(values, references, threshold=0.01):
    arguments = [
        np.float64(x) for pair in zip(values, references, strict=True) for x in pair
    ]
    with quiet(), np.errstate(divide="ignore", invalid="ignore"):
        return findContinuum.compute4LetterCode(*arguments, threshold)


def _map(values):
    """An image-shaped map of one time, frequency and polarization."""
    values = np.asarray(values)
    return xr.DataArray(
        values[None, None, None],
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            "time": [0.0],
            "frequency": [FIRST_FREQUENCY],
            "polarization": ["I"],
            "l": np.arange(values.shape[0], dtype=np.float64),
            "m": np.arange(values.shape[1], dtype=np.float64),
        },
    )


def _joint_mask(mask, footprint):
    return xr.Dataset(
        {"JOINT_MASK": _map(mask), "JOINT_MASK_FOOTPRINT": _map(footprint)},
        attrs={"kind": "phase_1", "primary_beam_based": False},
    )


def _round_maps(masked_pixels, amend_level, shape=(40, 40)):
    """A joint mask, maximum map, moment difference and primary-beam region
    with ``masked_pixels`` = (before, after) pixels in the joint mask before
    and after the amend level grows it."""
    before, after = masked_pixels or (0, 0)
    pixel = np.arange(shape[0] * shape[1]).reshape(shape)
    maximum = np.where((pixel >= before) & (pixel < after), amend_level + 1.0, 0.0)
    return (
        _joint_mask(pixel < before, np.ones(shape, dtype=bool)),
        _map(maximum),
        _map(np.zeros(shape)),
        _map(np.ones(shape, dtype=bool)),
    )


def _decision_from(decisions, name):
    """The decision ``name`` of a transition's ``decisions``."""
    return Decision(
        decisions[f"{name}_decision"],
        decisions[f"{name}_level"],
        decisions[f"{name}_sigma_used"],
    )


def _decision_kinds(decisions):
    return {
        key.removesuffix("_decision"): value
        for key, value in decisions.items()
        if key.endswith("_decision")
    }


def test_decisions_equal_findcontinuum():
    """Random cases with values on every threshold (fC 3790-3982)."""
    rng = random.Random(20261114)
    snrs = [7.9, 8.0, 8.1, 10.9, 11.0, 11.4, 11.5, 12.0]
    counts = [0, 6.0, 7.0, 8.0, 9.0, 20.0, 850.0, 851.0]
    for _ in range(N_CASES):
        bad_atmosphere = rng.choice([False, True])
        median, mad = rng.uniform(-1e-3, 3e-3), rng.uniform(1e-4, 1e-3)
        cube_mad = rng.choice([1e-4, 1e-3, 1e9, rng.uniform(1e-4, 2e-3)])
        snr, snr_outside = rng.choice(snrs), rng.choice(snrs)
        snr_cube = rng.choice([7.4, 7.5, 7.6])
        n_above, n_above_bad = rng.choice(counts), rng.choice(counts)
        n_above_2, n_above_2_bad = rng.choice([0, 1.0, 4.0, 22.0, 26.0]), 5.0
        n_cube, n_cube_2 = rng.choice([0, 9.0, 10.0, 20.0]), rng.choice([9, 19, 20, 40])
        n_anywhere, n_anywhere_bad = rng.choice(counts), rng.choice(counts)
        ten_event_sigma = rng.choice([4.4, 5.0, 5.5, 5.73475, NAN])
        fraction = rng.choice([0.0, 0.1, 0.11])
        cube_peak, cube_median = (
            rng.choice([0.0075, 0.008, 0.01]),
            rng.choice([0, 5e-4]),
        )
        with quiet():
            amend = findContinuum.amendMaskYesOrNo(
                bad_atmosphere,
                median,
                snr_outside,
                snr_cube,
                n_above,
                n_above_bad,
                n_cube,
                ten_event_sigma,
                mad,
                cube_mad,
                8.0,
                11.5,
                7.5,
                n_cube_2,
                fraction,
                n_above_2,
                n_above_2_bad,
                False,
                850,
            )
            only = findContinuum.onlyExtraMaskYesOrNo(
                bad_atmosphere,
                median,
                snr,
                snr_cube,
                n_anywhere,
                n_anywhere_bad,
                0,
                ten_event_sigma,
                mad,
                cube_mad,
                cube_peak,
                cube_median,
                8.0,
                11.0,
                7.5,
                7.5,
                7,
                False,
            )
            extra = findContinuum.extraMaskYesOrNo(
                bad_atmosphere,
                median,
                snr,
                snr_cube,
                0,
                0,
                0,
                ten_event_sigma,
                mad,
                cube_mad,
                8.0,
                11.5,
                7.5,
            )
        _assert_decision(
            amend_mask_decision(
                bad_atmosphere,
                median,
                snr_outside,
                snr_cube,
                n_above,
                n_above_2,
                n_above_bad,
                n_cube,
                n_cube_2,
                mad,
                ten_event_sigma,
                cube_mad,
                fraction,
            ),
            amend,
        )
        _assert_decision(
            only_extra_mask_decision(
                bad_atmosphere,
                median,
                snr,
                mad,
                n_anywhere,
                n_anywhere_bad,
                cube_peak,
                cube_median,
                cube_mad,
                ten_event_sigma,
            ),
            only,
        )
        _assert_decision(
            extra_mask_decision(bad_atmosphere, median, snr, mad, ten_event_sigma),
            extra,
        )


def test_change_code_equals_findcontinuum():
    """Random cases with zero references, ties and both thresholds
    (fC 3630-3687); Python floats with a zero reference do not raise."""
    rng = random.Random(2)
    base = [0.0, 1e-9, 0.004, 0.00404, 0.00396, 0.0041, 0.00408, 27.7, -0.002, NAN]
    for _ in range(N_CASES):
        values = tuple(rng.choice(base) for _ in range(4))
        references = tuple(rng.choice(base) for _ in range(4))
        threshold = rng.choice([0.01, 0.05])
        assert change_code(values, references, threshold) == (
            _findcontinuum_change_code(values, references, threshold)
        )
    assert change_code((1.0, 0.0, -1.0, 0.0), (0.0, 0.0, 0.0, 0.0)) == "HSLS"


def _extra_round_case(rng):
    """Channels, reference ranges, a round's ranges and an extra round kind;
    one in seven keeps the reference, one in seven intersects in one channel."""
    n_channels = rng.choice([8, 40, 120, 480, 501, 600, 1000])
    reference = random_ranges(rng, n_channels)
    if rng.random() < 0.3:
        rng.shuffle(reference)
    draw = rng.random()
    if draw < 1 / 7:
        ranges = [[0, n_channels]]
    elif draw < 2 / 7:
        channel = int(rng.choice(list(channels_of_ranges(reference))))
        ranges = [[channel, channel + 1]]
    else:
        ranges = random_ranges(rng, n_channels)
    return n_channels, reference, ranges, rng.choice(EXTRA_ROUND_KINDS)


def test_round_maps_ranges_equals_findcontinuum():
    """The intersection with the reference, its emptiness, no improvement and
    the widening of one channel (fC 2548-2550, 2594-2612) on random cases."""
    rng = random.Random(3)
    for _ in range(N_CASES):
        n_channels, reference, ranges, round_kind = _extra_round_case(rng)
        maps_ranges = round_maps_ranges(round_kind, ranges, reference, n_channels)
        with quiet():
            found = INTERSECTION(_selection(reference), _selection(ranges))
            if found["channelList"].size == 0:
                assert maps_ranges == MapsRanges(ranges, True, False, False)
                continue
            expected = INTERSECTED(
                found["channelList"],
                found["channelList0"],
                FINDCONTINUUM_ROUND[round_kind],
                n_channels,
                False,
                False,
                False,
            )
        assert maps_ranges.ranges == _ranges(expected["selection"])
        assert maps_ranges.ranges == _ranges(expected["selectionPreBluePruning"])
        assert not maps_ranges.empty_intersection
        assert maps_ranges.no_improvement == (
            list(found["channelList"]) == list(found["channelList0"])
        )
        assert maps_ranges.widened == ("onlyChannel" in expected)
    for round_kind in ("original", "single", "amended_mask"):
        assert round_maps_ranges(round_kind, [[3, 9]], None, 40) == MapsRanges(
            [[3, 9]], False, False, False
        )
    with pytest.raises(ValueError, match="round kind"):
        round_maps_ranges("amended", [[3, 9]], [[3, 9]], 40)


def test_after_extra_round_equals_findcontinuum():
    """Result 6, no improvement, the auto-lower decisions, the forced Y and
    result 7 (fC 2548-2612, 2683-2742) on random cases."""
    rng = random.Random(4)
    for _ in range(N_CASES):
        n_channels, reference, ranges, round_kind = _extra_round_case(rng)
        n_rounds_after_original = rng.choice([4, 4, 4, 1, 2, 3])
        bad_atmosphere = rng.choice([False, True])
        statistics = _statistics(
            (
                rng.choice([7.9, 8.0, 11.5, 14.9, 15.0, 30.0]),
                rng.uniform(4e-3, 8e-3),
                rng.uniform(1e-3, 3e-3),
                rng.uniform(1e-4, 5e-4),
                NAN,
                rng.uniform(4e-3, 8e-3),
                rng.uniform(20.0, 30.0),
                NAN,
            )
        )
        references = (
            rng.uniform(4e-3, 8e-3),
            rng.uniform(4e-3, 8e-3),
            rng.uniform(20.0, 30.0),
            rng.uniform(1e-4, 5e-4),
        )
        final_threshold_sigma = rng.choice([4.2, 4.5, 2.5, 3.6, 7.2])
        threshold_sigma_mode = rng.choice(["auto", "auto_lower"])
        ten_event_sigma = rng.choice([4.4, 5.73475])
        ranges_before_blue_pruning = random_ranges(rng, n_channels)
        aggregate_bandwidth = selection_bandwidth(ranges, CHANNEL_WIDTH)
        maps_ranges = round_maps_ranges(round_kind, ranges, reference, n_channels)
        with quiet():
            found = INTERSECTION(_selection(reference), _selection(ranges))
            empty = found["channelList"].size == 0
            if empty:
                expected = REFERENCE_RANGES(found["channelList0"])
            else:
                intersected = INTERSECTED(
                    found["channelList"],
                    found["channelList0"],
                    FINDCONTINUUM_ROUND[round_kind],
                    n_channels,
                    False,
                    False,
                    False,
                )
                state = dict(
                    amendMaskIterationName=FINDCONTINUUM_ROUND[round_kind],
                    amendMaskIterations=n_rounds_after_original,
                    amendMaskIteration=1,
                    amendMaskIterationNames=[".original"]
                    + [""] * n_rounds_after_original,
                    badAtmosphere=bad_atmosphere,
                    momDiffMedian=statistics.moment_difference_median,
                    momDiffSNR=statistics.moment_difference_snr,
                    momDiffSNRCube=0.0,
                    NpixMomDiff=0,
                    NpixMomDiffBadAtm=0,
                    NpixCubeDiffMedian=0,
                    TenEventSigma=ten_event_sigma,
                    momDiffMAD=statistics.moment_difference_scaled_mad,
                    MADCubeOutside=1e-3,
                    momDiffLevelForProceeding=8.0,
                    momDiffLevelBadAtm=11.5,
                    cubeLevel=7.5,
                    keepIntermediatePngs=True,
                    png="round.png",
                    signalRatioTier1=1.0,
                    sigmaFindContinuumMode={"auto": "auto", "auto_lower": "autolower"}[
                        threshold_sigma_mode
                    ],
                    sigmaFindContinuum=final_threshold_sigma,
                    finalSigmaFindContinuum=final_threshold_sigma,
                    selection=intersected["selection"],
                    selection0=_selection(reference),
                )
                expected = AUTO_LOWER(
                    **state,
                    chanInfo=_channel_info(n_channels),
                    smallBandwidthFraction=0.05,
                    smallSpreadFraction=0.33,
                    momDiffApproachLevel=15.0,
                    autoLowerDecision=intersected["autoLowerDecision"],
                )
                if "warnings" not in expected:
                    state["amendMaskIterationNames"] = [".original"] + [
                        ""
                    ] * n_rounds_after_original
                    expected = {
                        "autoLowerDecision": intersected["autoLowerDecision"],
                        **AUTO_LOWER_2(
                            **state,
                            autoLower2Decision=intersected["autoLower2Decision"],
                        ),
                    }
        warnings_found = expected.get("warnings", ["", ""])
        transition = after_extra_round(
            round_kind,
            maps_ranges,
            statistics,
            final_threshold_sigma,
            BandwidthWarnings(
                warnings_found[0] != "", warnings_found[1] != "", 1.0, 1.0
            ),
            bad_atmosphere,
            n_rounds_after_original,
            ten_event_sigma,
            references,
            ranges,
            ranges_before_blue_pruning,
            aggregate_bandwidth,
            reference,
            CHANNEL_WIDTH,
            threshold_sigma_mode,
        )
        code = _findcontinuum_change_code(_change_values(statistics), references)
        assert transition.round_record["empty_intersection"] == empty
        if empty:
            assert transition.result == "result_6"
            assert transition.continuum_ranges == _ranges(expected["selection"])
            assert transition.ranges_before_blue_pruning == ranges_before_blue_pruning
            assert transition.aggregate_bandwidth == aggregate_bandwidth
            assert transition.change_code == code
            assert transition.decisions == {}
            continue
        expected_kinds = {
            name: NATIVE_DECISION[value]
            for name, value in (
                ("extra_mask", intersected["extraMaskDecision"]),
                ("auto_lower", expected["autoLowerDecision"]),
                (
                    "auto_lower_2",
                    expected.get(
                        "autoLower2Decision", intersected["autoLower2Decision"]
                    ),
                ),
            )
            if value is not False
        }
        assert _decision_kinds(transition.decisions) == expected_kinds
        next_round = (expected["amendMaskIterationNames"] + [""])[2]
        if next_round:
            assert transition.result is None
            assert transition.next_round_kind == NATIVE_ROUND[next_round]
            assert _same(transition.threshold_sigma, expected["sigmaFindContinuum"])
            assert transition.threshold_sigma_mode == "auto_lower"
            assert expected["sigmaFindContinuumMode"] == "autolower"
            assert transition.selection_reference == _ranges(expected["selection0"])
            assert transition.signal_ratio_tier_1 == expected["signalRatioTier1"]
            assert transition.change_code is None
        else:
            assert transition.result == "result_7"
            assert transition.next_round_kind is None
            assert transition.change_code == code
            with quiet():
                assert transition.aggregate_bandwidth * 1e-9 == (
                    findContinuum.computeBandwidth(
                        intersected["selection"], CHANNEL_WIDTH, 0
                    )
                )
        assert transition.continuum_ranges == _ranges(intersected["selection"])
        assert transition.ranges_before_blue_pruning == _ranges(
            intersected["selectionPreBluePruning"]
        )
    with pytest.raises(ValueError, match="extra-mask round"):
        after_extra_round(
            "amended_mask",
            MapsRanges([[0, 4]], False, False, False),
            _statistics((8.0, 1.0, 0.0, 0.1, NAN, 1.0, 1.0, NAN)),
            4.2,
            BandwidthWarnings(False, False, 1.0, 1.0),
            False,
            4,
            5.0,
            (1.0, 1.0, 1.0, 0.1),
            [[0, 4]],
            [[0, 4]],
            1.0,
            [[0, 4]],
            CHANNEL_WIDTH,
            "auto",
        )


def test_auto_lower_rounds_of_large_128():
    """CASA's only auto-lower run (large_128): sigma 4.50 lowered to 6/7 below
    a signal-to-noise ratio of 15, again in round auto_lower_2, whose
    unchanged channels end with the forced Y: path code EDxY."""
    statistics = _statistics((9.356687, 0.008, 0.002, 4e-4, NAN, 0.007, 25.0, NAN))
    references = (0.009, 0.008, 26.0, 4e-4)
    ranges = [[5, 26], [46, 92], [118, 123]]
    common = (False, 4, 5.73475, references)
    lowered = after_extra_round(
        "only_extra_mask",
        MapsRanges(ranges, False, False, False),
        statistics,
        4.5,
        BandwidthWarnings(False, False, 0.5, 0.9),
        *common,
        [[3, 30], [40, 125]],
        [[3, 30], [40, 125]],
        1.0,
        [[5, 30], [46, 123]],
        CHANNEL_WIDTH,
        "auto",
    )
    assert lowered.next_round_kind == "auto_lower"
    assert lowered.threshold_sigma == 3.857142857142857
    assert (lowered.threshold_sigma_mode, lowered.signal_ratio_tier_1) == (
        "auto_lower",
        1.0,
    )
    assert lowered.selection_reference == ranges
    unchanged = MapsRanges(ranges, False, True, False)
    second = after_extra_round(
        "auto_lower",
        unchanged,
        statistics,
        lowered.threshold_sigma,
        BandwidthWarnings(False, False, 0.5, 0.9),
        *common,
        ranges,
        ranges,
        1.0,
        ranges,
        CHANNEL_WIDTH,
        "auto_lower",
    )
    assert second.next_round_kind == "auto_lower_2"
    assert second.threshold_sigma == 3.306122448979592
    last = after_extra_round(
        "auto_lower_2",
        unchanged,
        statistics,
        second.threshold_sigma,
        BandwidthWarnings(False, False, 0.5, 0.9),
        *common,
        ranges,
        ranges,
        1.0,
        ranges,
        CHANNEL_WIDTH,
        "auto_lower",
    )
    assert last.result == "result_7"
    assert last.decisions["auto_lower_2_decision"] == "yes_moment"
    assert last.aggregate_bandwidth * 1e-9 == 0.0703125
    original = {
        "amend_mask_decision": "no",
        "amend_mask_sigma_used": 0.0,
        "only_extra_mask_decision": "yes_moment",
    }
    decisions = [original, lowered.decisions, second.decisions, last.decisions]
    assert path_code(decisions, True) == "EDxY"
    assert path_code(decisions, False) == ""


def _low_spread_case(rng):
    """Ranges of a small spread, a noisy spectrum with lines, baseline channels
    and thresholds around their median (segment_6_check.py)."""
    n_channels = rng.randint(8, 520)
    spread = rng.randint(1, max(1, int(0.33 * n_channels) - 1))
    start = rng.randint(0, n_channels - spread)
    inner = set(
        rng.sample(range(start, start + spread), min(spread, rng.randint(1, 4)))
    )
    ranges = ranges_of_channels(np.array(sorted(inner | {start, start + spread - 1})))
    spectrum = np.array([rng.gauss(0.0, 1e-3) for _ in range(n_channels)])
    for _ in range(rng.randint(0, 4)):
        centre = rng.randint(0, n_channels - 1)
        spectrum[max(0, centre - 3) : centre + 4] += rng.uniform(0.002, 0.02)
    n_baseline = max(2, int(round(0.19 * n_channels)))
    if rng.random() < 0.7:
        baseline = np.argsort(spectrum)[:n_baseline]
    else:
        baseline = np.array(rng.sample(range(n_channels), n_baseline))
    median = float(np.median(spectrum[baseline]))
    scaled_mad = scaled_median_absolute_deviation(spectrum[baseline])
    positive = median + rng.uniform(0.5, 4) * scaled_mad
    negative = median - rng.uniform(0.5, 4) * scaled_mad
    return ranges, n_channels, baseline, spectrum, positive, negative


def _low_spread_oracle(ranges, n_channels, baseline, spectrum, positive, negative):
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return LOW_SPREAD(
            _selection(ranges),
            n_channels,
            0.33,
            [baseline, spectrum[baseline]],
            spectrum,
            negative,
            positive,
            ";",
            "",
        )


def _assert_low_spread(ranges, n_channels, baseline, spectrum, positive, negative):
    expected = _low_spread_oracle(
        ranges, n_channels, baseline, spectrum, positive, negative
    )
    got = low_spread_ranges(ranges, n_channels, baseline, spectrum, positive, negative)
    if got is None:
        assert expected["selection"] == _selection(ranges)
    else:
        assert _selection(got) == expected["selection"]
    return expected, got


def test_low_spread_ranges_equal_findcontinuum():
    """Random spectra through the low-spread fix (fC 2856-2920)."""
    rng = random.Random(5)
    for _ in range(N_CASES):
        _assert_low_spread(*_low_spread_case(rng))


def test_low_spread_ranges_branches():
    """The low side, the comparable-intensity scan, prepended runs in
    decreasing order, the retry after a dropped run and the cases that add
    nothing, each equal to findContinuum."""
    n_channels = 100
    channel = np.arange(n_channels)
    spectrum = 1e-3 * (((channel * 37) % 11) - 5) / 5.0
    # Low side, widest baseline run [10, 16): prepended.
    expected, got = _assert_low_spread(
        [[60, 70]],
        n_channels,
        np.array([12, 10, 11, 13, 14, 15, 80, 81]),
        spectrum,
        1e-2,
        -1e-2,
    )
    assert got == [[10, 16], [60, 70]]
    # No baseline run on the low side: the scan finds runs between spikes, each
    # prepended, so that they end in decreasing order.
    spiked = spectrum.copy()
    spiked[[10, 20, 30]] = 1.0
    expected, got = _assert_low_spread(
        [[70, 80]], n_channels, np.array([72, 73, 74]), spiked, 1e-2, -1e-2
    )
    assert expected["triedComparableIntensity"]
    assert got == [[31, 45], [21, 30], [11, 20], [0, 10], [70, 80]]
    # High side scan, appended.
    expected, got = _assert_low_spread(
        [[5, 15]], n_channels, np.array([5, 6, 7, 8]), spectrum, 1e-2, -1e-2
    )
    assert got[0] == [5, 15] and len(got) > 1
    # A run whose median lies more than a scaled MAD below the negative threshold
    # is dropped; the retry keeps the longest run of channels between the
    # thresholds widened by that excess.
    absorbed = spectrum.copy()
    absorbed[50:53] = -5e-3
    expected, got = _assert_low_spread(
        [[5, 15]],
        n_channels,
        np.array([50, 51, 52, 7, 9, 3, 1, 12, 14, 30, 33, 36]),
        absorbed,
        1e-3,
        -1e-3,
    )
    assert not expected["triedComparableIntensity"]
    assert expected["medianNegativeSigma"] > 1
    assert got == [[5, 15], [53, 100]]
    # Nothing comparable in the window: nothing added.
    flat = np.zeros(n_channels)
    flat[40:] = 1.0
    assert (
        _assert_low_spread(
            [[5, 15]], n_channels, np.array([5, 6, 7, 8]), flat, 1e-2, -1e-2
        )[1]
        is None
    )


def test_path_code_equals_findcontinuum():
    """findContinuum's decision code (fC 3424-3482) for every combination of
    decisions, the intersection flag, sigma used and an added selection."""
    amends = ["not_evaluated", "no", "yes_moment", "yes_cube"]
    kinds = ["not_evaluated", "no", "yes_moment", "yes_cube", "no_improvement"]
    others = [(sigma, added) for sigma in (-2.0, -1.0, 0.0, 5.5) for added in (0, 1)]
    for index, (
        amend,
        extra,
        extra_2,
        auto_lower,
        auto_lower_2,
        intersection,
    ) in enumerate(
        (a, e, e2, x, y, i)
        for a in amends
        for e in kinds
        for e2 in kinds
        for x in kinds
        for y in kinds
        for i in (False, True)
    ):
        sigma_used, added = others[index % len(others)]
        with quiet():
            expected = DECISION_CODE(
                FINDCONTINUUM_DECISION[amend],
                FINDCONTINUUM_DECISION[extra],
                FINDCONTINUUM_DECISION[extra_2],
                FINDCONTINUUM_DECISION[auto_lower],
                intersection,
                True,
                sigma_used,
                bool(added),
                FINDCONTINUUM_DECISION[auto_lower_2],
                4,
            )["decisionCode"]
        only = amend in ("not_evaluated", "no") and index % 2
        decisions = [
            {
                "amend_mask_decision": amend,
                "amend_mask_sigma_used": sigma_used,
                f"{'only_extra_mask' if only else 'extra_mask'}_decision": extra,
            },
            {
                "extra_mask_2_decision": extra_2,
                "auto_lower_decision": auto_lower,
                "auto_lower_2_decision": auto_lower_2,
                "intersection_of_selections": intersection,
            },
        ]
        if added:
            decisions.append({"added_selection": True})
        assert path_code(decisions, True) == expected.split(",")[1].strip(), (
            amend,
            extra,
            extra_2,
            auto_lower,
            auto_lower_2,
            intersection,
            sigma_used,
            added,
        )


def test_after_rounds_and_low_spread_reversion_equal_findcontinuum():
    """The reversion test and the low-spread condition (fC 2834-2854), and the
    reversion after the low-spread fix (fC 3040-3053), on random cases."""
    rng = random.Random(7)
    codes = ["HHHH", "HHHS", "HHSH", "SSSS", "LLLS", None]
    for _ in range(N_CASES):
        n_channels = rng.choice([8, 40, 480])
        ranges = random_ranges(rng, n_channels, max_ranges=3)
        code = rng.choice(codes)
        snr = rng.choice([7.5, 8.0, 8.000000000000002, 11.0, NAN])
        level = rng.choice([7.5, 8.0])
        with quiet():
            expected = REVERSION(
                True,
                code,
                None,
                snr,
                level,
                _selection(ranges),
                _channel_info(n_channels),
                0.05,
                0.33,
            )
        found = _warnings(ranges, n_channels)
        assert after_rounds(code, snr, level, found) == (
            expected["revert"],
            not expected["revert"] and expected["warnings"][1] != "",
        )
        low_bandwidth, low_spread = rng.random() < 0.5, rng.random() < 0.5
        warnings_strings = ["LowBW" if low_bandwidth else "", "spread" * low_spread]
        with quiet():
            revert = False
            if code is not None:
                revert = LOW_SPREAD_CODE_REVERSION(
                    code, None, ["HHHH", "HHHS"], False, snr, level
                )["revert"]
            revert = WARNINGS_REVERSION(warnings_strings, revert, [""])["revert"]
        assert (
            after_low_spread_fix(
                code, snr, level, BandwidthWarnings(low_bandwidth, low_spread, 0.5, 0.5)
            )
            == revert
        )


def test_amended_joint_mask_equals_port():
    """The amended joint mask equals the port's evaluation of findContinuum's
    expression, its level written with %f (fC 2096-2099), at ties too."""
    rng = np.random.default_rng(8)
    for level in [0.004775446333953805, 0.0047674799852493, 0.0010000005, 0.0, -3e-4]:
        joint = rng.random((30, 40)) < 0.2
        footprint = rng.random((30, 40)) < 0.9
        maximum = rng.normal(0.003, 0.002, (30, 40))
        maximum[rng.random((30, 40)) < 0.1] = NAN
        written = round_to_decimals(level, 6)
        maximum[0, :4] = [
            written,
            np.nextafter(written, 1),
            level,
            np.nextafter(written, -1),
        ]
        port = evaluate_lattice_expression(
            "iif((IM0==1)||(IM1>%f), 1, 0)" % level,  # noqa: UP031
            {"IM0": np.where(footprint, joint.astype(np.float64), NAN), "IM1": maximum},
        )
        amended = amended_joint_mask(
            _joint_mask(joint, footprint), _map(maximum), level
        )
        assert np.array_equal(amended["JOINT_MASK"].values[0, 0, 0], port == 1)
        assert amended["JOINT_MASK_FOOTPRINT"].values.all() and np.isfinite(port).all()
        assert amended.attrs == {"kind": "amended", "primary_beam_based": False}
        assert amended["JOINT_MASK"].dims == (
            "time",
            "frequency",
            "polarization",
            "l",
            "m",
        )
        float32 = amended_joint_mask(
            _joint_mask(joint, footprint), _map(maximum.astype(np.float32)), level
        )
        assert np.array_equal(
            float32["JOINT_MASK"].values[0, 0, 0],
            (joint & footprint)
            | (maximum.astype(np.float32).astype(np.float64) > written),
        )
    two_planes = xr.concat([_map(maximum)] * 2, dim="polarization")
    with pytest.raises(ValueError, match="one time and one polarization"):
        amended_joint_mask(_joint_mask(joint, footprint), two_planes, 0.004)


@pytest.mark.parametrize("case", list(PORT_ROUNDS))
def test_rounds_replay_port(case):
    """The decisions, transitions, change and path codes, final ranges and
    aggregate bandwidth of the port's rounds of seven cubes."""
    port = PORT_ROUNDS[case]
    n_channels = 8 if case == "eight_channels" else 480
    bad_atmosphere = port["bad_atmosphere"]
    ranges, before_blue = (_ranges(selection) for selection in port["ranges"])
    statistics = _statistics(port["statistics"])
    cube_noise = CubeNoiseStatistics(*port["cube_noise"])
    joint_mask, maximum, moment_difference, region = _round_maps(
        port["masked_pixels"], port["amend_mask"][1]
    )
    transition = after_original_round(
        bad_atmosphere,
        statistics,
        cube_noise,
        MomentDifferenceCounts(*port["counts"]),
        port["negative_fraction"],
        _warnings(ranges, n_channels).low_bandwidth,
        ranges,
        before_blue,
        selection_bandwidth(ranges, CHANNEL_WIDTH),
        port["final_threshold_sigma"],
        "auto",
        "auto",
        joint_mask,
        maximum,
        moment_difference,
        region,
    )
    decisions = [transition.decisions]
    assert transition.round_record["amend_mask_decision"] == port["amend_mask"][0]
    _assert_decision(
        _decision_from(transition.decisions, "amend_mask"), port["amend_mask"]
    )
    if "only_extra_mask" in port:
        _assert_decision(
            _decision_from(transition.decisions, "only_extra_mask"),
            port["only_extra_mask"],
        )
    second = port.get("second")
    if second is None:
        assert (transition.result, transition.change_code) == (
            "result_0",
            port["change_code"],
        )
        assert transition.change_code_references is None
    else:
        assert transition.result is None
        assert transition.next_round_kind == second["kind"]
        assert transition.threshold_sigma == second["threshold_sigma"]
        assert transition.threshold_sigma_mode == second["threshold_sigma_mode"]
        assert transition.signal_ratio_tier_1 == second["signal_ratio_tier_1"]
        assert transition.selection_reference == ranges
        assert transition.change_code_references == _change_values(statistics)
        own, own_before_blue = (_ranges(selection) for selection in second["ranges"])
        statistics_2 = _statistics(second["statistics"])
        if second["kind"] == "amended_mask":
            assert (
                transition.round_record["n_masked_before"] == port["masked_pixels"][0]
            )
            assert transition.round_record["n_masked_after"] == port["masked_pixels"][1]
            assert transition.amended_joint_mask.attrs["kind"] == "amended"
            transition = after_amended_round(
                bad_atmosphere,
                statistics_2,
                cube_noise.ten_event_sigma,
                _warnings(own, n_channels).low_bandwidth,
                0.0,
                own,
                own_before_blue,
                selection_bandwidth(own, CHANNEL_WIDTH),
                transition.selection_reference,
                transition.change_code_references,
                moment_difference,
                n_channels,
                transition.threshold_sigma,
                transition.threshold_sigma_mode,
            )
            _assert_decision(
                _decision_from(transition.decisions, "extra_mask"), second["decision"]
            )
            assert transition.result == "result_1"
        else:
            request = transition.extra_mask_request
            assert request.form == "only_extra_mask"
            assert request.level == port["only_extra_mask"][1]
            assert request.moment_difference is moment_difference
            assert request.continuum_ranges == ranges
            maps_ranges = round_maps_ranges(
                "only_extra_mask", own, transition.selection_reference, n_channels
            )
            transition = after_extra_round(
                "only_extra_mask",
                maps_ranges,
                statistics_2,
                second["final_threshold_sigma"],
                _warnings(maps_ranges.ranges, n_channels),
                bad_atmosphere,
                4,
                cube_noise.ten_event_sigma,
                transition.change_code_references,
                own,
                own_before_blue,
                selection_bandwidth(own, CHANNEL_WIDTH),
                transition.selection_reference,
                CHANNEL_WIDTH,
                transition.threshold_sigma_mode,
            )
            _assert_decision(
                _decision_from(transition.decisions, "auto_lower"), second["decision"]
            )
            assert transition.result == "result_7"
        decisions.append(transition.decisions)
        assert transition.change_code == port["change_code"]
    final_ranges = transition.continuum_ranges
    aggregate_bandwidth = transition.aggregate_bandwidth
    final = after_rounds(
        transition.change_code,
        _statistics((second or port)["statistics"]).moment_difference_snr,
        8.0,
        _warnings(final_ranges, n_channels),
    )
    assert not final.revert
    assert final.run_low_spread_fix == ("low_spread" in port)
    if final.run_low_spread_fix:
        fix = port["low_spread"]
        final_ranges = low_spread_ranges(
            final_ranges,
            n_channels,
            np.array(fix["baseline_channels"]),
            np.array(fix["spectrum"]),
            fix["positive_threshold"],
            fix["negative_threshold"],
        )
        aggregate_bandwidth = selection_bandwidth(final_ranges, CHANNEL_WIDTH)
        references = transition.change_code_references or _change_values(statistics)
        low_spread_code = change_code(fix["values"], references, 0.05)
        assert low_spread_code == fix["change_code"]
        assert not after_low_spread_fix(
            low_spread_code,
            fix["moment_difference_snr"],
            8.0,
            _warnings(final_ranges, n_channels),
        )
        decisions.append({"added_selection": True})
    selection, aggregate_bandwidth_ghz, expected_path_code = port["result"]
    assert _selection(final_ranges) == selection
    assert aggregate_bandwidth * 1e-9 == aggregate_bandwidth_ghz
    assert path_code(decisions, True) == expected_path_code


def test_after_original_round_guards():
    """A low bandwidth makes a yes of the only-extra-mask decision
    "not_evaluated" (fC 1985-1991); a joint mask without pixels in the
    primary beam keeps the threshold sigma (fC 2113); maps of two planes are
    refused."""
    port = PORT_ROUNDS["imaging"]
    ranges, before_blue = (_ranges(selection) for selection in port["ranges"])
    statistics = _statistics(port["statistics"])
    arguments = (
        False,
        statistics,
        CubeNoiseStatistics(*port["cube_noise"]),
        MomentDifferenceCounts(*port["counts"]),
        0.0,
    )
    maps = _round_maps(None, 0.0)
    guarded = after_original_round(
        *arguments,
        True,
        ranges,
        before_blue,
        1.0,
        4.2,
        "auto",
        "auto",
        *maps,
    )
    assert guarded.result == "result_0"
    assert guarded.decisions["only_extra_mask_decision"] == "not_evaluated"
    assert guarded.decisions["only_extra_mask_level"] == port["only_extra_mask"][1]
    assert guarded.extra_mask_request is None and guarded.change_code == "SSSS"
    assert path_code([guarded.decisions], True) == "S"
    amend = PORT_ROUNDS["sidelobes_3_6"]
    empty_mask = list(_round_maps((0, 1232), amend["amend_mask"][1]))
    transition = after_original_round(
        False,
        _statistics(amend["statistics"]),
        CubeNoiseStatistics(*amend["cube_noise"]),
        MomentDifferenceCounts(*amend["counts"]),
        0.0,
        False,
        ranges,
        before_blue,
        1.0,
        3.6,
        "auto",
        "auto",
        *empty_mask,
    )
    assert transition.round_record["n_masked_before"] == 0
    assert (transition.threshold_sigma, transition.threshold_sigma_mode) == (
        "auto",
        "auto",
    )
    assert transition.amended_joint_mask["JOINT_MASK"].values.sum() == 1232
    empty_mask[2] = xr.concat([empty_mask[2]] * 2, dim="polarization")
    with pytest.raises(ValueError, match="one time and one polarization"):
        after_original_round(
            *arguments,
            False,
            ranges,
            before_blue,
            1.0,
            4.2,
            "auto",
            "auto",
            *empty_mask,
        )


def test_after_amended_round_branches():
    """Result 2 with stale values, the intersection's maps request, result 3,
    the extra mask with and without intersection, the bad-atmosphere level
    and the "not_evaluated" that continues (fC 2155-2542)."""
    references = (0.0065, 0.0065, 27.7, 3.5e-4)
    original = [[10, 96], [172, 474]]
    amended = [[3, 87], [154, 256], [339, 473]]
    yes = _statistics((9.0, 0.0045, 0.0028, 3.5e-4, 9.0, 0.0045, 26.0, NAN))
    no = _statistics((5.0, 0.0045, 0.0028, 3.5e-4, 5.0, 0.0045, 26.0, NAN))
    difference = _map(np.zeros((4, 5)))

    def amended_round(statistics, difference_snr, ranges, bad_atmosphere=False):
        return after_amended_round(
            bad_atmosphere,
            statistics,
            4.585864264594227,
            False,
            difference_snr,
            ranges,
            [[3, 477]],
            123.0,
            original,
            references,
            difference,
            480,
            "auto",
            "auto",
        )

    level = 8.5 * np.sqrt(2)
    disjoint = amended_round(yes, np.nextafter(level, 20), [[100, 170]])
    assert disjoint.result == "result_2"
    assert disjoint.continuum_ranges == original
    assert disjoint.ranges_before_blue_pruning == [[3, 477]]
    assert disjoint.aggregate_bandwidth == 123.0
    assert disjoint.change_code == change_code(_change_values(yes), references)
    request = amended_round(yes, 13.0, amended)
    expected_intersection = ranges_of_channels(
        np.intersect1d(channels_of_ranges(original), channels_of_ranges(amended))
    )
    assert request.result is None and request.next_round_kind is None
    assert request.maps_request == expected_intersection
    assert request.intersection_of_selections
    assert request.decisions["intersection_of_selections"]
    result_3 = after_amended_intersection(
        False,
        no,
        4.585864264594227,
        request.maps_request,
        CHANNEL_WIDTH,
        references,
        difference,
        "auto",
        "auto",
    )
    assert result_3.result == "result_3"
    assert result_3.continuum_ranges == expected_intersection
    assert result_3.ranges_before_blue_pruning == expected_intersection
    assert result_3.aggregate_bandwidth == selection_bandwidth(
        expected_intersection, CHANNEL_WIDTH
    )
    assert result_3.change_code == change_code(_change_values(no), references)
    assert (
        path_code(
            [
                {"amend_mask_decision": "yes_moment"},
                request.decisions,
                result_3.decisions,
            ],
            True,
        )
        == "ADI"
    )
    with_intersection = after_amended_intersection(
        False,
        yes,
        4.585864264594227,
        request.maps_request,
        CHANNEL_WIDTH,
        references,
        difference,
        "auto",
        "auto",
    )
    assert with_intersection.next_round_kind == "extra_mask"
    assert with_intersection.selection_reference == expected_intersection
    assert with_intersection.signal_ratio_tier_1 == 1.0
    assert with_intersection.extra_mask_request.form == "with_intersection"
    assert with_intersection.extra_mask_request.level == (
        extra_mask_decision(False, 0.0028, 9.0, 3.5e-4, 4.585864264594227).level
    )
    assert (
        path_code(
            [
                {"amend_mask_decision": "yes_moment"},
                request.decisions,
                with_intersection.decisions,
            ],
            True,
        )
        == "ADIE"
    )
    for snr, bad_atmosphere in ((level, False), (13.0, True)):
        without = amended_round(
            _statistics((12.0, 0.0045, 0.0028, 3.5e-4, 12.0, 0.0045, 26.0, NAN)),
            snr,
            amended,
            bad_atmosphere,
        )
        assert without.next_round_kind == "extra_mask"
        assert without.extra_mask_request.form == "without_intersection"
        assert without.selection_reference == amended
        assert without.round_record["difference_snr"] == snr
    guarded = after_amended_round(
        False,
        no,
        4.585864264594227,
        True,
        1.0,
        amended,
        [[3, 477]],
        123.0,
        original,
        references,
        difference,
        480,
        4.2,
        "auto_lower",
    )
    assert guarded.decisions["extra_mask_decision"] == "not_evaluated"
    assert guarded.next_round_kind == "extra_mask"
    assert guarded.extra_mask_request.level == 0.0
    assert (guarded.threshold_sigma, guarded.threshold_sigma_mode) == (
        4.2,
        "auto_lower",
    )
    with pytest.raises(ValueError, match="one time and one polarization"):
        after_amended_intersection(
            False,
            no,
            5.0,
            amended,
            CHANNEL_WIDTH,
            references,
            xr.concat([difference] * 2, dim="time"),
            "auto",
            "auto",
        )
