"""Tests of the common beam and of the correcting beam."""

import importlib
import math
import re

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.common_beam import (
    _beam_covariance,
    _beam_from_covariance,
    _correcting_covariance,
    _eigenvalues,
    common_beam,
    correcting_beam,
)

# The module itself (the package re-exports the function under its name).
common_beam_module = importlib.import_module(
    "astroviper.processing_functions.image_analysis.common_beam"
)

# casacore's C::arcsec and C::degree: the radians casacore makes of arcsec and degrees.
ARCSEC = 4.84813681109535993589914098765e-6
DEGREE = 0.0174532925199432957692369

# Beam sets (major", minor", pa deg) of one polarization, in channel order.
CASA_SETS = {
    "all equal": [(1.0, 0.8, 30.0)] * 3,
    "largest contains all": [(1.0, 0.8, 30.0), (1.1, 0.88, 30.0), (1.2, 0.95, 32.0)],
    "two PAs": [(3.0, 1.0, 20.0), (2.8, 1.1, 60.0)],
    "mirrored PA": [(3.0, 1.0, 80.0), (2.9, 1.2, -80.0)],
    "right angle": [(3.0, 1.0, 0.0), (2.5, 1.2, 90.0)],
    "three PAs": [(2.0, 1.0, 0.0), (1.9, 1.05, 50.0), (1.95, 1.0, -45.0)],
    "near-identical": [(1.0, 0.8, 30.0), (1.0, 0.8, 30.0001), (0.99999, 0.80001, 30.0)],
    "PA near 90": [(1.5, 1.0, 89.0), (1.49, 1.01, -88.0), (1.48, 1.0, 87.0)],
}
# [channel][polarization]
TWO_POLARIZATIONS = [
    [(2.0, 1.0, 0.0), (1.95, 1.0, -45.0)],
    [(1.9, 1.05, 50.0), (2.0, 1.0, 10.0)],
]

# image.commonbeam() of casatools 6.7.6-14, in radians, for images with per-plane beams set
# as '<repr>rad' strings from exactly the doubles these tests give common_beam: the sets of
# CASA_SETS as float64 radians (False) and rounded to float32 (True).
CASA_COMMON_BEAMS = {
    ("all equal", False): (
        4.84813681109536e-06,
        3.878509448876288e-06,
        0.5235987755982988,
    ),
    ("all equal", True): (
        4.848136995860841e-06,
        3.878509232890792e-06,
        0.5235987901687622,
    ),
    ("largest contains all", False): (
        5.817764173314431e-06,
        4.6057299705405916e-06,
        0.5585053606381855,
    ),
    ("largest contains all", True): (
        5.817764304083539e-06,
        4.6057298277446534e-06,
        0.5585053563117981,
    ),
    ("two PAs", False): (
        1.6950237546376048e-05,
        8.612645519352222e-06,
        0.6623736215103411,
    ),
    ("two PAs", True): (
        1.6950237352037298e-05,
        8.612645765114705e-06,
        0.6623736459205128,
    ),
    ("mirrored PA", False): (
        2.249901641268438e-05,
        9.387114176023786e-06,
        -1.231686506689072,
    ),
    ("mirrored PA", True): (
        2.2499016341757695e-05,
        9.387114843929027e-06,
        -1.2316864272068928,
    ),
    ("right angle", False): (1.454441043328608e-05, 1.21203420277384e-05, 0.0),
    ("right angle", True): (
        1.4558954488165909e-05,
        1.2132462376939637e-05,
        -5.931605961961849e-08,
    ),
    ("three PAs", False): (
        1.1099743785965947e-05,
        8.841597273469036e-06,
        -0.18058868096631048,
    ),
    ("three PAs", True): (
        1.112053198727281e-05,
        8.851570482534604e-06,
        -0.18204151394690088,
    ),
    ("near-identical", False): (
        4.852984966375699e-06,
        3.88243649999807e-06,
        0.5235995516040775,
    ),
    ("near-identical", True): (
        4.852985150906644e-06,
        3.88243646030316e-06,
        0.5235995585544756,
    ),
    ("PA near 90", False): (
        7.5194174120194915e-06,
        5.098706735793113e-06,
        -1.5025279086071144,
    ),
    ("PA near 90", True): (
        7.519417507244098e-06,
        5.098706757866188e-06,
        -1.502527884800134,
    ),
}
# casatools 6.7.6-14 as above (float64): "three PAs" in reversed order, TWO_POLARIZATIONS,
# and "mirrored PA" turned by -45 deg.
THREE_PAS_REVERSED = (
    1.0989454762911535e-05,
    8.892541244405772e-06,
    0.13135402231873206,
)
TWO_POLARIZATIONS_EXPECTED = (
    1.1435255763981921e-05,
    8.682829099000295e-06,
    -0.10914975423278478,
)
MIRRORED_PA_TURNED = (1.6487903480947445e-05, 6.879137721400835e-06, 0.7953541936904902)


def radians(beams, single=False):
    """Beams in arcsec and degrees as radians, float64 or rounded to float32."""
    values = np.array(beams, dtype=np.float64) * [ARCSEC, ARCSEC, DEGREE]
    return values.astype(np.float32).astype(np.float64) if single else values


def cube(beams):
    """Beams of one time and one polarization, ``(time, frequency, polarization, 3)``."""
    return beams[np.newaxis, :, np.newaxis]


def assert_beam(result, expected):
    np.testing.assert_allclose(result[:2], expected[:2], rtol=1e-12)
    np.testing.assert_allclose(result[2], expected[2], rtol=0.0, atol=1e-12)


def casa_deconvolve(convolved, beam):
    """CASA's GaussianDeconvolver::deconvolve in radians, the deconvolved beam."""
    (major_s, minor_s, pa_s), (major_b, minor_b, pa_b) = convolved, beam
    cos_s, sin_s, cos_b, sin_b = np.cos(pa_s), np.sin(pa_s), np.cos(pa_b), np.sin(pa_b)
    alpha = (
        (major_s * cos_s) ** 2
        + (minor_s * sin_s) ** 2
        - (major_b * cos_b) ** 2
        - (minor_b * sin_b) ** 2
    )
    beta = (
        (major_s * sin_s) ** 2
        + (minor_s * cos_s) ** 2
        - (major_b * sin_b) ** 2
        - (minor_b * cos_b) ** 2
    )
    gamma = 2 * (
        (minor_s**2 - major_s**2) * sin_s * cos_s
        - (minor_b**2 - major_b**2) * sin_b * cos_b
    )
    s = alpha + beta
    t = np.sqrt((alpha - beta) ** 2 + gamma**2)
    return (
        np.sqrt(0.5 * (s + t)),
        np.sqrt(0.5 * (s - t)),
        0.5 * np.arctan2(-gamma, alpha - beta),
    )


@pytest.mark.parametrize(("name", "single"), list(CASA_COMMON_BEAMS))
def test_common_beam_equals_casatools(name, single):
    assert_beam(
        common_beam(cube(radians(CASA_SETS[name], single))),
        CASA_COMMON_BEAMS[name, single],
    )


def test_common_beam_of_nearly_identical_float32_beams():
    """CAS-14719: float32 beams whose position angles differ by one ulp make the scaled
    ellipse circular, where casatools 6.7.6-14 returns a major axis 23 times too long.
    One growth step contains both beams."""
    major, minor, pa = np.array(
        [900584750, 894658310, 1036331922], dtype=np.uint32
    ).view(np.float32)
    beams = np.array(
        [[major, minor, pa], [major, minor, np.nextafter(pa, np.float32(1))]],
        dtype=np.float64,
    )
    result = common_beam(cube(beams))
    np.testing.assert_allclose(result[:2] / beams[0, :2], 1.001, rtol=1e-8)
    correcting_beam(beams, result)


def test_common_beam_of_a_circle_and_a_beam_at_45_degrees():
    """CAS-14719: casatools 6.7.6 builds a NaN beam here and fails. 1.2012" x 1.001" at
    45 deg, from CASA's algorithm with the CAS-14719 transform (reproduced in Python)."""
    result = common_beam(cube(radians([(1.0, 1.0, 0.0), (1.2, 0.7, 45.0)])))
    assert_beam(
        result, (5.823581937487747e-06, 4.8529849479064545e-06, 0.7853981633974483)
    )


def test_common_beam_returns_at_a_right_angle_without_testing_the_other_beams():
    """The last beam that does not fit is at a right angle to the largest: the ellipse of
    their major axes, which misses the second beam; correcting_beam finds it."""
    beams = radians([(3.0, 1.0, 0.0), (3.2, 0.5, 20.0), (2.5, 1.1, 90.0)])
    result = common_beam(cube(beams))
    np.testing.assert_array_equal(result, [beams[0, 0], beams[2, 0], 0.0])
    with pytest.raises(ValueError, match=r"1 of 3 plane\(s\):\n  plane \(1,\)"):
        correcting_beam(beams, result)
    # The beam that does not fit has the larger major axis: its pa.
    beams = radians([(2.5, 1.3, 0.0), (3.0, 1.0, 90.0)])
    np.testing.assert_array_equal(
        common_beam(cube(beams)), [beams[1, 0], beams[0, 0], beams[1, 2]]
    )


def test_common_beam_keeps_the_mirrored_position_angle():
    """Position angles more than 90 deg apart: CASA's enclosing ellipse takes the mirrored
    angle and grows until it contains both beams, to 1.86 times the area of the common
    beam of the same pair turned by -45 deg."""
    pair = CASA_SETS["mirrored PA"]
    turned = common_beam(cube(radians([(a, b, pa - 45.0) for a, b, pa in pair])))
    assert_beam(turned, MIRRORED_PA_TURNED)
    mirrored = common_beam(cube(radians(pair)))
    assert mirrored[0] * mirrored[1] / (turned[0] * turned[1]) == pytest.approx(
        1.862, abs=1e-3
    )


def test_common_beam_gives_up_after_1000_growth_steps(monkeypatch):
    """CAS-14719: with the containment test forced to fail, the beam of largest area and a
    warning, after the enclosing ellipse and 1000 growth steps."""
    calls, warnings = [], []

    def never(convolved, beam):
        calls.append(beam)
        return False

    monkeypatch.setattr(common_beam_module, "_deconvolves_cleanly", never)
    monkeypatch.setattr("toolviper.utils.logger.warning", warnings.append)
    beams = radians(CASA_SETS["three PAs"])
    np.testing.assert_array_equal(common_beam(cube(beams)), beams[0])
    # two tests of the other beams against the largest, then 1 + 1000 ellipses
    assert len(calls) == 2 + 1001
    assert len(warnings) == 1 and "1000 growth steps" in warnings[0]


def test_common_beam_visits_frequency_fastest_then_polarization():
    planes = radians([beam for channel in TWO_POLARIZATIONS for beam in channel])
    planes = planes.reshape(1, 2, 2, 3)
    assert_beam(common_beam(planes), TWO_POLARIZATIONS_EXPECTED)
    polarization_fastest = common_beam(planes.reshape(1, 4, 1, 3))
    assert not np.allclose(polarization_fastest, TWO_POLARIZATIONS_EXPECTED)
    data_array = xr.DataArray(
        planes[0].transpose(1, 0, 2),
        dims=("polarization", "frequency", "beam_params_label"),
    )
    np.testing.assert_array_equal(common_beam(data_array), common_beam(planes))


def test_common_beam_visits_time_last():
    beams = radians(CASA_SETS["three PAs"] + CASA_SETS["PA near 90"])
    np.testing.assert_array_equal(
        common_beam(beams.reshape(2, 3, 1, 3)), common_beam(cube(beams))
    )


def test_common_beam_depends_on_the_order():
    beams = radians(CASA_SETS["three PAs"])
    assert_beam(common_beam(cube(beams[::-1])), THREE_PAS_REVERSED)
    assert not np.allclose(common_beam(cube(beams)), THREE_PAS_REVERSED)


def test_common_beam_leaves_out_null_beams_and_failed_fits():
    beams = radians(CASA_SETS["three PAs"])
    with_others = np.array(
        [
            [np.nan] * 3,
            beams[0],
            [0.0, 0.0, 0.0],
            beams[1],
            [beams[0, 0], 0.0, 0.3],
            [np.inf, beams[0, 1], 0.0],
            beams[2],
        ]
    )
    np.testing.assert_array_equal(
        common_beam(cube(with_others)), common_beam(cube(beams))
    )


@pytest.mark.parametrize(
    "beam", [[-2e-6, -3e-6, 0.0], [2e-6, 3e-6, 0.0], [2e-6, -1e-7, 0.0]]
)
def test_common_beam_refuses_negative_or_swapped_axes(beam):
    beams = np.array([radians([(1.0, 0.8, 30.0)])[0], beam])
    with pytest.raises(ValueError, match="0 <= minor <= major"):
        common_beam(cube(beams))


def test_common_beam_refuses_a_set_without_a_beam():
    beams = np.array([[np.nan] * 3, [0.0, 0.0, 0.0], [1e-6, 0.0, 0.2]])
    with pytest.raises(ValueError, match="zero axis"):
        common_beam(cube(beams))


def test_common_beam_refuses_other_layouts():
    beams = radians(CASA_SETS["two PAs"])
    with pytest.raises(ValueError, match=r"\(time, frequency, polarization, 3\)"):
        common_beam(beams)
    with pytest.raises(ValueError, match="among time, frequency and polarization"):
        common_beam(xr.DataArray(beams, dims=("channel", "beam_params_label")))


def test_common_beam_leaves_its_input_alone():
    beams = cube(radians(CASA_SETS["three PAs"]))
    before = beams.copy()
    common_beam(beams)
    common_beam(
        xr.DataArray(
            beams, dims=("time", "frequency", "polarization", "beam_params_label")
        )
    )
    np.testing.assert_array_equal(beams, before)


def test_common_beam_position_angle_in_minus_half_pi_to_half_pi():
    """The astroviper fit gives pa in [0, pi)."""
    beams = radians([(1.0, 0.8, 120.0), (1.0, 0.8, 120.0), (0.9, 0.7, 125.0)])
    result = common_beam(cube(beams))
    np.testing.assert_array_equal(result[:2], beams[0, :2])
    assert result[2] == pytest.approx(math.radians(-60.0), abs=1e-15)
    assert common_beam(cube(radians([(1.0, 0.8, -90.0)])))[2] == math.pi / 2


def test_common_beam_contains_every_beam():
    rng = np.random.default_rng(5)
    for _ in range(20):
        majors = rng.uniform(0.9, 1.1, 8)
        beams = radians(
            np.column_stack(
                [majors, majors * rng.uniform(0.5, 1.0, 8), rng.uniform(-90, 90, 8)]
            )
        )
        correcting_beam(beams, common_beam(cube(beams)))


def test_correcting_beam_of_the_memo_appendix():
    """CASA Memo 10, appendix."""
    psf = radians([(0.7956633567810059, 0.6768985390663147, 56.512718200683594)])
    params, area_ratio = correcting_beam(psf, radians([(1.05, 0.83, 88.7)])[0])
    np.testing.assert_allclose(params[0, :2] / ARCSEC, [0.732485, 0.404468], rtol=1e-6)
    assert math.degrees(params[0, 2]) == pytest.approx(-78.7936, abs=1e-4)
    assert area_ratio[0] == pytest.approx(1.6181, abs=1e-4)


@pytest.mark.parametrize(
    ("beam", "target"),
    [
        ((6.0, 5.0, 45.0), (7.0, 3.0, 45.0)),
        ((5.0, 5.0, 0.0), (6.0, 3.0, 45.0)),
        ((5.0, 4.0, 45.0), (6.0, 2.0, 135.0)),
    ],
)
def test_correcting_beam_refuses_the_pairs_memo_10_accepts(beam, target):
    """Memo 10's test (eqs. 24-26) accepts these, though the target does not contain the
    beam; the message names the plane and the growth factor that would contain it."""
    beams = radians([beam])
    target = radians([target])[0]
    with pytest.raises(ValueError, match=r"plane \(0,\): .*growth factor") as error:
        correcting_beam(beams, target)
    assert "round it up" in str(error.value)
    growth = float(re.search(r"grow by (\S+) to contain", str(error.value)).group(1))
    correcting_beam(beams, target * [growth * (1 + 1e-6), growth * (1 + 1e-6), 1.0])
    with pytest.raises(ValueError, match="does not contain"):
        correcting_beam(beams, target * [growth * (1 - 1e-4), growth * (1 - 1e-4), 1.0])


def test_correcting_beam_of_equal_and_touching_beams():
    beam = radians([(4.0, 3.0, 30.0)])
    params, area_ratio = correcting_beam(beam, beam[0])
    np.testing.assert_array_equal(params, [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(area_ratio, [1.0])
    params, _ = correcting_beam(
        radians([(4.0, 3.0, 0.0)]), radians([(6.0, 3.0, 0.0)])[0]
    )
    np.testing.assert_allclose(
        params[0], [math.sqrt(6.0**2 - 4.0**2) * ARCSEC, 0.0, 0.0], rtol=1e-12, atol=0.0
    )


def test_correcting_beam_tolerates_float32_rounding_of_the_target():
    """The float32 copy of a 10:1 beam is smaller along one direction, by 7.8e-8 of the
    beam's variance along it, within the tolerance; a target 1e-4 smaller along one axis is
    refused."""
    beam = radians([(10.0, 1.0, 80.0)])
    target = beam[0].astype(np.float32).astype(np.float64)
    assert _eigenvalues(_beam_covariance(target) - _beam_covariance(beam[0]))[0] < 0.0
    params, _ = correcting_beam(beam, target)
    assert np.isfinite(params).all()
    for axis in (0, 1):
        smaller = beam[0].copy()
        smaller[axis] *= 1.0 - 1e-4
        with pytest.raises(ValueError, match="does not contain"):
            correcting_beam(beam, smaller)


@pytest.mark.parametrize("axis_ratio", [1.0, 10.0, 300.0, 1000.0])
def test_containment_tolerance_does_not_grow_with_the_axis_ratio(axis_ratio):
    """A target narrower than the beam along an axis by 4e-7 of its width is rounding, by
    1e-6 it is not, at any axis ratio; also when the other axis is short by nearly the
    tolerance. Measured against the beam's largest variance, the tolerance let a target
    4.5% narrower than a 300:1 beam through."""
    beam = radians([(30.0, 30.0 / axis_ratio, 20.0)])
    for axis in (0, 1):
        for shortfall, state in ((4e-7, "contained"), (1e-6, "not contained")):
            target = beam[0].copy()
            target[axis] *= 1.0 - shortfall
            assert _correcting_covariance(beam, target)[2].tolist() == [state]
    hidden = beam[0] * [1.0 - 4.9e-7, 1.0 - 4e-6, 1.0]
    assert _correcting_covariance(beam, hidden)[2].tolist() == ["not contained"]
    with pytest.raises(ValueError, match="does not contain"):
        correcting_beam(beam, beam[0] * [1.0, 0.955, 1.0])


def test_correcting_beam_equals_the_casa_deconvolver():
    rng = np.random.default_rng(1)
    major = rng.uniform(1.0, 3.0, 50) * ARCSEC
    beams = np.stack(
        [major, major * rng.uniform(0.3, 1.0, 50), rng.uniform(-1.5, 1.5, 50)], axis=-1
    )
    major = rng.uniform(0.5, 2.0, 50) * ARCSEC
    kernels = np.stack(
        [major, major * rng.uniform(0.4, 1.0, 50), rng.uniform(-1.5, 1.5, 50)], axis=-1
    )
    targets = _beam_from_covariance(_beam_covariance(beams) + _beam_covariance(kernels))
    for beam, target in zip(beams, targets, strict=True):
        params, _ = correcting_beam(beam, target)
        expected = casa_deconvolve(target, beam)
        np.testing.assert_allclose(params[:2], expected[:2], rtol=1e-12)
        assert params[2] == pytest.approx(expected[2], abs=1e-12)


def test_correcting_beam_of_planes_without_a_fit():
    beams = np.array([[np.nan] * 3, [0.0, 0.0, 0.0], radians([(1.0, 0.8, 30.0)])[0]])
    params, area_ratio = correcting_beam(beams, radians([(1.2, 1.0, 30.0)])[0])
    assert np.isnan(params[:2]).all() and np.isnan(area_ratio[:2]).all()
    assert np.isfinite(params[2]).all() and np.isfinite(area_ratio[2])


def test_correcting_beam_names_the_frequencies_of_a_data_array():
    beams = xr.DataArray(
        radians([(1.0, 0.8, 30.0), (1.1, 0.9, 30.0), (1.3, 0.9, 30.0)])[:, np.newaxis],
        dims=("frequency", "polarization", "beam_params_label"),
        coords={"frequency": [1.0e11, 1.01e11, 1.02e11], "polarization": ["I"]},
    )
    with pytest.raises(
        ValueError, match=r"frequency 2 \(1\.02e\+11 Hz\), polarization 0: \["
    ):
        correcting_beam(beams, radians([(1.2, 1.0, 30.0)])[0])


def test_correcting_beam_names_ten_planes():
    beams = np.broadcast_to(radians([(1.3, 0.9, 30.0)]), (12, 3))
    with pytest.raises(ValueError, match=r"12 of 12 plane\(s\)") as error:
        correcting_beam(beams, radians([(1.2, 1.0, 30.0)])[0])
    assert str(error.value).count("growth factor") == 10
    assert "\n  and 2 more\n" in str(error.value)


@pytest.mark.parametrize("target", ["common", [2e-6, 3e-6, 0.0], [2e-6, 1e-6]])
def test_correcting_beam_refuses_a_target_that_is_not_a_beam(target):
    with pytest.raises(ValueError):
        correcting_beam(radians([(1.0, 0.8, 30.0)]), target)


def test_correcting_beam_keeps_the_shape_of_its_input():
    target = radians([(1.2, 1.0, 30.0)])[0]
    beams = np.broadcast_to(radians([(1.0, 0.8, 30.0)]), (2, 3, 1, 3))
    params, area_ratio = correcting_beam(beams, target)
    assert params.shape == (2, 3, 1, 3) and area_ratio.shape == (2, 3, 1)
    params, area_ratio = correcting_beam(beams[0, 0, 0], target)
    assert params.shape == (3,) and area_ratio.shape == ()
    with pytest.raises(ValueError, match=r"shaped \(\.\.\., 3\)"):
        correcting_beam(beams[..., :2], target)


def test_correcting_covariance_states_without_raising():
    target = radians([(1.2, 1.0, 30.0)])[0]
    beams = np.array(
        [
            target,
            radians([(1.0, 0.8, 30.0)])[0],
            radians([(1.3, 0.9, 30.0)])[0],
            [np.nan] * 3,
            [0.0, 0.0, 0.0],
        ]
    )
    covariance, area_ratio, state = _correcting_covariance(beams, target)
    assert state.tolist() == [
        "contained",
        "contained",
        "not contained",
        "no fit",
        "no fit",
    ]
    np.testing.assert_array_equal(covariance[0], np.zeros((2, 2)))
    assert area_ratio[0] == 1.0
    assert np.isfinite(covariance[1]).all() and np.isnan(covariance[2:]).all()
    assert np.isfinite(area_ratio[2]) and np.isnan(area_ratio[3:]).all()


def test_correcting_covariance_is_zero_for_the_beam_itself():
    """Exactly zero and an area ratio of exactly one, so the restore leaves the residual
    of such a plane bit for bit; also for the beam's common beam, whose pa is unwrapped
    from the fit's [0, pi) to (-pi/2, pi/2]."""
    rng = np.random.default_rng(2)
    major = rng.uniform(1.0, 3.0, (1, 4, 2)) * ARCSEC
    beams = np.stack(
        [
            major,
            major * rng.uniform(0.3, 1.0, major.shape),
            rng.uniform(0.0, np.pi, major.shape),
        ],
        axis=-1,
    )
    for index in np.ndindex(beams.shape[:-1]):
        for target in (beams[index], common_beam(beams[index][None, None, None])):
            covariance, area_ratio, state = _correcting_covariance(beams, target)
            np.testing.assert_array_equal(covariance[index], 0.0)
            assert area_ratio[index] == 1.0 and state[index] == "contained"
            covariance, area_ratio, state = _correcting_covariance(beams[index], target)
            np.testing.assert_array_equal(covariance, 0.0)
            assert area_ratio == 1.0 and state == "contained"


def test_correcting_covariance_sets_rounding_to_zero():
    beam = radians([(10.0, 1.0, 80.0)])[0]
    target = beam.astype(np.float32).astype(np.float64)
    difference = _beam_covariance(target) - _beam_covariance(beam)
    covariance, _, state = _correcting_covariance(beam, target)
    assert state == "contained"
    smallest, largest = _eigenvalues(covariance)
    np.testing.assert_allclose(largest, _eigenvalues(difference)[1], rtol=1e-12)
    assert abs(smallest) <= 1e-12 * largest


def test_beam_covariance_is_the_second_moment_of_the_restore_kernel():
    """The frame of the restore's clean beam: the major axis along (sin pa, -cos pa) in
    (l index, m index)."""
    from astroviper.processing_functions.imaging.restore import (
        _elliptical_gaussian_kernel,
    )

    major, minor, pa = 14.0, 6.0, 0.4
    kernel = _elliptical_gaussian_kernel(129, 129, major, minor, pa, np.float64)
    offsets = np.arange(129) - 64
    l_index, m_index = np.meshgrid(offsets, offsets, indexing="ij")
    weights = kernel / kernel.sum()
    moments = [
        [(weights * a * b).sum() for b in (l_index, m_index)]
        for a in (l_index, m_index)
    ]
    np.testing.assert_allclose(moments, _beam_covariance([major, minor, pa]), rtol=1e-9)


def test_beam_from_covariance_inverts_beam_covariance():
    rng = np.random.default_rng(4)
    major = rng.uniform(1.0, 3.0, 100) * ARCSEC
    beams = np.stack(
        [major, major * rng.uniform(0.2, 0.99, 100), rng.uniform(-1.5, 1.5, 100)],
        axis=-1,
    )
    result = _beam_from_covariance(_beam_covariance(beams))
    np.testing.assert_allclose(result[:, :2], beams[:, :2], rtol=1e-10)
    np.testing.assert_allclose(result[:, 2], beams[:, 2], rtol=0.0, atol=1e-12)
