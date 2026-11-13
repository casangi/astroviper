"""Tests of the common beam and of the correcting beam."""

import math

import numpy as np
import pytest

from astroviper.processing_functions.image_analysis.common_beam import (
    common_beam,
    correcting_beam,
)

ARCSEC = math.pi / 180 / 3600

# image.commonbeam() of casatools 6.7.6-14 for beam sets of (major", minor", pa deg),
# as (major", minor", pa deg).
CASA_COMMON_BEAMS = [
    ([(1.0, 0.8, 30.0)] * 3, (1.0, 0.8, 30.0)),
    ([(1.0, 0.8, 30.0), (1.1, 0.88, 30.0), (1.2, 0.95, 32.0)], (1.2, 0.95, 32.0)),
    (
        [(3.0, 1.0, 20.0), (2.8, 1.1, 60.0)],
        (3.4962374633455138, 1.7764856593241087, 37.95121297333834),
    ),
    (
        [(3.0, 1.0, 80.0), (2.9, 1.2, -80.0)],
        (4.640755261112584, 1.9362312867369176, -70.57043851649567),
    ),
    ([(3.0, 1.0, 0.0), (2.5, 1.2, 90.0)], (3.0, 2.5, 0.0)),
    (
        [(2.0, 1.0, 0.0), (1.9, 1.05, 50.0), (1.95, 1.0, -45.0)],
        (2.289486501404678, 1.8237103485269461, -10.346969247204092),
    ),
    (
        [(1.0, 0.8, 30.0), (1.0, 0.8, 30.0001), (0.99999, 0.80001, 30.0)],
        (1.0010000038095546, 0.8008100124387569, 30.000044453657992),
    ),
    (
        [(1.5, 1.0, 89.0), (1.49, 1.01, -88.0), (1.48, 1.0, 87.0)],
        (1.550991175581243, 1.0516837569691315, -86.08850776380594),
    ),
]


def _beam(major, minor, pa):
    """A beam of major and minor axes in arcsec and a position angle in degrees."""
    return [major * ARCSEC, minor * ARCSEC, math.radians(pa)]


def _assert_beam(result, major, minor, pa, rtol=1e-9, pa_tolerance=1e-9):
    np.testing.assert_allclose(result[:2] / ARCSEC, [major, minor], rtol=rtol)
    assert result[2] == pytest.approx(math.radians(pa) % math.pi, abs=pa_tolerance)


@pytest.mark.parametrize("beams, expected", CASA_COMMON_BEAMS)
def test_common_beam_equals_casa(beams, expected):
    _assert_beam(common_beam([_beam(*b) for b in beams]), *expected)


def test_common_beam_of_nearly_identical_beams():
    """Two float32 beams whose position angles differ by one ulp: CASA 6.7's ellipse
    scaling returns a common beam 23 times longer; CAS-14719 (CASA 6.8) fixed it."""
    major, minor, pa = np.array(
        [900584750, 894658310, 1036331922], dtype=np.uint32
    ).view(np.float32)
    beams = np.array(
        [[major, minor, pa], [major, minor, np.nextafter(pa, np.float32(1))]]
    )
    result = common_beam(beams)
    assert major <= result[0] <= 1.002 * major
    assert minor <= result[1] <= 1.002 * minor


def test_common_beam_leaves_out_failed_fits():
    beams = [
        _beam(1.0, 0.8, 30.0),
        [np.nan] * 3,
        [0.0, 0.0, 0.0],
        _beam(1.2, 0.95, 32.0),
    ]
    _assert_beam(common_beam(np.array(beams)[None, :, None]), 1.2, 0.95, 32.0)
    with pytest.raises(ValueError, match="finite, positive"):
        common_beam([[np.nan] * 3])


def test_common_beam_contains_every_beam():
    rng = np.random.default_rng(5)
    for _ in range(20):
        beams = [
            _beam(major, major * rng.uniform(0.5, 1.0), rng.uniform(-90, 90))
            for major in rng.uniform(0.9, 1.1, 8)
        ]
        target = common_beam(beams)
        for beam in beams:
            correcting_beam(beam, target)


def test_correcting_beam_of_the_memo():
    # CASA Memo 10: same position angle, and the appendix example.
    _assert_beam(
        correcting_beam(_beam(4.0, 3.0, 30.0), _beam(6.5, 4.5, 30.0)),
        math.sqrt(6.5**2 - 4.0**2),
        math.sqrt(4.5**2 - 3.0**2),
        30.0,
    )
    psf = _beam(0.7956633567810059, 0.6768985390663147, 56.512718200683594)
    _assert_beam(
        correcting_beam(psf, _beam(1.05, 0.83, 88.7)),
        0.732485,
        0.404468,
        -78.7936,
        rtol=1e-6,
        pa_tolerance=math.radians(1e-4),
    )


def test_correcting_beam_of_equal_and_touching_beams():
    np.testing.assert_array_equal(
        correcting_beam(_beam(4.0, 3.0, 30.0), _beam(4.0, 3.0, 30.0)), [0.0, 0.0, 0.0]
    )
    touching = correcting_beam(_beam(4.0, 3.0, 0.0), _beam(6.0, 3.0, 0.0))
    assert touching[1] == pytest.approx(0.0, abs=1e-12 * ARCSEC)
    assert touching[0] / ARCSEC == pytest.approx(math.sqrt(6.0**2 - 4.0**2))


def test_correcting_beam_refuses_a_target_that_does_not_contain_the_beam():
    with pytest.raises(ValueError, match="does not contain"):
        correcting_beam(_beam(6.0, 5.0, 45.0), _beam(7.0, 3.0, 45.0))
