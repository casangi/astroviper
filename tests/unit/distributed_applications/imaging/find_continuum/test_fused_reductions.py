"""Tests of the fused reduction backend of find_continuum_fused."""

import numpy as np
import pytest

from astroviper.distributed_applications.imaging.find_continuum import (
    casa_image_tasks as cit,
)
from astroviper.distributed_applications.imaging.find_continuum.fused_reductions import (
    FusedReductions,
)
from astroviper.node_tasks.imaging.find_continuum import write_small_image

N_CHANNELS = 24
METADATA = {
    "shape": [4, 3, 1, N_CHANNELS],
    "increment": [-1.0e-6, 1.0e-6, 1.0, 1.0e6],
    "reference_value": [1.2, -0.4, 1.0, 230.0e9],
    "reference_pixel": [2.0, 1.0, 0.0, 0.0],
    "rest_frequency": 230.0e9,
}
CUBE, PRIMARY_BEAM = "/data/cube.residual", "/data/cube.pb"


class FakePasses:
    """Records every pass and answers each request with its pass and position."""

    def __init__(self):
        self.passes = []

    def __call__(self, requests):
        self.passes.append(list(requests))
        return [{"pass": len(self.passes), "position": i} for i in range(len(requests))]


@pytest.fixture
def backend():
    passes = FakePasses()
    fused = FusedReductions(passes, CUBE, PRIMARY_BEAM)
    cubes = {CUBE: {"metadata": METADATA}, PRIMARY_BEAM: {"metadata": METADATA}}
    with cit.cube_session(cubes, fused):
        yield fused, passes


def _statistics(chans="", mask="", per_channel=False):
    return cit._cube_request(CUBE, "statistics", chans, mask, per_channel=per_channel)


def test_first_pass_adds_the_first_round(backend):
    fused, passes = backend
    request = cit._moments_request(PRIMARY_BEAM, [-1], "3,6,12,18,21")
    fused([request])
    (first,) = passes.passes
    assert first[0] == request
    assert len(first) == 4
    moments = [r for r in first if r["kind"] == "moments" and r["image"] == CUBE]
    assert moments[0]["mask"] == f'"{CUBE}"!=0' and moments[0]["moments"] == [0, 8]
    assert {(r["image"], r["kind"]) for r in first[2:]} == {
        (CUBE, "statistics"),
        (PRIMARY_BEAM, "statistics"),
    }

    # The predicted reductions are answered without another pass.
    assert fused([_statistics()])[0]["pass"] == 1
    assert len(passes.passes) == 1


def test_cache_answers_copies_of_identical_requests(backend):
    fused, passes = backend
    request = _statistics(chans="2~5", per_channel=True)
    result = fused([request])[0]
    result["pass"] = -1
    assert fused([dict(request)])[0]["pass"] == 1
    assert fused([_statistics(chans="2~6", per_channel=True)])[0]["pass"] == 2


def test_changed_mask_image_misses_the_cache(backend, tmp_path):
    fused, passes = backend
    mask_image = str(tmp_path / "joint.mask")
    write_small_image(
        mask_image, np.zeros((4, 3, 1, 1)), np.ones((4, 3, 1, 1), dtype=bool), METADATA
    )
    request = cit._cube_request(
        CUBE, "profile", "", f'"{mask_image}">0', function="mean"
    )
    assert fused([request])[0]["pass"] == 1
    assert fused([request])[0]["pass"] == 1
    write_small_image(
        mask_image, np.ones((4, 3, 1, 1)), np.ones((4, 3, 1, 1), dtype=bool), METADATA
    )
    assert fused([request])[0]["pass"] == 2


def test_selection_round_adds_peak_and_intersections(backend):
    fused, passes = backend
    first = cit._moments_request(CUBE, [0, 8], "2~10")
    fused([first])
    assert _statistics(chans="2~10") in passes.passes[0]

    second = cit._moments_request(CUBE, [0, 8], "6~14")
    fused([second])
    assert cit._moments_request(CUBE, [0, 8], "6~10") in passes.passes[1]
    assert _statistics(chans="6~14") in passes.passes[1]
    assert fused([cit._moments_request(CUBE, [0, 8], "6~10")])[0]["pass"] == 2


def test_placeholders_while_capturing(backend):
    fused, _ = backend

    def asks(chans):
        statistics = cit.imstat(CUBE, axes=[0, 1, 2], chans=chans)
        assert np.isnan(statistics["median"]).all() and statistics["median"].size == 4
        profile = cit.iatool()
        profile.open(CUBE)
        assert np.isnan(profile.getprofile(function="mean")["values"]).all()
        assert np.isnan(cit.imstat(CUBE)["max"][0])
        moments = fused([cit._moments_request(CUBE, [8], chans)])[0]
        assert moments[8].shape == (4, 3)

    captured = fused._capture(asks, "1~4")
    assert [r["kind"] for r in captured] == [
        "statistics",
        "profile",
        "statistics",
        "moments",
    ]
    assert fused.captured is None


def test_failed_prediction_does_not_stop_the_run(backend):
    fused, _ = backend

    def fails():
        raise ValueError("no joint mask")

    assert fused._capture(fails) == []
    assert fused.captured is None
