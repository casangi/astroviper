"""The two polarization modes of the imager: the requested Stokes planes fix the
correlations that are loaded and gridded."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.utils.imaging_polarization import (
    correlation_selection,
    correlations_for_stokes,
    is_four_correlation_basis,
)


@pytest.mark.parametrize(
    "stokes, basis, correlations",
    [
        (["I", "Q"], "linear", ["XX", "YY"]),
        (["I", "V"], "circular", ["RR", "LL"]),
        (["I", "Q", "U", "V"], "linear", ["XX", "XY", "YX", "YY"]),
        (["I", "Q", "U", "V"], "circular", ["RR", "RL", "LR", "LL"]),
    ],
)
def test_supported_modes(stokes, basis, correlations):
    assert correlations_for_stokes(stokes, basis) == correlations
    assert correlations_for_stokes(np.array(stokes), basis) == correlations
    assert correlations_for_stokes(tuple(stokes), basis) == correlations


@pytest.mark.parametrize(
    "stokes, basis",
    [
        (["I"], "linear"),  # a single plane
        (["Q", "I"], "linear"),  # wrong order
        (["I", "V"], "linear"),  # the parallel hands of linear feeds give I and Q
        (["I", "Q"], "circular"),  # ... and those of circular feeds I and V
        (["I", "Q", "U"], "linear"),
        (["XX", "YY"], "linear"),  # correlations are not Stokes planes
        ([], "linear"),
    ],
)
def test_unsupported_requests_are_refused(stokes, basis):
    with pytest.raises(ValueError, match="polarization_coords"):
        correlations_for_stokes(stokes, basis)


def test_unknown_basis_is_refused():
    with pytest.raises(ValueError, match="instrument_polarization_basis"):
        correlations_for_stokes(["I", "Q"], "elliptical")


def test_only_the_needed_correlations_are_selected():
    four = ["XX", "XY", "YX", "YY"]
    # nothing to select when the data holds exactly the needed correlations
    assert correlation_selection(["XX", "YY"], ["XX", "YY"]) is None
    assert correlation_selection(np.array(four), four) is None
    # the parallel hands of four-correlation data, and data in another order
    assert correlation_selection(four, ["XX", "YY"]) == [0, 3]
    assert correlation_selection(["YY", "XX"], ["XX", "YY"]) == [1, 0]
    assert correlation_selection(["RR", "RL", "LR", "LL"], ["RR", "LL"]) == [0, 3]
    with pytest.raises(ValueError, match=r"missing \['XY', 'YX'\]"):
        correlation_selection(["XX", "YY"], four, "ms_0")
    with pytest.raises(ValueError, match="instrument_polarization_basis"):
        correlation_selection(["RR", "LL"], ["XX", "YY"], "ms_0")


def test_four_correlation_basis():
    assert is_four_correlation_basis(["XX", "XY", "YX", "YY"])
    assert is_four_correlation_basis(np.array(["RR", "RL", "LR", "LL"]))
    assert not is_four_correlation_basis(["XX", "YY"])
    assert not is_four_correlation_basis(["I", "Q", "U", "V"])
    assert not is_four_correlation_basis(["XX", "XY", "LR", "LL"])
