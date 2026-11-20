"""The interface class Deconvolver declares no deconvolver and implements nothing."""

import inspect

import pytest

from astroviper.processing_functions.imaging.deconvolvers.deconvolver_base import (
    Deconvolver,
)


def test_deconvolver_declares_no_deconvolver():
    assert Deconvolver.name is None
    assert Deconvolver.model_basis is None
    assert Deconvolver.imaging_kinds == ()


@pytest.mark.parametrize("method", ["deconvolve", "prepare_restoration"])
def test_deconvolver_methods_are_static_and_not_implemented(method):
    assert isinstance(Deconvolver.__dict__[method], staticmethod)
    function = getattr(Deconvolver, method)
    n_positional = sum(
        parameter.kind is parameter.POSITIONAL_OR_KEYWORD
        for parameter in inspect.signature(function).parameters.values()
    )
    with pytest.raises(NotImplementedError):
        function(*[None] * n_positional)
