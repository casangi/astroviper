"""Tests of find_continuum_ultra's step results."""

import inspect
from typing import NamedTuple

from astroviper.processing_functions.imaging.find_continuum_ultra import step_results


class _Plain(NamedTuple):
    field: int


def _methods(cls):
    return {
        name
        for name, value in vars(cls).items()
        if inspect.isfunction(value)
        or isinstance(value, classmethod | staticmethod | property)
    }


def test_step_results_are_named_tuples_without_methods():
    classes = [
        value
        for value in vars(step_results).values()
        if inspect.isclass(value) and value.__module__ == step_results.__name__
    ]
    assert classes
    for cls in classes:
        assert cls.__mro__[1:] == (tuple, object), cls
        assert _methods(cls) == _methods(_Plain), cls
