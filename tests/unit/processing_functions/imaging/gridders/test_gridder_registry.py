"""The registry of the gridders maps each gridder's name to its class."""

from types import MappingProxyType

import pytest

from astroviper.processing_functions.imaging.gridders.anti_alias.gridder_anti_alias import (
    GridderAntiAlias,
)
from astroviper.processing_functions.imaging.gridders.direct_fourier_transform.gridder_direct_fourier_transform import (
    GridderDirectFourierTransform,
)
from astroviper.processing_functions.imaging.gridders.gridder_base import Gridder
from astroviper.processing_functions.imaging.gridders.gridder_registry import GRIDDERS
from astroviper.utils.interface import Interface

METHODS = (
    "new_grid",
    "grid",
    "grid_uv_sampling",
    "degrid",
    "uv_to_image",
    "image_to_uv",
)


def test_registry_maps_the_names_to_the_gridders():
    assert list(GRIDDERS.items()) == [
        ("anti_alias", GridderAntiAlias),
        ("direct_fourier_transform", GridderDirectFourierTransform),
    ]


@pytest.mark.parametrize("name", list(GRIDDERS))
def test_every_registered_gridder_is_an_interface_class_of_its_name(name):
    gridder = GRIDDERS[name]
    assert issubclass(gridder, Gridder)
    assert type(gridder) is Interface
    assert gridder.name == name
    assert gridder.normalization in ("sum_of_weights", "weight_image")
    assert isinstance(gridder.gcf_terms, tuple)
    for roles in (gridder.grid_roles, gridder.grid_uv_sampling_roles):
        assert isinstance(roles, MappingProxyType)
        assert len(roles) == 2
    assert gridder.degrid_role is not None
    for method in METHODS:
        assert isinstance(gridder.__dict__[method], staticmethod)
