"""The interface class Gridder obeys the rules of the interface classes."""

import inspect

import pytest

from astroviper.processing_functions.imaging.gridders.gridder_base import Gridder

METHODS = (
    "new_grid",
    "grid",
    "grid_uv_sampling",
    "degrid",
    "uv_to_image",
    "image_to_uv",
)
CONSTANTS = (
    "name",
    "normalization",
    "gcf_terms",
    "grid_roles",
    "grid_uv_sampling_roles",
    "degrid_role",
)


def test_gridder_cannot_be_instantiated():
    with pytest.raises(TypeError, match="Gridder cannot be instantiated"):
        Gridder()


@pytest.mark.parametrize("attribute", CONSTANTS + METHODS)
def test_gridder_attributes_cannot_be_reassigned_or_deleted(attribute):
    value = getattr(Gridder, attribute)
    with pytest.raises(
        AttributeError, match=f"Gridder.{attribute} cannot be reassigned"
    ):
        setattr(Gridder, attribute, None)
    with pytest.raises(AttributeError, match=f"Gridder.{attribute} cannot be deleted"):
        delattr(Gridder, attribute)
    assert getattr(Gridder, attribute) is value


@pytest.mark.parametrize("roles", ["grid_roles", "grid_uv_sampling_roles"])
def test_gridder_role_maps_are_read_only(roles):
    role_map = getattr(Gridder, roles)
    with pytest.raises(TypeError):
        role_map["visibility"] = "VISIBILITY"
    assert dict(role_map) == {}


def test_gridder_declares_no_gridder():
    assert Gridder.name is None
    assert Gridder.normalization is None
    assert Gridder.gcf_terms == ()
    assert Gridder.degrid_role is None


@pytest.mark.parametrize("method", METHODS)
def test_gridder_methods_are_static_and_not_implemented(method):
    assert isinstance(Gridder.__dict__[method], staticmethod)
    function = getattr(Gridder, method)
    n_parameters = len(inspect.signature(function).parameters)
    with pytest.raises(NotImplementedError):
        function(*[None] * n_parameters)
