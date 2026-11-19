import pytest

from astroviper.utils.interface import Interface, make_registry


class _Base(metaclass=Interface):
    name = None

    @staticmethod
    def add_one(value):
        return value + 1

    @staticmethod
    def run(value):
        raise NotImplementedError


class _First(_Base):
    name = "first"

    @staticmethod
    def run(value):
        return _Base.add_one(value)


class _Second(_Base):
    name = "second"


def test_static_methods_are_called_through_the_class():
    assert _First.run(1) == 2
    with pytest.raises(NotImplementedError):
        _Second.run(1)


def test_an_interface_class_cannot_be_instantiated():
    with pytest.raises(TypeError, match="_First cannot be instantiated"):
        _First()


def test_class_attributes_are_constant():
    with pytest.raises(AttributeError, match="_First.name cannot be reassigned"):
        _First.name = "other"
    with pytest.raises(AttributeError, match="_First.name cannot be deleted"):
        del _First.name
    assert _First.name == "first"


@pytest.mark.parametrize(
    "namespace",
    [
        {"run": lambda value: value},
        {"run": classmethod(lambda cls, value: value)},
        {"value": property(lambda self: 1)},
        {"__init__": lambda self: None},
    ],
    ids=["function", "classmethod", "property", "init"],
)
def test_a_class_body_holds_static_methods_only(namespace):
    with pytest.raises(TypeError, match="static"):
        Interface("_Refused", (), namespace)


def test_registry_maps_each_name_to_its_class():
    registry = make_registry(_First, _Second)
    assert registry == {"first": _First, "second": _Second}
    assert list(registry) == ["first", "second"]


def test_registry_refuses_two_classes_of_one_name():
    with pytest.raises(ValueError, match="'first'"):
        make_registry(_First, _First)
