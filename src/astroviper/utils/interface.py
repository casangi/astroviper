"""Metaclass and registry of the interface classes.

An interface class groups the static methods that several implementations
share (imagers, gridders, deconvolvers, solvers). It holds no data: all state
lives in the datasets its methods receive.
"""

import inspect


class Interface(type):
    """Metaclass of the interface classes.

    A class with this metaclass cannot be instantiated, its attributes cannot
    be reassigned or deleted once the class body is defined, and every
    function in its body other than a dunder method is a ``staticmethod``.

    Raises
    ------
    TypeError
        If a class body defines ``__init__``, ``__init_subclass__``,
        ``__class_getitem__``, a ``classmethod``, a ``property`` or a
        function that is neither a ``staticmethod`` nor a dunder method, or
        if the class is called.
    AttributeError
        If a class attribute is assigned or deleted.
    """

    def __new__(mcs, name, bases, namespace):
        for key, value in namespace.items():
            if key in ("__init__", "__init_subclass__", "__class_getitem__") or (
                isinstance(value, (classmethod, property))
            ):
                raise TypeError(
                    f"{name}.{key}: an interface class has static methods only."
                )
            if inspect.isfunction(value) and not key.startswith("__"):
                raise TypeError(f"{name}.{key} must be a staticmethod.")
        return super().__new__(mcs, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} cannot be instantiated.")

    def __setattr__(cls, name, value):
        raise AttributeError(f"{cls.__name__}.{name} cannot be reassigned.")

    def __delattr__(cls, name):
        raise AttributeError(f"{cls.__name__}.{name} cannot be deleted.")


def make_registry(*classes):
    """Map the ``name`` of each interface class to the class.

    Parameters
    ----------
    *classes : type
        Interface classes, each with a distinct string ``name``.

    Returns
    -------
    dict of str to type
        The classes keyed by name, in the order given.

    Raises
    ------
    ValueError
        If two classes have the same name.
    """
    registry = {}
    for cls in classes:
        if cls.name in registry:
            raise ValueError(f"Two interface classes are named {cls.name!r}.")
        registry[cls.name] = cls
    return registry
