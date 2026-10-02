"""Helpers for tests that check that objects die by reference counting."""

from __future__ import annotations

import collections
import gc


def _is_self_cached_unit(obj):
    """True for an astropy ``CompositeUnit`` whose ``_decomposed_cache`` is
    the unit itself."""
    if type(obj).__module__ != "astropy.units.core":
        return False
    if type(obj).__qualname__ != "CompositeUnit":
        return False
    return getattr(obj, "__dict__", {}).get("_decomposed_cache") is obj


def cyclic_garbage(garbage):
    """Types of the objects in ``garbage`` (what a collection found, kept with
    ``gc.DEBUG_SAVEALL``) that are not part of an astropy unit's self-cycle,
    counted.

    xradio's ``make_empty_sky_image`` converts the speed of light with
    ``_c.to("m/s")``, which parses "m/s" into a new astropy ``CompositeUnit``
    on every call; ``decompose()`` stores the unit in its own
    ``_decomposed_cache``, a cycle of about 0.6 kB per call (xradio 1.2.3,
    astropy 8). Only that cycle is left out: each ``CompositeUnit`` whose
    ``_decomposed_cache`` is the unit itself, and the plain dicts, lists and
    tuples of its own state it reaches through the garbage (its ``__dict__``,
    the lists of its bases and powers, the tuples of its physical type). Any
    other object is counted, an astropy ``Quantity``, another unit or an
    array held in one of those containers included. Drop the exception once
    xradio uses ``_c.value``.
    """
    in_garbage = {id(obj) for obj in garbage}
    explained = set()
    for root in garbage:
        if not _is_self_cached_unit(root):
            continue
        explained.add(id(root))
        stack = [root]
        while stack:
            for referent in gc.get_referents(stack.pop()):
                if (
                    id(referent) in in_garbage
                    and id(referent) not in explained
                    and type(referent) in (dict, list, tuple)
                ):
                    explained.add(id(referent))
                    stack.append(referent)
    return collections.Counter(
        f"{type(obj).__module__}.{type(obj).__qualname__}"
        for obj in garbage
        if id(obj) not in explained
    )
