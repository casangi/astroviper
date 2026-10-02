"""The cyclic garbage check of the memory tests leaves out the astropy unit
self-cycle xradio creates, and nothing else."""

from __future__ import annotations

import gc

import numpy as np

from tests.unit.node_tasks.imaging.cycle_test_utils import cyclic_garbage


def _garbage_of(make):
    """What a collection finds after ``make()``, with the collector off;
    a first call (imports, the parser's tables) is made outside."""
    make()
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    gc.freeze()
    try:
        make()
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        garbage = gc.garbage[:]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.unfreeze()
        if gc_was_enabled:
            gc.enable()
    return garbage


def _self_cached_units():
    """New astropy units in their own ``_decomposed_cache``, as xradio's
    ``_c.to("m/s")`` makes: every call parses "m/s" into a new unit, which
    the parser holds until it parses the next string."""
    import astropy.units as u

    speed = 2.99792458e08 * u.m / u.s
    for _ in range(3):
        speed.to("m/s")
    u.Unit("km")


def test_the_self_cached_unit_is_left_out():
    garbage = _garbage_of(_self_cached_units)
    assert any(type(obj).__qualname__ == "CompositeUnit" for obj in garbage)
    assert not cyclic_garbage(garbage)


def test_a_cycle_through_an_astropy_quantity_is_counted():
    import astropy.units as u

    def make():
        _self_cached_units()
        flux = 1.0 * u.Jy
        flux.holder = {"data": np.ones(10), "flux": flux}

    left = cyclic_garbage(_garbage_of(make))
    assert left["builtins.dict"] >= 1, left
    assert left["astropy.units.quantity.Quantity"] == 1, left
