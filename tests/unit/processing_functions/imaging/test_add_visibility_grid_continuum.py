"""Continuum gridding validates direct calls and trusts checked configuration."""

import pytest


def test_direct_gridder_checks_once_then_trusts_checked_input(monkeypatch):
    import astroviper.processing_functions.imaging.config_check as checks
    from astroviper.processing_functions.imaging.add_visibility_grid_continuum import (
        add_visibility_grid_continuum_single_field,
    )

    calls = []

    def shared(config, **kwargs):
        calls.append(config)
        raise ValueError("shared checker reached")

    monkeypatch.setattr(checks, "check_residual_update", shared)
    with pytest.raises(ValueError, match="shared checker reached"):
        add_visibility_grid_continuum_single_field(None, None, None, -1, 1e9)
    assert len(calls) == 1

    class RuntimeInput:
        @property
        def attrs(self):
            raise RuntimeError("runtime input reached")

    with pytest.raises(RuntimeError, match="runtime input reached"):
        add_visibility_grid_continuum_single_field(
            RuntimeInput(), None, None, 1, 1e9, input_checked=True
        )
    assert len(calls) == 1
