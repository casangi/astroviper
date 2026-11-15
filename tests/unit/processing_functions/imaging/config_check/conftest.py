"""Capture validation diagnostics through the production ToolVIPER logger."""

import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import toolviper.utils.logger as logger


@pytest.fixture
def diagnostic_warnings(monkeypatch):
    """Return a scoped recorder for warning-level validation log messages."""

    @contextmanager
    def capture(*, match=None):
        messages = []
        with monkeypatch.context() as patch:
            patch.setattr(
                logger,
                "warning",
                lambda message, **kwargs: messages.append(
                    SimpleNamespace(message=message)
                ),
            )
            yield messages
        if match is not None:
            assert any(re.search(match, str(item.message)) for item in messages), (
                messages
            )

    return capture
