"""Shared fixtures of the utils unit tests.

* ``logged``: the messages logged to the toolviper logger during the test;
  ``logged.read()`` returns those logged since the last read, each followed by
  a newline, and forgets them (as ``capsys.readouterr().out`` does for printed
  text).

Only the standard library and pytest are imported at module level, as in
``tests/unit/conftest.py``.
"""

import logging

import pytest


class _LoggedMessages(logging.Handler):
    """Collects the messages of the records it handles."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def read(self):
        """The messages logged since the last read, each followed by a newline."""
        text = "".join(f"{message}\n" for message in self.messages)
        self.messages.clear()
        return text


@pytest.fixture
def logged():
    import toolviper.utils.logger as logger

    target = logger.get_logger()
    handler = _LoggedMessages()
    target.addHandler(handler)
    yield handler
    target.removeHandler(handler)
