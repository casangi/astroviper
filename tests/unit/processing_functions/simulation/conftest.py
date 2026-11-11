"""pytest configuration for the simulation unit tests."""

# ``data/generate_legacy_fixtures.py`` is a stand-alone script that regenerates the
# ``legacy_*.npz`` oracles from the original SIRIUS package, which is not a test
# dependency (see its docstring). The CI test template runs pytest with
# ``--doctest-modules``, which imports every ``.py`` file under ``tests/``, so keep
# pytest from importing the generator.
collect_ignore = ["data/generate_legacy_fixtures.py"]
