# viper

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Contributing

This project uses [poetry](https://python-poetry.org/) to manage dependencies and generate builds as well as [pre-commit](https://pre-commit.com/) to run linters and enforce code standards. To manage dependencies, first [manually install](https://python-poetry.org/docs/#installing-manually ) poetry then run `poetry run pre-commit install` the first time you clone the repository, and subsequent commits will run the git hooks specified in the pre-commit config file. Note that this includes the automatic generation a requirements.txt corresponding to the poetry lock file.

To add a new project-level dependency, run `poetry add my-new-module` to update pyproject.toml and the lock file. Use the `--dev` flag to add it as a development-specific requirement.

To run tests locally, run `poetry install` and then `poetry run pytest`.
