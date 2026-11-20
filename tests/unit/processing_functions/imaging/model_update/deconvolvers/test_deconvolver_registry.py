"""The registry of the deconvolvers, and the names config_check and the schemas take from it."""

import importlib
import json
import pathlib

import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
)
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_deconvolver,
)
from astroviper.processing_functions.imaging.model_update.deconvolvers.aspclean.deconvolver_asp import (
    DeconvolverAsp,
)
from astroviper.processing_functions.imaging.model_update.deconvolvers.deconvolver_base import (
    Deconvolver,
)
from astroviper.processing_functions.imaging.model_update.deconvolvers.deconvolver_registry import (
    DECONVOLVERS,
)
from astroviper.processing_functions.imaging.model_update.deconvolvers.hogbom.deconvolver_hogbom import (
    DeconvolverHogbom,
)
from astroviper.processing_functions.imaging.model_update.deconvolvers.hogbom.deconvolver_hogbom_many_threads import (
    DeconvolverHogbomManyThreads,
)
from astroviper.processing_functions.imaging.model_update.deconvolvers.mtmfs.deconvolver_mtmfs import (
    DeconvolverMtmfs,
)
from astroviper.utils.interface import Interface

# (model_basis, imaging_kinds) of each deconvolver.
CONSTANTS = {
    "hogbom": ("pixels", ("cube", "continuum")),
    "hogbom_many_threads": ("pixels", ("cube",)),
    "asp": ("pixels", ("cube", "continuum")),
    "mtmfs": ("taylor_terms", ("continuum",)),
}


def test_registry_maps_the_names_to_the_deconvolvers():
    assert list(DECONVOLVERS.items()) == [
        ("hogbom", DeconvolverHogbom),
        ("hogbom_many_threads", DeconvolverHogbomManyThreads),
        ("asp", DeconvolverAsp),
        ("mtmfs", DeconvolverMtmfs),
    ]


@pytest.mark.parametrize("name", list(CONSTANTS))
def test_every_registered_deconvolver_declares_its_basis_and_imaging_kinds(name):
    deconvolver = DECONVOLVERS[name]
    assert issubclass(deconvolver, Deconvolver)
    assert type(deconvolver) is Interface
    assert deconvolver.name == name
    assert (deconvolver.model_basis, deconvolver.imaging_kinds) == CONSTANTS[name]
    for method in ("deconvolve", "prepare_restoration"):
        assert isinstance(deconvolver.__dict__[method], staticmethod)


# The messages of check_deconvolver before the registry existed.
@pytest.mark.parametrize(
    "mode, name, message",
    [
        (
            CUBE_SINGLE_FIELD,
            "mtmfs",
            "Deconvolution algorithm 'mtmfs' not recognized; cube deconvolver must be "
            "one of ('hogbom', 'hogbom_many_threads', 'asp'); received 'mtmfs'.",
        ),
        (
            CONTINUUM_SINGLE_FIELD,
            "hogbom_many_threads",
            "Deconvolution algorithm 'hogbom_many_threads' not recognized; continuum "
            "deconvolver must be one of ('hogbom', 'asp', 'mtmfs'); received "
            "'hogbom_many_threads'.",
        ),
    ],
)
def test_check_deconvolver_lists_the_deconvolvers_of_the_imaging_kind(
    mode, name, message
):
    with pytest.raises(ValueError) as error:
        check_deconvolver(name, mode=mode)
    assert str(error.value) == message


@pytest.mark.parametrize(
    "application, imaging_kind",
    [
        ("image_cube_single_field", "cube"),
        ("image_continuum_single_field", "continuum"),
    ],
)
def test_application_schemas_allow_the_deconvolvers_of_their_imaging_kind(
    application, imaging_kind
):
    module = importlib.import_module(
        f"astroviper.distributed_applications.imaging.{application}"
    )
    schema = json.loads(
        pathlib.Path(module.__file__).with_name(f"{application}.param.json").read_text()
    )
    assert schema[application]["deconvolver"]["allowed"] == [
        name
        for name, deconvolver in DECONVOLVERS.items()
        if imaging_kind in deconvolver.imaging_kinds
    ]
