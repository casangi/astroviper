"""The registry of the deconvolvers: ``DECONVOLVERS`` maps each deconvolver's name to its class."""

from astroviper.processing_functions.imaging.deconvolvers.aspclean.deconvolver_asp import (
    DeconvolverAsp,
)
from astroviper.processing_functions.imaging.deconvolvers.hogbom.deconvolver_hogbom import (
    DeconvolverHogbom,
)
from astroviper.processing_functions.imaging.deconvolvers.hogbom.deconvolver_hogbom_many_threads import (
    DeconvolverHogbomManyThreads,
)
from astroviper.processing_functions.imaging.deconvolvers.mtmfs.deconvolver_mtmfs import (
    DeconvolverMtmfs,
)
from astroviper.utils.interface import make_registry

# check_deconvolver lists the names of an imaging kind in this order.
DECONVOLVERS = make_registry(
    DeconvolverHogbom, DeconvolverHogbomManyThreads, DeconvolverAsp, DeconvolverMtmfs
)
