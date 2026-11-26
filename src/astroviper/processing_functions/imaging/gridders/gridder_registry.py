"""The registry of the gridders: ``GRIDDERS`` maps each gridder's name to its class."""

from astroviper.processing_functions.imaging.gridders.anti_alias.gridder_anti_alias import (
    GridderAntiAlias,
)
from astroviper.processing_functions.imaging.gridders.direct_fourier_transform.gridder_direct_fourier_transform import (
    GridderDirectFourierTransform,
)
from astroviper.utils.interface import make_registry

GRIDDERS = make_registry(GridderAntiAlias, GridderDirectFourierTransform)
