"""The residual each deconvolver gives the restore."""

import pytest
import xarray as xr

from astroviper.processing_functions.imaging.deconvolvers.deconvolver_registry import (
    DECONVOLVERS,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    _mtmfs_principal_residual_for_restoration,
)
from tests.utils.continuum_images import make_runaway_taylor_image


@pytest.mark.parametrize("name", ["hogbom", "hogbom_many_threads", "asp"])
def test_pixel_deconvolvers_restore_the_residual_itself(name):
    image = make_runaway_taylor_image()
    residual_da = image.SKY_RESIDUAL
    residual = DECONVOLVERS[name].prepare_restoration(
        residual_da, image.POINT_SPREAD_FUNCTION
    )
    assert residual is residual_da


def test_mtmfs_restores_the_principal_solution_of_a_copy():
    image = make_runaway_taylor_image()
    # PSF order one couples the two Taylor terms at the PSF centre.
    image.POINT_SPREAD_FUNCTION.values[0, 1, 0, 16, 16] = 0.25
    snapshot = image.SKY_RESIDUAL.copy(deep=True)
    residual = DECONVOLVERS["mtmfs"].prepare_restoration(
        image.SKY_RESIDUAL, image.POINT_SPREAD_FUNCTION
    )
    principal = _mtmfs_principal_residual_for_restoration(
        image.SKY_RESIDUAL, image.POINT_SPREAD_FUNCTION
    )
    assert not principal.equals(snapshot)
    xr.testing.assert_identical(residual, principal)
    xr.testing.assert_identical(image.SKY_RESIDUAL, snapshot)
