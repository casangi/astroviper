"""Unit tests for the CASA-compatible C++ PSF beam fit (psf_gaussian_fit_cpp)."""

import numpy as np
import pytest

from astroviper.processing_functions.image_analysis.psf_gaussian_fit_cpp import (
    casa_psf_fit_available,
    fit_psf_beam,
)
from astroviper.processing_functions.imaging.restore import (
    _elliptical_gaussian_kernel,
)

ARCSEC = np.pi / (180 * 3600)

pytestmark = pytest.mark.skipif(
    not casa_psf_fit_available(), reason="psf_gaussian_fit C++ extension not built"
)


@pytest.mark.parametrize(
    "major_pix, minor_pix, pa",
    [(9.0, 5.0, 0.6), (12.0, 4.0, -1.1), (6.0, 5.9, 1.2)],
)
def test_recovers_synthetic_gaussian(major_pix, minor_pix, pa):
    """A noiseless Gaussian PSF is recovered in the restore-kernel convention."""
    cell = 0.5 * ARCSEC
    kernel = _elliptical_gaussian_kernel(128, 128, major_pix, minor_pix, pa, np.float64)
    beams = fit_psf_beam(kernel[None, None, None], np.array([-cell, cell]))
    major, minor, fit_pa = beams[0, 0, 0]
    np.testing.assert_allclose(major / cell, major_pix, rtol=5e-3)
    np.testing.assert_allclose(minor / cell, minor_pix, rtol=5e-3)
    pa_difference = (fit_pa - pa + np.pi / 2) % np.pi - np.pi / 2
    assert abs(pa_difference) < 5e-3


def test_float32_matches_float64_and_threads():
    cell = 2.0 * ARCSEC
    planes = np.stack(
        [
            _elliptical_gaussian_kernel(96, 96, 8.0, 5.0, 0.3, np.float64),
            _elliptical_gaussian_kernel(96, 96, 10.0, 6.0, -0.8, np.float64),
        ]
    )[None, :, None]
    double = fit_psf_beam(planes, np.array([-cell, cell]))
    single = fit_psf_beam(
        planes.astype(np.float32),
        np.array([-cell, cell]),
        processing_function_threads=2,
    )
    np.testing.assert_allclose(single, double, rtol=1e-4)


def test_peak_checks():
    cell = ARCSEC
    with pytest.raises(RuntimeError, match="zero"):
        fit_psf_beam(np.zeros((1, 1, 1, 64, 64)), np.array([-cell, cell]))
    off_centre = np.zeros((1, 1, 1, 64, 64))
    off_centre[..., 2, 2] = 1.0
    with pytest.raises(RuntimeError, match="inner quarter"):
        fit_psf_beam(off_centre, np.array([-cell, cell]))


def test_pipeline_fitting_method(tmp_path):
    """fitting_method='casa' is selectable in the fit processing function and
    agrees with the astroviper fit on a clean synthetic PSF."""
    import xarray as xr
    from xradio.image import make_empty_sky_image

    from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
        point_spread_function_gaussian_fit,
    )

    cell = np.array([-1.0, 1.0]) * ARCSEC
    img_xds = make_empty_sky_image(
        phase_center=np.array([0.0, 0.5]),
        image_size=[128, 128],
        cell_size=cell,
        frequency_coords=np.array([1.0e11]),
        pol_coords=["I"],
        time_coords=[0],
        do_sky_coords=False,
    )
    img_xds.attrs["type"] = "image_dataset"
    psf = _elliptical_gaussian_kernel(128, 128, 9.0, 5.0, 0.6, np.float64)
    img_xds["POINT_SPREAD_FUNCTION"] = xr.DataArray(
        psf[None, None, None], dims=("time", "frequency", "polarization", "l", "m")
    )
    img_xds = img_xds.xr_img.add_data_group(
        new_data_group_name="residual",
        new_data_group={
            "description": "test",
            "date": "2026",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        },
    )
    beams = {}
    for method in ("astroviper", "casa"):
        fitted = point_spread_function_gaussian_fit(
            img_xds.copy(deep=True),
            image_data_group_in_name="residual",
            image_data_group_out_name="residual",
            image_data_group_out_modified={
                "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
                "max_sidelobe_point_spread_function": "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
            },
            fitting_method=method,
        )
        beams[method] = fitted.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0, 0, 0]
    # Both recover the synthetic beam (the algorithms differ at the few-percent
    # level; the CASA method is the tighter one on a noiseless Gaussian).
    np.testing.assert_allclose(beams["casa"][0] / ARCSEC, 9.0, rtol=0.01)
    np.testing.assert_allclose(beams["casa"][1] / ARCSEC, 5.0, rtol=0.01)
    np.testing.assert_allclose(beams["astroviper"][0] / ARCSEC, 9.0, rtol=0.05)
    np.testing.assert_allclose(beams["astroviper"][1] / ARCSEC, 5.0, rtol=0.05)
    with pytest.raises(ValueError, match="fitting_method"):
        point_spread_function_gaussian_fit(
            img_xds.copy(deep=True),
            image_data_group_in_name="residual",
            image_data_group_out_name="residual",
            fitting_method="bogus",
        )
