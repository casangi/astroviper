"""Unit tests for the single-field point-spread-function processing function.

Exercises
:func:`astroviper.processing_functions.imaging.make_point_spread_function.make_single_field_point_spread_function`
on the local TW Hya processing set used by the stakeholder imaging test (the
download fixture for this data is not yet published, so the test is skipped when
the local copy is absent).  The PSF normalization (peak ``1.0`` at the image
centre) is the same convention the stakeholder test validates against CASA.
"""

from pathlib import Path

import numpy as np
import pytest

# The processing set lives next to the stakeholder imaging test.
PS_STORE = (
    Path(__file__).resolve().parents[3]
    / "stakeholder"
    / "twhya_selfcal_lsrk_5chans.ps.zarr"
)

pytestmark = pytest.mark.skipif(
    not PS_STORE.exists(),
    reason=f"local processing set not available: {PS_STORE}",
)

IMAGE_SIZE = [250, 250]
CELL_SIZE = np.array([-0.1, 0.1]) * np.pi / (180 * 3600)


@pytest.fixture(scope="module")
def psf_result():
    """Build the PSF for the local processing set (shared across tests).

    Mirrors the once-per-chunk setup: compute the imaging weights, then make the
    point spread function.
    """
    from xradio.measurement_set import open_processing_set
    from xradio.measurement_set.load_processing_set import load_processing_set
    from xradio.image import make_empty_sky_image
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )
    from astroviper.processing_functions.imaging.make_point_spread_function import (
        make_single_field_point_spread_function,
    )

    ps_meta = open_processing_set(str(PS_STORE))
    combined = ps_meta.xr_ps.get_combined_field_and_source_xds()
    phase_direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values
    frequency_coords = ps_meta.xr_ps.get_freq_axis().values

    ps_xdt = load_processing_set(
        str(PS_STORE), data_group_name="base", load_sub_datasets=False
    )

    # Empty image in the correlation basis, with an empty residual data group.
    img_xds = make_empty_sky_image(
        phase_center=phase_direction,
        image_size=IMAGE_SIZE,
        cell_size=CELL_SIZE,
        frequency_coords=frequency_coords,
        pol_coords=["XX", "YY"],
        time_coords=[0],
        do_sky_coords=False,
    )
    img_xds.attrs["type"] = "image_dataset"
    img_xds = img_xds.xr_img.add_data_group(
        new_data_group_name="residual",
        new_data_group={"description": "test", "date": "2026"},
    )

    # Imaging weights (WEIGHT_IMAGING) are required by the UV-sampling gridder.
    calculate_imaging_weights(
        ps_xdt,
        img_xds,
        imaging_weights_params={
            "weighting": "briggs",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        return_weight_density_grid=False,
        ms_data_group_in_name="base",
        ms_data_group_out_name="base",
        ms_data_group_out_modified={"weight_imaging": "WEIGHT_IMAGING"},
    )

    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }
    img_xds, return_df = make_single_field_point_spread_function(
        ps_xdt,
        img_xds,
        image_params,
        ms_data_group_in_name="base",
        image_data_group_name="residual",
        num_threads=1,
        fft_backend="scipy",
        complex_dtype=np.complex128,
    )
    n_freq = len(frequency_coords)
    return img_xds, return_df, n_freq


def test_point_spread_function_shape_and_finite(psf_result):
    img_xds, _, n_freq = psf_result
    assert "POINT_SPREAD_FUNCTION" in img_xds.data_vars
    psf = img_xds["POINT_SPREAD_FUNCTION"]
    assert psf.dims == ("time", "frequency", "polarization", "l", "m")
    assert psf.shape == (1, n_freq, 2, IMAGE_SIZE[0], IMAGE_SIZE[1])
    assert np.all(np.isfinite(psf.values))


def test_point_spread_function_registered_in_data_group(psf_result):
    img_xds, _, _ = psf_result
    residual_group = img_xds.attrs["data_groups"]["residual"]
    assert residual_group["point_spread_function"] == "POINT_SPREAD_FUNCTION"


def test_point_spread_function_peak_normalized_at_centre(psf_result):
    """The PSF peaks at the image centre with value 1.0 (CASA convention)."""
    img_xds, _, n_freq = psf_result
    psf = img_xds["POINT_SPREAD_FUNCTION"].values
    cy, cx = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    for f in range(n_freq):
        plane = psf[0, f, 0]
        assert plane[cy, cx] == pytest.approx(1.0, abs=1e-5)
        peak = np.unravel_index(np.argmax(plane), plane.shape)
        assert peak == (cy, cx)


def test_point_spread_function_timing_columns(psf_result):
    _, return_df, _ = psf_result
    for column in ("T_gcf", "T_vis_mask", "T_uv_sampling_grid", "T_fft_norm"):
        assert column in return_df.columns
        assert float(return_df[column].iloc[0]) >= 0.0
