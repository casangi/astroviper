"""Component test: AstroVIPER moments vs pre-computed CASA ``immoments`` references.

The CASA reference planes live in ``data/casa_moments_reference.npz``,
generated once with ``data/generate_casa_moments_reference.py`` (which needs
``casatools``; the test suite itself does not import CASA).  The fixture holds
the synthetic line cube, the CASA ``fromarray`` frequency axis, the rest
frequency, and every CASA moment plane (+ mask) compared below.

Value moments (mean, median, std, rms, abs mean dev, max, min) are compared
directly.  CASA expresses spectral coordinate-valued moments (and the
``integrated`` moment's channel width) in radio velocity (km/s) while
AstroVIPER uses the native frequency axis (Hz); those are compared through the
exact linear radio-velocity conversion ``v = c * (1 - f / f0)`` with ``f0``
and the per-channel frequencies stored in the reference file.
"""

import os

import numpy as np
import pytest
import xarray as xr
from xradio.image import load_image, make_empty_sky_image, write_image

from astroviper.distributed_applications.image_analysis import moments

C_KM_S = 299792.458  # speed of light in km/s
INCLUDE_RANGE = (0.1, 1e30)  # significant-emission cut for moments 1 and 2

REFERENCE_FILE = os.path.join(
    os.path.dirname(__file__), "data", "casa_moments_reference.npz"
)

VALUE_MOMENTS = [
    "mean",
    "median",
    "standard_deviation",
    "rms",
    "abs_mean_dev",
    "maximum",
    "minimum",
]
COORD_MOMENTS = ["weighted_coord", "maximum_coord", "minimum_coord"]


def _casa_moment(reference, name):
    """The pre-computed CASA (l, m) plane and mask of one moment."""
    return reference[f"{name}_plane"], reference[f"{name}_mask"]


@pytest.fixture(scope="module")
def cubes(tmp_path_factory):
    """AstroVIPER moments of the reference cube, plus the CASA references."""
    tmp_path = tmp_path_factory.mktemp("moments_casa")
    reference = dict(np.load(REFERENCE_FILE))
    sky = reference["sky"]  # (chan, l, m); no NaNs, identical pixels both sides
    frequency = reference["frequency"]
    n_l, n_m = sky.shape[1], sky.shape[2]

    # AstroVIPER image with the SAME per-channel frequencies as the CASA image
    # the references were computed from.
    rad_per_arcsec = np.pi / 180 / 3600
    img_xds = make_empty_sky_image(
        phase_center=[0.6, -0.2],
        image_size=[n_l, n_m],
        cell_size=[15 * rad_per_arcsec, 15 * rad_per_arcsec],
        frequency_coords=frequency,
        pol_coords=["I"],
        time_coords=[0],
    )
    img_xds["SKY"] = xr.DataArray(
        sky[np.newaxis, :, np.newaxis],
        dims=["time", "frequency", "polarization", "l", "m"],
    )
    img_xds["SKY"].attrs["units"] = "Jy/beam"
    img_xds.attrs["data_groups"] = {"base": {"sky": "SKY"}}
    zarr_input = str(tmp_path / "astroviper_input.img.zarr")
    write_image(img_xds, imagename=zarr_input, out_format="zarr", overwrite=True)

    moments_store = str(tmp_path / "astroviper_moments.img.zarr")
    moments(
        input_image_store=zarr_input,
        moments_image_store=moments_store,
        moments=VALUE_MOMENTS
        + COORD_MOMENTS
        + ["integrated", "weighted_dispersion_coord"],
        moment_axis="frequency",
        n_chunks=2,
        overwrite=True,
    )
    # Second run restricted to significant emission: the CASA-recommended way
    # to compute the intensity-weighted moments 1 and 2.
    included_store = str(tmp_path / "astroviper_moments_included.img.zarr")
    moments(
        input_image_store=zarr_input,
        moments_image_store=included_store,
        moments=["weighted_coord", "weighted_dispersion_coord"],
        moment_axis="frequency",
        include_pixel_range=list(INCLUDE_RANGE),
        n_chunks=2,
        overwrite=True,
    )
    return {
        "reference": reference,
        "astroviper": load_image(moments_store),
        "astroviper_included": load_image(included_store),
        "frequency": frequency,
        "rest_frequency": float(reference["rest_frequency"]),
    }


def _astroviper_plane(moments_xds, name):
    """AstroVIPER moment map transposed to CASA's (x=l, y=m) ordering."""
    return (
        moments_xds["SKY_MOMENT_" + name.upper()]
        .isel(time=0, frequency=0, polarization=0)
        .values
    )


@pytest.mark.parametrize("name", VALUE_MOMENTS)
def test_value_moments_match_casa(cubes, name):
    casa_plane, _ = _casa_moment(cubes["reference"], name)
    np.testing.assert_allclose(
        _astroviper_plane(cubes["astroviper"], name),
        casa_plane,
        rtol=2e-5,
        atol=1e-7,
        err_msg=f"moment '{name}' differs from CASA",
    )


@pytest.mark.parametrize("name", COORD_MOMENTS)
def test_coordinate_moments_match_casa_in_velocity(cubes, name):
    casa_plane, _ = _casa_moment(cubes["reference"], name)
    ours_hz = _astroviper_plane(cubes["astroviper"], name)
    # radio velocity: v[km/s] = c * (1 - f / f0)
    ours_km_s = C_KM_S * (1 - ours_hz / cubes["rest_frequency"])
    np.testing.assert_allclose(
        ours_km_s,
        casa_plane,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"moment '{name}' differs from CASA (after velocity conversion)",
    )


def test_integrated_matches_casa_up_to_channel_width_units(cubes):
    casa_plane, _ = _casa_moment(cubes["reference"], "integrated")
    ours = _astroviper_plane(cubes["astroviper"], "integrated")  # Jy/beam.Hz
    # CASA integrates over radio velocity: dv = (c / f0) * df, in km/s.
    scale = C_KM_S / cubes["rest_frequency"]
    matches_signed = np.allclose(casa_plane, scale * ours, rtol=1e-5, atol=1e-6)
    matches_flipped = np.allclose(casa_plane, -scale * ours, rtol=1e-5, atol=1e-6)
    assert matches_signed or matches_flipped, (
        "integrated moment differs from CASA beyond the velocity-axis sign"
    )


def test_dispersion_matches_casa_scaled_with_include_range(cubes):
    """Moment 2 is only meaningful on significant positive emission (CASA docs
    recommend ``includepix``); on noise pixels both codes emit numerically
    meaningless values (CASA takes ``sqrt(abs(...))`` where AstroVIPER yields
    NaN), so the comparison is made under the same include range."""
    casa_plane, casa_mask = _casa_moment(
        cubes["reference"], "weighted_dispersion_coord_included"
    )
    ours = _astroviper_plane(
        cubes["astroviper_included"], "weighted_dispersion_coord"
    )  # Hz
    scale = C_KM_S / cubes["rest_frequency"]
    both = casa_mask & np.isfinite(ours)
    assert both.sum() > 100, "too few jointly valid pixels to compare"
    np.testing.assert_allclose(
        scale * ours[both],
        casa_plane[both],
        rtol=1e-3,
        atol=1e-3,
        err_msg="dispersion moment differs from CASA (after velocity scaling)",
    )


def test_weighted_coord_matches_casa_with_include_range(cubes):
    casa_plane, casa_mask = _casa_moment(cubes["reference"], "weighted_coord_included")
    ours_hz = _astroviper_plane(cubes["astroviper_included"], "weighted_coord")
    ours_km_s = C_KM_S * (1 - ours_hz / cubes["rest_frequency"])
    both = casa_mask & np.isfinite(ours_hz)
    assert both.sum() > 100, "too few jointly valid pixels to compare"
    np.testing.assert_allclose(
        ours_km_s[both],
        casa_plane[both],
        rtol=1e-4,
        atol=1e-4,
        err_msg="weighted_coord differs from CASA under the include range",
    )
