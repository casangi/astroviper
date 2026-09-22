"""Component test: AstroVIPER moments vs CASA ``immoments`` on the same cube.

Skipped automatically when ``casatasks``/``casatools`` are not installed.

Value moments (mean, median, std, rms, abs mean dev, max, min) are compared
directly.  CASA expresses spectral coordinate-valued moments (and the
``integrated`` moment's channel width) in radio velocity (km/s) while
AstroVIPER uses the native frequency axis (Hz); those are compared through the
exact linear radio-velocity conversion ``v = c * (1 - f / f0)`` with ``f0``
and the per-channel frequencies read back from the CASA image's coordinate
system.
"""

import numpy as np
import pytest
import xarray as xr

casatools = pytest.importorskip("casatools")

from xradio.image import load_image, make_empty_sky_image, write_image  # noqa: E402

from astroviper.distributed_applications.image_analysis import moments  # noqa: E402

C_KM_S = 299792.458  # speed of light in km/s
INCLUDE_RANGE = (0.1, 1e30)  # significant-emission cut for moments 1 and 2

# (AstroVIPER name, CASA immoments code, CASA output suffix-free kind)
VALUE_MOMENTS = [
    ("mean", -1),
    ("median", 3),
    ("standard_deviation", 5),
    ("rms", 6),
    ("abs_mean_dev", 7),
    ("maximum", 8),
    ("minimum", 10),
]
COORD_MOMENTS = [
    ("weighted_coord", 1),
    ("maximum_coord", 9),
    ("minimum_coord", 11),
]


def _casa_moment(casa_image, code, outfile, includepix=None):
    """Run CASA moments for one moment; return the (l, m) plane and its mask.

    Uses the ``casatools`` image-tool ``moments`` method (the compute engine
    behind ``casatasks.immoments``); the task wrapper's ``logsink`` logging
    bus-errors inside pytest on macOS, the tool does not.
    """
    ia = casatools.image()
    ia.open(casa_image)
    kwargs = {} if includepix is None else {"includepix": includepix}
    moment_image = ia.moments(
        moments=[code], axis=3, outfile=outfile, drop=False, **kwargs
    )
    ia.close()
    pixels = np.squeeze(moment_image.getchunk())  # (x=l, y=m)
    mask = np.squeeze(moment_image.getchunk(getmask=True))
    moment_image.close()
    return pixels, mask


@pytest.fixture(scope="module")
def cubes(tmp_path_factory):
    """One synthetic line cube written both as a CASA image and a Zarr image."""
    tmp_path = tmp_path_factory.mktemp("moments_casa")
    n_chan, n_l, n_m = 24, 48, 40
    rng = np.random.default_rng(11)
    l_idx, m_idx = np.meshgrid(np.arange(n_l), np.arange(n_m), indexing="ij")
    spatial = np.exp(
        -(((l_idx - n_l / 2) ** 2 + (m_idx - n_m / 2) ** 2) / (2 * 6.0**2))
    )
    chan = np.arange(n_chan)
    center = n_chan / 2 + 4 * (l_idx - n_l / 2) / n_l
    profile = np.exp(-((chan[:, None, None] - center[None]) ** 2) / (2 * 2.5**2))
    sky = (spatial[None] * profile + rng.normal(0, 0.01, profile.shape)).astype(
        np.float64
    )  # (chan, l, m); no NaNs so CASA and AstroVIPER see identical pixels

    # CASA image: pixels ordered (x=l, y=m, stokes, chan) with the default
    # coordinate system fromarray creates.
    casa_image = str(tmp_path / "casa_input.im")
    ia = casatools.image()
    ia.fromarray(
        outfile=casa_image,
        pixels=np.transpose(sky, (1, 2, 0))[:, :, np.newaxis, :],
        overwrite=True,
    )
    frequency = np.array(
        [ia.toworld([0, 0, 0, c])["numeric"][3] for c in range(n_chan)]
    )
    # Put the rest frequency at band centre.  casacore stores the per-profile
    # moment results in float32 (``calcMoments`` is ``Vector<Float>``), so with
    # a rest frequency far from the band the velocities are ~1000 km/s and
    # CASA's moment-2 cancellation error alone is ~0.25 km/s; centring keeps
    # the velocities (and hence the float32 rounding) small on both sides.
    rest_frequency = float(frequency.mean())
    cs = ia.coordsys()
    cs.setrestfrequency(value=f"{rest_frequency}Hz")
    ia.setcoordsys(cs.torecord())
    cs.done()
    ia.close()

    # AstroVIPER image with the SAME per-channel frequencies.
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
        moments=[name for name, _ in VALUE_MOMENTS + COORD_MOMENTS]
        + ["integrated", "weighted_dispersion_coord"],
        moment_axis="frequency",
        n_mapping_parallelism={"m": 2},
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
        n_mapping_parallelism={"m": 2},
        overwrite=True,
    )
    return {
        "tmp_path": tmp_path,
        "casa_image": casa_image,
        "astroviper": load_image(moments_store),
        "astroviper_included": load_image(included_store),
        "frequency": frequency,
        "rest_frequency": rest_frequency,
    }


def _astroviper_plane(moments_xds, name):
    """AstroVIPER moment map transposed to CASA's (x=l, y=m) ordering."""
    return (
        moments_xds["SKY_MOMENT_" + name.upper()]
        .isel(time=0, frequency=0, polarization=0)
        .values
    )


@pytest.mark.parametrize(("name", "code"), VALUE_MOMENTS)
def test_value_moments_match_casa(cubes, name, code):
    casa_plane, _ = _casa_moment(
        cubes["casa_image"], code, str(cubes["tmp_path"] / f"casa_{name}.im")
    )
    np.testing.assert_allclose(
        _astroviper_plane(cubes["astroviper"], name),
        casa_plane,
        rtol=2e-5,
        atol=1e-7,
        err_msg=f"moment '{name}' differs from CASA",
    )


@pytest.mark.parametrize(("name", "code"), COORD_MOMENTS)
def test_coordinate_moments_match_casa_in_velocity(cubes, name, code):
    casa_plane, _ = _casa_moment(
        cubes["casa_image"], code, str(cubes["tmp_path"] / f"casa_{name}.im")
    )
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
    casa_plane, _ = _casa_moment(
        cubes["casa_image"], 0, str(cubes["tmp_path"] / "casa_integrated.im")
    )
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
        cubes["casa_image"],
        2,
        str(cubes["tmp_path"] / "casa_dispersion.im"),
        includepix=list(INCLUDE_RANGE),
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
    casa_plane, casa_mask = _casa_moment(
        cubes["casa_image"],
        1,
        str(cubes["tmp_path"] / "casa_weighted_coord_inc.im"),
        includepix=list(INCLUDE_RANGE),
    )
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
