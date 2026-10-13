"""Unit tests for ``image_statistics.image_residual_entropy``.

The entropy of Homan, Roth and Pushkarev (2024, AJ 167, 11) is computed from
the fractions of the pixels of a plane in cells of (spatial bin, flux bin),
the flux taken in units of the RMS of the plane.
"""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis import image_statistics
from astroviper.processing_functions.image_analysis.image_statistics import (
    image_residual_entropy,
)

DIMS = ("time", "frequency", "polarization", "l", "m")


def _img_xds(residual, mask=None):
    residual = np.asarray(residual, dtype=float)
    while residual.ndim < 5:
        residual = residual[None]
    img_xds = xr.Dataset({"SKY_RESIDUAL": (DIMS, residual)})
    data_group = {"sky": "SKY_RESIDUAL"}
    if mask is not None:
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), residual.shape).copy()
        img_xds["MASK"] = (DIMS, mask)
        data_group["mask"] = "MASK"
    img_xds.attrs["data_groups"] = {"residual": data_group}
    return img_xds


def _direct_entropy(plane, spatial_bins=7, flux_bins=10, edge=None):
    """The definition, written down without any blocks. ``edge`` is the
    outermost flux bin, for values that lie beyond it."""
    n_l, n_m = plane.shape
    rms = np.sqrt(np.mean(plane**2))
    flux = np.floor(plane / rms * flux_bins).astype(np.int64)
    if edge is not None:
        flux = np.clip(flux, -edge, edge - 1)
    flux -= flux.min()
    cell = (np.arange(n_l) * spatial_bins // n_l)[:, None] * spatial_bins + (
        np.arange(n_m) * spatial_bins // n_m
    )[None, :]
    counts = np.bincount((cell * (flux.max() + 1) + flux).ravel())
    fraction = counts[counts > 0] / plane.size
    return float(-(fraction * np.log(fraction)).sum())


def test_noise_has_the_entropy_of_a_gaussian_in_every_spatial_bin():
    # Gaussian noise in flux bins 0.1 sigma wide carries
    # 0.5 * ln(2 pi e) - ln(0.1) = 3.7215, the 49 spatial bins add ln(49).
    rng = np.random.default_rng(1)
    entropy, snr = image_residual_entropy(
        _img_xds(rng.standard_normal((1400, 1400))), "residual"
    )
    expected = np.log(49) + 0.5 * np.log(2 * np.pi * np.e) - np.log(0.1)
    assert entropy.shape == snr.shape == (1, 1, 1)
    assert entropy.item() == pytest.approx(expected, abs=0.01)
    assert 4.0 < snr.item() < 6.0


def test_matches_the_definition():
    rng = np.random.default_rng(2)
    plane = rng.standard_normal((250, 250))
    plane[100:140, 60:90] *= 0.3  # a region cleaned below its surroundings
    entropy, _ = image_residual_entropy(_img_xds(plane), "residual")
    assert entropy.item() == pytest.approx(_direct_entropy(plane), rel=0, abs=1e-12)
    for spatial_bins, flux_bins in ((1, 10), (5, 4), (12, 25)):
        entropy, _ = image_residual_entropy(
            _img_xds(plane), "residual", spatial_bins, flux_bins
        )
        assert entropy.item() == pytest.approx(
            _direct_entropy(plane, spatial_bins, flux_bins), rel=0, abs=1e-12
        )


def test_image_of_several_blocks_matches_the_definition():
    # 1300 columns: the histogram is filled in blocks of 806 rows
    rng = np.random.default_rng(3)
    plane = rng.standard_normal((1500, 1300))
    entropy, _ = image_residual_entropy(_img_xds(plane), "residual")
    assert entropy.item() == pytest.approx(_direct_entropy(plane), rel=0, abs=1e-12)


def test_structure_lowers_the_entropy():
    # The same pixel values, sorted: the flux distribution is unchanged, but
    # every spatial bin now holds a narrow range of fluxes.
    rng = np.random.default_rng(4)
    plane = rng.standard_normal((256, 256))
    ordered = np.sort(plane, axis=None).reshape(plane.shape)
    noise, _ = image_residual_entropy(_img_xds(plane), "residual")
    structure, _ = image_residual_entropy(_img_xds(ordered), "residual")
    without_space, _ = image_residual_entropy(_img_xds(plane), "residual", 1)
    without_space_ordered, _ = image_residual_entropy(_img_xds(ordered), "residual", 1)
    assert structure.item() < noise.item() - 1.0
    assert without_space.item() == pytest.approx(without_space_ordered.item())


def test_region_cleaned_below_its_surroundings_lowers_the_entropy():
    rng = np.random.default_rng(5)
    plane = rng.standard_normal((256, 256))
    noise, _ = image_residual_entropy(_img_xds(plane), "residual")
    cleaned = plane.copy()
    cleaned[64:192, 64:192] *= 0.5
    over_cleaned, _ = image_residual_entropy(_img_xds(cleaned), "residual")
    assert over_cleaned.item() < noise.item() - 0.05


def test_scale_of_the_image_does_not_matter():
    rng = np.random.default_rng(6)
    plane = rng.standard_normal((128, 128))
    one, snr_one = image_residual_entropy(_img_xds(plane), "residual")
    other, snr_other = image_residual_entropy(_img_xds(1e-4 * plane), "residual")
    assert other.item() == pytest.approx(one.item(), abs=1e-12)
    assert snr_other.item() == pytest.approx(snr_one.item(), rel=1e-12)


def test_full_dimensionality_and_planes_are_independent():
    rng = np.random.default_rng(7)
    cube = rng.standard_normal((2, 3, 4, 64, 64))
    cube[1, 2, 3] *= np.linspace(0.2, 1.0, 64)[:, None]
    entropy, snr = image_residual_entropy(_img_xds(cube), "residual")
    assert entropy.shape == snr.shape == (2, 3, 4)
    for index in np.ndindex(2, 3, 4):
        assert entropy[index] == pytest.approx(_direct_entropy(cube[index]), abs=1e-12)
    assert entropy[1, 2, 3] < entropy[:1].min()


def test_strong_peak_gives_no_entropy():
    rng = np.random.default_rng(8)
    plane = rng.standard_normal((128, 128))
    plane[40, 50] = -30.0
    entropy, snr = image_residual_entropy(_img_xds(plane), "residual")
    assert np.isnan(entropy.item())
    assert snr.item() == pytest.approx(30.0 / np.sqrt(np.mean(plane**2)))
    # a higher limit lets the plane in
    entropy, _ = image_residual_entropy(_img_xds(plane), "residual", max_snr=40.0)
    assert entropy.item() == pytest.approx(_direct_entropy(plane), abs=1e-12)


def test_peak_is_searched_inside_the_mask():
    rng = np.random.default_rng(9)
    plane = rng.standard_normal((128, 128))
    plane[5, 5] = 30.0  # outside the mask
    mask = np.zeros((128, 128), dtype=bool)
    mask[32:96, 32:96] = True
    entropy, snr = image_residual_entropy(_img_xds(plane, mask), "residual")
    rms = np.sqrt(np.mean(plane**2))
    assert snr.item() == pytest.approx(np.abs(plane[mask]).max() / rms)
    assert snr.item() < 6.0
    # the entropy itself covers the whole plane; the pixel far beyond
    # 2 * max_snr = 12 times the RMS is counted in the outermost flux bin
    assert plane[5, 5] > 12 * rms
    assert entropy.item() == pytest.approx(_direct_entropy(plane, edge=120), abs=1e-12)
    # without a mask the strong pixel is the peak
    entropy, snr = image_residual_entropy(_img_xds(plane), "residual")
    assert snr.item() == pytest.approx(30.0 / rms)
    assert np.isnan(entropy.item())


def test_histogram_stays_small_for_any_limit():
    # A limit of 1e9 on the ratio of peak to RMS would ask for 2e10 flux bins
    # per spatial bin; the histogram is capped instead and the entropy of a
    # noise image is not affected.
    rng = np.random.default_rng(13)
    plane = rng.standard_normal((128, 128))
    plane[3, 3] = 1e4
    entropy, snr = image_residual_entropy(_img_xds(plane), "residual", max_snr=1e9)
    rms = np.sqrt(np.mean(plane**2))
    assert snr.item() == pytest.approx(1e4 / rms)
    edge = image_statistics.MAX_ENTROPY_CELLS // (2 * 49)
    assert entropy.item() == pytest.approx(_direct_entropy(plane, edge=edge), abs=1e-12)
    # very many spatial bins: fewer flux bins, still a number
    entropy, _ = image_residual_entropy(
        _img_xds(plane), "residual", spatial_bins=128, flux_bins=1000, max_snr=1e9
    )
    assert np.isfinite(entropy.item())


def test_pixels_that_are_not_finite_are_left_out():
    rng = np.random.default_rng(10)
    plane = rng.standard_normal((128, 128))
    holes = plane.copy()
    holes[:, :16] = np.nan
    holes[3, 40] = np.inf
    entropy, snr = image_residual_entropy(_img_xds(holes), "residual")
    finite = np.isfinite(holes)
    rms = np.sqrt(np.mean(holes[finite] ** 2))
    assert snr.item() == pytest.approx(np.abs(holes[finite]).max() / rms)
    assert np.isfinite(entropy.item())
    assert 6.5 < entropy.item() < 7.6


def test_plane_without_signal_gives_no_number():
    cube = np.zeros((1, 1, 3, 32, 32))
    cube[0, 0, 1] = np.nan
    cube[0, 0, 2] = np.random.default_rng(11).standard_normal((32, 32))
    entropy, snr = image_residual_entropy(_img_xds(cube), "residual")
    assert np.isnan(entropy[0, 0, :2]).all() and np.isnan(snr[0, 0, :2]).all()
    assert np.isfinite(entropy[0, 0, 2]) and np.isfinite(snr[0, 0, 2])


def test_single_precision_image():
    rng = np.random.default_rng(12)
    plane = rng.standard_normal((128, 128)).astype(np.float32)
    img_xds = xr.Dataset({"SKY_RESIDUAL": (DIMS, plane[None, None, None])})
    img_xds.attrs["data_groups"] = {"residual": {"sky": "SKY_RESIDUAL"}}
    entropy, _ = image_residual_entropy(img_xds, "residual")
    assert entropy.item() == pytest.approx(
        _direct_entropy(plane.astype(float)), abs=1e-3
    )
