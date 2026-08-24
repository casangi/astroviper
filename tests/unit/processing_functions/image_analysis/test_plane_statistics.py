"""Unit tests for the per-plane (l, m) image statistics."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.plane_statistics import (
    DEFAULT_STATISTICS_VARIABLES,
    PLANE_STATISTIC_NAMES,
    calculate_plane_statistics,
    concatenate_plane_statistics,
    plane_statistics_to_dataframe,
)

DIMS = ("time", "frequency", "polarization", "l", "m")


def _make_image(
    n_freq=3,
    n_pol=2,
    n_lm=16,
    with_mask=True,
    with_primary_beam=False,
    dtype=np.float32,
):
    rng = np.random.default_rng(1234)
    shape = (1, n_freq, n_pol, n_lm, n_lm)
    residual = rng.normal(size=shape).astype(dtype)
    residual[0, 0, 0, 0, 0] = np.nan  # a NaN pixel must be ignored
    residual[0, 1, 1, 3, 3] = -10.0  # signed peak (negative)
    restored = (residual + 5.0).astype(dtype)
    data_vars = {
        "SKY_RESIDUAL": (DIMS, residual),
        "SKY_RESTORED": (DIMS, restored),
    }
    data_groups = {
        "residual": {"sky": "SKY_RESIDUAL"},
        "restored": {"sky": "SKY_RESTORED"},
    }
    if with_mask:
        mask = np.zeros(shape, dtype=bool)
        mask[..., 4:12, 4:12] = True
        data_vars["MASK"] = (DIMS, mask)
        data_groups["residual"]["mask"] = "MASK"
    if with_primary_beam:
        primary_beam = np.zeros(shape)
        primary_beam[..., 2:14, 2:14] = 0.5  # > the 0.2 default limit
        data_vars["PRIMARY_BEAM"] = (DIMS, primary_beam)
        data_groups["residual"]["primary_beam"] = "PRIMARY_BEAM"
    return xr.Dataset(
        data_vars,
        coords={
            "time": [0.0],
            "frequency": 1.0e9 + 1.0e8 * np.arange(n_freq),
            "polarization": ["I", "Q"][:n_pol],
            "l": np.arange(n_lm),
            "m": np.arange(n_lm),
        },
        attrs={"data_groups": data_groups},
    )


def test_statistics_match_numpy_and_ignore_nans():
    img_xds = _make_image()
    stats = calculate_plane_statistics(img_xds)
    assert set(stats) == {"sky_residual", "sky_restored"}

    residual = img_xds["SKY_RESIDUAL"].values.astype(np.float64)
    mask = img_xds["MASK"].values
    ds = stats["sky_residual"]
    assert ds.attrs == {
        "data_variable": "SKY_RESIDUAL",
        "mask_present": True,
        "mask_source": "MASK",
    }
    assert dict(ds.sizes) == {"time": 1, "frequency": 3, "polarization": 2}
    assert set(ds.data_vars) == set(PLANE_STATISTIC_NAMES) | {
        f"{name}_masked" for name in PLANE_STATISTIC_NAMES
    }

    for f in range(3):
        for p in range(2):
            plane = residual[0, f, p]
            valid = plane[~np.isnan(plane)]
            inside = plane[~np.isnan(plane) & mask[0, f, p]]
            sel = ds.isel(time=0, frequency=f, polarization=p)
            assert sel["mean"] == pytest.approx(valid.mean())
            assert sel["median"] == pytest.approx(np.median(valid))
            assert sel["max"] == pytest.approx(valid.max())
            assert sel["min"] == pytest.approx(valid.min())
            assert sel["peak"] == pytest.approx(valid[np.abs(valid).argmax()])
            assert sel["sum"] == pytest.approx(valid.sum())
            assert sel["rms"] == pytest.approx(np.sqrt(np.mean(valid**2)))
            assert sel["std"] == pytest.approx(valid.std())
            assert sel["mad_sigma"] == pytest.approx(
                1.4826 * np.median(np.abs(valid - np.median(valid)))
            )
            assert sel["n_pixels"] == valid.size
            assert sel["mean_masked"] == pytest.approx(inside.mean())
            assert sel["median_masked"] == pytest.approx(np.median(inside))
            assert sel["max_masked"] == pytest.approx(inside.max())
            assert sel["peak_masked"] == pytest.approx(inside[np.abs(inside).argmax()])
            assert sel["n_pixels_masked"] == inside.size

    # The NaN pixel is excluded from the count; the negative peak keeps its sign.
    assert ds["n_pixels"].values[0, 0, 0] == 16 * 16 - 1
    assert ds["peak"].values[0, 1, 1] == pytest.approx(-10.0)
    assert ds["min"].values[0, 1, 1] == pytest.approx(-10.0)
    # The restored image is the residual + 5, so its mean shifts by 5.
    assert stats["sky_restored"]["mean"].values == pytest.approx(
        ds["mean"].values + 5.0, abs=1e-5
    )


def test_no_mask_and_no_primary_beam_gives_nan_masked_statistics():
    img_xds = _make_image(with_mask=False)
    ds = calculate_plane_statistics(img_xds)["sky_residual"]
    assert ds.attrs["mask_present"] is False
    assert ds.attrs["mask_source"] is None
    assert np.isfinite(ds["mean"].values).all()
    for name in PLANE_STATISTIC_NAMES:
        if name == "n_pixels":
            assert (ds["n_pixels_masked"].values == 0).all()
        else:
            assert np.isnan(ds[f"{name}_masked"].values).all()


def test_primary_beam_fallback_mask_when_no_clean_mask():
    """No MASK (a niter=0 run): masked stats fall back to the valid-sky area
    PRIMARY_BEAM > primary_beam_limit."""
    img_xds = _make_image(with_mask=False, with_primary_beam=True)
    ds = calculate_plane_statistics(img_xds)["sky_residual"]
    assert ds.attrs["mask_present"] is True
    assert ds.attrs["mask_source"] == "PRIMARY_BEAM > 0.2"
    sky = img_xds["SKY_RESIDUAL"].values.astype(np.float64)
    inside = sky[0, 1, 0, 2:14, 2:14]
    sel = ds.isel(time=0, frequency=1, polarization=0)
    assert sel["n_pixels_masked"] == 12 * 12
    assert sel["mean_masked"] == pytest.approx(np.nanmean(inside))
    assert sel["max_masked"] == pytest.approx(np.nanmax(inside))
    # A stricter limit shrinks the area; limit=None disables the fallback.
    strict = calculate_plane_statistics(img_xds, primary_beam_limit=0.6)["sky_residual"]
    assert (strict["n_pixels_masked"].values == 0).all()
    off = calculate_plane_statistics(img_xds, primary_beam_limit=None)["sky_residual"]
    assert off.attrs["mask_present"] is False
    assert np.isnan(off["mean_masked"].values).all()


def test_clean_mask_wins_over_primary_beam_fallback():
    img_xds = _make_image(with_mask=True, with_primary_beam=True)
    ds = calculate_plane_statistics(img_xds)["sky_residual"]
    assert ds.attrs["mask_source"] == "MASK"
    assert (ds["n_pixels_masked"].values == 8 * 8).all()  # the MASK box, not the PB box


def test_all_nan_plane_is_nan_with_zero_pixels():
    img_xds = _make_image(with_mask=False)
    img_xds["SKY_RESIDUAL"].values[0, 2, 0] = np.nan
    ds = calculate_plane_statistics(img_xds)["sky_residual"]
    sel = ds.isel(time=0, frequency=2, polarization=0)
    assert np.isnan(sel["mean"]) and np.isnan(sel["peak"])
    assert sel["n_pixels"] == 0
    assert np.isfinite(ds["mean"].values[0, 2, 1])


def test_explicit_variables_and_mask_name_and_missing_variables_skipped():
    img_xds = _make_image()
    img_xds["MY_MASK"] = img_xds["MASK"].copy()
    stats = calculate_plane_statistics(
        img_xds,
        image_data_variables=["sky_restored", "sky_model", "sky_dirty"],
        mask_name="MY_MASK",
    )
    assert list(stats) == ["sky_restored"]  # sky_model / sky_dirty absent
    assert stats["sky_restored"].attrs["mask_present"] is True
    assert DEFAULT_STATISTICS_VARIABLES[:2] == ("sky_residual", "sky_restored")


def test_concatenate_orders_by_frequency_and_skips_empty_chunks():
    img_xds = _make_image(n_freq=5)
    full = calculate_plane_statistics(img_xds)
    chunk_a = calculate_plane_statistics(img_xds.isel(frequency=[3, 4]))
    chunk_b = calculate_plane_statistics(img_xds.isel(frequency=[0]))
    chunk_c = calculate_plane_statistics(img_xds.isel(frequency=[1, 2]))
    merged = concatenate_plane_statistics([chunk_a, {}, chunk_b, chunk_c])
    for key in full:
        xr.testing.assert_allclose(merged[key], full[key])
    # Composes under tree reduction: a merged result is a valid input again.
    partial = concatenate_plane_statistics([chunk_a, chunk_c])
    xr.testing.assert_allclose(
        concatenate_plane_statistics([partial, chunk_b])["sky_residual"],
        full["sky_residual"],
    )
    assert concatenate_plane_statistics([{}, {}]) == {}
    # A single chunk is returned as is (no concat needed).
    assert (
        concatenate_plane_statistics([chunk_b])["sky_residual"]
        is (chunk_b["sky_residual"])
    )


def test_to_dataframe_long_format():
    img_xds = _make_image(n_freq=2)
    stats = calculate_plane_statistics(img_xds)
    df = plane_statistics_to_dataframe(stats)
    assert list(df.columns) == [
        "image_variable",
        "statistic",
        "time",
        "frequency",
        "polarization",
        "value",
    ]
    n_stats = 2 * len(PLANE_STATISTIC_NAMES)
    assert len(df) == 2 * n_stats * 2 * 2  # variables * stats * freq * pol
    row = df[
        (df.image_variable == "sky_residual")
        & (df.statistic == "mean")
        & (df.polarization == "Q")
        & (df.frequency == 1.1e9)
    ]
    assert len(row) == 1
    assert row["value"].iloc[0] == pytest.approx(
        stats["sky_residual"]["mean"].values[0, 1, 1]
    )
    assert plane_statistics_to_dataframe({}).empty
