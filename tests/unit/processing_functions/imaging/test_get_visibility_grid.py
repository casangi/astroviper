"""Unit tests for `get_visibility_grid_single_field` (prolate-spheroidal degrid).

Coverage:
  * Smoke: uniform UV model produces constant visibilities.
  * Output allocation and data-group registration on first call.
  * Channel-mode `cube` vs `continuum`.
  * NaN UVW samples and out-of-bounds samples are skipped.
  * `overwrite=False` guards against accidental overwrites on a second call.
  * Agreement with a direct call to `prolate_spheroidal_degrid_jit`.
"""

import unittest

import numpy as np
import xarray as xr

# Registers the `xr_img` accessor used by the function under test.
import xradio.image.image_xds  # noqa: F401

from astroviper.processing_functions.imaging.get_visibility_grid import (
    get_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_degrid import (
    prolate_spheroidal_degrid_jit,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
)

SUPPORT = 7
OVERSAMPLING = 100


def _build_datasets(
    n_l=80,
    n_m=80,
    n_time=1,
    n_baseline=16,
    n_chan=2,
    n_pol=2,
    fft_padding=1.2,
    delta=2.0e-5,
    uv_extent=20.0,
    sky_value=2.0 + 0.0j,
    seed=0,
):
    """Build a minimal (ms_xds, img_xds, n_uv) triple for the degridder.

    `img_xds` holds a UV-domain model grid (not a sky image) with shape
    `(time, frequency, polarization, u, v)`.
    """
    rng = np.random.default_rng(seed)
    n_uv = (fft_padding * np.array([n_l, n_m])).astype(int)

    # l/m coordinates centred on zero so `get_lm_cell_size` returns `delta`.
    l_coord = (np.arange(n_l) - n_l / 2) * delta
    m_coord = (np.arange(n_m) - n_m / 2) * delta
    freq = np.linspace(1.0e9, 1.1e9, n_chan)

    uvw = np.concatenate(
        [
            rng.uniform(-uv_extent, uv_extent, (n_time, n_baseline, 2)),
            np.zeros((n_time, n_baseline, 1)),
        ],
        axis=-1,
    )

    ms_xds = xr.Dataset(
        data_vars={
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.zeros((n_time, n_baseline, n_chan, n_pol), dtype=np.complex128),
            ),
            "UVW": (("time", "baseline_id", "uvw_label"), uvw),
            "WEIGHT_IMAGING": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.ones((n_time, n_baseline, n_chan, n_pol)),
            ),
        },
        coords={"frequency": freq},
    )
    ms_xds.attrs["data_groups"] = {
        "base": {
            "correlated_data": "VISIBILITY",
            "uvw": "UVW",
            "weight_imaging": "WEIGHT_IMAGING",
        }
    }

    # UV model grid: shape (m_time, m_chan, m_pol, n_u, n_v)
    sky_model = np.full(
        (n_time, n_chan, n_pol, int(n_uv[0]), int(n_uv[1])),
        sky_value,
        dtype=np.complex128,
    )
    img_xds = xr.Dataset(
        data_vars={
            "SKY_MODEL": (
                ("time", "frequency", "polarization", "u", "v"),
                sky_model,
            ),
        },
        coords={"l": l_coord, "m": m_coord, "frequency": freq},
    )
    img_xds.attrs["type"] = "image_dataset"
    # get_visibility_grid_single_field degrids the image-side "visibility" uv
    # grid (default input data group "model") into ms model visibilities.
    img_xds.attrs["data_groups"] = {
        "model": {"visibility": "SKY_MODEL"},
    }

    return ms_xds, img_xds, n_uv


class TestGetVisibilityGridSingleField(unittest.TestCase):
    # ------------------------------------------------------------------
    # Smoke / analytical
    # ------------------------------------------------------------------
    def test_uniform_grid_yields_constant_visibilities(self):
        """A uniform UV model degrids to a constant: sum(k*c)/sum(k) == c."""
        sky_value = 2.0 + 0.0j
        ms_xds, img_xds, _ = _build_datasets(sky_value=sky_value)
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds)

        out = ms_xds["VISIBILITY_MODEL"].values
        # The degridder allocates the model visibilities in double precision
        # (visibilities stay double even when the image-domain grid is single).
        self.assertEqual(out.dtype, np.complex128)
        self.assertEqual(out.shape, (1, 16, 2, 2))
        np.testing.assert_allclose(out, np.full_like(out, sky_value), rtol=0, atol=1e-5)

    def test_zero_grid_yields_zero_visibilities(self):
        """An all-zero UV model produces all-zero visibilities."""
        ms_xds, img_xds, _ = _build_datasets(sky_value=0.0 + 0.0j)
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds)

        out = ms_xds["VISIBILITY_MODEL"].values
        self.assertTrue(np.all(out == 0.0))

    # ------------------------------------------------------------------
    # Dataset plumbing
    # ------------------------------------------------------------------
    def test_allocates_output_variable(self):
        """First call creates the output data variable with the right shape/dtype."""
        ms_xds, img_xds, _ = _build_datasets()
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        self.assertNotIn("VISIBILITY_MODEL", ms_xds)
        get_visibility_grid_single_field(ms_xds, cgk, img_xds)

        self.assertIn("VISIBILITY_MODEL", ms_xds)
        self.assertEqual(
            ms_xds["VISIBILITY_MODEL"].dims,
            ms_xds["VISIBILITY"].dims,
        )
        self.assertEqual(
            ms_xds["VISIBILITY_MODEL"].shape,
            ms_xds["VISIBILITY"].shape,
        )
        self.assertEqual(ms_xds["VISIBILITY_MODEL"].dtype, np.complex128)

    def test_registers_output_data_group(self):
        """The `model` data group is added with the correct role mappings."""
        ms_xds, img_xds, _ = _build_datasets()
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds)

        self.assertIn("model", ms_xds.attrs["data_groups"])
        model_group = ms_xds.attrs["data_groups"]["model"]
        self.assertEqual(model_group["correlated_data"], "VISIBILITY_MODEL")
        # Roles inherited from `base`.
        self.assertEqual(model_group["uvw"], "UVW")
        self.assertEqual(model_group["weight_imaging"], "WEIGHT_IMAGING")
        # Audit-trail fields added by `modify_data_groups_xds`.
        self.assertIn("date", model_group)
        self.assertIn("description", model_group)

    def test_overwrite_false_second_call_raises(self):
        """With overwrite=False, re-running on a dataset that already has the
        output variable raises (guard against accidental clobber)."""
        ms_xds, img_xds, _ = _build_datasets()
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds)
        with self.assertRaises(AssertionError):
            get_visibility_grid_single_field(ms_xds, cgk, img_xds, overwrite=False)

    # ------------------------------------------------------------------
    # Skip behaviour
    # ------------------------------------------------------------------
    def test_nan_uvw_sample_stays_zero(self):
        """A visibility with NaN UVW is skipped; its output stays at the
        initial zero value."""
        ms_xds, img_xds, _ = _build_datasets(sky_value=3.0 + 0.0j)
        # Set the first baseline's UVW to NaN for all time steps.
        ms_xds["UVW"].values[:, 0, :] = np.nan
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds)

        out = ms_xds["VISIBILITY_MODEL"].values
        # Skipped baseline retains the zero initial value.
        np.testing.assert_array_equal(out[:, 0], np.zeros_like(out[:, 0]))
        # Other baselines still get the constant-grid value.
        np.testing.assert_allclose(
            out[:, 1:], np.full_like(out[:, 1:], 3.0 + 0.0j), atol=1e-12
        )

    def test_out_of_bounds_uvw_stays_zero(self):
        """A UVW coordinate whose kernel support falls outside the grid is
        skipped (output stays at the zero initial value)."""
        ms_xds, img_xds, _ = _build_datasets(sky_value=1.0 + 0.0j, n_baseline=1)
        # 1e12 m baseline maps far outside the padded 96x96 grid.
        ms_xds["UVW"].values[:] = 1.0e12
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds)

        out = ms_xds["VISIBILITY_MODEL"].values
        self.assertTrue(np.all(out == 0.0))

    # ------------------------------------------------------------------
    # Channel modes
    # ------------------------------------------------------------------
    def test_continuum_mode_uses_single_image_channel(self):
        """In continuum mode every visibility channel samples image chan 0."""
        n_chan = 3
        ms_xds, img_xds, n_uv = _build_datasets(n_chan=n_chan, sky_value=0.0 + 0.0j)
        # Only image channel 0 has a non-zero model; others are zero.
        sky = img_xds["SKY_MODEL"].values
        sky[:, 0] = 5.0 + 0.0j
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds, chan_mode="continuum")

        out = ms_xds["VISIBILITY_MODEL"].values
        # All vis channels read from image chan 0 => all equal 5.0.
        np.testing.assert_allclose(out, np.full_like(out, 5.0 + 0.0j), atol=1e-12)

    def test_cube_mode_maps_chan_to_chan(self):
        """Cube mode maps vis chan i to image chan i."""
        n_chan = 3
        ms_xds, img_xds, _ = _build_datasets(n_chan=n_chan, sky_value=0.0 + 0.0j)
        # Give each image channel a distinct value.
        sky = img_xds["SKY_MODEL"].values
        chan_values = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        for c in range(n_chan):
            sky[:, c] = chan_values[c]
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        get_visibility_grid_single_field(ms_xds, cgk, img_xds, chan_mode="cube")

        out = ms_xds["VISIBILITY_MODEL"].values
        # For each vis channel the output must match that image channel.
        for c in range(n_chan):
            np.testing.assert_allclose(
                out[:, :, c],
                np.full_like(out[:, :, c], chan_values[c]),
                atol=1e-12,
            )

    # ------------------------------------------------------------------
    # Oracle: direct call to prolate_spheroidal_degrid_jit
    # ------------------------------------------------------------------
    def test_matches_direct_degrid_jit_call(self):
        """Degridding via the wrapper must match a direct jit call."""
        ms_xds, img_xds, n_uv = _build_datasets(sky_value=0.0 + 0.0j, seed=42)
        # Populate the UV grid with deterministic-but-nontrivial data.
        rng = np.random.default_rng(7)
        sky = img_xds["SKY_MODEL"].values
        sky[...] = (
            rng.standard_normal(sky.shape) + 1j * rng.standard_normal(sky.shape)
        ).astype(np.complex128)
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT).astype(
            np.float64
        )

        # --- Wrapper call ---
        get_visibility_grid_single_field(ms_xds, cgk, img_xds)
        wrapper_out = ms_xds["VISIBILITY_MODEL"].values.copy()

        # --- Direct jit call with the same inputs ---
        n_time = ms_xds.sizes["time"]
        n_chan = ms_xds.sizes["frequency"]
        n_pol = ms_xds.sizes["polarization"]
        frequency_map = np.arange(n_chan, dtype=np.int64)
        time_map = np.zeros(n_time, dtype=np.int64)
        pol_map = np.arange(n_pol, dtype=np.int64)
        delta_lm = img_xds.xr_img.get_lm_cell_size()

        direct_vis = np.zeros(ms_xds["VISIBILITY"].shape, dtype=np.complex128)
        prolate_spheroidal_degrid_jit(
            sky,
            direct_vis,
            ms_xds["UVW"].values,
            ms_xds["frequency"].values,
            frequency_map,
            time_map,
            pol_map,
            cgk,
            n_uv.astype(np.int64),
            delta_lm,
            support=SUPPORT,
            oversampling=OVERSAMPLING,
        )

        # The wrapper degrids in single precision; the direct oracle call here
        # accumulates in complex128, so compare at single-precision tolerance.
        np.testing.assert_allclose(wrapper_out, direct_vis, rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
