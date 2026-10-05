"""Unit tests for ``degrid_visibility_grid_single_field`` (shared primitive).

Coverage:
  * Input contracts: frequency-map length and range, grid dimensionality,
    polarization-axis agreement with the measurement set.
  * A direct call with an identity frequency map reproduces
    ``get_visibility_grid_single_field`` in cube mode.
  * Output allocation (complex128) and data-group registration.
"""

import unittest

import numpy as np

from astroviper.processing_functions.imaging.degrid_visibility_grid import (
    degrid_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.get_visibility_grid import (
    get_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
)
from tests.unit.processing_functions.imaging.degrid_test_datasets import (
    OVERSAMPLING,
    SUPPORT,
    build_datasets,
)


class TestDegridVisibilityGridSingleField(unittest.TestCase):
    def setUp(self):
        self.cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

    # ------------------------------------------------------------------
    # Input contracts
    # ------------------------------------------------------------------
    def test_rejects_frequency_map_with_wrong_length(self):
        """The frequency map must contain one entry per visibility channel."""
        ms_xds, img_xds, _ = build_datasets(n_chan=2)

        with self.assertRaisesRegex(ValueError, "one grid-plane index"):
            degrid_visibility_grid_single_field(
                ms_xds,
                self.cgk,
                img_xds,
                img_xds["SKY_MODEL"].values,
                np.array([0], dtype=np.int64),
            )

    def test_rejects_non_five_dimensional_grid(self):
        """The shared primitive requires a five-dimensional UV grid."""
        ms_xds, img_xds, _ = build_datasets(n_chan=2)

        with self.assertRaisesRegex(ValueError, "grid must have dimensions"):
            degrid_visibility_grid_single_field(
                ms_xds,
                self.cgk,
                img_xds,
                img_xds["SKY_MODEL"].values[0],
                np.array([0, 1], dtype=np.int64),
            )

    def test_rejects_out_of_range_frequency_map(self):
        """Every mapped visibility channel must name an existing grid plane."""
        ms_xds, img_xds, _ = build_datasets(n_chan=2)

        for bad_map in ([0, 2], [-1, 1]):
            with self.subTest(frequency_map=bad_map):
                with self.assertRaisesRegex(ValueError, "outside the UV grid"):
                    degrid_visibility_grid_single_field(
                        ms_xds,
                        self.cgk,
                        img_xds,
                        img_xds["SKY_MODEL"].values,
                        np.array(bad_map, dtype=np.int64),
                    )

    def test_rejects_polarization_axis_mismatch(self):
        """The C++ kernel does not bounds-check the (identity) polarization map,
        so a grid with fewer polarization planes than the MS is rejected."""
        ms_xds, img_xds, _ = build_datasets(n_chan=2, n_pol=2)

        with self.assertRaisesRegex(ValueError, "polarization axis must match"):
            degrid_visibility_grid_single_field(
                ms_xds,
                self.cgk,
                img_xds,
                img_xds["SKY_MODEL"].values[:, :, :1],
                np.array([0, 1], dtype=np.int64),
            )
        # Nothing was written / registered before the check fired.
        self.assertNotIn("VISIBILITY_MODEL", ms_xds)

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    def test_identity_map_matches_cube_wrapper(self):
        """A direct call with an identity frequency map reproduces the cube
        wrapper bit for bit."""
        ms_direct, img_xds, _ = build_datasets(n_chan=3, sky_value=0.0 + 0.0j)
        sky = img_xds["SKY_MODEL"].values
        for channel in range(3):
            sky[:, channel] = channel + 1.0
        ms_wrapper = ms_direct.copy(deep=True)

        get_visibility_grid_single_field(ms_wrapper, self.cgk, img_xds)
        degrid_visibility_grid_single_field(
            ms_direct,
            self.cgk,
            img_xds,
            sky,
            np.arange(3, dtype=np.int64),
        )

        np.testing.assert_array_equal(
            ms_direct["VISIBILITY_MODEL"].values, ms_wrapper["VISIBILITY_MODEL"].values
        )
        for channel in range(3):
            np.testing.assert_allclose(
                ms_direct["VISIBILITY_MODEL"].values[:, :, channel],
                channel + 1.0,
                atol=1e-12,
            )

    def test_allocates_complex128_output_and_registers_group(self):
        """The output is allocated as complex128 (even for a complex64 grid)
        and the output data group is registered with the given description."""
        ms_xds, img_xds, _ = build_datasets(n_chan=2)
        grid = img_xds["SKY_MODEL"].values.astype(np.complex64)

        degrid_visibility_grid_single_field(
            ms_xds,
            self.cgk,
            img_xds,
            grid,
            np.arange(2, dtype=np.int64),
            description="test description",
        )

        out = ms_xds["VISIBILITY_MODEL"]
        self.assertEqual(out.dtype, np.complex128)
        self.assertEqual(out.dims, ms_xds["VISIBILITY"].dims)
        groups = ms_xds.attrs["data_groups"]
        self.assertEqual(groups["model"]["correlated_data"], "VISIBILITY_MODEL")
        self.assertIn("test description", groups["model"]["description"])
        np.testing.assert_allclose(out.values, 2.0 + 0.0j, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
