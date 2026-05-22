# run using eg
# python -m pytest astroviper/tests/unit/processing_functions/imaging/test_padding.py

import numpy as np
import unittest

from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
    add_padding,
    remove_padding,
)


class PaddingTest(unittest.TestCase):

    def test_add_padding_basic(self):
        rng = np.random.default_rng(0)
        src = rng.standard_normal((4, 4))
        dst = np.full((8, 8), 99.0)

        returned = add_padding(src, dst)

        self.assertIs(returned, dst)
        np.testing.assert_array_equal(dst[2:6, 2:6], src)
        border_mask = np.ones((8, 8), dtype=bool)
        border_mask[2:6, 2:6] = False
        self.assertTrue(np.all(dst[border_mask] == 0))

    def test_add_padding_no_allocation(self):
        # Confirm dst is mutated in place by checking its data pointer is unchanged.
        src = np.ones((3, 3))
        dst = np.full((7, 7), 5.0)
        dst_ptr = dst.__array_interface__["data"][0]
        add_padding(src, dst)
        self.assertEqual(dst.__array_interface__["data"][0], dst_ptr)

    def test_add_padding_preserves_dst_dtype(self):
        for dtype in (np.float32, np.float64, np.complex128):
            src = np.ones((3, 3))
            dst = np.empty((7, 7), dtype=dtype)
            add_padding(src, dst)
            self.assertEqual(dst.dtype, dtype)

    def test_add_padding_leading_dims(self):
        rng = np.random.default_rng(1)
        src = rng.standard_normal((2, 3, 5, 5))
        dst = np.full((2, 3, 9, 9), 99.0)

        add_padding(src, dst)

        np.testing.assert_array_equal(dst[..., 2:7, 2:7], src)
        self.assertTrue(np.all(dst[..., :2, :] == 0))
        self.assertTrue(np.all(dst[..., 7:, :] == 0))
        self.assertTrue(np.all(dst[..., 2:7, :2] == 0))
        self.assertTrue(np.all(dst[..., 2:7, 7:] == 0))

    def test_add_padding_asymmetric_sizes(self):
        # Odd/even mix exercises the // 2 centring.
        src = np.full((3, 5), 5.0)
        dst = np.empty((8, 9))
        add_padding(src, dst)

        start_r = 8 // 2 - 3 // 2  # 3
        start_c = 9 // 2 - 5 // 2  # 2
        self.assertTrue(
            np.all(dst[start_r : start_r + 3, start_c : start_c + 5] == 5.0)
        )
        self.assertTrue(np.all(dst[:start_r] == 0))
        self.assertTrue(np.all(dst[start_r + 3 :] == 0))
        self.assertTrue(np.all(dst[:, :start_c] == 0))
        self.assertTrue(np.all(dst[:, start_c + 5 :] == 0))

    def test_add_padding_full_size_copies_no_borders(self):
        # When src and dst share last-two-axes shape, dst becomes a copy of src
        # and there are no border strips to zero.
        rng = np.random.default_rng(5)
        src = rng.standard_normal((6, 6))
        dst = np.full((6, 6), 99.0)
        add_padding(src, dst)
        np.testing.assert_array_equal(dst, src)

    def test_round_trip_add_then_remove(self):
        rng = np.random.default_rng(3)
        for src_shape, padded_size in [
            ((4, 4), (8, 8)),
            ((5, 5), (8, 8)),
            ((3, 5), (9, 8)),
            ((6, 7), (13, 11)),
        ]:
            src = rng.standard_normal(src_shape)
            dst = np.empty(padded_size)
            add_padding(src, dst)
            recovered = remove_padding(dst, src_shape)
            np.testing.assert_array_equal(recovered, src)

    def test_round_trip_remove_view_then_add_same_buffer(self):
        # remove_padding returns a view of the centre; passing that view back
        # as `src` with the same buffer as `dst` leaves the centre intact and
        # zeros whatever was outside.
        rng = np.random.default_rng(2)
        padded = rng.standard_normal((10, 10))
        centre_snapshot = padded[3:7, 3:7].copy()

        view = remove_padding(padded, (4, 4))
        add_padding(view, padded)

        np.testing.assert_array_equal(padded[3:7, 3:7], centre_snapshot)
        self.assertTrue(np.all(padded[:3] == 0))
        self.assertTrue(np.all(padded[7:] == 0))
        self.assertTrue(np.all(padded[3:7, :3] == 0))
        self.assertTrue(np.all(padded[3:7, 7:] == 0))

    def test_round_trip_with_leading_dims(self):
        rng = np.random.default_rng(4)
        src = rng.standard_normal((2, 3, 6, 6))
        dst = np.empty((2, 3, 14, 14))
        add_padding(src, dst)
        recovered = remove_padding(dst, (6, 6))
        np.testing.assert_array_equal(recovered, src)

    def test_dst_buffer_reuse_across_iterations(self):
        # The same dst buffer can be reused across many calls without
        # reallocation — the call-site pattern in fft_norm_img_xds.
        rng = np.random.default_rng(6)
        dst = np.empty((8, 8))
        dst_ptr = dst.__array_interface__["data"][0]

        for _ in range(5):
            src = rng.standard_normal((4, 4))
            add_padding(src, dst)
            np.testing.assert_array_equal(dst[2:6, 2:6], src)
            self.assertEqual(dst.__array_interface__["data"][0], dst_ptr)

    def test_dst_can_be_a_slice_view(self):
        # Mirrors the upstream usage where dst is a slice into a larger
        # pre-allocated output buffer (e.g. out_arr[t, f, p]).
        rng = np.random.default_rng(7)
        big = np.full((2, 3, 8, 8), 99.0)
        src = rng.standard_normal((4, 4))
        big_ptr = big.__array_interface__["data"][0]

        add_padding(src, big[1, 2])

        np.testing.assert_array_equal(big[1, 2, 2:6, 2:6], src)
        self.assertTrue(np.all(big[1, 2, :2] == 0))
        self.assertTrue(np.all(big[1, 2, 6:] == 0))
        # Other slices are untouched.
        self.assertTrue(np.all(big[0, 0] == 99.0))
        # No reallocation of the underlying buffer.
        self.assertEqual(big.__array_interface__["data"][0], big_ptr)


if __name__ == "__main__":
    unittest.main()
