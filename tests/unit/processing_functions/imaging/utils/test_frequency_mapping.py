"""Unit tests for ``map_visibility_frequencies_to_image``.

Coverage:
  * Identity axes and sparse (partitioned) visibility axes map to the expected
    image planes.
  * Offsets up to half an image channel width map to the nearest channel;
    larger offsets raise.
  * Single-channel image / visibility axes and many-to-one mapping.
  * Invalid coordinates (non-1-D, empty, non-finite) raise.
  * The visibility, PSF and model paths share the same physical map.
"""

import unittest

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.add_visibility_grid import (
    add_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.get_visibility_grid import (
    get_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
)
from astroviper.processing_functions.imaging.make_point_spread_function import (
    add_uv_sampling_grid_single_field,
)
from astroviper.processing_functions.imaging.utils.frequency_mapping import (
    map_visibility_frequencies_to_image,
)
from tests.unit.processing_functions.imaging.degrid_test_datasets import (
    OVERSAMPLING,
    SUPPORT,
    build_datasets,
)


class TestFrequencyMapping(unittest.TestCase):
    """Nearest-channel mapping with a half-channel-width tolerance."""

    def test_maps_identity_and_partitioned_frequency_axes(self):
        """Identity axes and sparse child axes map to the expected planes."""
        cases = (
            ([1.0e9, 1.1e9], [1.0e9, 1.1e9], [0, 1]),
            ([1.1e9, 1.3e9], [1.0e9, 1.1e9, 1.2e9, 1.3e9], [1, 3]),
            ([1.3e9, 1.1e9], [1.0e9, 1.1e9, 1.2e9, 1.3e9], [3, 1]),
        )
        for visibility, image, expected in cases:
            with self.subTest(visibility=visibility, image=image):
                result = map_visibility_frequencies_to_image(visibility, image)
                np.testing.assert_array_equal(result, expected)
                self.assertEqual(result.dtype, np.int64)

    def test_offsets_within_half_channel_width_map_to_nearest(self):
        """An image axis that is not a verbatim copy of the visibility axis
        (regridded / shifted, or a second MS with slightly different channel
        centres) still maps, as long as each visibility channel is within
        half an image channel width of an image channel centre."""
        image = np.linspace(1.0e9, 1.4e9, 5)  # width 1e8
        width = 1.0e8
        for fraction in (1.0e-12, 0.1, 0.25, 0.49):
            with self.subTest(fraction=fraction):
                result = map_visibility_frequencies_to_image(
                    image + fraction * width, image
                )
                np.testing.assert_array_equal(result, np.arange(5))
                result = map_visibility_frequencies_to_image(
                    image - fraction * width, image
                )
                np.testing.assert_array_equal(result, np.arange(5))

    def test_offsets_beyond_half_channel_width_raise(self):
        """A visibility channel more than half a channel width from every
        image channel centre (i.e. outside the band covered by the image
        channels) is rejected with a message naming the channel. Inside the
        band every frequency is within half a width of some channel, so an
        interior offset of 0.51 widths simply maps to the next channel."""
        image = np.linspace(1.0e9, 1.4e9, 5)  # width 1e8
        visibility = image.copy()
        visibility[2] += 0.51e8
        np.testing.assert_array_equal(
            map_visibility_frequencies_to_image(visibility, image), [0, 1, 3, 3, 4]
        )
        # Beyond the last / before the first channel by 0.6 widths.
        with self.assertRaisesRegex(ValueError, "visibility channel indices=\\[1\\]"):
            map_visibility_frequencies_to_image([1.0e9, 1.46e9], image)
        with self.assertRaisesRegex(ValueError, "visibility channel indices=\\[0\\]"):
            map_visibility_frequencies_to_image([0.94e9, 1.1e9], image)
        # Entirely outside the image band.
        with self.assertRaisesRegex(ValueError, "half an image channel width"):
            map_visibility_frequencies_to_image([2.0e9], image)

    def test_many_visibility_channels_map_onto_one_image_channel(self):
        """A coarser image axis receives several visibility channels per
        plane (channel averaging onto the image grid)."""
        image = [1.0e9, 1.2e9]  # width 2e8
        visibility = [0.95e9, 1.0e9, 1.05e9, 1.15e9, 1.2e9, 1.25e9]
        result = map_visibility_frequencies_to_image(visibility, image)
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 1, 1])

    def test_single_channel_axes(self):
        """A single image channel uses the visibility spacing as its width;
        a single channel on both sides always maps onto plane 0."""
        # Single image channel, several visibility channels (width from vis).
        result = map_visibility_frequencies_to_image([1.0e9, 1.1e9], [1.05e9])
        np.testing.assert_array_equal(result, [0, 0])
        with self.assertRaisesRegex(ValueError, "half an image channel width"):
            map_visibility_frequencies_to_image([1.0e9, 1.1e9], [1.2e9])
        # Single channel on both sides: any frequency maps onto plane 0.
        result = map_visibility_frequencies_to_image([1.7e9], [1.0e9])
        np.testing.assert_array_equal(result, [0])

    def test_rejects_invalid_frequency_coordinates(self):
        """Non-1-D, empty, and non-finite coordinates fail."""
        cases = (
            ([[1.0e9]], [1.0e9], "one-dimensional"),
            ([], [1.0e9], "must not be empty"),
            ([1.0e9], [], "must not be empty"),
            ([np.nan], [1.0e9], "finite"),
            ([1.0e9], [np.inf], "finite"),
        )
        for visibility, image, message in cases:
            with self.subTest(visibility=visibility, image=image):
                with self.assertRaisesRegex(ValueError, message):
                    map_visibility_frequencies_to_image(visibility, image)


class TestCubePathsShareFrequencyMap(unittest.TestCase):
    """The visibility, PSF and model paths all use the physical-frequency map."""

    def test_cube_paths_map_partitioned_channels_to_full_image_axis(self):
        visibility_frequencies = [1.1e9, 1.3e9]
        image_frequencies = [1.0e9, 1.1e9, 1.2e9, 1.3e9]
        ms_xds, img_xds, _ = build_datasets(
            visibility_frequencies=visibility_frequencies,
            image_frequencies=image_frequencies,
            sky_value=0.0 + 0.0j,
        )
        # add_visibility_grid / add_uv_sampling_grid read the "residual" group.
        img_xds.attrs["data_groups"]["residual"] = {}
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        model_plane_values = np.array([1.0, 2.0, 3.0, 4.0])
        for channel, value in enumerate(model_plane_values):
            img_xds["SKY_MODEL"].values[:, channel] = value
        get_visibility_grid_single_field(ms_xds, cgk, img_xds)
        expected_model = np.broadcast_to(
            np.array([2.0, 4.0])[np.newaxis, np.newaxis, :],
            ms_xds["VISIBILITY_MODEL"].values[:, :, :, 0].shape,
        )
        np.testing.assert_allclose(
            ms_xds["VISIBILITY_MODEL"].values[:, :, :, 0],
            expected_model,
            atol=1e-5,
        )

        ms_xds["VISIBILITY"].values[...] = 1.0 + 0.0j
        add_visibility_grid_single_field(ms_xds, cgk, img_xds)
        visibility_norm = img_xds["VISIBILITY_NORMALIZATION"].values[0, :, 0]
        np.testing.assert_array_equal(visibility_norm[[0, 2]], 0.0)
        self.assertTrue(np.all(visibility_norm[[1, 3]] > 0.0))

        add_uv_sampling_grid_single_field(ms_xds, cgk, img_xds)
        sampling_norm = img_xds["UV_SAMPLING_NORMALIZATION"].values[0, :, 0]
        np.testing.assert_array_equal(sampling_norm[[0, 2]], 0.0)
        self.assertTrue(np.all(sampling_norm[[1, 3]] > 0.0))

    def test_cube_paths_accept_slightly_offset_image_axis(self):
        """An image axis offset by a tenth of a channel (as a regridded or
        second-MS axis would be) is accepted by all three paths and maps each
        visibility channel onto its nearest image channel."""
        visibility_frequencies = np.linspace(1.0e9, 1.1e9, 3)
        image_frequencies = visibility_frequencies + 0.1 * 0.05e9
        ms_xds, img_xds, _ = build_datasets(
            visibility_frequencies=visibility_frequencies,
            image_frequencies=image_frequencies,
            sky_value=0.0 + 0.0j,
        )
        img_xds.attrs["data_groups"]["residual"] = {}
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

        for channel in range(3):
            img_xds["SKY_MODEL"].values[:, channel] = channel + 1.0
        get_visibility_grid_single_field(ms_xds, cgk, img_xds)
        for channel in range(3):
            np.testing.assert_allclose(
                ms_xds["VISIBILITY_MODEL"].values[:, :, channel],
                channel + 1.0,
                atol=1e-5,
            )

        ms_xds["VISIBILITY"].values[...] = 1.0 + 0.0j
        add_visibility_grid_single_field(ms_xds, cgk, img_xds)
        self.assertTrue(
            np.all(img_xds["VISIBILITY_NORMALIZATION"].values[0, :, 0] > 0.0)
        )
        add_uv_sampling_grid_single_field(ms_xds, cgk, img_xds)
        self.assertTrue(
            np.all(img_xds["UV_SAMPLING_NORMALIZATION"].values[0, :, 0] > 0.0)
        )


@pytest.mark.parametrize(
    "visibility,image,expected",
    [
        ([1.3e9, 1.1e9], [1.0e9, 1.1e9, 1.2e9, 1.3e9], [3, 1]),
        ([1.0e9 + 0.0005], [1.0e9], [0]),
    ],
)
def test_exact_matching_preserves_unique_channels(visibility, image, expected):
    np.testing.assert_array_equal(
        map_visibility_frequencies_to_image(visibility, image, matching="exact"),
        expected,
    )


@pytest.mark.parametrize(
    "visibility,image",
    [
        ([1.01e9, 1.11e9], [1.0e9, 1.1e9]),
        ([1.0e9, 1.0e9], [1.0e9, 1.1e9]),
        ([1.0e9], [1.0e9, 1.0e9]),
        ([1.7e9], [1.0e9]),
    ],
)
def test_exact_rejects_cases_accepted_by_nearest(visibility, image):
    map_visibility_frequencies_to_image(visibility, image, matching="nearest")
    with pytest.raises(ValueError):
        map_visibility_frequencies_to_image(visibility, image, matching="exact")


def test_exact_tolerance_is_configurable():
    with pytest.raises(ValueError):
        map_visibility_frequencies_to_image([1.0e9 + 1], [1.0e9], matching="exact")
    np.testing.assert_array_equal(
        map_visibility_frequencies_to_image(
            [1.0e9 + 1], [1.0e9], matching="exact", rtol=0, atol=2
        ),
        [0],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"matching": "invalid"},
        {"matching": "exact", "rtol": -1},
        {"matching": "exact", "atol": np.inf},
    ],
)
def test_invalid_matching_policy_or_tolerance(kwargs):
    with pytest.raises(ValueError):
        map_visibility_frequencies_to_image([1.0e9], [1.0e9], **kwargs)


@pytest.mark.parametrize("offset", [0.0, 1.0e6])
def test_mvc_requires_exact_image_frequencies(offset):
    from astroviper.processing_functions.imaging.add_visibility_grid_continuum_mvc import (
        add_visibility_grid_mvc_single_field,
    )

    ms, image, _ = build_datasets(n_chan=2, n_pol=1)
    ms["VISIBILITY"].values[...] = 1.0 + 0.0j
    image = image.assign_coords(
        time=[0.0], polarization=["I"], frequency=image.frequency.values + offset
    )
    image.attrs["data_groups"]["residual"] = {}
    cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)
    if offset:
        with pytest.raises(ValueError, match="exactly one image frequency"):
            add_visibility_grid_mvc_single_field(ms, cgk, image)
    else:
        add_visibility_grid_mvc_single_field(ms, cgk, image)
        assert np.all(image.VISIBILITY_NORMALIZATION.values > 0)


@pytest.mark.parametrize("matching", ["nearest", "exact"])
@pytest.mark.parametrize("offset", [0.0, 1.0e6])
def test_weighting_uses_selected_frequency_policy(matching, offset):
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )

    ms, image, _ = build_datasets(n_chan=2)
    ms["FLAG"] = xr.zeros_like(ms.WEIGHT_IMAGING, dtype=bool)
    ms.attrs["data_groups"]["base"].update(weight="WEIGHT_IMAGING", flag="FLAG")
    ps = xr.DataTree.from_dict({"ms": ms})
    image = image.assign_coords(frequency=image.frequency.values + offset)

    def run():
        calculate_imaging_weights(
            ps,
            image,
            {
                "weighting": "briggs",
                "robust": 0.5,
                "casa_weighting_implementation": True,
            },
            frequency_matching=matching,
            truncate_uv_cells=True,
            ms_data_group_out_modified={"weight_imaging": "WEIGHT_RESULT"},
        )

    if offset and matching == "exact":
        with pytest.raises(ValueError, match="exactly one image frequency"):
            run()
    else:
        run()
        result = ps["ms"]["WEIGHT_RESULT"].values
        assert np.all(np.isfinite(result)) and np.all(result > 0)


if __name__ == "__main__":
    unittest.main()
