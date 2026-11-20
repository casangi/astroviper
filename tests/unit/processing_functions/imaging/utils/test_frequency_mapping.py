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
    calculate_frequency_boundaries,
    map_visibility_frequencies_to_image,
    select_frequency_boundaries,
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

    def test_same_map_as_graphviper(self):
        """The map is graphviper's ``map_to_nearest_coordinate``, by which the
        graph selects the visibility channels of a task, also on an irregular
        image axis, where a visibility channel between two image channels
        always maps onto the nearer one."""
        from graphviper.graph_tools.coordinate_utils import (
            map_to_nearest_coordinate,
        )

        image = np.array([1.0e9, 1.01e9, 1.1e9, 1.4e9])
        visibility = np.array([0.996e9, 1.004e9, 1.05e9, 1.06e9, 1.3e9, 1.5e9])
        expected = [0, 0, 1, 2, 3, 3]
        np.testing.assert_array_equal(
            map_visibility_frequencies_to_image(visibility, image), expected
        )
        np.testing.assert_array_equal(
            map_to_nearest_coordinate(visibility, image), expected
        )

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


class TestPartitionedFrequencyMapping(unittest.TestCase):
    """Task boundaries must retain the full image axis's mapping policy."""

    def test_reported_mvc_single_channel_task(self):
        full = 4107516045.6904545 + np.arange(8) * 533950857.6451702
        observed = 4380616641.112875 + np.arange(13) * 8001126.285138
        with self.assertRaises(ValueError):
            map_visibility_frequencies_to_image(observed, full[1:2])
        np.testing.assert_array_equal(
            map_visibility_frequencies_to_image(
                observed, full[1:2], full_image_frequencies=full
            ),
            np.zeros(13, dtype=np.int64),
        )

    def test_partition_invariance_and_global_ties(self):
        from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

        for full in (
            np.arange(8, dtype=float) * 10,
            np.array([0.0, 2.0, 8.0, 10.0, 30.0, 60.0, 65.0, 100.0]),
            np.array([100.0, 65.0, 60.0, 30.0, 10.0, 8.0, 2.0, 0.0]),
        ):
            sorted_axis = np.sort(full)
            observed = np.concatenate(
                (
                    full,
                    (sorted_axis[1:] + sorted_axis[:-1]) / 2,
                    full + 1e-6,
                    full - 1e-6,
                )
            )
            expected = map_to_nearest_coordinate(observed, full)
            for chunks in (1, 2, 4, 8):
                actual = np.full(observed.size, -1)
                for indices in np.array_split(np.arange(8), chunks):
                    selected = np.isin(expected, indices)
                    local = map_visibility_frequencies_to_image(
                        observed[selected], full[indices], full_image_frequencies=full
                    )
                    actual[selected] = indices[local]
                np.testing.assert_array_equal(actual, expected)

    def test_wrong_task_and_outside_global_band_remain_errors(self):
        with self.assertRaisesRegex(ValueError, "outside this task"):
            map_visibility_frequencies_to_image(
                [20.0], [10.0], full_image_frequencies=[0.0, 10.0, 20.0]
            )
        with self.assertRaisesRegex(ValueError, "half an image channel"):
            map_visibility_frequencies_to_image(
                [26.0], [20.0], full_image_frequencies=[0.0, 10.0, 20.0]
            )

    def test_invalid_full_axis_or_subset(self):
        for full in ([], [[1.0, 2.0]], [np.nan], [np.inf]):
            with self.assertRaisesRegex(ValueError, "Full image frequencies"):
                map_visibility_frequencies_to_image(
                    [1.0], [1.0], full_image_frequencies=full
                )
        for local, full in (([1.0], [1.0, 1.0]), ([1.0, 1.0], [1.0, 2.0])):
            with self.assertRaisesRegex(ValueError, "unique"):
                map_visibility_frequencies_to_image(
                    [1.0], local, full_image_frequencies=full
                )
        with self.assertRaisesRegex(ValueError, "subset"):
            map_visibility_frequencies_to_image(
                [1.0], [1.1], full_image_frequencies=[1.0, 2.0]
            )

    def test_true_single_channel_preserves_existing_behavior(self):
        for observed in ([12.0], [9.0, 11.0]):
            np.testing.assert_array_equal(
                map_visibility_frequencies_to_image(observed, [10.0]),
                map_visibility_frequencies_to_image(
                    observed, [10.0], full_image_frequencies=[10.0]
                ),
            )


@pytest.mark.parametrize("context_kind", ["legacy", "boundaries"])
def test_single_plane_gridding_and_prediction_preserve_original_boundaries(
    context_kind,
):
    """The context reaches PSF, visibility and prediction kernels after slicing."""
    full = np.array([1.0e9, 1.2e9, 1.4e9])
    ms, image, _ = build_datasets(
        visibility_frequencies=[1.15e9, 1.16e9, 1.17e9],
        image_frequencies=full,
        sky_value=2.0 + 0j,
    )
    if context_kind == "legacy":
        image.frequency.attrs["full_image_frequencies"] = full.tolist()
    else:
        image.frequency.attrs["frequency_boundaries"] = calculate_frequency_boundaries(
            full
        )
    image = image.isel(frequency=slice(1, 2))
    image.attrs["data_groups"]["residual"] = {}
    cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)
    get_visibility_grid_single_field(ms, cgk, image)
    np.testing.assert_allclose(ms.VISIBILITY_MODEL.values, 2.0, atol=1e-5)
    ms.VISIBILITY.values[:] = 1.0
    add_visibility_grid_single_field(ms, cgk, image)
    assert np.all(image.VISIBILITY_NORMALIZATION.values > 0)
    add_uv_sampling_grid_single_field(ms, cgk, image)
    assert np.all(image.UV_SAMPLING_NORMALIZATION.values > 0)


if __name__ == "__main__":
    unittest.main()


def test_precomputed_boundaries_match_graph_at_adjacent_floats():
    """Closed float64 intervals preserve actual graph arithmetic, not rounded midpoints."""
    import json

    from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

    rng = np.random.default_rng(456)
    for axis in (
        np.array([0.0, 2.0, 8.0, 10.0, 30.0, 60.0, 65.0, 100.0]),
        np.array([100.0, 65.0, 60.0, 30.0, 10.0, 8.0, 2.0, 0.0]),
        np.array([30.0, 0.0, 100.0, 2.0, 65.0, 8.0, 60.0, 10.0]),
        4107516045.6904545 + np.arange(8) * 533950857.6451702,
        rng.uniform(1.0e9, 2.0e9, 40),
        np.array([-1.0e9, np.nextafter(1.0e9, np.inf)]),
    ):
        boundaries = json.loads(json.dumps(calculate_frequency_boundaries(axis)))
        endpoints = np.asarray(list(boundaries.values())).ravel()
        midpoints = (np.sort(axis)[:-1] + np.sort(axis)[1:]) / 2
        probes = np.concatenate(
            (
                axis,
                endpoints,
                midpoints,
                np.nextafter(endpoints, -np.inf),
                np.nextafter(endpoints, np.inf),
                np.nextafter(midpoints, -np.inf),
                np.nextafter(midpoints, np.inf),
                rng.uniform(axis.min() - np.ptp(axis), axis.max() + np.ptp(axis), 100),
            )
        )
        expected = map_to_nearest_coordinate(probes, axis)
        for indices in (
            np.arange(axis.size),
            np.arange(axis.size)[::2],
            np.array([0]),
            np.array([axis.size - 1]),
        ):
            local_axis = axis[indices]
            context = select_frequency_boundaries(boundaries, local_axis)
            assert len(context) == len(indices)
            for value, owner in zip(probes, expected, strict=True):
                if owner in indices:
                    actual = map_visibility_frequencies_to_image(
                        [value], local_axis, frequency_boundaries=context
                    )
                    assert indices[actual[0]] == owner
                else:
                    with np.testing.assert_raises_regex(
                        ValueError, "outside this task"
                    ):
                        map_visibility_frequencies_to_image(
                            [value], local_axis, frequency_boundaries=context
                        )


def test_boundary_preparation_rejects_invalid_axes_and_missing_channels():
    for axis in ([], [[1.0, 2.0]], [np.inf], [1.0, 1.0]):
        with np.testing.assert_raises(ValueError):
            calculate_frequency_boundaries(axis)
    with np.testing.assert_raises_regex(ValueError, "subset"):
        select_frequency_boundaries(calculate_frequency_boundaries([1.0, 2.0]), [3.0])


def test_precomputed_boundary_payload_scales_with_owned_channels():
    import pickle

    lengths = []
    for count in (100, 10000):
        axis = np.arange(count, dtype=float)
        boundaries = calculate_frequency_boundaries(axis)
        local = select_frequency_boundaries(boundaries, axis[50:51])
        assert len(local) == 1
        lengths.append(len(pickle.dumps(local)))
    assert lengths[0] == lengths[1]


@pytest.mark.parametrize(
    "observed",
    [
        [1.0e9, 1.1e9, 1.2e9, 2.0e9],
        [2.0e9, 1.2e9, 1.1e9, 1.0e9],
        [1.35e9, 1.45e9, 1.55e9, 1.65e9],
        [1.7e9],
        [1.5e9, 1.5e9],
    ],
)
def test_single_image_channel_uses_full_observation_boundaries(observed):
    import json

    from graphviper.graph_tools.coordinate_utils import map_to_nearest_coordinate

    observed = np.asarray(observed)
    image = [1.5e9]
    boundaries = json.loads(
        json.dumps(calculate_frequency_boundaries(image, observed), allow_nan=False)
    )
    selected = observed[map_to_nearest_coordinate(observed, image) == 0]
    if selected.size:
        np.testing.assert_array_equal(
            map_visibility_frequencies_to_image(
                selected, image, frequency_boundaries=boundaries
            ),
            np.zeros(selected.size, dtype=np.int64),
        )
    lo, hi = boundaries[str(image[0])]
    tolerance = 0.5 * np.diff(np.sort(observed)).max() if len(observed) > 1 else np.inf
    with np.errstate(over="ignore"):
        probes = (lo, hi, np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))
    for probe in probes:
        if not np.isfinite(probe):
            continue
        if abs(probe - image[0]) <= tolerance:
            np.testing.assert_array_equal(
                map_visibility_frequencies_to_image(
                    [probe], image, frequency_boundaries=boundaries
                ),
                [0],
            )
        else:
            with pytest.raises(ValueError, match="outside this task"):
                map_visibility_frequencies_to_image(
                    [probe], image, frequency_boundaries=boundaries
                )


def test_single_channel_observations_keep_distinct_intervals_after_slicing():
    import xarray as xr

    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        attach_observation_frequency_boundaries,
        calculate_observation_frequency_boundaries,
        get_frequency_boundaries,
    )

    full = {
        "wide": xr.Dataset(coords={"frequency": [1.0e9, 1.1e9, 1.2e9, 2.0e9]}),
        "narrow": xr.Dataset(coords={"frequency": [1.35e9, 1.45e9, 1.55e9, 1.65e9]}),
    }
    bounds = calculate_observation_frequency_boundaries([1.5e9], full)
    assert bounds["wide"] != bounds["narrow"]
    selected = {name: child.isel(frequency=slice(1, 3)) for name, child in full.items()}
    attach_observation_frequency_boundaries(selected, bounds)
    image = xr.Dataset(coords={"frequency": [1.5e9]})
    for child in selected.values():
        child = child.isel(frequency=slice(0, 1))
        np.testing.assert_array_equal(
            map_visibility_frequencies_to_image(
                child.frequency.values,
                image.frequency.values,
                frequency_boundaries=get_frequency_boundaries(image, child),
            ),
            [0],
        )
    with pytest.raises(ValueError, match="outside this task"):
        map_visibility_frequencies_to_image(
            [1.2e9], [1.5e9], frequency_boundaries=bounds["narrow"]
        )
    assert calculate_observation_frequency_boundaries([1.0e9, 2.0e9], full) == {}


def test_finished_image_metadata_cleanup_preserves_input_and_array_buffers():
    import copy

    from astroviper.processing_functions.imaging.utils.frequency_mapping import (
        without_frequency_mapping_metadata,
    )

    _, image, _ = build_datasets(
        visibility_frequencies=[1e9, 1.2e9],
        image_frequencies=[1e9, 1.2e9],
        sky_value=2.0 + 0j,
    )
    scientific_attrs = {
        "units": ["Hz"],
        "observer": "lsrk",
        "rest_frequency": {"data": 1.1e9, "units": ["Hz"]},
    }
    image.frequency.attrs = {
        **scientific_attrs,
        "full_image_frequencies": [1e9, 1.2e9],
        "frequency_boundaries": {"1000000000.0": [0.9e9, 1.1e9]},
        "image_frequency_boundaries": {},
        "observation_frequency_boundaries": {},
    }
    original_attrs = copy.deepcopy(image.frequency.attrs)
    cleaned = without_frequency_mapping_metadata(image)
    assert image.frequency.attrs == original_attrs
    assert cleaned.frequency.attrs == scientific_attrs
    np.testing.assert_array_equal(cleaned.frequency, image.frequency)
    for name in image.data_vars:
        assert cleaned[name].data is image[name].data
    assert without_frequency_mapping_metadata(cleaned) is cleaned
