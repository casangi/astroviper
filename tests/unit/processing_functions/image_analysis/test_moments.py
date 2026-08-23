"""Unit tests for the moments (immoments) processing function."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.image_analysis.moments import (
    CASA_MOMENT_CODES,
    COORDINATE_VALUED_MOMENTS,
    MOMENT_NAMES,
    moments,
    normalize_moments,
    normalize_pixel_range,
)
from tests.unit.processing_functions.image_analysis.moments_test_utils import (
    ALL_MOMENTS,
    assert_moments_match,
    make_test_image_xds,
    reference_moments,
)


class TestNormalization:
    def test_casa_codes_map_to_names(self):
        assert normalize_moments(list(range(-1, 12))) == MOMENT_NAMES

    def test_names_pass_through_and_dedupe(self):
        assert normalize_moments(["mean", 0, "mean", "integrated"]) == [
            "mean",
            "integrated",
        ]

    def test_single_scalar(self):
        assert normalize_moments("rms") == ["rms"]
        assert normalize_moments(8) == ["maximum"]

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown moment"):
            normalize_moments(["not_a_moment"])

    def test_unknown_code_raises(self):
        with pytest.raises(ValueError, match="Unknown CASA moment code"):
            normalize_moments([12])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one moment"):
            normalize_moments([])

    def test_pixel_range_two_values(self):
        assert normalize_pixel_range([-0.5, 2.0], "include_pixel_range") == (-0.5, 2.0)

    def test_pixel_range_single_value_is_symmetric(self):
        assert normalize_pixel_range([2.0], "include_pixel_range") == (-2.0, 2.0)
        assert normalize_pixel_range(-3, "include_pixel_range") == (-3.0, 3.0)

    def test_pixel_range_none(self):
        assert normalize_pixel_range(None, "include_pixel_range") is None

    def test_pixel_range_errors(self):
        with pytest.raises(ValueError, match="one or two values"):
            normalize_pixel_range([1, 2, 3], "include_pixel_range")
        with pytest.raises(ValueError, match="greater than"):
            normalize_pixel_range([2.0, 1.0], "exclude_pixel_range")


class TestMomentsFrequencyAxis:
    def test_all_moments_match_reference(self):
        img_xds = make_test_image_xds()
        result = moments(img_xds, moments=ALL_MOMENTS, moment_axis="frequency")
        reference = reference_moments(
            img_xds.SKY.values, axis=1, coord_values=img_xds.frequency.values
        )
        assert_moments_match(result, reference, axis=1)

    def test_include_pixel_range(self):
        img_xds = make_test_image_xds()
        result = moments(
            img_xds,
            moments=ALL_MOMENTS,
            moment_axis="frequency",
            include_pixel_range=[0.0, 1.5],
        )
        reference = reference_moments(
            img_xds.SKY.values,
            axis=1,
            coord_values=img_xds.frequency.values,
            include_range=(0.0, 1.5),
        )
        assert_moments_match(result, reference, axis=1)

    def test_exclude_pixel_range(self):
        img_xds = make_test_image_xds()
        result = moments(
            img_xds,
            moments=ALL_MOMENTS,
            moment_axis="frequency",
            exclude_pixel_range=[-0.2, 0.2],
        )
        reference = reference_moments(
            img_xds.SKY.values,
            axis=1,
            coord_values=img_xds.frequency.values,
            exclude_range=(-0.2, 0.2),
        )
        assert_moments_match(result, reference, axis=1)

    def test_use_mask(self):
        img_xds = make_test_image_xds()
        result = moments(
            img_xds, moments=ALL_MOMENTS, moment_axis="frequency", use_mask=True
        )
        reference = reference_moments(
            img_xds.SKY.values,
            axis=1,
            coord_values=img_xds.frequency.values,
            mask=img_xds.MASK.values,
        )
        assert_moments_match(result, reference, axis=1)

    def test_input_not_mutated(self):
        img_xds = make_test_image_xds()
        before = img_xds.SKY.values.copy()
        moments(
            img_xds,
            moments=ALL_MOMENTS,
            moment_axis="frequency",
            include_pixel_range=[0.0, 1.0],
        )
        np.testing.assert_array_equal(
            img_xds.SKY.values, before, err_msg="input SKY was mutated"
        )

    def test_all_nan_profile_gives_nan(self):
        img_xds = make_test_image_xds(with_nans=True)
        result = moments(img_xds, moments=ALL_MOMENTS, moment_axis="frequency")
        # (l=5, m=6) has an all-NaN frequency profile in the synthetic image.
        for name in ALL_MOMENTS:
            value = result["SKY_MOMENT_" + name.upper()].values[0, 0, 0, 5, 6]
            assert np.isnan(value), f"moment '{name}' should be NaN"


class TestMomentsOtherAxes:
    def test_polarization_axis_uses_plane_index(self):
        img_xds = make_test_image_xds(n_polarization=3)
        result = moments(
            img_xds,
            moments=["maximum_coord", "minimum_coord"],
            moment_axis="polarization",
        )
        reference = reference_moments(
            img_xds.SKY.values, axis=2, coord_values=np.arange(3)
        )
        assert_moments_match(result, reference, axis=2)
        maximum_coord = result.SKY_MOMENT_MAXIMUM_COORD.values
        finite = maximum_coord[np.isfinite(maximum_coord)]
        assert set(np.unique(finite)).issubset({0.0, 1.0, 2.0})

    def test_l_axis(self):
        img_xds = make_test_image_xds()
        result = moments(img_xds, moments=ALL_MOMENTS, moment_axis="l")
        reference = reference_moments(
            img_xds.SKY.values, axis=3, coord_values=img_xds.l.values
        )
        assert_moments_match(result, reference, axis=3)

    def test_m_axis(self):
        img_xds = make_test_image_xds()
        result = moments(img_xds, moments=["mean", "rms"], moment_axis="m")
        reference = reference_moments(
            img_xds.SKY.values, axis=4, coord_values=img_xds.m.values
        )
        assert_moments_match(result, reference, axis=4)

    def test_single_plane_time_axis(self):
        img_xds = make_test_image_xds(with_nans=False)
        result = moments(img_xds, moments=["mean", "integrated"], moment_axis="time")
        np.testing.assert_allclose(
            np.squeeze(result.SKY_MOMENT_MEAN.values, axis=0),
            img_xds.SKY.values[0],
            rtol=1e-6,
        )
        # A single plane has unit coordinate width, so integrated == mean.
        np.testing.assert_allclose(
            result.SKY_MOMENT_INTEGRATED.values,
            result.SKY_MOMENT_MEAN.values,
            rtol=1e-6,
        )


class TestMomentsMetadata:
    def test_output_structure(self):
        img_xds = make_test_image_xds()
        result = moments(img_xds, moments=ALL_MOMENTS, moment_axis="frequency")
        assert result.sizes["frequency"] == 1
        np.testing.assert_allclose(
            result.frequency.values, [img_xds.frequency.values.mean()]
        )
        # velocity rides on frequency and must collapse with it
        if "velocity" in result.coords:
            assert result.velocity.sizes["frequency"] == 1
        for name in ALL_MOMENTS:
            variable_name = "SKY_MOMENT_" + name.upper()
            assert variable_name in result.data_vars
            assert result.attrs["data_groups"]["moment_" + name]["sky"] == variable_name
            assert "description" in result.attrs["data_groups"]["moment_" + name]

    def test_dtypes_follow_precision_rules(self):
        img_xds = make_test_image_xds(dtype=np.float32)
        result = moments(img_xds, moments=ALL_MOMENTS, moment_axis="frequency")
        for name in ALL_MOMENTS:
            dtype = result["SKY_MOMENT_" + name.upper()].dtype
            if name in COORDINATE_VALUED_MOMENTS:
                assert dtype == np.float64, name
            else:
                assert dtype == np.float32, name

    def test_units(self):
        img_xds = make_test_image_xds()
        result = moments(
            img_xds,
            moments=["mean", "integrated", "weighted_coord"],
            moment_axis="frequency",
        )
        assert result.SKY_MOMENT_MEAN.attrs["units"] == "Jy/beam"
        assert result.SKY_MOMENT_INTEGRATED.attrs["units"] == "Jy/beam.Hz"
        assert result.SKY_MOMENT_WEIGHTED_COORD.attrs["units"] == "Hz"

    def test_fallback_to_sky_without_data_groups(self):
        img_xds = make_test_image_xds(with_mask=False)
        img_xds.attrs.pop("data_groups")
        result = moments(img_xds, moments=["mean"], moment_axis="frequency")
        assert "SKY_MOMENT_MEAN" in result.data_vars

    def test_missing_data_group_raises(self):
        img_xds = make_test_image_xds()
        with pytest.raises(AssertionError, match="not found in image data_groups"):
            moments(img_xds, moments=["mean"], image_data_group_in_name="nope")


class TestMomentsErrors:
    def test_both_ranges_raise(self):
        img_xds = make_test_image_xds()
        with pytest.raises(ValueError, match="Only one of"):
            moments(
                img_xds,
                moments=["mean"],
                include_pixel_range=[0, 1],
                exclude_pixel_range=[0, 1],
            )

    def test_bad_moment_axis_raises(self):
        img_xds = make_test_image_xds()
        with pytest.raises(ValueError, match="not in allowed axes"):
            moments(img_xds, moments=["mean"], moment_axis="u")

    def test_moment_axis_not_a_dim_raises(self):
        img_xds = make_test_image_xds()
        img_xds = img_xds.rename({"time": "epoch"})
        img_xds["SKY"] = xr.DataArray(
            img_xds.SKY.values, dims=["epoch", "frequency", "polarization", "l", "m"]
        )
        with pytest.raises(ValueError, match="is not a dimension"):
            moments(img_xds, moments=["mean"], moment_axis="time")


def test_moment_registry_consistency():
    """The CASA code map and the canonical name list must stay in sync."""
    assert sorted(CASA_MOMENT_CODES.values()) == sorted(MOMENT_NAMES)
    assert ALL_MOMENTS == MOMENT_NAMES


# ---------------------------------------------------------------------------
# Streaming accumulator API (the memory-efficient production path)
# ---------------------------------------------------------------------------
class TestStreamingAccumulator:
    STREAMABLE = [name for name in ALL_MOMENTS if name != "median"]

    @staticmethod
    def _stream(img_xds, names, n_block, axis="frequency", use_mask=True, **kw):
        from astroviper.processing_functions.image_analysis.moments import (
            moments_streamed,
        )

        sky = img_xds["SKY"]
        ax = sky.dims.index(axis)
        mask = img_xds["MASK"] if use_mask else None
        calls = []

        def read_planes(start, stop):
            stop = min(stop, start + n_block)
            calls.append((start, stop))
            sel = {axis: slice(start, stop)}
            planes = np.moveaxis(sky.isel(sel).values, ax, 0)
            mask_planes = (
                None if mask is None else np.moveaxis(mask.isel(sel).values, ax, 0)
            )
            return planes, mask_planes

        result = moments_streamed(
            read_planes,
            coords_xds=img_xds,
            sky_dims=sky.dims,
            sky_name="SKY",
            sky_attrs=dict(sky.attrs),
            attrs=img_xds.attrs,
            moments=names,
            moment_axis=axis,
            **kw,
        )
        return result, calls

    @pytest.mark.parametrize("n_block", [1, 3, 100])
    @pytest.mark.parametrize("axis", ["frequency", "polarization", "m"])
    def test_streamed_matches_in_memory(self, n_block, axis):
        img_xds = make_test_image_xds(dtype=np.float32)
        expected = moments(
            img_xds, moments=self.STREAMABLE, moment_axis=axis, use_mask=True
        )
        result, calls = self._stream(img_xds, self.STREAMABLE, n_block, axis=axis)
        n = img_xds.sizes[axis]
        # Two passes (abs_mean_dev / median_coord) over ceil(n / n_block) blocks.
        assert len(calls) == 2 * -(-n // n_block)
        for name in self.STREAMABLE:
            variable = "SKY_MOMENT_" + name.upper()
            assert result[variable].dtype == expected[variable].dtype, name
            np.testing.assert_allclose(
                result[variable].values,
                expected[variable].values,
                rtol=1e-6,
                equal_nan=True,
                err_msg=name,
            )
        assert (
            result.attrs["data_groups"].keys() == expected.attrs["data_groups"].keys()
        )
        assert result.sizes == expected.sizes

    def test_single_pass_streams_once_and_pixel_ranges_apply(self):
        img_xds = make_test_image_xds()
        names = ["maximum", "maximum_coord", "integrated", "rms"]
        expected = moments(
            img_xds,
            moments=names,
            moment_axis="frequency",
            include_pixel_range=[-0.5, 2.0],
            use_mask=False,
        )
        result, calls = self._stream(
            img_xds, names, 2, use_mask=False, include_pixel_range=[-0.5, 2.0]
        )
        assert len(calls) == -(-img_xds.sizes["frequency"] // 2)  # one pass
        for name in names:
            variable = "SKY_MOMENT_" + name.upper()
            np.testing.assert_allclose(
                result[variable].values,
                expected[variable].values,
                rtol=1e-6,
                equal_nan=True,
                err_msg=name,
            )

    def test_median_is_rejected(self):
        img_xds = make_test_image_xds()
        with pytest.raises(ValueError, match="median"):
            self._stream(img_xds, ["median"], 1)

    def test_memory_model(self):
        from astroviper.processing_functions.image_analysis.moments import (
            moments_memory_model,
        )

        assert moments_memory_model(["maximum", "mean"]) == {
            "n_passes": 1,
            "requires_full_profile": False,
            "mergeable": True,
        }
        assert moments_memory_model(["abs_mean_dev"]) == {
            "n_passes": 2,
            "requires_full_profile": False,
            "mergeable": False,
        }
        assert moments_memory_model(["median", "rms"])["requires_full_profile"]
        assert not moments_memory_model(["median"])["mergeable"]

    def test_merge_of_moment_axis_segments_is_exact(self):
        """Partial accumulators over disjoint frequency segments merge to the
        full result (the building block of a moment-axis map + reduce)."""
        from astroviper.processing_functions.image_analysis.moments import (
            MomentsAccumulator,
            _moment_axis_values,
        )

        img_xds = make_test_image_xds(dtype=np.float64)
        names = [
            "mean",
            "integrated",
            "weighted_coord",
            "weighted_dispersion_coord",
            "standard_deviation",
            "rms",
            "maximum",
            "maximum_coord",
            "minimum",
            "minimum_coord",
        ]
        sky = img_xds["SKY"].values[0]  # frequency first already
        mask = img_xds["MASK"].values[0]
        coord = _moment_axis_values(img_xds, "frequency")
        map_shape = sky.shape[1:]

        full = MomentsAccumulator(names, coord, map_shape)
        parts = [MomentsAccumulator(names, coord, map_shape) for _ in range(3)]
        for i in range(sky.shape[0]):
            full.add_plane(i, sky[i], mask[i])
            parts[i % 3].add_plane(i, sky[i], mask[i])
        merged = parts[2].merge(parts[0]).merge(parts[1])
        a, b = full.finalize(), merged.finalize()
        for name in names:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-12, equal_nan=True)

        two_pass = MomentsAccumulator(["abs_mean_dev"], coord, map_shape)
        with pytest.raises(ValueError, match="single-pass"):
            two_pass.merge(MomentsAccumulator(["abs_mean_dev"], coord, map_shape))
