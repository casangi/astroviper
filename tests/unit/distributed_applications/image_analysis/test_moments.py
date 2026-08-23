"""Unit tests for the moments distributed application (full 3-layer runs)."""

import numpy as np
import pandas as pd
import pytest
from xradio.image import load_image

from astroviper.distributed_applications.image_analysis import moments
from tests.unit.processing_functions.image_analysis.moments_test_utils import (
    ALL_MOMENTS,
    assert_moments_match,
    make_test_image_xds,
    reference_moments,
    write_test_image,
)


@pytest.fixture
def input_image(tmp_path):
    """Synthetic image written to a Zarr store; returns (path, in-memory xds)."""
    img_xds = make_test_image_xds()
    path = tmp_path / "input.img.zarr"
    write_test_image(img_xds, path)
    return str(path), img_xds


class TestMomentsFullStack:
    def test_frequency_moment_parallel_over_m(self, input_image, tmp_path):
        """Moment over frequency: the graph parallelizes over m (never the moment axis)."""
        path, img_xds = input_image
        out = str(tmp_path / "moments.img.zarr")
        timing = moments(
            input_image_store=path,
            moments_image_store=out,
            moments=ALL_MOMENTS,
            moment_axis="frequency",
            n_mapping_parallelism={"m": 3},
        )
        assert isinstance(timing, pd.DataFrame)
        assert len(timing) == 3  # one row per m chunk
        result = load_image(out)
        assert result.sizes["frequency"] == 1
        np.testing.assert_allclose(
            result.frequency.values, [img_xds.frequency.values.mean()]
        )
        reference = reference_moments(
            img_xds.SKY.values, axis=1, coord_values=img_xds.frequency.values
        )
        assert_moments_match(result, reference, axis=1)
        # per-moment data groups registered on the store
        for name in ALL_MOMENTS:
            group = result.attrs["data_groups"]["moment_" + name]
            assert group["sky"] == "SKY_MOMENT_" + name.upper()

    def test_resource_monitoring_series_in_timing_frame(self, input_image, tmp_path):
        """monitor_resources_seconds folds graphviper's per-task CPU / memory
        series into the timing frame as list-valued columns (imaging layout)."""
        pytest.importorskip("psutil")
        path, img_xds = input_image
        out = str(tmp_path / "moments_monitored.img.zarr")
        timing = moments(
            input_image_store=path,
            moments_image_store=out,
            moments=["maximum"],
            moment_axis="frequency",
            n_mapping_parallelism={"m": 2},
            monitor_resources_seconds=0.01,
        )
        assert len(timing) == 2
        for column in ("time_seconds", "cpu_percent", "memory_rss_bytes"):
            assert column in timing.columns, column
            assert all(isinstance(v, list) for v in timing[column])
        assert (timing["sample_interval_seconds"] == 0.01).all()
        assert "start_unixtime" in timing.columns
        reference = reference_moments(
            img_xds.SKY.values, axis=1, coord_values=img_xds.frequency.values
        )
        assert_moments_match(load_image(out), reference, axis=1)

    def test_m_moment_parallel_over_frequency(self, input_image, tmp_path):
        path, img_xds = input_image
        out = str(tmp_path / "moments_m.img.zarr")
        moments(
            input_image_store=path,
            moments_image_store=out,
            moments=["mean", "maximum", "maximum_coord"],
            moment_axis="m",
            n_mapping_parallelism={"frequency": 2},
        )
        result = load_image(out)
        assert result.sizes["m"] == 1
        reference = reference_moments(
            img_xds.SKY.values, axis=4, coord_values=img_xds.m.values
        )
        assert_moments_match(result, reference, axis=4)

    def test_casa_integer_codes_and_mask_and_include(self, input_image, tmp_path):
        path, img_xds = input_image
        out = str(tmp_path / "moments_casa.img.zarr")
        moments(
            input_image_store=path,
            moments_image_store=out,
            moments=[-1, 0, 8],  # mean, integrated, maximum
            moment_axis="frequency",
            include_pixel_range=[0.0, 2.0],
            use_mask=True,
            n_mapping_parallelism={"m": 2},
        )
        result = load_image(out)
        assert sorted(result.data_vars) == [
            "SKY_MOMENT_INTEGRATED",
            "SKY_MOMENT_MAXIMUM",
            "SKY_MOMENT_MEAN",
        ]
        reference = reference_moments(
            img_xds.SKY.values,
            axis=1,
            coord_values=img_xds.frequency.values,
            mask=img_xds.MASK.values,
            include_range=(0.0, 2.0),
        )
        assert_moments_match(result, reference, axis=1)

    def test_polarization_moment_with_frequency_selection(self, input_image, tmp_path):
        path, img_xds = input_image
        out = str(tmp_path / "moments_pol.img.zarr")
        selection = {"frequency": slice(1, 5)}
        moments(
            input_image_store=path,
            moments_image_store=out,
            moments=["mean", "weighted_coord"],
            moment_axis="polarization",
            selection=selection,
            n_mapping_parallelism={"m": 2},
        )
        result = load_image(out)
        assert result.sizes["polarization"] == 1
        assert result.sizes["frequency"] == 4
        selected = img_xds.isel(selection)
        reference = reference_moments(
            selected.SKY.values,
            axis=2,
            coord_values=np.arange(selected.sizes["polarization"]),
        )
        assert_moments_match(result, reference, axis=2)

    def test_auto_chunking_with_thread_info(self, input_image, tmp_path):
        path, img_xds = input_image
        out = str(tmp_path / "moments_auto.img.zarr")
        timing = moments(
            input_image_store=path,
            moments_image_store=out,
            moments=["median", "rms"],
            moment_axis="frequency",
            thread_info={"n_threads": 2, "memory_per_thread": 4.0},
        )
        assert len(timing) >= 1
        result = load_image(out)
        reference = reference_moments(
            img_xds.SKY.values, axis=1, coord_values=img_xds.frequency.values
        )
        assert_moments_match(result, reference, axis=1)

    def test_units_on_disk(self, input_image, tmp_path):
        path, _ = input_image
        out = str(tmp_path / "moments_units.img.zarr")
        moments(
            input_image_store=path,
            moments_image_store=out,
            moments=["integrated", "weighted_coord", "mean"],
            moment_axis="frequency",
            n_mapping_parallelism={"m": 1},
        )
        result = load_image(out)
        assert result.SKY_MOMENT_INTEGRATED.attrs["units"] == "Jy/beam.Hz"
        assert result.SKY_MOMENT_WEIGHTED_COORD.attrs["units"] == "Hz"
        assert result.SKY_MOMENT_MEAN.attrs["units"] == "Jy/beam"


class TestMomentsOverwriteAndErrors:
    def test_no_overwrite_raises(self, input_image, tmp_path):
        path, _ = input_image
        out = tmp_path / "exists.img.zarr"
        out.mkdir()
        with pytest.raises(RuntimeError, match="will not be overwritten"):
            moments(
                input_image_store=path,
                moments_image_store=str(out),
                moments=["mean"],
                n_mapping_parallelism={"m": 1},
            )

    def test_overwrite_true_runs(self, input_image, tmp_path):
        path, _ = input_image
        out = str(tmp_path / "overwrite.img.zarr")
        for _ in range(2):
            moments(
                input_image_store=path,
                moments_image_store=out,
                moments=["mean"],
                n_mapping_parallelism={"m": 1},
                overwrite=True,
            )
        assert "SKY_MOMENT_MEAN" in load_image(out).data_vars

    def test_mapping_parallelism_axis_equal_to_moment_axis_raises(
        self, input_image, tmp_path
    ):
        path, _ = input_image
        with pytest.raises(ValueError, match="cannot be used for parallelism"):
            moments(
                input_image_store=path,
                moments_image_store=str(tmp_path / "x.img.zarr"),
                moments=["mean"],
                moment_axis="frequency",
                n_mapping_parallelism={"frequency": 1},
            )

    def test_mapping_parallelism_axis_l_raises(self, input_image, tmp_path):
        path, _ = input_image
        with pytest.raises(ValueError, match="not in allowed axes"):
            moments(
                input_image_store=path,
                moments_image_store=str(tmp_path / "x.img.zarr"),
                moments=["mean"],
                moment_axis="frequency",
                n_mapping_parallelism={"l": 1},
            )

    def test_mapping_parallelism_multi_entry_dict_raises(self, input_image, tmp_path):
        path, _ = input_image
        with pytest.raises(ValueError, match="exactly one entry"):
            moments(
                input_image_store=path,
                moments_image_store=str(tmp_path / "x.img.zarr"),
                moments=["mean"],
                moment_axis="frequency",
                n_mapping_parallelism={"m": 2, "time": 2},
            )

    def test_mapping_parallelism_bad_count_raises(self, input_image, tmp_path):
        path, _ = input_image
        with pytest.raises(ValueError, match="positive int"):
            moments(
                input_image_store=path,
                moments_image_store=str(tmp_path / "x.img.zarr"),
                moments=["mean"],
                moment_axis="frequency",
                n_mapping_parallelism={"m": 0},
            )

    def test_mapping_parallelism_none_count_auto_chunks(self, input_image, tmp_path):
        """A None count picks the axis but auto-determines the chunk count."""
        path, _ = input_image
        out = str(tmp_path / "auto_count.img.zarr")
        moments(
            input_image_store=path,
            moments_image_store=out,
            moments=["mean"],
            moment_axis="frequency",
            n_mapping_parallelism={"m": None},
            thread_info={"n_threads": 2, "memory_per_thread": 4.0},
        )
        assert "SKY_MOMENT_MEAN" in load_image(out).data_vars

    def test_selection_on_parallel_axis_raises(self, input_image, tmp_path):
        path, _ = input_image
        with pytest.raises(ValueError, match="must not select along"):
            moments(
                input_image_store=path,
                moments_image_store=str(tmp_path / "x.img.zarr"),
                moments=["mean"],
                moment_axis="frequency",
                selection={"m": slice(0, 5)},
            )

    def test_both_pixel_ranges_raise(self, input_image, tmp_path):
        path, _ = input_image
        with pytest.raises(ValueError, match="Only one of"):
            moments(
                input_image_store=path,
                moments_image_store=str(tmp_path / "x.img.zarr"),
                moments=["mean"],
                include_pixel_range=[0, 1],
                exclude_pixel_range=[2, 3],
            )

    def test_unknown_moment_raises(self, input_image, tmp_path):
        path, _ = input_image
        with pytest.raises(ValueError, match="Unknown moment"):
            moments(
                input_image_store=path,
                moments_image_store=str(tmp_path / "x.img.zarr"),
                moments=["bogus"],
            )
