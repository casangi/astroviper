"""The gridding kernel object, its parameters and the table layout the
gridders read."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    coordinates,
    create_prolate_spheroidal_correcting_image_1D,
    create_prolate_spheroidal_kernel_1D,
    prolate_spheroidal_function,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    DEFAULT_GRIDDING_KERNEL_PARAMS,
    GriddingKernel,
    as_gridding_kernel,
    make_gridding_kernel,
    resolve_gridding_kernel_params,
)


class TestParams:
    def test_defaults_and_overrides(self):
        assert resolve_gridding_kernel_params() == DEFAULT_GRIDDING_KERNEL_PARAMS
        assert resolve_gridding_kernel_params(None) == DEFAULT_GRIDDING_KERNEL_PARAMS
        params = resolve_gridding_kernel_params({"oversampling": 1000})
        assert params == {**DEFAULT_GRIDDING_KERNEL_PARAMS, "oversampling": 1000}
        # a new dict every time, the defaults untouched
        params["oversampling"] = 5
        assert DEFAULT_GRIDDING_KERNEL_PARAMS["oversampling"] == 100

    @pytest.mark.parametrize(
        "bad, match",
        [
            ({"oversamplng": 10}, "unknown"),
            ({"type": "boxcar"}, "not one of"),
            ({"support": 6}, "odd positive"),
            ({"support": 9}, "spans 7 grid points"),
            ({"oversampling": 0}, "positive integer"),
            ({"oversampling": 2.5}, "positive integer"),
        ],
    )
    def test_refusals(self, bad, match):
        with pytest.raises(ValueError, match=match):
            resolve_gridding_kernel_params(bad)


class TestKernel:
    def test_default_kernel_is_the_prolate_spheroidal_table(self):
        kernel = make_gridding_kernel()
        assert kernel.support == 7 and kernel.oversampling == 100
        np.testing.assert_array_equal(
            kernel.table, create_prolate_spheroidal_kernel_1D(100, 7)
        )
        assert kernel.table.size == 100 * 4
        assert kernel.params == DEFAULT_GRIDDING_KERNEL_PARAMS

    def test_table_layout(self):
        """Entry k is the kernel k / oversampling cells from the centre and the
        last cell of the table is zero."""
        kernel = make_gridding_kernel({"oversampling": 50})
        assert kernel.table.size == 50 * 4
        distance = np.arange(50 * 3) / 50  # cells
        _, expected = prolate_spheroidal_function(distance / 3)
        np.testing.assert_allclose(kernel.table[: 50 * 3], expected, rtol=1e-12)
        assert np.all(kernel.table[50 * 3 :] == 0.0)
        assert kernel.table[0] == kernel.table.max()

    def test_correction_is_the_spheroidal_transform(self):
        kernel = make_gridding_kernel()
        n = 64
        expected_l, expected_m = create_prolate_spheroidal_correcting_image_1D([n, 48])
        np.testing.assert_array_equal(kernel.correction_1D(n), expected_l)
        np.testing.assert_array_equal(kernel.correction_1D(48), expected_m)
        np.testing.assert_array_equal(
            kernel.correction_1D(n),
            prolate_spheroidal_function(np.abs(2.0 * coordinates(n)))[0],
        )

    def test_bare_table_is_wrapped(self):
        table = create_prolate_spheroidal_kernel_1D(25, 7)
        kernel = as_gridding_kernel(table)
        assert isinstance(kernel, GriddingKernel)
        assert (kernel.support, kernel.oversampling) == (7, 25)
        assert as_gridding_kernel(kernel) is kernel
        with pytest.raises(ValueError, match="bare kernel table"):
            as_gridding_kernel(np.zeros(10))

    def test_table_size_is_checked(self):
        with pytest.raises(ValueError, match="entries"):
            GriddingKernel(np.zeros(399), 7, 100)
