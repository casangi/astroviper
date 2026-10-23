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
    CORRECTION_FLOOR,
    DEFAULT_GRIDDING_KERNEL_PARAMS,
    KERNEL_TYPES,
    GriddingKernel,
    as_gridding_kernel,
    kept_half_width,
    make_gridding_kernel,
    resolve_gridding_kernel_params,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.kernel_lookup import (
    stencil_lookup,
    stencil_values,
    table_transform,
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
            ({"oversamplng": 10}, "Unknown gridding_kernel_params"),
            ({"type": "boxcar"}, "must be one of"),
            ({"support": 6}, "support must be odd"),
            ({"support": 1, "type": "exponential_semicircle"}, "support must be >= 3"),
            ({"support": 9}, "support=7"),
            ({"oversampling": 0}, "oversampling must be >= 1"),
            ({"oversampling": 2.5}, "oversampling must be an integer"),
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
        assert kernel.kernel_type == "prolate_spheroidal_casa"

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
        assert kernel.kernel_type == "prolate_spheroidal_casa"
        assert as_gridding_kernel(kernel) is kernel
        with pytest.raises(ValueError, match="bare kernel table"):
            as_gridding_kernel(np.zeros(10))

    def test_table_size_is_checked(self):
        with pytest.raises(ValueError, match="entries"):
            GriddingKernel(np.zeros(399), 7, 100)


class TestInterpolation:
    def test_default_is_nearest_and_linear_is_selectable(self):
        assert make_gridding_kernel().interpolation == "nearest"
        assert make_gridding_kernel().interpolation_code == 0
        kernel = make_gridding_kernel({"interpolation": "linear"})
        assert kernel.interpolation == "linear" and kernel.interpolation_code == 1
        assert kernel.params["interpolation"] == "linear"
        assert "linear" in repr(kernel)

    def test_unknown_interpolation_is_refused(self):
        with pytest.raises(ValueError, match="interpolation"):
            resolve_gridding_kernel_params({"interpolation": "cubic"})
        with pytest.raises(ValueError, match="interpolation"):
            GriddingKernel(np.zeros(400), 7, 100, interpolation="cubic")


def test_prolate_spheroidal_table_exists_for_support_7_only():
    with pytest.raises(ValueError, match="support=7"):
        create_prolate_spheroidal_kernel_1D(100, 9)
    assert create_prolate_spheroidal_kernel_1D(100, 7).size == 400


class TestLookup:
    """The Python twin of kernel_lookup.hpp."""

    def test_matches_the_reference_gridder_lookup(self):
        import importlib.util
        import pathlib

        reference = (
            pathlib.Path(__file__).parents[1] / "gridders" / "reference_gridders.py"
        )
        spec = importlib.util.spec_from_file_location("reference_gridders", reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(3)
        for support, oversampling in ((7, 100), (9, 16), (3, 7)):
            table = rng.uniform(0.1, 1.0, oversampling * (support // 2 + 1))
            offsets = rng.uniform(-0.5, 0.5, 40)
            offsets[0], offsets[1] = 0.5, -0.5 + 1e-12
            for interpolation, code in (("nearest", 0), ("linear", 1)):
                values = stencil_values(
                    table, support, oversampling, interpolation, offsets
                )
                for offset, row in zip(offsets, values, strict=True):
                    np.testing.assert_array_equal(
                        row,
                        module.kernel_values(
                            table, support, oversampling, code, offset
                        ),
                    )

    def test_weights_sum_to_one(self):
        _, weights = stencil_lookup(7, 50, "linear", np.linspace(-0.5, 0.5, 11))
        np.testing.assert_allclose(weights.sum(axis=-2), 1.0)
        _, weights = stencil_lookup(7, 50, "nearest", np.linspace(-0.5, 0.5, 11))
        np.testing.assert_array_equal(weights.sum(axis=-2), 1.0)

    def test_table_transform_is_the_transform_of_the_kernel_applied(self):
        """Brute force: the piecewise constant (nearest) and piecewise linear
        (linear) kernels built from the table, transformed numerically."""
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle import (
            create_exponential_semicircle_kernel_1D,
        )

        oversampling, support = 8, 5
        table = create_exponential_semicircle_kernel_1D(oversampling, support, 2.0, 0.5)
        distance = np.linspace(-3.0, 3.0, 240001)
        step = distance[1] - distance[0]
        index = np.minimum(
            np.rint(np.abs(distance) * oversampling).astype(int), table.size - 1
        )
        nearest = table[index]
        linear = np.interp(
            np.abs(distance) * oversampling, np.arange(table.size), table
        )
        for x in (0.0, 0.17, 0.5, 1.3):
            phase = np.cos(2.0 * np.pi * distance * x)
            np.testing.assert_allclose(
                table_transform(table, oversampling, "nearest", x),
                np.sum(nearest * phase) * step,
                rtol=1e-6,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                table_transform(table, oversampling, "linear", x),
                np.sum(linear * phase) * step,
                rtol=1e-6,
                atol=1e-8,
            )


class TestExponentialSemicircle:
    def test_params_accept_any_odd_support(self):
        params = resolve_gridding_kernel_params(
            {"type": "exponential_semicircle", "support": 9}
        )
        assert params["support"] == 9
        assert "exponential_semicircle" in KERNEL_TYPES

    def test_padding_is_required(self):
        with pytest.raises(ValueError, match="fft_padding"):
            make_gridding_kernel({"type": "exponential_semicircle"})
        with pytest.raises(ValueError, match="fft_padding"):
            make_gridding_kernel({"type": "least_misfit"}, fft_padding=0.9)
        assert kept_half_width(2.0) == 0.25
        assert kept_half_width(1.25) == 0.4

    def test_table_layout(self):
        kernel = make_gridding_kernel(
            {"type": "exponential_semicircle", "support": 9, "oversampling": 20},
            fft_padding=2.0,
        )
        assert kernel.table.size == 20 * 5
        assert kernel.table[0] == 1.0 == kernel.table.max()
        distance = np.arange(kernel.table.size) / 20
        # zero beyond the half width of 4.5 cells, positive and falling within
        assert np.all(kernel.table[distance > 4.5] == 0.0)
        inside = kernel.table[distance <= 4.5]
        assert np.all(inside > 0.0) and np.all(np.diff(inside) < 0.0)
        assert kernel.shape["fft_padding"] == 2.0
        assert 0.4 < kernel.shape["mu"] < 0.7 and 1.0 < kernel.shape["beta"] < 3.0

    def test_shape_table_matches_the_published_optimum(self):
        """Arras et al. (2021, appendix A) give beta 2.1278, mu 0.5186 and an
        error 7.7e-7 for support 7 at a padding of 2; our error measure
        reproduces that and our optimum is at least as good."""
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle import (
            SHAPE_TABLE,
            aliasing_error,
            exponential_semicircle_parameters,
            exponential_semicircle_transform,
        )

        x = np.linspace(0.0, 0.25, 129)
        published = np.max(
            aliasing_error(
                lambda p: exponential_semicircle_transform(p, 7, 2.1278, 0.5186), x
            )
        )
        assert published == pytest.approx(7.7e-7, rel=0.05)
        beta, mu, error = exponential_semicircle_parameters(7, 0.25)
        assert (beta, mu) == SHAPE_TABLE[(7, 2.0)]
        assert error <= published * 1.02
        # the error falls steeply with the support and rises as the padding shrinks
        assert exponential_semicircle_parameters(9, 0.25)[2] < error / 20
        assert exponential_semicircle_parameters(7, 0.4)[2] > error * 100

    def test_unlisted_padding_runs_the_optimiser(self):
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle import (
            exponential_semicircle_parameters,
        )

        beta, mu, error = exponential_semicircle_parameters(5, 0.5 / 1.7)
        assert round(beta, 4) == beta and round(mu, 4) == mu
        between = (
            exponential_semicircle_parameters(5, 0.5 / 1.5)[2],
            exponential_semicircle_parameters(5, 0.5 / 2.0)[2],
        )
        assert between[1] < error < between[0]


class TestLeastMisfit:
    def test_beats_the_exponential_semicircle_when_the_lookup_allows(self):
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_least_misfit import (
            misfit,
        )

        for interpolation, oversampling in (("linear", 200), ("nearest", 20)):
            params = {
                "support": 7,
                "oversampling": oversampling,
                "interpolation": interpolation,
            }
            least = make_gridding_kernel(
                {"type": "least_misfit", **params}, fft_padding=1.5
            )
            semicircle = make_gridding_kernel(
                {"type": "exponential_semicircle", **params}, fft_padding=1.5
            )
            x0 = kept_half_width(1.5)
            own = misfit(least.table, 7, oversampling, interpolation, x0)
            other = misfit(semicircle.table, 7, oversampling, interpolation, x0)
            assert own <= other * 1.001
            assert least.shape["misfit"] == pytest.approx(own, rel=1e-6)
            assert np.all(
                least.table[np.arange(least.table.size) / oversampling > 3.5] == 0.0
            )
        # with linear lookup the design is aliasing limited and clearly better
        assert own < 0.7 * other or interpolation == "nearest"

    def test_linear_design_is_clearly_better(self):
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_least_misfit import (
            misfit,
        )

        params = {"support": 7, "oversampling": 300, "interpolation": "linear"}
        least = make_gridding_kernel(
            {"type": "least_misfit", **params}, fft_padding=2.0
        )
        semicircle = make_gridding_kernel(
            {"type": "exponential_semicircle", **params}, fft_padding=2.0
        )
        assert least.shape["misfit"] < 0.7 * misfit(
            semicircle.table, 7, 300, "linear", 0.25
        )

    def test_fine_tables_are_resampled(self):
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_least_misfit import (
            MAX_SOLVED_TABLE,
            create_least_misfit_kernel_1D,
        )

        oversampling = MAX_SOLVED_TABLE // 2 + 50  # support 3: two cells
        table, info = create_least_misfit_kernel_1D(oversampling, 3, "linear", 0.25)
        assert table.size == 2 * oversampling
        assert info["solved_oversampling"] < oversampling
        coarse, coarse_info = create_least_misfit_kernel_1D(
            info["solved_oversampling"], 3, "linear", 0.25
        )
        assert info["misfit"] == pytest.approx(coarse_info["misfit"], rel=0.05)
        np.testing.assert_allclose(
            table_transform(table, oversampling, "linear", 0.0), 1.0
        )


class TestCorrection:
    @pytest.mark.parametrize(
        "kernel_type", ["prolate_spheroidal", "exponential_semicircle", "least_misfit"]
    )
    def test_numerical_corrections_are_the_table_transform(self, kernel_type):
        kernel = make_gridding_kernel(
            {"type": kernel_type, "oversampling": 50, "interpolation": "linear"},
            fft_padding=2.0,
        )
        n = 96
        x = coordinates(n)
        expected = table_transform(kernel.table, 50, "linear", x)
        expected /= table_transform(kernel.table, 50, "linear", 0.0)
        np.testing.assert_allclose(kernel.correction_1D(n), expected)
        assert kernel.correction_1D(n)[n // 2] == 1.0
        # positive everywhere, far above the floor over the image kept
        assert np.all(kernel.correction_1D(n) > CORRECTION_FLOOR)
        assert np.all(kernel.correction(x[np.abs(x) <= 0.25]) > 0.1)

    def test_map_error_of_the_prolate_kernel_is_its_lookup_error(self):
        """Nearest lookup at 100 samples per cell: the rms position error of a
        visibility is 1 / (100 sqrt(12)) cells, a phase error 2 pi x of that
        at image position x."""
        kernel = make_gridding_kernel()
        x = np.array([0.05, 0.1, 0.2])
        np.testing.assert_allclose(
            kernel.map_error(x), 2.0 * np.pi * x / (100 * np.sqrt(12.0)), rtol=0.05
        )

    def test_map_error_falls_with_oversampling_and_interpolation(self):
        x = np.array([0.1, 0.25])
        nearest = make_gridding_kernel(
            {"type": "exponential_semicircle"}, 2.0
        ).map_error(x)
        finer = make_gridding_kernel(
            {"type": "exponential_semicircle", "oversampling": 1000}, 2.0
        ).map_error(x)
        linear = make_gridding_kernel(
            {"type": "exponential_semicircle", "interpolation": "linear"}, 2.0
        ).map_error(x)
        np.testing.assert_allclose(nearest / finer, 10.0, rtol=0.05)
        assert np.all(linear < nearest / 200)
        # the prolate spheroidal kernel does not profit from the padding: its
        # aliasing shows once the lookup error is gone
        prolate = make_gridding_kernel(
            {"interpolation": "linear", "oversampling": 1000}
        )
        assert np.all(prolate.map_error(x) > 10 * linear)


class TestDefaultAndTableCorrections:
    def test_only_the_default_keeps_the_analytic_correction(self):
        """The default is bit for bit the analytic correction; any other sampling
        or lookup of the prolate kernel is corrected by its table transform."""
        n = 3750
        default = make_gridding_kernel()
        np.testing.assert_array_equal(
            default.correction_1D(n),
            create_prolate_spheroidal_correcting_image_1D([n, n])[0],
        )
        for params in ({"oversampling": 1000}, {"interpolation": "linear"}):
            kernel = make_gridding_kernel(params)
            x = coordinates(n)
            expected = table_transform(
                kernel.table, kernel.oversampling, kernel.interpolation, x
            ) / table_transform(
                kernel.table, kernel.oversampling, kernel.interpolation, 0.0
            )
            np.testing.assert_allclose(
                kernel.correction_1D(n), np.maximum(expected, CORRECTION_FLOOR)
            )

    def test_analytic_correction_of_the_sampled_table_is_biased(self):
        """Sampled at 100 per cell with the nearest lookup, the table's transform is
        the analytic correction times sinc(x / 100): a flux bias of
        -(pi x / M)^2 / 6 for the analytic correction, 2.9e-5 at x = 0.42."""
        kernel = make_gridding_kernel()
        x = np.linspace(0.0, 0.42, 43)
        ratio = table_transform(kernel.table, 100, "nearest", x) / table_transform(
            kernel.table, 100, "nearest", 0.0
        )
        ratio /= kernel.correction(x)
        np.testing.assert_allclose(ratio, np.sinc(x / 100), rtol=0, atol=1e-6)
        assert 1.0 - ratio[-1] == pytest.approx((np.pi * 0.42 / 100) ** 2 / 6, rel=0.05)

    @pytest.mark.parametrize(
        "kernel_type", ["prolate_spheroidal", "exponential_semicircle", "least_misfit"]
    )
    def test_designed_kernels_refuse_a_small_padding(self, kernel_type):
        with pytest.raises(ValueError, match="fft_padding >= 1.05"):
            make_gridding_kernel({"type": kernel_type}, 1.02)

    @pytest.mark.parametrize(
        "kernel_type", ["prolate_spheroidal", "exponential_semicircle", "least_misfit"]
    )
    def test_designed_kernels_refuse_a_correction_below_the_floor(self, kernel_type):
        """Support 17 at a padding of 1.05: the correction at the edge of the
        image kept is about 1e-7, so the floor would clamp the outer image."""
        with pytest.raises(
            ValueError,
            match=f"the {kernel_type} gridding kernel of support 17 designed for "
            "fft_padding 1.05 has a correction of .* below the floor of 1e-06",
        ):
            make_gridding_kernel({"type": kernel_type, "support": 17}, 1.05)

    @pytest.mark.parametrize(
        "kernel_type", ["prolate_spheroidal", "exponential_semicircle"]
    )
    def test_every_listed_shape_builds(self, kernel_type):
        """Every support and padding of the shape tables is well above the
        floor at the edge of the image kept (about 1e-4 at the least, support
        17 at 1.2)."""
        from astroviper.processing_functions.imaging.gridding_convolution_functions import (
            gcf_exponential_semicircle,
            gcf_generalized_prolate_spheroidal,
        )

        shapes = {
            "prolate_spheroidal": gcf_generalized_prolate_spheroidal.SHAPE_TABLE,
            "exponential_semicircle": gcf_exponential_semicircle.SHAPE_TABLE,
        }[kernel_type]
        for support, fft_padding in shapes:
            kernel = make_gridding_kernel(
                {"type": kernel_type, "support": support}, fft_padding
            )
            x0 = kept_half_width(fft_padding)
            edge = kernel.transform(x0) / kernel.transform(0.0)
            assert edge > 10 * CORRECTION_FLOOR, (support, fft_padding)

    def test_default_kernel_is_not_checked(self):
        """The prolate_spheroidal_casa kernel does not depend on the padding: at
        one that refuses the designed kernels of support 17 the default is the
        same kernel, with its analytic correction bit for bit."""
        n = 3750
        default = make_gridding_kernel()
        assert make_gridding_kernel(None, 1.05) is default
        np.testing.assert_array_equal(
            default.table, create_prolate_spheroidal_kernel_1D(100, 7)
        )
        np.testing.assert_array_equal(
            default.correction_1D(n),
            create_prolate_spheroidal_correcting_image_1D([n, n])[0],
        )


def test_least_misfit_table_is_the_same_in_another_process():
    """The least misfit solve gives the same table on every process."""
    import subprocess
    import sys

    code = (
        "from astroviper.processing_functions.imaging.gridding_convolution_functions."
        "gridding_kernel import make_gridding_kernel; import hashlib; "
        "k = make_gridding_kernel({'type': 'least_misfit', 'oversampling': 100, "
        "'interpolation': 'linear'}, 1.2); print(hashlib.sha256(k.table.tobytes()).hexdigest())"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(digests) == 1
