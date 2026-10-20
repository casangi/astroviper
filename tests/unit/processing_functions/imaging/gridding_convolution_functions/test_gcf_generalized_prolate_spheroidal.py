"""The generalised prolate spheroidal gridding kernel (the ``prolate_spheroidal``
kernel type)."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle import (
    aliasing_error,
    exponential_semicircle_parameters,
    exponential_semicircle_transform,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_generalized_prolate_spheroidal import (
    SHAPE_TABLE,
    create_generalized_prolate_spheroidal_kernel_1D,
    generalized_prolate_spheroidal,
    generalized_prolate_spheroidal_kernel,
    generalized_prolate_spheroidal_parameters,
    generalized_prolate_spheroidal_transform,
    nominal_bandwidth,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    prolate_spheroidal_function,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    CORRECTION_FLOOR,
    KERNEL_TYPES,
    kept_half_width,
    make_gridding_kernel,
    resolve_gridding_kernel_params,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.kernel_lookup import (
    table_transform,
)


def _max_aliasing_error(transform, x0):
    return float(np.max(aliasing_error(transform, np.linspace(0.0, x0, 129))))


class TestFunction:
    def test_reproduces_schwab_rational_approximation(self):
        """c = 3 pi, alpha = 1 over a half width of 3 cells (Schwab's m = 6) is
        the function behind the prolate_spheroidal_casa kernel: the kernel and
        its correction agree with the rational approximation to its accuracy
        (a few parts in 1e7)."""
        eta = np.linspace(0.0, 1.0, 3001)
        correction, kernel = prolate_spheroidal_function(eta)
        ours = generalized_prolate_spheroidal_kernel(eta, 3.0 * np.pi, 1.0)
        kernel_difference = np.max(np.abs(ours - kernel / correction[0]))
        x = np.linspace(0.0, 0.5, 501)  # 2 x = eta out to the edge of the band
        transform = generalized_prolate_spheroidal_transform(x, 6, 3.0 * np.pi, 1.0)
        correction_difference = np.max(
            np.abs(
                transform / transform[0]
                - prolate_spheroidal_function(2.0 * x)[0] / correction[0]
            )
        )
        print(
            f"Schwab anchor: kernel {kernel_difference:.2e}, "
            f"correction {correction_difference:.2e}"
        )
        assert kernel_difference < 1e-6
        assert correction_difference < 1e-6

    @pytest.mark.parametrize(
        "support, c, alpha",
        [
            (3, 3.0, 0.5),
            (7, 12.6665, 0.0),
            (7, 9.0, 1.7),
            (17, 56.0, 0.0),
            (17, 40.0, 3.0),
        ],
    )
    def test_transform_is_the_function(self, support, c, alpha):
        """The eigen relation: the transform of the kernel, (W / 2) lambda
        psi(pi W x / c), is the function itself across the band, for any c
        (c = 56 at alpha = 0 is far beyond where the leading eigenvalues of
        the transform coincide in double precision)."""
        x = np.linspace(-1.0, 1.0, 801) * c / (np.pi * support)
        transform = generalized_prolate_spheroidal_transform(x, support, c, alpha)
        centre = generalized_prolate_spheroidal_transform(0.0, support, c, alpha)
        eta = np.clip(np.pi * support * x / c, -1.0, 1.0)
        np.testing.assert_allclose(
            transform / centre,
            generalized_prolate_spheroidal(eta, c, alpha),
            rtol=0,
            atol=1e-12,
        )
        # the eigenvalue: (W / 2) lambda = transform at the centre, with
        # |lambda| below the sqrt(2 pi / c) of the unweighted transform
        eigenvalue = 2.0 * centre / support
        assert 0.0 < eigenvalue <= np.sqrt(2.0 * np.pi / c) * (1.0 + 1e-12)

    @pytest.mark.parametrize("alpha", [0.0, 1.0])
    def test_integral_equation_by_independent_quadrature(self, alpha):
        """lambda psi(eta) = integral of exp(i c eta t) (1 - t^2)^alpha psi(t)
        over [-1, 1], with Gauss-Legendre quadrature (the weight is a
        polynomial for these alpha) instead of the Gauss-Jacobi of the
        module."""
        from scipy.special import roots_legendre

        c = 25.0
        t, w = roots_legendre(400)
        weighted = (
            w * (1.0 - t * t) ** alpha * generalized_prolate_spheroidal(t, c, alpha)
        )
        eta = np.linspace(0.0, 1.0, 51)
        integral = np.cos(c * np.multiply.outer(eta, t)) @ weighted
        eigenvalue = integral[0]
        np.testing.assert_allclose(
            integral / eigenvalue,
            generalized_prolate_spheroidal(eta, c, alpha),
            rtol=0,
            atol=1e-13,
        )

    def test_order_zero_for_a_large_bandwidth(self):
        """The function of order zero has no zero in [-1, 1] and falls
        monotonically from the centre, also where the transform's leading
        eigenvectors could not be separated (c = 30, alpha = 0)."""
        eta = np.linspace(0.0, 1.0, 2001)
        psi = generalized_prolate_spheroidal(eta, 30.0, 0.0)
        assert psi[0] == 1.0
        assert np.all(psi > 0.0)
        resolved = psi > 1e-12
        assert np.all(np.diff(psi[resolved]) < 0.0)
        np.testing.assert_array_equal(
            generalized_prolate_spheroidal(-eta, 30.0, 0.0), psi
        )

    def test_expansion_holds_on_the_band_only(self):
        with pytest.raises(ValueError, match="eta"):
            generalized_prolate_spheroidal(np.array([0.5, 1.5]), 10.0, 1.0)


class TestKernel:
    def test_registered(self):
        assert "prolate_spheroidal" in KERNEL_TYPES
        params = resolve_gridding_kernel_params(
            {"type": "prolate_spheroidal", "support": 11}
        )
        assert params["support"] == 11
        with pytest.raises(ValueError, match="fft_padding"):
            make_gridding_kernel({"type": "prolate_spheroidal"})

    def test_table_layout(self):
        """The table layout of the other kernels: entry k is the kernel
        k / oversampling cells from the centre, 1 at the centre, zero beyond
        the half width W / 2, and the kernel is even."""
        kernel = make_gridding_kernel(
            {"type": "prolate_spheroidal", "support": 9, "oversampling": 20},
            fft_padding=2.0,
        )
        c, alpha = SHAPE_TABLE[(9, 2.0)]
        assert kernel.shape["c"] == c and kernel.shape["alpha"] == alpha
        assert kernel.shape["fft_padding"] == 2.0
        assert kernel.table.size == 20 * 5
        assert kernel.table[0] == 1.0 == kernel.table.max()
        distance = np.arange(kernel.table.size) / 20
        np.testing.assert_array_equal(
            kernel.table,
            generalized_prolate_spheroidal_kernel(2.0 * distance / 9, c, alpha),
        )
        # zero from the half width of 4.5 cells on (where alpha > 0) or beyond
        # it, positive and falling within
        assert np.all(kernel.table[distance > 4.5] == 0.0)
        inside = kernel.table[distance < 4.5]
        assert np.all(inside > 0.0) and np.all(np.diff(inside) < 0.0)
        z = np.linspace(0.0, 1.2, 61)
        np.testing.assert_array_equal(
            generalized_prolate_spheroidal_kernel(-z, c, alpha),
            generalized_prolate_spheroidal_kernel(z, c, alpha),
        )
        stencil = kernel.stencil_values(np.array([0.0]))[0]
        np.testing.assert_array_equal(stencil, stencil[::-1])
        assert "c=" in repr(kernel) and "alpha=" in repr(kernel)

    def test_aliasing_at_support_7_padding_1p2(self):
        """The chosen (c, alpha) beat Schwab's kernel (half width 3 cells,
        c = 3 pi, alpha = 1) and the nominal bandwidth with alpha = 0; the
        exponential of a semicircle of the same support is printed for
        reference."""
        x0 = kept_half_width(1.2)
        c, alpha, error = generalized_prolate_spheroidal_parameters(7, x0)
        assert (c, alpha) == SHAPE_TABLE[(7, 1.2)]
        casa = _max_aliasing_error(
            lambda p: generalized_prolate_spheroidal_transform(p, 6, 3.0 * np.pi, 1.0),
            x0,
        )
        nominal = _max_aliasing_error(
            lambda p: generalized_prolate_spheroidal_transform(
                p, 7, nominal_bandwidth(7, x0), 0.0
            ),
            x0,
        )
        beta, mu, _ = exponential_semicircle_parameters(7, x0)
        semicircle = _max_aliasing_error(
            lambda p: exponential_semicircle_transform(p, 7, beta, mu), x0
        )
        print(
            f"largest aliasing error, W 7, padding 1.2: prolate_spheroidal "
            f"{error:.2e} (c {c}, alpha {alpha}), prolate_spheroidal_casa "
            f"{casa:.2e}, nominal c with alpha 0 {nominal:.2e}, "
            f"exponential_semicircle {semicircle:.2e}"
        )
        assert error < casa / 10
        assert error < nominal / 2
        assert error < 2.0 * semicircle

    @pytest.mark.parametrize("support", range(3, 18, 2))
    def test_any_odd_support(self, support):
        kernel = make_gridding_kernel(
            {"type": "prolate_spheroidal", "support": support, "oversampling": 10},
            fft_padding=1.25,
        )
        assert kernel.table.size == 10 * (support // 2 + 1)
        assert np.all(np.isfinite(kernel.table)) and kernel.table[0] == 1.0
        distance = np.arange(kernel.table.size) / 10
        assert np.all(kernel.table[distance > support / 2] == 0.0)
        assert np.all(np.isfinite(kernel.correction_1D(64)))

    @pytest.mark.parametrize(
        "support, fft_padding, smaller, larger",
        [(3, 1.1, None, 1.2), (7, 1.3, 1.25, 1.5), (17, 3.0, 2.0, None)],
    )
    def test_unlisted_padding_runs_the_optimiser(
        self, support, fft_padding, smaller, larger
    ):
        kernel = make_gridding_kernel(
            {"type": "prolate_spheroidal", "support": support, "oversampling": 10},
            fft_padding=fft_padding,
        )
        c, alpha = kernel.shape["c"], kernel.shape["alpha"]
        assert round(c, 4) == c and round(alpha, 4) == alpha
        assert (
            0.6 <= c / nominal_bandwidth(support, kept_half_width(fft_padding)) <= 1.4
        )
        assert 0.0 <= alpha <= 3.0
        assert np.all(np.isfinite(kernel.table)) and kernel.table[0] == 1.0
        # the error falls with the padding (down to the floor the quadrature
        # resolves), between those of the neighbouring paddings listed
        error = kernel.shape["aliasing_error"]
        for padding, is_smaller in ((smaller, True), (larger, False)):
            if padding is not None:
                listed = generalized_prolate_spheroidal_parameters(
                    support, kept_half_width(padding)
                )[2]
                assert error <= listed if is_smaller else error > listed

    def test_correction_is_the_table_transform(self):
        kernel = make_gridding_kernel(
            {
                "type": "prolate_spheroidal",
                "oversampling": 300,
                "interpolation": "linear",
            },
            fft_padding=1.2,
        )
        x = np.linspace(-0.5, 0.5, 101)
        np.testing.assert_allclose(
            kernel.correction(x),
            np.maximum(
                table_transform(kernel.table, 300, "linear", x)
                / table_transform(kernel.table, 300, "linear", 0.0),
                CORRECTION_FLOOR,
            ),
        )
        # the transform of the table applied is close to the continuous one
        c, alpha = kernel.shape["c"], kernel.shape["alpha"]
        kept = x[np.abs(x) <= kept_half_width(1.2)]
        continuous = generalized_prolate_spheroidal_transform(kept, 7, c, alpha)
        continuous /= generalized_prolate_spheroidal_transform(0.0, 7, c, alpha)
        np.testing.assert_allclose(kernel.correction(kept), continuous, rtol=1e-5)

    def test_create_table_matches_the_kernel_function(self):
        table = create_generalized_prolate_spheroidal_kernel_1D(8, 5, 9.0, 0.5)
        assert table.size == 8 * 3
        cells = np.arange(24) / 8
        np.testing.assert_array_equal(
            table, generalized_prolate_spheroidal_kernel(2.0 * cells / 5, 9.0, 0.5)
        )
        assert table[cells == 2.5] == 0.0 and np.all(table[cells > 2.5] == 0.0)
