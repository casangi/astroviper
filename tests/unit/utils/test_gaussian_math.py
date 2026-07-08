"""Tests for Gaussian deconvolution helpers in _gaussian_math.py."""

import numpy as np
import pytest

from astroviper.utils._gaussian_math import (
    FWHM2SIG,
    SIG2FWHM,
    _cov_to_gauss,
    _gauss_to_cov,
    deconvolve_gaussian,
    deconvolve_gaussian_with_errors,
)

# ---------------------------------------------------------------------------
# _gauss_to_cov / _cov_to_gauss round-trip
# ---------------------------------------------------------------------------


class TestCovRoundTrip:
    """Verify covariance matrix construction and decomposition are inverses."""

    @pytest.mark.parametrize(
        "sigma_maj, sigma_min, pa",
        [
            (5.0, 3.0, 0.0),
            (5.0, 3.0, np.pi / 4),
            (5.0, 3.0, np.pi / 2),
            (5.0, 3.0, 3 * np.pi / 4),
            (4.0, 4.0, 0.3),  # circular — PA is degenerate
            (10.0, 1.0, 1.2),
        ],
    )
    def test_round_trip_scalar(self, sigma_maj, sigma_min, pa):
        c00, c01, c11 = _gauss_to_cov(sigma_maj, sigma_min, pa)
        maj_out, min_out, pa_out = _cov_to_gauss(c00, c01, c11)
        np.testing.assert_allclose(maj_out, sigma_maj, atol=1e-12)
        np.testing.assert_allclose(min_out, sigma_min, atol=1e-12)
        if not np.isclose(sigma_maj, sigma_min):
            # PA wraps to [0, pi), so compare modulo pi
            delta = (pa_out - (pa % np.pi)) % np.pi
            delta = min(delta, np.pi - delta)
            assert delta < 1e-12

    def test_round_trip_array(self):
        majs = np.array([5.0, 8.0, 3.0])
        mins = np.array([3.0, 2.0, 1.0])
        pas = np.array([0.3, 1.0, 0.0])
        c00, c01, c11 = _gauss_to_cov(majs, mins, pas)
        maj_out, min_out, pa_out = _cov_to_gauss(c00, c01, c11)
        np.testing.assert_allclose(maj_out, majs, atol=1e-12)
        np.testing.assert_allclose(min_out, mins, atol=1e-12)


# ---------------------------------------------------------------------------
# deconvolve_gaussian
# ---------------------------------------------------------------------------


class TestDeconvolveGaussian:
    """Tests for the core deconvolution function."""

    def test_circular_source_and_beam(self):
        """Convolve a circular source with a circular beam, then deconvolve."""
        src_true_fwhm = 6.0
        beam_fwhm = 4.0
        # Convolved: sqrt(6² + 4²) = sqrt(52)
        conv_fwhm = np.sqrt(src_true_fwhm**2 + beam_fwhm**2)
        d_maj, d_min, d_pa, unres = deconvolve_gaussian(
            conv_fwhm, conv_fwhm, 0.0, beam_fwhm, beam_fwhm, 0.0
        )
        np.testing.assert_allclose(d_maj, src_true_fwhm, atol=1e-10)
        np.testing.assert_allclose(d_min, src_true_fwhm, atol=1e-10)
        assert not unres

    def test_elliptical_aligned(self):
        """Elliptical source convolved with circular beam, same PA."""
        src_maj = 8.0
        src_min = 4.0
        beam = 3.0
        pa = 0.5
        # Convolved sigmas add in quadrature along each principal axis
        conv_maj = np.sqrt(src_maj**2 + beam**2)
        conv_min = np.sqrt(src_min**2 + beam**2)
        d_maj, d_min, d_pa, unres = deconvolve_gaussian(
            conv_maj, conv_min, pa, beam, beam, 0.0
        )
        # The PA will shift because source and beam have different PAs,
        # but for circular beam the source PA should be preserved.
        np.testing.assert_allclose(d_maj, src_maj, atol=1e-8)
        np.testing.assert_allclose(d_min, src_min, atol=1e-8)
        assert not unres

    def test_unresolved(self):
        """Source smaller than beam → unresolved."""
        d_maj, d_min, d_pa, unres = deconvolve_gaussian(3.0, 2.0, 0.0, 5.0, 5.0, 0.0)
        assert unres
        assert np.isnan(d_maj)
        assert np.isnan(d_min)
        assert np.isnan(d_pa)

    def test_equal_to_beam(self):
        """Source equal to beam → unresolved (zero deconvolved size)."""
        d_maj, d_min, d_pa, unres = deconvolve_gaussian(5.0, 3.0, 0.7, 5.0, 3.0, 0.7)
        # Deconvolved covariance is zero → eigenvalues are zero → det = 0
        assert unres

    def test_vectorized(self):
        """Multiple sources at once."""
        src_maj = np.array([10.0, 3.0, 8.0])
        src_min = np.array([8.0, 2.0, 6.0])
        src_pa = np.array([0.0, 0.5, 1.0])
        beam_maj = np.array([5.0, 5.0, 5.0])
        beam_min = np.array([5.0, 5.0, 5.0])
        beam_pa = np.array([0.0, 0.0, 0.0])
        d_maj, d_min, d_pa, unres = deconvolve_gaussian(
            src_maj, src_min, src_pa, beam_maj, beam_min, beam_pa
        )
        assert d_maj.shape == (3,)
        # Second source (3x2) is smaller than beam (5x5) → unresolved
        assert not unres[0]
        assert unres[1]
        assert not unres[2]

    def test_elliptical_source_elliptical_beam_different_pa(self):
        """Elliptical source and beam at different PAs — full covariance test."""
        # Build ground truth by constructing covariance matrices directly
        src_s_maj, src_s_min, src_pa = 6.0, 3.0, np.deg2rad(30)
        beam_s_maj, beam_s_min, beam_pa = 4.0, 2.0, np.deg2rad(60)
        # Convolved = source + beam in covariance space
        sc = _gauss_to_cov(FWHM2SIG * src_s_maj, FWHM2SIG * src_s_min, src_pa)
        bc = _gauss_to_cov(FWHM2SIG * beam_s_maj, FWHM2SIG * beam_s_min, beam_pa)
        conv_c = (sc[0] + bc[0], sc[1] + bc[1], sc[2] + bc[2])
        conv_s_maj, conv_s_min, conv_pa = _cov_to_gauss(*conv_c)
        conv_fwhm_maj = SIG2FWHM * conv_s_maj
        conv_fwhm_min = SIG2FWHM * conv_s_min

        d_maj, d_min, d_pa, unres = deconvolve_gaussian(
            conv_fwhm_maj,
            conv_fwhm_min,
            conv_pa,
            beam_s_maj,
            beam_s_min,
            beam_pa,
        )
        assert not unres
        np.testing.assert_allclose(d_maj, src_s_maj, atol=1e-8)
        np.testing.assert_allclose(d_min, src_s_min, atol=1e-8)
        delta = (d_pa - (src_pa % np.pi)) % np.pi
        delta = min(delta, np.pi - delta)
        assert delta < 1e-8


# ---------------------------------------------------------------------------
# deconvolve_gaussian_with_errors
# ---------------------------------------------------------------------------


class TestDeconvolveGaussianWithErrors:
    """Tests for deconvolution with analytic error propagation."""

    def _finite_diff_errors(
        self,
        src_fwhm_maj,
        src_fwhm_min,
        src_pa,
        beam_fwhm_maj,
        beam_fwhm_min,
        beam_pa,
        eps=1e-5,
    ):
        """Compute deconvolved parameter derivatives via finite differences."""
        base = deconvolve_gaussian(
            src_fwhm_maj,
            src_fwhm_min,
            src_pa,
            beam_fwhm_maj,
            beam_fwhm_min,
            beam_pa,
        )
        jac = np.zeros((3, 3))  # [output_idx, input_idx]
        for j, (arr, idx) in enumerate([("maj", 0), ("min", 1), ("pa", 2)]):
            params = [src_fwhm_maj, src_fwhm_min, src_pa]
            params[j] = params[j] + eps
            plus = deconvolve_gaussian(
                params[0],
                params[1],
                params[2],
                beam_fwhm_maj,
                beam_fwhm_min,
                beam_pa,
            )
            params[j] = params[j] - 2 * eps
            minus = deconvolve_gaussian(
                params[0],
                params[1],
                params[2],
                beam_fwhm_maj,
                beam_fwhm_min,
                beam_pa,
            )
            for i in range(3):
                jac[i, j] = (plus[i] - minus[i]) / (2 * eps)
        return jac

    def test_errors_match_finite_diff_circular(self):
        """Analytic errors should match finite-difference Jacobian for circular case.

        For a circular deconvolved source the PA is degenerate, so only the
        FWHM errors are compared.
        """
        src_fwhm = 10.0
        beam_fwhm = 4.0
        conv_fwhm = np.sqrt(src_fwhm**2 + beam_fwhm**2)
        errs = np.array([0.3, 0.3, 0.05])

        result = deconvolve_gaussian_with_errors(
            conv_fwhm,
            conv_fwhm,
            0.0,
            errs[0],
            errs[1],
            errs[2],
            beam_fwhm,
            beam_fwhm,
            0.0,
        )
        analytic_errs = np.array([result[3], result[4]])

        jac = self._finite_diff_errors(
            conv_fwhm,
            conv_fwhm,
            0.0,
            beam_fwhm,
            beam_fwhm,
            0.0,
        )
        fd_errs = np.sqrt((jac**2) @ (errs**2))
        # Only compare FWHM errors; PA error is degenerate for circular sources
        np.testing.assert_allclose(analytic_errs, fd_errs[:2], rtol=1e-3)

    def test_errors_match_finite_diff_elliptical(self):
        """Analytic errors should match finite-difference Jacobian for elliptical case."""
        src_true_maj, src_true_min, src_true_pa = 8.0, 4.0, 0.6
        beam_maj, beam_min, beam_pa = 3.0, 2.0, 0.2
        # Build convolved source
        sc = _gauss_to_cov(
            FWHM2SIG * src_true_maj, FWHM2SIG * src_true_min, src_true_pa
        )
        bc = _gauss_to_cov(FWHM2SIG * beam_maj, FWHM2SIG * beam_min, beam_pa)
        conv_s_maj, conv_s_min, conv_pa = _cov_to_gauss(
            sc[0] + bc[0], sc[1] + bc[1], sc[2] + bc[2]
        )
        conv_fwhm_maj = SIG2FWHM * conv_s_maj
        conv_fwhm_min = SIG2FWHM * conv_s_min

        errs = np.array([0.2, 0.15, 0.03])
        result = deconvolve_gaussian_with_errors(
            conv_fwhm_maj,
            conv_fwhm_min,
            conv_pa,
            errs[0],
            errs[1],
            errs[2],
            beam_maj,
            beam_min,
            beam_pa,
        )
        analytic_errs = np.array([result[3], result[4], result[5]])

        jac = self._finite_diff_errors(
            conv_fwhm_maj,
            conv_fwhm_min,
            conv_pa,
            beam_maj,
            beam_min,
            beam_pa,
        )
        fd_errs = np.sqrt((jac**2) @ (errs**2))
        np.testing.assert_allclose(analytic_errs, fd_errs, rtol=1e-2)

    def test_unresolved_gives_nan_errors(self):
        """Unresolved source should have NaN errors."""
        result = deconvolve_gaussian_with_errors(
            3.0,
            2.0,
            0.0,
            0.5,
            0.5,
            0.1,
            5.0,
            5.0,
            0.0,
        )
        assert result[6]  # is_unresolved
        assert np.isnan(result[3])  # fwhm_maj_err
        assert np.isnan(result[4])  # fwhm_min_err
        assert np.isnan(result[5])  # pa_err

    def test_marginally_resolved_flagged(self):
        """Source barely larger than beam should be flagged marginally resolved."""
        beam_fwhm = 5.0
        # Source just barely larger — minor axis only slightly above beam
        src_fwhm_maj = 5.1
        src_fwhm_min = 5.001  # very close to beam
        result = deconvolve_gaussian_with_errors(
            src_fwhm_maj,
            src_fwhm_min,
            0.0,
            0.1,
            0.1,
            0.01,
            beam_fwhm,
            beam_fwhm,
            0.0,
        )
        # Should be resolved (not unresolved) but marginally so
        assert not result[6]  # is_unresolved
        assert result[7]  # is_marginally_resolved
        assert np.isnan(result[3])  # errors NaN when marginal
