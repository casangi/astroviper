"""Shared Gaussian math constants and conversion helpers.

All Gaussian-related code in the package should import from here rather than
inlining conversion factors or angle formulas.
"""

from __future__ import annotations

import numpy as np

# Conversion factors between Gaussian sigma and full width at half maximum.
# FWHM = SIG2FWHM * sigma,  sigma = FWHM2SIG * FWHM
SIG2FWHM: float = 2.0 * np.sqrt(2.0 * np.log(2.0))
FWHM2SIG: float = 1.0 / SIG2FWHM


def fwhm_from_sigma(sigma) -> np.ndarray:
    """Convert Gaussian sigma values to full width at half maximum.

    Parameters
    ----------
    sigma : array-like
        Scalar or array of Gaussian sigma values.

    Returns
    -------
    np.ndarray
        Values converted to FWHM using the standard Gaussian relation
        ``FWHM = 2 * sqrt(2 * ln(2)) * sigma``.

    Notes
    -----
    No positivity check is applied; callers are expected to pass physically
    meaningful widths.
    """
    return SIG2FWHM * np.asarray(sigma)


def sigma_from_fwhm(fwhm) -> np.ndarray:
    """Convert full width at half maximum values to Gaussian sigma.

    Parameters
    ----------
    fwhm : array-like
        Scalar or array of FWHM values.

    Returns
    -------
    np.ndarray
        Values converted to sigma using the standard Gaussian relation
        ``sigma = FWHM / (2 * sqrt(2 * ln(2)))``.

    Notes
    -----
    No positivity check is applied; callers are expected to pass physically
    meaningful widths.
    """
    return FWHM2SIG * np.asarray(fwhm)


# ---------------------------------------------------------------------------
# PA ↔ math angle conversion
#
# "math" convention: measured from +x toward +y.
# "pa"   convention: astronomical position angle, measured from +y toward +x.
# Relation: theta_math = pi/2 - PA  (and its inverse is identical in form).
# ---------------------------------------------------------------------------


def theta_pa_to_math(pa: np.ndarray) -> np.ndarray:
    """Convert position-angle values to the internal math-angle convention.

    Parameters
    ----------
    pa : array-like
        Angle values in radians measured from +y toward +x.

    Returns
    -------
    np.ndarray
        Equivalent angles in radians measured from +x toward +y, wrapped
        into ``[0, 2*pi)``.

    Notes
    -----
    This conversion is purely geometric and does not apply handedness flips.
    """
    return (np.pi / 2.0 - np.asarray(pa)) % (2.0 * np.pi)


def theta_math_to_pa(theta_math: np.ndarray) -> np.ndarray:
    """Convert internal math-angle values to position-angle values.

    Parameters
    ----------
    theta_math : array-like
        Angle values in radians measured from +x toward +y.

    Returns
    -------
    np.ndarray
        Equivalent PA angles in radians measured from +y toward +x, wrapped
        into ``[0, 2*pi)``.

    Notes
    -----
    This is the inverse mapping of :func:`theta_pa_to_math`.
    """
    return (np.pi / 2.0 - np.asarray(theta_math)) % (2.0 * np.pi)


def normalize_angle(angle_value: float, *, angle: str, degrees: bool) -> float:
    """Convert a user-provided angle to the internal math convention in radians.

    Parameters
    ----------
    angle_value : float
        Input angle, either in radians or degrees depending on ``degrees``.
    angle : {"math", "pa"}
        Convention of the input angle.
        Supported choices are:

        - ``"math"``: measured from +x toward +y; returned unchanged.
        - ``"pa"``: astronomical position angle measured from +y toward +x;
          converted via ``theta_math = pi/2 - PA``.
    degrees : bool
        If ``True``, ``angle_value`` is in degrees and is converted to radians
        before any convention mapping.

    Returns
    -------
    float
        Angle in radians in the internal math convention. No wrapping is
        applied; the raw formula result is returned.

    Notes
    -----
    Unlike :func:`theta_pa_to_math`, this function operates on a single scalar
    and does not wrap the result modulo ``2*pi``.
    """
    theta = float(np.deg2rad(angle_value) if degrees else angle_value)
    return (np.pi / 2.0) - theta if angle == "pa" else theta


# ---------------------------------------------------------------------------
# Gaussian deconvolution
#
# Given a measured (convolved) source Gaussian and a beam Gaussian, recover
# the intrinsic (deconvolved) source Gaussian by subtracting the beam's
# covariance matrix from the source's.
# ---------------------------------------------------------------------------


def _gauss_to_cov(sigma_maj, sigma_min, pa):
    """Build a 2×2 covariance matrix from Gaussian parameters.

    Parameters
    ----------
    sigma_maj : array-like
        Major-axis sigma (not FWHM).
    sigma_min : array-like
        Minor-axis sigma (not FWHM).
    pa : array-like
        Position angle in radians, measured from +y toward −x (east of north,
        standard astronomical convention).

    Returns
    -------
    tuple of (c00, c01, c11)
        Upper-triangle elements of the symmetric 2×2 covariance matrix in the
        (l, m) frame.  Full matrix is [[c00, c01], [c01, c11]].

    Notes
    -----
    PA is converted to the math angle (from +x toward +y) internally so that
    the rotation matrix aligns the major axis correctly in the (l, m) plane.
    """
    sigma_maj = np.asarray(sigma_maj, dtype=float)
    sigma_min = np.asarray(sigma_min, dtype=float)
    pa = np.asarray(pa, dtype=float)
    # Convert PA to math angle for rotation matrix
    th = np.pi / 2.0 - pa
    c = np.cos(th)
    s = np.sin(th)
    s2maj = sigma_maj**2
    s2min = sigma_min**2
    # C = R @ diag(σ_maj², σ_min²) @ R.T
    c00 = c * c * s2maj + s * s * s2min
    c01 = c * s * (s2maj - s2min)
    c11 = s * s * s2maj + c * c * s2min
    return c00, c01, c11


def _cov_to_gauss(c00, c01, c11):
    """Extract Gaussian parameters from a 2×2 covariance matrix.

    Parameters
    ----------
    c00, c01, c11 : array-like
        Upper-triangle elements of a symmetric positive-definite 2×2 matrix.

    Returns
    -------
    sigma_maj : np.ndarray
        Major-axis sigma.
    sigma_min : np.ndarray
        Minor-axis sigma.
    pa : np.ndarray
        Position angle in radians (astronomical convention: from +y toward −x).

    Notes
    -----
    Eigenvalues of the 2×2 matrix give σ² values. The math-convention angle
    of the eigenvector is converted back to PA.
    """
    c00 = np.asarray(c00, dtype=float)
    c01 = np.asarray(c01, dtype=float)
    c11 = np.asarray(c11, dtype=float)
    tr = c00 + c11
    det = c00 * c11 - c01 * c01
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    lam1 = tr / 2.0 + disc  # larger eigenvalue
    lam2 = tr / 2.0 - disc  # smaller eigenvalue
    sigma_maj = np.sqrt(np.maximum(lam1, 0.0))
    sigma_min = np.sqrt(np.maximum(lam2, 0.0))
    # Math-convention angle of the major eigenvector
    th_math = 0.5 * np.arctan2(2.0 * c01, c00 - c11)
    # Convert to PA
    pa = (np.pi / 2.0 - th_math) % np.pi
    return sigma_maj, sigma_min, pa


def deconvolve_gaussian(
    src_fwhm_maj,
    src_fwhm_min,
    src_pa,
    beam_fwhm_maj,
    beam_fwhm_min,
    beam_pa,
):
    """Deconvolve a beam Gaussian from a measured source Gaussian.

    Parameters
    ----------
    src_fwhm_maj : array-like
        Measured source major-axis FWHM.
    src_fwhm_min : array-like
        Measured source minor-axis FWHM.
    src_pa : array-like
        Measured source position angle in radians (east of north).
    beam_fwhm_maj : array-like
        Beam major-axis FWHM (same units as source).
    beam_fwhm_min : array-like
        Beam minor-axis FWHM.
    beam_pa : array-like
        Beam position angle in radians (east of north).

    Returns
    -------
    deconv_fwhm_maj : np.ndarray
        Deconvolved major-axis FWHM. NaN where unresolved.
    deconv_fwhm_min : np.ndarray
        Deconvolved minor-axis FWHM. NaN where unresolved.
    deconv_pa : np.ndarray
        Deconvolved position angle in radians. NaN where unresolved.
    is_unresolved : np.ndarray
        Boolean, True where the source is unresolved (deconvolution failed
        because the source is smaller than or equal to the beam).

    Notes
    -----
    The algorithm subtracts the beam covariance matrix from the source
    covariance matrix. If the resulting matrix is not positive-definite,
    the source is unresolved and NaN values are returned for the
    deconvolved parameters.
    """
    src_s_maj = FWHM2SIG * np.asarray(src_fwhm_maj, dtype=float)
    src_s_min = FWHM2SIG * np.asarray(src_fwhm_min, dtype=float)
    beam_s_maj = FWHM2SIG * np.asarray(beam_fwhm_maj, dtype=float)
    beam_s_min = FWHM2SIG * np.asarray(beam_fwhm_min, dtype=float)

    sc00, sc01, sc11 = _gauss_to_cov(src_s_maj, src_s_min, src_pa)
    bc00, bc01, bc11 = _gauss_to_cov(beam_s_maj, beam_s_min, beam_pa)

    dc00 = sc00 - bc00
    dc01 = sc01 - bc01
    dc11 = sc11 - bc11

    # Check positive-definiteness: both eigenvalues must be > 0.
    tr = dc00 + dc11
    det = dc00 * dc11 - dc01 * dc01
    # Positive-definite iff trace > 0 and determinant > 0.
    resolved = (tr > 0) & (det > 0)

    d_s_maj, d_s_min, d_pa = _cov_to_gauss(dc00, dc01, dc11)

    deconv_fwhm_maj = np.where(resolved, SIG2FWHM * d_s_maj, np.nan)
    deconv_fwhm_min = np.where(resolved, SIG2FWHM * d_s_min, np.nan)
    deconv_pa = np.where(resolved, d_pa, np.nan)
    is_unresolved = ~resolved

    return deconv_fwhm_maj, deconv_fwhm_min, deconv_pa, is_unresolved


def deconvolve_gaussian_with_errors(
    src_fwhm_maj,
    src_fwhm_min,
    src_pa,
    src_fwhm_maj_err,
    src_fwhm_min_err,
    src_pa_err,
    beam_fwhm_maj,
    beam_fwhm_min,
    beam_pa,
):
    """Deconvolve a beam from a source with analytic error propagation.

    Parameters
    ----------
    src_fwhm_maj : array-like
        Measured source major-axis FWHM.
    src_fwhm_min : array-like
        Measured source minor-axis FWHM.
    src_pa : array-like
        Measured source PA in radians (east of north).
    src_fwhm_maj_err : array-like
        1σ uncertainty on source major-axis FWHM.
    src_fwhm_min_err : array-like
        1σ uncertainty on source minor-axis FWHM.
    src_pa_err : array-like
        1σ uncertainty on source PA in radians.
    beam_fwhm_maj : array-like
        Beam major-axis FWHM (assumed exact — no uncertainty).
    beam_fwhm_min : array-like
        Beam minor-axis FWHM.
    beam_pa : array-like
        Beam PA in radians.

    Returns
    -------
    deconv_fwhm_maj : np.ndarray
        Deconvolved major-axis FWHM. NaN where unresolved.
    deconv_fwhm_min : np.ndarray
        Deconvolved minor-axis FWHM. NaN where unresolved.
    deconv_pa : np.ndarray
        Deconvolved PA in radians. NaN where unresolved.
    deconv_fwhm_maj_err : np.ndarray
        Propagated 1σ uncertainty on deconvolved major FWHM. NaN where
        unresolved or marginally resolved.
    deconv_fwhm_min_err : np.ndarray
        Propagated 1σ uncertainty on deconvolved minor FWHM.
    deconv_pa_err : np.ndarray
        Propagated 1σ uncertainty on deconvolved PA.
    is_unresolved : np.ndarray
        Boolean, True where source is unresolved.
    is_marginally_resolved : np.ndarray
        Boolean, True where the smaller eigenvalue of the deconvolved
        covariance is positive but very small relative to the beam, making
        error estimates unreliable.

    Notes
    -----
    Error propagation uses the analytic Jacobian of the eigendecomposition
    of the 2×2 deconvolved covariance matrix with respect to the source
    parameters. The beam is treated as exact (zero uncertainty).

    Near the unresolved boundary the Jacobian diverges, so components whose
    smaller deconvolved eigenvalue is less than 1% of the beam's smaller
    eigenvalue are flagged as marginally resolved and their uncertainties
    are set to NaN.
    """
    src_fwhm_maj = np.asarray(src_fwhm_maj, dtype=float)
    src_fwhm_min = np.asarray(src_fwhm_min, dtype=float)
    src_pa = np.asarray(src_pa, dtype=float)
    src_fwhm_maj_err = np.asarray(src_fwhm_maj_err, dtype=float)
    src_fwhm_min_err = np.asarray(src_fwhm_min_err, dtype=float)
    src_pa_err = np.asarray(src_pa_err, dtype=float)

    # First get the deconvolved values
    d_fwhm_maj, d_fwhm_min, d_pa, is_unresolved = deconvolve_gaussian(
        src_fwhm_maj,
        src_fwhm_min,
        src_pa,
        beam_fwhm_maj,
        beam_fwhm_min,
        beam_pa,
    )

    # Convert source to sigma for Jacobian computation
    src_s_maj = FWHM2SIG * src_fwhm_maj
    src_s_min = FWHM2SIG * src_fwhm_min
    beam_s_min = FWHM2SIG * np.asarray(beam_fwhm_min, dtype=float)

    # Build deconvolved covariance matrix elements
    sc00, sc01, sc11 = _gauss_to_cov(src_s_maj, src_s_min, src_pa)
    bc00, bc01, bc11 = _gauss_to_cov(
        FWHM2SIG * np.asarray(beam_fwhm_maj, dtype=float),
        beam_s_min,
        np.asarray(beam_pa, dtype=float),
    )
    dc00 = sc00 - bc00
    dc01 = sc01 - bc01
    dc11 = sc11 - bc11

    # Eigenvalues of the deconvolved covariance
    tr = dc00 + dc11
    det = dc00 * dc11 - dc01 * dc01
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    lam1 = tr / 2.0 + disc  # major eigenvalue
    lam2 = tr / 2.0 - disc  # minor eigenvalue

    # Flag marginally resolved: minor eigenvalue < 1% of beam minor eigenvalue
    beam_lam_min = beam_s_min**2
    marginal_threshold = 0.01 * beam_lam_min
    is_marginally_resolved = (~is_unresolved) & (lam2 < marginal_threshold)

    # --- Analytic Jacobian of (fwhm_maj_d, fwhm_min_d, pa_d) w.r.t.
    #     (fwhm_maj_s, fwhm_min_s, pa_s) ---
    #
    # Chain: source params → covariance matrix entries → eigenvalues/vectors
    #        → deconvolved Gaussian params.
    #
    # Step 1: ∂(c00, c01, c11)/∂(σ_maj, σ_min, pa) for the source.
    #         Since C_deconv = C_src - C_beam and beam is constant,
    #         ∂C_deconv/∂(source params) = ∂C_src/∂(source params).

    th_math = np.pi / 2.0 - src_pa
    c = np.cos(th_math)
    s = np.sin(th_math)

    # ∂C_src/∂σ_maj (chain through FWHM: ∂σ/∂fwhm = FWHM2SIG)
    dc00_dsmaj = 2.0 * src_s_maj * c * c
    dc01_dsmaj = 2.0 * src_s_maj * c * s
    dc11_dsmaj = 2.0 * src_s_maj * s * s

    dc00_dsmin = 2.0 * src_s_min * s * s
    dc01_dsmin = -2.0 * src_s_min * c * s
    dc11_dsmin = 2.0 * src_s_min * c * c

    # ∂C_src/∂pa (note: ∂th_math/∂pa = -1)
    s2maj = src_s_maj**2
    s2min = src_s_min**2
    diff_s2 = s2maj - s2min
    dc00_dpa = 2.0 * c * s * diff_s2  # negated twice: ∂/∂th * (-1)
    dc01_dpa = (c * c - s * s) * diff_s2 * (-1.0)
    dc11_dpa = -2.0 * c * s * diff_s2

    # Step 2: ∂(lam1, lam2, th_eig)/∂(c00, c01, c11) for 2×2 symmetric matrix.
    #
    # For a 2×2 symmetric [[a, b], [b, d]]:
    #   lam = (a+d)/2 ± sqrt(((a-d)/2)² + b²)
    #   th = 0.5 * arctan2(2b, a-d)
    safe_disc = np.where(disc > 0, disc, 1.0)  # avoid division by zero

    # ∂lam1/∂(c00, c01, c11)
    dlam1_dc00 = 0.5 + 0.5 * (dc00 - dc11) / (2.0 * safe_disc)
    dlam1_dc01 = 2.0 * dc01 / (2.0 * safe_disc)
    dlam1_dc11 = 0.5 - 0.5 * (dc00 - dc11) / (2.0 * safe_disc)

    dlam2_dc00 = 0.5 - 0.5 * (dc00 - dc11) / (2.0 * safe_disc)
    dlam2_dc01 = -2.0 * dc01 / (2.0 * safe_disc)
    dlam2_dc11 = 0.5 + 0.5 * (dc00 - dc11) / (2.0 * safe_disc)

    # ∂th_eig/∂(c00, c01, c11) where th_eig = 0.5 * arctan2(2*c01, c00-c11)
    denom_th = (dc00 - dc11) ** 2 + 4.0 * dc01**2
    safe_denom_th = np.where(denom_th > 0, denom_th, 1.0)
    dth_dc00 = -dc01 / safe_denom_th
    dth_dc01 = (dc00 - dc11) / safe_denom_th
    dth_dc11 = dc01 / safe_denom_th

    # Step 3: ∂(σ_maj_d, σ_min_d)/∂(lam1, lam2) = 1/(2*σ)
    safe_lam1 = np.where(lam1 > 0, lam1, 1.0)
    safe_lam2 = np.where(lam2 > 0, lam2, 1.0)
    dsmaj_dlam1 = 0.5 / np.sqrt(safe_lam1)
    dsmin_dlam2 = 0.5 / np.sqrt(safe_lam2)

    # ∂fwhm/∂σ = SIG2FWHM, ∂σ_src/∂fwhm_src = FWHM2SIG
    # Full chain for each output w.r.t. each input:
    # ∂fwhm_d/∂fwhm_s = SIG2FWHM * ∂σ_d/∂lam * ∂lam/∂C * ∂C/∂σ_s * FWHM2SIG

    def _chain_fwhm_err(dlam_dc00, dlam_dc01, dlam_dc11, dsig_dlam):
        """Propagate error for one deconvolved FWHM output."""
        # ∂fwhm_d/∂fwhm_maj_s
        grad_maj = (
            SIG2FWHM
            * dsig_dlam
            * (dlam_dc00 * dc00_dsmaj + dlam_dc01 * dc01_dsmaj + dlam_dc11 * dc11_dsmaj)
            * FWHM2SIG
        )
        # ∂fwhm_d/∂fwhm_min_s
        grad_min = (
            SIG2FWHM
            * dsig_dlam
            * (dlam_dc00 * dc00_dsmin + dlam_dc01 * dc01_dsmin + dlam_dc11 * dc11_dsmin)
            * FWHM2SIG
        )
        # ∂fwhm_d/∂pa_s
        grad_pa = (
            SIG2FWHM
            * dsig_dlam
            * (dlam_dc00 * dc00_dpa + dlam_dc01 * dc01_dpa + dlam_dc11 * dc11_dpa)
        )
        return np.sqrt(
            (grad_maj * src_fwhm_maj_err) ** 2
            + (grad_min * src_fwhm_min_err) ** 2
            + (grad_pa * src_pa_err) ** 2
        )

    fwhm_maj_err = _chain_fwhm_err(dlam1_dc00, dlam1_dc01, dlam1_dc11, dsmaj_dlam1)
    fwhm_min_err = _chain_fwhm_err(dlam2_dc00, dlam2_dc01, dlam2_dc11, dsmin_dlam2)

    # PA error: ∂pa_d/∂pa_s chain
    # pa_d = pi/2 - th_eig, so ∂pa_d/∂th_eig = -1
    # ∂pa_d/∂C = -∂th_eig/∂C
    grad_pa_maj = (
        -(dth_dc00 * dc00_dsmaj + dth_dc01 * dc01_dsmaj + dth_dc11 * dc11_dsmaj)
        * FWHM2SIG
    )
    grad_pa_min = (
        -(dth_dc00 * dc00_dsmin + dth_dc01 * dc01_dsmin + dth_dc11 * dc11_dsmin)
        * FWHM2SIG
    )
    grad_pa_pa = -(dth_dc00 * dc00_dpa + dth_dc01 * dc01_dpa + dth_dc11 * dc11_dpa)
    pa_err = np.sqrt(
        (grad_pa_maj * src_fwhm_maj_err) ** 2
        + (grad_pa_min * src_fwhm_min_err) ** 2
        + (grad_pa_pa * src_pa_err) ** 2
    )

    # Mask out errors where unresolved or marginally resolved
    bad = is_unresolved | is_marginally_resolved
    deconv_fwhm_maj_err = np.where(bad, np.nan, fwhm_maj_err)
    deconv_fwhm_min_err = np.where(bad, np.nan, fwhm_min_err)
    deconv_pa_err = np.where(bad, np.nan, pa_err)

    return (
        d_fwhm_maj,
        d_fwhm_min,
        d_pa,
        deconv_fwhm_maj_err,
        deconv_fwhm_min_err,
        deconv_pa_err,
        is_unresolved,
        is_marginally_resolved,
    )
