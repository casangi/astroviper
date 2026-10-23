"""Diagnose spatially coupled Taylor Hessians without a pixel-space matrix."""

import warnings

import numpy as np


def check_spatial_hessian_blocks(
    transfer,
    frequency_l,
    frequency_m,
    *,
    precision,
    frequency_units,
    context=None,
):
    """Inspect Fourier blocks of a periodic, shift-invariant Taylor operator.

    Parameters
    ----------
    transfer : ndarray
        Fourier Taylor PSFs, shape ``(2*nterms-1, n_l_modes, n_m_modes)``.
        These may come from actual PSFs or an analytic Gaussian model.
    frequency_l, frequency_m : array-like
        Spatial Fourier coordinates along the two image axes.
    precision : {"float32", "float64"}
        Working precision for support and numerical-rank tolerances.
    frequency_units : str
        Units of the supplied spatial coordinates, e.g. cycles_per_radian.
    context : mapping, optional
        Stage, field, time, and polarization identifiers for warnings.

    Returns
    -------
    dict
        Counts, bounded examples of problematic modes, and worst order-zero
        amplification. Missing spatial modes are reported, not inverted.

    Notes
    -----
    For delta components, H_(k,a),(l,b) = P_(k+l)(theta_a-theta_b).
    A spatial Fourier transform diagonalizes convolution, leaving the small
    blocks H_kl(u) = Fourier(P_(k+l))(u). This preserves interactions between
    distinct component positions; it does not replace PSFs by central values.
    The boundary model is periodic, and spatial masks are not included.
    """
    transfer = np.asarray(transfer)
    frequency_l = np.asarray(frequency_l)
    frequency_m = np.asarray(frequency_m)
    if (
        transfer.ndim != 3
        or frequency_l.ndim != 1
        or frequency_m.ndim != 1
        or not np.all(np.isfinite(frequency_l))
        or not np.all(np.isfinite(frequency_m))
        or transfer.shape[0] % 2 != 1
        or transfer.shape[1:] != (frequency_l.size, frequency_m.size)
        or not transfer.size
        or not np.all(np.isfinite(transfer))
    ):
        raise ValueError(
            "Spatial Taylor-Hessian transfers must be finite, nonempty, and have matching axes and an odd Taylor count."
        )
    nterms = (transfer.shape[0] + 1) // 2
    if np.dtype(precision) not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Spatial Taylor-Hessian precision must be float32 or float64.")
    epsilon = np.finfo(precision).eps
    relative_tolerance = nterms * epsilon
    values = transfer.reshape(transfer.shape[0], -1)

    # --- Spatial support ---
    # Interferometers need not measure every image-grid Fourier mode. A weak
    # order-zero response is missing support, not evidence that the requested
    # Taylor model is invalid. Use a working-precision relative floor rather
    # than inverting those modes or imposing a scientific bandwidth cutoff.
    # The floor is a numerical convention, not a sensitivity/noise estimate.
    support_scale = float(np.max(np.abs(values[0])))
    support_tolerance = relative_tolerance * support_scale
    order_scales = np.max(np.abs(values), axis=1)
    result = {
        "representation": "periodic_spatial_fourier_blocks",
        "frequency_units": frequency_units,
        "spatial_wavelength_units": "radians"
        if frequency_units == "cycles_per_radian"
        else "pixels",
        "boundary_assumption": "periodic_shift_invariant",
        "covariance_assumption": "proportional_to_spatial_hessian_block",
        "n_taylor_terms": nterms,
        "mode_count": values.shape[1],
        "support_relative_tolerance": float(relative_tolerance),
        "support_absolute_tolerance": float(support_tolerance),
        "unsupported_mode_count": 0,
        "supported_mode_count": 0,
        "non_hermitian_mode_count": 0,
        "non_positive_mode_count": 0,
        "rank_deficient_mode_count": 0,
        "assessed_mode_count": 0,
        "excess_noise_mode_count": 0,
        "order_zero_noise_amplification_limit": 3.0,
        "max_order_zero_noise_amplification": None,
        "minimum_relative_eigenvalue": None,
        "worst_mode": None,
        "examples": {},
    }

    def mode_coordinates(index):
        row, column = divmod(int(index), frequency_m.size)
        first, second = float(frequency_l[row]), float(frequency_m[column])
        radius = np.hypot(first, second)
        return {
            "frequency_l": first,
            "frequency_m": second,
            "spatial_wavelength": float(1 / radius) if radius > 0 else None,
        }

    def record(name, indices):
        result[f"{name}_mode_count"] += int(len(indices))
        examples = result["examples"].setdefault(name, [])
        examples.extend(
            mode_coordinates(index) for index in indices[: max(0, 4 - len(examples))]
        )

    # --- Taylor blocks on supported modes ---
    # Assemble only bounded batches of p-by-p matrices. Never allocate the
    # (p * n_pixels)-squared spatial Hessian. The Fourier PSFs remain read-only.
    order_indices = np.add.outer(np.arange(nterms), np.arange(nterms))
    for start in range(0, values.shape[1], 4096):
        block_values = values[:, start : start + 4096]
        indices = np.arange(start, start + block_values.shape[1])
        supported = np.abs(block_values[0]) > support_tolerance
        result["unsupported_mode_count"] += int((~supported).sum())
        result["supported_mode_count"] += int(supported.sum())

        # A convolution Hessian must be self-adjoint. An imaginary transfer
        # indicates an asymmetric PSF after centering. Report it rather than
        # silently discarding the imaginary part and asserting positive rank.
        hermitian = np.all(
            np.abs(block_values.imag) <= relative_tolerance * order_scales[:, None],
            axis=0,
        )
        record("non_hermitian", indices[supported & ~hermitian])
        selected = supported & hermitian
        indices = indices[selected]
        if not indices.size:
            continue
        matrices = block_values[:, selected].real[order_indices].transpose(2, 0, 1)
        diagonal = np.diagonal(matrices, axis1=1, axis2=2)
        positive_diagonal = np.all(diagonal > 0, axis=1)
        negative_diagonal = np.any(diagonal < 0, axis=1)
        record("non_positive", indices[~positive_diagonal & negative_diagonal])
        record("rank_deficient", indices[~positive_diagonal & ~negative_diagonal])
        matrices = matrices[positive_diagonal]
        indices = indices[positive_diagonal]
        if not indices.size:
            continue

        # --- Positive curvature and independent Taylor directions ---
        # Equilibrate before testing eigenvalues, so small higher-order units
        # do not masquerade as dependence. A deficient spatial mode is reported
        # separately from a globally unusable Taylor selection/local inverse.
        diagonal_sqrt = np.sqrt(np.diagonal(matrices, axis1=1, axis2=2))
        equilibrated = matrices / diagonal_sqrt[:, :, None] / diagonal_sqrt[:, None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
        relative_eigenvalues = eigenvalues[:, 0] / np.max(np.abs(eigenvalues), axis=1)
        minimum = float(relative_eigenvalues.min())
        if (
            result["minimum_relative_eigenvalue"] is None
            or minimum < result["minimum_relative_eigenvalue"]
        ):
            result["minimum_relative_eigenvalue"] = minimum
        tolerances = relative_tolerance * np.max(np.abs(eigenvalues), axis=1)
        negative = eigenvalues[:, 0] < -tolerances
        deficient = (eigenvalues[:, 0] <= tolerances) & ~negative
        record("non_positive", indices[negative])
        record("rank_deficient", indices[deficient])
        safe = ~(negative | deficient)
        indices = indices[safe]
        if not indices.size:
            continue

        # --- Order-zero noise amplification ---
        # If C_R(u) = tau**2 * H(u), comparison with a constant-only solve gives
        # G0(u)**2 = H00(u) * inverse(H(u))00. Unit-diagonal equilibration makes
        # this sum_j |V_0j|**2 / lambda_j. The covariance relation is assumed,
        # not measured. The limit three is the agreed diagnostic policy.
        amplification = np.sqrt(
            np.sum(eigenvectors[safe, 0, :] ** 2 / eigenvalues[safe], axis=1)
        )
        result["assessed_mode_count"] += int(indices.size)
        record("excess_noise", indices[amplification > 3.0])
        maximum_index = int(np.argmax(amplification))
        maximum = float(amplification[maximum_index])
        if (
            result["max_order_zero_noise_amplification"] is None
            or maximum > result["max_order_zero_noise_amplification"]
        ):
            result["max_order_zero_noise_amplification"] = maximum
            result["worst_mode"] = mode_coordinates(indices[maximum_index])

    # --- Diagnostic policy ---
    # Missing Fourier modes are expected, and a global pixel-space inverse is
    # not the CLEAN algorithm. Warn about problematic supported blocks instead
    # of rejecting the configuration or changing the model/PSF. Local native
    # inversion guards still reject unsafe component updates independently.
    if (
        any(
            result[f"{name}_mode_count"]
            for name in (
                "non_hermitian",
                "non_positive",
                "rank_deficient",
                "excess_noise",
            )
        )
        or not result["supported_mode_count"]
    ):
        counts = {
            key: value for key, value in result.items() if key.endswith("mode_count")
        }
        warnings.warn(
            f"Spatial Taylor-Hessian diagnostic ({context or {}}): {counts}; max_order_zero_noise_amplification={result['max_order_zero_noise_amplification']}; covariance_assumption={result['covariance_assumption']}; examples={result['examples']}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def check_gaussian_spatial_hessian(
    frequencies,
    weights,
    longest_projected_baselines,
    *,
    n_taylor_terms,
    reference_frequency,
    image_size,
    cell_size,
    precision,
    context=None,
):
    """Approximate spatial PSF blocks with natural-resolution channel Gaussians.

    Parameters
    ----------
    frequencies, weights, longest_projected_baselines : array-like
        Usable frequencies (Hz), effective weights, and longest usable projected
        baselines (metres) per sample. The latter may be a common scalar.
    n_taylor_terms : int
        Number of Taylor coefficients.
    reference_frequency : float
        Taylor reference frequency in Hz.
    image_size, cell_size : array-like
        Two image dimensions and corresponding signed angular pixels in radians.
    precision : {"float32", "float64"}
        Working precision of the later imaging calculation.
    context : mapping, optional
        Identifying metadata for diagnostic messages.

    Returns
    -------
    dict
        Shared spatial-block diagnostics and Gaussian approximation metadata.
    """
    frequencies, weights = np.asarray(frequencies), np.asarray(weights)
    if (
        frequencies.ndim != 1
        or not frequencies.size
        or weights.shape != frequencies.shape
        or not np.all(np.isfinite(frequencies))
        or not np.all(np.isfinite(weights))
        or np.any(frequencies <= 0)
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "Gaussian spatial-Hessian inputs must be matching positive finite frequency and usable-weight arrays."
        )
    baselines = np.broadcast_to(longest_projected_baselines, frequencies.shape)
    if not np.all(np.isfinite(baselines)) or np.any(baselines <= 0):
        raise ValueError(
            "Gaussian spatial-Hessian preflight requires positive finite usable projected baselines in metres."
        )
    size, cells = np.asarray(image_size), np.asarray(cell_size)
    if (
        size.shape != (2,)
        or np.any(size < 2)
        or size.dtype.kind not in "iu"
        or cells.shape != (2,)
        or not np.all(np.isfinite(cells))
        or np.any(cells == 0)
    ):
        raise ValueError(
            "Gaussian spatial-Hessian preflight requires two image sizes >= 2 and finite nonzero angular cells."
        )

    # --- Gaussian PSF blocks ---
    # FWHM_i = lambda_i / Bmax_i, with Bmax measured in projected metres.
    # A peak-one Gaussian g_i(theta) has transform proportional to
    # sigma_i**2 * exp(-2*pi**2*sigma_i**2*|u|**2). Keep its area factor:
    # independently normalizing each Fourier Gaussian would alter Taylor
    # coupling. Only a common area and weight scale may cancel here.
    fwhm = (299792458.0 / frequencies) / baselines
    sigma = fwhm / np.sqrt(8 * np.log(2))
    areas = (sigma / sigma.max()) ** 2
    q = weights / weights.max()
    x = (frequencies - reference_frequency) / reference_frequency

    # Sample the image's half-Fourier grid, at most 33 coordinates per axis.
    # This is a cheap spatial approximation, not an exhaustive certification;
    # actual PSFs below are assessed on every stored Fourier mode.
    first_axis = np.fft.fftfreq(int(size[0]), d=cells[0])
    second_axis = np.fft.rfftfreq(int(size[1]), d=cells[1])
    first = first_axis[
        np.unique(
            np.linspace(0, first_axis.size - 1, min(33, first_axis.size))
            .round()
            .astype(int)
        )
    ]
    second = second_axis[
        np.unique(
            np.linspace(0, second_axis.size - 1, min(33, second_axis.size))
            .round()
            .astype(int)
        )
    ]
    radii_squared = (first[:, None] ** 2 + second[None, :] ** 2).ravel()
    transfer = np.zeros((2 * n_taylor_terms - 1, radii_squared.size))
    # Bound temporary storage even when the selection contains many channels.
    for start in range(0, frequencies.size, 1024):
        selection = slice(start, start + 1024)
        gaussian = np.exp(
            -2 * np.pi**2 * sigma[selection, None] ** 2 * radii_squared[None, :]
        )
        gaussian *= (q[selection] * areas[selection])[:, None]
        taylor = np.vander(x[selection], N=2 * n_taylor_terms - 1, increasing=True)
        transfer += taylor.T @ gaussian
    result = check_spatial_hessian_blocks(
        transfer.reshape(-1, first.size, second.size),
        first,
        second,
        precision=precision,
        frequency_units="cycles_per_radian",
        context={**(context or {}), "stage": "gaussian_preflight"},
    )
    result.update(
        approximation="circular_gaussian_channel_psfs",
        sampling="at_most_33_by_33_half_fourier_grid",
        gaussian_fwhm_radians=[float(fwhm.min()), float(fwhm.max())],
        resolution_recipe="wavelength_over_longest_usable_projected_baseline",
    )
    return result


def check_psf_spatial_hessian(psf, *, precision, cell_size=None, context=None):
    """Assess every half-Fourier mode of an actual Taylor PSF stack.

    Parameters
    ----------
    psf : ndarray
        Real, centered Taylor PSFs of one plane, shape ``(2*nterms-1, l, m)``.
    precision : {"float32", "float64"}
        Working image precision used for support/rank thresholds.
    cell_size : array-like, optional
        Angular pixel sizes in radians. Without these, report cycles per pixel.
    context : mapping, optional
        Time/polarization identifiers.

    Returns
    -------
    dict
        Shared spatial-block diagnostics for the actual PSF convolution.
    """
    psf = np.asarray(psf)
    if (
        psf.ndim != 3
        or psf.shape[0] % 2 != 1
        or not psf.size
        or np.iscomplexobj(psf)
        or not np.all(np.isfinite(psf))
    ):
        raise ValueError(
            "Spatial Taylor PSFs must be a finite real stack with an odd Taylor count."
        )
    cells = np.ones(2) if cell_size is None else np.asarray(cell_size)
    if cells.shape != (2,) or not np.all(np.isfinite(cells)) or np.any(cells == 0):
        raise ValueError("Spatial-Hessian cell sizes must be finite and nonzero.")
    # Center-to-origin shifting removes the common Fourier phase of a centered
    # convolution kernel. No even-symmetry assumption or PSF symmetrization is
    # applied: imaginary transfer components are diagnosed by the shared check.
    transfer = np.empty(
        (psf.shape[0], psf.shape[1], psf.shape[2] // 2 + 1), dtype=np.complex128
    )
    for order in range(psf.shape[0]):
        transfer[order] = np.fft.rfft2(np.fft.ifftshift(psf[order]))
    result = check_spatial_hessian_blocks(
        transfer,
        np.fft.fftfreq(psf.shape[1], d=cells[0]),
        np.fft.rfftfreq(psf.shape[2], d=cells[1]),
        precision=precision,
        frequency_units="cycles_per_pixel"
        if cell_size is None
        else "cycles_per_radian",
        context={**(context or {}), "stage": "actual_psf"},
    )
    result["sampling"] = "all_half_fourier_modes"
    return result
