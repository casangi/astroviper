"""Identify the runtime inputs of reusable Taylor-Hessian diagnostics."""

import hashlib

import numpy as np


def model_hessian_cache_key(
    psf, psf_values, diagnostics, *, small_scale_bias, cell_size
):
    """Fingerprint actual PSF contents and every input to its diagnostics.

    Parameters
    ----------
    psf : xarray.DataArray
        Current PSF, including its coordinate labels.
    psf_values : numpy.ndarray
        PSF values in canonical time/Taylor/polarization/l/m order.
    diagnostics : mapping
        Validated Taylor count, precision, geometry, scales and reference frequency.
    small_scale_bias : float
        Native per-scale Hessian bias.
    cell_size : array-like or None
        Angular sampling used by spatial diagnostics.

    Returns
    -------
    tuple
        Serializable cache key, independent of array identity or memory address.

    Notes
    -----
    A content digest detects in-place edits and replacement arrays, including
    after transfer between workers. One linear read of the PSF is cheaper than
    repeating its scale convolutions and spatial Fourier diagnostics. The key
    holds no image buffers, and configuration-validation flags cannot grant a
    cache hit. Native cleaner safety checks remain independent.
    """
    values = np.ascontiguousarray(psf_values)
    digest = hashlib.blake2b(memoryview(values).cast("B"), digest_size=32).hexdigest()
    coordinates = hashlib.blake2b(digest_size=32)
    # Axis labels define the product geometry. Ancillary sky-coordinate grids
    # do not enter the Hessian calculation and need not be materialized here.
    for name in sorted(psf.dims):
        if name in psf.coords:
            coordinates.update(repr((name, psf[name].values.tolist())).encode("utf-8"))
    return (
        1,  # Increment when the diagnostic algorithm or its dependencies change.
        digest,
        values.shape,
        values.dtype.str,
        coordinates.hexdigest(),
        diagnostics["n_taylor_terms"],
        diagnostics["working_precision"],
        tuple(diagnostics["image_geometry"]),
        tuple(diagnostics["effective_scales"]),
        diagnostics["reference_frequency"],
        float(small_scale_bias),
        None if cell_size is None else tuple(np.asarray(cell_size).tolist()),
    )
