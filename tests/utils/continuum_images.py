"""Small generated images for continuum software tests; no external data."""

import numpy as np
import xarray as xr


def make_continuum_image(
    psf_axis="psf_taylor_order",
    beam_axis=None,
    legacy=False,
    image_shape=(8, 8),
):
    """Create a small deterministic continuum image for unit tests.

    Parameters
    ----------
    psf_axis : str or None, optional
        Optional Taylor-like dimension added to the point-spread function.
    beam_axis : str or None, optional
        Optional Taylor-like dimension added to the fitted beam parameters.
    legacy : bool, optional
        Omit PSF and beam-fit roles from the residual data group when true.
    image_shape : tuple of int, optional
        Spatial ``(l, m)`` image shape.

    Returns
    -------
    xarray.Dataset
        Synthetic two-term model/residual image with PSF, beam, and primary
        beam products.
    """
    nl, nm = image_shape
    coords = dict(
        time=[0.0],
        taylor_term=[0, 1],
        polarization=["I"],
        l=np.arange(nl) * -1e-5,
        m=np.arange(nm) * 1e-5,
    )
    dims = ("time", "taylor_term", "polarization", "l", "m")
    model = np.zeros((1, 2, 1, nl, nm))
    model[0, 0, 0, nl // 2, nm // 2] = 2.0
    model[0, 1, 0, nl // 2, nm // 2] = 17.0  # Higher Taylor terms must not be restored.
    image = xr.Dataset(
        {"SKY_MODEL": (dims, model), "SKY_RESIDUAL": (dims, np.full_like(model, 0.25))},
        coords=coords,
    )
    plane = xr.DataArray(
        np.ones((1, 1, nl, nm)),
        dims=("time", "polarization", "l", "m"),
        coords={k: image.coords[k] for k in ("time", "polarization", "l", "m")},
    )
    psf = plane
    beam = xr.DataArray(
        np.array([3e-5, 2e-5, 0.3]).reshape(1, 1, 3),
        dims=("time", "polarization", "beam_params"),
        coords={
            "time": [0.0],
            "polarization": ["I"],
            "beam_params": ["major", "minor", "pa"],
        },
    )
    for axis, kind in ((psf_axis, "psf"), (beam_axis, "beam")):
        if axis is not None:
            values = image[axis].values if axis in image.coords else [100.0]
            if kind == "psf":
                psf = psf.expand_dims({axis: values}, axis=1)
            else:
                beam = beam.expand_dims({axis: values}, axis=1)
    image["POINT_SPREAD_FUNCTION"] = psf
    image["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = beam
    image["PRIMARY_BEAM"] = plane * 0.5
    residual = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
    if not legacy:
        residual.update(
            point_spread_function="POINT_SPREAD_FUNCTION",
            beam_fit_params_point_spread_function="BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        )
    image.attrs = {
        "reference_frequency": 100.0,
        "data_groups": {"residual": residual, "model": {"sky": "SKY_MODEL"}},
    }
    return image


def make_runaway_taylor_image(dtype=np.float64):
    """Two-term Taylor residual on which every CLEAN component makes it grow.

    The PSF and checkerboard of the MT-MFS divergence tests
    (``_negative_transfer_problem`` in ``test_mtmfs_clean.py``): a component
    adds 0.9 of itself to its four neighbours, which have the opposite sign.
    The amplitudes are spread by up to 20 percent so that no rounding decides
    a tie. Taylor term one of the residual is zero; PSF orders one and two
    are 0 and 0.25 times order zero (a band symmetric about the reference
    frequency). One time and one polarization, no mask or primary beam.

    Parameters
    ----------
    dtype : numpy.dtype, optional
        Numeric dtype of the generated image products.

    Returns
    -------
    xarray.Dataset
        Synthetic two-term residual and three-order PSF that exercise the
        MT-MFS divergence guard.
    """
    n = 32
    psf = np.zeros((n, n))
    psf[16, 16] = 1.0
    psf[[15, 17, 16, 16], [16, 16, 15, 17]] = 0.9
    yy, xx = np.indices((n, n))
    box = (abs(yy - 16) < 6) & (abs(xx - 16) < 6)
    amplitude = 1.0 + 0.2 * np.random.default_rng(7).uniform(-1.0, 1.0, (n, n))
    residual = np.where((yy + xx) % 2 == 0, 1.0, -1.0) * box * amplitude
    return xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.stack([residual, np.zeros((n, n))])[None, :, None].astype(dtype),
            ),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                np.stack([psf, 0.0 * psf, 0.25 * psf])[None, :, None].astype(dtype),
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                np.zeros((1, 1), dtype=dtype),
            ),
        },
        coords={
            "time": [0.0],
            "taylor_term": [0, 1],
            "psf_taylor_order": [0, 1, 2],
            "polarization": ["I"],
            "l": np.arange(n),
            "m": np.arange(n),
        },
        attrs={
            "reference_frequency": 100.0,
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                }
            },
        },
    )


def make_single_term_continuum_image(dtype=np.float64, shape=(25, 21)):
    """One-Taylor-term continuum image of resolved and unresolved emission.

    Two times and two polarizations. The sky is a 1 Jy point source at pixel
    (8, 7) and a Gaussian of peak 0.1 Jy and width 2 pixels at (15, 12); every
    ``(time, polarization)`` plane scales it by ``1 + 0.25 t + 0.125 p``. The
    PSF is a unit-peak Gaussian of width 1.5 pixels centred on pixel
    ``(n_l // 2, n_m // 2)``, and the residual is the linear convolution of
    the sky with it (no wrap-around). The primary beam is one everywhere,
    laid out as the continuum primary beam is,
    ``(time, frequency, polarization, l, m)`` with one frequency.

    Returns
    -------
    image : xarray.Dataset
        Residual, PSF, primary beam and PSF sidelobe level, with the
        ``residual`` data group.
    sky : numpy.ndarray
        The unscaled ``(l, m)`` sky.
    scale : numpy.ndarray
        The ``(time, polarization)`` scale factors.
    """
    nl, nm = shape
    ntime, npol = 2, 2
    yy, xx = np.indices(shape)
    psf = np.exp(-((yy - nl // 2) ** 2 + (xx - nm // 2) ** 2) / (2 * 1.5**2))
    sky = np.zeros(shape)
    sky[8, 7] = 1.0
    sky += 0.1 * np.exp(-((yy - 15) ** 2 + (xx - 12) ** 2) / (2 * 2.0**2))
    padded = (2 * nl, 2 * nm)
    full = np.fft.irfft2(np.fft.rfft2(sky, padded) * np.fft.rfft2(psf, padded), padded)
    dirty = full[nl // 2 : nl // 2 + nl, nm // 2 : nm // 2 + nm]
    scale = 1.0 + 0.25 * np.arange(ntime)[:, None] + 0.125 * np.arange(npol)[None, :]
    residual = scale[:, None, :, None, None] * dirty[None, None, None]
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                residual.astype(dtype),
            ),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                np.tile(psf, (ntime, 1, npol, 1, 1)).astype(dtype),
            ),
            "PRIMARY_BEAM": (
                ("time", "frequency", "polarization", "l", "m"),
                np.ones((ntime, 1, npol, nl, nm), dtype=dtype),
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                ("time", "polarization"),
                np.zeros((ntime, npol), dtype=dtype),
            ),
        },
        coords={
            "time": np.arange(ntime, dtype=float),
            "taylor_term": [0],
            "psf_taylor_order": [0],
            "frequency": [100.0],
            "polarization": ["I", "Q"],
            "l": np.arange(nl),
            "m": np.arange(nm),
        },
        attrs={
            "reference_frequency": 100.0,
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "primary_beam": "PRIMARY_BEAM",
                }
            },
        },
    )
    return image, sky, scale


def make_single_term_runaway_image():
    """Taylor term zero of :func:`make_runaway_taylor_image`, float64, with a
    primary beam of ones laid out as the continuum primary beam is."""
    image = make_runaway_taylor_image().isel(
        taylor_term=slice(0, 1), psf_taylor_order=slice(0, 1)
    )
    image = image.assign_coords(frequency=[100.0])
    image["PRIMARY_BEAM"] = (
        ("time", "frequency", "polarization", "l", "m"),
        np.ones((1, 1, 1, 32, 32)),
    )
    image.attrs["data_groups"]["residual"]["primary_beam"] = "PRIMARY_BEAM"
    return image
