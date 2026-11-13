"""Simulated dirty cubes for the find_continuum tests."""

import numpy as np
import xarray as xr
from xradio.image import make_empty_sky_image

RAD_PER_ARCSEC = np.pi / (180.0 * 3600.0)
FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))


def _gaussian_2d(shape, center, fwhm_pixels):
    l_index, m_index = np.indices(shape, dtype=np.float64)
    sigma = fwhm_pixels * FWHM_TO_SIGMA
    radius2 = (l_index - center[0]) ** 2 + (m_index - center[1]) ** 2
    return np.exp(-0.5 * radius2 / sigma**2)


def _convolve_with_beam(plane, beam_fwhm_pixels):
    n_l, n_m = plane.shape
    sigma = beam_fwhm_pixels * FWHM_TO_SIGMA
    u = np.fft.fftfreq(n_l)[:, None]
    v = np.fft.fftfreq(n_m)[None, :]
    transfer = np.exp(-2.0 * (np.pi * sigma) ** 2 * (u**2 + v**2))
    return np.fft.ifft2(np.fft.fft2(plane) * transfer).real


def make_find_continuum_image_xds(
    n_pixels=64,
    n_channels=480,
    first_frequency=230.0e9,
    channel_width=0.9765625e6,
    cell_arcsec=0.2,
    beam_fwhm_arcsec=1.0,
    primary_beam_fwhm_arcsec=25.0,
    noise_jy=1.0e-3,
    continuum_jy=5.0e-3,
    lines=(),
    seed=7,
):
    """Dirty Stokes I cube with continuum, Gaussian lines and correlated noise.

    Parameters
    ----------
    n_pixels : int, default 64
        Image size along ``l`` and ``m``.
    n_channels : int, default 480
        Number of frequency channels.
    first_frequency, channel_width : float
        LSRK frequency of channel 0 and the channel width (Hz).
    cell_arcsec, beam_fwhm_arcsec, primary_beam_fwhm_arcsec : float
        Pixel size, synthesized beam FWHM and primary beam FWHM at
        ``first_frequency`` (arcsec).
    noise_jy : float, default 1e-3
        Noise per channel (Jy/beam), correlated on the beam scale.
    continuum_jy : float, default 5e-3
        Flux density of a point source at the phase center (Jy).
    lines : sequence of dict
        Each line has ``channel``, ``fwhm_channels``, ``peak_jy`` (Jy/beam at
        the line center), ``offset_pixels`` (``(dl, dm)``) and
        ``size_fwhm_pixels`` (spatial FWHM before the beam), and optionally
        ``sidelobe_fraction``: the amplitude, relative to ``peak_jy``, of a
        ripple across the whole image in the line's channels, as the
        sidelobes of a bright line in a dirty cube.
    seed : int, default 7
        Seed of the noise.

    Returns
    -------
    xarray.Dataset
        Image dataset with data group ``residual`` (``SKY_RESIDUAL``,
        ``PRIMARY_BEAM``, ``POINT_SPREAD_FUNCTION`` and
        ``BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION``).
    """
    rng = np.random.default_rng(seed)
    frequency = first_frequency + channel_width * np.arange(n_channels)
    img_xds = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[n_pixels, n_pixels],
        cell_size=[cell_arcsec * RAD_PER_ARCSEC, cell_arcsec * RAD_PER_ARCSEC],
        frequency_coords=frequency,
        pol_coords=["I"],
        time_coords=[0.0],
    )
    shape = (n_pixels, n_pixels)
    center = ((n_pixels - 1) / 2.0, (n_pixels - 1) / 2.0)
    beam_fwhm_pixels = beam_fwhm_arcsec / cell_arcsec
    point_source = _gaussian_2d(shape, center, beam_fwhm_pixels)
    channel_index = np.arange(n_channels)

    sky = np.empty((1, n_channels, 1, n_pixels, n_pixels), dtype=np.float32)
    primary_beam = np.empty_like(sky)
    noise_gain = 1.0 / np.sqrt(
        np.mean(_convolve_with_beam(rng.standard_normal(shape), beam_fwhm_pixels) ** 2)
    )
    line_shapes = [
        _convolve_with_beam(
            _gaussian_2d(
                shape,
                (
                    center[0] + line["offset_pixels"][0],
                    center[1] + line["offset_pixels"][1],
                ),
                line["size_fwhm_pixels"],
            ),
            beam_fwhm_pixels,
        )
        for line in lines
    ]
    line_shapes = [shape_ / shape_.max() for shape_ in line_shapes]
    l_index, m_index = np.indices(shape, dtype=np.float64)
    ripple = np.cos(2.0 * np.pi * (l_index / 11.0 + m_index / 17.0))
    for line, line_shape in zip(lines, line_shapes, strict=True):
        line_shape += line.get("sidelobe_fraction", 0.0) * ripple
    for chan in range(n_channels):
        pb_fwhm_pixels = (
            primary_beam_fwhm_arcsec * first_frequency / frequency[chan] / cell_arcsec
        )
        pb = _gaussian_2d(shape, center, pb_fwhm_pixels)
        plane = continuum_jy * point_source
        for line, line_shape in zip(lines, line_shapes, strict=True):
            sigma = line["fwhm_channels"] * FWHM_TO_SIGMA
            profile = np.exp(
                -0.5 * ((channel_index[chan] - line["channel"]) / sigma) ** 2
            )
            plane = plane + line["peak_jy"] * profile * line_shape
        noise = _convolve_with_beam(rng.standard_normal(shape), beam_fwhm_pixels)
        sky[0, chan, 0] = plane * pb + noise_jy * noise_gain * noise
        primary_beam[0, chan, 0] = pb

    dims = ["time", "frequency", "polarization", "l", "m"]
    img_xds["SKY_RESIDUAL"] = xr.DataArray(sky, dims=dims, attrs={"units": "Jy/beam"})
    img_xds["PRIMARY_BEAM"] = xr.DataArray(primary_beam, dims=dims)
    psf = np.broadcast_to(point_source.astype(np.float32), sky.shape).copy()
    img_xds["POINT_SPREAD_FUNCTION"] = xr.DataArray(psf, dims=dims)
    beam = np.empty((1, n_channels, 1, 3))
    beam[..., 0] = beam[..., 1] = beam_fwhm_arcsec * RAD_PER_ARCSEC
    beam[..., 2] = 0.0
    img_xds["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = xr.DataArray(
        beam,
        dims=["time", "frequency", "polarization", "beam_params_label"],
        attrs={"units": "rad"},
    )
    img_xds.attrs["data_groups"] = {
        "residual": {
            "sky": "SKY_RESIDUAL",
            "primary_beam": "PRIMARY_BEAM",
            "point_spread_function": "POINT_SPREAD_FUNCTION",
            "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        }
    }
    return img_xds


#: A compact bright line, a faint extended line offset from it, and a narrow line.
LINE_SET = (
    {
        "channel": 120,
        "fwhm_channels": 14,
        "peak_jy": 0.04,
        "offset_pixels": (0, 0),
        "size_fwhm_pixels": 6,
    },
    {
        "channel": 300,
        "fwhm_channels": 40,
        "peak_jy": 2.5e-3,
        "offset_pixels": (14, -10),
        "size_fwhm_pixels": 26,
    },
    {
        "channel": 410,
        "fwhm_channels": 4,
        "peak_jy": 0.02,
        "offset_pixels": (-6, 8),
        "size_fwhm_pixels": 4,
    },
)
