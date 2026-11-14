"""Simulated dirty cubes and reduction helpers for the find_continuum tests."""

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
COMPACT_LINE = dict(
    channel=120,
    fwhm_channels=14,
    peak_jy=0.04,
    offset_pixels=(0, 0),
    size_fwhm_pixels=6,
)
# Channel selection, all-continuum flag and moment difference SNR of CASA
# 6.7.6's findContinuum (pipeline 2026.3.0.7) on the same cubes, with CASA's
# atmosphere model. The windows have no atmospheric line, and a flat atmosphere
# selects the same channels. Last, the passes of find_continuum that read the
# dirty cube.
CASA_RESULTS = {
    "no_lines": (dict(lines=()), "9~464", True, 4.4041, 7),
    "line_set": (dict(lines=LINE_SET), "6~85;154~367;433~472", False, 6.2507, 7),
    # The primary beam falls below 0.2 in the corners: masked there, as by
    # tclean's pblimit in the pipeline (and in the CASA reference).
    "small_beam": (
        dict(lines=LINE_SET, primary_beam_fwhm_arcsec=8.0),
        "6~85;154~367;433~467",
        False,
        7.3877,
        7,
    ),
    # The ripple of the bright line's channels amends the joint mask.
    "sidelobes": (
        dict(
            n_pixels=96,
            lines=(
                dict(COMPACT_LINE, sidelobe_fraction=0.2),
                dict(
                    channel=300,
                    fwhm_channels=20,
                    peak_jy=6e-3,
                    offset_pixels=(22, -18),
                    size_fwhm_pixels=24,
                ),
            ),
        ),
        "3~86;154~255;339~472",
        False,
        5.0069,
        8,
    ),
}
# Cubes of findContinuum's paths without rounds (time-division mode), with an
# extra joint mask after round original and with the low-spread fix, with
# CASA's results as in CASA_RESULTS.
NATIVE_CASES = {
    "tdm": (
        dict(
            lines=LINE_SET[:1],
            n_channels=120,
            channel_width=15.625e6,
            beam_fwhm_arcsec=0.3,
            cell_arcsec=0.06,
        ),
        "3~106",
        False,
        0.0,
    ),
    "only_extra_mask": (
        dict(
            n_pixels=96,
            lines=(
                dict(COMPACT_LINE, sidelobe_fraction=0.1),
                dict(
                    channel=300,
                    fwhm_channels=20,
                    peak_jy=6e-3,
                    offset_pixels=(22, -18),
                    size_fwhm_pixels=24,
                ),
            ),
        ),
        "7~95;172~260;325~473",
        False,
        4.8890,
    ),
    "eight_channels": (
        dict(
            n_channels=8,
            lines=(
                dict(
                    channel=4,
                    fwhm_channels=1.5,
                    peak_jy=0.04,
                    offset_pixels=(0, 0),
                    size_fwhm_pixels=6,
                ),
            ),
        ),
        "1~1;6~7",
        False,
        4.1222,
    ),
}
# The passes of find_continuum_ultra that read the dirty cube, without a
# first-round store and with the imager's (design section 14.3 item 5).
NATIVE_RESIDUAL_READS = {
    "no_lines": (3, 2),
    "line_set": (3, 2),
    "small_beam": (3, 2),
    "sidelobes": (5, 4),
    "tdm": (2, 1),
    "only_extra_mask": (5, 4),
    "eight_channels": (4, 3),
}


def graphviper_tree(leaves, merge):
    """GraphVIPER's binary tree reduce (``_tree_combine``): neighbours paired
    level by level, an odd last node carried up, ``merge(left, right)``."""
    level = list(leaves)
    while len(level) > 1:
        level = [
            merge(level[i], level[i + 1]) if i + 1 < len(level) else level[i]
            for i in range(0, len(level), 2)
        ]
    return level[0]
