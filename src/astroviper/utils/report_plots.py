"""Figures of the distributed application reports (:mod:`astroviper.utils.report`).

Every function takes plain arrays (or an :class:`xarray.DataArray`) and returns
a :class:`matplotlib.figure.Figure` made without pyplot, so that a report can
draw and close its figures in any process. Images follow the orientation of
:func:`~astroviper.utils.plotting.generate_plot`: ``data[l, m]`` drawn as
``data.T`` with ``origin="lower"``, with offsets in arcseconds, east to the
left.

A Stokes plane keeps its color and marker in every figure
(:data:`STOKES_STYLES`); a stop code keeps its color whatever codes occur.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "STOKES_STYLES",
    "plot_antenna_positions",
    "plot_deconvolution_summary",
    "plot_point_spread_function_cross_sections",
    "plot_spectra",
    "plot_stokes_images",
    "plot_uv_coverage",
    "point_spread_function_cross_sections",
]

RADIANS_TO_ARCSEC = 180.0 / np.pi * 3600.0

# Colorblind-safe Okabe-Ito colors (the task stream's) in a fixed order, with a
# marker each as the second encoding.
_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
_MARKERS = ("o", "s", "^", "D", "v", "P")
STOKES_STYLES = {
    stokes: {"color": color, "marker": marker}
    for stokes, color, marker in zip(
        ("I", "Q", "U", "V"), _COLORS, _MARKERS, strict=False
    )
}
_GRID = {"color": "#d0d7de", "linewidth": 0.5}


def _stokes_style(stokes, index):
    """Color and marker of a Stokes plane (fixed for I, Q, U, V)."""
    return STOKES_STYLES.get(
        str(stokes),
        {"color": _COLORS[index % len(_COLORS)], "marker": _MARKERS[index % 6]},
    )


def _figure(n_rows, n_columns, panel_size=(4.0, 3.0), **kwargs):
    """A pyplot-free figure with a grid of panels (always a 2-D axes array)."""
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows),
        layout="constrained",
    )
    axes = figure.subplots(n_rows, n_columns, squeeze=False, **kwargs)
    return figure, axes


def point_spread_function_cross_sections(
    psf_plane, beam, cell_size, extent=None, n_samples=401
):
    """Cuts of a point spread function along the axes of its fitted Gaussian.

    The cuts pass through the peak of the PSF along the major axis, the minor
    axis and the diagonal between them (45 degrees from both). The PSF is
    interpolated linearly between pixels; the Gaussian is the fit evaluated on
    the same lines, scaled to the PSF peak. The axes follow the convention of
    :func:`~astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit.point_spread_function_gaussian_fit`:
    in the frame of the pixel axes ``(l, m)``, in radians, the major axis
    points along ``(sin(pa), -cos(pa))`` and the minor axis along
    ``(cos(pa), sin(pa))``.

    Parameters
    ----------
    psf_plane : numpy.ndarray, [n_l, n_m]
        One plane of the point spread function.
    beam : array_like, [3]
        Fitted ``[major, minor, pa]``: full widths at half maximum and position
        angle, radians.
    cell_size : array_like, [2]
        Pixel size along ``l`` and ``m``, radians (the sign is ignored).
    extent : float, optional
        Half length of the cuts, radians. Default three major axes, at most
        the distance from the peak to the nearest image edge.
    n_samples : int, optional
        Samples along each cut. Default 401.

    Returns
    -------
    dict
        ``{"major" | "minor" | "diagonal": {"offset", "psf", "gaussian",
        "fwhm"}}``: the offset from the peak along the cut (radians), the
        PSF and the Gaussian at those offsets, and the full width at half
        maximum of the Gaussian along the cut (radians).
    """
    from scipy.ndimage import map_coordinates

    psf_plane = np.asarray(psf_plane, dtype=np.float64)
    major, minor, pa = (float(value) for value in beam)
    delta_l, delta_m = (abs(float(value)) for value in cell_size)
    finite = np.where(np.isfinite(psf_plane), psf_plane, -np.inf)
    peak_l, peak_m = np.unravel_index(np.argmax(finite), psf_plane.shape)
    peak = psf_plane[peak_l, peak_m]
    if extent is None:
        edge = min(
            peak_l * delta_l,
            (psf_plane.shape[0] - 1 - peak_l) * delta_l,
            peak_m * delta_m,
            (psf_plane.shape[1] - 1 - peak_m) * delta_m,
        )
        extent = min(3.0 * major, edge)
    offset = np.linspace(-extent, extent, n_samples)
    major_axis = np.array([np.sin(pa), -np.cos(pa)])
    minor_axis = np.array([np.cos(pa), np.sin(pa)])
    cuts = {}
    for name, direction, angle in (
        ("major", major_axis, 0.0),
        ("minor", minor_axis, 0.5 * np.pi),
        ("diagonal", (major_axis + minor_axis) / np.sqrt(2.0), 0.25 * np.pi),
    ):
        pixels = np.array(
            [
                peak_l + offset * direction[0] / delta_l,
                peak_m + offset * direction[1] / delta_m,
            ]
        )
        values = map_coordinates(psf_plane, pixels, order=1, cval=np.nan)
        # 1 / FWHM**2 along a line at ``angle`` from the major axis
        inverse_square = (np.cos(angle) / major) ** 2 + (np.sin(angle) / minor) ** 2
        gaussian = peak * np.exp(-4.0 * np.log(2.0) * offset**2 * inverse_square)
        cuts[name] = {
            "offset": offset,
            "psf": values,
            "gaussian": gaussian,
            "fwhm": 1.0 / np.sqrt(inverse_square),
        }
    return cuts


def plot_point_spread_function_cross_sections(psf_plane, beam, cell_size, title=None):
    """Three panels: the PSF and its fitted Gaussian along the major axis, the
    minor axis and the diagonal between them
    (:func:`point_spread_function_cross_sections`).

    Parameters
    ----------
    psf_plane, beam, cell_size
        As for :func:`point_spread_function_cross_sections`.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cuts = point_spread_function_cross_sections(psf_plane, beam, cell_size)
    figure, axes = _figure(1, 3, panel_size=(4.2, 3.2), sharey=True)
    labels = {
        "major": "major axis",
        "minor": "minor axis",
        "diagonal": "diagonal (45 deg between them)",
    }
    for ax, (name, cut) in zip(axes[0], cuts.items(), strict=True):
        offset = cut["offset"] * RADIANS_TO_ARCSEC
        ax.plot(offset, cut["psf"], color=_COLORS[0], linewidth=1.5, label="PSF")
        ax.plot(
            offset,
            cut["gaussian"],
            color=_COLORS[1],
            linewidth=1.5,
            linestyle="--",
            label="fitted Gaussian",
        )
        ax.axhline(0.0, color="#57606a", linewidth=0.5)
        ax.set_title(
            f'{labels[name]}, FWHM {cut["fwhm"] * RADIANS_TO_ARCSEC:.4g}"',
            fontsize=10,
        )
        ax.set_xlabel("offset from the peak (arcsec)")
        ax.grid(True, **_GRID)
    axes[0, 0].set_ylabel("PSF")
    axes[0, 0].legend(fontsize=8)
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def plot_stokes_images(
    planes, title=None, colorbar_label=None, cmap="viridis", stretch="linear"
):
    """One image per Stokes plane, each with its own color bar.

    Parameters
    ----------
    planes : xarray.DataArray
        Dims ``(polarization, l, m)`` with the ``l`` and ``m`` coordinates in
        radians.
    title : str, optional
        Figure title.
    colorbar_label : str, optional
        Label of the color bars (e.g. the units).
    cmap : str, optional
        Matplotlib colormap. Default ``"viridis"``.
    stretch : str, optional
        ``"linear"`` (default) or ``"asinh"``, linear below a hundredth of the
        largest absolute value of the plane and logarithmic above, so that
        faint extended emission shows next to bright point sources.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.colors import AsinhNorm

    planes = planes.transpose("polarization", "l", "m")
    l_offset = (planes["l"].values - planes["l"].values.mean()) * RADIANS_TO_ARCSEC
    m_offset = (planes["m"].values - planes["m"].values.mean()) * RADIANS_TO_ARCSEC
    extent = [l_offset[0], l_offset[-1], m_offset[0], m_offset[-1]]
    n_stokes = planes.sizes["polarization"]
    figure, axes = _figure(1, n_stokes, panel_size=(3.6, 3.2))
    for ax, stokes in zip(axes[0], planes["polarization"].values, strict=True):
        values = np.asarray(planes.sel(polarization=stokes).values, dtype=np.float64)
        norm = None
        largest = np.nanmax(np.abs(values)) if np.any(np.isfinite(values)) else 0.0
        if stretch == "asinh" and largest > 0:
            norm = AsinhNorm(linear_width=0.01 * largest)
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="equal",
            norm=norm,
        )
        colorbar = figure.colorbar(image, ax=ax, label=colorbar_label, shrink=0.85)
        if norm is not None:
            colorbar.set_ticks(_asinh_ticks(norm.linear_width, values))
        ax.set_title(f"Stokes {stokes}", fontsize=10)
        ax.set_xlabel("l offset (arcsec)")
    axes[0, 0].set_ylabel("m offset (arcsec)")
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def _asinh_ticks(linear_width, values):
    """Color bar ticks of an asinh stretch: 0 and the powers of ten (of the
    signs present) outside its linear part, which crowd no tick near 0."""
    finite = values[np.isfinite(values)]
    ticks = [0.0]
    for sign, largest in ((1.0, finite.max()), (-1.0, -finite.min())):
        if largest > 2.0 * linear_width:
            exponents = range(
                int(np.ceil(np.log10(2.0 * linear_width))),
                int(np.floor(np.log10(largest))) + 1,
            )
            ticks += [sign * 10.0**exponent for exponent in exponents]
    return sorted(ticks)


def _frequency_axis(ax, frequency):
    """Label ``ax`` for a frequency axis in GHz, with the frequencies written
    out (no offset for narrow channels), and return the values."""
    from matplotlib.ticker import MaxNLocator

    ax.set_xlabel("frequency (GHz)")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelrotation=20)
    return np.asarray(frequency, dtype=np.float64) / 1e9


def plot_spectra(frequency, values, stokes, ylabel, title=None):
    """One panel per Stokes plane of a quantity against frequency.

    Parameters
    ----------
    frequency : array_like, [n_frequency]
        Hz.
    values : array_like, [n_frequency, n_stokes]
    stokes : list of str
        The Stokes labels of the columns of ``values``.
    ylabel : str
        Label of the quantity (with its units).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = np.asarray(values, dtype=np.float64).reshape(len(frequency), len(stokes))
    figure, axes = _figure(1, len(stokes), panel_size=(3.6, 2.8))
    for index, (ax, label) in enumerate(zip(axes[0], stokes, strict=True)):
        x = _frequency_axis(ax, frequency)
        style = _stokes_style(label, index)
        ax.plot(
            x,
            values[:, index],
            color=style["color"],
            marker=style["marker"],
            markersize=4,
            linewidth=1.5,
        )
        ax.set_title(f"Stokes {label}", fontsize=10)
        ax.grid(True, **_GRID)
    axes[0, 0].set_ylabel(ylabel)
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def _per_stokes_panel(ax, x, values, stokes, ylabel, title):
    """One line per Stokes plane (columns of ``values``) against ``x``."""
    for index, label in enumerate(stokes):
        style = _stokes_style(label, index)
        ax.plot(
            x,
            values[:, index],
            color=style["color"],
            marker=style["marker"],
            markersize=4,
            linewidth=1.5,
            label=f"Stokes {label}",
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, **_GRID)
    if len(stokes) > 1:
        ax.legend(fontsize=7)


def plot_deconvolution_summary(
    frequency,
    stokes,
    imaging_stop_code,
    model_update_stop_code_counts,
    iterations,
    imaging_cycles,
    entropy,
    residual_rms,
    imaging_stop_descriptions,
    model_update_stop_descriptions,
):
    """Six panels of the per channel record of a cube's deconvolution.

    Parameters
    ----------
    frequency : array_like, [n_frequency]
        Hz.
    stokes : list of str
        The Stokes planes.
    imaging_stop_code : array_like, [n_frequency, n_stokes]
        The cycle control stop code of every plane.
    model_update_stop_code_counts : dict
        ``{code: [n_frequency] counts}``, the number of model updates of each
        channel (over its Stokes planes) that ended with ``code``.
    iterations : array_like, [n_frequency, n_stokes]
        Total iterations of every plane.
    imaging_cycles : array_like, [n_frequency]
        Imaging cycles of every channel.
    entropy, residual_rms : array_like, [n_frequency, n_stokes]
        Final residual entropy and RMS of every plane.
    imaging_stop_descriptions, model_update_stop_descriptions : dict
        ``{code: description}`` of the two kinds of stop code.

    Returns
    -------
    matplotlib.figure.Figure
    """
    shape = (len(frequency), len(stokes))
    imaging_stop_code = np.asarray(imaging_stop_code, dtype=np.float64).reshape(shape)
    figure, axes = _figure(3, 2, panel_size=(5.4, 2.9), sharex=True)
    x = np.asarray(frequency, dtype=np.float64) / 1e9

    ax = axes[0, 0]
    for index, label in enumerate(stokes):
        style = _stokes_style(label, index)
        ax.plot(
            x,
            imaging_stop_code[:, index],
            linestyle="none",
            color=style["color"],
            marker=style["marker"],
            markersize=5,
            label=f"Stokes {label}",
        )
    codes = sorted(
        {int(code) for code in imaging_stop_code.ravel() if np.isfinite(code)}
    )
    ax.set_yticks(codes)
    ax.set_yticklabels(
        [f"{code} {imaging_stop_descriptions.get(code, '')[:30]}" for code in codes],
        fontsize=7,
    )
    ax.set_title("cycle control stop code", fontsize=10)
    ax.grid(True, **_GRID)
    if len(stokes) > 1:
        ax.legend(fontsize=7)

    ax = axes[0, 1]
    known = sorted(model_update_stop_descriptions)
    for code in sorted(model_update_stop_code_counts):
        index = known.index(code) if code in known else len(known) + code
        ax.plot(
            x,
            np.asarray(model_update_stop_code_counts[code], dtype=np.float64),
            color=_COLORS[index % len(_COLORS)],
            marker=_MARKERS[index % len(_MARKERS)],
            markersize=4,
            linewidth=1.5,
            label=f"{code} {model_update_stop_descriptions.get(code, '')}",
        )
    ax.set_ylabel("model updates")
    ax.set_title("model update stop codes: occurrences", fontsize=10)
    ax.grid(True, **_GRID)
    if model_update_stop_code_counts:
        ax.legend(fontsize=7)

    _per_stokes_panel(
        axes[1, 0],
        x,
        np.asarray(iterations, dtype=np.float64).reshape(shape),
        stokes,
        "iterations",
        "total iterations",
    )
    ax = axes[1, 1]
    ax.plot(
        x,
        np.asarray(imaging_cycles, dtype=np.float64),
        color=_COLORS[0],
        marker="o",
        markersize=4,
        linewidth=1.5,
    )
    ax.set_ylabel("imaging cycles")
    ax.set_title("imaging cycles", fontsize=10)
    ax.grid(True, **_GRID)

    _per_stokes_panel(
        axes[2, 0],
        x,
        np.asarray(entropy, dtype=np.float64).reshape(shape),
        stokes,
        "entropy",
        "final residual entropy",
    )
    _per_stokes_panel(
        axes[2, 1],
        x,
        np.asarray(residual_rms, dtype=np.float64).reshape(shape),
        stokes,
        "RMS",
        "residual RMS",
    )
    for ax in axes[-1]:
        _frequency_axis(ax, frequency)
    return figure


def plot_antenna_positions(antenna_name, east, north, up, title=None):
    """Two panels: the antennas in the local horizontal plane, each with its
    name, and the height of every antenna against its name.

    Parameters
    ----------
    antenna_name : array_like of str, [n_antenna]
    east, north, up : array_like, [n_antenna]
        Local tangent plane offsets from the array reference position, metres
        (:func:`~astroviper.utils.telescope_layout.itrf_to_local_tangent_plane`).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.figure import Figure

    antenna_name = [str(name) for name in antenna_name]
    east, north, up = (
        np.asarray(value, dtype=np.float64) for value in (east, north, up)
    )
    figure = Figure(figsize=(12.0, 5.0), layout="constrained")
    ax_plane, ax_height = figure.subplots(
        1, 2, gridspec_kw={"width_ratios": [1.0, 1.4]}
    )
    ax_plane.plot(east, north, "o", color=_COLORS[0], markersize=5)
    for name, x, y in zip(antenna_name, east, north, strict=True):
        ax_plane.annotate(
            name,
            (x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color="#1f2328",
        )
    ax_plane.set_aspect("equal", adjustable="datalim")
    ax_plane.set_xlabel("east (m)")
    ax_plane.set_ylabel("north (m)")
    ax_plane.set_title("antenna positions", fontsize=10)
    ax_plane.grid(True, **_GRID)

    index = np.arange(len(antenna_name))
    ax_height.plot(index, up, "o", color=_COLORS[0], markersize=5)
    ax_height.set_xticks(index)
    ax_height.set_xticklabels(antenna_name, rotation=90, fontsize=6)
    ax_height.set_xlim(-0.5, len(antenna_name) - 0.5)
    ax_height.set_ylabel("up (m)")
    ax_height.set_title("antenna heights", fontsize=10)
    ax_height.grid(True, **_GRID)
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def plot_uv_coverage(u, v, title=None, max_points=200_000):
    """The uv coverage: every sample and its conjugate ``(-u, -v)``.

    Parameters
    ----------
    u, v : array_like
        uv coordinates, wavelengths.
    title : str, optional
        Figure title.
    max_points : int, optional
        At most this many samples are drawn (evenly strided).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.figure import Figure

    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    keep = np.isfinite(u) & np.isfinite(v)
    u, v = u[keep], v[keep]
    stride = max(1, int(np.ceil(u.size / max_points)))
    u, v = u[::stride] / 1e3, v[::stride] / 1e3
    figure = Figure(figsize=(5.6, 5.4), layout="constrained")
    ax = figure.subplots()
    for sign in (1.0, -1.0):
        ax.plot(
            sign * u,
            sign * v,
            linestyle="none",
            marker=".",
            markersize=1.5,
            color=_COLORS[0],
            rasterized=True,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("u (kilo wavelengths)")
    ax.set_ylabel("v (kilo wavelengths)")
    ax.grid(True, **_GRID)
    if title:
        ax.set_title(title, fontsize=10)
    return figure
