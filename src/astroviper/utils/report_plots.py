"""Figures of the distributed application reports (:mod:`astroviper.utils.report`).

Every plotting function takes plain arrays (an :class:`xarray.DataArray`, or the
convergence record of a deconvolution) and returns a
:class:`matplotlib.figure.Figure` made without pyplot, so that a report can
draw and close its figures in any process. Images follow the orientation of
:func:`~astroviper.utils.plotting.generate_plot`: ``data[l, m]`` drawn as
``data.T`` with ``origin="lower"``, the sky images with offsets in arcseconds,
east to the left, the images of a deconvolution with pixel indices.

A Stokes plane keeps its color and marker in every figure
(:data:`STOKES_STYLES`); a stop code keeps its color whatever codes occur.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "STOKES_STYLES",
    "plot_antenna_positions",
    "plot_deconvolution_summary",
    "plot_plane_deconvolution_summary",
    "plot_point_spread_function_cross_sections",
    "plot_signed_images",
    "plot_spectra",
    "plot_stokes_images",
    "plot_uv_coverage",
    "plot_visibility_comparison",
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
_INK = "#57606a"
# The kinds of deconvolution component, each with its style: the iteration
# history gives every component its scale (-1 a residual-shaped component of
# Asp, 0 a point, above 0 an Asp Gaussian); the convergence record counts them
# per model update, the components of Asp's significance stage (counted as
# points or Asp Gaussians too) and its Hogbom runs as well.
_COMPONENT_STYLES = {
    "residual-shaped": {"color": _COLORS[4], "marker": "*"},
    "Asp Gaussian": {"color": _COLORS[2], "marker": "o"},
    "significance": {"color": _COLORS[1], "marker": "D"},
    "point": {"color": _COLORS[0], "marker": "s"},
    "Hogbom run": {"color": _COLORS[3], "marker": "^"},
}


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


def _image_panel(figure, ax, plane, title, cmap, norm, mask=None):
    """``plane`` drawn as ``plane.T`` with ``origin="lower"``, its color bar
    (ticks outside the linear part of an asinh scale) and, if given, the
    outline of ``mask``."""
    from matplotlib.ticker import FormatStrFormatter

    ax.set_xlabel("l pixel")
    if not np.isfinite(plane).any():
        ax.set_title(f"{title} (no finite pixel)", fontsize=10)
        return
    image = ax.imshow(
        plane.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest"
    )
    if mask is not None and np.any(mask) and not np.all(mask):
        ax.contour(
            np.asarray(mask, dtype=np.float64).T,
            levels=[0.5],
            colors="#1f2328",
            linewidths=0.8,
        )
    colorbar = figure.colorbar(image, ax=ax, shrink=0.8)
    if hasattr(norm, "linear_width"):
        colorbar.set_ticks(_asinh_ticks(norm.linear_width, plane))
    else:  # every tick in full: an exponent above the bar would hit the titles
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
    ax.set_title(title, fontsize=10)


def _largest(plane):
    """The largest finite absolute value, 0 without one."""
    return float(np.max(np.abs(plane), where=np.isfinite(plane), initial=0.0))


def _signed_norm(plane, limit=None):
    """A linear scale symmetric about 0, to ``limit`` or the largest value."""
    from matplotlib.colors import Normalize

    limit = (_largest(plane) if limit is None else limit) or 1.0
    return Normalize(-limit, limit)


def plot_signed_images(planes, titles, masks=None, title=None):
    """Signed images (differences, residuals) side by side, each on a linear
    scale symmetric about 0, with the outline of its mask.

    Parameters
    ----------
    planes : sequence of numpy.ndarray, [n_l, n_m]
    titles : sequence of str
        Title of every panel.
    masks : sequence of (numpy.ndarray of bool, [n_l, n_m], or None), optional
        The mask outlined on every panel.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axes = _figure(1, len(planes), panel_size=(4.2, 3.6))
    masks = [None] * len(planes) if masks is None else masks
    for ax, plane, label, mask in zip(axes[0], planes, titles, masks, strict=True):
        plane = np.asarray(plane, dtype=np.float64)
        finite = plane[np.isfinite(plane)]
        if finite.size and not finite.any():
            label += " (all zero)"
        _image_panel(figure, ax, plane, label, "RdBu_r", _signed_norm(plane), mask)
    axes[0, 0].set_ylabel("m pixel")
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def plot_visibility_comparison(uv_distance, visibility, expected, noise, title=None):
    """Visibilities against an independent prediction of them, and their noise
    against the noise their weights give.

    Parameters
    ----------
    uv_distance : array_like, [n]
        Wavelengths.
    visibility, expected : array_like of complex, [n]
        The visibilities and their prediction, at the same samples.
    noise : array_like
        Noise samples in units of the standard deviation the weights give, for
        example the real and imaginary parts of the noise times
        ``sqrt(WEIGHT)``.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    uv_distance = np.asarray(uv_distance, dtype=np.float64) / 1e3
    visibility = np.asarray(visibility, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    noise = np.asarray(noise, dtype=np.float64).ravel()
    figure, axes = _figure(1, 3, panel_size=(4.6, 3.6))

    ax = axes[0, 0]
    ax.plot(uv_distance, np.abs(expected), "o", color=_COLORS[0], markersize=5)
    ax.plot(uv_distance, np.abs(visibility), ".", color=_COLORS[1], markersize=3)
    ax.legend(["prediction", "visibilities"], fontsize=7)
    ax.set_xlabel("uv distance (kilo wavelengths)")
    ax.set_ylabel("amplitude")
    ax.set_title("visibilities", fontsize=10)
    ax.grid(True, **_GRID)

    ax = axes[0, 1]
    difference = np.abs(visibility - expected)
    equal = difference == 0
    ax.plot(uv_distance[~equal], difference[~equal], ".", color=_COLORS[0])
    ax.set_yscale("log")
    ax.set_xlabel("uv distance (kilo wavelengths)")
    ax.set_ylabel("|visibility - prediction|")
    ax.set_title(f"difference ({int(equal.sum())} samples equal)", fontsize=10)
    ax.grid(True, **_GRID)

    ax = axes[0, 2]
    edges = np.linspace(-5.0, 5.0, 81)
    # a density over all the samples, those beyond the axis counted
    beyond = np.count_nonzero(np.abs(noise) > edges[-1])
    ax.hist(
        noise,
        bins=edges,
        weights=np.full(noise.size, 1.0 / (noise.size * np.diff(edges)[0])),
        histtype="step",
        color=_COLORS[0],
        linewidth=1.5,
        label="noise" + (f" ({beyond} beyond ±5)" if beyond else ""),
    )
    x = np.linspace(edges[0], edges[-1], 201)
    ax.plot(
        x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "--", color=_INK, label="N(0, 1)"
    )
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel("noise / standard deviation of the weights")
    ax.set_title(
        f"noise: mean {noise.mean():.3f}, standard deviation {noise.std():.3f}",
        fontsize=10,
    )
    ax.grid(True, **_GRID)
    if title:
        figure.suptitle(title, fontsize=11)
    return figure


def _record_values(record, name, n):
    """The first ``n`` values of a field of a convergence record, float, or
    ``None`` without the field."""
    if name not in record:
        return None
    return np.asarray(
        [np.nan if value is None else value for value in list(record[name])[:n]],
        dtype=np.float64,
    )


def _moments(values):
    """Skew and excess kurtosis."""
    deviation = values - values.mean()
    variance = np.mean(deviation**2)
    if variance == 0:
        return np.nan, np.nan
    return (
        np.mean(deviation**3) / variance**1.5,
        np.mean(deviation**4) / variance**2 - 3.0,
    )


def _residual_histogram(ax, residual, mask, sigma):
    """The distribution of the finite residual pixels, in and outside the mask,
    in units of ``sigma`` (or of their standard deviation), against N(0, 1)."""
    finite = np.isfinite(residual)
    unit = sigma or (float(np.std(residual[finite])) if finite.any() else 0.0) or 1.0
    if mask is None:
        regions = [("all pixels", residual[finite])]
    else:
        regions = [
            ("in the mask", residual[mask & finite]),
            ("outside", residual[~mask & finite]),
        ]
    edges = np.linspace(-6.0, 6.0, 97)
    for index, (label, values) in enumerate(regions):
        if values.size:
            # a density over all the values, those beyond the axis counted
            beyond = np.count_nonzero(np.abs(values / unit) > edges[-1])
            ax.hist(
                values / unit,
                bins=edges,
                weights=np.full(values.size, 1.0 / (values.size * np.diff(edges)[0])),
                histtype="step",
                color=_COLORS[index],
                linewidth=1.5,
                label=label + (f" ({beyond} beyond ±6)" if beyond else ""),
            )
    x = np.linspace(edges[0], edges[-1], 241)
    ax.plot(
        x,
        np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
        "--",
        color=_INK,
        linewidth=1,
        label="N(0, 1)",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel("residual / sigma" if sigma else "residual / its standard deviation")
    ax.set_title("residual distribution", fontsize=10)
    ax.grid(True, **_GRID)
    ax.legend(fontsize=7)


def _model_update_panels(
    figure,
    record,
    residual,
    mask,
    true_flux,
    threshold,
    sigma,
    stop_descriptions,
    final_entropy,
):
    """The panels of the convergence record, one point per model update and,
    after them, a star for the residual written (``final``)."""
    from matplotlib.ticker import MaxNLocator

    n = len(record["model_update_stop_code"])
    update = np.arange(1, n + 1)
    final = n + 1
    region = "in the image" if mask is None else "in the mask"
    inside = residual.ravel() if mask is None else residual[mask]
    inside = inside[np.isfinite(inside)]
    axes = figure.subplots(1, 7)

    ax = axes[0]
    ax.plot(
        update,
        np.abs(_record_values(record, "start_peakres", n)),
        "o-",
        color=_COLORS[0],
        markersize=4,
        label="at the start",
    )
    end = _record_values(record, "peakres", n)
    if end is not None:
        ax.plot(
            update,
            np.abs(end),
            "o",
            markerfacecolor="none",
            color=_COLORS[0],
            markersize=5,
            label="at the end",
        )
    if inside.size:
        ax.plot(
            final,
            np.abs(inside).max(),
            "*",
            color=_COLORS[0],
            markersize=9,
            label="final",
        )
    ax.hlines(
        _record_values(record, "threshold_per_cycle", n),
        update - 0.4,
        update + 0.4,
        color=_COLORS[1],
        linestyle="--",
        label="threshold_per_cycle",
    )
    if threshold:
        ax.axhline(threshold, color=_INK, linestyle=":", label="threshold")
    ax.set_yscale("log")
    ax.set_title(f"|peak residual| {region}", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    flux = _record_values(record, "model_flux", n)
    ax.plot(update, flux, "o-", color=_COLORS[0], markersize=4, label="model")
    if true_flux is not None:
        ax.axhline(true_flux, color=_INK, linestyle=":", label="truth")
    negative = _record_values(record, "model_flux_negative", n)
    label = "model flux"
    if negative is not None and flux[-1]:
        label += f" (negative {100 * abs(negative[-1] / flux[-1]):.2f} %)"
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[2]
    iterations = _record_values(record, "iter_done", n)
    codes = np.asarray(record["model_update_stop_code"][:n], dtype=int)
    descriptions = stop_descriptions or {}
    known = sorted(descriptions)
    for code in sorted(set(codes.tolist())):
        index = known.index(code) if code in known else len(known) + code
        chosen = codes == code
        ax.bar(
            update[chosen],
            iterations[chosen],
            color=_COLORS[index % len(_COLORS)],
            label=f"{code} {descriptions.get(code, '')}".strip(),
        )
    ax.set_ylim(0, 1.45 * max(_largest(iterations), 1.0))
    ax.set_title(f"iterations ({int(np.nansum(iterations))})", fontsize=10)
    ax.legend(fontsize=6, title="model update stop", title_fontsize=7, loc="upper left")

    ax = axes[3]
    for kind, name, label in (
        ("residual-shaped", "n_residual_components", "residual-shaped"),
        ("Asp Gaussian", "n_scale_components", "Asp Gaussian"),
        ("significance", "n_significance_components", "significance (a subset)"),
        ("point", "n_point_components", "point"),
        ("Hogbom run", "n_hogbom_runs", "Hogbom runs"),
    ):
        counts = _record_values(record, name, n)
        if counts is None or not np.nansum(counts):
            continue
        style = _COMPONENT_STYLES[kind]
        ax.plot(
            update,
            np.where(counts > 0, counts, np.nan),
            marker=style["marker"],
            color=style["color"],
            markersize=5,
            label=f"{label} ({int(np.nansum(counts))})",
        )
    ax.set_yscale("log")
    ax.set_title("components and Hogbom runs per model update", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.set_ylim(top=30 * ax.get_ylim()[1])  # room for the legend
        ax.legend(fontsize=7)

    ax = axes[4]
    rms_in = _record_values(record, "start_rms_in_mask", n)
    ax.plot(update, rms_in, "o-", color=_COLORS[0], markersize=4, label=f"rms {region}")
    if inside.size:
        ax.plot(final, np.sqrt(np.mean(inside**2)), "*", color=_COLORS[0], markersize=9)
    mad = _record_values(record, "start_mad_rms_outside_mask", n)
    if mask is not None and mad is not None and np.isfinite(mad).any():
        outside = residual[~mask]
        outside = outside[np.isfinite(outside)]
        ax.plot(
            update, mad, "s-", color=_COLORS[1], markersize=4, label="MAD rms outside"
        )
        if outside.size:
            ax.plot(
                final,
                1.4826 * np.median(np.abs(outside - np.median(outside))),
                "*",
                color=_COLORS[1],
                markersize=9,
            )
    if sigma:
        ax.axhline(sigma, color=_INK, linestyle=":", label="sigma")
    ax.set_yscale("log")
    ax.set_title("residual at the start (star: final)", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[5]
    moments = _moments(inside) if inside.size else (np.nan, np.nan)
    for index, (name, label) in enumerate(
        (
            ("start_skew_in_mask", "skew"),
            ("start_excess_kurtosis_in_mask", "excess kurtosis"),
        )
    ):
        values = _record_values(record, name, n)
        if values is not None:
            ax.plot(
                update,
                values,
                marker=_MARKERS[index],
                color=_COLORS[index],
                markersize=4,
                label=label,
            )
        ax.plot(final, moments[index], "*", color=_COLORS[index], markersize=9)
    ax.axhline(0.0, color=_INK, linewidth=0.8)
    ax.set_title(f"residual {region} (star: final)", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)

    ax = axes[6]
    entropy = _record_values(record, "entropy", n + 1)
    ax.plot(update, entropy[:n], "o-", color=_COLORS[0], markersize=4)
    # cube imaging records the closing stop test too
    final_entropy = entropy[n] if entropy.size > n else final_entropy
    title = "residual entropy at the start"
    if final_entropy is not None and np.isfinite(final_entropy):
        ax.plot(final, final_entropy, "*", color=_COLORS[0], markersize=9)
        title += " (star: final)"
    ax.set_title(title, fontsize=10)

    for ax in axes:
        ax.set_xlim(0.5, final + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("model update")
        ax.grid(True, **_GRID)


def _iteration_history_panels(figure, record, mask, true_flux, shape):
    """The panels of the iteration history, one point per component."""
    from matplotlib.ticker import MaxNLocator

    n = len(record["model_update_stop_code"])

    def flat(name):
        return np.concatenate(
            [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in record[name][:n]]
        )

    scale = flat("iteration_component_scale")
    counts = [np.size(v) for v in record["iteration_component_scale"][:n]]
    boundaries = np.cumsum(counts)[:-1]
    component = np.arange(scale.size)
    kinds = np.where(
        scale < 0, "residual-shaped", np.where(scale > 0, "Asp Gaussian", "point")
    )
    present = [kind for kind in _COMPONENT_STYLES if np.any(kinds == kind)]
    # the most numerous kind first, so that the rarer ones are drawn on top
    drawn = sorted(present, key=lambda kind: -np.sum(kinds == kind))
    axes = figure.subplots(1, 6)

    def by_kind(ax, values, size=4):
        for kind in drawn:
            chosen = kinds == kind
            ax.scatter(
                component[chosen],
                values[chosen],
                s=size,
                color=_COMPONENT_STYLES[kind]["color"],
                marker=_COMPONENT_STYLES[kind]["marker"],
                label=f"{kind} ({int(chosen.sum())})",
                rasterized=True,
            )

    ax = axes[0]
    by_kind(ax, np.abs(flat("iteration_peak_residual")))
    threshold = _record_values(record, "threshold_per_cycle", n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    for k in range(n):
        ax.hlines(
            threshold[k], starts[k], starts[k + 1], color=_COLORS[1], linestyle="--"
        )
    ax.set_yscale("log")
    ax.set_title("|peak residual| before every component", fontsize=10)
    ax.legend(fontsize=7, markerscale=2, loc="upper right")

    ax = axes[1]
    by_kind(ax, flat("iteration_residual_rms"))
    ax.set_yscale("log")
    region = "in the image" if mask is None else "in the mask"
    ax.set_title(f"residual rms {region} after it", fontsize=10)

    ax = axes[2]
    ax.plot(component, flat("iteration_model_flux"), color=_COLORS[0], linewidth=1.2)
    if true_flux is not None:
        ax.axhline(true_flux, color=_INK, linestyle=":")
    ax.set_title("model flux after it", fontsize=10)

    ax = axes[3]
    by_kind(ax, scale)
    ax.set_title("component scale (pixels; -1 residual-shaped)", fontsize=10)

    for ax in axes[:4]:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        for boundary in boundaries:
            ax.axvline(boundary, color="#afb8c1", linewidth=0.6)
        ax.set_xlabel("component")
        ax.grid(True, **_GRID)

    ax = axes[4]
    flux = flat("iteration_component_flux")
    l_pixel, m_pixel = flat("iteration_component_l"), flat("iteration_component_m")
    largest = np.abs(flux).max() if flux.size else 1.0
    for kind in drawn:
        chosen = kinds == kind
        ax.scatter(
            l_pixel[chosen],
            m_pixel[chosen],
            s=2 + 30 * np.sqrt(np.abs(flux[chosen]) / largest),
            color=_COMPONENT_STYLES[kind]["color"],
            marker=_COMPONENT_STYLES[kind]["marker"],
            alpha=0.5,
            rasterized=True,
        )
    if mask is not None and mask.any() and not mask.all():
        ax.contour(
            np.asarray(mask, dtype=np.float64).T,
            levels=[0.5],
            colors="#1f2328",
            linewidths=0.8,
        )
    ax.set_xlim(-0.5, shape[0] - 0.5)
    ax.set_ylim(-0.5, shape[1] - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("l pixel")
    ax.set_ylabel("m pixel")
    ax.set_title("where the components are", fontsize=10)

    ax = axes[5]
    totals = [flux[kinds == kind].sum() for kind in present]
    ax.bar(
        present, totals, color=[_COMPONENT_STYLES[kind]["color"] for kind in present]
    )
    for x, (kind, total) in enumerate(zip(present, totals, strict=True)):
        ax.annotate(
            f"{int(np.sum(kinds == kind))}",
            (x, total),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_title("net flux and number of components", fontsize=10)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, axis="y", **_GRID)


def plot_plane_deconvolution_summary(
    record,
    model,
    residual,
    restored=None,
    mask=None,
    truth=None,
    model_error=None,
    true_flux=None,
    threshold=None,
    sigma=None,
    stop_descriptions=None,
    final_entropy=None,
    title=None,
):
    """What the deconvolution of one image plane did: its images, the
    convergence record of its model updates and, if the record holds it, the
    iteration history of its components.

    Three rows of panels. The images: the truth at the resolution of the
    restored image, the restored image, the model, the residual (the clean mask
    outlined), the model error and the distribution of the residual against
    that of Gaussian noise. One point per model update: the peak residual at its
    start and end with ``threshold_per_cycle``, the model flux, the iterations
    by stop code, the components of every kind, the rms of the residual and its
    skew, excess kurtosis and entropy; a star marks the residual written. With
    ``record_iteration_history`` (the ``iteration_*`` fields), one point per
    component: the peak residual before it, the rms of the residual after it,
    the model flux, its scale, where the components are and their net flux by
    kind. Without the history the third row is left out, without a model
    update the second too.

    Parameters
    ----------
    record : dict or None
        The convergence record of the plane: its fields in the ``"deconvolution"``
        record the imagers return, each with one value per model update (cube
        imaging adds the closing stop test to the fields of the residual). The
        fields ``model_update_stop_code``, ``iter_done``, ``start_peakres``,
        ``threshold_per_cycle``, ``model_flux``, ``start_rms_in_mask`` and
        ``entropy`` are needed; the others are drawn when present. A run that
        made no model update has no record (continuum imaging) or one without
        ``model_update_stop_code`` (cube imaging).
    model, residual : numpy.ndarray, [n_l, n_m]
    restored : numpy.ndarray, [n_l, n_m], optional
    mask : numpy.ndarray of bool, [n_l, n_m], optional
        The clean mask; ``None`` for none (the whole image).
    truth, model_error : numpy.ndarray, [n_l, n_m], optional
        The true sky at the resolution of the restored image, and the error of
        the model at that resolution, in the units of the restored image.
    true_flux : float, optional
        The flux of the true sky the model holds.
    threshold : float, optional
        The stopping threshold.
    sigma : float, optional
        The noise of the image; the residual is shown to 5 sigma and its
        distribution in units of sigma.
    stop_descriptions : dict, optional
        ``{code: description}`` of the model update stop codes.
    final_entropy : float, optional
        The entropy of the residual written, where the record does not hold it
        (continuum imaging returns it in its ``"residual_statistics"``).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.colors import AsinhNorm
    from matplotlib.figure import Figure

    residual = np.asarray(residual, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    n = len((record or {}).get("model_update_stop_code", []))
    history = n > 0 and any(
        np.size(v) for v in record.get("iteration_component_scale", [])[:n]
    )
    n_rows = 1 + (n > 0) + history
    figure = Figure(figsize=(26.0, 4.2 * n_rows), layout="constrained")
    rows = figure.subfigures(n_rows, 1, squeeze=False)[:, 0]

    panels = [
        ("truth at the clean beam", truth, "asinh"),
        ("restored image", restored, "asinh"),
        ("model", model, "asinh"),
        ("residual", residual, "signed"),
        ("model error", model_error, "signed"),
    ]
    panels = [panel for panel in panels if panel[1] is not None]
    axes = rows[0].subplots(1, len(panels) + 1)
    for ax, (label, plane, stretch) in zip(axes, panels, strict=False):
        plane = np.asarray(plane, dtype=np.float64)
        if stretch == "asinh":
            _image_panel(
                rows[0],
                ax,
                plane,
                label,
                "viridis",
                AsinhNorm(linear_width=0.01 * (_largest(plane) or 1.0)),
            )
        elif label == "residual":
            limit = 5 * sigma if sigma else None
            _image_panel(
                rows[0],
                ax,
                plane,
                "residual" + (" (5 sigma)" if sigma else ""),
                "RdBu_r",
                _signed_norm(plane, limit),
                mask,
            )
        else:
            _image_panel(rows[0], ax, plane, label, "RdBu_r", _signed_norm(plane))
    axes[0].set_ylabel("m pixel")
    _residual_histogram(axes[-1], residual, mask, sigma)

    if n:
        _model_update_panels(
            rows[1],
            record,
            residual,
            mask,
            true_flux,
            threshold,
            sigma,
            stop_descriptions,
            final_entropy,
        )
    else:
        title = f"{title}: no model update" if title else "no model update"
    if history:
        _iteration_history_panels(rows[2], record, mask, true_flux, residual.shape)
    if title:
        figure.suptitle(title, fontsize=12)
    return figure
