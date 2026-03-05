"""Utilities for plotting image-like arrays with astronomy-friendly orientation."""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import xarray as xr


def generate_astro_plot(
    data,
    wcs=None,
    show_world_axes: bool = False,
    cmap: str = "magma",
    figsize: Tuple[float, float] = (8.0, 8.0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot a 2D image array using astronomy-style pixel orientation.

    Parameters
    ----------
    data : numpy.ndarray
        2D image array with shape ``(nx, ny)`` in the package's internal convention.
    wcs : astropy.wcs.WCS or None, optional
        Celestial WCS associated with ``data``.
        Supported choices are:
        - ``None``: no WCS context; valid only when ``show_world_axes=False``.
        - ``astropy.wcs.WCS``: used when ``show_world_axes=True`` to render RA/Dec
          axes with correct handedness.
        Default is ``None``.
    show_world_axes : bool, optional
        Axis mode for the output plot.
        Supported choices are:
        - ``False``: regular matplotlib pixel axes.
        - ``True``: WCSAxes world-coordinate axes (e.g., RA/Dec).
        Default is ``False``.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    figsize : tuple[float, float], optional
        Figure size in inches as ``(width, height)``. Default is ``(8.0, 8.0)``.
    vmin : float or None, optional
        Lower bound for image normalization. If ``None``, matplotlib chooses automatically.
    vmax : float or None, optional
        Upper bound for image normalization. If ``None``, matplotlib chooses automatically.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes containing the rendered image.

    Notes
    -----
    ``imshow`` expects image memory order ``[row, col] == [y, x]``. This function
    transposes the input ``data`` before plotting so a source written to
    ``data[x, y]`` is displayed at pixel ``(x, y)`` in the rendered image.
    """
    if show_world_axes and wcs is None:
        raise ValueError("wcs must be provided when show_world_axes=True.")

    if show_world_axes:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
    else:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    return fig, ax


def plot_correct_orientation(
    xda: xr.DataArray, horizontal: str = "l", vertical: str = "m", vmin=None, vmax=None
):
    """Plot an ``xarray.DataArray`` and invert axes based on coordinate monotonicity.

    Parameters
    ----------
    xda : xr.DataArray
        Two-dimensional array to plot.
    horizontal : str, optional
        Name of the horizontal coordinate in ``xda``. Default is ``"l"``.
    vertical : str, optional
        Name of the vertical coordinate in ``xda``. Default is ``"m"``.
    vmin : float or None, optional
        Lower bound for image normalization. If ``None``, matplotlib chooses automatically.
    vmax : float or None, optional
        Upper bound for image normalization. If ``None``, matplotlib chooses automatically.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The plotted pcolormesh object.

    Notes
    -----
    This helper only checks coordinate monotonicity for axis inversion. It does
    not correct row/column versus x/y indexing conventions in raw 2D arrays.
    """
    mesh = xda.plot.pcolormesh(
        x=horizontal, y=vertical, vmin=vmin, vmax=vmax, cmap="viridis"
    )
    xvals = xda[horizontal].values
    if xvals[1] - xvals[0] < 0:
        mesh.axes.invert_xaxis()
    yvals = xda[vertical].values
    if yvals[1] - yvals[0] < 0:
        mesh.axes.invert_yaxis()
    return mesh
