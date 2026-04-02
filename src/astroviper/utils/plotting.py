"""Utilities for plotting image-like arrays with astronomy-friendly orientation."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr


def _resolve_plot_coords(
    data,
    x_coords: Optional[Union[str, np.ndarray]] = None,
    y_coords: Optional[Union[str, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve a 2-D plotting payload and its x/y coordinate vectors.

    Parameters
    ----------
    data : numpy.ndarray | xarray.DataArray
        Two-dimensional image-like input in the package convention ``data[x, y]``.
    x_coords : str | numpy.ndarray | None, optional
        X-axis coordinate selector. Supported choices are:
        - ``None``: use the 0th axis coordinate from an ``xarray.DataArray`` when
          available, otherwise use pixel indices ``0..nx-1``.
        - ``str``: use the named coordinate from the ``xarray.DataArray``.
        - array-like: explicit x coordinate values with length ``nx``.
    y_coords : str | numpy.ndarray | None, optional
        Y-axis coordinate selector. Supported choices are:
        - ``None``: use the 1st axis coordinate from an ``xarray.DataArray`` when
          available, otherwise use pixel indices ``0..ny-1``.
        - ``str``: use the named coordinate from the ``xarray.DataArray``.
        - array-like: explicit y coordinate values with length ``ny``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple ``(values, x_values, y_values)`` where ``values`` is the 2-D numeric
        image array and ``x_values`` / ``y_values`` are one-dimensional coordinate
        vectors for the 0th and 1st axes, respectively.

    Notes
    -----
    This helper preserves the package's x-first, y-second storage convention. The
    returned coordinate vectors therefore align with ``values.shape == (nx, ny)``.
    """
    if isinstance(data, xr.DataArray):
        if data.ndim != 2:
            raise ValueError(
                "data must be a 2D array-like object with shape (nx, ny) for plotting."
            )
        # Normalise to internal (x, y) axis order so dim_x / coord vectors are
        # always consistent regardless of how the caller stored the DataArray.
        # A (y, x) array is transposed here; other named layouts fall back to
        # positional order (dim_0 = x, dim_1 = y).
        dims = list(data.dims)
        if "x" in dims and "y" in dims and dims != ["x", "y"]:
            data = data.transpose("x", "y")
        dim_x = data.dims[0]
        dim_y = data.dims[1]
        nx, ny = data.shape

        def _coord_from_spec(spec, dim_name: str, axis_size: int) -> np.ndarray:
            if spec is None:
                if dim_name in data.coords:
                    coord_values = np.asarray(data.coords[dim_name].values, dtype=float)
                else:
                    coord_values = np.arange(axis_size, dtype=float)
            elif isinstance(spec, str):
                if spec not in data.coords:
                    raise ValueError(f"Coordinate {spec!r} not found in DataArray.")
                coord_values = np.asarray(data.coords[spec].values, dtype=float)
            else:
                coord_values = np.asarray(spec, dtype=float)
            if coord_values.ndim != 1 or coord_values.size != axis_size:
                raise ValueError(
                    f"Coordinate values for {dim_name!r} must be 1-D with length {axis_size}."
                )
            return coord_values

        x_values = _coord_from_spec(x_coords, dim_x, nx)
        y_values = _coord_from_spec(y_coords, dim_y, ny)
        # Materialize here — plotting always requires concrete values.
        return np.asarray(data.values), x_values, y_values

    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(
            "data must be a 2D array-like object with shape (nx, ny) for plotting."
        )

    def _coord_from_array(
        spec: Optional[Union[str, np.ndarray]], axis_size: int, axis_name: str
    ) -> np.ndarray:
        if spec is None:
            return np.arange(axis_size, dtype=float)
        if isinstance(spec, str):
            raise ValueError(
                f"String {axis_name} coordinate selectors require an xarray.DataArray input."
            )
        coord_values = np.asarray(spec, dtype=float)
        if coord_values.ndim != 1 or coord_values.size != axis_size:
            raise ValueError(
                f"Coordinate values for {axis_name} must be 1-D with length {axis_size}."
            )
        return coord_values

    x_values = _coord_from_array(x_coords, values.shape[0], "x")
    y_values = _coord_from_array(y_coords, values.shape[1], "y")
    return values, x_values, y_values


def generate_plot(
    data,
    wcs=None,
    show_world_axes: bool = False,
    x_coords: Optional[Union[str, np.ndarray]] = None,
    y_coords: Optional[Union[str, np.ndarray]] = None,
    title: Optional[str] = None,
    cmap: str = "magma",
    figsize: Tuple[float, float] = (8.0, 8.0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Plot a 2-D image using x/y coordinates or celestial WCS when available.

    Parameters
    ----------
    data : numpy.ndarray | xarray.DataArray
        Two-dimensional image array in the package convention ``data[x, y]``.
    wcs : astropy.wcs.WCS | None, optional
        Celestial WCS associated with ``data``.
        Supported choices are:
        - ``None``: no WCS projection; if ``show_world_axes=True`` the helper uses
          explicit or inferred x/y coordinate vectors instead.
        - ``astropy.wcs.WCS``: render celestial world axes through WCSAxes.
        Default is ``None``.
    show_world_axes : bool, optional
        Axis mode for the output plot.
        Supported choices are:
        - ``False``: regular matplotlib pixel axes.
        - ``True``: use WCS axes when ``wcs`` is provided, otherwise use x/y
          coordinate vectors.
        Default is ``False``.
    x_coords : str | numpy.ndarray | None, optional
        X coordinate selector. For ``xarray.DataArray`` input this may be a coordinate
        name or a one-dimensional array. For plain arrays this must be ``None`` or a
        one-dimensional array. If omitted, the 0th axis is treated as x and uses the
        DataArray's matching coordinate or pixel indices.
    y_coords : str | numpy.ndarray | None, optional
        Y coordinate selector. For ``xarray.DataArray`` input this may be a coordinate
        name or a one-dimensional array. For plain arrays this must be ``None`` or a
        one-dimensional array. If omitted, the 1st axis is treated as y and uses the
        DataArray's matching coordinate or pixel indices.
    title : str | None, optional
        Plot title. If provided, it is displayed on the rendered axes.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"magma"``.
    figsize : tuple[float, float], optional
        Figure size in inches as ``(width, height)``. Default is ``(8.0, 8.0)``.
    vmin : float | None, optional
        Lower bound for image normalization. If ``None``, matplotlib chooses automatically.
    vmax : float | None, optional
        Upper bound for image normalization. If ``None``, matplotlib chooses automatically.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axes containing the rendered image.

    Notes
    -----
    The helper preserves the package's x-first array convention by plotting
    ``data.T`` with ``origin="lower"``. When ``show_world_axes=True`` and no WCS is
    supplied, the helper renders a coordinate-aware plot using the resolved x/y
    coordinate vectors rather than requiring an ``astropy.wcs.WCS`` object.
    """
    values, x_values, y_values = _resolve_plot_coords(
        data=data, x_coords=x_coords, y_coords=y_coords
    )

    if isinstance(data, xr.DataArray):
        # Mirror the axis-normalisation applied in _resolve_plot_coords: if the
        # DataArray has named "x"/"y" dims in (y, x) order, swap to match the
        # resolved (x, y) layout so labels stay consistent.
        dims = list(data.dims)
        if "x" in dims and "y" in dims and dims != ["x", "y"]:
            x_dim, y_dim = "x", "y"
        else:
            x_dim, y_dim = dims[0], dims[1]
        default_x_label = x_coords if isinstance(x_coords, str) else x_dim
        default_y_label = y_coords if isinstance(y_coords, str) else y_dim
        colorbar_label = data.name if data.name is not None else "value"
    else:
        default_x_label = "x"
        default_y_label = "y"
        colorbar_label = "value"

    import matplotlib.pyplot as plt

    if show_world_axes and wcs is not None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
        image = ax.imshow(values.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        # WCSAxes derives world-coordinate labels (e.g. RA/Dec) from the WCS
        # object itself; only override when the caller explicitly requested a
        # custom label via a string coordinate selector.
        if isinstance(x_coords, str):
            ax.coords[0].set_axislabel(default_x_label)
        if isinstance(y_coords, str):
            ax.coords[1].set_axislabel(default_y_label)
        if title is not None:
            ax.set_title(title)
        fig.colorbar(image, ax=ax, label=colorbar_label)
        return fig, ax

    if show_world_axes:
        fig, ax = plt.subplots(figsize=figsize)
        mesh = ax.pcolormesh(
            x_values,
            y_values,
            values.T,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
        ax.set_ylim(float(y_values[0]), float(y_values[-1]))
        ax.set_aspect("equal")
        ax.set_xlabel(default_x_label)
        ax.set_ylabel(default_y_label)
        if title is not None:
            ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label=colorbar_label)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(values.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel(default_x_label)
    ax.set_ylabel(default_y_label)
    if title is not None:
        ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    return fig, ax
