import xarray as xr


def plot_correct_orientation(
    xda: xr.DataArray, horizontal: str = "l", vertical: str = "m", vmin=None, vmax=None
):
    """
    Plot the DataArray with correct orientation. By default, matplotlib will always
    plot the axes with increasing values from left to right and bottom to top, no matter
    the orientation of the data. This function checks the values of the horizontal and
    vertical coordinates and inverts the axes if necessary.
    Parameters
    ----------
    xda : xr.DataArray
        The DataArray to plot.
    horizontal : str, optional
        The name of the horizontal coordinate variable, by default "l".
    vertical : str, optional
        The name of the vertical coordinate variable, by default "m".
    """
    m = xda.plot.pcolormesh(
        x=horizontal, y=vertical, vmin=vmin, vmax=vmax, cmap="viridis"
    )
    xvals = xda[horizontal].values
    if xvals[1] - xvals[0] < 0:
        m.axes.invert_xaxis()
    yvals = xda[vertical].values
    if yvals[1] - yvals[0] < 0:
        m.axes.invert_yaxis()
