"""Small matplotlib helpers for the simulation tutorials (Jones / Mueller montages).

Ported from SIRIUS ``display_tools.display_J`` / ``display_M``; kept next to the
notebooks so that they do not add a plotting dependency to the package.
"""

from __future__ import annotations

import numpy as np

_UNIT_SCALE = {
    "rad": 1.0,
    "deg": 180 / np.pi,
    "arcmin": 180 * 60 / np.pi,
    "arcsec": 180 * 3600 / np.pi,
}


def _component(values, val_type):
    if val_type == "abs":
        return np.abs(values)
    if val_type == "phase":
        return np.angle(values)
    if val_type == "real":
        return values.real
    if val_type == "imag":
        return values.imag
    raise ValueError("val_type must be 'abs', 'phase', 'real' or 'imag'")


def display_jones(
    jones_xds,
    parallactic_angle=0,
    frequency=0,
    val_type="abs",
    units="arcmin",
    figsize=(8, 7),
):
    """2x2 montage of the Jones beam image elements of one (parallactic angle, frequency)."""
    import matplotlib.pyplot as plt

    scale = _UNIT_SCALE[units]
    extent = [
        jones_xds.l.values[0] * scale,
        jones_xds.l.values[-1] * scale,
        jones_xds.m.values[0] * scale,
        jones_xds.m.values[-1] * scale,
    ]
    pols = list(jones_xds.polarization.values)
    from astroviper.utils.measurement_set_tools import polarization_index

    index = polarization_index(pols)
    data = jones_xds.JONES.isel(
        parallactic_angle=parallactic_angle, frequency=frequency
    ).values
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, p in zip(index, pols, strict=True):
        ax = axes.ravel()[i]
        ax.set_axis_on()
        im = ax.imshow(
            _component(data[pols.index(p)], val_type).T,
            origin="lower",
            extent=extent,
            cmap="viridis" if i in (0, 3) else "inferno",
        )
        ax.set_title(f"J {p} ({val_type})")
        ax.set_xlabel(f"l [{units}]")
        ax.set_ylabel(f"m [{units}]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(
        f"pa = {float(jones_xds.parallactic_angle.values[parallactic_angle]) * 180 / np.pi:.1f} deg, "
        f"frequency = {float(jones_xds.frequency.values[frequency]) / 1e9:.3f} GHz"
    )
    return fig


def display_mueller(
    mueller_xds,
    parallactic_angle=0,
    frequency=0,
    val_type="abs",
    units="arcmin",
    figsize=(12, 11),
):
    """4x4 montage of the selected Mueller matrix elements of one (parallactic angle, frequency)."""
    import matplotlib.pyplot as plt

    scale = _UNIT_SCALE[units]
    extent = [
        mueller_xds.l.values[0] * scale,
        mueller_xds.l.values[-1] * scale,
        mueller_xds.m.values[0] * scale,
        mueller_xds.m.values[-1] * scale,
    ]
    data = mueller_xds.MUELLER.isel(
        parallactic_angle=parallactic_angle, frequency=frequency
    ).values
    elements = list(mueller_xds.mueller_element.values)
    fig, axes = plt.subplots(4, 4, figsize=figsize, constrained_layout=True)
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, element in enumerate(elements):
        ax = axes.ravel()[int(element)]
        ax.set_axis_on()
        diagonal = element in (0, 5, 10, 15)
        im = ax.imshow(
            _component(data[k], val_type).T,
            origin="lower",
            extent=extent,
            cmap="viridis" if diagonal else "inferno",
        )
        ax.set_title(
            f"M[{int(element) // 4},{int(element) % 4}] {mueller_xds.polarization_1.values[k]}x{mueller_xds.polarization_2.values[k]}*"
        )
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Mueller matrix ({val_type}), units {units}")
    return fig
