"""Telescope array layouts as MSv4 ``antenna_xds`` datasets.

Array configurations are read from CASA/``simobserve`` style ``.cfg`` files (a
copy of the standard set ships with AstroVIPER under
``astroviper/data/simulation/telescope_layouts``) and returned as an XRADIO
Measurement Set v4 antenna dataset (``xradio.measurement_set.schema.AntennaXds``)
so that the same object can be used by the simulator and by any tool that
understands processing sets.

``.cfg`` files have a header of ``# key=value`` comments followed by one line per
antenna ``x y z diameter name``.  Supported ``coordsys`` values are ``XYZ`` (ITRF
geocentric metres) and ``LOC`` (local tangent-plane east/north/up metres relative
to the observatory; converted with the same WGS84 formula as CASA's
``simutil.locxyz2itrf``).  ``UTM`` files are not supported — convert them to
``XYZ`` first (the shipped ``alma.out0*.cfg`` have been converted).

Observatory reference positions are the ITRF values of CASA's ``Observatories``
table (copied from the legacy SIRIUS ``tel.zarr`` products) so that uvw and
parallactic angles agree with CASA/SIRIUS.
"""

from __future__ import annotations

import importlib.resources
import os
from collections.abc import Sequence

import numpy as np
import xarray as xr

# ITRF geocentric (x, y, z) in metres of the array reference position, as given
# by CASA's ``me.observatory(name)`` measure converted to ITRF (values taken from
# the SIRIUS ``*.tel.zarr`` ``site_pos`` attributes).  ``None`` means "use the
# centroid of the antenna positions".
OBSERVATORY_ITRF_POSITIONS: dict[str, tuple[float, float, float] | None] = {
    "ACA": (2225066.2462818427, -5440107.534105327, -2481532.700718496),
    "ALMA": (2225142.180268967, -5440307.370348562, -2481029.851873547),
    "ALMASD": (2225142.180268967, -5440307.370348562, -2481029.851873547),
    "ATCA": (-4750915.837000001, 2792906.182, -3200483.747),
    "CARMA": (-2397389.6519749276, -4482068.562521395, 3843528.4147928944),
    "EVLA": (-1601156.673287362, -5041988.986065895, 3554879.2368205097),
    "NGVLA": (-1601156.673287362, -5041988.986065895, 3554879.2368205097),
    "IRAM_PDB": (4523998.400000571, 468045.24000039644, 4460309.760000601),
    "MEERKAT": (5109360.133, 2006852.586, -3238948.127),
    "SMA": (-5462448.949394941, -2491969.9165984807, 2164446.105479709),
    "VLA": (-1601185.3650000016, -5041977.546999999, 3554875.8700000006),
    "VLBA": None,
    # WGS84 lon/lat/height (0.11526445268379215 rad, 0.923574425836796 rad, 5 m)
    "WSRT": (3828488.8640655093, 443253.4204053785, 5064977.777133618),
}

# Default receptor (feed) polarization basis per observatory.
_DEFAULT_POLARIZATION_TYPE: dict[str, tuple[str, str]] = {
    "VLA": ("R", "L"),
    "EVLA": ("R", "L"),
    "VLBA": ("R", "L"),
}

# CASA ``.cfg`` names use ``observatory=VLA`` for both the (E)VLA and the legacy
# VLA; the telescope name recorded in the dataset is kept as written.

_WGS84_A = 6378137.0
_WGS84_B = 6356752.3142


def telescope_layout_directory() -> str:
    """Return the directory holding the shipped ``.cfg`` telescope layouts."""
    return str(
        importlib.resources.files("astroviper")
        / "data"
        / "simulation"
        / "telescope_layouts"
    )


def list_telescope_layouts() -> list[str]:
    """Names (without ``.cfg``) of the shipped telescope layouts."""
    return sorted(
        f[:-4] for f in os.listdir(telescope_layout_directory()) if f.endswith(".cfg")
    )


def observatory_position(
    telescope_name: str, antenna_position: np.ndarray | None = None
):
    """ITRF ``(x, y, z)`` metres of an observatory's reference position.

    Parameters
    ----------
    telescope_name : str
        Observatory name (case-insensitive), e.g. ``"VLA"``, ``"ALMA"``, ``"NGVLA"``.
    antenna_position : np.ndarray, [n_antenna, 3], optional
        Used as a fallback (centroid) when the observatory is unknown.

    Returns
    -------
    np.ndarray, [3]
    """
    key = telescope_name.upper()
    pos = OBSERVATORY_ITRF_POSITIONS.get(key)
    if pos is not None:
        return np.array(pos, dtype=np.float64)
    if antenna_position is None:
        raise ValueError(
            f"Unknown observatory '{telescope_name}' and no antenna positions given "
            "to compute a centroid."
        )
    return np.mean(np.asarray(antenna_position, dtype=np.float64), axis=0)


def local_tangent_plane_to_itrf(
    local_xyz: np.ndarray, reference_itrf: np.ndarray
) -> np.ndarray:
    """Convert local tangent-plane offsets (east, north, up; metres) to ITRF.

    Port of CASA ``simutil.locxyz2itrf``: the reference point is the WGS84
    geodetic position of ``reference_itrf``; ``x`` points east, ``y`` north and
    ``z`` along the ellipsoid normal.

    Parameters
    ----------
    local_xyz : np.ndarray, [n, 3]
    reference_itrf : np.ndarray, [3]

    Returns
    -------
    np.ndarray, [n, 3]
    """
    from astropy.coordinates import EarthLocation

    loc = EarthLocation.from_geocentric(*reference_itrf, unit="m")
    lon = float(loc.lon.rad)
    lat = float(loc.lat.rad)
    alt = float(loc.height.value)

    local_xyz = np.atleast_2d(np.asarray(local_xyz, dtype=np.float64))
    locx, locy, locz = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]
    sphi, cphi = np.sin(lat), np.cos(lat)
    ae = np.arccos(_WGS84_B / _WGS84_A)
    n = _WGS84_A / np.sqrt(1.0 - (np.sin(ae) * sphi) ** 2)
    term = (n + locz + alt) * cphi - locy * sphi
    clmb, slmb = np.cos(lon), np.sin(lon)
    x = term * clmb - locx * slmb
    y = term * slmb + locx * clmb
    z = (n * (_WGS84_B / _WGS84_A) ** 2 + locz + alt) * sphi + locy * cphi
    return np.stack([x, y, z], axis=-1)


def make_antenna_xds(
    antenna_name: Sequence[str],
    antenna_position: np.ndarray,
    dish_diameter: np.ndarray | float,
    telescope_name: str,
    station_name: Sequence[str] | None = None,
    mount: str | Sequence[str] = "ALT-AZ",
    polarization_type: Sequence[str] | None = None,
    blockage_diameter: np.ndarray | float | None = None,
    relocatable_antennas: bool = False,
) -> xr.Dataset:
    """Build an MSv4 ``antenna_xds`` from arrays.

    Parameters
    ----------
    antenna_name : sequence of str, [n_antenna]
    antenna_position : np.ndarray, [n_antenna, 3]
        ITRF geocentric positions in metres.
    dish_diameter : np.ndarray [n_antenna] or float, metres
    telescope_name : str
        Observatory name, stored per antenna (``telescope_name`` coordinate) and
        as ``overall_telescope_name``.
    station_name : sequence of str, optional
        Defaults to ``antenna_name``.
    mount : str or sequence of str
        Mount type(s); default ``"ALT-AZ"``.
    polarization_type : sequence of two str, optional
        Receptor polarization basis, e.g. ``("R", "L")`` or ``("X", "Y")``.
        Defaults to ``("R", "L")`` for the VLA family and ``("X", "Y")`` otherwise.
    blockage_diameter : np.ndarray [n_antenna] or float, metres, optional
        Stored as ``ANTENNA_BLOCKAGE`` when given.
    relocatable_antennas : bool

    Returns
    -------
    xr.Dataset
        Dataset following ``xradio.measurement_set.schema.AntennaXds``.
    """
    antenna_name = np.asarray(antenna_name, dtype=str)
    n_antenna = len(antenna_name)
    antenna_position = np.asarray(antenna_position, dtype=np.float64).reshape(
        n_antenna, 3
    )
    dish_diameter = np.broadcast_to(
        np.asarray(dish_diameter, dtype=np.float64), (n_antenna,)
    )
    if station_name is None:
        station_name = antenna_name
    station_name = np.asarray(station_name, dtype=str)
    mount_arr = np.broadcast_to(np.asarray(mount, dtype=str), (n_antenna,))
    if polarization_type is None:
        polarization_type = _DEFAULT_POLARIZATION_TYPE.get(
            telescope_name.upper(), ("X", "Y")
        )
    polarization_type = list(polarization_type)
    receptor_label = [f"pol_{i}" for i in range(len(polarization_type))]

    xds = xr.Dataset(
        coords={
            "antenna_name": ("antenna_name", antenna_name),
            "station_name": ("antenna_name", station_name),
            "mount": ("antenna_name", np.array(mount_arr)),
            "telescope_name": (
                "antenna_name",
                np.array([telescope_name] * n_antenna, dtype=str),
            ),
            "receptor_label": ("receptor_label", receptor_label),
            "polarization_type": (
                ("antenna_name", "receptor_label"),
                np.tile(np.array(polarization_type, dtype=str), (n_antenna, 1)),
            ),
            "cartesian_pos_label": ("cartesian_pos_label", ["x", "y", "z"]),
        }
    )
    xds["ANTENNA_POSITION"] = xr.DataArray(
        antenna_position,
        dims=("antenna_name", "cartesian_pos_label"),
        attrs={
            "type": "location",
            "units": "m",
            "frame": "ITRS",
            "coordinate_system": "geocentric",
            "origin_object_name": "earth",
        },
    )
    xds["ANTENNA_DISH_DIAMETER"] = xr.DataArray(
        np.array(dish_diameter),
        dims=("antenna_name",),
        attrs={"type": "quantity", "units": "m"},
    )
    if blockage_diameter is not None:
        xds["ANTENNA_BLOCKAGE"] = xr.DataArray(
            np.array(
                np.broadcast_to(
                    np.asarray(blockage_diameter, dtype=np.float64), (n_antenna,)
                )
            ),
            dims=("antenna_name",),
            attrs={"type": "quantity", "units": "m"},
        )
    xds.attrs.update(
        {
            "type": "antenna",
            "overall_telescope_name": telescope_name,
            "relocatable_antennas": bool(relocatable_antennas),
        }
    )
    return xds


def _parse_cfg(path: str) -> tuple[dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    header: dict[str, str] = {}
    xyz, diam, names = [], [], []
    with open(path) as fh:
        for line_no, raw in enumerate(fh):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line.lstrip("#").strip()
                if "=" in body:
                    key, _, value = body.partition("=")
                    header[key.strip().lower()] = value.strip()
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"{path}:{line_no + 1}: expected 'x y z diameter [name]'"
                )
            xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
            diam.append(float(parts[3]))
            names.append(parts[4] if len(parts) > 4 else f"A{len(names):03d}")
    return (
        header,
        np.array(xyz, dtype=np.float64),
        np.array(diam),
        np.array(names, dtype=str),
    )


def read_telescope_layout(
    layout: str,
    telescope_name: str | None = None,
    polarization_type: Sequence[str] | None = None,
    antenna_selection: Sequence[str] | Sequence[int] | None = None,
) -> xr.Dataset:
    """Read a CASA ``.cfg`` telescope layout into an MSv4 ``antenna_xds``.

    Parameters
    ----------
    layout : str
        Either the name of a shipped layout (e.g. ``"vla.d"``, ``"alma.cycle7.1"``,
        ``"ngvla-main-revC"``; see :func:`list_telescope_layouts`) or a path to a
        ``.cfg`` file.
    telescope_name : str, optional
        Overrides the ``# observatory=`` header (e.g. ``"EVLA"`` for a VLA layout
        when the EVLA reference position should be used).
    polarization_type : sequence of two str, optional
        Receptor polarization basis; see :func:`make_antenna_xds`.
    antenna_selection : sequence of str or int, optional
        Keep only these antennas (names or indices), in the given order.

    Returns
    -------
    xr.Dataset
        ``antenna_xds`` (see :func:`make_antenna_xds`).

    Raises
    ------
    FileNotFoundError
        If the layout is neither a shipped name nor an existing file.
    NotImplementedError
        For ``coordsys=UTM`` files.
    """
    path = layout
    if not os.path.isfile(path):
        candidate = os.path.join(telescope_layout_directory(), layout + ".cfg")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(
                f"Telescope layout '{layout}' is neither a file nor one of the shipped "
                f"layouts {list_telescope_layouts()}."
            )
        path = candidate

    header, xyz, diam, names = _parse_cfg(path)
    if telescope_name is None:
        telescope_name = header.get("observatory", os.path.basename(path).split(".")[0])
    coordsys = header.get("coordsys", "XYZ").upper()

    if coordsys.startswith("XYZ"):
        position = xyz
    elif coordsys.startswith("LOC"):
        position = local_tangent_plane_to_itrf(
            xyz, observatory_position(telescope_name)
        )
    elif coordsys.startswith("UTM"):
        raise NotImplementedError(
            f"{path}: coordsys=UTM is not supported; convert the file to coordsys=XYZ."
        )
    else:
        raise ValueError(f"{path}: unknown coordsys '{coordsys}'.")

    xds = make_antenna_xds(
        names,
        position,
        diam,
        telescope_name,
        polarization_type=polarization_type,
    )
    if antenna_selection is not None:
        sel = list(antenna_selection)
        if len(sel) and isinstance(sel[0], str):
            xds = xds.sel(antenna_name=sel)
        else:
            xds = xds.isel(antenna_name=np.asarray(sel, dtype=int))
    return xds
