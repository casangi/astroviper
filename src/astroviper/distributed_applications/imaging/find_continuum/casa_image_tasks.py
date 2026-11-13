"""The CASA tasks and tools of the vendored findContinuum, on AstroVIPER images.

findContinuum addresses images by path, as CASA does. The dirty cube, its
primary beam and its point spread function are registered with
:func:`cube_session`; every reduction of a registered cube (``immoments``,
``imstat``, ``image.getprofile``) is a request to the session's ``reduce``
callable, which runs it as a map-reduce over the image store. Every other path
is a small image written here (:func:`astroviper.node_tasks.imaging.find_continuum.write_small_image`).
The tools that read a Measurement Set or need CASA's atmosphere model are not
provided: findContinuum takes the atmosphere from the caller and runs without
a Measurement Set.
"""

import contextlib
import os
import re

import numpy as np
import toolviper.utils.logger as logger
from astropy import units

from astroviper.node_tasks.imaging.find_continuum import (
    read_small_image,
    write_small_image,
)
from astroviper.processing_functions.image_analysis.make_auto_mask import _prune
from astroviper.processing_functions.imaging.find_continuum import (
    CASA_MOMENTS,
    evaluate_lattice_expression,
    image_statistics,
    lattice_expression_names,
)

# The CASA version whose findContinuum behaviour is reproduced.
casaVersion = "6.7.6"
AXIS_NAMES = ["Right Ascension", "Declination", "Stokes", "Frequency"]
MOMENT_SUFFIXES = {-1: ".average", 0: ".integrated", 8: ".maximum", 10: ".minimum"}

_session = None


class _Session:
    def __init__(self, cubes, reduce):
        self.cubes = {path.rstrip("/"): cube for path, cube in cubes.items()}
        self.reduce = reduce


@contextlib.contextmanager
def cube_session(cubes, reduce):
    """Register the cubes findContinuum reads and the reduction backend.

    Parameters
    ----------
    cubes : dict
        Path findContinuum uses to a dict with ``store`` (the image store),
        ``variable`` (its data variable), ``metadata`` (CASA image metadata),
        optionally ``primary_beam`` and ``primary_beam_limit`` (pixels where
        that variable is below the limit are masked, as in the cubes of
        ``tclean`` with ``pblimit``) and, for the dirty cube, ``atmosphere``
        (``{"transmission": ..., "tsky": ...}``, one value per channel). Each
        path must exist as a directory.
    reduce : callable
        ``reduce(requests)`` returns the finalized results of a list of
        reduction requests over the registered cubes.
    """
    global _session
    _session = _Session(cubes, reduce)
    try:
        yield _session
    finally:
        _session = None


def _cube(path):
    return None if _session is None else _session.cubes.get(path.rstrip("/"))


def _reduce(requests):
    return _session.reduce(requests)


def atmosphere_spectrum(img, value):
    """The atmospheric ``"transmission"`` or sky temperature ``"tsky"`` of the cube.

    Parameters
    ----------
    img : str
        Path of the registered dirty cube.
    value : {"transmission", "tsky"}
        Spectrum to return.

    Returns
    -------
    numpy.ndarray
        One value per channel.
    """
    cube = _cube(img)
    if cube is None or cube.get("atmosphere") is None:
        raise ValueError(f"No atmospheric spectra were given for {img}")
    return np.asarray(cube["atmosphere"][value], dtype=np.float64)


def parse_channel_selection(chans, nchan):
    """Channel indices of a CASA ``chans`` string (``"5~10;20,22"``), all if empty."""
    if chans in ("", None):
        return None
    channels = []
    for item in re.split(r"[;,]", str(chans)):
        item = item.strip()
        if not item:
            continue
        if "~" in item:
            first, last = (int(value) for value in item.split("~"))
            channels.extend(range(first, last + 1))
        else:
            channels.append(int(item))
    return np.array(sorted({c for c in channels if 0 <= c < nchan}), dtype=int)


def _read(path):
    """Values (masked pixels NaN), mask and metadata of a small image."""
    values, mask, metadata = read_small_image(path.rstrip("/"))
    return values, mask, metadata


def _read_cube_channels(path, channels, spatial=(slice(None), slice(None))):
    import xarray as xr

    cube = _cube(path)
    lazy_xds = xr.open_zarr(cube["store"], chunks=None)
    selection = dict(
        frequency=slice(None) if channels is None else list(channels),
        l=spatial[0],
        m=spatial[1],
    )

    def plane(variable):
        planes = lazy_xds[variable].isel(selection).sel(polarization=["I"]).values[0]
        return np.transpose(planes, (2, 3, 1, 0))

    values = plane(cube["variable"]).astype(np.float64)
    if cube.get("primary_beam_limit") is not None:
        values[plane(cube["primary_beam"]) < cube["primary_beam_limit"]] = np.nan
    return values, np.isfinite(values), dict(cube["metadata"], shape=list(values.shape))


def _small_image_metadata(metadata, nchan=1):
    return dict(metadata, shape=list(metadata["shape"][:3]) + [nchan])


def _expression_arrays(expression, extra=None):
    """Arrays of the small images a lattice expression refers to (masked pixels NaN)."""
    arrays = dict(extra or {})
    for name in lattice_expression_names(expression):
        if name in arrays:
            continue
        if _cube(name) is not None:
            raise NotImplementedError(
                f"{expression!r} needs the whole cube {name} in memory"
            )
        values, mask, _ = _read(name)
        arrays[name] = np.where(mask, values, np.nan)
    return arrays


def _mask_references(expression):
    """Small images and cubes a reduction mask refers to."""
    if not expression:
        return [], []
    names = lattice_expression_names(expression)
    return [n for n in names if _cube(n) is None], [
        n for n in names if _cube(n) is not None
    ]


def _cube_request(path, kind, chans="", mask="", **fields):
    nchan = _cube(path)["metadata"]["shape"][3]
    channels = parse_channel_selection(chans, nchan)
    return dict(
        kind=kind,
        image=path.rstrip("/"),
        mask=mask or "",
        channels=None if channels is None else channels.tolist(),
        **fields,
    )


class _Casalog:
    def post(self, message, priority="INFO", origin=""):
        if priority in ("WARN", "SEVERE"):
            logger.warning(str(message))
        else:
            logger.debug(str(message))

    def origin(self, origin):
        pass


casalog = _Casalog()


def immoments(
    imagename, moments=(0,), axis="spectral", chans="", mask="", outfile="", **kwargs
):
    """CASA ``immoments`` along the spectral axis (moments -1, 0, 8 and 10)."""
    if kwargs:
        raise NotImplementedError(
            f"immoments arguments {sorted(kwargs)} are not supported"
        )
    if axis not in ("spectral", 3):
        raise NotImplementedError("immoments only collapses the spectral axis")
    moments = list(moments)
    for moment in moments:
        if moment not in CASA_MOMENTS:
            raise NotImplementedError(f"immoments moment {moment} is not supported")
    cube = _cube(imagename)
    if cube is None:
        raise NotImplementedError("immoments is only supported on the registered cubes")
    metadata = cube["metadata"]
    request = _cube_request(
        imagename,
        "moments",
        chans,
        mask,
        moments=moments,
        frequency=(
            metadata["reference_value"][3]
            + metadata["increment"][3]
            * (np.arange(metadata["shape"][3]) - metadata["reference_pixel"][3])
        ).tolist(),
        rest_frequency=metadata["rest_frequency"],
    )
    maps = _reduce([request])[0]
    for moment in moments:
        path = outfile + MOMENT_SUFFIXES[moment] if len(moments) > 1 else outfile
        values = maps[moment][:, :, None, None]
        write_small_image(
            path, values, np.isfinite(values), _small_image_metadata(metadata)
        )
    return True


def _statistics_dict(stats, maxpos=None, minpos=None):
    # CASA returns empty arrays when no pixel of the image is valid.
    if np.ndim(stats["npts"]) == 0 and not stats["npts"]:
        return {key: np.array([], dtype=np.float64) for key in stats}
    result = {
        key: np.atleast_1d(np.asarray(value, dtype=np.float64))
        for key, value in stats.items()
    }
    if maxpos is not None:
        result["maxpos"] = np.asarray(maxpos)
        result["minpos"] = np.asarray(minpos)
    return result


def _box_slices(box, shape):
    if not box:
        return (slice(None), slice(None))
    x0, y0, x1, y1 = (int(value) for value in str(box).split(","))
    return (
        slice(max(x0, 0), min(x1, shape[0] - 1) + 1),
        slice(max(y0, 0), min(y1, shape[1] - 1) + 1),
    )


def imstat(
    imagename,
    axes=-1,
    box="",
    chans="",
    mask="",
    algorithm="classic",
    maxiter=-1,
    stretch=False,
    listit=True,
    verbose=True,
    **kwargs,
):
    """CASA ``imstat``: whole-image or per-channel (``axes=[0,1,2]``) statistics."""
    if kwargs:
        raise NotImplementedError(
            f"imstat arguments {sorted(kwargs)} are not supported"
        )
    per_channel = axes not in (-1, None, [])
    if per_channel and list(axes) != [0, 1, 2]:
        raise NotImplementedError(f"imstat axes={axes} is not supported")
    if _cube(imagename) is not None and not box:
        if algorithm != "classic":
            raise NotImplementedError("Only classic statistics are supported on a cube")
        request = _cube_request(
            imagename, "statistics", chans, mask, per_channel=per_channel
        )
        return _statistics_dict(_reduce([request])[0])
    if _cube(imagename) is not None:
        shape = _cube(imagename)["metadata"]["shape"]
        channels = parse_channel_selection(chans, shape[3])
        spatial = _box_slices(box, shape)
        values, valid, _ = _read_cube_channels(imagename, channels, spatial)
    else:
        values, valid, _ = _read(imagename)
        channels = parse_channel_selection(chans, values.shape[3])
        if channels is not None:
            values, valid = values[..., channels], valid[..., channels]
        spatial = _box_slices(box, values.shape)
        values, valid = values[spatial], valid[spatial]
    valid = valid & np.isfinite(values)
    if mask:
        valid &= np.broadcast_to(
            np.asarray(
                evaluate_lattice_expression(
                    mask, _expression_arrays(mask, {imagename.rstrip("/"): values})
                ),
                dtype=bool,
            ),
            values.shape,
        )
    if per_channel:
        rows = [
            image_statistics(values[..., c][valid[..., c]], algorithm, maxiter)
            for c in range(values.shape[3])
        ]
        return {key: np.array([row[key] for row in rows]) for key in rows[0]}
    stats = image_statistics(values[valid], algorithm, maxiter)
    if not valid.any():
        return _statistics_dict(stats)
    masked = np.where(valid, values, np.nan)
    positions = []
    # Pixel positions in the whole image, as CASA gives them.
    for index in (np.nanargmax(masked), np.nanargmin(masked)):
        position = [int(i) for i in np.unravel_index(index, values.shape)]
        position[0] += spatial[0].start or 0
        position[1] += spatial[1].start or 0
        if channels is not None:
            position[3] = int(channels[position[3]])
        positions.append(position)
    return _statistics_dict(stats, *positions)


def immath(
    imagename="", mode="evalexpr", expr="", outfile="", mask="", chans="", **kwargs
):
    """CASA ``immath`` in ``evalexpr`` mode."""
    if kwargs or mode != "evalexpr":
        raise NotImplementedError("immath supports mode='evalexpr' only")
    names = [imagename] if isinstance(imagename, str) else list(imagename)
    arrays, metadata, shape = {}, None, None
    for index, name in enumerate(names):
        if _cube(name) is not None:
            channels = parse_channel_selection(
                chans, _cube(name)["metadata"]["shape"][3]
            )
            if channels is None:
                raise NotImplementedError("immath on a whole cube is not supported")
            values, valid, image_metadata = _read_cube_channels(name, channels)
        else:
            values, valid, image_metadata = _read(name)
        arrays[f"IM{index}"] = arrays[name.rstrip("/")] = np.where(
            valid, values, np.nan
        )
        metadata = metadata or image_metadata
        shape = shape or values.shape
    result = np.broadcast_to(
        np.asarray(
            evaluate_lattice_expression(expr, _expression_arrays(expr, arrays)),
            dtype=np.float64,
        ),
        shape,
    )
    valid = np.isfinite(result)
    if mask:
        valid &= np.broadcast_to(
            np.asarray(
                evaluate_lattice_expression(mask, _expression_arrays(mask, arrays)),
                dtype=bool,
            ),
            shape,
        )
    write_small_image(
        outfile, np.array(result), valid, _small_image_metadata(metadata, shape[3])
    )
    return True


def imsubimage(imagename, outfile="", chans="", mask="", overwrite=False, **kwargs):
    """CASA ``imsubimage``: a channel selection and a pixel mask."""
    if kwargs:
        raise NotImplementedError(
            f"imsubimage arguments {sorted(kwargs)} are not supported"
        )
    if _cube(imagename) is not None:
        channels = parse_channel_selection(
            chans, _cube(imagename)["metadata"]["shape"][3]
        )
        if channels is None:
            raise NotImplementedError("imsubimage of a whole cube is not supported")
        values, valid, metadata = _read_cube_channels(imagename, channels)
    else:
        values, valid, metadata = _read(imagename)
        channels = parse_channel_selection(chans, values.shape[3])
        if channels is not None:
            values, valid = values[..., channels], valid[..., channels]
    if mask:
        arrays = _expression_arrays(
            mask, {imagename.rstrip("/"): np.where(valid, values, np.nan)}
        )
        valid = valid & np.broadcast_to(
            np.asarray(evaluate_lattice_expression(mask, arrays), dtype=bool),
            values.shape,
        )
    if os.path.exists(outfile) and not overwrite:
        raise FileExistsError(outfile)
    write_small_image(
        outfile, values, valid, _small_image_metadata(metadata, values.shape[3])
    )
    return True


def makemask(
    mode="copy", inpimage="", inpmask="", output="", overwrite=False, **kwargs
):
    """CASA ``makemask`` in ``copy`` mode: a 0/1 mask image from pixel masks or mask images."""
    if kwargs or mode != "copy":
        raise NotImplementedError("makemask supports mode='copy' only")
    if ":" in os.path.basename(output):
        raise NotImplementedError("makemask into an internal mask is not supported")
    _, _, metadata = (
        _read(inpimage)
        if _cube(inpimage) is None
        else (None, None, _cube(inpimage)["metadata"])
    )
    combined = None
    for source in [inpmask] if isinstance(inpmask, str) else list(inpmask):
        if source.endswith(":mask0"):
            _, valid, _ = _read(source[: -len(":mask0")])
            selected = valid
        else:
            values, valid, _ = _read(source)
            selected = valid & (values > 0)
        combined = selected if combined is None else combined | selected
    if os.path.exists(output) and not overwrite:
        raise FileExistsError(output)
    write_small_image(
        output,
        combined.astype(np.float64),
        np.ones(combined.shape, dtype=bool),
        _small_image_metadata(metadata, combined.shape[3]),
    )
    return True


def imhead(imagename, mode="summary", hdkey="", **kwargs):
    """CASA ``imhead``: ``mode='get'`` of the ``object`` keyword."""
    if mode != "get" or hdkey != "object" or kwargs:
        raise NotImplementedError("imhead supports mode='get', hdkey='object' only")
    cube = _cube(imagename)
    metadata = cube["metadata"] if cube is not None else _read(imagename)[2]
    return metadata["object"]


def imcollapse(
    imagename, outfile="", function="max", axes=3, overwrite=False, **kwargs
):
    """CASA ``imcollapse`` of a small image along one axis (``max``, ``min``, ``mean``)."""
    if kwargs or function not in ("max", "min", "mean"):
        raise NotImplementedError("imcollapse supports max, min and mean only")
    values, valid, metadata = _read(imagename)
    masked = np.where(valid, values, np.nan)
    collapse = {"max": np.nanmax, "min": np.nanmin, "mean": np.nanmean}[function]
    with np.errstate(all="ignore"):
        result = collapse(masked, axis=axes, keepdims=True)
    if os.path.exists(outfile) and not overwrite:
        raise FileExistsError(outfile)
    shape = list(result.shape)
    write_small_image(outfile, result, np.isfinite(result), dict(metadata, shape=shape))
    return True


def imregrid(*args, **kwargs):
    raise NotImplementedError("imregrid (a user mask on another grid) is not supported")


def imsmooth(*args, **kwargs):
    raise NotImplementedError(
        "imsmooth (meanSpectrumMethod='peakOverMad') is not supported"
    )


class _CoordinateSystem:
    def __init__(self, metadata):
        self.metadata = metadata

    def names(self):
        return list(AXIS_NAMES)

    def findaxisbyname(self, name, allowfriendlyname=True):
        lowered = name.lower()
        for index, axis_name in enumerate(AXIS_NAMES):
            if axis_name.lower().startswith(lowered) or (
                lowered == "spectral" and axis_name == "Frequency"
            ):
                return index
        raise ValueError(f"No axis named {name}")

    def increment(self, format="n", type=""):
        return {"numeric": np.array(self.metadata["increment"], dtype=np.float64)}

    def referencevalue(self, format="n", type=""):
        return {"numeric": np.array(self.metadata["reference_value"], dtype=np.float64)}

    def referencepixel(self, type=""):
        return {"numeric": np.array(self.metadata["reference_pixel"], dtype=np.float64)}

    def restfrequency(self):
        return {"unit": "Hz", "value": np.array([self.metadata["rest_frequency"]])}

    def referencecode(self, type="spectral", list=False):
        return [
            self.metadata["spectral_frame"]
            if type == "spectral"
            else self.metadata["direction_frame"]
        ]

    def epoch(self):
        return {
            "type": "epoch",
            "refer": "UTC",
            "m0": {"unit": "d", "value": self.metadata["epoch_mjd"]},
        }

    def telescope(self):
        return self.metadata["telescope"]

    def toworld(self, value, format="n"):
        pixel = np.asarray(value, dtype=np.float64)
        world = np.array(self.metadata["reference_value"], dtype=np.float64) + (
            pixel - np.array(self.metadata["reference_pixel"], dtype=np.float64)
        ) * np.array(self.metadata["increment"], dtype=np.float64)
        return {"numeric": world}

    def done(self):
        return True


def _quantity(value, unit):
    return {"unit": unit, "value": float(value)}


class iatool:
    """The CASA image tool methods findContinuum calls."""

    def __init__(self):
        self.path = None

    def open(self, infile, **kwargs):
        self.path = infile.rstrip("/")
        if _cube(self.path) is None and not os.path.exists(self.path):
            raise FileNotFoundError(infile)
        return True

    def close(self):
        self.path = None
        return True

    done = close

    def _metadata(self):
        cube = _cube(self.path)
        return cube["metadata"] if cube is not None else _read(self.path)[2]

    def shape(self):
        return np.array(self._metadata()["shape"], dtype=int)

    def coordsys(self):
        return _CoordinateSystem(self._metadata())

    def brightnessunit(self):
        return self._metadata()["brightness_unit"]

    def restoringbeam(self, channel=-1, polarization=-1):
        beams = self._metadata()["beams"]
        if beams is None:
            return {}

        def beam(major, minor, pa):
            return {
                "major": _quantity(np.degrees(major) * 3600.0, "arcsec"),
                "minor": _quantity(np.degrees(minor) * 3600.0, "arcsec"),
                "positionangle": _quantity(np.degrees(pa), "deg"),
            }

        if len(beams) == 1:
            return beam(*beams[0])
        return {
            "beams": {f"*{c}": {"*0": beam(*b)} for c, b in enumerate(beams)},
            "nChannels": len(beams),
            "nStokes": 1,
        }

    def maskhandler(self, op="default", name=None):
        if op != "default":
            raise NotImplementedError("maskhandler supports op='default' only")
        if _cube(self.path) is not None:
            return [""]
        _, valid, _ = _read(self.path)
        return ["" if valid.all() else "mask0"]

    def getregion(self, region=None, getmask=False, axes=None, **kwargs):
        if region is not None or axes is not None or kwargs:
            raise NotImplementedError(
                "getregion of a region or along axes is not supported"
            )
        if _cube(self.path) is not None:
            raise NotImplementedError("getregion of a whole cube is not supported")
        values, valid, _ = _read(self.path)
        return valid if getmask else values

    def getprofile(self, axis=3, function="mean", mask="", stretch=False, **kwargs):
        if kwargs or axis != 3:
            raise NotImplementedError("getprofile along the spectral axis only")
        if _cube(self.path) is None:
            raise NotImplementedError(
                "getprofile is only supported on the registered cubes"
            )
        request = _cube_request(self.path, "profile", "", mask, function=function)
        return {"values": _reduce([request])[0]}

    def statistics(self, **kwargs):
        if kwargs:
            raise NotImplementedError("statistics arguments are not supported")
        return imstat(self.path)


class qatool:
    """The CASA quanta tool methods findContinuum calls."""

    def convert(self, value, outunit):
        if isinstance(value, dict):
            quantity = np.asarray(value["value"]) * units.Unit(value["unit"])
        else:
            quantity = units.Quantity(value)
        converted = quantity.to(units.Unit(outunit)).value
        return {"unit": outunit, "value": converted}

    def quantity(self, value, unit=""):
        if isinstance(value, str):
            q = units.Quantity(value)
            return {"unit": str(q.unit), "value": float(q.value)}
        return {"unit": unit, "value": value}

    def done(self):
        return True


class synthesismaskhandler:
    """``pruneregions`` of the CASA synthesis mask handler."""

    def pruneregions(self, inmaskname, prunesize=0.0, chanflag=None, outmaskname=""):
        values, valid, metadata = _read(inmaskname)
        mask = valid & (values > 0)
        pruned = np.empty_like(mask)
        n_regions, n_pruned = [], []
        import scipy.ndimage

        for index in np.ndindex(mask.shape[2:]):
            plane = mask[(slice(None), slice(None), *index)]
            kept = _prune(plane, prunesize)
            pruned[(slice(None), slice(None), *index)] = kept
            n_regions.append(scipy.ndimage.label(plane)[1])
            n_pruned.append(n_regions[-1] - scipy.ndimage.label(kept)[1])
        write_small_image(
            outmaskname or inmaskname.rstrip("/") + ".pruned",
            pruned.astype(np.float64),
            np.ones(pruned.shape, dtype=bool),
            metadata,
        )
        return {"N_reg": np.array(n_regions), "N_reg_pruned": np.array(n_pruned)}

    def done(self):
        return True


def _unsupported_tool(name):
    class Tool:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"findContinuum reached a path that needs the CASA {name} tool, which AstroVIPER does not provide"
            )

    Tool.__name__ = name
    return Tool


attool = _unsupported_tool("atmosphere")
metool = _unsupported_tool("measures")
msmdtool = _unsupported_tool("msmetadata")
mstool = _unsupported_tool("ms")
rgtool = _unsupported_tool("regionmanager")
tbtool = _unsupported_tool("table")
