"""Performance-oriented (``_skunk_works``) FITS output for cube imaging.

FITS counterpart of the direct-Zarr writers in
:mod:`astroviper.node_tasks.imaging.utils.skunk_works`, following the same
two-phase pattern the Zarr image store uses:

1. The **driver** (``distributed_applications`` layer) calls
   :func:`create_empty_fits_images` once: for every kept image variable it
   creates ``<image_store>/<VARIABLE>.fits`` with the complete FITS header --
   generated from the empty image dataset so that XRADIO's FITS reader
   (:func:`xradio.image.open_image`) reconstructs the same coordinates and
   attributes as the Zarr store -- and a **sparse** (``ftruncate``\\ d) data
   area sized for the full cube.
2. Each **node task** calls :func:`write_result_chunk_to_fits_skunk_works` to
   ``pwrite`` its frequency chunk directly into the pre-created files. The
   frequency axis is deliberately the *last* (slowest-varying) FITS axis, so a
   task's block of consecutive channels is one contiguous byte range: parallel
   tasks write disjoint ranges with no locking and no file creation --
   the same "single parallel file" pattern as the sharded Zarr writer.

Layout of each FITS file (conforming to what
:func:`xradio.image._util._fits.xds_from_fits._fits_image_to_xds` reads):

* PRIMARY HDU: 4 axes ``RA---<proj>, DEC--<proj>, STOKES, FREQ`` with big-endian
  IEEE floats (``BITPIX`` -32 or -64). The boolean ``MASK`` is stored as floats
  (FITS images have no boolean type).
* Optional ``BEAMS`` binary-table extension (``CASAMBM = T`` in the primary
  header) carrying the per-channel/per-polarization beam-fit parameters
  (``BMAJ, BMIN, BPA, CHAN, POL`` -- CASA's multi-beam convention). Rows are
  fixed-size and channel-major, so node tasks ``pwrite`` their channels' rows
  at disjoint offsets exactly like the image planes.

Unwritten planes of a sparse data area read back as 0.0 (filesystem holes),
not NaN -- unlike the NaN-filled empty Zarr store. Every channel is written by
exactly one task in normal operation, so this only matters for chunks whose
task failed and was skipped.

Only image-domain variables (dims ``time, frequency, polarization, l, m``) and
beam-fit-parameter variables are supported; uv-domain and complex-valued
variables have no FITS representation and raise at creation time.
"""

from __future__ import annotations

import os
import shutil
import struct

import numpy as np

# The FITS writers share the transient-I/O (Lustre eviction) retry machinery
# with the direct-Zarr skunk-works module.
from astroviper.node_tasks.imaging.utils.skunk_works import (
    _EIO_RETRY_BACKOFF_SECONDS,
    _TRANSIENT_IO_ERRNOS,
    _retry_transient_io,
)

_FITS_BLOCK = 2880

# BUNIT / BTYPE written per logical image variable (defaults below for the rest).
_VARIABLE_FITS_UNITS = {
    "sky_model": ("Jy/pixel", "Intensity"),
    "sky_deconvolved": ("Jy/pixel", "Intensity"),
    "point_spread_function": ("", "Intensity"),
    "primary_beam": ("", "Intensity"),
    "mask": ("", "Intensity"),
}
_DEFAULT_FITS_UNITS = ("Jy/beam", "Intensity")

# Image variables whose FITS file carries the BEAMS extension (when a
# beam-fit-params variable is kept). Masks and primary beams have no
# restoring beam.
_BEAM_CARRYING_VARIABLES = frozenset(
    {
        "sky",
        "sky_dirty",
        "sky_residual",
        "sky_model",
        "sky_restored",
        "sky_deconvolved",
        "point_spread_function",
    }
)

# Fixed BEAMS-table row layout written by create_empty_fits_images and assumed
# by the row writer: BMAJ, BMIN, BPA as big-endian float64 then CHAN, POL as
# big-endian int32 (CASA's multi-beam column order; readers use positions).
_BEAMS_ROW_STRUCT = struct.Struct(">dddii")

_FULL_DIMS_LM = ("time", "frequency", "polarization", "l", "m")
_BEAM_PARAMS_DIMS = ("time", "frequency", "polarization", "beam_params_label")


def _pad_to_block(nbytes):
    """Round ``nbytes`` up to a whole number of 2880-byte FITS blocks."""
    return -(-nbytes // _FITS_BLOCK) * _FITS_BLOCK


def _fits_float_dtype(double_precision):
    """(numpy dtype, BITPIX) of the on-disk image floats."""
    return (np.dtype(">f8"), -64) if double_precision else (np.dtype(">f4"), -32)


def _stokes_axis(pol_labels):
    """FITS ``STOKES`` axis ``(crval, cdelt, crpix)`` for the polarization labels.

    A FITS axis is linear, and XRADIO's reader decodes label ``i`` as
    ``stokes_types[crval - cdelt*(crpix-1) + i]`` for unit ``cdelt``, so the
    labels must map to consecutive ascending casacore Stokes codes (true for
    the Stokes outputs of the imager, e.g. ``["I"]`` or ``["I", "Q"]``).
    """
    from xradio.measurement_set._utils._utils.stokes_types import stokes_types

    label_to_code = {label: code for code, label in stokes_types.items()}
    try:
        codes = [label_to_code[label] for label in pol_labels]
    except KeyError as exc:
        raise ValueError(f"Unknown polarization label {exc} for FITS output.") from exc
    if list(codes) != list(range(codes[0], codes[0] + len(codes))):
        raise ValueError(
            f"Polarization labels {list(pol_labels)} map to non-consecutive "
            f"Stokes codes {codes}; a linear FITS STOKES axis cannot represent "
            "them."
        )
    return float(codes[0]), 1.0, 1.0


def _frequency_axis(freq_values):
    """FITS ``FREQ`` axis ``(crval, cdelt, crpix, max_deviation_hz)`` for the
    channel centers.

    A FITS axis is linear, and the imager never regrids spectrally, so a
    non-uniform channel grid (normal when the imaged processing set
    concatenates spectral windows) cannot be represented exactly. Such a grid
    is approximated by the uniform axis through the first and last channel
    centers, with a logged warning; ``max_deviation_hz`` is the worst
    ``|true - linear|`` channel-center error (``0.0`` when the axis is exact)
    and is recorded as a ``HISTORY`` card by :func:`_build_primary_header`.
    """
    import toolviper.utils.logger as logger

    freq_values = np.asarray(freq_values, dtype=np.float64)
    if freq_values.size == 1:
        # Any nonzero increment is valid for a single-channel axis (the reader
        # evaluates the axis only at the reference pixel).
        return float(freq_values[0]), 1.0, 1.0, 0.0
    cdelt = float((freq_values[-1] - freq_values[0]) / (freq_values.size - 1))
    if cdelt == 0.0:
        raise ValueError(
            "Frequency coordinates have zero overall span; a FITS FREQ axis "
            "cannot represent them."
        )
    linear = freq_values[0] + cdelt * np.arange(freq_values.size)
    max_deviation = float(np.max(np.abs(freq_values - linear)))
    if max_deviation <= 1e-6 * abs(cdelt):
        return float(freq_values[0]), cdelt, 1.0, 0.0
    logger.warning(
        "FITS FREQ axis: channel centers are not uniformly spaced (e.g. "
        "concatenated spectral windows), which a linear FITS axis cannot "
        "represent. Writing the uniform axis through the first and last "
        f"centers instead; worst channel-center error {max_deviation:.6g} Hz "
        f"({max_deviation / abs(cdelt):.3g} channel widths). A HISTORY card "
        "records the approximation."
    )
    return float(freq_values[0]), cdelt, 1.0, max_deviation


def _direction_axis_values(coord_values):
    """``(cdelt_deg, crpix)`` of the ``l`` or ``m`` coordinate (linear, rad)."""
    values = np.asarray(coord_values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("FITS output needs at least 2 pixels per direction axis.")
    return float(np.degrees(values[1] - values[0])), int(np.argmin(np.abs(values))) + 1


def _build_primary_header(
    img_xds, bitpix, bunit, btype, telescope_name, has_beams_extension
):
    """Primary-HDU header for one image variable, generated from the empty image.

    Every keyword XRADIO's FITS reader requires is written (direction axes +
    ``RADESYS``/``EQUINOX``/``LONPOLE``/``LATPOLE``/``PC``, spectral axis +
    ``RESTFRQ``/``SPECSYS``, ``TELESCOP``, ``DATE-OBS``/``TIMESYS``), and every
    keyword written is one the reader consumes or excludes from the user attrs,
    so nothing leaks into ``attrs["user"]`` on read. A non-uniform frequency
    grid is approximated (see :func:`_frequency_axis`) and noted in a
    ``HISTORY`` card (also excluded by the reader).
    """
    from astropy.io import fits
    from astropy.time import Time

    if img_xds.sizes["time"] != 1:
        raise ValueError(
            "FITS output supports a single time plane (4 FITS axes); got "
            f"{img_xds.sizes['time']} time planes."
        )

    csys = img_xds.attrs["coordinate_system_info"]
    reference_direction = csys["reference_direction"]
    frame = reference_direction["attrs"]["frame"].upper()
    if frame not in ("FK5", "ICRS"):
        raise ValueError(
            f"Direction frame {frame!r} is not supported for FITS output "
            "(XRADIO's FITS reader supports RA/DEC axes: FK5 or ICRS)."
        )
    projection = csys["projection"]

    cdelt1, crpix1 = _direction_axis_values(img_xds.l.values)
    cdelt2, crpix2 = _direction_axis_values(img_xds.m.values)
    crval3, cdelt3, crpix3 = _stokes_axis(img_xds.polarization.values)
    crval4, cdelt4, crpix4, freq_max_deviation = _frequency_axis(
        img_xds.frequency.values
    )
    pc = csys["pixel_coordinate_transformation_matrix"]
    lonpole_deg, latpole_deg = np.degrees(csys["native_pole_direction"]["data"])
    time_attrs = img_xds.time.attrs
    obs_time = Time(
        float(img_xds.time.values[0]),
        format=time_attrs.get("format", "mjd"),
        scale=time_attrs.get("scale", "utc"),
    )

    header = fits.Header()
    header["SIMPLE"] = (True, "conforms to FITS standard")
    header["BITPIX"] = bitpix
    header["NAXIS"] = 4
    header["NAXIS1"] = img_xds.sizes["l"]
    header["NAXIS2"] = img_xds.sizes["m"]
    header["NAXIS3"] = img_xds.sizes["polarization"]
    header["NAXIS4"] = img_xds.sizes["frequency"]
    header["EXTEND"] = True
    if bunit:
        header["BUNIT"] = bunit
    header["BTYPE"] = btype
    header["CTYPE1"] = f"RA---{projection}"
    header["CRVAL1"] = float(np.degrees(reference_direction["data"][0]))
    header["CDELT1"] = cdelt1
    header["CRPIX1"] = float(crpix1)
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRVAL2"] = float(np.degrees(reference_direction["data"][1]))
    header["CDELT2"] = cdelt2
    header["CRPIX2"] = float(crpix2)
    header["CUNIT2"] = "deg"
    header["CTYPE3"] = "STOKES"
    header["CRVAL3"] = crval3
    header["CDELT3"] = cdelt3
    header["CRPIX3"] = crpix3
    header["CUNIT3"] = ""
    header["CTYPE4"] = "FREQ"
    header["CRVAL4"] = crval4
    header["CDELT4"] = cdelt4
    header["CRPIX4"] = crpix4
    header["CUNIT4"] = "Hz"
    for i in (0, 1):
        for j in (0, 1):
            header[f"PC{i + 1}_{j + 1}"] = float(pc[i][j])
    header["LONPOLE"] = float(lonpole_deg)
    header["LATPOLE"] = float(latpole_deg)
    header["RADESYS"] = frame
    if frame != "ICRS":
        header["EQUINOX"] = 2000.0
    header["RESTFRQ"] = float(img_xds.frequency.attrs["rest_frequency"]["data"])
    header["SPECSYS"] = img_xds.frequency.attrs.get("observer", "lsrk").upper()
    header["TELESCOP"] = telescope_name
    header["DATE-OBS"] = obs_time.isot
    header["TIMESYS"] = time_attrs.get("scale", "utc").upper()
    header["ORIGIN"] = "AstroVIPER"
    if freq_max_deviation != 0.0:
        header["HISTORY"] = (
            "FREQ axis approximates non-uniform channel centers (worst error "
            f"{freq_max_deviation:.6g} Hz = "
            f"{freq_max_deviation / abs(cdelt4):.3g} channel widths)."
        )
    if has_beams_extension:
        header["CASAMBM"] = (True, "CASA multiple beams per channel/polarization")
    return header


def _build_beams_extension_bytes(n_chan, n_pol):
    """Serialized ``BEAMS`` binary-table extension (header + zero-filled rows).

    One fixed-size row per (channel, polarization), channel-major, in CASA's
    multi-beam column order ``BMAJ, BMIN, BPA, CHAN, POL`` (radians). Beam
    values start as zero and are overwritten in place by the node tasks; the
    ``CHAN``/``POL`` index columns are correct from the start.
    """
    from astropy.io import fits

    n_rows = n_chan * n_pol
    zeros = np.zeros(n_rows, dtype=np.float64)
    columns = [
        fits.Column(name="BMAJ", format="D", unit="rad", array=zeros),
        fits.Column(name="BMIN", format="D", unit="rad", array=zeros),
        fits.Column(name="BPA", format="D", unit="rad", array=zeros),
        fits.Column(
            name="CHAN",
            format="J",
            array=np.repeat(np.arange(n_chan, dtype=np.int32), n_pol),
        ),
        fits.Column(
            name="POL",
            format="J",
            array=np.tile(np.arange(n_pol, dtype=np.int32), n_chan),
        ),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="BEAMS")
    hdu.header["NCHAN"] = n_chan
    hdu.header["NPOL"] = n_pol
    if hdu.header["NAXIS1"] != _BEAMS_ROW_STRUCT.size:
        raise AssertionError(
            f"BEAMS row is {hdu.header['NAXIS1']} bytes; the fixed-slot row "
            f"writer assumes {_BEAMS_ROW_STRUCT.size} (BMAJ,BMIN,BPA,CHAN,POL)."
        )
    header_bytes = hdu.header.tostring().encode("ascii")
    data_bytes = hdu.data.tobytes()
    return (
        header_bytes
        + data_bytes
        + b"\0" * (_pad_to_block(len(data_bytes)) - len(data_bytes))
    )


def _classify_kept_variables(image_data_variables_keep, double_precision):
    """Split the keep list into image variables and beam-fit-params variables.

    Uses the imaging variable registry in :mod:`astroviper.utils.io` (the
    single source of truth for names/dims/dtypes) and rejects variables with no
    FITS representation (uv-domain / complex normalization variables).
    """
    from astroviper.utils.io import (
        imaging_data_variables_and_dims_double_precision,
        imaging_data_variables_and_dims_single_precision,
    )

    definitions = (
        imaging_data_variables_and_dims_double_precision
        if double_precision
        else imaging_data_variables_and_dims_single_precision
    )
    image_variables, beam_variables, unsupported = [], [], []
    for key in image_data_variables_keep:
        dims = tuple(definitions[key]["dims"])
        if dims == _FULL_DIMS_LM:
            image_variables.append(key)
        elif dims == _BEAM_PARAMS_DIMS:
            beam_variables.append(key)
        else:
            unsupported.append(key)
    if unsupported:
        raise ValueError(
            f"Variables {unsupported} have no FITS representation (only "
            "image-domain time/frequency/polarization/l/m variables and "
            "beam-fit-params variables can be written to FITS)."
        )
    return image_variables, beam_variables


def create_empty_fits_images(
    image_store,
    img_xds,
    image_data_variables_keep,
    double_precision=False,
    telescope_name="UNKNOWN",
    overwrite=False,
):
    """Pre-create one XRADIO-conformant FITS file per kept image variable.

    Run ONCE in the driver (the FITS counterpart of writing the empty Zarr
    image + :func:`~astroviper.utils.io.create_empty_data_variables_on_disk`).
    Creates the directory ``image_store`` containing ``<VARIABLE>.fits`` for
    every image-domain variable in ``image_data_variables_keep``, each with:

    * a complete primary header generated from the (empty) ``img_xds`` so that
      :func:`xradio.image.open_image` reads back the same coordinates,
      reference direction, spectral frame, and observation date;
    * a sparse (``ftruncate``\\ d) primary data area sized for the full cube --
      no pixel data is materialised, so creation is metadata-cost only;
    * when a beam-fit-params variable is kept, a zero-filled ``BEAMS``
      binary-table extension (and ``CASAMBM = T``) on the beam-carrying
      variables, whose fixed-size rows the node tasks later overwrite in place.

    The concurrent write phase then creates no files and writes only disjoint
    byte ranges (:func:`write_result_chunk_to_fits_skunk_works`).

    Parameters
    ----------
    image_store : str
        Output directory; created (or replaced when ``overwrite``) to hold one
        ``<VARIABLE>.fits`` file per kept image variable.
    img_xds : xarray.Dataset
        Empty image dataset from :func:`xradio.image.make_empty_sky_image`
        holding the full cube's coordinates and attributes. Must have a single
        time plane (FITS has 4 image axes).
    image_data_variables_keep : list of str
        Logical image-variable keys to create (e.g. ``"sky_residual"``); each
        is upper-cased to the FITS file name. Variables with no FITS
        representation (uv-domain / complex) raise ``ValueError``.
    double_precision : bool, optional
        If ``True`` write 64-bit floats (``BITPIX`` -64), else 32-bit
        (``BITPIX`` -32). Default ``False``.
    telescope_name : str, optional
        Value of the ``TELESCOP`` keyword (required by XRADIO's FITS reader).
        Default ``"UNKNOWN"``.
    overwrite : bool, optional
        Replace an existing ``image_store``. Default ``False`` (raise
        ``FileExistsError``).

    Returns
    -------
    dict
        ``{variable_key: fits_path}`` of the created files.
    """
    image_variables, beam_variables = _classify_kept_variables(
        image_data_variables_keep, double_precision
    )
    if not image_variables:
        raise ValueError(
            "image_data_variables_keep contains no image-domain variables to "
            "write to FITS."
        )

    if os.path.exists(image_store):
        if not overwrite:
            raise FileExistsError(
                f"{image_store} already exists. Set overwrite=True to replace it."
            )
        if os.path.isdir(image_store):
            shutil.rmtree(image_store)
        else:
            os.remove(image_store)
    os.makedirs(image_store)

    dtype, bitpix = _fits_float_dtype(double_precision)
    n_chan = img_xds.sizes["frequency"]
    n_pol = img_xds.sizes["polarization"]
    data_bytes = (
        img_xds.sizes["l"] * img_xds.sizes["m"] * n_pol * n_chan * dtype.itemsize
    )
    beams_extension_bytes = (
        _build_beams_extension_bytes(n_chan, n_pol) if beam_variables else None
    )

    paths = {}
    for key in image_variables:
        bunit, btype = _VARIABLE_FITS_UNITS.get(key, _DEFAULT_FITS_UNITS)
        has_beams = (
            beams_extension_bytes is not None and key in _BEAM_CARRYING_VARIABLES
        )
        header = _build_primary_header(
            img_xds, bitpix, bunit, btype, telescope_name, has_beams
        )
        header_bytes = header.tostring().encode("ascii")
        path = os.path.join(image_store, key.upper() + ".fits")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, header_bytes)
            data_end = len(header_bytes) + _pad_to_block(data_bytes)
            os.ftruncate(fd, data_end)  # sparse data area (holes read as 0.0)
            if has_beams:
                os.pwrite(fd, beams_extension_bytes, data_end)
        finally:
            os.close(fd)
        paths[key] = path
    return paths


# ---------------------------------------------------------------------------
# Node-task side: direct chunk writes into the pre-created files
# ---------------------------------------------------------------------------
def _fits_file_structure(path):
    """Read one pre-created FITS file's offsets/dtype (headers only, no data).

    Returns ``{"data_start", "dtype", "shape" (freq, pol, m, l), "beams"}``
    where ``beams`` is ``None`` or ``{"data_start", "row_bytes", "n_pol"}``.
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdulist:
        primary = hdulist[0]
        header = primary.header
        if header["NAXIS"] != 4:
            raise ValueError(f"{path}: expected a 4-axis FITS image.")
        structure = {
            "data_start": primary.fileinfo()["datLoc"],
            "dtype": np.dtype(">f8" if header["BITPIX"] == -64 else ">f4"),
            "shape": tuple(header[f"NAXIS{i}"] for i in (4, 3, 2, 1)),
            "beams": None,
        }
        for hdu in hdulist[1:]:
            if hdu.name == "BEAMS":
                structure["beams"] = {
                    "data_start": hdu.fileinfo()["datLoc"],
                    "row_bytes": hdu.header["NAXIS1"],
                    "n_pol": hdu.header["NPOL"],
                }
    return structure


def _encode_image_chunk(variable_key, image_store, chan_start, img_xds):
    """Encode one image variable's chunk to ``(path, offset, blob)``.

    The xds plane order ``(frequency, polarization, l, m)`` is transposed to
    FITS order ``(frequency, polarization, m, l)`` (``NAXIS1`` = RA = ``l``
    varies fastest) and converted to the file's big-endian float type; because
    ``FREQ`` is the last FITS axis, the whole channel block is one contiguous
    byte range at ``data_start + chan_start * plane_bytes``.
    """
    name = variable_key.upper()
    path = os.path.join(image_store, name + ".fits")
    try:
        structure = _retry_transient_io(
            f"FITS structure of {path}", lambda: _fits_file_structure(path)
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"FITS file {path} not found; create_empty_fits_images must run "
            "in the driver before the FITS chunk writer."
        ) from exc
    values = np.asarray(img_xds[name].values)
    chunk = values[0].transpose(0, 1, 3, 2).astype(structure["dtype"])
    n_chan, n_pol, n_m, n_l = chunk.shape
    if (n_pol, n_m, n_l) != structure["shape"][1:]:
        raise ValueError(
            f"{name}: chunk plane shape {(n_pol, n_m, n_l)} does not match the "
            f"FITS file's {structure['shape'][1:]}."
        )
    plane_bytes = n_pol * n_m * n_l * structure["dtype"].itemsize
    offset = structure["data_start"] + chan_start * plane_bytes
    return path, offset, np.ascontiguousarray(chunk).tobytes(), structure


def _encode_beam_rows(beam_variable_key, chan_start, img_xds):
    """Encode this chunk's ``BEAMS`` rows (full rows, including CHAN/POL)."""
    values = np.asarray(img_xds[beam_variable_key.upper()].values)[0]  # (chan, pol, 3)
    rows = bytearray()
    for local_chan in range(values.shape[0]):
        for pol in range(values.shape[1]):
            major, minor, pa = (float(v) for v in values[local_chan, pol, :3])
            rows += _BEAMS_ROW_STRUCT.pack(
                major, minor, pa, chan_start + local_chan, pol
            )
    return bytes(rows)


def _pwrite_with_eio_retry(path, offset, blob):
    """``pwrite`` one blob at a fixed offset, retrying transient Lustre errnos
    with a FRESH file descriptor (same schedule and rationale as the sharded
    Zarr writer: after a client eviction the old descriptor stays poisoned, and
    ``close()`` is part of the retried unit because an eviction can surface a
    failed writeback as EIO at close)."""
    import errno
    import time

    import toolviper.utils.logger as logger

    last_exc = None
    n_attempts = 1 + len(_EIO_RETRY_BACKOFF_SECONDS)
    for attempt, delay in enumerate((0.0,) + _EIO_RETRY_BACKOFF_SECONDS):
        if delay:
            time.sleep(delay)
        try:
            fd = os.open(path, os.O_WRONLY)  # pre-created; no O_CREAT
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"FITS file {path} not found; create_empty_fits_images must run "
                "in the driver before the FITS chunk writer."
            ) from exc
        except OSError as exc:
            if exc.errno not in _TRANSIENT_IO_ERRNOS:
                raise
            last_exc = exc
            logger.warning(
                f"FITS writer: {errno.errorcode.get(exc.errno, exc.errno)} "
                f"opening {path} (attempt {attempt + 1}/{n_attempts}); retrying."
            )
            continue
        try:
            os.pwrite(fd, blob, offset)
        except OSError as exc:
            try:
                os.close(fd)
            except OSError:
                pass  # the write already failed; a close error adds nothing
            failure = exc
        else:
            try:
                os.close(fd)
            except OSError as exc:
                failure = exc
            else:
                if attempt:
                    logger.warning(
                        f"FITS writer: pwrite to {path} succeeded on attempt "
                        f"{attempt + 1} after transient I/O error."
                    )
                return
        if failure.errno not in _TRANSIENT_IO_ERRNOS:
            raise failure
        last_exc = failure
        logger.warning(
            f"FITS writer: {errno.errorcode.get(failure.errno, failure.errno)} "
            f"writing {len(blob)} B at offset {offset} of {path} (attempt "
            f"{attempt + 1}/{n_attempts}); reopening and retrying."
        )
    raise last_exc


def write_result_chunk_to_fits_skunk_works(
    image_store,
    image_data_variables_keep,
    task_coords,
    img_xds,
    processing_function_threads=1,
):
    """Write this task's frequency chunk directly into the pre-created FITS files.

    FITS counterpart of
    :func:`~astroviper.node_tasks.imaging.utils.write_result_chunk_to_disk_using_zarr_skunk_works`,
    with the same two decoupled phases:

    1. **Encode** each kept image variable's chunk (transpose to FITS axis
       order + convert to the file's big-endian float type) concurrently
       across up to ``processing_function_threads`` threads. There is no
       compression -- FITS image data is raw -- so this phase is much cheaper
       than the Zarr encode.
    2. **Write** the blobs serially, one ``pwrite`` per file at this chunk's
       fixed offset (channel blocks are contiguous because ``FREQ`` is the last
       FITS axis). Concurrent tasks touch disjoint byte ranges of the shared,
       pre-created files: no locking, no file creation.

    A kept beam-fit-params variable is written as full ``BEAMS``-table rows
    (values plus their ``CHAN``/``POL`` indices) into every kept image file
    that carries a ``BEAMS`` extension.

    Parameters
    ----------
    image_store : str
        Directory of pre-created FITS files (:func:`create_empty_fits_images`).
    image_data_variables_keep : list of str
        Logical image-variable keys to write (e.g. ``"sky_residual"``); each is
        upper-cased to ``<VARIABLE>.fits``.
    task_coords : dict
        Per-chunk coordinate mapping; ``task_coords["frequency"]["slice"]``
        gives the global channel range this task owns.
    img_xds : xarray.Dataset
        The computed image holding this task's chunk for each variable.
    processing_function_threads : int, optional
        Maximum number of threads used to encode the variables concurrently
        (the ``pwrite``\\ s are always serial). Default ``1``.
    """
    frequency_slice = task_coords["frequency"].get("slice")
    chan_start = (frequency_slice.start or 0) if frequency_slice is not None else 0

    beam_variables = [
        dv for dv in image_data_variables_keep if dv.startswith("beam_fit_params")
    ]
    image_variables = [
        dv for dv in image_data_variables_keep if not dv.startswith("beam_fit_params")
    ]
    # The clean beam (PSF fit) labels the BEAMS tables when several are kept.
    beam_variable = (
        "beam_fit_params_point_spread_function"
        if "beam_fit_params_point_spread_function" in beam_variables
        else (beam_variables[0] if beam_variables else None)
    )

    # Phase 1: encode each image variable's chunk (optionally concurrent).
    if processing_function_threads <= 1 or len(image_variables) <= 1:
        encoded = [
            _encode_image_chunk(dv, image_store, chan_start, img_xds)
            for dv in image_variables
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(processing_function_threads, len(image_variables))
        ) as executor:
            futures = [
                executor.submit(
                    _encode_image_chunk, dv, image_store, chan_start, img_xds
                )
                for dv in image_variables
            ]
            encoded = [f.result() for f in futures]  # order matches; re-raises

    writes = [(path, offset, blob) for path, offset, blob, _structure in encoded]

    # Beam rows go into every kept image file that carries a BEAMS extension.
    if beam_variable is not None:
        for path, _offset, _blob, structure in encoded:
            beams = structure["beams"]
            if beams is None:
                continue
            if beams["row_bytes"] != _BEAMS_ROW_STRUCT.size:
                raise ValueError(
                    f"{path}: BEAMS rows are {beams['row_bytes']} B; expected "
                    f"{_BEAMS_ROW_STRUCT.size} (BMAJ,BMIN,BPA,CHAN,POL)."
                )
            rows = _encode_beam_rows(beam_variable, chan_start, img_xds)
            offset = (
                beams["data_start"] + chan_start * beams["n_pol"] * beams["row_bytes"]
            )
            writes.append((path, offset, rows))

    # Phase 2: write the blobs serially (one open file at a time for this task).
    for path, offset, blob in writes:
        _pwrite_with_eio_retry(path, offset, blob)
