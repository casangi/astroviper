"""FITS output of the chunk I/O engine (``io_engine="chunk"``) for cube imaging.

FITS counterpart of the direct-Zarr writers in
:mod:`astroviper.utils.chunk_io.zarr_chunks`, following the same
two-phase pattern the Zarr image store uses:

1. The **driver** (``distributed_applications`` layer) calls
   :func:`create_empty_fits_images` once: for every kept image variable it
   creates ``<image_store>/<VARIABLE>.fits`` with the complete FITS header --
   generated from the empty image dataset so that XRADIO's FITS reader
   (:func:`xradio.image.open_image`) reconstructs the same coordinates and
   attributes as the Zarr store -- and a **sparse** (``ftruncate``\\ d) data
   area sized for the full cube.
2. Each **node task** calls :func:`write_fits` to
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
  (``BMAJ, BMIN, BPA, CHAN, POL`` -- CASA's multi-beam convention, as
  casacore reads it: float32 axes and position angle in radians, int32
  indices). Rows are fixed-size and channel-major, so node tasks
  ``pwrite`` their channels' rows at disjoint offsets exactly like the image
  planes. casacore refuses a table with a zero or non-finite beam, so after
  the run :func:`fill_fits_beams_without_fit` gives the rows of planes without
  a fit the beam ``tclean`` gives them.
* The restored images of a run with a ``restoring_beam`` instead carry that
  one beam as ``BMAJ``, ``BMIN`` and ``BPA`` cards (degrees) of the primary
  header, as ``tclean`` writes a restored image, and no ``BEAMS`` table.
  Without one, the beams the restore applied fill the ``BEAMS`` table of the
  restored image, and the primary beam corrected image has no beam.

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

import numpy as np

# The FITS writers share the transient-I/O (Lustre eviction) retry machinery
# with the Zarr chunk writers.
from astroviper.utils.chunk_io.zarr_chunks import (
    _EIO_RETRY_BACKOFF_SECONDS,
    _TRANSIENT_IO_ERRNOS,
    _encode_and_write,
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
        "sky_restored_flat_noise",
        "sky_deconvolved",
        "point_spread_function",
    }
)
# With beam_fit_params_sky_restored kept, these carry a restoring beam in their
# header (cube and continuum restored images).
_RESTORED_VARIABLES = frozenset(
    {
        "sky_restored",
        "sky_restored_primary_beam_corrected",
        "sky_restored_flat_noise",
        "sky_restored_flat_sky",
    }
)

# Fixed BEAMS-table row layout written by create_empty_fits_images and assumed
# by the row writer, as casacore's ImageFITSConverter reads it:
# BMAJ, BMIN, BPA as big-endian float32 in the units below, then CHAN, POL as
# big-endian int32.
_BEAMS_ROW_DTYPE = np.dtype(
    [("BMAJ", ">f4"), ("BMIN", ">f4"), ("BPA", ">f4"), ("CHAN", ">i4"), ("POL", ">i4")]
)
_BEAMS_UNITS = ("rad", "rad", "rad")
# tclean's beam for every plane when no plane has a fit.
_NO_FIT_BEAM = tuple(np.radians([1e-6 / 3600.0, 1e-6 / 3600.0, 0.0]))

_FULL_DIMS_LM = ("time", "frequency", "polarization", "l", "m")
_BEAM_PARAMS_DIMS = ("time", "frequency", "polarization", "beam_params_label")


def _header_beam(image_variable, beam_variables, restoring_beam):
    """Whether the file of ``image_variable`` carries ``restoring_beam`` as its
    header beam: a restored image, when there is a ``restoring_beam`` and
    ``beam_fit_params_sky_restored`` is kept."""
    return (
        restoring_beam is not None
        and image_variable in _RESTORED_VARIABLES
        and "beam_fit_params_sky_restored" in beam_variables
    )


def _beams_table_variable(image_variable, beam_variables):
    """The beam-fit-params variable whose rows fill the ``BEAMS`` table of
    ``image_variable`` (``None`` for a file without one): for the restored
    image the beams the restore applied (with a restoring beam the file has
    its header beam instead, :func:`_header_beam`); else the clean beam (PSF
    fit), else the first other beam-fit-params variable kept."""
    if image_variable not in _BEAM_CARRYING_VARIABLES:
        return None
    if (
        image_variable in ("sky_restored", "sky_restored_flat_noise")
        and "beam_fit_params_sky_restored" in beam_variables
    ):
        return "beam_fit_params_sky_restored"
    others = [dv for dv in beam_variables if dv != "beam_fit_params_sky_restored"]
    if "beam_fit_params_point_spread_function" in others:
        return "beam_fit_params_point_spread_function"
    return others[0] if others else None


def _require_finite_positive(name, value):
    """Nonempty, finite, strictly positive real values, not coerced."""
    if hasattr(value, "unit"):
        raise TypeError(
            f"{name} must use plain numerical SI values, not a unit-bearing quantity."
        )
    values = np.asarray(value)
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers.")
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be nonempty and finite.")
    if np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return values


def check_fits_coordinates(image_params):
    """Require image coordinates exactly representable by the cube FITS writer.

    Parameters
    ----------
    image_params : mapping
        Supplied image coordinates. Only available keys are checked.
    """
    # --- Number of axes ---
    # The implemented four-axis FITS image has no independent time axis.
    if "time_coords" in image_params and np.size(image_params["time_coords"]) != 1:
        raise ValueError("FITS output supports a single time plane.")
    if "image_size" in image_params and np.any(
        np.asarray(image_params["image_size"]) < 2
    ):
        raise ValueError("FITS output needs at least 2 pixels per direction axis.")

    # --- Polarization axis ---
    # A linear FITS axis can encode consecutive Stokes/correlation codes only.
    if "polarization_coords" in image_params:
        codes_by_label = {
            "I": 1,
            "Q": 2,
            "U": 3,
            "V": 4,
            "RR": -1,
            "LL": -2,
            "RL": -3,
            "LR": -4,
            "XX": -5,
            "YY": -6,
            "XY": -7,
            "YX": -8,
        }
        labels = image_params["polarization_coords"]
        if (
            not isinstance(labels, list | tuple | np.ndarray)
            or np.ndim(labels) != 1
            or not len(labels)
        ):
            raise ValueError("FITS polarization_coords must be a nonempty vector.")
        if any(label not in codes_by_label for label in labels):
            raise ValueError("FITS polarization_coords contains an unknown label.")
        codes = [codes_by_label[label] for label in labels]
        if codes != list(range(codes[0], codes[0] + len(codes))):
            raise ValueError("FITS output requires consecutive polarization codes.")

    # --- Spectral axis ---
    # Do not substitute the first-to-last linear approximation for real centres.
    # Use the writer's tolerance (one millionth of a channel) for round-off.
    if "frequency_coords" in image_params:
        frequencies = _require_finite_positive(
            "FITS frequency_coords", image_params["frequency_coords"]
        )
        if frequencies.ndim != 1:
            raise ValueError("FITS frequency_coords must be one-dimensional.")
        if frequencies.size > 1:
            increment = (frequencies[-1] - frequencies[0]) / (frequencies.size - 1)
            if increment == 0:
                raise ValueError("FITS frequency coordinates have zero overall span.")
            linear = frequencies[0] + increment * np.arange(frequencies.size)
            if np.max(np.abs(frequencies - linear)) > 1e-6 * abs(increment):
                raise ValueError(
                    "FITS output requires uniformly spaced frequency_coords; use Zarr to preserve the channel centres."
                )

    # --- Direction frame ---
    # The writer/reader pair supports RA/DEC in these two reference frames.
    if "direction_reference" in image_params:
        frame = image_params["direction_reference"]
        if not isinstance(frame, str) or frame.upper() not in ("FK5", "ICRS"):
            raise ValueError("FITS direction_reference must be FK5 or ICRS.")


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

    check_fits_coordinates({"polarization_coords": pol_labels})
    label_to_code = {label: code for code, label in stokes_types.items()}
    return float(label_to_code[pol_labels[0]]), 1.0, 1.0


def _frequency_axis(freq_values):
    """FITS ``FREQ`` axis ``(crval, cdelt, crpix, max_deviation_hz)`` for the
    channel centers.

    A linear FITS axis must represent the supplied channel centres within the
    shared round-off tolerance. Nonuniform axes raise rather than replacing
    the frequencies with a first-to-last approximation.
    """

    check_fits_coordinates({"frequency_coords": freq_values})
    freq_values = np.asarray(freq_values, dtype=np.float64)
    if freq_values.size == 1:
        return float(freq_values[0]), 1.0, 1.0, 0.0
    cdelt = float((freq_values[-1] - freq_values[0]) / (freq_values.size - 1))
    return float(freq_values[0]), cdelt, 1.0, 0.0


def _direction_axis_values(coord_values):
    """``(cdelt_deg, crpix)`` of the ``l`` or ``m`` coordinate (linear, rad)."""
    values = np.asarray(coord_values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("FITS output needs at least 2 pixels per direction axis.")
    return float(np.degrees(values[1] - values[0])), int(np.argmin(np.abs(values))) + 1


def _build_primary_header(
    img_xds, bitpix, bunit, btype, telescope_name, has_beams_extension, beam=None
):
    """Primary-HDU header for one image variable, generated from the empty image.

    Every keyword XRADIO's FITS reader requires is written (direction axes +
    ``RADESYS``/``EQUINOX``/``LONPOLE``/``LATPOLE``/``PC``, spectral axis +
    ``RESTFRQ``/``SPECSYS``, ``TELESCOP``, ``DATE-OBS``/``TIMESYS``), and every
    keyword written is one the reader consumes or excludes from the user attrs,
    so nothing leaks into ``attrs["user"]`` on read. Nonuniform frequency
    grids are rejected by the shared coordinate check. ``beam``
    (``[major, minor, pa]`` in radians) is written as the ``BMAJ``, ``BMIN``
    and ``BPA`` cards in degrees, the one beam of every plane.
    """
    from astropy.io import fits
    from astropy.time import Time

    csys = img_xds.attrs["coordinate_system_info"]
    reference_direction = csys["reference_direction"]
    frame = reference_direction["attrs"]["frame"].upper()
    check_fits_coordinates(
        {"time_coords": img_xds.time.values, "direction_reference": frame}
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
    if beam is not None:
        for keyword, value in zip(("BMAJ", "BMIN", "BPA"), beam, strict=True):
            header[keyword] = float(np.degrees(value))
    return header


def _build_beams_extension_bytes(n_chan, n_pol):
    """Serialized ``BEAMS`` binary-table extension (header + rows).

    One fixed-size row per (channel, polarization), channel-major, in CASA's
    multi-beam column order ``BMAJ, BMIN, BPA, CHAN, POL`` (``_BEAMS_ROW_DTYPE``
    in ``_BEAMS_UNITS``). Beam values start as zero and are overwritten in
    place by the node tasks; the ``CHAN``/``POL`` index columns are correct from
    the start, also in rows no task writes.
    """
    from astropy.io import fits

    # Big-endian, as the file holds them (astropy's in-memory table is not).
    rows = np.zeros(n_chan * n_pol, dtype=_BEAMS_ROW_DTYPE)
    rows["CHAN"] = np.repeat(np.arange(n_chan), n_pol)
    rows["POL"] = np.tile(np.arange(n_pol), n_chan)
    columns = [
        fits.Column(name=name, format="E", unit=unit, array=rows[name])
        for name, unit in zip(("BMAJ", "BMIN", "BPA"), _BEAMS_UNITS, strict=True)
    ]
    columns += [
        fits.Column(name="CHAN", format="J", array=rows["CHAN"]),
        fits.Column(name="POL", format="J", array=rows["POL"]),
    ]
    hdu = fits.BinTableHDU.from_columns(columns, name="BEAMS")
    hdu.header["NCHAN"] = n_chan
    hdu.header["NPOL"] = n_pol
    if hdu.header["NAXIS1"] != _BEAMS_ROW_DTYPE.itemsize:
        raise AssertionError(
            f"BEAMS row is {hdu.header['NAXIS1']} bytes; the fixed-slot row "
            f"writer assumes {_BEAMS_ROW_DTYPE.itemsize} (BMAJ,BMIN,BPA,CHAN,POL)."
        )
    header_bytes = hdu.header.tostring().encode("ascii")
    data_bytes = rows.tobytes()
    return (
        header_bytes
        + data_bytes
        + b"\0" * (_pad_to_block(len(data_bytes)) - len(data_bytes))
    )


def _classify_kept_variables(image_data_variables_keep, data_variable_definitions):
    """Split the keep list into image variables and beam-fit-params variables.

    ``data_variable_definitions`` maps each key to its ``"dims"``; a variable
    with other dimensions (uv-domain or normalization variables) has no FITS
    representation and raises ``ValueError``.
    """
    image_variables, beam_variables = [], []
    for key in image_data_variables_keep:
        dims = tuple(data_variable_definitions[key]["dims"])
        if dims == _FULL_DIMS_LM:
            image_variables.append(key)
        elif dims == _BEAM_PARAMS_DIMS:
            beam_variables.append(key)
        else:
            raise ValueError(f"{key!r} has no FITS representation (dims {dims}).")
    return image_variables, beam_variables


def create_empty_fits_images(
    image_store,
    img_xds,
    image_data_variables_keep,
    double_precision=False,
    telescope_name="UNKNOWN",
    overwrite=False,
    *,
    data_variable_definitions,
    restoring_beam=None,
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
    * a ``BEAMS`` binary-table extension of zero beams (and ``CASAMBM = T``)
      on every variable a kept beam-fit-params variable labels
      (:func:`_beams_table_variable`), whose fixed-size rows the node tasks
      later overwrite in place and :func:`fill_fits_beams_without_fit`
      completes after the run;
    * with ``beam_fit_params_sky_restored`` kept, ``restoring_beam`` as the
      header beam of the restored images, which then have no ``BEAMS`` table
      (:func:`_header_beam`).

    The concurrent write phase then creates no files and writes only disjoint
    byte ranges (:func:`write_fits`).

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
        is upper-cased to the FITS file name.
    double_precision : bool, optional
        If ``True`` write 64-bit floats (``BITPIX`` -64), else 32-bit
        (``BITPIX`` -32). Default ``False``.
    telescope_name : str, optional
        Value of the ``TELESCOP`` keyword (required by XRADIO's FITS reader).
        Default ``"UNKNOWN"``.
    overwrite : bool, optional
        Replace an existing ``image_store``. Default ``False`` (raise
        ``FileExistsError``).
    data_variable_definitions : dict
        ``{key: {"dims": [...], ...}}`` of the kept variables; a variable
        without the image or beam-fit-params dimensions raises ``ValueError``.
    restoring_beam : sequence of float, optional
        The run's target beam ``[major, minor, pa]`` in radians; ``None``
        without one.

    Returns
    -------
    dict
        ``{variable_key: fits_path}`` of the created files.
    """
    image_variables, beam_variables = _classify_kept_variables(
        image_data_variables_keep, data_variable_definitions
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
        header_beam = _header_beam(key, beam_variables, restoring_beam)
        has_beams = (
            not header_beam and _beams_table_variable(key, beam_variables) is not None
        )
        header = _build_primary_header(
            img_xds,
            bitpix,
            bunit,
            btype,
            telescope_name,
            has_beams,
            beam=restoring_beam if header_beam else None,
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
    where ``beams`` is ``None`` or ``{"data_start", "row_bytes", "n_rows",
    "n_pol"}``.
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
                    "n_rows": hdu.header["NAXIS2"],
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
    values = np.asarray(img_xds[beam_variable_key.upper()].values)
    n_chan, n_pol = values.shape[1:3]
    beams = values[0, :, :, :3].reshape(-1, 3)
    rows = np.empty(n_chan * n_pol, dtype=_BEAMS_ROW_DTYPE)
    rows["BMAJ"] = beams[:, 0]
    rows["BMIN"] = beams[:, 1]
    rows["BPA"] = beams[:, 2]
    rows["CHAN"] = chan_start + np.repeat(np.arange(n_chan), n_pol)
    rows["POL"] = np.tile(np.arange(n_pol), n_chan)
    return rows.tobytes()


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


def write_fits(
    image_store,
    image_data_variables_keep,
    task_coords,
    img_xds,
    processing_function_threads=1,
):
    """Write this task's frequency chunk directly into the pre-created FITS files.

    FITS counterpart of
    :func:`~astroviper.utils.chunk_io.write_zarr`,
    with the same batches of ``processing_function_threads`` variables and two
    decoupled phases per batch:

    1. **Encode** each image variable's chunk of the batch (transpose to FITS
       axis order + convert to the file's big-endian float type) concurrently,
       one thread per variable. There is no
       compression -- FITS image data is raw -- so this phase is much cheaper
       than the Zarr encode.
    2. **Write** the blobs serially, one ``pwrite`` per file at this chunk's
       fixed offset (channel blocks are contiguous because ``FREQ`` is the last
       FITS axis). Concurrent tasks touch disjoint byte ranges of the shared,
       pre-created files: no locking, no file creation.

    Every kept image file that carries a ``BEAMS`` extension gets full table
    rows (values plus their ``CHAN``/``POL`` indices) of the beam-fit-params
    variable that :func:`_beams_table_variable` assigns to it, the clean beam
    (PSF fit) when it is kept, the restored beam for the restored image; the
    rows of planes without a fit are filled after the run
    (:func:`fill_fits_beams_without_fit`). A restoring beam is not written
    here: the restored images carry it in their header from the start.

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

    # Beam rows go into every kept image file that carries a BEAMS extension,
    # encoded once per beam variable.
    rows = {}

    def write(encoded):
        key, (path, offset, blob, structure) = encoded
        _pwrite_with_eio_retry(path, offset, blob)
        beams = structure["beams"]
        beam_variable = _beams_table_variable(key, beam_variables)
        if beams is None or beam_variable is None:
            return
        if beams["row_bytes"] != _BEAMS_ROW_DTYPE.itemsize:
            raise ValueError(
                f"{path}: BEAMS rows are {beams['row_bytes']} B; expected "
                f"{_BEAMS_ROW_DTYPE.itemsize} (BMAJ,BMIN,BPA,CHAN,POL)."
            )
        if beam_variable not in rows:
            rows[beam_variable] = _encode_beam_rows(beam_variable, chan_start, img_xds)
        _pwrite_with_eio_retry(
            path,
            beams["data_start"] + chan_start * beams["n_pol"] * beams["row_bytes"],
            rows[beam_variable],
        )

    _encode_and_write(
        image_variables,
        lambda dv: (dv, _encode_image_chunk(dv, image_store, chan_start, img_xds)),
        write,
        processing_function_threads,
    )


def write_fits_image(
    image_store,
    img_xds,
    image_data_variables_keep,
    *,
    data_variable_definitions,
    double_precision=False,
    telescope_name="UNKNOWN",
    overwrite=False,
    restoring_beam=None,
):
    """Write a whole image dataset to FITS files.

    :func:`create_empty_fits_images` followed by :func:`write_fits` of every
    channel and :func:`fill_fits_beams_without_fit`, for an image written once
    at the end of a run rather than by the node tasks.

    Parameters
    ----------
    image_store : str
        Output directory holding one ``<VARIABLE>.fits`` per image variable.
    img_xds : xarray.Dataset
        The image, with its coordinates, attributes and the kept variables.
    image_data_variables_keep : list of str
        Keys of the variables to write (beam-fit-params variables go into the
        ``BEAMS`` tables).
    data_variable_definitions : dict
        ``{key: {"dims": [...], ...}}`` of the kept variables.
    double_precision : bool, optional
        64-bit instead of 32-bit floats.
    telescope_name : str, optional
        ``TELESCOP`` keyword.
    overwrite : bool, optional
        Replace an existing ``image_store``.
    restoring_beam : sequence of float, optional
        The run's target beam ``[major, minor, pa]`` in radians; ``None``
        without one.

    Returns
    -------
    dict
        ``{key: path}`` of the FITS files written.
    """
    paths = create_empty_fits_images(
        image_store,
        img_xds,
        image_data_variables_keep,
        double_precision=double_precision,
        telescope_name=telescope_name,
        overwrite=overwrite,
        data_variable_definitions=data_variable_definitions,
        restoring_beam=restoring_beam,
    )
    write_fits(
        image_store,
        image_data_variables_keep,
        {"frequency": {"slice": slice(0, img_xds.sizes["frequency"])}},
        img_xds,
    )
    fill_fits_beams_without_fit(image_store, image_data_variables_keep)
    return paths


def fill_fits_beams_without_fit(image_store, image_data_variables_keep):
    """Give the ``BEAMS`` rows of planes without a fit the beam ``tclean`` gives them.

    Run ONCE in the driver, after every node task has written its rows.
    casacore refuses a ``BEAMS`` table that holds a zero or non-finite beam,
    so the rows of planes without a fit (a flagged channel, a failed fit, a
    skipped task) take the beam ``tclean`` gives such planes
    (``SIImageStore::makeImageBeamSet``): the first beam of largest area, the
    planes taken channel by channel as ``tclean`` fits them, or a circle of
    1e-6 arcsec at 0 degrees when no plane has a fit. A plane has a fit when
    its beam is finite and both axes are positive.

    Parameters
    ----------
    image_store : str
        Directory of FITS files written by
        :func:`write_fits`.
    image_data_variables_keep : list of str
        The run's kept variables; the ``BEAMS`` tables of the files of its image
        variables are filled.

    Returns
    -------
    dict
        ``{image variable key: numpy.ndarray}``: for the file of every image
        variable with a ``BEAMS`` table, which of its planes ``(frequency,
        polarization)`` had no fit.
    """
    planes_without_fit = {}
    for key in image_data_variables_keep:
        if not key.startswith("beam_fit_params"):
            without_fit = _fill_beams_table_without_fit(
                os.path.join(image_store, key.upper() + ".fits")
            )
            if without_fit is not None:
                planes_without_fit[key] = without_fit
    return planes_without_fit


def _fill_beams_table_without_fit(path):
    """:func:`fill_fits_beams_without_fit` for the file ``path``: which of its
    planes had no fit, ``None`` for a file without a ``BEAMS`` table."""
    import toolviper.utils.logger as logger

    beams = _retry_transient_io(
        f"FITS structure of {path}", lambda: _fits_file_structure(path)
    )["beams"]
    if beams is None:
        return None

    def read_rows():
        with open(path, "rb") as file:
            file.seek(beams["data_start"])
            return np.frombuffer(
                file.read(beams["n_rows"] * _BEAMS_ROW_DTYPE.itemsize),
                dtype=_BEAMS_ROW_DTYPE,
            ).copy()

    rows = _retry_transient_io(f"BEAMS rows of {path}", read_rows)
    values = np.stack([rows[name] for name in ("BMAJ", "BMIN", "BPA")], axis=-1)
    fitted = np.isfinite(values).all(axis=1) & (values[:, 0] > 0) & (values[:, 1] > 0)
    without_fit = (~fitted).reshape(-1, beams["n_pol"])
    if fitted.all():
        return without_fit
    if fitted.any():
        area = np.full(len(rows), -np.inf)
        area[fitted] = values[fitted, 0].astype(np.float64) * values[fitted, 1]
        fill = values[np.argmax(area)]
    else:
        fill = _NO_FIT_BEAM
    for name, value in zip(("BMAJ", "BMIN", "BPA"), fill, strict=True):
        rows[name][~fitted] = value
    logger.debug(
        f"{path}: {int((~fitted).sum())} BEAMS rows without a fit take "
        f"{[float(value) for value in fill]} ({', '.join(_BEAMS_UNITS)})."
    )
    _pwrite_with_eio_retry(path, beams["data_start"], rows.tobytes())
    return without_fit
