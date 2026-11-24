"""Unit tests for the FITS output of the chunk I/O engine.

Creates empty XRADIO-conformant FITS files from a tiny empty image dataset,
writes frequency chunks into them the way independent node tasks would
(disjoint ``pwrite`` ranges, including the BEAMS-table rows), and checks that
the files round-trip both through plain astropy and through XRADIO's FITS
reader (:func:`xradio.image.open_image`) -- coordinates, data, beams, and
pointing metadata. Also covers the unsupported-variable / axis-shape error
paths and the sparse (unwritten-channel) behavior.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from astroviper.utils.chunk_io import (
    create_empty_fits_images,
    write_fits,
)

IMAGE_SIZE = [16, 12]  # (l, m) -> FITS (NAXIS1, NAXIS2)
CELL = np.array([-1.0, 1.0]) * 4.8e-7
PHASE_CENTER = [2.888, -0.606]
FREQS = 3.4e11 + 1.2e6 * np.arange(5)
POLS = ["I", "Q"]
FULL_DIMS = ["time", "frequency", "polarization", "l", "m"]
BEAM_DIMS = ["time", "frequency", "polarization", "beam_params_label"]
KEEP = [
    "sky_residual",
    "mask",
    "point_spread_function",
    "beam_fit_params_point_spread_function",
]


def _create_empty_fits_images(*args, **kwargs):
    from astroviper.utils.io import imaging_data_variables_and_dims_single_precision

    return create_empty_fits_images(
        *args,
        data_variable_definitions=imaging_data_variables_and_dims_single_precision,
        **kwargs,
    )


def _empty_image(frequency_coords=FREQS, pol_coords=POLS, time_coords=(60000.0,)):
    from xradio.image import make_empty_sky_image

    return make_empty_sky_image(
        phase_center=PHASE_CENTER,
        image_size=IMAGE_SIZE,
        cell_size=CELL,
        frequency_coords=np.asarray(frequency_coords),
        pol_coords=list(pol_coords),
        time_coords=list(time_coords),
        do_sky_coords=False,
    )


def _full_arrays(seed=7, n_chan=None):
    if n_chan is None:
        n_chan = len(FREQS)
    rng = np.random.default_rng(seed)
    shape = (1, n_chan, len(POLS), IMAGE_SIZE[0], IMAGE_SIZE[1])
    return {
        "SKY_RESIDUAL": rng.standard_normal(shape).astype("f4"),
        "MASK": rng.random(shape) > 0.5,
        "POINT_SPREAD_FUNCTION": rng.standard_normal(shape).astype("f4"),
        "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": rng.random((1, n_chan, len(POLS), 3)),
    }


def _casa_rows(beams):
    """Beams ``[major, minor, pa]`` in radians as a ``BEAMS`` table holds them:
    float32 radians."""
    return np.asarray(beams, dtype=np.float32)


def _write_chunk(store, full, chan_slice, threads=1, keep=KEEP):
    """Write one simulated node task's channel chunk into the FITS store."""
    chunk_xds = _empty_image(frequency_coords=FREQS[chan_slice])
    for name, array in full.items():
        dims = FULL_DIMS if array.ndim == 5 else BEAM_DIMS
        chunk_xds[name] = (dims, array[:, chan_slice])
    task_coords = {"frequency": {"data": FREQS[chan_slice], "slice": chan_slice}}
    write_fits(store, keep, task_coords, chunk_xds, processing_function_threads=threads)


@pytest.fixture
def written_store(tmp_path):
    """Empty FITS store + full cube written as two independent chunks."""
    store = str(tmp_path / "img.fits")
    img_xds = _empty_image()
    _create_empty_fits_images(store, img_xds, KEEP, telescope_name="ALMA")
    full = _full_arrays()
    _write_chunk(store, full, slice(0, 2), threads=2)
    _write_chunk(store, full, slice(2, 5), threads=1)
    return store, img_xds, full


# --------------------------------------------------------------------------- #
# astropy-level round trip
# --------------------------------------------------------------------------- #
def test_astropy_round_trip(written_store):
    from astropy.io import fits

    store, _img_xds, full = written_store
    with fits.open(os.path.join(store, "SKY_RESIDUAL.fits")) as hdulist:
        header = hdulist[0].header
        assert header["BITPIX"] == -32
        assert [header[f"CTYPE{i}"] for i in (1, 2, 3, 4)] == [
            "RA---SIN",
            "DEC--SIN",
            "STOKES",
            "FREQ",
        ]
        # FITS stores (freq, pol, m, l); the xds planes are (freq, pol, l, m).
        expected = full["SKY_RESIDUAL"][0].transpose(0, 1, 3, 2)
        assert np.array_equal(hdulist[0].data, expected)
        # BEAMS rows are channel-major with prefilled CHAN/POL indices.
        beams = hdulist["BEAMS"].data
        assert len(beams) == len(FREQS) * len(POLS)
        assert np.array_equal(beams["CHAN"][:4], [0, 0, 1, 1])
        assert np.array_equal(beams["POL"][:4], [0, 1, 0, 1])
        expected = _casa_rows(full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"][0])
        for param, column in enumerate(("BMAJ", "BMIN", "BPA")):
            np.testing.assert_array_equal(
                beams[column].reshape(len(FREQS), len(POLS)), expected[..., param]
            )


def test_beams_tables_have_the_types_and_units_casacore_reads(written_store):
    """casacore's ImageFITSConverter::readBeamsTable reads BMAJ, BMIN and BPA
    only as float32 ('E') with angular TUNITs and CHAN, POL as int32 ('J'),
    with NCHAN and NPOL keywords; the beams are in radians, the unit of the
    beam-fit-params variables."""
    from astropy.io import fits

    store, _img_xds, _full = written_store
    for key in ("SKY_RESIDUAL", "POINT_SPREAD_FUNCTION"):
        with fits.open(os.path.join(store, key + ".fits")) as hdulist:
            table = hdulist["BEAMS"]
            columns = table.columns
            assert columns.names == ["BMAJ", "BMIN", "BPA", "CHAN", "POL"]
            assert columns.formats == list("EEEJJ")
            assert columns.units[:3] == ["rad", "rad", "rad"]
            assert table.header["NAXIS1"] == 20
            assert (table.header["NCHAN"], table.header["NPOL"]) == (5, 2)


def test_mask_written_as_floats_and_no_beams(written_store):
    from astropy.io import fits

    store, _img_xds, full = written_store
    with fits.open(os.path.join(store, "MASK.fits")) as hdulist:
        assert len(hdulist) == 1  # mask carries no BEAMS extension
        assert "CASAMBM" not in hdulist[0].header
        expected = full["MASK"][0].transpose(0, 1, 3, 2).astype("f4")
        assert np.array_equal(hdulist[0].data, expected)
    with fits.open(os.path.join(store, "POINT_SPREAD_FUNCTION.fits")) as hdulist:
        assert hdulist[0].header["CASAMBM"] is True
        assert hdulist["BEAMS"].header["NCHAN"] == len(FREQS)


def test_unwritten_channels_read_as_zero(tmp_path):
    """A sparse file's unwritten planes are holes -> 0.0 (not NaN)."""
    from astropy.io import fits

    store = str(tmp_path / "img.fits")
    _create_empty_fits_images(store, _empty_image(), KEEP)
    full = _full_arrays()
    _write_chunk(store, full, slice(2, 5))  # channels 0-1 never written
    with fits.open(os.path.join(store, "SKY_RESIDUAL.fits")) as hdulist:
        data = hdulist[0].data
    assert np.all(data[:2] == 0.0)
    assert np.array_equal(data[2:], full["SKY_RESIDUAL"][0, 2:].transpose(0, 1, 3, 2))


def test_double_precision(tmp_path):
    from astropy.io import fits

    store = str(tmp_path / "img.fits")
    _create_empty_fits_images(
        store, _empty_image(), ["sky_residual"], double_precision=True
    )
    full = {"SKY_RESIDUAL": _full_arrays()["SKY_RESIDUAL"].astype("f8")}
    _write_chunk(store, full, slice(0, 5), keep=["sky_residual"])
    with fits.open(os.path.join(store, "SKY_RESIDUAL.fits")) as hdulist:
        assert hdulist[0].header["BITPIX"] == -64
        assert hdulist[0].data.dtype.itemsize == 8
        assert np.array_equal(
            hdulist[0].data, full["SKY_RESIDUAL"][0].transpose(0, 1, 3, 2)
        )


# --------------------------------------------------------------------------- #
# The beam of each file
# --------------------------------------------------------------------------- #
RESTORED = ["sky_restored", "sky_primary_beam_corrected_restored"]
TARGET = np.array([3.0e-5, 2.0e-5, 0.5])


def _beams_table(store, key):
    """The ``BEAMS`` rows of ``<KEY>.fits`` as (channel, polarization, 3), as
    stored (:func:`_casa_rows`), or None."""
    from astropy.io import fits

    with fits.open(os.path.join(store, key.upper() + ".fits")) as hdulist:
        if "BEAMS" not in hdulist:
            return None
        beams = hdulist["BEAMS"].data
        rows = np.stack([beams[column] for column in ("BMAJ", "BMIN", "BPA")], -1)
    return rows.reshape(len(FREQS), len(POLS), 3)


def _restored_arrays(keep):
    full = _full_arrays()
    full["SKY_RESTORED"] = full["SKY_RESIDUAL"] + 1
    full["SKY_PRIMARY_BEAM_CORRECTED_RESTORED"] = full["SKY_RESIDUAL"] + 2
    full["BEAM_FIT_PARAMS_SKY_RESTORED"] = np.broadcast_to(
        TARGET, full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].shape
    ).copy()
    return {name: array for name, array in full.items() if name.lower() in keep}


def _header_beam(store, key):
    """``BMAJ``, ``BMIN``, ``BPA`` (degrees) of ``<KEY>.fits``, or None."""
    from astropy.io import fits

    header = fits.getheader(os.path.join(store, key.upper() + ".fits"))
    if "BMAJ" not in header:
        return None
    return np.array([header[keyword] for keyword in ("BMAJ", "BMIN", "BPA")])


def test_without_a_restoring_beam_the_corrected_file_carries_its_beam(tmp_path):
    """Without a restoring beam the restored file carries the PSF beams and
    the corrected file none, as before restoring_beam existed."""
    from astropy.io import fits

    store = str(tmp_path / "img.fits")
    keep = KEEP + RESTORED
    _create_empty_fits_images(store, _empty_image(), keep)
    full = _restored_arrays(keep)
    _write_chunk(store, full, slice(0, 5), keep=keep)
    np.testing.assert_array_equal(
        _beams_table(store, "sky_restored"),
        _casa_rows(full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"][0]),
    )
    path = os.path.join(store, "SKY_PRIMARY_BEAM_CORRECTED_RESTORED.fits")
    with fits.open(path) as hdulist:
        assert len(hdulist) == 2
        assert "CASAMBM" in hdulist[0].header
        assert "BMAJ" not in hdulist[0].header
        np.testing.assert_array_equal(
            hdulist[0].data,
            full["SKY_PRIMARY_BEAM_CORRECTED_RESTORED"][0].transpose(0, 1, 3, 2),
        )


def test_restored_files_carry_the_restoring_beam(tmp_path, monkeypatch):
    """The restored files carry the restoring beam as their header beam, in
    degrees, as tclean writes it, with no BEAMS table; the others the PSF
    beams, encoded once per chunk."""
    import xradio.image
    from astropy.io import fits

    import astroviper.utils.chunk_io.fits_blocks as module

    encode_beam_rows = module._encode_beam_rows
    encoded = []

    def counting_encode_beam_rows(beam_variable_key, chan_start, img_xds):
        encoded.append(beam_variable_key)
        return encode_beam_rows(beam_variable_key, chan_start, img_xds)

    monkeypatch.setattr(module, "_encode_beam_rows", counting_encode_beam_rows)
    store = str(tmp_path / "img.fits")
    keep = KEEP + RESTORED + ["beam_fit_params_sky_restored"]
    _create_empty_fits_images(store, _empty_image(), keep, restoring_beam=TARGET)
    full = _restored_arrays(keep)
    _write_chunk(store, full, slice(0, 2), keep=keep)
    _write_chunk(store, full, slice(2, 5), keep=keep)
    assert encoded == 2 * ["beam_fit_params_point_spread_function"]
    for key in RESTORED:
        path = os.path.join(store, key.upper() + ".fits")
        with fits.open(path) as hdulist:
            assert len(hdulist) == 1
            assert "CASAMBM" not in hdulist[0].header
            np.testing.assert_array_equal(
                hdulist[0].data, full[key.upper()][0].transpose(0, 1, 3, 2)
            )
        np.testing.assert_allclose(
            _header_beam(store, key), np.degrees(TARGET), rtol=1e-15
        )
        xds = xradio.image.open_image({key: path})
        np.testing.assert_allclose(
            xds["BEAM_FIT_PARAMS_" + key.upper()].values,
            full["BEAM_FIT_PARAMS_SKY_RESTORED"],
            rtol=1e-14,
        )
    for key in ("sky_residual", "point_spread_function"):
        assert _header_beam(store, key) is None
        np.testing.assert_array_equal(
            _beams_table(store, key),
            _casa_rows(full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"][0]),
        )


def test_restoring_beam_labels_no_other_file(tmp_path):
    store = str(tmp_path / "img.fits")
    keep = ["sky_residual", "beam_fit_params_sky_restored", *RESTORED]
    _create_empty_fits_images(store, _empty_image(), keep, restoring_beam=TARGET)
    full = _restored_arrays(keep)
    _write_chunk(store, full, slice(0, 5), keep=keep)
    assert _beams_table(store, "sky_residual") is None
    assert _header_beam(store, "sky_residual") is None
    for key in RESTORED:
        assert _beams_table(store, key) is None
        np.testing.assert_allclose(
            _header_beam(store, key), np.degrees(TARGET), rtol=1e-15
        )


def test_without_a_restoring_beam_the_restored_file_carries_the_restored_beam(
    tmp_path,
):
    """Without a restoring beam the BEAMS table of the restored file holds the
    beams the restore applied, not the PSF beams, and it has no header beam;
    the corrected file has no beam. After the run the rows of the flagged
    channel hold the beam of largest area, channel 2's, which XRADIO reads."""
    import xradio.image
    from astropy.io import fits

    from astroviper.utils.chunk_io.fits_blocks import (
        fill_fits_beams_without_fit,
    )

    store = str(tmp_path / "img.fits")
    keep = KEEP + RESTORED + ["beam_fit_params_sky_restored"]
    _create_empty_fits_images(store, _empty_image(), keep)
    full = _restored_arrays(keep)
    restored = np.array(
        [
            [2.0, 1.0, 50.0],
            [0.0, 0.0, 0.0],  # flagged
            [3.0, 2.0, 30.0],
            [2.5, 1.5, 20.0],
            [2.2, 1.9, -40.0],
        ]
    )
    restored = np.radians(restored / [3600.0, 3600.0, 1.0])
    full["BEAM_FIT_PARAMS_SKY_RESTORED"] = np.repeat(
        restored[None, :, None], len(POLS), axis=2
    )
    _write_chunk(store, full, slice(0, 2), keep=keep)
    _write_chunk(store, full, slice(2, 5), keep=keep)
    np.testing.assert_array_equal(
        _beams_table(store, "sky_restored"),
        _casa_rows(full["BEAM_FIT_PARAMS_SKY_RESTORED"][0]),
    )
    assert _header_beam(store, "sky_restored") is None
    for key in ("sky_residual", "point_spread_function"):
        np.testing.assert_array_equal(
            _beams_table(store, key),
            _casa_rows(full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"][0]),
        )
    path = os.path.join(store, "SKY_PRIMARY_BEAM_CORRECTED_RESTORED.fits")
    with fits.open(path) as hdulist:
        assert len(hdulist) == 2
        assert "CASAMBM" in hdulist[0].header
        assert "BMAJ" not in hdulist[0].header

    fill_fits_beams_without_fit(store, keep)
    filled = full["BEAM_FIT_PARAMS_SKY_RESTORED"][0].copy()
    filled[1] = restored[2]
    np.testing.assert_array_equal(
        _beams_table(store, "sky_restored"), _casa_rows(filled)
    )
    xds = xradio.image.open_image(
        {"sky_restored": os.path.join(store, "SKY_RESTORED.fits")}
    )
    np.testing.assert_allclose(
        xds["BEAM_FIT_PARAMS_SKY_RESTORED"].values[0],
        filled,
        rtol=np.finfo(np.float32).eps,
    )


def test_unwritten_beams_rows_hold_their_channel_and_polarization(tmp_path):
    """Rows no task writes (a kept variable the run does not make, a skipped
    task) hold zero beams with their CHAN and POL, so the file still opens."""
    import xradio.image
    from astropy.io import fits

    store = str(tmp_path / "img.fits")
    _create_empty_fits_images(store, _empty_image(), KEEP)
    path = os.path.join(store, "SKY_RESIDUAL.fits")
    with fits.open(path) as hdulist:
        beams = hdulist["BEAMS"].data
        np.testing.assert_array_equal(beams["CHAN"], np.repeat(range(5), 2))
        np.testing.assert_array_equal(beams["POL"], np.tile(range(2), 5))
        assert not np.any([beams[column] for column in ("BMAJ", "BMIN", "BPA")])
    xds = xradio.image.open_image({"sky_residual": path})
    assert not xds["BEAM_FIT_PARAMS_SKY_RESIDUAL"].values.any()


def test_rows_without_a_fit_take_the_beam_of_largest_area(tmp_path):
    """After the run the rows of planes without a fit, the zeros of a flagged
    channel and the NaN of a failed fit, which casacore refuses, hold the beam
    of largest area, channel 4's, as tclean fills them, not the first fit's or
    a neighbour's. The other rows, the indices and the files without a BEAMS
    table stay as they are; XRADIO reads the filled beams."""
    import xradio.image
    from astropy.io import fits

    from astroviper.utils.chunk_io.fits_blocks import (
        fill_fits_beams_without_fit,
    )

    store = str(tmp_path / "img.fits")
    keep = KEEP + RESTORED + ["beam_fit_params_sky_restored"]
    _create_empty_fits_images(store, _empty_image(), keep, restoring_beam=TARGET)
    full = _restored_arrays(keep)
    psf = np.array(
        [
            [2.0, 1.0, 50.0],
            [2.5, 1.5, 20.0],
            [0.0, 0.0, 0.0],  # flagged
            [2.2, 1.9, -40.0],
            [3.0, 2.0, 30.0],
        ]
    )
    psf = np.radians(psf / [3600.0, 3600.0, 1.0])
    psf = np.repeat(psf[None, :, None], len(POLS), axis=2)
    psf[0, 3, 1] = np.nan  # a failed fit
    full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"] = psf
    _write_chunk(store, full, slice(0, 2), keep=keep)
    _write_chunk(store, full, slice(2, 5), keep=keep)

    def file_bytes(key):
        with open(os.path.join(store, key.upper() + ".fits"), "rb") as file:
            return file.read()

    without_table = {key: file_bytes(key) for key in RESTORED + ["mask"]}

    fill_fits_beams_without_fit(store, keep)

    filled = psf[0].copy()
    filled[2, :] = psf[0, 4, 0]
    filled[3, 1] = psf[0, 4, 0]
    for key in ("sky_residual", "point_spread_function"):
        np.testing.assert_array_equal(_beams_table(store, key), _casa_rows(filled))
        path = os.path.join(store, key.upper() + ".fits")
        with fits.open(path) as hdulist:
            beams = hdulist["BEAMS"].data
            np.testing.assert_array_equal(beams["CHAN"], np.repeat(range(5), 2))
            np.testing.assert_array_equal(beams["POL"], np.tile(range(2), 5))
        xds = xradio.image.open_image({key: path})
        np.testing.assert_allclose(
            xds[f"BEAM_FIT_PARAMS_{key.upper()}"].values[0],
            filled,
            rtol=np.finfo(np.float32).eps,
        )
    for key, content in without_table.items():
        assert file_bytes(key) == content, key


def test_without_any_fit_rows_take_the_tclean_default_beam(tmp_path):
    """Without a fit in any plane (every channel flagged, or rows no task
    wrote) every row takes tclean's circle of 1e-6 arcsec at 0 degrees."""
    from astroviper.utils.chunk_io.fits_blocks import (
        fill_fits_beams_without_fit,
    )

    store = str(tmp_path / "img.fits")
    _create_empty_fits_images(store, _empty_image(), KEEP)
    fill_fits_beams_without_fit(store, KEEP)
    for key in ("sky_residual", "point_spread_function"):
        np.testing.assert_array_equal(
            _beams_table(store, key),
            np.broadcast_to(
                _casa_rows(np.radians([1e-6 / 3600, 1e-6 / 3600, 0.0])),
                (len(FREQS), len(POLS), 3),
            ),
        )


PSF_BEAM = "beam_fit_params_point_spread_function"
RESTORED_BEAM = "beam_fit_params_sky_restored"
RESIDUAL_BEAM = "beam_fit_params_sky_residual"


@pytest.mark.parametrize(
    "image_variable, beam_variables, expected",
    [
        ("sky_restored", [PSF_BEAM], PSF_BEAM),
        ("sky_restored", [RESTORED_BEAM, PSF_BEAM], RESTORED_BEAM),
        ("sky_primary_beam_corrected_restored", [PSF_BEAM], PSF_BEAM),
        (
            "sky_primary_beam_corrected_restored",
            [RESTORED_BEAM, PSF_BEAM],
            RESTORED_BEAM,
        ),
        ("sky_restored", [RESTORED_BEAM, PSF_BEAM], RESTORED_BEAM),
        (
            "sky_primary_beam_corrected_restored",
            [RESTORED_BEAM, PSF_BEAM],
            RESTORED_BEAM,
        ),
        ("sky_residual", [RESTORED_BEAM, PSF_BEAM], PSF_BEAM),
        ("sky_residual", [RESTORED_BEAM, RESIDUAL_BEAM], RESIDUAL_BEAM),
        ("sky_residual", [RESTORED_BEAM], None),
        ("mask", [RESTORED_BEAM, PSF_BEAM], None),
    ],
)
def test_beams_table_variable(image_variable, beam_variables, expected):
    from astroviper.utils.chunk_io.fits_blocks import (
        _beams_table_variable,
    )

    assert _beams_table_variable(image_variable, beam_variables) == expected


# --------------------------------------------------------------------------- #
# XRADIO-conformance round trip
# --------------------------------------------------------------------------- #
def test_xradio_open_image_round_trip(written_store):
    from xradio.image import open_image

    store, img_xds, full = written_store
    xds = open_image({"sky": os.path.join(store, "SKY_RESIDUAL.fits")})

    assert np.allclose(xds.frequency.values, FREQS)
    assert list(xds.polarization.values) == POLS
    assert np.allclose(xds.l.values, img_xds.l.values)
    assert np.allclose(xds.m.values, img_xds.m.values)
    assert np.allclose(xds.time.values, img_xds.time.values)

    sky = xds.SKY.isel(time=0).transpose("frequency", "polarization", "l", "m").values
    assert np.allclose(sky, full["SKY_RESIDUAL"][0])

    # The BEAMS table becomes the beam-fit-params variable, the radians it holds.
    np.testing.assert_array_equal(
        xds["BEAM_FIT_PARAMS_SKY"].values,
        _casa_rows(full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]),
    )

    assert xds.SKY.attrs["telescope"]["name"] == "ALMA"
    assert np.allclose(xds.SKY.attrs["pointing_center"]["data"], PHASE_CENTER)
    reference = xds.attrs["coordinate_system_info"]["reference_direction"]
    assert np.allclose(reference["data"], PHASE_CENTER)
    assert reference["attrs"]["frame"].lower() == "fk5"
    assert np.isclose(
        xds.frequency.attrs["rest_frequency"]["data"],
        img_xds.frequency.attrs["rest_frequency"]["data"],
    )


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_unsupported_variable_raises(tmp_path):
    with pytest.raises(ValueError, match="no FITS representation"):
        _create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(),
            ["sky_residual", "visibility_normalization"],
        )


def test_non_consecutive_stokes_raises(tmp_path):
    with pytest.raises(ValueError, match="consecutive polarization codes"):
        _create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(pol_coords=["I", "V"]),
            ["sky_residual"],
        )


def test_non_uniform_frequency_rejected(tmp_path):
    """Direct FITS creation follows the preflight policy without approximating."""
    freqs = np.concatenate([FREQS, 2e7 + FREQS])
    with pytest.raises(ValueError, match="uniformly spaced frequency_coords"):
        _create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(frequency_coords=freqs),
            ["sky_residual"],
        )


def test_zero_span_frequency_raises(tmp_path):
    with pytest.raises(ValueError, match="zero overall span"):
        _create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(frequency_coords=[1e9, 1e9]),
            ["sky_residual"],
        )


def test_multiple_time_planes_raise(tmp_path):
    with pytest.raises(ValueError, match="single time plane"):
        _create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(time_coords=[60000.0, 60001.0]),
            ["sky_residual"],
        )


def test_overwrite_semantics(tmp_path):
    store = str(tmp_path / "img.fits")
    _create_empty_fits_images(store, _empty_image(), ["sky_residual"])
    with pytest.raises(FileExistsError):
        _create_empty_fits_images(store, _empty_image(), ["sky_residual"])
    paths = _create_empty_fits_images(
        store, _empty_image(), ["sky_residual"], overwrite=True
    )
    assert os.path.exists(paths["sky_residual"])


def test_writer_requires_precreated_files(tmp_path):
    store = str(tmp_path / "img.fits")
    os.makedirs(store)
    with pytest.raises(FileNotFoundError, match="create_empty_fits_images"):
        _write_chunk(store, _full_arrays(), slice(0, 2))
