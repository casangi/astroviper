"""Unit tests for the ``_skunk_works`` FITS output path.

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

from astroviper.node_tasks.imaging.utils import (
    create_empty_fits_images,
    write_result_chunk_to_fits_skunk_works,
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


def _write_chunk(store, full, chan_slice, threads=1, keep=KEEP):
    """Write one simulated node task's channel chunk into the FITS store."""
    chunk_xds = _empty_image(frequency_coords=FREQS[chan_slice])
    for name, array in full.items():
        dims = FULL_DIMS if array.ndim == 5 else BEAM_DIMS
        chunk_xds[name] = (dims, array[:, chan_slice])
    task_coords = {"frequency": {"data": FREQS[chan_slice], "slice": chan_slice}}
    write_result_chunk_to_fits_skunk_works(
        store, keep, task_coords, chunk_xds, processing_function_threads=threads
    )


@pytest.fixture
def written_store(tmp_path):
    """Empty FITS store + full cube written as two independent chunks."""
    store = str(tmp_path / "img.fits")
    img_xds = _empty_image()
    create_empty_fits_images(store, img_xds, KEEP, telescope_name="ALMA")
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
        for column, param in zip(("BMAJ", "BMIN", "BPA"), range(3), strict=False):
            assert np.allclose(
                beams[column].reshape(len(FREQS), len(POLS)),
                full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"][0, :, :, param],
            )


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
    create_empty_fits_images(store, _empty_image(), KEEP)
    full = _full_arrays()
    _write_chunk(store, full, slice(2, 5))  # channels 0-1 never written
    with fits.open(os.path.join(store, "SKY_RESIDUAL.fits")) as hdulist:
        data = hdulist[0].data
    assert np.all(data[:2] == 0.0)
    assert np.array_equal(data[2:], full["SKY_RESIDUAL"][0, 2:].transpose(0, 1, 3, 2))


def test_double_precision(tmp_path):
    from astropy.io import fits

    store = str(tmp_path / "img.fits")
    create_empty_fits_images(
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

    # The BEAMS table becomes the beam-fit-params variable (radians).
    assert np.allclose(
        xds["BEAM_FIT_PARAMS_SKY"].values,
        full["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"],
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
        create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(),
            ["sky_residual", "visibility_normalization"],
        )


def test_non_consecutive_stokes_raises(tmp_path):
    with pytest.raises(ValueError, match="non-consecutive"):
        create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(pol_coords=["I", "V"]),
            ["sky_residual"],
        )


def test_non_uniform_frequency_approximated(tmp_path):
    """A concatenated-SPW-style axis gets the uniform FREQ axis through the
    first/last channel centers plus a HISTORY note (a linear FITS axis cannot
    hold the true centers); the note stays out of XRADIO's user attrs."""
    from astropy.io import fits
    from xradio.image import open_image

    freqs = np.concatenate([FREQS, 2e7 + FREQS])  # two "SPWs" 20 MHz apart
    paths = create_empty_fits_images(
        str(tmp_path / "img.fits"),
        _empty_image(frequency_coords=freqs),
        ["sky_residual"],
    )
    header = fits.getheader(paths["sky_residual"])
    cdelt = (freqs[-1] - freqs[0]) / (freqs.size - 1)
    assert header["CRVAL4"] == freqs[0]
    assert np.isclose(header["CDELT4"], cdelt, rtol=1e-12)
    assert header["CRPIX4"] == 1.0
    assert any("non-uniform" in card for card in header["HISTORY"])
    xds = open_image({"sky": paths["sky_residual"]})
    assert np.allclose(xds.frequency.values, freqs[0] + cdelt * np.arange(freqs.size))
    assert "history" not in xds.attrs.get("user", {})


def test_zero_span_frequency_raises(tmp_path):
    with pytest.raises(ValueError, match="zero overall span"):
        create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(frequency_coords=[1e9, 1e9]),
            ["sky_residual"],
        )


def test_multiple_time_planes_raise(tmp_path):
    with pytest.raises(ValueError, match="single time plane"):
        create_empty_fits_images(
            str(tmp_path / "img.fits"),
            _empty_image(time_coords=[60000.0, 60001.0]),
            ["sky_residual"],
        )


def test_overwrite_semantics(tmp_path):
    store = str(tmp_path / "img.fits")
    create_empty_fits_images(store, _empty_image(), ["sky_residual"])
    with pytest.raises(FileExistsError):
        create_empty_fits_images(store, _empty_image(), ["sky_residual"])
    paths = create_empty_fits_images(
        store, _empty_image(), ["sky_residual"], overwrite=True
    )
    assert os.path.exists(paths["sky_residual"])


def test_writer_requires_precreated_files(tmp_path):
    store = str(tmp_path / "img.fits")
    os.makedirs(store)
    with pytest.raises(FileNotFoundError, match="create_empty_fits_images"):
        _write_chunk(store, _full_arrays(), slice(0, 2))
