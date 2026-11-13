"""Tests of the CASA tasks and tools that find_continuum gives findContinuum."""

import os

import numpy as np
import pytest
import xarray as xr
from xradio.image import write_image

from astroviper.distributed_applications.imaging.find_continuum import (
    casa_image_tasks as cit,
)
from astroviper.distributed_applications.imaging.find_continuum import findContinuum
from astroviper.distributed_applications.imaging.find_continuum.find_continuum import (
    _cube_metadata,
)
from astroviper.node_tasks.imaging.find_continuum import (
    read_small_image,
    write_small_image,
)
from astroviper.processing_functions.imaging.find_continuum import (
    finalize_find_continuum_reductions,
    find_continuum_cube_reductions,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    make_find_continuum_image_xds,
)

N_PIXELS = 12
N_CHANNELS = 10
METADATA = {
    "shape": [5, 4, 1, 1],
    "increment": [-1.0e-6, 1.0e-6, 1.0, 1.0e6],
    "reference_value": [1.2, -0.4, 1.0, 230.0e9],
    "reference_pixel": [2.0, 2.0, 0.0, 0.0],
    "rest_frequency": 230.0e9,
    "spectral_frame": "LSRK",
    "direction_frame": "J2000",
    "beams": [[5.0e-6, 2.5e-6, 0.5]],
    "brightness_unit": "Jy/beam",
    "telescope": "ALMA",
    "object": "M100",
    "epoch_mjd": None,
}


def _casa_order(img_xds, variable):
    return np.transpose(img_xds[variable].values[0], (2, 3, 1, 0)).astype(np.float64)


@pytest.fixture
def session(tmp_path):
    """Registered dirty cube and primary beam, reduced in one process."""
    line = dict(
        channel=5,
        fwhm_channels=2,
        peak_jy=0.05,
        offset_pixels=(0, 0),
        size_fwhm_pixels=3,
    )
    img_xds = make_find_continuum_image_xds(
        n_pixels=N_PIXELS, n_channels=N_CHANNELS, lines=(line,)
    )
    store = str(tmp_path / "cube.img.zarr")
    write_image(img_xds, imagename=store, out_format="zarr", overwrite=True)
    lazy_xds = xr.open_zarr(store, chunks=None)
    residual, pb = str(tmp_path / "cube.residual"), str(tmp_path / "cube.pb")
    beam = "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
    cubes = {
        residual: {
            "store": store,
            "variable": "SKY_RESIDUAL",
            "metadata": _cube_metadata(lazy_xds, "SKY_RESIDUAL", beam, "ALMA"),
            "atmosphere": {
                "transmission": np.full(N_CHANNELS, 0.9),
                "tsky": np.full(N_CHANNELS, 20.0),
            },
        },
        pb: {
            "store": store,
            "variable": "PRIMARY_BEAM",
            "metadata": _cube_metadata(lazy_xds, "SKY_RESIDUAL", None, "ALMA"),
        },
    }
    image = lazy_xds.sel(polarization=["I"]).load()
    variables = {path: cube["variable"] for path, cube in cubes.items()}

    def reduce(requests):
        images = {}
        for request in requests:
            for name in cit._mask_references(request["mask"])[0]:
                values, mask, _ = read_small_image(name)
                images[name] = np.where(mask, values, np.nan)
        partials = find_continuum_cube_reductions(
            image, requests, variables, images, np.arange(N_CHANNELS)
        )
        return finalize_find_continuum_reductions(partials, requests)

    with cit.cube_session(cubes, reduce):
        yield {
            "residual": residual,
            "pb": pb,
            "sky": _casa_order(img_xds, "SKY_RESIDUAL"),
            "pb_values": _casa_order(img_xds, "PRIMARY_BEAM"),
            "frequency": img_xds.frequency.values,
            "tmp_path": tmp_path,
        }


@pytest.fixture
def small(tmp_path):
    """A 5 x 4 small image with its first pixel masked."""
    values = np.arange(20.0).reshape(5, 4, 1, 1)
    valid = np.ones(values.shape, dtype=bool)
    valid[0, 0] = False
    path = str(tmp_path / "small.image")
    write_small_image(path, values, valid, METADATA)
    return path, values, valid


def test_parse_channel_selection():
    np.testing.assert_array_equal(
        cit.parse_channel_selection("5~7;9,11", 10), [5, 6, 7, 9]
    )
    np.testing.assert_array_equal(cit.parse_channel_selection(3, 10), [3])
    assert cit.parse_channel_selection("", 10) is None


class TestCubeTasks:
    @pytest.mark.filterwarnings("ignore:All-NaN slice:RuntimeWarning")
    def test_immoments(self, session):
        out = str(session["tmp_path"] / "mom")
        mask = f'"{session["pb"]}">0.5'
        cit.immoments(
            session["residual"], moments=[0, 8], chans="3~7", mask=mask, outfile=out
        )
        values, valid, metadata = read_small_image(out + ".maximum")
        pb_inside = session["pb_values"][..., 3:8] > 0.5
        inside = pb_inside.any(axis=3, keepdims=True)
        expected = np.nanmax(
            np.where(pb_inside, session["sky"][..., 3:8], np.nan), axis=3, keepdims=True
        )
        np.testing.assert_allclose(values, expected, equal_nan=True)
        np.testing.assert_array_equal(valid, inside)
        assert metadata["shape"] == [N_PIXELS, N_PIXELS, 1, 1]
        assert read_small_image(out + ".integrated")[1].sum() == inside.sum()

        cit.immoments(session["residual"], moments=[-1], outfile=out + ".mean")
        np.testing.assert_allclose(
            read_small_image(out + ".mean")[0][..., 0], session["sky"].mean(axis=3)
        )

    def test_imstat(self, session):
        sky = session["sky"]
        stats = cit.imstat(session["residual"], chans="2~4")
        assert stats["npts"][0] == sky[..., 2:5].size
        assert stats["max"][0] == pytest.approx(sky[..., 2:5].max())
        per_channel = cit.imstat(session["residual"], axes=[0, 1, 2])
        np.testing.assert_allclose(per_channel["max"], sky.max(axis=(0, 1, 2)))

        boxed = cit.imstat(session["residual"], box="2,3,5,6", chans="4")
        region = sky[2:6, 3:7, :, 4]
        assert boxed["npts"][0] == region.size
        i, j, _ = np.unravel_index(region.argmax(), region.shape)
        assert list(boxed["maxpos"]) == [2 + i, 3 + j, 0, 4]
        whole_box = cit.imstat(session["residual"], box="0,0,3,3", axes=[0, 1, 2])
        np.testing.assert_allclose(whole_box["max"], sky[:4, :4].max(axis=(0, 1, 2)))

    def test_imstat_box_below_primary_beam_limit(self, session):
        """A box below the primary beam limit has no valid pixel: CASA gives
        empty arrays, which findContinuum counts to find masked corners."""
        middle = N_CHANNELS // 2
        pb = session["pb_values"][..., 0, middle]
        half = N_PIXELS // 2
        corners = pb[[0, 0, -1, -1], [0, -1, 0, -1]]
        edges = pb[[0, -1, half, half], [half, half, 0, -1]]
        assert corners.max() < edges.min()
        limit = 0.5 * (corners.max() + edges.min())
        for path in (session["residual"], session["pb"]):
            cit._cube(path).update(
                primary_beam="PRIMARY_BEAM", primary_beam_limit=limit
            )
            os.makedirs(path)

        empty = cit.imstat(session["pb"], chans=str(middle), box="0,0,0,0")
        assert empty and all(value.size == 0 for value in empty.values())
        assert cit.imstat(session["pb"], chans=str(middle), box="0,6,0,6")["npts"] == 1
        assert findContinuum.cornersEmptyEdgesNotEmpty(
            session["pb"], middleChannel=middle
        )

    def test_image_tool(self, session):
        ia = cit.iatool()
        ia.open(session["residual"])
        profile = ia.getprofile(axis=3, function="mean", mask=f'"{session["pb"]}">0')
        np.testing.assert_allclose(
            profile["values"], session["sky"].mean(axis=(0, 1, 2))
        )
        np.testing.assert_array_equal(ia.shape(), [N_PIXELS, N_PIXELS, 1, N_CHANNELS])
        assert ia.brightnessunit() == "Jy/beam"
        beams = ia.restoringbeam()
        assert beams["nChannels"] == N_CHANNELS
        assert beams["beams"]["*0"]["*0"]["major"]["value"] == pytest.approx(1.0)
        assert ia.maskhandler("default") == [""]
        assert ia.statistics()["npts"][0] == session["sky"].size
        with pytest.raises(NotImplementedError):
            ia.getregion()

        coordsys = ia.coordsys()
        assert coordsys.names()[3] == "Frequency"
        assert coordsys.findaxisbyname("spectral") == 3
        assert coordsys.findaxisbyname("right") == 0
        assert coordsys.referencecode("spectral") == ["LSRK"]
        assert coordsys.referencecode("direction") == ["J2000"]
        assert coordsys.toworld([0, 0, 0, 2])["numeric"][3] == pytest.approx(
            session["frequency"][2]
        )
        assert coordsys.increment()["numeric"][3] == pytest.approx(
            np.diff(session["frequency"])[0]
        )
        assert coordsys.referencepixel()["numeric"][3] == 0.0
        assert coordsys.referencevalue()["numeric"][1] == pytest.approx(-0.4)
        assert coordsys.restfrequency()["unit"] == "Hz"
        assert coordsys.telescope() == "ALMA"
        assert coordsys.epoch()["m0"]["value"] is None
        assert coordsys.done()
        with pytest.raises(ValueError):
            coordsys.findaxisbyname("velocity")
        assert ia.close()

        ia.open(session["pb"])
        assert ia.restoringbeam() == {}
        ia.done()

    def test_subsets_of_a_cube(self, session):
        tmp_path = session["tmp_path"]
        twice = str(tmp_path / "twice")
        cit.immath(session["residual"], expr="IM0 * 2", chans="3~4", outfile=twice)
        values, _, metadata = read_small_image(twice)
        np.testing.assert_allclose(values, 2 * session["sky"][..., 3:5])
        assert metadata["shape"] == [N_PIXELS, N_PIXELS, 1, 2]

        plane = str(tmp_path / "pb1")
        cit.imsubimage(
            session["pb"], outfile=plane, chans="1", mask=f'"{session["pb"]}">0.5'
        )
        np.testing.assert_array_equal(
            read_small_image(plane)[1][..., 0], session["pb_values"][..., 1] > 0.5
        )
        assert cit.imhead(session["residual"], mode="get", hdkey="object") == ""
        np.testing.assert_array_equal(
            cit.atmosphere_spectrum(session["residual"], "tsky"), 20.0
        )
        with pytest.raises(ValueError, match="atmospheric"):
            cit.atmosphere_spectrum(session["pb"], "tsky")

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: cit.immath(
                s["residual"], expr="IM0", outfile=str(s["tmp_path"] / "x")
            ),
            lambda s: cit.imsubimage(s["residual"], outfile=str(s["tmp_path"] / "x")),
            lambda s: cit.imstat(s["residual"], algorithm="chauvenet", maxiter=5),
            lambda s: cit.imstat(
                s["residual"], chans="1", box="0,0,1,1", mask=f'"{s["pb"]}">0'
            ),
        ],
    )
    def test_whole_cube_in_memory_is_refused(self, session, call):
        with pytest.raises(NotImplementedError):
            call(session)


class TestSmallImageTasks:
    def test_imstat(self, small):
        path, values, valid = small
        stats = cit.imstat(path, mask=f'"{path}">2')
        expected = values[valid & (values > 2)]
        assert stats["npts"][0] == expected.size
        assert list(stats["minpos"]) == [0, 3, 0, 0]
        assert list(stats["maxpos"]) == [4, 3, 0, 0]
        chauvenet = cit.imstat(path, algorithm="chauvenet", maxiter=5)
        assert chauvenet["npts"][0] == 19
        empty = cit.imstat(path, mask=f'"{path}">100')
        assert empty["npts"].size == 0 and "maxpos" not in empty

    def test_immath_and_imsubimage(self, small, tmp_path):
        path, values, valid = small
        other = str(tmp_path / "other.image")
        write_small_image(
            other, np.ones_like(values), np.ones(values.shape, dtype=bool), METADATA
        )
        difference = str(tmp_path / "difference")
        cit.immath(
            [path, other], expr="IM0 - IM1", mask=f'"{path}">3', outfile=difference
        )
        result, result_valid, _ = read_small_image(difference)
        np.testing.assert_allclose(
            result[result_valid], (values - 1)[valid & (values > 3)]
        )

        subimage = str(tmp_path / "sub")
        cit.imsubimage(path, outfile=subimage, mask=f'"{path}"<10')
        np.testing.assert_array_equal(
            read_small_image(subimage)[1], valid & (values < 10)
        )
        with pytest.raises(FileExistsError):
            cit.imsubimage(path, outfile=subimage)
        cit.imsubimage(path, outfile=subimage, overwrite=True)

    def test_makemask(self, small, tmp_path):
        path, values, valid = small
        threshold = str(tmp_path / "threshold")
        write_small_image(
            threshold,
            (values > 15).astype(float),
            np.ones(values.shape, dtype=bool),
            METADATA,
        )
        output = str(tmp_path / "mask")
        cit.makemask(
            mode="copy",
            inpimage=path,
            inpmask=[path + ":mask0", threshold],
            output=output,
        )
        mask_values, mask_valid, _ = read_small_image(output)
        np.testing.assert_array_equal(mask_values, valid.astype(float))
        assert mask_valid.all()
        with pytest.raises(FileExistsError):
            cit.makemask(mode="copy", inpimage=path, inpmask=threshold, output=output)

    def test_imcollapse(self, tmp_path):
        values = np.arange(12.0).reshape(2, 2, 1, 3)
        path = str(tmp_path / "cube3")
        write_small_image(
            path,
            values,
            np.ones(values.shape, dtype=bool),
            dict(METADATA, shape=[2, 2, 1, 3]),
        )
        for function, expected in (
            ("max", values.max(3)),
            ("min", values.min(3)),
            ("mean", values.mean(3)),
        ):
            out = str(tmp_path / function)
            cit.imcollapse(path, outfile=out, function=function, axes=3)
            collapsed, _, metadata = read_small_image(out)
            np.testing.assert_allclose(collapsed[..., 0], expected)
            assert metadata["shape"] == [2, 2, 1, 1]
        with pytest.raises(FileExistsError):
            cit.imcollapse(path, outfile=str(tmp_path / "max"), function="max")

    def test_image_tool(self, small):
        path, values, valid = small
        ia = cit.iatool()
        ia.open(path + "/")
        np.testing.assert_array_equal(ia.getregion(), values)
        np.testing.assert_array_equal(ia.getregion(getmask=True), valid)
        assert ia.maskhandler("default") == ["mask0"]
        assert ia.restoringbeam()["positionangle"]["value"] == pytest.approx(
            np.degrees(0.5)
        )
        assert ia.coordsys().referencecode("direction") == ["J2000"]
        assert cit.imhead(path, mode="get", hdkey="object") == "M100"
        with pytest.raises(FileNotFoundError):
            ia.open(path + ".missing")

    def test_pruneregions(self, tmp_path):
        mask = np.zeros((8, 8, 1, 1))
        mask[0, 0] = 1.0
        mask[3:6, 3:6] = 1.0
        path = str(tmp_path / "mask")
        write_small_image(
            path,
            mask,
            np.ones(mask.shape, dtype=bool),
            dict(METADATA, shape=[8, 8, 1, 1]),
        )
        result = cit.synthesismaskhandler().pruneregions(path, prunesize=3.0)
        assert list(result["N_reg"]) == [2] and list(result["N_reg_pruned"]) == [1]
        pruned = read_small_image(path + ".pruned")[0]
        assert pruned[0, 0, 0, 0] == 0.0 and pruned[3:6, 3:6].all()


def test_quanta_and_log():
    qa = cit.qatool()
    assert qa.convert({"value": 1.0, "unit": "arcmin"}, "arcsec")[
        "value"
    ] == pytest.approx(60.0)
    assert qa.convert("2deg", "arcmin")["value"] == pytest.approx(120.0)
    assert qa.quantity("3GHz") == {"unit": "GHz", "value": 3.0}
    assert qa.quantity(5, "Hz") == {"unit": "Hz", "value": 5}
    assert qa.done()
    cit.casalog.post("a warning", "WARN")
    cit.casalog.post("a message")
    cit.casalog.origin("findContinuum")


@pytest.mark.parametrize(
    "call",
    [
        lambda p: cit.imregrid(p),
        lambda p: cit.imsmooth(p),
        lambda p: cit.attool(),
        lambda p: cit.mstool(),
        lambda p: cit.immoments(p, moments=[0], outfile="x", box="1,1,2,2"),
        lambda p: cit.immoments(p, moments=[0], axis=0, outfile="x"),
        lambda p: cit.immoments(p, moments=[1], outfile="x"),
        lambda p: cit.immoments(p, moments=[0], outfile="x"),
        lambda p: cit.imstat(p, region="r"),
        lambda p: cit.imstat(p, axes=[0, 1]),
        lambda p: cit.immath(p, mode="spix"),
        lambda p: cit.imsubimage(p, box="1,1,2,2"),
        lambda p: cit.makemask(mode="expand"),
        lambda p: cit.makemask(mode="copy", inpimage=p, inpmask=p, output=p + ":mask1"),
        lambda p: cit.imhead(p, mode="summary"),
        lambda p: cit.imcollapse(p, function="median"),
    ],
)
def test_unsupported_calls(small, call):
    with pytest.raises(NotImplementedError):
        call(small[0])


@pytest.mark.parametrize(
    "call",
    [
        lambda ia: ia.maskhandler("set"),
        lambda ia: ia.getregion(region="r"),
        lambda ia: ia.getprofile(axis=2),
        lambda ia: ia.getprofile(axis=3),
        lambda ia: ia.statistics(robust=True),
    ],
)
def test_unsupported_image_tool_calls(small, call):
    ia = cit.iatool()
    ia.open(small[0])
    with pytest.raises(NotImplementedError):
        call(ia)
