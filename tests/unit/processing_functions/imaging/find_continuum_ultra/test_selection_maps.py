"""Tests of find_continuum_ultra's selection maps and their statistics against
the vendored findContinuum and the simple port."""

import random

import numpy as np
import pytest
import xarray as xr
from xradio.image import write_image

from astroviper.distributed_applications.imaging import find_continuum
from astroviper.node_tasks.imaging.find_continuum import (
    read_small_image,
    write_small_image,
)
from astroviper.processing_functions.image_analysis.moments import (
    collapsed_moment_axis_coords,
)
from astroviper.processing_functions.imaging.find_continuum import (
    evaluate_lattice_expression,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    channels_of_ranges,
    round_to_decimals,
    round_to_significant_digits,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.selection_maps import (
    count_above_level,
    difference_map_snr,
    image_snr,
    image_snr_outside,
    lowered_moment_difference_level,
    map_statistics,
    moment_difference_counts,
    negative_pixel_fraction,
    original_round_cube_statistics,
    round_zero_regions,
    selection_maps,
    selection_statistics,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    CASA_RESULTS,
    make_find_continuum_image_xds,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    findContinuum,
    quiet,
    random_ranges,
    vendored_block,
)

MAP_DIMS = ("time", "frequency", "polarization", "l", "m")
# The metadata of the port's small images.
METADATA = {
    "shape": [1, 1, 1, 1],
    "increment": [-1.0e-6, 1.0e-6, 1.0, 1.0e6],
    "reference_value": [1.2, -0.4, 1.0, 230.0e9],
    "reference_pixel": [0.0, 0.0, 0.0, 0.0],
    "rest_frequency": 230.0e9,
    "spectral_frame": "LSRK",
    "direction_frame": "J2000",
    "beams": None,
    "brightness_unit": "Jy/beam",
    "telescope": "ALMA",
    "object": "",
    "epoch_mjd": None,
}
SHAPES = [(9, 13), (16, 16), (24, 31), (40, 40)]
# The functions of findContinuum that the port run records.
RECORDED = [
    "amendMaskYesOrNo",
    "computeNpixCubeMedian",
    "computeNpixMom8Median",
    "computeNpixMom8MedianBadAtm",
    "countPixelsAboveZero",
    "cubeNoiseLevel",
    "extraMaskYesOrNo",
    "findOuterAnnulusForPBCube",
    "imageSNR",
    "imageSNRAnnulus",
    "immath",
    "immoments",
    "imstat",
    "oneEvent",
    "onlyExtraMaskYesOrNo",
]


def _same(value, expected):
    """Equal bit for bit, NaN equal to NaN."""
    return np.array_equal(np.float64(value), np.float64(expected), equal_nan=True)


def _all_same(values, expected):
    return len(values) == len(expected) and all(
        _same(value, reference)
        for value, reference in zip(values, expected, strict=True)
    )


def _map(values):
    """An ``(l, m)`` array as an image-shaped map."""
    return xr.DataArray(values[None, None, None], dims=MAP_DIMS)


def _plane(image):
    """The ``(l, m)`` plane of an image-shaped map."""
    return np.asarray(image)[0, 0, 0]


def _joint_mask(joint, footprint):
    """A joint mask of 0 and 1 values, valid in its footprint, as a native
    joint mask (False outside the footprint)."""
    return xr.Dataset(
        {
            "JOINT_MASK": _map((joint == 1) & footprint),
            "JOINT_MASK_FOOTPRINT": _map(footprint),
        }
    )


def _map_coords(shape):
    return xr.Dataset(
        coords={
            "time": [0.0],
            "frequency": [230.0e9],
            "polarization": ["I"],
            "l": np.arange(shape[0]) * 1e-6,
            "m": np.arange(shape[1]) * 1e-6,
        }
    )


def _write(path, values, valid=None):
    """An ``(l, m)`` map as a small image of the port, invalid where NaN
    unless ``valid`` is given."""
    valid = np.isfinite(values) if valid is None else valid
    write_small_image(
        str(path),
        values[:, :, None, None],
        valid[:, :, None, None],
        dict(METADATA, shape=[*values.shape, 1, 1]),
    )
    return str(path)


def _read(path):
    """A small image of the port as an ``(l, m)`` map, NaN where invalid."""
    values, valid, _ = read_small_image(path)
    return np.where(valid, values, np.nan)[:, :, 0, 0]


def _selected(text, arrays):
    """The pixels that a mask expression of findContinuum selects, as the port
    evaluates it."""
    return np.asarray(evaluate_lattice_expression(text, arrays), dtype=bool)


def _selection(ranges):
    """findContinuum's selection string of ``[start, stop)`` ranges."""
    return ";".join(f"{start}~{stop - 1}" for start, stop in ranges)


@pytest.fixture(scope="module")
def blocks():
    """The lines of findContinuum that the tests run, compiled once."""
    return {
        "round_zero_masks": vendored_block(
            1712,
            "if jointMask is None:",
            1723,
            "jointMaskTestAnnulus = ''",
            ["jointMask", "useAnnulus", "pbcube", "pbmom", "subimage", "imageInfo"],
        ),
        "moment_difference_mask": vendored_block(
            1767,
            "if pbcube is None:",
            1770,
            "mymask = '\"%s\">0.23' % (pbmom)",
            ["pbcube", "pbmom"],
        ),
        "cube_noise_mask": vendored_block(
            3722,
            "if mask != '' and mask is not None:",
            3728,
            'mymask = \'"%s">%f && "%s"<%f\' % (pbcube, lowerAnnulusLevel, '
            "pbcube, higherAnnulusLevel)",
            ["mask", "pbcube", "lowerAnnulusLevel", "higherAnnulusLevel"],
        ),
        "normalization_mask": vendored_block(
            8502,
            "if jointMaskForNormalize not in ['', None]:",
            8505,
            'outsideMaskQuoted = \'"%s">%g && "%s"<%g\'%(pbimg,lowerAnnulusLevel,'
            "pbimg,higherAnnulusLevel)",
            [
                "jointMaskForNormalize",
                "pbimg",
                "lowerAnnulusLevel",
                "higherAnnulusLevel",
            ],
        ),
        "count_mask": vendored_block(
            7669,
            "if pbmom is None:",
            7672,
            'mask = \'"%s">0 && "%s">0.21\' % (img,pbmom)',
            ["img", "pbmom"],
        ),
        "selection_maps": vendored_block(
            1755,
            "mom0fcScaled = mymom0fc+'.scaled'",
            1773,
            "outfile=mymomDiff)",
            [
                "mymom0fc",
                "mymom8fc",
                "channelWidth",
                "meanFreqHz",
                "selection",
                "momDiff",
                "amendMaskIterationName",
                "pbcube",
                "pbmom",
            ],
        ),
        "negative_fraction": vendored_block(
            1939,
            "npixels = imstat(mymom8fc, listit=imstatListit)['npts'][0]",
            1947,
            "fractionNegativePixels = float(negativePixels)/npixels",
            ["mymom8fc", "jointMaskTest"],
        ),
        "difference": vendored_block(
            2201,
            "momDiff['.diff'] = momDiff['.amendedMask'] + '.diff'",
            2207,
            "diffStats = imageSNRAnnulus(momDiff['.diff'], jointMaskTest, "
            "jointMaskTestAnnulus, applyMaskToAll=False)",
            ["momDiff", "useAnnulus", "jointMaskTest", "jointMaskTestAnnulus"],
        ),
        "cube_noise": vendored_block(
            3733,
            "cubeMedian = results['median'][0]",
            3743,
            "return cubeScaledMAD[0], cubeMedian",
            ["results", "percentile"],
        ),
        "cube_snr": vendored_block(
            1912,
            "momDiffSNRCube = (momDiffPeakOutside-momDiffMedian)/MADCubeOutside",
            1912,
            "momDiffSNRCube = (momDiffPeakOutside-momDiffMedian)/MADCubeOutside",
            ["momDiffPeakOutside", "momDiffMedian", "MADCubeOutside"],
        ),
        "lowering": vendored_block(
            1915,
            "if mom8fcSNRMom8 > 0:",
            1920,
            "casalogPost('Reducing momDiffLevel by 0.5 to %f' % (momDiffLevel))",
            ["mom8fcSNRMom8", "momDiffSNR", "momDiffLevel"],
        ),
    }


def _scene(rng, directory, blocks, shape, empty=None):
    """A mean primary beam (NaN where no channel is valid), a joint mask with
    its footprint, annulus levels, findContinuum's mask expressions of them and
    the native regions.

    The levels bracket the middle of the beam outside the joint mask, and some
    pixels of the beam lie exactly on the levels as findContinuum writes them.
    ``empty`` makes the region outside the joint mask (``"outside"``) or the
    noise annulus (``"annulus"``) empty.
    """
    l_index, m_index = np.indices(shape)
    radius2 = (l_index - rng.uniform(0, shape[0])) ** 2 + (
        m_index - rng.uniform(0, shape[1])
    ) ** 2
    primary_beam = np.exp(-radius2 / (max(shape) ** 2 * rng.uniform(0.1, 0.8)))
    primary_beam[primary_beam < 0.2] = np.nan
    joint = (rng.random(shape) < rng.uniform(0.05, 0.4)).astype(np.float64)
    low, high = np.quantile(
        primary_beam[np.isfinite(primary_beam) & (joint == 0)], [0.3, 0.7]
    )
    cube_levels, mean_levels = (
        AnnulusLevels(rng.uniform(0.2, low), rng.uniform(high, 1.0)) for _ in range(2)
    )
    if empty == "outside":
        joint[:] = 1.0
    if empty == "annulus":
        cube_levels = AnnulusLevels(0.5, 0.5)
    ties = [
        *(round_to_decimals(level, 6) for level in (*cube_levels, *mean_levels)),
        *(round_to_significant_digits(level, 6) for level in mean_levels),
        0.21,
        0.23,
        0.0,
    ]
    primary_beam.flat[rng.choice(primary_beam.size, len(ties), replace=False)] = ties
    footprint = primary_beam > 0
    paths = {
        "joint": _write(directory / "joint", joint, footprint),
        "primary_beam": _write(directory / "pbmom", primary_beam),
        "ones": _write(directory / "ones", np.ones(shape)),
    }
    masks = blocks["round_zero_masks"]
    masks.__globals__["findOuterAnnulusForPBCube"] = lambda *args, **kwargs: tuple(
        cube_levels
    )
    with quiet():
        round_zero = masks(
            paths["joint"], True, "cube.pb", paths["primary_beam"], False, None
        )
        texts = {
            "OUTSIDE_JOINT_MASK": round_zero["jointMaskTest"],
            "NOISE_ANNULUS": round_zero["jointMaskTestAnnulus"],
            "CUBE_NOISE_ANNULUS": blocks["cube_noise_mask"](
                round_zero["jointMaskTest"], paths["primary_beam"], *mean_levels
            )["mymask"],
            "MOMENT_DIFFERENCE_REGION": blocks["moment_difference_mask"](
                "cube.pb", paths["primary_beam"]
            )["mymask"],
            "PRIMARY_BEAM_ABOVE_SPECTRUM_LEVEL": blocks["count_mask"](
                paths["ones"], paths["primary_beam"]
            )["mask"],
            "ONLY_EXTRA_MASK_NORMALIZATION_REGION": blocks["normalization_mask"](
                None, paths["primary_beam"], *mean_levels
            )["outsideMaskQuoted"],
            "PRIMARY_BEAM_FOOTPRINT": blocks["count_mask"](paths["primary_beam"], None)[
                "mask"
            ],
        }
    regions = round_zero_regions(
        _joint_mask(joint, footprint),
        _map(primary_beam),
        cube_levels,
        mean_levels,
    )
    arrays = {
        paths["joint"]: np.where(footprint, joint, np.nan),
        paths["primary_beam"]: primary_beam,
        paths["ones"]: np.ones(shape),
    }
    return {
        "shape": shape,
        "primary_beam": primary_beam,
        "paths": paths,
        "texts": texts,
        "arrays": arrays,
        "regions": regions,
        "outside": _plane(regions["OUTSIDE_JOINT_MASK"]),
        "annulus": _plane(regions["NOISE_ANNULUS"]),
    }


def _map_values(rng, scene, kind):
    """A map whose medians and deviations outside the joint mask and in the
    noise annulus are of one kind, NaN where invalid.

    ``"noise"`` has positive deviations in both regions, ``"outside_flat"``
    a zero deviation outside the joint mask, ``"annulus_flat"`` in the noise
    annulus, ``"flat"`` in both (the standard deviation divides),
    ``"constant"`` a zero standard deviation too, and ``"single"`` one valid
    pixel outside the joint mask.
    """
    shape, outside, annulus = scene["shape"], scene["outside"], scene["annulus"]
    values = rng.standard_normal(shape) * 10.0 ** rng.uniform(-4, 1) + rng.normal()
    if rng.random() < 0.3:
        values = np.round(values, 1)
    mostly = rng.random(shape) < 0.9
    if kind == "outside_flat":
        values[outside & ~annulus & mostly] = 0.5
    elif kind == "annulus_flat":
        values[annulus & mostly] = 0.5
    elif kind == "flat":
        values[outside & mostly] = 0.5
    elif kind == "constant":
        values[outside] = 1.0
    values[~np.isfinite(scene["primary_beam"]) | (rng.random(shape) < 0.05)] = np.nan
    if kind == "single":
        values[outside] = np.nan
        values.flat[np.flatnonzero(annulus)[0]] = rng.normal()
    return values


KINDS = ["noise", "outside_flat", "annulus_flat", "flat", "constant", "single"]


def test_round_zero_regions_equal_findcontinuums_masks(blocks, tmp_path):
    """The seven regions select the pixels of findContinuum's mask expressions,
    pixels on a level as written included."""
    rng = np.random.default_rng(11)
    for index in range(40):
        directory = tmp_path / str(index)
        directory.mkdir()
        scene = _scene(rng, directory, blocks, SHAPES[index % len(SHAPES)])
        assert sorted(scene["regions"]) == sorted(scene["texts"])
        for name, text in scene["texts"].items():
            region = scene["regions"][name]
            assert region.dtype == bool and region.dims == MAP_DIMS
            np.testing.assert_array_equal(
                _plane(region), _selected(text, scene["arrays"]), err_msg=name
            )


def test_map_statistics_equal_the_ports_imstat(blocks, tmp_path):
    """imstat of the port with findContinuum's masks: float32 and float64 maps,
    one value, ties and even counts; no value gives None."""
    rng = np.random.default_rng(20)
    keys = ("npts", "sum", "max", "min", "median", "medabsdevmed", "sigma")
    for index in range(12):
        directory = tmp_path / str(index)
        directory.mkdir()
        scene = _scene(rng, directory, blocks, SHAPES[index % len(SHAPES)])
        values = _map_values(rng, scene, KINDS[index % len(KINDS)])
        if index % 2:
            values = values.astype(np.float32)
        path = _write(directory / "map", values.astype(np.float64))
        for name in ("OUTSIDE_JOINT_MASK", "NOISE_ANNULUS"):
            with quiet():
                expected = findContinuum.imstat(path, mask=scene["texts"][name])
            statistics = map_statistics(values, _plane(scene["regions"][name]))
            assert _all_same(statistics, [expected[key][0] for key in keys])
            assert type(statistics.count) is int
    assert map_statistics(np.full((3, 3), np.nan), np.ones((3, 3), bool)) is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_image_snrs_equal_findcontinuum(blocks, tmp_path):
    """imageSNR, imageSNRAnnulus and the sums of the port's imstat on random
    maps, through every branch of the deviation and the fallback."""
    rng = np.random.default_rng(12)
    for index in range(48):
        directory = tmp_path / str(index)
        directory.mkdir()
        scene = _scene(rng, directory, blocks, SHAPES[index % len(SHAPES)])
        outside_text = scene["texts"]["OUTSIDE_JOINT_MASK"]
        annulus_text = scene["texts"]["NOISE_ANNULUS"]
        regions = scene["regions"]
        images = {}
        for name in ("SKY_MOMENT_MAXIMUM_SELECTION", "MOMENT_DIFFERENCE"):
            values = _map_values(rng, scene, KINDS[index % len(KINDS)])
            path = _write(directory / name, values)
            with quiet():
                snr = findContinuum.imageSNR(
                    path,
                    mask=outside_text,
                    maskWithAnnulus=annulus_text,
                    useAnnulus=True,
                    returnAllStats=True,
                    applyMaskToAll=False,
                )
                outside = [
                    findContinuum.imageSNRAnnulus(
                        path, outside_text, annulus_text, applyMaskToAll=peak_outside
                    )
                    for peak_outside in (True, False)
                ]
                total = findContinuum.imstat(path)["sum"][0]
            image = _map(values)
            assert _all_same(
                image_snr(
                    image, regions["OUTSIDE_JOINT_MASK"], regions["NOISE_ANNULUS"]
                ),
                snr,
            )
            for peak_outside, expected in zip((True, False), outside, strict=True):
                assert _all_same(
                    image_snr_outside(
                        image,
                        regions["OUTSIDE_JOINT_MASK"],
                        regions["NOISE_ANNULUS"],
                        peak_outside=peak_outside,
                    ),
                    expected,
                )
            images[name] = (image, [*snr, *outside[0], total])
        statistics = selection_statistics(
            xr.Dataset({name: image for name, (image, _) in images.items()}), regions
        )
        expected = [value for _, values in images.values() for value in values]
        assert _all_same(statistics, [*expected, expected[1]])


@pytest.mark.parametrize("empty", ["outside", "annulus"])
def test_an_empty_region_raises(blocks, tmp_path, empty):
    """findContinuum's imageSNR indexes an empty imstat result."""
    rng = np.random.default_rng(13)
    scene = _scene(rng, tmp_path, blocks, (16, 16), empty=empty)
    values = _map_values(rng, scene, "noise")
    path = _write(tmp_path / "map", values)
    texts, regions = scene["texts"], scene["regions"]
    with quiet(), pytest.raises(IndexError):
        findContinuum.imageSNR(
            path,
            mask=texts["OUTSIDE_JOINT_MASK"],
            maskWithAnnulus=texts["NOISE_ANNULUS"],
            useAnnulus=True,
        )
    maps = xr.Dataset(
        {
            name: _map(values)
            for name in ("SKY_MOMENT_MAXIMUM_SELECTION", "MOMENT_DIFFERENCE")
        }
    )
    for call in (
        lambda: image_snr(
            _map(values), regions["OUTSIDE_JOINT_MASK"], regions["NOISE_ANNULUS"]
        ),
        lambda: image_snr_outside(
            _map(values), regions["OUTSIDE_JOINT_MASK"], regions["NOISE_ANNULUS"]
        ),
        lambda: selection_statistics(maps, regions),
    ):
        with pytest.raises(ValueError, match="noise annulus"):
            call()


def test_counts_above_a_level_equal_findcontinuum(blocks, tmp_path):
    """computeNpixMom8Median, computeNpixMom8MedianBadAtm and
    computeNpixCubeMedian, with thresholds that fall on pixel values once
    written with "%f"."""
    rng = np.random.default_rng(14)
    functions = [
        findContinuum.computeNpixMom8Median,
        findContinuum.computeNpixMom8MedianBadAtm,
        findContinuum.computeNpixCubeMedian,
    ]
    for index in range(24):
        directory = tmp_path / str(index)
        directory.mkdir()
        scene = _scene(rng, directory, blocks, SHAPES[index % len(SHAPES)])
        values = np.round(_map_values(rng, scene, "noise"), 6)
        path = _write(directory / "map", values)
        valid = values[np.isfinite(values)]
        for _ in range(6):
            level = float(rng.choice([5.0, 5.5, 7.25, 7.5, 8.0, 8.5, 11.0, 11.5, 12.0]))
            scaled_mad = abs(rng.normal()) * 10.0 ** rng.uniform(-3, 0)
            median = float(rng.choice(valid)) - level * scaled_mad
            for function, text, region in (
                (functions[0], scene["texts"]["OUTSIDE_JOINT_MASK"], "outside"),
                (functions[1], scene["texts"]["OUTSIDE_JOINT_MASK"], "outside"),
                (functions[2], "", None),
            ):
                with quiet():
                    expected = function(median, level, scaled_mad, path, text)
                count = count_above_level(
                    _map(values),
                    median,
                    level,
                    scaled_mad,
                    None if region is None else scene["regions"]["OUTSIDE_JOINT_MASK"],
                )
                assert type(count) is int and count == expected


def test_negative_pixel_fraction_equals_findcontinuum(blocks, tmp_path):
    rng = np.random.default_rng(15)
    for index in range(12):
        directory = tmp_path / str(index)
        directory.mkdir()
        scene = _scene(rng, directory, blocks, SHAPES[index % len(SHAPES)])
        values = _map_values(rng, scene, "noise")
        if index % 4 == 3:
            values = np.abs(values)
        path = _write(directory / "maximum", values)
        with quiet():
            expected = blocks["negative_fraction"](
                path, scene["texts"]["OUTSIDE_JOINT_MASK"]
            )["fractionNegativePixels"]
        fraction = negative_pixel_fraction(
            _map(values), scene["regions"]["OUTSIDE_JOINT_MASK"]
        )
        assert _same(fraction, expected)
    empty = np.full((4, 4), np.nan)
    with pytest.raises(ValueError, match="no valid pixel"):
        negative_pixel_fraction(_map(empty), _map(np.ones((4, 4), bool)))


def test_selection_maps_equal_findcontinuums_images(blocks, tmp_path):
    """The port's scaled moment 0 and moment difference (findContinuum's
    immath lines) on random maps and selections, overlapping ranges included;
    the moment difference is made in the integral's buffer."""
    rng = np.random.default_rng(16)
    ranges_rng = random.Random(16)
    widths = [976562.5, 488281.25, 1.953125e6, 15.625e6, 31.25e6]
    for index in range(24):
        directory = tmp_path / str(index)
        directory.mkdir()
        shape = SHAPES[index % len(SHAPES)]
        scene = _scene(rng, directory, blocks, shape)
        integral = rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 2)
        maximum = (rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 0)).astype(
            np.float32
        )
        empty = (rng.random(shape) < 0.1) | ~np.isfinite(scene["primary_beam"])
        integral[empty], maximum[empty] = np.nan, np.nan
        n_channels = int(rng.choice([8, 40, 480, 3840]))
        ranges = random_ranges(ranges_rng, n_channels)
        if index % 3 == 2:
            ranges.append(list(ranges[0]))
        channel_width = float(rng.choice(widths))
        mean_frequency = rng.uniform(84e9, 950e9)
        paths = {
            "integral": _write(directory / "cube.integrated", integral),
            "maximum": _write(directory / "cube.maximum", maximum.astype(np.float64)),
        }
        with quiet():
            names = blocks["selection_maps"](
                paths["integral"],
                paths["maximum"],
                channel_width,
                mean_frequency,
                _selection(ranges),
                {},
                ".original",
                "cube.pb",
                scene["paths"]["primary_beam"],
            )
        buffer = integral.copy()
        maps = selection_maps(
            buffer if index % 2 else buffer[None, None, None],
            maximum if index % 4 < 2 else maximum.astype(np.float64),
            len(channels_of_ranges(ranges)),
            channel_width,
            mean_frequency,
            scene["regions"]["MOMENT_DIFFERENCE_REGION"],
            _map_coords(shape),
        )
        assert np.array_equal(
            _plane(maps["MOMENT_DIFFERENCE"]), _read(names["mymomDiff"]), equal_nan=True
        )
        assert np.array_equal(
            _plane(maps["SKY_MOMENT_MAXIMUM_SELECTION"]),
            _read(paths["maximum"]),
            equal_nan=True,
        )
        assert maps["SKY_MOMENT_MAXIMUM_SELECTION"].dtype == np.float64
        assert np.shares_memory(maps["MOMENT_DIFFERENCE"].values, buffer)
        assert maps["MOMENT_DIFFERENCE"].dims == MAP_DIMS


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_difference_map_snr_equals_findcontinuum(blocks, tmp_path):
    """The image of findContinuum's difference of the moment differences and
    its imageSNRAnnulus with the peak anywhere."""
    rng = np.random.default_rng(17)
    for index in range(12):
        directory = tmp_path / str(index)
        directory.mkdir()
        scene = _scene(rng, directory, blocks, SHAPES[index % len(SHAPES)])
        reference = _map_values(rng, scene, "noise")
        amended = _map_values(rng, scene, KINDS[index % len(KINDS)])
        momdiff = {
            ".original": _write(directory / "original", reference),
            ".amendedMask": _write(directory / "amended", amended),
        }
        with quiet():
            expected = blocks["difference"](
                momdiff,
                True,
                scene["texts"]["OUTSIDE_JOINT_MASK"],
                scene["texts"]["NOISE_ANNULUS"],
            )["diffStats"][0]
        regions = scene["regions"]
        assert _same(
            difference_map_snr(
                _map(amended),
                _map(reference),
                regions["OUTSIDE_JOINT_MASK"],
                regions["NOISE_ANNULUS"],
            ),
            expected,
        )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_original_round_cube_statistics_equal_findcontinuum(blocks):
    """cubeNoiseLevel's percentile and first median of per-channel rows in any
    row order (channels without an annulus pixel are NaN), oneEvent and the
    cube signal-to-noise ratio, zero deviations included."""
    rng = np.random.default_rng(18)
    for n_rows in (1, 2, 3, 4, 5, 8, 33, 34, 480):
        for _ in range(8):
            medians = rng.normal(size=n_rows) * 1e-3
            mads = np.abs(rng.normal(size=n_rows)) * 1e-3
            empty = rng.random(n_rows) < 0.15
            medians[empty], mads[empty] = np.nan, np.nan
            if rng.random() < 0.2:
                mads[:] = 0.0
            channels = np.sort(rng.choice(4000, n_rows, replace=False))
            n_valid_voxels = int(rng.integers(100, 11250**2))
            peak, median = rng.normal(size=2) * 1e-3
            expected_mad, expected_median = blocks["cube_noise"](
                {"median": medians, "medabsdevmed": mads}, 50
            )
            order = rng.permutation(n_rows)
            statistics = original_round_cube_statistics(
                medians[order],
                mads[order],
                channels[order],
                n_valid_voxels,
                peak,
                median,
            )
            assert _all_same(
                statistics,
                [
                    expected_mad,
                    expected_median,
                    findContinuum.oneEvent(np.float64(n_valid_voxels), 10),
                    blocks["cube_snr"](peak, median, expected_mad)["momDiffSNRCube"],
                ],
            )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_lowered_moment_difference_level_equals_findcontinuum(blocks):
    rng = np.random.default_rng(19)
    values = [0.0, -1.0, 1e-300, np.inf, -np.inf, np.nan]
    with quiet():
        for _ in range(2000):
            maximum_snr_outside = (
                float(rng.choice(values)) if rng.random() < 0.2 else rng.normal() * 20
            )
            snr = (
                float(rng.choice(values))
                if rng.random() < 0.1
                else maximum_snr_outside * rng.choice([0.3, rng.uniform(0, 1)])
            )
            level = float(rng.choice([8.0, 11.5]))
            expected = blocks["lowering"](maximum_snr_outside, snr, level)[
                "momDiffLevel"
            ]
            assert _same(
                lowered_moment_difference_level(level, snr, maximum_snr_outside),
                expected,
            )


@pytest.mark.parametrize(
    "call",
    [
        lambda planes: round_zero_regions(
            xr.Dataset({"JOINT_MASK": planes, "JOINT_MASK_FOOTPRINT": planes}),
            planes.astype(float),
            AnnulusLevels(0.3, 0.8),
            AnnulusLevels(0.3, 0.8),
        ),
        lambda planes: selection_maps(
            np.zeros(planes.shape),
            np.zeros(planes.shape),
            1,
            1e6,
            230e9,
            planes,
            xr.Dataset(coords=planes.coords),
        ),
        lambda planes: image_snr(planes.astype(float), planes, planes),
        lambda planes: image_snr_outside(planes.astype(float), planes, planes),
        lambda planes: selection_statistics(
            xr.Dataset(
                {
                    "SKY_MOMENT_MAXIMUM_SELECTION": planes.astype(float),
                    "MOMENT_DIFFERENCE": planes.astype(float),
                }
            ),
            xr.Dataset({"OUTSIDE_JOINT_MASK": planes, "NOISE_ANNULUS": planes}),
        ),
        lambda planes: count_above_level(planes.astype(float), 0.0, 1.0, 1.0, None),
        lambda planes: moment_difference_counts(
            planes.astype(float), 0.0, 1.0, 1.0, 8.0, planes
        ),
        lambda planes: negative_pixel_fraction(planes.astype(float), planes),
        lambda planes: difference_map_snr(
            planes.astype(float), planes.astype(float), planes, planes
        ),
    ],
)
def test_maps_have_one_time_and_one_polarization(call):
    planes = xr.DataArray(
        np.ones((1, 1, 2, 4, 5), dtype=bool),
        dims=MAP_DIMS,
        coords={"polarization": ["I", "Q"]},
    )
    with pytest.raises(ValueError, match="one time and one polarization"):
        call(planes)


@pytest.fixture(scope="module", params=["line_set", "small_beam", "sidelobes"])
def port_run(request, tmp_path_factory):
    """The port on a synthetic cube of the CASA comparison (the common path,
    pixels outside the primary-beam limit, the amended round), with the calls of
    findContinuum's helpers recorded as (name, args, kwargs, result)."""
    img_xds = make_find_continuum_image_xds(**CASA_RESULTS[request.param][0])
    directory = tmp_path_factory.mktemp(request.param)
    store = str(directory / "cube.img.zarr")
    write_image(img_xds, imagename=store, out_format="zarr", overwrite=True)
    calls = []

    def recorder(name, function):
        def record(*args, **kwargs):
            result = function(*args, **kwargs)
            calls.append((name, args, kwargs, result))
            return result

        return record

    n_channels = img_xds.sizes["frequency"]
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in RECORDED:
            monkeypatch.setattr(
                findContinuum, name, recorder(name, getattr(findContinuum, name))
            )
        find_continuum(
            input_image_store=store,
            atmospheric_transmission=np.full(n_channels, 0.92),
            atmospheric_sky_temperature=np.full(n_channels, 27.0),
            work_directory=str(directory / "work"),
            n_mapping_parallelism={"frequency": 3},
        )
    return img_xds, calls


def test_the_ports_maps_and_statistics(port_run):
    """Regions, maps, signal-to-noise ratios, sums, counts, the negative-pixel
    fraction and the cube noise of every round of the port, bit for bit, from
    the port's mean primary beam, joint mask, annulus levels and moments."""
    img_xds, calls = port_run

    def recorded(function):
        return [(a, k, r) for name, a, k, r in calls if name == function]

    image_snrs = [
        (args, kwargs, result)
        for args, kwargs, result in recorded("imageSNR")
        if kwargs.get("applyMaskToAll") is False and kwargs.get("returnAllStats")
    ]
    outside_text = image_snrs[0][1]["mask"]
    annulus_text = image_snrs[0][1]["maskWithAnnulus"]
    joint_path = outside_text.split('"')[1]
    ((_, noise_kwargs, noise_result),) = recorded("cubeNoiseLevel")
    primary_beam_path = noise_kwargs["pbcube"]
    levels = {
        args[0]: result for args, _, result in recorded("findOuterAnnulusForPBCube")
    }
    cube_levels = next(
        result for path, result in levels.items() if path.endswith(".pb")
    )
    joint_values, joint_valid, _ = read_small_image(joint_path)
    joint, footprint = joint_values[:, :, 0, 0], joint_valid[:, :, 0, 0]
    primary_beam = _read(primary_beam_path)
    regions = round_zero_regions(
        _joint_mask(joint, footprint),
        _map(primary_beam),
        AnnulusLevels(*cube_levels),
        AnnulusLevels(*levels[primary_beam_path]),
    )
    arrays = {
        joint_path: np.where(footprint, joint, np.nan),
        primary_beam_path: primary_beam,
    }
    ((_, noise_statistics_kwargs, noise_rows),) = (
        call for call in recorded("imstat") if call[1].get("axes") == [0, 1, 2]
    )
    for name, text in (
        ("OUTSIDE_JOINT_MASK", outside_text),
        ("NOISE_ANNULUS", annulus_text),
        ("CUBE_NOISE_ANNULUS", noise_statistics_kwargs["mask"]),
        (
            "MOMENT_DIFFERENCE_REGION",
            next(
                k["mask"]
                for _, k, _ in recorded("immath")
                if k.get("expr") == "IM0-IM1"
            ),
        ),
    ):
        np.testing.assert_array_equal(
            _plane(regions[name]), _selected(text, arrays), err_msg=name
        )
    np.testing.assert_array_equal(_plane(regions["PRIMARY_BEAM_FOOTPRINT"]), footprint)
    in_beam = (
        (joint > 0) & footprint & _plane(regions["PRIMARY_BEAM_ABOVE_SPECTRUM_LEVEL"])
    )
    # The first round counts with the mean primary beam's relative path.
    joint_counts = [
        result
        for (path, beam_path), _, result in recorded("countPixelsAboveZero")
        if path == joint_path and beam_path is not None
    ]
    assert joint_counts and all(
        count == np.count_nonzero(in_beam) for count in joint_counts
    )

    for name in (
        "computeNpixMom8Median",
        "computeNpixMom8MedianBadAtm",
        "computeNpixCubeMedian",
    ):
        for (median, level, scaled_mad, path, mask, *_), _, count in recorded(name):
            region = regions["OUTSIDE_JOINT_MASK"] if mask else None
            assert (
                count_above_level(_map(_read(path)), median, level, scaled_mad, region)
                == count
            )

    frequency = img_xds.frequency.values
    increment = frequency[1] - frequency[0]
    mean_frequency = np.mean(
        [frequency[0], frequency[0] + (frequency.size - 1) * increment]
    )
    map_coords = collapsed_moment_axis_coords(img_xds, "frequency")
    rounds = [
        (args, kwargs)
        for args, kwargs, _ in recorded("immoments")
        if args[0].endswith(".residual") and kwargs.get("chans")
    ]
    assert rounds
    for _, kwargs in rounds:
        root = kwargs["outfile"]
        maximum_path = root + ".maximum"
        difference_path = maximum_path + ".minus.mom0fcScaled"
        maximum = _read(maximum_path)
        maps = selection_maps(
            _read(root + ".integrated").copy(),
            maximum.astype(np.float32),
            len(findContinuum.convertSelectionIntoChannelList(kwargs["chans"])),
            abs(increment),
            mean_frequency,
            regions["MOMENT_DIFFERENCE_REGION"],
            map_coords,
        )
        assert np.array_equal(
            _plane(maps["SKY_MOMENT_MAXIMUM_SELECTION"]), maximum, equal_nan=True
        )
        assert np.array_equal(
            _plane(maps["MOMENT_DIFFERENCE"]), _read(difference_path), equal_nan=True
        )
        statistics = selection_statistics(maps, regions)
        for path, first in ((maximum_path, 0), (difference_path, 7)):
            snrs = [r for a, k, r in image_snrs if a[0] == path]
            outside = [r for a, _, r in recorded("imageSNRAnnulus") if a[0] == path]
            sums = [
                r["sum"][0]
                for a, k, r in recorded("imstat")
                if a[0] == path and set(k) == {"listit"}
            ]
            assert snrs and outside and sums
            assert all(_all_same(statistics[first : first + 4], snr) for snr in snrs)
            assert all(_all_same(statistics[first + 4 : first + 6], o) for o in outside)
            assert all(_same(statistics[first + 6], total) for total in sums)
        assert _same(statistics.cube_peak, statistics.maximum_peak)
        if root.endswith(".amendedMask"):
            for args, _, _ in recorded("extraMaskYesOrNo"):
                assert _all_same(
                    [
                        statistics.moment_difference_median,
                        statistics.moment_difference_snr,
                        statistics.moment_difference_scaled_mad,
                    ],
                    [args[1], args[2], args[8]],
                )
        if not root.endswith(".original"):
            continue
        channels = np.unique(
            findContinuum.convertSelectionIntoChannelList(kwargs["chans"])
        )
        (n_valid_voxels,) = (a[0] for a, _, _ in recorded("oneEvent") if a[1] == 10)
        cube = original_round_cube_statistics(
            noise_rows["median"],
            noise_rows["medabsdevmed"],
            channels,
            int(n_valid_voxels),
            statistics.moment_difference_peak_outside,
            statistics.moment_difference_median,
        )
        ((amend, _, _),) = recorded("amendMaskYesOrNo")
        assert _all_same(cube[:2], noise_result)
        assert _all_same(
            [
                statistics.moment_difference_median,
                statistics.moment_difference_snr_outside,
                cube.moment_difference_snr_cube,
                cube.ten_event_sigma,
                statistics.moment_difference_scaled_mad,
                cube.cube_scaled_mad_outside,
            ],
            [amend[1], amend[2], amend[3], amend[7], amend[8], amend[9]],
        )
        level = lowered_moment_difference_level(
            8.0, statistics.moment_difference_snr, statistics.maximum_snr_outside
        )
        levels_used = [
            args[1]
            for args, _, _ in recorded("computeNpixMom8Median")
            if args[3] == difference_path
        ]
        assert levels_used[0] == level
        counts = moment_difference_counts(
            maps["MOMENT_DIFFERENCE"],
            statistics.moment_difference_median,
            statistics.moment_difference_scaled_mad,
            cube.cube_scaled_mad_outside,
            level,
            regions["OUTSIDE_JOINT_MASK"],
        )
        assert [
            counts.n_above,
            counts.n_above_bad_atmosphere,
            counts.n_above_cube,
            counts.n_above_cube_2,
            counts.n_above_2,
            counts.n_above_2_bad_atmosphere,
        ] == [amend[4], amend[5], amend[6], amend[13], amend[15], amend[16]]
        assert _same(
            negative_pixel_fraction(
                maps["SKY_MOMENT_MAXIMUM_SELECTION"], regions["OUTSIDE_JOINT_MASK"]
            ),
            amend[14],
        )
        for args, _, _ in recorded("onlyExtraMaskYesOrNo"):
            assert [
                counts.n_above_anywhere,
                counts.n_above_anywhere_bad_atmosphere,
            ] == [
                args[4],
                args[5],
            ]
            assert _all_same(
                [statistics.cube_peak, cube.first_channel_median_outside],
                [args[10], args[11]],
            )
