"""Tests of find_continuum_ultra's first-round maps against the simple port.

The port's runs on the synthetic cubes leave the mean primary beam, the moment
maps and the joint masks in their work directory, and findContinuum's returns
are recorded. findContinuum's first-round lines run through the port's image
tasks on synthetic maps for the branches no cube takes, and the port's
reductions and helpers on random inputs.
"""

import functools
import os

import numpy as np
import pytest
import xarray as xr
from xradio.image import write_image

from astroviper.distributed_applications.imaging import find_continuum
from astroviper.distributed_applications.imaging.find_continuum import casa_image_tasks
from astroviper.node_tasks.imaging.find_continuum import (
    read_small_image,
    write_small_image,
)
from astroviper.processing_functions.image_analysis.moments import (
    collapsed_moment_axis_coords,
)
from astroviper.processing_functions.imaging.find_continuum import (
    finalize_find_continuum_reductions,
    find_continuum_cube_reductions,
    merge_find_continuum_reductions,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.first_round_maps import (
    MAP_DIMS,
    _raised_threshold,
    add_primary_beam_planes,
    amended_round_statistics,
    annulus_levels,
    count_in_primary_beam,
    first_round_joint_mask,
    map_edge_samples,
    plane_edge_samples,
    primary_beam_based_joint_mask,
    primary_beam_mean,
    prune_size,
    prune_small_regions,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    chauvenet_statistics,
    round_to_decimals,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    BeamGeometry,
    EdgeSamples,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    CASA_RESULTS,
    graphviper_tree,
    make_find_continuum_image_xds,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    findContinuum,
    quiet,
    vendored_block,
)

PRIMARY_BEAM_LIMIT = 0.2
RAD_PER_ARCSEC = np.pi / (180.0 * 3600.0)
N_PIXELS = 64
# The 17 values meanSpectrumFromMom0Mom8JointMask returns (fC 8300).
RETURN_NAMES = (
    "intensity",
    "normalized",
    "numberPixelsInMask",
    "pbBasedMask",
    "initialQuadraticRemoved",
    "initialQuadraticImprovementRatio",
    "mom0snrs",
    "mom8snrs",
    "regionsPruned",
    "numberPixelsInMom8Mask",
    "mom0peak",
    "mom8peak",
    "jointMask",
    "nbin",
    "initialPeakOverMad",
    "mom0peakOverMad",
    "mom8peakOverMad",
)
# The locals findContinuum's first round (fC 7890-8204) reads.
FIRST_ROUND_PARAMETERS = (
    "cube",
    "pbcube",
    "psfcube",
    "pbmom",
    "mom0",
    "mom8",
    "moment0name",
    "moment8name",
    "imageInfo",
    "outdir",
    "minbeamfrac",
    "phase2",
    "snrThreshold",
    "mom0minsnr",
    "mom8minsnr",
    "overwriteMasks",
    "overwritePhase2",
    "userJointMask",
    "avoidExtremaInNoiseCalcForJointMask",
    "bidirectionalMaskPhase2",
    "minPixelsInJointMask",
    "subimage",
)


def _map(values):
    """A map of ``(l, m)`` values."""
    return xr.DataArray(np.asarray(values)[None, None, None], dims=MAP_DIMS)


def _edge_lines(plane):
    return {
        "first_l": plane[0],
        "last_l": plane[-1],
        "first_m": plane[:, 0],
        "last_m": plane[:, -1],
    }


def _blanked(primary_beam):
    """The primary beam with NaN below the limit (compared in its dtype)."""
    return np.where(primary_beam < PRIMARY_BEAM_LIMIT, np.nan, primary_beam)


def _primary_beam_channels(n_channels):
    """The channels of findContinuum's mean primary beam (fC 7793-7796)."""
    if n_channels < 16:
        return list(range(n_channels))
    return [n * n_channels // 8 for n in (1, 2, 4, 6, 7)]


def _metadata(shape, cell_arcsec=0.2, beam_arcsec=None):
    """Image metadata of a small image of the port."""
    return {
        "shape": list(shape),
        "increment": [
            -cell_arcsec * RAD_PER_ARCSEC,
            cell_arcsec * RAD_PER_ARCSEC,
            1.0,
            1e6,
        ],
        "reference_value": [1.2, -0.4, 1.0, 1e11],
        "reference_pixel": [0.0, 0.0, 0.0, 0.0],
        "rest_frequency": 1e11,
        "spectral_frame": "LSRK",
        "direction_frame": "J2000",
        "beams": None
        if beam_arcsec is None
        else [[beam_arcsec * RAD_PER_ARCSEC, beam_arcsec * RAD_PER_ARCSEC, 0.0]],
        "brightness_unit": "Jy/beam",
        "telescope": "ALMA",
        "object": "",
        "epoch_mjd": None,
    }


def _write(path, values, valid=None, **metadata):
    """A small image of ``(l, m)`` or ``(frequency, l, m)`` values, valid where
    finite unless ``valid`` is given."""
    values = np.asarray(values, dtype=np.float64)
    values = values[None] if values.ndim == 2 else values
    valid = (
        np.isfinite(values) if valid is None else np.broadcast_to(valid, values.shape)
    )
    casa = np.transpose(values, (1, 2, 0))[:, :, None]
    mask = np.transpose(valid, (1, 2, 0))[:, :, None]
    write_small_image(path, casa, mask, _metadata(casa.shape, **metadata))
    return path


def _read(path):
    """Values and validity of the first plane of a small image."""
    values, valid, _ = read_small_image(path)
    return values[:, :, 0, 0], valid[:, :, 0, 0]


def _masked(path):
    values, valid = _read(path)
    return np.where(valid, values, np.nan)


def _assert_joint_mask(joint_mask, path):
    """``joint_mask`` equals findContinuum's joint-mask image at ``path``."""
    values, valid = _read(path)
    np.testing.assert_array_equal(
        joint_mask.JOINT_MASK_FOOTPRINT.values[0, 0, 0], valid
    )
    np.testing.assert_array_equal(
        joint_mask.JOINT_MASK.values[0, 0, 0], valid & (values > 0.5)
    )


def _assert_statistics(statistics, port):
    """The statistics equal findContinuum's values by its names."""
    assert statistics.integrated_peak == port["mom0peak"]
    assert statistics.integrated_snr == port["mom0snrs"][0]
    assert statistics.maximum_peak == port["mom8peak"]
    assert statistics.maximum_snr == port["mom8snrs"][0]
    assert statistics.integrated_peak_over_mad == port["mom0peakOverMad"]
    assert statistics.maximum_peak_over_mad == port["mom8peakOverMad"]
    assert statistics.n_pixels_maximum_mask == port["numberPixelsInMom8Mask"]
    assert statistics.regions_pruned == port["regionsPruned"]
    assert statistics.needs_primary_beam_based_mask == port["pbBasedMask"]


def _assert_first_round(joint_mask, statistics, port):
    """The first round equals findContinuum's (fC 7889-8186)."""
    _assert_statistics(statistics, port)
    assert statistics.n_pixels_in_mask == port["numberPixelsInMask"]
    phase_2 = port["jointMask"].endswith(".joint.mask2")
    assert joint_mask.attrs["kind"] == ("phase_2" if phase_2 else "phase_1")


def _mean_primary_beam(img_xds):
    """PRIMARY_BEAM_MEAN of a synthetic cube."""
    primary_beam = img_xds.PRIMARY_BEAM.values
    channels = _primary_beam_channels(primary_beam.shape[1])
    total = np.zeros((1, 1, 1) + primary_beam.shape[-2:])
    count = np.zeros(total.shape, dtype=np.uint8)
    planes = primary_beam[:, channels]
    add_primary_beam_planes(total, count, planes, ~(planes < PRIMARY_BEAM_LIMIT))
    return primary_beam_mean(
        total, count, collapsed_moment_axis_coords(img_xds, "frequency")
    )


def _port_primary_beam_mean(primary_beam, n_chunks):
    """The port's mean primary beam of ``(frequency, l, m)`` planes, reduced in
    frequency chunks merged in GraphVIPER's tree (fC 7798)."""
    n_channels = primary_beam.shape[0]
    request = {
        "kind": "moments",
        "image": "pb",
        "mask": "",
        "channels": None if n_channels < 16 else _primary_beam_channels(n_channels),
        "moments": [-1],
        "frequency_axis": (1e11, 1e6, 0.0, n_channels),
        "rest_frequency": 1e11,
    }
    img_xds = xr.Dataset({"PRIMARY_BEAM": (MAP_DIMS, primary_beam[None, :, None])})
    leaves = [
        find_continuum_cube_reductions(
            img_xds.isel(frequency=chunk),
            [request],
            {"pb": "PRIMARY_BEAM"},
            {},
            chunk,
            primary_beam="PRIMARY_BEAM",
            primary_beam_limit=PRIMARY_BEAM_LIMIT,
        )
        for chunk in np.array_split(np.arange(n_channels), n_chunks)
    ]
    merged = graphviper_tree(leaves, merge_find_continuum_reductions)
    return finalize_find_continuum_reductions(merged, [request])[0][-1]


@pytest.mark.parametrize("n_channels", [8, 15, 16, 41])
def test_primary_beam_mean_equals_the_ports_reduction(n_channels):
    rng = np.random.default_rng(n_channels)
    shape = (2, n_channels, 3, 9, 7)
    primary_beam = rng.uniform(0.1, 1.0, shape).astype(np.float32)
    primary_beam[rng.random(shape) < 0.1] = np.nan
    primary_beam[..., 0, 0] = 0.15
    channels = _primary_beam_channels(n_channels)
    planes = primary_beam[:, channels]
    limit_mask = ~(planes < PRIMARY_BEAM_LIMIT)
    total = np.zeros((2, 1, 3, 9, 7))
    count = np.zeros(total.shape, dtype=np.uint8)
    add_primary_beam_planes(total, count, planes, limit_mask)
    map_coords = xr.Dataset(
        coords={
            "time": [0.0, 1.0],
            "frequency": [1e11],
            "polarization": ["I", "Q", "U"],
            "l": np.arange(9.0),
            "m": np.arange(7.0),
        }
    )
    beam = primary_beam_mean(total, count, map_coords)
    assert beam.name == "PRIMARY_BEAM_MEAN" and beam.dims == MAP_DIMS
    assert np.isnan(beam.values[..., 0, 0]).all()
    for time in range(2):
        for polarization in range(3):
            for n_chunks in (1, 3):
                expected = _port_primary_beam_mean(
                    primary_beam[time, :, polarization], n_chunks
                )
                np.testing.assert_array_equal(
                    beam.values[time, 0, polarization], expected
                )
    total_reversed, count_reversed = np.zeros_like(total), np.zeros_like(count)
    add_primary_beam_planes(
        total_reversed, count_reversed, planes[:, ::-1], limit_mask[:, ::-1]
    )
    np.testing.assert_array_equal(
        primary_beam_mean(total_reversed, count_reversed, map_coords), beam
    )


def _recorder(function, calls):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        calls.append((args, result))
        return result

    return wrapper


@pytest.fixture(scope="module", params=["line_set", "small_beam", "sidelobes"])
def port_run(request, tmp_path_factory):
    """The port's run on a synthetic cube, with what findContinuum's first
    round computed: ``(case, img_xds, work-directory stem, calls by function)``."""
    case = request.param
    img_xds = make_find_continuum_image_xds(**CASA_RESULTS[case][0])
    directory = tmp_path_factory.mktemp(case)
    store = str(directory / f"{case}.img.zarr")
    write_image(img_xds, imagename=store, out_format="zarr", overwrite=True)
    calls = {}
    n_channels = img_xds.sizes["frequency"]
    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "countUnmaskedPixels",
            "findOuterAnnulusForPBCube",
            "getImageInfo",
            "meanSpectrumFromMom0Mom8JointMask",
        ):
            patch.setattr(
                findContinuum,
                name,
                _recorder(getattr(findContinuum, name), calls.setdefault(name, [])),
            )
        find_continuum(
            input_image_store=store,
            work_directory=str(directory / "work"),
            atmospheric_transmission=np.full(n_channels, 0.92),
            atmospheric_sky_temperature=np.full(n_channels, 27.0),
        )
    return case, img_xds, str(directory / "work" / case), calls


def test_primary_beam_mean_equals_the_ports_run(port_run):
    _, img_xds, stem, _ = port_run
    beam = _mean_primary_beam(img_xds)
    assert beam.frequency.values[0] == img_xds.frequency.values.mean()
    np.testing.assert_array_equal(beam.values[0, 0, 0], _masked(stem + ".pbmom"))


def test_annulus_levels_equal_the_ports_run(port_run):
    _, img_xds, _, calls = port_run
    primary_beam = img_xds.PRIMARY_BEAM.values[0, :, 0]
    limit_mask = ~(primary_beam < PRIMARY_BEAM_LIMIT)
    middle = primary_beam.shape[0] // 2
    beam = _mean_primary_beam(img_xds)
    levels = {
        ".pb": annulus_levels(
            primary_beam[limit_mask & np.isfinite(primary_beam)].min(),
            plane_edge_samples(
                _edge_lines(primary_beam[middle]), _edge_lines(limit_mask[middle])
            ),
        ),
        ".pbmom": annulus_levels(np.nanmin(beam.values), map_edge_samples(beam)),
    }
    recorded = {
        os.path.splitext(args[0])[1]: result
        for args, result in calls["findOuterAnnulusForPBCube"]
    }
    assert levels == recorded


def test_first_round_joint_mask_equals_the_ports_run(port_run):
    case, img_xds, stem, calls = port_run
    beam = _mean_primary_beam(img_xds)
    image_info = next(
        result for args, result in calls["getImageInfo"] if args[0].endswith(".psf")
    )
    joint_mask, statistics = first_round_joint_mask(
        beam.copy(data=_masked(stem + ".residual.mom.integrated")[None, None, None]),
        beam.copy(data=_masked(stem + ".residual.mom.maximum")[None, None, None]),
        beam,
        calls["countUnmaskedPixels"][0][1],
        BeamGeometry(*(image_info[index] for index in (0, 1, 3, 4, 11))),
    )
    rounds = [
        dict(zip(RETURN_NAMES, result, strict=True))
        for _, result in calls["meanSpectrumFromMom0Mom8JointMask"]
    ]
    _assert_first_round(joint_mask, statistics, rounds[0])
    _assert_joint_mask(joint_mask, rounds[0]["jointMask"])
    assert joint_mask.attrs["primary_beam_based"] is False
    assert statistics.prune_size == 5
    region = beam.copy(data=(beam.values > 0.21))
    assert (
        count_in_primary_beam(
            joint_mask.JOINT_MASK, joint_mask.JOINT_MASK_FOOTPRINT, region
        )
        == rounds[0]["numberPixelsInMask"]
    )
    # sidelobes amends the joint mask; the round of the amended mask makes no map.
    assert len(rounds) == 1 + (case == "sidelobes")
    for amended in rounds[1:]:
        _assert_statistics(amended_round_statistics(statistics), amended)
        values, valid = _read(amended["jointMask"])
        assert (
            count_in_primary_beam(
                beam.copy(data=(values > 0)[None, None, None]),
                beam.copy(data=valid[None, None, None]),
                region,
            )
            == amended["numberPixelsInMask"]
        )


def _gaussian(center, fwhm):
    l_index, m_index = np.indices((N_PIXELS, N_PIXELS), dtype=np.float64)
    radius2 = (l_index - center[0]) ** 2 + (m_index - center[1]) ** 2
    return np.exp(-4.0 * np.log(2.0) * radius2 / fwhm**2)


def _moment_maps(rng, sources):
    """Integrated and maximum moment maps of noise and Gaussian sources
    ``(center, fwhm, integrated peak, maximum peak)``."""
    integrated = rng.normal(0.0, 0.05, (N_PIXELS, N_PIXELS))
    maximum = 0.004 + rng.normal(0.0, 0.001, (N_PIXELS, N_PIXELS))
    for center, fwhm, integrated_peak, maximum_peak in sources:
        integrated += integrated_peak * _gaussian(center, fwhm)
        maximum += maximum_peak * _gaussian(center, fwhm)
    return integrated, maximum


def _case(name, rng):
    """Moment maps, mean primary beam and voxel count of a first round that
    takes the branch ``name``."""
    beam = _blanked(_gaussian((31.5, 31.5), 50.0))
    n_valid_voxels = N_PIXELS * N_PIXELS * 480
    geometry = {}
    if name == "bright":
        # A source where the mean primary beam is blanked: in the mask before
        # the footprint, outside it after (fC 8037-8043).
        integrated, maximum = _moment_maps(
            rng,
            [
                ((20, 30), 3, 10.0, 0.2),
                ((40, 22), 2, 6.0, 0.12),
                ((3, 3), 2, 8.0, 0.15),
            ],
        )
    elif name == "faint":
        integrated, maximum = _moment_maps(
            rng, [((25, 30), 3, 2.0, 0.02), ((40, 40), 4, 1.5, 0.015)]
        )
        n_valid_voxels = 10**8
    elif name == "no_phase_2":
        integrated, maximum = _moment_maps(rng, [((30, 30), 4, 0.6, 0.008)])
    elif name == "flat_maximum":
        # Most of the maximum map is one value: its clipped median absolute
        # deviation is 0 and the standard deviation scales the threshold.
        integrated, maximum = _moment_maps(rng, [((30, 30), 3, 10.0, 0.0)])
        maximum = rng.choice(
            [0.0049, 0.005, 0.005, 0.005, 0.005, 0.0051], maximum.shape
        )
        maximum += 0.1 * _gaussian((30, 30), 3)
    elif name == "maximum_above_every_level":
        # Outside the joint mask the maximum is one value, whose 6 decimals
        # fall below every pixel of the map.
        beam = _blanked(_gaussian((31.5, 31.5), 40.0))
        integrated, _ = _moment_maps(rng, [((30, 30), 3, 10.0, 0.0)])
        maximum = np.where(np.isfinite(beam), 0.0123454, 0.0123452)
        maximum += 0.05 * _gaussian((30, 30), 3)
    elif name in ("aca_retry", "aca_all_pruned"):
        # 7 m: regions of 4 pixels, kept only by the retry with half the
        # fraction, or of 1 pixel, pruned by both.
        integrated, maximum = _moment_maps(rng, [])
        size = 2 if name == "aca_retry" else 1
        for corner in ((20, 20), (40, 35), (28, 45)):
            box = (
                slice(corner[0], corner[0] + size),
                slice(corner[1], corner[1] + size),
            )
            integrated[box] += 5.0
            maximum[box] += 0.05
        geometry = {"cell_arcsec": 2.0, "beam_arcsec": 12.0}
    elif name == "few_pixels_in_primary_beam":
        beam = np.full((N_PIXELS, N_PIXELS), 0.9)
        beam[40:50, 40:50] = 0.205
        integrated, maximum = _moment_maps(rng, [((45, 45), 3, 10.0, 0.2)])
    elif name == "flat_integrated":
        _, maximum = _moment_maps(rng, [((30, 30), 3, 0.0, 0.2)])
        integrated = 0.1 + 10.0 * _gaussian((30, 30), 3)
    elif name == "negative_integrated":
        # Every valid pixel is in the two-sided phase-1 mask: nothing outside.
        _, maximum = _moment_maps(rng, [((30, 30), 3, 0.0, 0.2)])
        integrated = -1.0 + rng.normal(0.0, 1e-3, maximum.shape)
        integrated += 10.0 * _gaussian((30, 30), 3)
    elif name == "outside_filled_by_phase_2":
        # The phase-2 level of the integrated map, written with 6 decimals,
        # falls below every pixel of the footprint.
        beam = _blanked(_gaussian((31.5, 31.5), 40.0))
        _, maximum = _moment_maps(rng, [((30, 30), 3, 0.0, 0.2)])
        integrated = np.where(np.isfinite(beam), 0.0123454, 0.01)
        integrated += 10.0 * _gaussian((30, 30), 3)
    elif name == "no_footprint":
        integrated, maximum = _moment_maps(rng, [((30, 30), 3, 10.0, 0.2)])
        beam = np.full((N_PIXELS, N_PIXELS), np.nan)
    else:
        integrated, maximum = _moment_maps(rng, [((30, 30), 3, 10.0, 0.2)])
        integrated[:] = np.nan
    return integrated, maximum, beam, n_valid_voxels, geometry


def _primary_beam_cube():
    """float32 ``(frequency, l, m)`` primary beam narrowing with frequency."""
    return np.stack(
        [_gaussian((31.5, 31.5), 50.0 / (1 + 0.02 * channel)) for channel in range(6)]
    ).astype(np.float32)


def _port_first_round(directory, integrated, maximum, beam, n_valid_voxels, geometry):
    """findContinuum's first round (fC 7890-8204) on small images, with
    ``n_valid_voxels`` as the cube's count: its locals and its geometry."""
    stem = os.path.join(directory, "cube")
    paths = {
        "cube": stem + ".residual",
        "pbcube": _write(stem + ".pb", _blanked(_primary_beam_cube())),
        "psfcube": _write(
            stem + ".psf",
            np.zeros((N_PIXELS, N_PIXELS)),
            **{"cell_arcsec": 0.2, "beam_arcsec": 1.0, **geometry},
        ),
        "pbmom": _write(stem + ".pbmom", beam),
        "mom0": _write(stem + ".residual.mom.integrated", integrated),
        "mom8": _write(stem + ".residual.mom.maximum", maximum),
    }
    block = vendored_block(
        7890,
        "imstatResults = imstat(moment0name)",
        8204,
        "removeIfNecessary(jointMaskTemp)",
        FIRST_ROUND_PARAMETERS,
    )
    block.__globals__["countUnmaskedPixels"] = lambda image, outdir: n_valid_voxels
    # findContinuum divides by a median absolute deviation of 0 on some maps.
    with (
        quiet(),
        casa_image_tasks.cube_session({}, None),
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        image_info = findContinuum.getImageInfo(paths["psfcube"])
        port = block(
            **paths,
            moment0name=paths["mom0"],
            moment8name=paths["mom8"],
            imageInfo=image_info,
            outdir="",
            minbeamfrac=0.3,
            phase2=True,
            snrThreshold=23,
            mom0minsnr=5,
            mom8minsnr=4,
            overwriteMasks=True,
            overwritePhase2=True,
            userJointMask="",
            avoidExtremaInNoiseCalcForJointMask=False,
            bidirectionalMaskPhase2=False,
            minPixelsInJointMask=3,
            subimage=False,
        )
    return port, BeamGeometry(*(image_info[index] for index in (0, 1, 3, 4, 11)))


def _port_amended_round(port):
    """findContinuum's round of a given joint mask (fC 8206-8219) on the maps
    of ``port``."""
    block = vendored_block(
        8206,
        "classicResult = imstat(mom0, listit=imstatListit, verbose=imstatVerbose)",
        8219,
        "numberPixelsInMom8Mask = countPixelsAboveZero(mom8mask, pbmom)",
        ("mom0", "mom8", "mom8mask", "pbmom"),
    )
    with (
        quiet(),
        casa_image_tasks.cube_session({}, None),
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        return block(
            **{name: port[name] for name in ("mom0", "mom8", "mom8mask", "pbmom")}
        )


@pytest.mark.parametrize(
    "case",
    [
        "bright",
        "faint",
        "no_phase_2",
        "flat_maximum",
        "maximum_above_every_level",
        "aca_retry",
        "aca_all_pruned",
        "few_pixels_in_primary_beam",
    ],
)
def test_first_round_joint_mask_equals_findcontinuum(case, tmp_path):
    integrated, maximum, beam, n_valid_voxels, geometry = _case(
        case, np.random.default_rng(len(case))
    )
    port, beam_geometry = _port_first_round(
        str(tmp_path), integrated, maximum, beam, n_valid_voxels, geometry
    )
    joint_mask, statistics = first_round_joint_mask(
        _map(integrated), _map(maximum), _map(beam), n_valid_voxels, beam_geometry
    )
    _assert_first_round(joint_mask, statistics, port)
    phase_2 = joint_mask.attrs["kind"] == "phase_2"
    assert statistics.thresholds[-2:] == tuple(
        round_to_decimals(port[name], 6 if phase_2 else 9)
        for name in ("mom0threshold", "mom8threshold")
    )
    assert phase_2 == (case not in ("no_phase_2", "aca_all_pruned"))
    if statistics.needs_primary_beam_based_mask:
        assert case in ("aca_all_pruned", "few_pixels_in_primary_beam")
        primary_beam = _primary_beam_cube()
        limit_mask = ~(primary_beam < PRIMARY_BEAM_LIMIT)
        levels = annulus_levels(
            primary_beam[limit_mask].min(),
            plane_edge_samples(
                _edge_lines(primary_beam[3]), _edge_lines(limit_mask[3])
            ),
        )
        assert levels == (port["lowerAnnulusLevel"], port["higherAnnulusLevel"])
        joint_mask = primary_beam_based_joint_mask(
            _map(_blanked(primary_beam[1])),
            joint_mask.JOINT_MASK_FOOTPRINT,
            levels.higher,
            joint_mask.attrs["kind"],
        )
        assert joint_mask.attrs["primary_beam_based"] is True
    _assert_joint_mask(joint_mask, port["jointMask"])
    if case == "faint":
        assert port["mom0minsnr"] == 5 and port["mom0sigma"] > 5
    if case in ("flat_maximum", "maximum_above_every_level"):
        assert chauvenet_statistics(maximum.ravel())[1] == 0.0
    if case == "maximum_above_every_level":
        assert port["mom8mask2exists"] is False
        assert (maximum > statistics.thresholds[3]).all()
    if case == "aca_retry":
        # The prune size removes every region kept: the retry kept them.
        joint = joint_mask.JOINT_MASK.values
        assert joint.any()
        assert not prune_small_regions(joint, statistics.prune_size)[0].any()

    amended = _port_amended_round(port)
    assert statistics.integrated_chauvenet_mad == amended["myMAD8"]
    amended_statistics = amended_round_statistics(statistics)
    assert (
        amended_statistics.integrated_peak,
        amended_statistics.integrated_snr,
        amended_statistics.maximum_peak,
        amended_statistics.maximum_snr,
        amended_statistics.n_pixels_maximum_mask,
    ) == (
        amended["mom0peak"],
        amended["mom0snr"],
        amended["mom8peak"],
        amended["mom8snr"],
        amended["numberPixelsInMom8Mask"],
    )
    assert (
        amended_statistics._replace(
            maximum_snr=statistics.maximum_snr,
            regions_pruned=statistics.regions_pruned,
            thresholds=statistics.thresholds,
            needs_primary_beam_based_mask=statistics.needs_primary_beam_based_mask,
        )
        == statistics
    )
    assert (
        amended_statistics.regions_pruned,
        amended_statistics.thresholds,
        amended_statistics.needs_primary_beam_based_mask,
    ) == (0, (), False)


@pytest.mark.parametrize(
    "case, reason",
    [
        ("negative_integrated", "outside the joint mask"),
        ("outside_filled_by_phase_2", "outside the joint mask"),
        ("no_footprint", "no valid positive pixel"),
        ("no_valid_integrated", "no valid pixel"),
    ],
)
def test_first_round_joint_mask_raises_where_findcontinuum_does(case, reason, tmp_path):
    integrated, maximum, beam, n_valid_voxels, geometry = _case(
        case, np.random.default_rng(len(case))
    )
    with pytest.raises(IndexError):
        _port_first_round(
            str(tmp_path), integrated, maximum, beam, n_valid_voxels, geometry
        )
    with pytest.raises(ValueError, match=reason):
        first_round_joint_mask(
            _map(integrated),
            _map(maximum),
            _map(beam),
            n_valid_voxels,
            BeamGeometry(1.0, 1.0, -0.2, 0.2, 600.0),
        )


def test_first_round_joint_mask_raises_where_findcontinuum_loops():
    """findContinuum never ends the loop of fC 7955-7959 when more than half
    of the clipped integrated map equals its minimum."""
    integrated, maximum, beam, n_valid_voxels, _ = _case(
        "flat_integrated", np.random.default_rng(0)
    )
    with pytest.raises(ValueError, match="cannot rise"):
        first_round_joint_mask(
            _map(integrated),
            _map(maximum),
            _map(beam),
            n_valid_voxels,
            BeamGeometry(1.0, 1.0, -0.2, 0.2, 600.0),
        )


@pytest.mark.parametrize(
    "sigma, scaled_mad, median, minimum, peak, raised",
    [
        (5.0, 1e-17, 1.0, 1.0, 2.0, True),
        (np.int64(8), 2e-18, -0.5, -0.5, 1.0, True),
        (5.3, 0.01, 0.2, 0.1, 3.0, False),
        (4.0, 1e-17, 1.0, 1.0, 1.0, False),
    ],
)
def test_raised_threshold_equals_findcontinuums_loop(
    sigma, scaled_mad, median, minimum, peak, raised
):
    """fC 7951-7959 on values the sigma raises a few times; no map takes the
    loop: a positive median absolute deviation puts the threshold above the
    clipped median, which is not below the minimum."""
    block = vendored_block(
        7951,
        "mom0threshold = mom0sigma*scaledMAD + result['median'][0]",
        7959,
        "mom0threshold = mom0sigma*scaledMAD + result['median'][0]",
        ("mom0sigma", "scaledMAD", "result", "classicResult"),
    )
    with quiet():
        expected = block(
            sigma,
            scaled_mad,
            {"median": np.array([median])},
            {"min": np.array([minimum]), "max": np.array([peak])},
        )
    threshold = _raised_threshold(sigma, scaled_mad, median, minimum, peak)
    assert threshold == expected["mom0threshold"]
    assert (expected["mom0sigma"] > sigma) == raised


def _random_primary_beam(rng, n_channels, n_l, n_m):
    """float32 ``(frequency, l, m)`` primary beam of a random Gaussian."""
    center = rng.uniform(0.3, 0.7, 2) * (n_l - 1, n_m - 1)
    fwhm = rng.uniform(0.5, 4.0) * max(n_l, n_m)
    l_index, m_index = np.indices((n_l, n_m), dtype=np.float64)
    radius2 = (l_index - center[0]) ** 2 + (m_index - center[1]) ** 2
    planes = [
        np.exp(-4.0 * np.log(2.0) * radius2 / (fwhm / (1 + 0.05 * c)) ** 2)
        for c in range(n_channels)
    ]
    primary_beam = np.stack(planes).astype(np.float32)
    primary_beam[rng.random(primary_beam.shape) < 0.02] = np.nan
    return primary_beam


def _port_annulus_levels(path, n_l, n_m):
    """findContinuum's annulus levels of a small image (fC 8370-8411)."""
    image_info = [None] * 5 + [n_l, n_m] + [None] * 6
    with quiet(), casa_image_tasks.cube_session({}, None):
        return findContinuum.findOuterAnnulusForPBCube(
            path, False, False, False, image_info
        )


def test_annulus_levels_equal_findcontinuum(tmp_path):
    rng = np.random.default_rng(9)
    # Corners blanked and edge midpoints valid; an edge line blanked.
    centred = _gaussian((10, 10), 16.0)[None, :21, :21].astype(np.float32)
    blanked_line = np.full((2, 9, 8), 0.9, dtype=np.float32)
    blanked_line[:, 0] = 0.1
    cubes = [centred, blanked_line] + [
        _random_primary_beam(rng, int(rng.integers(1, 6)), *rng.integers(5, 30, 2))
        for _ in range(40)
    ]
    branches = set()
    for index, primary_beam in enumerate(cubes):
        n_channels, n_l, n_m = primary_beam.shape
        limit_mask = ~(primary_beam < PRIMARY_BEAM_LIMIT)
        middle = n_channels // 2
        valid = limit_mask & np.isfinite(primary_beam)
        if not valid.any():
            continue
        path = _write(str(tmp_path / f"pb_{index}"), primary_beam, valid)
        edge_samples = plane_edge_samples(
            _edge_lines(primary_beam[middle]), _edge_lines(limit_mask[middle])
        )
        minimum = primary_beam[valid].min()
        branches.add((minimum > 0.22, edge_samples.corners_empty_edges_valid))
        try:
            expected = _port_annulus_levels(path, n_l, n_m)
        except ValueError:
            with pytest.raises(ValueError, match="edge line"):
                annulus_levels(minimum, edge_samples)
            branches.add("empty edge line")
            continue
        assert annulus_levels(minimum, edge_samples) == expected
        if n_channels == 1:
            plane = _map(np.where(valid[0], primary_beam[0], np.nan))
            assert map_edge_samples(plane) == edge_samples
            assert annulus_levels(np.nanmin(plane), map_edge_samples(plane)) == expected
    assert branches >= {(True, False), (False, True), (False, False), "empty edge line"}


def test_annulus_levels_refuse_a_level_outside_the_unit_interval(tmp_path):
    primary_beam = np.full((1, 6, 5), 1.5, dtype=np.float32)
    path = _write(str(tmp_path / "pb"), primary_beam)
    with pytest.raises(TypeError):
        _port_annulus_levels(path, 6, 5)
    edge_samples = plane_edge_samples(
        _edge_lines(primary_beam[0]), _edge_lines(np.ones((6, 5), dtype=bool))
    )
    assert edge_samples == EdgeSamples(False, 1.5)
    with pytest.raises(ValueError, match="not in"):
        annulus_levels(1.5, edge_samples)


def test_primary_beam_based_joint_mask_equals_findcontinuum(tmp_path):
    """fC 8200-8203: the primary beam of channel 1 above the higher level
    written with 6 decimals, in the footprint of the mean primary beam."""
    block = vendored_block(
        8200,
        "imsubimage(pbcube, chans='1', mask='\"%s\">%f && \"%s\">0' % "
        "(pbcube,higherAnnulusLevel,pbmom), outfile=myMask1chan, overwrite=True)  "
        "# PIPE-1629 for the pb-based jointmask case",
        8203,
        "imsubimage(jointMaskTemp, mask='\"%s\">0' % (pbmom), outfile=jointMask, "
        "overwrite=True) # PIPE-1629 for the pb-based jointmask case",
        (
            "pbcube",
            "higherAnnulusLevel",
            "pbmom",
            "myMask1chan",
            "jointMaskTemp",
            "jointMask",
        ),
    )
    rng = np.random.default_rng(10)
    for index in range(10):
        primary_beam = _random_primary_beam(rng, 4, 23, 18)
        beam = _blanked(primary_beam.mean(axis=0, dtype=np.float64))
        beam[rng.random(beam.shape) < 0.05] = np.nan
        values = primary_beam[1][np.isfinite(primary_beam[1])]
        higher = float(rng.choice(values) if index % 2 else rng.uniform(0.2, 1.0))
        higher = higher if index % 3 else round_to_decimals(higher, 6) + 4e-7
        stem = str(tmp_path / f"cube_{index}")
        paths = {
            "pbcube": _write(stem + ".pb", _blanked(primary_beam)),
            "pbmom": _write(stem + ".pbmom", beam),
            "myMask1chan": stem + ".joint.mask.1chan",
            "jointMaskTemp": stem + ".joint.mask.tmp",
            "jointMask": stem + ".joint.mask",
        }
        with quiet(), casa_image_tasks.cube_session({}, None):
            block(higherAnnulusLevel=higher, **paths)
        footprint = _map(np.isfinite(beam) & (beam > 0))
        joint_mask = primary_beam_based_joint_mask(
            _map(_blanked(primary_beam[1])), footprint, higher, "phase_2"
        )
        assert joint_mask.attrs == {"kind": "phase_2", "primary_beam_based": True}
        _assert_joint_mask(joint_mask, paths["jointMask"])


def test_count_in_primary_beam_equals_findcontinuum(tmp_path):
    rng = np.random.default_rng(11)
    for index in range(10):
        mask = rng.random((17, 13)) < 0.4
        valid = rng.random((17, 13)) < 0.8
        beam = rng.uniform(0.15, 0.3, (17, 13))
        beam[rng.random(beam.shape) < 0.1] = np.nan
        beam[0, 0] = 0.21
        path = _write(str(tmp_path / f"mask_{index}"), mask, valid)
        pbmom = _write(str(tmp_path / f"pbmom_{index}"), beam)
        with quiet(), casa_image_tasks.cube_session({}, None):
            expected = findContinuum.countPixelsAboveZero(path, pbmom)
        region = _map(np.isfinite(beam) & (beam > 0.21))
        assert count_in_primary_beam(_map(mask), _map(valid), region) == expected


def test_prune_size_equals_findcontinuum():
    size = vendored_block(
        12021,
        "pixelsPerBeam = myinfo[0]*myinfo[1]*np.pi/4./np.abs(myinfo[3]*myinfo[4])",
        12023,
        "prunesize = np.max([4,int(pixelsPerBeam*minbeamfrac)])",
        ("myinfo", "minbeamfrac"),
    )
    aca_size = vendored_block(
        12047,
        "newprunesize = np.max([4,int(pixelsPerBeam*minbeamfrac*0.5)])",
        12047,
        "newprunesize = np.max([4,int(pixelsPerBeam*minbeamfrac*0.5)])",
        ("pixelsPerBeam", "minbeamfrac"),
    )
    rng = np.random.default_rng(12)
    for _ in range(20000):
        beam = rng.uniform(0.05, 20.0, 2)
        cell = rng.uniform(0.01, 3.0, 2) * rng.choice([-1, 1], 2)
        geometry = (*np.round(beam, int(rng.integers(1, 4))), *cell)
        expected = size([*geometry[:2], 0.0, *geometry[2:], *[0.0] * 7], 0.3)
        assert prune_size(*geometry) == expected["prunesize"]
        assert (
            prune_size(*geometry, 0.3 * 0.5)
            == aca_size(expected["pixelsPerBeam"], 0.3)["newprunesize"]
        )


def test_prune_small_regions_equals_the_port(tmp_path):
    rng = np.random.default_rng(13)
    for index, density in enumerate((0.0, 0.1, 0.3, 0.5)):
        mask = rng.random((2, 1, 3, 31, 24)) < density
        size = int(rng.integers(1, 9))
        pruned, n_pruned = prune_small_regions(mask, size)
        planes = mask.reshape(-1, 31, 24)
        path = _write(str(tmp_path / f"mask_{index}"), planes)
        with casa_image_tasks.cube_session({}, None):
            regions = casa_image_tasks.synthesismaskhandler().pruneregions(path, size)
        expected = read_small_image(path + ".pruned")[0][:, :, 0] > 0
        np.testing.assert_array_equal(
            pruned.reshape(-1, 31, 24), np.transpose(expected, (2, 0, 1))
        )
        assert n_pruned == regions["N_reg_pruned"].sum()
        plane, n_plane = prune_small_regions(mask[1, 0, 2], size)
        np.testing.assert_array_equal(plane, pruned[1, 0, 2])
        assert n_plane == regions["N_reg_pruned"][-1]
    unpruned, n_pruned = prune_small_regions(mask, 1)
    np.testing.assert_array_equal(unpruned, mask)
    assert n_pruned == 0


@pytest.mark.parametrize("sizes", [{"polarization": 2}, {"time": 2}, {"frequency": 3}])
def test_maps_have_one_plane(sizes):
    shape = tuple(sizes.get(dim, 1) for dim in MAP_DIMS[:3]) + (5, 4)
    values = xr.DataArray(np.ones(shape), dims=MAP_DIMS)
    with pytest.raises(ValueError, match="one time, frequency and polarization"):
        map_edge_samples(values)
    with pytest.raises(ValueError, match="one time, frequency and polarization"):
        count_in_primary_beam(values > 0, values > 0, values > 0)
