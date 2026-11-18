"""Tests of find_continuum_ultra's cube reductions against the simple port's
reductions (``find_continuum_cube_reductions``) of the same planes, its chunk
partials merged in GraphVIPER's binary tree."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum import (
    evaluate_lattice_expression,
    finalize_find_continuum_reductions,
    find_continuum_cube_reductions,
    merge_find_continuum_reductions,
    moment_velocity_widths,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
    SPECTRUM_PRIMARY_BEAM_LEVEL,
    accumulate_moments,
    finalize_moments,
    first_round_channel_rows,
    masked_channel_means,
    masked_channel_medians_and_mads,
    masked_channel_scaled_mads,
    merge_moment_partials,
    merge_moment_partials_in_tree_order,
    new_moment_partials,
    pack_plane_masks,
    primary_beam_masks,
    tile_occupancy,
    unpack_plane_masks,
    valid_voxels,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    round_to_decimals,
    round_to_significant_digits,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    graphviper_tree,
)

LIMIT = 0.2
DIMS = ("time", "frequency", "polarization", "l", "m")
VARIABLES = {"cube": "SKY", "pb": "PB"}
REST_FREQUENCY = 230.5e9
# Planes over three row blocks of the reductions (625, 625 and 50 rows).
TALL = (1300, 7)
# Annulus levels as findContinuum computes them, before it writes them. The
# float32 values of the written levels lie inside the annulus, so a comparison
# in float32 would select other voxels.
LOWER, HIGHER = 0.2198765432, 0.3241592653
LOWER_CUBE_NOISE = 0.2134567891
ANNULUS = AnnulusLevels(
    round_to_significant_digits(LOWER, 6), round_to_significant_digits(HIGHER, 6)
)


def _float_neighbours(values, dtype, n_steps):
    """``values`` in ``dtype`` and their ``n_steps`` neighbours on each side."""
    integer = {np.float32: np.int32, np.float64: np.int64}[dtype]
    steps = np.arange(-n_steps, n_steps + 1, dtype=integer)
    return (
        (np.asarray(values, dtype).view(integer)[:, None] + steps).view(dtype).ravel()
    )


# Primary-beam values at the limit, the spectrum level, the moment-difference
# region level and the annulus levels.
EDGE_BEAM = np.append(
    _float_neighbours(
        [LIMIT, SPECTRUM_PRIMARY_BEAM_LEVEL, 0.23, *ANNULUS], np.float32, 1
    ),
    np.float32([np.nan, 0.0, 1.0]),
)


def _planes(seed, n_time, n_channels, n_polarization, shape=TALL):
    """float32 residual and primary-beam planes with NaN, infinities, zeros,
    negative zeros and primary-beam values at the levels of the masks. The
    residual spans more binades than float64 holds beyond float32, so that its
    sums depend on the order of the additions."""
    rng = np.random.default_rng(seed)
    full = (n_time, n_channels, n_polarization, *shape)
    exponent = rng.uniform(-3, 3, (1, n_channels, 1, 1, 1)) + rng.uniform(-6, 6, full)
    sky = (rng.normal(size=full) * 10.0**exponent).astype(np.float32)
    for value, fraction in (
        (np.nan, 0.05),
        (np.inf, 0.01),
        (-np.inf, 0.01),
        (0.0, 0.05),
        (-0.0, 0.05),
    ):
        sky[rng.random(full) < fraction] = value
    l, m = np.meshgrid(
        np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij"
    )
    narrowing = 1.0 + 0.03 * np.arange(n_channels)[:, None, None]
    beam = np.exp(-2.0 * (l**2 + m**2) * narrowing)
    beam = np.broadcast_to(beam[None, :, None], full).astype(np.float32)
    edge = rng.random(full) < 0.1
    beam[edge] = rng.choice(EDGE_BEAM, np.count_nonzero(edge))
    return sky, beam


def _images(seed, shape=TALL):
    """Two-dimensional images, NaN where invalid, that regions are made of: a
    joint mask ``J``, a moment difference ``D``, a mean primary beam ``P`` and
    a normalization region ``N``."""
    rng = np.random.default_rng(seed)
    l, m = np.meshgrid(
        np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij"
    )
    images = {
        "J": (rng.random(shape) < 0.3).astype(np.float64),
        "D": rng.normal(scale=0.01, size=shape),
        "P": np.exp(-2.0 * (l**2 + m**2)),
        "N": (rng.random(shape) < 0.5).astype(np.float64),
    }
    for values in images.values():
        values[rng.random(shape) < 0.1] = np.nan
    return images


def _port(sky, beam, requests, images=None, chunks=(slice(None),)):
    """The port's results of ``requests`` on the planes of one time and
    polarization, its partials of ``chunks`` merged in GraphVIPER's tree."""
    img_xds = xr.Dataset({"SKY": (DIMS, sky), "PB": (DIMS, beam)})
    casa_images = {
        name: values[:, :, None, None] for name, values in (images or {}).items()
    }
    channels = np.arange(sky.shape[1])
    partials = [
        find_continuum_cube_reductions(
            img_xds.isel(frequency=chunk),
            requests,
            VARIABLES,
            casa_images,
            channels[chunk],
            "PB",
            LIMIT,
        )
        for chunk in chunks
    ]
    return finalize_find_continuum_reductions(
        graphviper_tree(partials, merge_find_continuum_reductions), requests
    )


def _plane(array, time, polarization):
    return array[time : time + 1, :, polarization : polarization + 1]


def _batches(channels, planes_per_batch):
    return [
        channels[start : start + planes_per_batch]
        for start in range(0, len(channels), planes_per_batch)
    ]


def _batch(sky, beam, channels):
    """A node task's batch of ``channels``: the planes, their primary beam,
    valid voxels and spectrum mask."""
    sky_batch, beam_batch = sky[:, channels], beam[:, channels]
    limit_mask, spectrum_mask = primary_beam_masks(beam_batch, LIMIT)
    return sky_batch, beam_batch, valid_voxels(sky_batch, limit_mask), spectrum_mask


def _assert_bitwise(actual, expected):
    """Equal bit for bit, NaN equal to NaN."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == np.ascontiguousarray(expected).tobytes()


def _assert_bitwise_but_zero_sign(actual, expected):
    """Equal bit for bit except the sign of a zero: which of ``+0.0`` and
    ``-0.0`` a minimum or maximum picks can depend on NumPy's reduction path,
    and the port's median of a middle ``-0.0`` is ``-0.0`` where
    ``numpy.median`` gives ``+0.0``."""
    _assert_bitwise(np.asarray(actual) + 0.0, np.asarray(expected) + 0.0)


class TestPrimaryBeamMasks:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("limit", [0.2, 0.21, 0.25, 0.1999999])
    def test_equal_the_ports_blanking_and_mask_expression(self, dtype, limit):
        beam = np.append(
            _float_neighbours([0.2, 0.21, 0.23, 0.25], dtype, 300),
            np.array([np.nan, 0.0, 1.0], dtype),
        ).reshape(1, 1, 1, 1, -1)
        blanked = beam.astype(np.float64)
        blanked[beam < limit] = np.nan
        for level, text in ((SPECTRUM_PRIMARY_BEAM_LEVEL, "0.21"), (0.23, "0.23")):
            expected = evaluate_lattice_expression(f'"pb">{text}', {"pb": blanked})
            for given in (limit, np.float64(limit)):
                limit_mask, spectrum_mask = primary_beam_masks(beam, given, level)
                np.testing.assert_array_equal(limit_mask, ~(beam < limit))
                np.testing.assert_array_equal(spectrum_mask, expected)
        assert limit_mask[..., -3] and not spectrum_mask[..., -3]

    def test_valid_voxels(self):
        sky, beam = _planes(0, 2, 3, 2, (9, 8))
        limit_mask = primary_beam_masks(beam, LIMIT)[0]
        np.testing.assert_array_equal(
            valid_voxels(sky, limit_mask), np.isfinite(sky) & ~(beam < LIMIT)
        )


@pytest.mark.parametrize("planes_per_batch", [1, 2, 5])
def test_first_round_channel_rows_equal_the_ports_statistics(planes_per_batch):
    sky, beam = _planes(1, 2, 6, 3)
    sky[0, 2, 1] = np.nan
    beam[1, 4, 0] = 0.0
    batches = [
        first_round_channel_rows(
            sky[:, channels],
            beam[:, channels],
            primary_beam_masks(beam[:, channels], LIMIT)[0],
        )
        for channels in _batches(list(range(6)), planes_per_batch)
    ]
    rows = {
        key: np.concatenate([batch[key] for batch in batches], axis=1)
        for key in batches[0]
    }
    assert rows["sky_valid_count"].dtype == np.int64
    assert rows["primary_beam_valid_count"].dtype == np.int64
    requests = [
        {
            "kind": "statistics",
            "image": image,
            "mask": "",
            "channels": None,
            "per_channel": True,
        }
        for image in ("cube", "pb")
    ]
    for time, polarization in np.ndindex(2, 3):
        sky_statistics, beam_statistics = _port(
            _plane(sky, time, polarization), _plane(beam, time, polarization), requests
        )
        plane = (time, slice(None), polarization)
        np.testing.assert_array_equal(
            rows["sky_valid_count"][plane], sky_statistics["npts"]
        )
        np.testing.assert_array_equal(
            rows["primary_beam_valid_count"][plane], beam_statistics["npts"]
        )
        for key, expected in (
            ("sky_minimum", sky_statistics["min"]),
            ("sky_maximum", sky_statistics["max"]),
            ("primary_beam_minimum", beam_statistics["min"]),
        ):
            _assert_bitwise_but_zero_sign(rows[key][plane], expected)
    assert rows["sky_valid_count"][0, 2, 1] == 0
    assert rows["primary_beam_valid_count"][0, 2, 1] > 0
    assert np.isnan(rows["sky_minimum"][0, 2, 1])
    assert np.isnan(rows["sky_maximum"][0, 2, 1])
    assert rows["sky_valid_count"][1, 4, 0] == 0
    assert rows["primary_beam_valid_count"][1, 4, 0] == 0
    assert np.isnan(rows["primary_beam_minimum"][1, 4, 0])


# Expression of the port's mask, the native region of the images, and whether
# the voxels must be in the spectrum mask.
MEAN_FORMS = {
    # The spectrum of a joint mask (fC 8447-8453).
    "joint_mask": ('"J">0 && "pb">0.21', lambda images: images["J"] > 0, True),
    # The only-extra-mask spectrum (fC 2041).
    "only_extra_mask": (
        '"D">0.004 && "P">0.23',
        lambda images: (images["D"] > 0.004) & (images["P"] > 0.23),
        False,
    ),
    # The extra-mask spectra (fC 2441, 2513).
    "extra_mask": ('"D">0.004 && "pb">0.21', lambda images: images["D"] > 0.004, True),
}


@pytest.mark.parametrize("form", sorted(MEAN_FORMS))
@pytest.mark.parametrize("planes_per_batch", [1, 2, 5])
def test_masked_channel_means_equal_the_ports_profiles(form, planes_per_batch):
    expression, region_of, in_spectrum_mask = MEAN_FORMS[form]
    sky, beam = _planes(2, 2, 7, 2)
    images = _images(3)
    region = region_of(images)
    means = []
    for channels in _batches(list(range(7)), planes_per_batch):
        sky_batch, _, valid, spectrum_mask = _batch(sky, beam, channels)
        means.append(
            masked_channel_means(
                sky_batch, valid, region, spectrum_mask if in_spectrum_mask else None
            )
        )
    means = np.concatenate(means, axis=1)
    request = {
        "kind": "profile",
        "image": "cube",
        "mask": expression,
        "channels": None,
        "function": "mean",
    }
    for time, polarization in np.ndindex(2, 2):
        (profile,) = _port(
            _plane(sky, time, polarization),
            _plane(beam, time, polarization),
            [request],
            images,
            (slice(0, 3), slice(3, 7)),
        )
        _assert_bitwise(means[time, :, polarization], profile)


@pytest.mark.parametrize(
    "rows, columns",
    [
        (slice(600, 700), slice(2, 5)),
        (slice(1200, 1300), slice(4, 7)),
        (slice(0, 0), slice(0, 0)),
    ],
)
def test_masked_channel_means_of_a_region_in_part_of_the_planes(rows, columns):
    sky, beam = _planes(4, 1, 3, 1)
    region = np.zeros(TALL, dtype=bool)
    region[rows, columns] = True
    sky_batch, _, valid, spectrum_mask = _batch(sky, beam, [0, 1, 2])
    means = masked_channel_means(sky_batch, valid, region, spectrum_mask)
    request = {
        "kind": "profile",
        "image": "cube",
        "mask": '"R">0 && "pb">0.21',
        "channels": None,
        "function": "mean",
    }
    (profile,) = _port(sky, beam, [request], {"R": region.astype(np.float64)})
    _assert_bitwise(means[0, :, 0], profile)
    assert np.isfinite(means).all() if region.any() else np.isnan(means).all()


# Expression of the port's mask, the native region of the images, and the
# primary-beam levels.
SCALED_MAD_FORMS = {
    # The normalization of the spectrum of a joint mask (fC 8503).
    "annulus_outside_joint_mask": (
        f'"pb">{LOWER:g} && "pb"<{HIGHER:g} && "J"<0.5',
        lambda images: images["J"] < 0.5,
        ANNULUS,
    ),
    # Without a joint mask (fC 8505).
    "annulus": (
        f'"pb">{LOWER:g} && "pb"<{HIGHER:g}',
        lambda images: np.ones(TALL, dtype=bool),
        ANNULUS,
    ),
    # The only-extra-mask normalization, a two-dimensional region (fC 2041).
    "region": ('"N">0.5', lambda images: images["N"] > 0.5, None),
}


@pytest.mark.parametrize("form", sorted(SCALED_MAD_FORMS))
@pytest.mark.parametrize("planes_per_batch", [1, 2, 5])
def test_masked_channel_scaled_mads_equal_the_ports_profiles(form, planes_per_batch):
    expression, region_of, levels = SCALED_MAD_FORMS[form]
    sky, beam = _planes(5, 2, 7, 2)
    images = _images(6)
    region = region_of(images)
    mads = []
    for channels in _batches(list(range(7)), planes_per_batch):
        sky_batch, beam_batch, valid, _ = _batch(sky, beam, channels)
        mads.append(
            masked_channel_scaled_mads(
                sky_batch,
                valid,
                region,
                None if levels is None else beam_batch,
                levels,
            )
        )
    mads = np.concatenate(mads, axis=1)
    request = {
        "kind": "profile",
        "image": "cube",
        "mask": expression,
        "channels": None,
        "function": "xmadm",
    }
    for time, polarization in np.ndindex(2, 2):
        (profile,) = _port(
            _plane(sky, time, polarization),
            _plane(beam, time, polarization),
            [request],
            images,
            (slice(0, 4), slice(4, 7)),
        )
        _assert_bitwise(mads[time, :, polarization], profile)
    assert np.isfinite(mads).all()


@pytest.mark.parametrize("planes_per_batch", [1, 2, 5])
def test_masked_channel_medians_and_mads_equal_the_ports_statistics(planes_per_batch):
    sky, beam = _planes(7, 2, 8, 2)
    images = _images(8)
    lower = round_to_decimals(LOWER_CUBE_NOISE, 6)
    higher = round_to_decimals(HIGHER, 6)
    region = (images["P"] > lower) & (images["P"] < higher) & (images["J"] == 0)
    selected = [0, 2, 3, 6, 7]
    medians, mads = [], []
    for channels in _batches(selected, planes_per_batch):
        sky_batch, _, valid, _ = _batch(sky, beam, channels)
        batch_medians, batch_mads = masked_channel_medians_and_mads(
            sky_batch, valid, region
        )
        medians.append(batch_medians)
        mads.append(batch_mads)
    medians, mads = np.concatenate(medians, axis=1), np.concatenate(mads, axis=1)
    request = {
        "kind": "statistics",
        "image": "cube",
        "mask": f'"P">{LOWER_CUBE_NOISE:f} && "P"<{HIGHER:f} && "J"==0',
        "channels": selected,
        "per_channel": True,
    }
    for time, polarization in np.ndindex(2, 2):
        (statistics,) = _port(
            _plane(sky, time, polarization),
            _plane(beam, time, polarization),
            [request],
            images,
            (slice(0, 3), slice(3, 5), slice(5, 8)),
        )
        _assert_bitwise(mads[time, :, polarization], statistics["medabsdevmed"])
        _assert_bitwise_but_zero_sign(
            medians[time, :, polarization], statistics["median"]
        )
    assert np.isfinite(medians).all()


def test_masked_reductions_of_a_plane_without_a_voxel():
    sky, beam = _planes(9, 1, 2, 1, (9, 8))
    sky_batch, beam_batch, valid, spectrum_mask = _batch(sky, beam, [0, 1])
    valid[0, 1] = False
    region = np.ones((9, 8), dtype=bool)
    means = masked_channel_means(sky_batch, valid, region, spectrum_mask)
    mads = masked_channel_scaled_mads(sky_batch, valid, region, beam_batch, ANNULUS)
    medians, unscaled = masked_channel_medians_and_mads(sky_batch, valid, region)
    for values in (means, mads, medians, unscaled):
        assert np.isnan(values[0, 1, 0]) and np.isfinite(values[0, 0, 0])


@pytest.mark.parametrize("planes_per_batch", [1, 2, 5])
@pytest.mark.parametrize("kind", ["selection", "first_round"])
def test_moments_equal_the_ports(kind, planes_per_batch):
    """Moments of a channel selection (fC 1741) and of every channel with
    ``"cube"!=0`` (fC 7870), frequency chunks merged in GraphVIPER's tree."""
    sky, beam = _planes(10, 2, 12, 2)
    axis = (230.0e9, 0.9765625e6, 0.0, 12)
    weights = moment_velocity_widths(axis, REST_FREQUENCY)
    first_round = kind == "first_round"
    channels = list(range(12)) if first_round else [1, 2, 3, 6, 9, 10]
    chunks = (slice(0, 4), slice(4, 5), slice(5, 9), slice(9, 12))
    map_shape = (2, 1, 2, *TALL)
    leaves = []
    for chunk in chunks:
        partials = None
        in_chunk = [
            channel for channel in channels if chunk.start <= channel < chunk.stop
        ]
        for batch in _batches(in_chunk, planes_per_batch):
            sky_batch, _, valid, _ = _batch(sky, beam, batch)
            if first_round:
                valid &= sky_batch != 0
            if partials is None:
                partials = new_moment_partials(map_shape, sky.dtype)
            accumulate_moments(partials, sky_batch, valid, weights[batch])
        leaves.append(partials)
    integral, maximum = finalize_moments(
        merge_moment_partials_in_tree_order(leaves), map_shape, sky.dtype
    )
    assert integral.dtype == np.float64 and maximum.dtype == np.float32
    request = {
        "kind": "moments",
        "image": "cube",
        "mask": '"cube"!=0' if first_round else "",
        "channels": None if first_round else channels,
        "moments": [0, 8],
        "frequency_axis": axis,
        "rest_frequency": REST_FREQUENCY,
    }
    for time, polarization in np.ndindex(2, 2):
        (maps,) = _port(
            _plane(sky, time, polarization),
            _plane(beam, time, polarization),
            [request],
            chunks=chunks,
        )
        _assert_bitwise(integral[time, 0, polarization], maps[0])
        _assert_bitwise_but_zero_sign(
            maximum[time, 0, polarization].astype(np.float64), maps[8]
        )
    assert np.isnan(integral).any() and not np.isnan(integral).all()


@pytest.mark.parametrize("n_slots", range(1, 34))
def test_tree_order_equals_the_ports_tree(n_slots):
    rng = np.random.default_rng(200 + n_slots)
    n_channels = 2 * n_slots
    shape = (1, n_channels, 1, 5, 4)
    scale = 10.0 ** rng.uniform(-6, 6, (1, n_channels, 1, 1, 1))
    sky = (rng.normal(size=shape) * scale).astype(np.float32)
    sky[rng.random(shape) < 0.1] = np.nan
    beam = np.ones(shape, dtype=np.float32)
    axis = (230.0e9, 0.9765625e6, 0.0, n_channels)
    weights = moment_velocity_widths(axis, REST_FREQUENCY)
    channels = np.flatnonzero(rng.random(n_channels) < 0.5).tolist()
    chunks = [slice(2 * slot, 2 * slot + 2) for slot in range(n_slots)]
    leaves = []
    for chunk in chunks:
        in_chunk = [
            channel for channel in channels if chunk.start <= channel < chunk.stop
        ]
        partials = None
        if in_chunk:
            partials = new_moment_partials((1, 1, 1, 5, 4), sky.dtype)
            sky_batch, _, valid, _ = _batch(sky, beam, in_chunk)
            accumulate_moments(partials, sky_batch, valid, weights[in_chunk])
        leaves.append(partials)
    merged = merge_moment_partials_in_tree_order(iter(leaves))
    if not channels:
        assert merged is None
        return
    integral, maximum = finalize_moments(merged, (1, 1, 1, 5, 4), sky.dtype)
    request = {
        "kind": "moments",
        "image": "cube",
        "mask": "",
        "channels": channels,
        "moments": [0, 8],
        "frequency_axis": axis,
        "rest_frequency": REST_FREQUENCY,
    }
    (maps,) = _port(sky, beam, [request], chunks=chunks)
    _assert_bitwise(integral[0, 0, 0], maps[0])
    _assert_bitwise_but_zero_sign(maximum[0, 0, 0].astype(np.float64), maps[8])


def test_tree_order_differs_from_a_left_fold():
    """The tree test checks the order of the merges: a left fold of the same
    partials adds moment 0 in another order and differs."""
    rng = np.random.default_rng(300)
    differs = 0
    for n_slots in range(3, 34):
        partials = []
        for _ in range(n_slots):
            plane = rng.normal(size=(1, 1, 1, 8, 8)) * 10.0 ** rng.uniform(-6, 6)
            partial = new_moment_partials((1, 1, 1, 8, 8), np.float32)
            accumulate_moments(
                partial, plane.astype(np.float32), np.ones(plane.shape, bool), [1.0]
            )
            partials.append(partial)
        copies = [{key: value.copy() for key, value in p.items()} for p in partials]
        tree = merge_moment_partials_in_tree_order(partials)
        fold = None
        for partial in copies:
            fold = merge_moment_partials(fold, partial)
        differs += not np.array_equal(tree["integrated"], fold["integrated"])
    assert differs > 0


def test_moment_partials_without_channels():
    map_shape = (1, 1, 2, 3, 4)
    partials = new_moment_partials(map_shape, np.float32)
    assert partials["integrated"].dtype == np.float64
    assert partials["maximum"].dtype == np.float32
    assert merge_moment_partials(None, None) is None
    assert merge_moment_partials(partials, None) is partials
    assert merge_moment_partials(None, partials) is partials
    assert merge_moment_partials_in_tree_order([]) is None
    assert merge_moment_partials_in_tree_order([None, None, None]) is None
    integral, maximum = finalize_moments(None, map_shape, np.float32)
    assert integral.dtype == np.float64 and maximum.dtype == np.float32
    assert np.isnan(integral).all() and np.isnan(maximum).all()
    sky = np.full((1, 1, 2, 3, 4), -2.5, dtype=np.float32)
    valid = np.zeros(sky.shape, dtype=bool)
    valid[0, 0, 1, 2, 3] = True
    accumulate_moments(partials, sky, valid, np.array([2.0]))
    integral, maximum = finalize_moments(partials, map_shape, np.float32)
    assert integral[0, 0, 1, 2, 3] == -5.0 and maximum[0, 0, 1, 2, 3] == -2.5
    assert np.isnan(integral).sum() == np.isnan(maximum).sum() == integral.size - 1
    assert partials["maximum"][0, 0, 0, 0, 0] == -np.inf


@pytest.mark.parametrize("shape", [(1, 3, 1, 37, 21), (2, 1, 2, 1300, 13)])
def test_pack_round_trip(shape):
    mask = np.random.default_rng(31).random(shape) < 0.3
    packed = pack_plane_masks(mask)
    assert packed.dtype == np.uint8
    assert packed.shape == shape[:-1] + (-(-shape[-1] // 8),)
    np.testing.assert_array_equal(unpack_plane_masks(packed, shape[-1]), mask)
    out = np.ones(shape, dtype=bool)
    assert unpack_plane_masks(packed, shape[-1], out=out) is out
    np.testing.assert_array_equal(out, mask)


def test_tile_occupancy_with_ragged_tiles():
    mask = np.random.default_rng(32).random((2, 2, 1, 37, 21)) < 0.004
    tiles = tile_occupancy(mask, (8, 5))
    assert tiles.shape == (2, 2, 1, 5, 5)
    for i, j in np.ndindex(5, 5):
        np.testing.assert_array_equal(
            tiles[..., i, j],
            mask[..., 8 * i : 8 * i + 8, 5 * j : 5 * j + 5].any(axis=(-2, -1)),
        )
    assert tile_occupancy(mask, (37, 21)).shape == (2, 2, 1, 1, 1)


def test_refusals():
    sky = np.zeros((1, 2, 1, 3, 4), dtype=np.float32)
    valid = np.ones(sky.shape, dtype=bool)
    region = np.ones((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="dimensions"):
        masked_channel_means(sky[0], valid[0], region)
    with pytest.raises(ValueError, match="valid has shape"):
        masked_channel_medians_and_mads(sky, valid[:, :1], region)
    with pytest.raises(ValueError, match="region has shape"):
        masked_channel_means(sky, valid, region.T)
    with pytest.raises(ValueError, match="spectrum_mask has shape"):
        masked_channel_means(sky, valid, region, valid[..., :2])
    with pytest.raises(ValueError, match="go together"):
        masked_channel_scaled_mads(sky, valid, region, primary_beam=sky)
    with pytest.raises(ValueError, match="go together"):
        masked_channel_scaled_mads(sky, valid, region, primary_beam_levels=ANNULUS)
    partials = new_moment_partials((1, 1, 1, 3, 4), np.float32)
    with pytest.raises(ValueError, match="3 channel weights for 2 planes"):
        accumulate_moments(partials, sky, valid, np.ones(3))
    with pytest.raises(ValueError, match="Moment partials"):
        accumulate_moments(
            new_moment_partials((1, 1, 1, 4, 3), np.float32), sky, valid, np.ones(2)
        )
    with pytest.raises(ValueError, match="limit_mask has shape"):
        first_round_channel_rows(sky, sky, valid[:, :1])
