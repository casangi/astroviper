"""Generate the CASA reference of ``test_primary_beam_annulus_statistics_casa.py``.

Run by hand where ``casatools`` imports (the committed file was made with
casatools 6.7.6-14); the test suite itself never imports CASA:

    cd /tmp && python <repository>/tests/component/data/generate_casa_primary_beam_annulus_reference.py

It writes ``casa_primary_beam_annulus_reference.npz`` next to this script: the
images of :func:`make_images` and what the CASA pipeline (commit 0091d599c)
measures on them with casatools:

- the primary beam levels of ``pblimits``
  (``pipeline/hif/heuristics/imageparams_base.py:1533-1569``);
- per plane (``axes=[0, 1]``), ``ia.statistics(robust=True,
  algorithm="chauvenet", maxiter=5, stretch=True)`` in the whole annulus,
  outside the clean mask of each plane, and outside the clean mask collapsed
  with ``max`` over polarizations and channels, with the mask expressions of
  ``pipeline/hif/heuristics/cleanbox.py:222-332`` and the levels written with
  ``%f``;
- the pipeline's median, minimum and maximum of the RMS over the continuum
  channels after its switch to the whole annulus (``cleanbox.py:292-371``).

The clipping of every plane keeps the same pixels at every pass for any
z-score in ``[z(n), z(n + 1))``, the values casacore's bisection can return,
so the reference does not depend on the z-score CASA finds: the first seed
from ``FIRST_SEED`` with that property is used.
"""

import math
import os

import numpy as np

PIPELINE_COMMIT = "0091d599c"
NOMINAL_ANNULUS_LIMITS = (0.2, 0.3)
MAX_ITERATIONS = 5
FIRST_SEED = 1
CASA_STATISTICS = ("npts", "rms", "sigma", "median", "medabsdevmed")
VARIANTS = ("whole", "plane_mask", "union_mask")


def _offsets(n_l, n_m, center):
    return np.meshgrid(
        np.arange(n_l) - center[0], np.arange(n_m) - center[1], indexing="ij"
    )


def _gaussian_beams(n_l, n_m, widths):
    """Primary beams ``(frequency, l, m)`` centred on ``(n_l // 2, n_m // 2)``."""
    l_offset, m_offset = _offsets(n_l, n_m, (n_l // 2, n_m // 2))
    radius2 = (l_offset**2 + m_offset**2)[np.newaxis]
    return np.exp(-radius2 / (2.0 * np.asarray(widths)[:, None, None] ** 2))


def _blob(n_l, n_m, center, width):
    l_offset, m_offset = _offsets(n_l, n_m, center)
    return np.exp(-(l_offset**2 + m_offset**2) / (2.0 * width**2))


def _disk(n_l, n_m, center, radius):
    l_offset, m_offset = _offsets(n_l, n_m, center)
    return l_offset**2 + m_offset**2 <= radius**2


def _noise(rng, shape, sigma):
    """Gaussian noise with a heavy-tailed (Student t, 3 degrees) part."""
    return sigma * (rng.normal(size=shape) + 0.5 * rng.standard_t(3, size=shape))


def _add_outliers(rng, sky, region, n_outliers, sigma):
    """Set ``n_outliers`` pixels of ``region`` of a plane to 6 to 10 ``sigma``."""
    pixels = np.flatnonzero(region)
    chosen = rng.choice(pixels, size=n_outliers, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_outliers)
    sky.flat[chosen] = signs * rng.uniform(6.0, 10.0, size=n_outliers) * sigma


def _nominal_images(rng):
    """Stokes I cube with an annulus inside the image: the nominal levels.

    Channel 6 has a zero primary beam (no annulus), a source of the annulus
    is masked in every channel, a second masked region of the annulus is in
    channel 3 only, and one annulus pixel of channel 2 has a primary beam of
    exactly ``float32(0.2)``.
    """
    n_chan, n_l, n_m = 8, 48, 48
    widths = 10.5 / (1.0 + 0.025 * (np.arange(n_chan) - n_chan // 2))
    beam = _gaussian_beams(n_l, n_m, widths).astype(np.float32)
    beam[6] = 0.0
    noise = 1e-3 * (1.0 + 0.1 * np.arange(n_chan))
    sky = _noise(rng, (n_chan, n_l, n_m), noise[:, None, None])
    source = (n_l // 2 + 17, n_m // 2)
    sky += 0.03 * _blob(n_l, n_m, source, 1.3)
    mask = np.zeros(sky.shape, dtype=bool)
    mask |= _disk(n_l, n_m, source, 4) | _disk(n_l, n_m, (n_l // 2, n_m // 2), 6)
    mask[3] |= _disk(n_l, n_m, (n_l // 2, n_m // 2 - 17), 4)
    for c in range(n_chan):
        annulus = (beam[c] > 0.2) & (beam[c] < 0.3) & ~mask[c]
        if annulus.any():
            _add_outliers(rng, sky[c], annulus, 4, noise[c])
    level = np.where((beam[2] > 0.2) & ~mask.any(axis=0), beam[2], np.inf)
    edge_pixel = np.unravel_index(np.argmin(level), level.shape)
    beam[2][edge_pixel] = np.float32(0.2)
    sky[2][edge_pixel] = 0.0
    return {
        "sky": sky[:, None].astype(np.float32),
        "primary_beam": beam[:, None],
        "mask": mask[:, None],
        "channel_indices": np.hstack([np.arange(0, 3), np.arange(2, 4), [5, 6, 7]]),
    }


def _stokes_images(rng):
    """I, Q, U, V cube with a clean mask that differs between the planes."""
    n_chan, n_pol, n_l, n_m = 3, 4, 32, 32
    widths = 7.0 / (1.0 + 0.03 * (np.arange(n_chan) - n_chan // 2))
    beam = np.repeat(_gaussian_beams(n_l, n_m, widths)[:, None], n_pol, axis=1)
    beam = beam.astype(np.float32)
    noise = np.array([1e-3, 6e-4, 6e-4, 6e-4])
    sky = _noise(rng, (n_chan, n_pol, n_l, n_m), noise[None, :, None, None])
    source = (n_l // 2 + 12, n_m // 2)
    blob = _blob(n_l, n_m, source, 1.2)
    sky[:, 0] += 0.02 * blob
    sky[:, 1] += 0.002 * blob
    mask = np.zeros(sky.shape, dtype=bool)
    mask[:, :2] |= _disk(n_l, n_m, source, 3)
    for c in range(n_chan):
        mask[c, 2] |= _disk(n_l, n_m, (n_l // 2 - 12 + c, n_m // 2), 3)
        mask[c, 3] |= _disk(n_l, n_m, (n_l // 2, n_m // 2 + 12 - c), 2)
        for p in range(n_pol):
            annulus = (beam[c, p] > 0.2) & (beam[c, p] < 0.3) & ~mask[c, p]
            _add_outliers(rng, sky[c, p], annulus, 2, noise[p])
    return {
        "sky": sky.astype(np.float32),
        "primary_beam": beam,
        "mask": mask,
        "channel_indices": np.arange(n_chan),
    }


def _edge_images(rng):
    """Small Stokes I image whose primary beam is above 0.2 at the edge.

    The levels come from the primary beam, and their ``%f`` rounding moves the
    pixels at the two sampled radii into the annulus. The clean masks of the
    channels cover different sectors of the annulus: each leaves most of it,
    their union only three narrow gaps.
    """
    n_chan, n_l, n_m = 3, 24, 24
    widths = 10.0 / (1.0 + 0.02 * (np.arange(n_chan) - n_chan // 2))
    beam = _gaussian_beams(n_l, n_m, widths).astype(np.float32)
    sky = _noise(rng, (n_chan, n_l, n_m), 1e-3)
    center = (n_l // 2, n_m // 2)
    sky += 0.05 * _blob(n_l, n_m, center, 1.5)
    l_offset, m_offset = _offsets(n_l, n_m, center)
    radius = np.hypot(l_offset, m_offset)
    angle = np.degrees(np.arctan2(m_offset, l_offset)) % 360.0
    ring = (radius >= 8) & (radius <= 13)
    mask = np.zeros(sky.shape, dtype=bool)
    for c in range(n_chan):
        mask[c] = _disk(n_l, n_m, center, 4)
        mask[c] |= ring & (angle >= 120 * c) & (angle < 120 * c + 110)
        annulus = (radius > 9.5) & (radius < 11.5) & ~mask[c]
        _add_outliers(rng, sky[c], annulus, 1, 1e-3)
    return {
        "sky": sky[:, None].astype(np.float32),
        "primary_beam": beam[:, None],
        "mask": mask[:, None],
        "channel_indices": np.arange(n_chan),
    }


def make_images(seed):
    """The images of the comparison (single source of truth).

    Arrays are on ``(frequency, polarization, l, m)``: float32 sky and primary
    beam, as CASA's ``ia.fromarray`` stores them, and a bool clean mask.
    """
    rng = np.random.default_rng(seed)
    return {
        "nominal": _nominal_images(rng),
        "stokes": _stokes_images(rng),
        "edge": _edge_images(rng),
    }


def numpy_pblimits(primary_beam):
    """``pblimits`` of the pipeline on an image without a pixel mask."""
    n_f, _, n_l, n_m = primary_beam.shape
    column = primary_beam[n_f // 2, 0, n_l // 2].astype(np.float64)
    edge = 0
    while column[edge] == 0.0 and edge < n_m // 2:
        edge += 1
    if column[edge] > 0.2:
        inner = max(edge, int(0.05 * n_m))
        return float(column[inner]), float(column[inner + int(0.05 * n_m)])
    return NOMINAL_ANNULUS_LIMITS


def selections(images, limits):
    """The pixels of every variant, as the pipeline's mask expressions select them."""
    lower, upper = (float(f"{limit:f}") for limit in limits)
    beam = images["primary_beam"].astype(np.float64)
    annulus = (beam > lower) & (beam < upper)
    mask = images["mask"]
    return {
        "whole": annulus,
        "plane_mask": annulus & ~mask,
        "union_mask": annulus & ~mask.any(axis=(0, 1)),
    }


def _exact_z_score(n_points):
    import scipy.special

    return math.sqrt(2.0) * float(scipy.special.erfcinv(0.5 / n_points))


def _casacore_passes(values, z_score):
    """Kept masks of casacore's Chauvenet ranges, each applied to all values."""
    kept = np.ones(values.size, dtype=bool)
    passes = []
    n_previous = -1
    for _ in range(MAX_ITERATIONS + 1):
        sample = values[kept]
        if sample.size == n_previous:
            break
        n_previous = sample.size
        deviation = sample.std(ddof=1) if sample.size > 1 else 0.0
        half_width = z_score(sample.size) * deviation if sample.size > 1 else 0.0
        mean = sample.mean() if sample.size else 0.0
        kept = (values >= mean - half_width) & (values <= mean + half_width)
        passes.append(kept)
    return passes


def _nested_kept(values):
    """Kept mask of AstroVIPER's clipping: each pass filters the previous one."""
    kept = np.arange(values.size)
    for _ in range(MAX_ITERATIONS + 1):
        if kept.size < 2:
            break
        sample = values[kept]
        deviation = sample.std(ddof=1)
        inside = np.abs(sample - sample.mean()) <= _exact_z_score(kept.size) * deviation
        if inside.all():
            break
        kept = kept[inside]
    result = np.zeros(values.size, dtype=bool)
    result[kept] = True
    return result


def clipping_is_z_insensitive(values):
    """Whether every pass keeps the same values for z-scores in ``[z(n), z(n + 1)]``."""
    if values.size < 2:
        return True
    low = _casacore_passes(values, lambda n: _exact_z_score(n) * (1.0 - 1e-9))
    high = _casacore_passes(values, lambda n: _exact_z_score(n + 1))
    return (
        len(low) == len(high)
        and all(np.array_equal(a, b) for a, b in zip(low, high, strict=True))
        and np.array_equal(low[-1], _nested_kept(values))
    )


def plane_values(images, selection, c, p):
    """Selected values of a plane in CASA's order (``l`` fastest)."""
    return images["sky"][c, p].T[selection[c, p].T].astype(np.float64)


def kept_counts(images, selection):
    """``(frequency, polarization)`` number of values the clipping keeps."""
    n_chan, n_pol = selection.shape[:2]
    counts = np.zeros((n_chan, n_pol), dtype=np.int64)
    for c in range(n_chan):
        for p in range(n_pol):
            values = plane_values(images, selection, c, p)
            if values.size:
                counts[c, p] = np.count_nonzero(
                    _casacore_passes(values, _exact_z_score)[-1]
                )
    return counts


def _z_insensitive(scenarios):
    for images in scenarios.values():
        limits = numpy_pblimits(images["primary_beam"])
        for selection in selections(images, limits).values():
            n_chan, n_pol = selection.shape[:2]
            for c in range(n_chan):
                for p in range(n_pol):
                    values = plane_values(images, selection, c, p)
                    if not clipping_is_z_insensitive(values):
                        return False
    return True


def main():
    import shutil
    import tempfile

    target = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "casa_primary_beam_annulus_reference.npz",
    )
    seed = FIRST_SEED
    scenarios = make_images(seed)
    while not _z_insensitive(scenarios):
        seed += 1
        scenarios = make_images(seed)

    work = tempfile.mkdtemp(prefix="casa_annulus_ref_")
    os.chdir(work)
    import casatools

    ia = casatools.image()

    def write(name, cube):
        ia.fromarray(
            outfile=name, pixels=np.transpose(cube, (2, 3, 1, 0)), overwrite=True
        )
        ia.close()

    def statistics(mask, axes):
        ia.open("sky.im")
        try:
            return ia.statistics(
                mask=mask,
                robust=True,
                axes=axes,
                algorithm="chauvenet",
                maxiter=MAX_ITERATIONS,
                stretch=True,
            )
        finally:
            ia.close()

    def pblimits():
        # imageparams_base.py:1533-1569 on pb.im, which has no pixel mask.
        pblimit_image, pblimit_cleanmask = NOMINAL_ANNULUS_LIMITS
        ia.open("pb.im")
        try:
            nx, ny, _, nf = ia.shape()
            if not ia.getchunk(
                [nx // 2, 0, 0, nf // 2], [nx // 2, 0, 0, nf // 2], getmask=True
            ).flatten()[0]:
                return pblimit_image, pblimit_cleanmask
            pb_edge = 0.0
            i_pb_edge = -1
            while pb_edge == 0.0 and i_pb_edge < ny // 2:
                i_pb_edge += 1
                pixel = [nx // 2, i_pb_edge, 0, nf // 2]
                pb_edge = ia.getchunk(pixel, pixel).flatten()[0]
            if pb_edge > 0.2:
                i_pb_image = max(i_pb_edge, int(0.05 * ny))
                pixel = [nx // 2, i_pb_image, 0, nf // 2]
                pblimit_image = ia.getchunk(pixel, pixel).flatten()[0]
                pixel = [nx // 2, i_pb_image + int(0.05 * ny), 0, nf // 2]
                pblimit_cleanmask = ia.getchunk(pixel, pixel).flatten()[0]
        finally:
            ia.close()
        return float(pblimit_image), float(pblimit_cleanmask)

    out = {
        "casatools_version": np.array(casatools.version_string()),
        "pipeline_commit": np.array(PIPELINE_COMMIT),
        "seed": np.array(seed),
    }
    for name, images in scenarios.items():
        n_chan, n_pol = images["sky"].shape[:2]
        write("sky.im", images["sky"])
        write("pb.im", images["primary_beam"])
        write("cleanmask.mask", images["mask"].astype(np.float32))
        ia.open("cleanmask.mask")
        ia.collapse(
            function="max",
            axes=[2, 3],
            outfile="cleanmask.mask.flattened",
            overwrite=True,
        ).done()
        ia.close()

        limits = pblimits()
        assert limits == numpy_pblimits(images["primary_beam"]), limits
        annulus = f'("pb.im" > {limits[0]:f}) && ("pb.im" < {limits[1]:f})'
        masks = {
            "whole": annulus,
            "plane_mask": '("cleanmask.mask" < 0.1) && ' + annulus,
            "union_mask": '("cleanmask.mask.flattened" < 0.1) && ' + annulus,
        }
        expected = selections(images, limits)
        for variant, mask in masks.items():
            result = statistics(mask, [0, 1])
            for key in CASA_STATISTICS:
                out[f"{name}_{variant}_{key}"] = np.reshape(
                    result[key], (n_pol, n_chan)
                ).T
            npts = out[f"{name}_{variant}_npts"]
            assert np.array_equal(npts, kept_counts(images, expected[variant])), (
                name,
                variant,
            )

        # cleanbox.py:292-371 with the continuum channels of the scenario.
        image_stats = statistics(masks["union_mask"], [0, 1, 2])
        image_stats_iquv = statistics(masks["union_mask"], [0, 1])
        full_annulus = bool(
            image_stats["npts"].shape == (0,) or np.median(image_stats["npts"]) < 10.0
        )
        if full_annulus:
            image_stats = statistics(masks["whole"], [0, 1, 2])
            image_stats_iquv = statistics(masks["whole"], [0, 1])
        cont_chan_indices = images["channel_indices"]
        if n_pol == 1:
            rms_vs_chan = image_stats["rms"][cont_chan_indices]
            summary = [np.median(rms_vs_chan), np.min(rms_vs_chan), np.max(rms_vs_chan)]
        else:
            rms_vs_chan_iquv = image_stats_iquv["rms"][:, cont_chan_indices]
            summary = [
                function(rms_vs_chan_iquv, axis=(1,))
                for function in (np.median, np.min, np.max)
            ]
        for key, value in zip(
            ("rms_median", "rms_min", "rms_max"), summary, strict=True
        ):
            out[f"{name}_summary_{key}"] = np.reshape(value, (n_pol,))
        out[f"{name}_summary_full_annulus"] = np.array(full_annulus)

        for key, value in images.items():
            out[f"{name}_{key}"] = value
        out[f"{name}_annulus_limits"] = np.array(limits)

    edge = scenarios["edge"]
    lower, upper = numpy_pblimits(edge["primary_beam"])
    middle = edge["primary_beam"][1, 0].astype(np.float64)
    rounded = selections(edge, (lower, upper))["whole"][1, 0]
    assert np.count_nonzero((middle > lower) & (middle < upper)) != rounded.sum()
    assert out["nominal_summary_full_annulus"] == np.array(False)
    assert out["stokes_summary_full_annulus"] == np.array(False)
    assert out["edge_summary_full_annulus"] == np.array(True)

    np.savez_compressed(target, **out)
    os.chdir(os.path.dirname(target))
    shutil.rmtree(work, ignore_errors=True)
    print("seed", seed, "wrote", target, os.path.getsize(target) // 1024, "KiB")


if __name__ == "__main__":
    main()
