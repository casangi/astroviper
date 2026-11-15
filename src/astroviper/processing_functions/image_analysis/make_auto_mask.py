"""Automatic CLEAN masks from the residual image (CASA ``auto-multithresh``).

The algorithm of Kepley et al. 2020, PASP 132, 024505, as ``tclean`` runs it
before every model update.
"""

import math

import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.special
import toolviper.utils.logger as logger

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs

AUTO_MASK_DEFAULTS = {
    "sidelobe_threshold": 3.0,
    "noise_threshold": 5.0,
    "low_noise_threshold": 1.5,
    "negative_threshold": 0.0,
    "smooth_factor": 1.0,
    "min_beam_fraction": 0.3,
    "cut_threshold": 0.01,
    "grow_iterations": 75,
    "grow_prune": True,
    "min_percent_change": -1.0,
    "fast_noise": True,
}

MAD_TO_SIGMA = 1.4826
CHAUVENET_MAX_ITERATIONS = 5
# Half width of the smoothing kernel in units of its major axis FWHM.
KERNEL_HALF_WIDTH = 3.0


def _chauvenet_sample(values):
    """``values`` without the outliers of Chauvenet's criterion.

    As casacore's ``ChauvenetCriterionStatistics`` with five iterations: the
    z-score of ``n`` points is the one at which ``n * erfc(z / sqrt(2)) = 0.5``.
    """
    for _ in range(CHAUVENET_MAX_ITERATIONS + 1):
        if values.size < 2:
            break
        mean, deviation = values.mean(), values.std(ddof=1)
        z_score = math.sqrt(2.0) * float(scipy.special.erfcinv(0.5 / values.size))
        kept = values[np.abs(values - mean) <= z_score * deviation]
        if kept.size == values.size:
            break
        values = kept
    return values


def _warn_when_unresolved_sources_are_pruned(sidelobe, params):
    """Warn when the sidelobe level alone prunes every unresolved source.

    A Gaussian holds ln(1 / f) beam areas above the fraction f of its peak, so
    an unresolved source passes pruning only for f below
    exp(-``min_beam_fraction``), whatever its signal to noise. Left out of the
    mask it stays the peak, and the sidelobe level stays above everything else.

    Parameters
    ----------
    sidelobe : numpy.ndarray
        Largest PSF sidelobe of every plane.
    params : dict
        Complete ``auto_mask_params``.
    """
    limit = math.exp(-params["min_beam_fraction"])
    fraction = params["sidelobe_threshold"] * sidelobe
    n_planes = np.count_nonzero(fraction > limit)
    if params["min_beam_fraction"] > 0 and n_planes:
        logger.warning(
            f"auto_mask_params: sidelobe_threshold times the largest PSF sidelobe "
            f"is up to {np.nanmax(fraction):.2f} in {n_planes} of {fraction.size} "
            f"planes, above exp(-min_beam_fraction) = {limit:.2f}. Pruning then "
            "removes every unresolved source, whatever its signal to noise. Lower "
            f"sidelobe_threshold below {limit / np.nanmax(sidelobe):.2f} or "
            f"min_beam_fraction below {-math.log(np.nanmax(fraction)):.2f}."
        )


def _bounding_box(mask, margin=0):
    """Slices around the pixels of ``mask``, ``margin`` wider, inside the plane."""
    box = []
    for axis, size in enumerate(mask.shape):
        pixels = np.flatnonzero(mask.any(axis=1 - axis))
        box.append(
            slice(max(pixels[0] - margin, 0), min(pixels[-1] + margin + 1, size))
        )
    return tuple(box)


def _location_and_noise(plane, support, previous_mask, fast_noise):
    """Median and MAD based noise of a residual plane.

    ``fast_noise`` takes every pixel of ``support``; otherwise those outside
    ``previous_mask`` or, when no such pixels or no mask exist, those Chauvenet's
    criterion leaves.
    """
    if fast_noise:
        sample = plane[support]
    else:
        outside = support & ~previous_mask
        if previous_mask.any() and outside.any():
            sample = plane[outside]
        else:
            sample = _chauvenet_sample(plane[support])
    location = float(np.median(sample, overwrite_input=True))
    sample -= location
    np.abs(sample, out=sample)
    noise = MAD_TO_SIGMA * float(np.median(sample, overwrite_input=True))
    return location, noise


def _plane_thresholds(peak, sidelobe, location, noise, params):
    """Threshold of the mask, of its growth and of negative features.

    Returns
    -------
    tuple
        The mask, growth and negative thresholds as floats, each including the
        median ``location``, and a bool: the noise sets the mask threshold.
    """
    sidelobe_level = sidelobe * params["sidelobe_threshold"] * peak
    noise_level = params["noise_threshold"] * noise
    return (
        max(sidelobe_level, noise_level) + location,
        max(sidelobe_level, params["low_noise_threshold"] * noise) + location,
        location - max(sidelobe_level, params["negative_threshold"] * noise),
        noise_level > sidelobe_level,
    )


def _prune(mask, min_pixels):
    """``mask`` without its 4-connected regions of fewer than ``min_pixels``."""
    if not mask.any():
        return mask
    box = _bounding_box(mask)
    labels, _ = scipy.ndimage.label(mask[box])
    keep = np.bincount(labels[mask[box]]) >= min_pixels
    keep[0] = False
    pruned = np.zeros_like(mask)
    pruned[box] = keep[labels]
    return pruned


def _convolve(box, kernel):
    """Convolution of a bool ``box``, padded to the FFT size, two spectra at most."""
    shape = [
        scipy.fft.next_fast_len(size + kernel.shape[0] - 1, real=True)
        for size in box.shape
    ]
    spectrum = scipy.fft.rfft2(box.astype(np.float64), shape)
    spectrum *= scipy.fft.rfft2(kernel, shape)
    return scipy.fft.irfft2(spectrum, shape)


def _smooth_and_cut(mask, kernel, cut_threshold):
    """``mask`` smoothed with ``kernel`` and cut at a fraction of its peak."""
    if not mask.any():
        return mask
    # Beyond the reach of the kernel the smoothed mask is 0, so only the box of
    # the mask is transformed.
    half_width = kernel.shape[0] // 2
    box = _bounding_box(mask)
    reach = _bounding_box(mask, half_width)
    smoothed = _convolve(mask[box], kernel)[
        tuple(
            slice(
                wide.start - narrow.start + half_width,
                wide.stop - narrow.start + half_width,
            )
            for wide, narrow in zip(reach, box, strict=True)
        )
    ]
    cut = np.zeros_like(mask)
    cut[reach] = smoothed > cut_threshold * smoothed.max()
    return cut


def _grow(mask, constraint, iterations):
    """``mask`` inside ``constraint``, dilated through ``constraint``, 4-connected."""
    grown = mask & constraint
    if grown.any():
        # A dilation stays within ``iterations`` pixels of where it starts.
        box = _bounding_box(grown, iterations)
        grown[box] = scipy.ndimage.binary_dilation(
            grown[box], iterations=iterations, mask=constraint[box]
        )
    return grown


def _update_plane_mask(
    plane, support, previous_mask, sidelobe, beam, delta, threshold_reached, params
):
    """The mask of one plane after one residual update.

    Parameters
    ----------
    plane : numpy.ndarray
        ``(l, m)`` residual.
    support : numpy.ndarray
        ``(l, m)`` bool: the finite pixels the mask may hold.
    previous_mask : numpy.ndarray
        ``(l, m)`` bool mask of the preceding residual update.
    sidelobe : float
        Largest PSF sidelobe.
    beam : numpy.ndarray
        ``[major, minor, pa]`` of the clean beam, radians.
    delta : float
        Pixel size, radians.
    threshold_reached : bool
        The preceding ``threshold_per_cycle`` of this plane was the absolute
        ``threshold``; only then does ``min_percent_change`` stop its mask.
    params : dict
        Complete ``auto_mask_params``.

    Returns
    -------
    mask : numpy.ndarray
        ``(l, m)`` bool: ``previous_mask`` with what this update adds.
    stop : bool
        Do not update the mask of this plane again.
    """
    from astroviper.processing_functions.imaging.restore import (
        _elliptical_gaussian_kernel,
    )

    major, minor, position_angle = (float(value) for value in beam)
    if (
        not np.isfinite([major, minor, position_angle]).all()
        or min(major, minor) <= 0
        or not support.any()
    ):
        return previous_mask, False

    location, noise = _location_and_noise(
        plane, support, previous_mask, params["fast_noise"]
    )
    mask_threshold, grow_threshold, negative_threshold, noise_sets_threshold = (
        _plane_thresholds(
            max(
                float(plane.max(where=support, initial=-np.inf)),
                -float(plane.min(where=support, initial=np.inf)),
            ),
            sidelobe,
            location,
            noise,
            params,
        )
    )

    major_pixels = params["smooth_factor"] * major / delta
    kernel_size = 2 * math.ceil(KERNEL_HALF_WIDTH * major_pixels) + 1
    kernel = _elliptical_gaussian_kernel(
        kernel_size,
        kernel_size,
        major_pixels,
        params["smooth_factor"] * minor / delta,
        position_angle,
        np.float64,
    )
    prune_pixels = (
        params["min_beam_fraction"]
        * math.pi
        * major
        * minor
        / (4.0 * math.log(2.0) * delta**2)
    )

    mask = plane > mask_threshold
    if params["min_beam_fraction"] > 0:
        mask = _prune(mask & support, prune_pixels)
    mask = _smooth_and_cut(mask, kernel, params["cut_threshold"])
    mask |= previous_mask

    if params["grow_iterations"] > 0 and previous_mask.any():
        grown = _grow(previous_mask, plane > grow_threshold, params["grow_iterations"])
        if params["min_beam_fraction"] > 0 and params["grow_prune"]:
            grown = _prune(grown, prune_pixels)
        mask |= _smooth_and_cut(grown, kernel, params["cut_threshold"])

    if params["negative_threshold"] > 0:
        mask |= _smooth_and_cut(
            plane < negative_threshold, kernel, params["cut_threshold"]
        )

    mask &= support
    n_pixels = int(np.count_nonzero(mask))
    n_previous = int(np.count_nonzero(previous_mask))
    change = abs(n_pixels - n_previous)
    fraction = params["min_percent_change"] / 100.0
    stop = noise_sets_threshold and (
        n_pixels == 0
        or (
            fraction >= 0
            and threshold_reached
            and (change == 0 or change < fraction * n_previous)
        )
    )
    return mask, stop


@shares_param_docs
def make_auto_mask(
    img_xds,
    auto_mask_params,
    primary_beam_limit=0.0,
    planes_stopped=None,
    threshold_reached=None,
    processing_function_threads=1,
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    image_data_group_out_modified=None,
    *,
    input_checked: bool = False,
):
    """Update the automatic CLEAN mask from the residual of every plane.

    Every plane gets the pixels above the larger of ``sidelobe_threshold`` times
    the largest PSF sidelobe times the peak residual (the sidelobe level) and
    ``noise_threshold`` times the noise, without the regions smaller than
    ``min_beam_fraction`` clean beams, smoothed with ``smooth_factor`` clean
    beams and cut at ``cut_threshold`` of the peak of the smoothed mask. A mask
    of the preceding residual update also grows through the pixels above the
    larger of the sidelobe level and ``low_noise_threshold`` times the noise,
    and with ``negative_threshold`` above 0 the mask takes the negative features
    too. A mask never loses pixels.

    Parameters
    ----------
    img_xds : xarray.Dataset
        Image with the ``sky``, ``beam_fit_params_point_spread_function`` and
        ``max_sidelobe_point_spread_function`` roles in the data group and
        optionally ``primary_beam`` and ``mask``, the mask of the preceding
        residual update. The mask is written to it as the ``mask`` of the
        output group.
    auto_mask_params : dict
        Keys of :data:`AUTO_MASK_DEFAULTS`; those left out take these defaults.
    primary_beam_limit : float, optional
        Fraction of the peak primary beam: the pixels below it and where the
        primary beam is 0 stay outside the statistics and the mask.
    planes_stopped : numpy.ndarray, optional
        ``(time, frequency, polarization)`` bool, the planes whose masks no
        longer change, as this function returned it after the preceding
        residual update.
    threshold_reached : numpy.ndarray, optional
        ``(time, frequency, polarization)`` bool, the planes whose preceding
        ``threshold_per_cycle`` was the absolute ``threshold``. Only then does
        ``min_percent_change`` stop a mask.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    image_data_group_in_name : str, optional
        Data group of the residual. Default ``"residual"``.
    image_data_group_out_name : str, optional
        Data group that gets the mask. Default ``"residual"``.
    image_data_group_out_modified : dict, optional
        Role to data-variable name of the output group. Default
        ``{"mask": "MASK"}``.
    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    numpy.ndarray
        ``(time, frequency, polarization)`` bool ``planes_stopped``, updated: a
        plane stops when the noise sets its threshold and its mask is empty or,
        with ``min_percent_change`` at or above 0 and ``threshold_reached``,
        did not change or changed by less than that percentage.
    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check._predicates import (
            check_auto_mask_params,
            check_primary_beam_limit,
        )

        check_auto_mask_params(auto_mask_params)
        check_primary_beam_limit(primary_beam_limit)

    params = {**AUTO_MASK_DEFAULTS, **auto_mask_params}
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {"mask": "MASK"}
    group, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=True,
    )
    plane_dims = ("time", "frequency", "polarization")
    sky = img_xds[group["sky"]]
    residual = sky.values
    planes_shape = residual.shape[:3]

    primary_beam = None
    primary_beam_name = group.get("primary_beam")
    if primary_beam_name is not None and primary_beam_name in img_xds:
        primary_beam = img_xds[primary_beam_name].values
        primary_beam_floor = primary_beam_limit * np.nanmax(primary_beam)

    mask_name = group.get("mask")
    if mask_name is None or mask_name not in img_xds:
        mask = np.zeros(residual.shape, dtype=bool)
    else:
        previous = img_xds[mask_name].values
        in_place = (
            mask_name == image_data_group_out["mask"] and previous.flags.writeable
        )
        mask = previous.astype(bool, copy=not in_place)

    beam = img_xds[group["beam_fit_params_point_spread_function"]]
    beam = beam.transpose(*plane_dims, ...).values
    sidelobe = img_xds[group["max_sidelobe_point_spread_function"]]
    sidelobe = sidelobe.transpose(*plane_dims).values
    delta = abs(float(img_xds.l[1] - img_xds.l[0]))

    if planes_stopped is None:
        _warn_when_unresolved_sources_are_pruned(sidelobe, params)
        planes_stopped = np.zeros(planes_shape, dtype=bool)
    else:
        planes_stopped = np.array(planes_stopped, dtype=bool)
    if threshold_reached is None:
        threshold_reached = np.zeros(planes_shape, dtype=bool)

    with scipy.fft.set_workers(processing_function_threads):
        for index in np.ndindex(planes_shape):
            if planes_stopped[index]:
                continue
            support = np.isfinite(residual[index])
            if primary_beam is not None:
                support &= primary_beam[index] >= primary_beam_floor
                support &= primary_beam[index] != 0
            mask[index], planes_stopped[index] = _update_plane_mask(
                residual[index],
                support,
                mask[index],
                float(sidelobe[index]),
                beam[index],
                delta,
                bool(threshold_reached[index]),
                params,
            )

    img_xds[image_data_group_out["mask"]] = (
        sky.dims,
        mask,
        {"description": "Automatic deconvolution mask.", "type": "mask"},
    )
    modify_data_groups_xds(
        img_xds,
        image_data_group_out_name,
        image_data_group_out,
        description="Automatic deconvolution mask updated.",
    )
    return planes_stopped
