"""Component test: cube and continuum imaging of one channel against a known truth.

The field is the 7 m array case (A3) of RADPS Memo 13a, simulated in the test:

* the ACA 7 m array (``aca.cycle8``: 10 antennas, baselines up to 45 m), a
  50 min track of 10 s integrations centred on the zenith transit, one
  15.6 MHz channel at 950 GHz, the four correlations of linear feeds and no
  noise: 13,500 rows;
* six polarized point sources on pixel centres of the 80 x 80 imaging grid
  (cells of 225.7 mas, five across the minor axis of the fitted beam with
  Briggs 0.5 weights; the image covers the radius where the primary beam
  power falls to 0.2, 34 pixels): 1 Jy on the l axis at 0.96 of that radius,
  where the primary beam is 0.23, 0.5 Jy at the phase centre and four fainter
  sources down to 1 mJy; Q, U and V are each 5 to 30 percent of I, of either
  sign (``SOURCES``, the memo's draw as a constant table);
* twice: with the w term, and without it (``uvw_params={"w_term": False}``).

Both imagers image it alike: Briggs 0.5 weights, which both assign to the
nearest density cell; the primary beam of the simulator (the obscured Airy
disk of 6.25 m with a 0.75 m blockage); the clean mask where the primary beam
is above 0.2; the Stokes planes I, Q, U and V in double precision; one thread;
the continuum in MFS with one Taylor term at the channel frequency.

The true model is known, so one residual update with it measures the error of
the residual update alone: for the one model update of each run, the model
update of both imagers (``model_update_cube_single_field``, which the
continuum's Hogbom model update calls as well) leaves the true model, the
truth times the exact primary beam (:func:`oracle_model_update`), and the
residual written is the residual of the true model.

* ``test_direct_fourier_transform_truth``: with the exact direct Fourier
  transform (``gridding_kernel_params={"type": "dft"}``, w term included)
  there is neither a kernel nor a w error. The imaging weights the PSF is made
  of, the PSF, the clean mask, the dirty image and the fitted beam of the two
  imagers agree to ``PRODUCT_RTOL`` of their peaks (measured on Linux x86-64:
  they are the same bit for bit), the primary beam to
  ``CONTINUUM_PRIMARY_BEAM_ATOL`` (the continuum interpolates its Airy disk in
  a table). The imaging (visibilities to image) is checked on its own: the
  dirty image and the PSF of both imagers at the source pixels and at seeded
  others are those of a NumPy direct Fourier transform of the store's
  visibilities with the imaging weights the PSF transform got
  (:func:`reference_image`), to ``REFERENCE_RTOL`` of the brightest apparent
  Stokes I flux and of the PSF's peak. The prediction (model to visibilities)
  is checked by the residual of the true model, which is at the rounding of
  the input, at most ``TRUTH_RESIDUAL_RTOL`` of the brightest apparent Stokes I
  flux in every Stokes plane. That residual alone does not check the imaging:
  it is the image of visibilities that are zero to rounding, whatever the
  imaging does.
* ``test_default_kernel_floor``: with the default kernel
  (``DEFAULT_GRIDDING_KERNEL_PARAMS``) the two imagers agree as above, and
  the residual of the true model minus that of the direct Fourier transform on
  the same data is the error of the gridded residual update: the floor of the
  kernel, and with the w term that of the two dimensional gridder, which does
  not model it. Its RMS inside the clean mask and away from the sources, per
  Stokes plane, is the measured value of ``KERNEL_FLOORS`` to within
  ``FLOOR_RTOL`` in both imagers, and away from the sources it is at most
  ``DYNAMIC_RANGE_GATE`` of the brightest apparent Stokes I flux in every
  Stokes plane: kernel errors of at most 0.1 sigma at a dynamic range of 1e4,
  the rule by which Memo 13a chose the default kernel.
* ``test_deconvolution_cube_against_continuum``: the deconvolution, whose
  model update the tests above replace. On the store without the w term both
  imagers clean with Hogbom, the default kernel and the image-sized PSF (the
  continuum has no other PSF subtraction) under ``DECONVOLUTION_CONTROLS``,
  every Stokes plane to ``DECONVOLUTION_THRESHOLD`` of its brightest apparent
  flux, and restore. Every plane stops on its threshold and the residual
  written is below it; both imagers make the same iterations in every model
  update, and their model, residual and restored image agree to
  ``DECONVOLUTION_RTOL`` of the brightest apparent Stokes I flux. Against the
  truth (:func:`deconvolution_errors`): the model flux at the sources, the
  model elsewhere and the restored image against the truth convolved with the
  fitted beam, computed here, are at most ``DECONVOLUTION_MARGINS`` times
  their measured values (``DECONVOLUTION_ERRORS``, 0.1 to 4.4 percent: errors
  of Hogbom on the PSF of the 7 m array, which fall with the threshold, far
  above the kernel floor).

The direct Fourier transform's residual is subtracted because it holds the
rounding of the input, which can differ between platforms: the source
directions the simulator is given round trip to the pixel centres only to
about half the spacing of doubles (4e-16 rad). Moving them by one unit in the
last place (ten seeded draws) raises the peak of that residual from 1.9e-11 to
up to 1.9e-10 of the brightest flux and moves the RMS of the default kernel's
residual of the true model by up to 7.4e-5 of itself, but the RMS of the
difference of the two residuals by at most 3.6e-10 of itself (1.2e-9 for up to
four units).

Caught: a change of the imaging weights, the PSF, the primary beam, the mask
or the dirty image in one imager only. In the direct Fourier transform's
imaging, by the NumPy transform: errors of the Stokes conversion, the cross
hand conjugation, the normalization, the sign of w and the n factor (in the
imaging only, w of the opposite sign moves the dirty image by 4.8e-6 of the
brightest flux and the PSF by 5.1e-6 of its peak, a missing n by 4.1e-10 and
3.3e-10, 330 times ``REFERENCE_RTOL``). In its prediction, by the residual of
the true model: errors of the Stokes conversion, the cross hand conjugation,
the normalization and the sign of w (in the prediction only, w of the
opposite sign gives a residual of 1.8e-6 of the brightest flux, 180 times
``TRUTH_RESIDUAL_RTOL``, a scale of 1 + 1e-6 one of 1.1e-6, conjugated cross
hands 0.49). Any change of the kernel, its table, its lookup or the padding
(the smallest one Memo 13a lists, oversampling 300 to 1000, moves the floors
by up to 0.84 percent without the w term and 0.15 percent with it, at least
1.5e3 times ``FLOOR_RTOL``). By the deconvolution (in mutated copies of the
package): a restoring beam 2 percent wider (a restored error 4.3 times its
measured value) or with a position angle of the opposite sign (32 times), a
continuum model increment scaled by 1 + 1e-6 or a threshold tolerance of 0.5
in the cube only (other iterations or cycles in the two imagers), a threshold
stop at 1.5 times the threshold (a residual written 1.29 times it) and a PSF
fraction of half the largest sidelobe (model flux error 3.5 times, spurious
flux 5.5 times). Not caught: several channels or Taylor terms, MVC, exact PSF
subtraction, the other deconvolvers, single precision, noise and the weights
it sets, and the 1 / n factor of the prediction (3e-10 of the brightest flux
here, below ``TRUTH_RESIDUAL_RTOL``; its unit tests cover it).

The builders and the analysis are plain functions so that the memo scripts can
call the same code.
"""

from __future__ import annotations

from unittest import mock

import dask
import numpy as np
import pytest
from xradio.image import load_image, make_empty_sky_image
from xradio.measurement_set import load_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from astroviper.processing_functions.imaging import model_update
from astroviper.processing_functions.imaging.gridders import (
    direct_fourier_transform_cpp,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    DEFAULT_GRIDDING_KERNEL_PARAMS,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_THRESHOLD,
)
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    THRESHOLD_TOLERANCE,
)
from astroviper.processing_functions.simulation.antenna_beams import (
    pack_beam_models,
    symmetric_beam_response,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import inverse_sin_project
from astroviper.utils.telescope_layout import read_telescope_layout

STOKES = ["I", "Q", "U", "V"]
LAYOUT = "aca.cycle8"
TELESCOPE = "aca"
DISH_DIAMETER = 6.25  # m, the obscured Airy disk of airy_disk_model("aca")
BLOCKAGE_DIAMETER = 0.75  # m
TIME_PARAMS = {
    "time_start": "2019-10-03T19:35:00.000",
    "time_delta": 10.0,
    "n_samples": 300,
}
FREQUENCY = 950e9
SPEED_OF_LIGHT = 299792458.0  # m/s
FREQUENCY_PARAMS = {
    "freq_start": FREQUENCY,
    "freq_delta": 15.625e6,
    "n_channels": 1,
    "channel_width": 15.625e6,
    "spectral_window_name": "A3_k1",
}
# the zenith over the array centre at mid track (RA, Dec in radians)
PHASE_CENTER = np.array([4.2618808927534415, -0.4011964523381274])
IMAGE_SIZE = [80, 80]
CELL_SIZE = np.array([-1.0942096267345813e-06, 1.0942096267345813e-06])
FFT_PADDING = 1.2
PRIMARY_BEAM_LIMIT = 0.2
IMAGING_WEIGHTS_PARAMS = {"weighting": "briggs", "robust": 0.5}
# pixel (l, m), Stokes I (Jy) and the Q, U and V fractions of I
SOURCES = [
    (7, 40, 1.0, -0.09693934206715836, 0.11417645255470553, 0.17366983179721285),
    (40, 40, 0.5, -0.21519788937104634, -0.2380282154190192, 0.23077446703491922),
    (25, 48, 0.001, 0.13510929987602005, 0.19938514712453903, -0.29264427812801813),
    (55, 66, 0.06691310059954052, -0.22667603511667722, 0.10425868386641811,
     0.08517507734877484),
    (55, 14, 0.016340733664249773, -0.22679937390681593, -0.2657226927925211,
     -0.1473814296278526),
    (40, 57, 0.028773528450070824, 0.07869014453610808, -0.18389586264934488,
     0.18200505682815582),
]  # fmt: skip
# one model update, which leaves the true model, and the residual update after it
ORACLE_CONTROLS = {
    "max_iter": 1,
    "max_cycles": 1,
    "threshold": 0.0,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.0,
    "max_iter_per_cycle": 1,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}
KEEP = [
    "sky_model",
    "sky_residual",
    "point_spread_function",
    "primary_beam",
    "beam_fit_params_point_spread_function",
]
# the far region: the mask beyond 3 fitted major axes of every source
FAR_RADIUS_BEAMS = 3.0
DFT = {"type": "dft"}
# the NumPy direct Fourier transform is evaluated at the source pixels and at
# this many seeded others
REFERENCE_PIXELS = 58
REFERENCE_SEED = 20261026
# Tolerances, with the values measured on Linux x86-64 (GCC 15.2, NumPy 2.5,
# FFTW through pyfftw).
# The two imagers' products, relative to their peaks: the same bit for bit.
PRODUCT_RTOL = 1e-12
# The continuum's primary beam (a 1e4 point table of the Airy disk) against the
# cube's, absolute: 9.1e-9.
CONTINUUM_PRIMARY_BEAM_ATOL = 1e-7
# The cube's primary beam at the sources against the simulator's: equal.
EXACT_PRIMARY_BEAM_ATOL = 1e-12
# The dirty image of both imagers at the reference pixels against the NumPy
# direct Fourier transform's, over the brightest apparent Stokes I flux, and
# their PSF against its, over its peak of 1: 7.8e-14 and 1.8e-14 with the w
# term, 8.1e-14 and 1.8e-14 without, the kernel's rounding (the NumPy
# transform is within 3.2e-16 of one in extended precision); with the source
# directions moved by up to four units in the last place (ten draws) up to
# 8.7e-14 and 1.8e-14.
REFERENCE_RTOL = 1e-12
# The peak in the mask of the DFT residual of the true model over the brightest
# apparent Stokes I flux, every plane: 1.9e-11; with the simulated source
# directions moved by one unit in the last place (ten draws) up to 1.9e-10,
# by up to four units 6.8e-10 (a margin of 15).
TRUTH_RESIDUAL_RTOL = 1e-8
# The kernel floors: more threads move them by 2e-14, scipy's FFT for
# pyfftw's by 5.1e-10, source directions moved by up to four units in the
# last place by 1.2e-9 (the floors without the DFT residual subtracted by
# 2.6e-4).
FLOOR_RTOL = 1e-6
DYNAMIC_RANGE_GATE = 1e-5
# RMS of the default kernel's residual of the true model minus the DFT's, in the
# mask and in the far region, over the brightest apparent flux of the plane;
# measured in both imagers (the same bit for bit) with the default kernel of
# RADPS Memo 13a, the least misfit kernel 9/300/linear at fft_padding 1.2 (its
# shipped table). Without the w term they are the floors of the memo's A3
# oracle runs, which did not subtract the DFT residual, to 1.5e-6.
KERNEL_FLOORS = {
    "no_w_term": {
        "I": {"rms_mask": 1.136631199e-07, "rms_far": 9.870115311e-08},
        "Q": {"rms_mask": 5.059796836e-08, "rms_far": 4.364251025e-08},
        "U": {"rms_mask": 5.484421317e-08, "rms_far": 4.689962098e-08},
        "V": {"rms_mask": 8.622940225e-08, "rms_far": 7.474301641e-08},
    },
    "w_term": {
        "I": {"rms_mask": 2.832981255e-07, "rms_far": 2.812413678e-07},
        "Q": {"rms_mask": 1.301449324e-07, "rms_far": 1.304552470e-07},
        "U": {"rms_mask": 1.374143750e-07, "rms_far": 1.369213155e-07},
        "V": {"rms_mask": 2.129672565e-07, "rms_far": 2.111950436e-07},
    },
}
# The deconvolution of test_deconvolution_cube_against_continuum: Hogbom with
# the image-sized PSF (exact_psf_subtraction False, the continuum's only PSF
# subtraction), every plane down to DECONVOLUTION_THRESHOLD of its brightest
# apparent flux. With psf_sidelobe_factor 1 a model update cleans down to the
# PSF's largest sidelobe, 0.41 of the peak residual (min_psf_fraction 0.2 does
# not act: 0.05 gives the same iterations). Measured: a PSF fraction of 0.2
# (0.1) leaves 2.6 to 26 (8.9 to 24) times the spurious flux (below) in 64 (89)
# cycles, and gain 0.3 1.3 to 3.7 times the errors in 41. The threshold 1e-4
# takes 36 cycles (2.0 s in the cube, 7.3 s in the continuum, one thread), 1e-5
# 131 cycles (6.3 s and 26 s); at 1e-6 I, U and V reach the 200th.
DECONVOLUTION_THRESHOLD = 1e-4
DECONVOLUTION_CONTROLS = {
    "max_iter": 100000,
    "max_cycles": 72,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.0,
    "min_psf_fraction": 0.2,
    "max_psf_fraction": 0.8,
    "max_iter_per_cycle": -1,
}
# The model, residual and restored image of the two imagers, over the brightest
# apparent Stokes I flux: 5.3e-16 (their iterations are the same).
DECONVOLUTION_RTOL = 1e-12
# Errors of the deconvolution against the truth, per Stokes plane (measured in
# both imagers; :func:`deconvolution_errors`): the model in the 3 x 3 pixels of
# a source minus its apparent flux (largest over the sources) and the restored
# image minus the truth convolved with the fitted beam (largest in the mask),
# over the brightest apparent flux of the plane, and the model outside the
# sources' pixels over their apparent flux. The data with the w term (another
# sequence of iterations, 35 cycles) move them by at most 1.3 percent, source
# directions moved by up to four units in the last place by at most 1.9e-7 (ten
# draws, the same iterations), scipy's FFT in place of pyfftw's by 5.6e-11.
DECONVOLUTION_ERRORS = {
    "I": {"model_flux": 2.910e-3, "spurious_flux": 3.859e-2, "restored": 3.394e-3},
    "Q": {"model_flux": 1.154e-3, "spurious_flux": 7.548e-3, "restored": 1.138e-3},
    "U": {"model_flux": 2.229e-3, "spurious_flux": 2.575e-2, "restored": 2.267e-3},
    "V": {"model_flux": 3.094e-3, "spurious_flux": 4.381e-2, "restored": 2.877e-3},
}
# The errors may rise by these factors (design G.2 of Memo 13a: 3 for the
# largest values, 2 for sums).
DECONVOLUTION_MARGINS = {"model_flux": 3.0, "spurious_flux": 2.0, "restored": 3.0}


def primary_beam_power(l, m):
    """Primary beam power at ``l, m``, as the simulator and the cube compute it."""
    (model,) = pack_beam_models([airy_disk_model(TELESCOPE)])
    return np.asarray(
        symmetric_beam_response(
            model, np.asarray(l), np.asarray(m), FREQUENCY, ipower=2
        ),
        dtype=np.float64,
    )


def build_sky():
    """The sources on their pixel centres: components for the simulator and the truth.

    Returns a dict with the simulator components, the pixels ``[n_source, 2]``,
    the Stokes fluxes ``[n_source, 4]`` (I, Q, U, V), the primary beam at the
    sources, the true model (Stokes, l, m) in Jy per pixel with the primary
    beam, ``f_ref``, the brightest apparent flux of every Stokes plane, and
    the image's ``l`` and ``m`` coordinates.
    """
    table = np.array(SOURCES, dtype=np.float64)
    pixels = table[:, :2].astype(int)
    stokes_i = table[:, 2]
    stokes = np.column_stack([stokes_i] + [table[:, k] * stokes_i for k in (3, 4, 5)])
    grid = make_empty_sky_image(
        PHASE_CENTER,
        IMAGE_SIZE,
        CELL_SIZE,
        np.array([FREQUENCY]),
        ["I"],
        [0.0],
        do_sky_coords=False,
    )
    lm = np.stack([grid.l.values[pixels[:, 0]], grid.m.values[pixels[:, 1]]], axis=-1)
    ra_dec = inverse_sin_project(PHASE_CENTER, lm)
    primary_beam = primary_beam_power(lm[:, 0], lm[:, 1])
    components = []
    for s, (i, q, u, v) in enumerate(stokes):
        # linear feeds: XX = I + Q, XY = U + iV, YX = U - iV, YY = I - Q
        flux = np.array([[i + q, u + 1j * v, u - 1j * v, i - q]])
        components.append(
            {"kind": "point", "flux": flux, "ra_dec": ra_dec[s], "name": f"source_{s}"}
        )
    true_model = np.zeros((len(STOKES), *IMAGE_SIZE))
    true_model[:, pixels[:, 0], pixels[:, 1]] = (stokes * primary_beam[:, None]).T
    apparent = np.abs(stokes) * primary_beam[:, None]
    return {
        "components": components,
        "pixels": pixels,
        "stokes": stokes,
        "primary_beam": primary_beam,
        "true_model": true_model,
        "f_ref": dict(zip(STOKES, apparent.max(axis=0), strict=True)),
        "l": grid.l.values,
        "m": grid.m.values,
    }


def simulate(work_dir, sky, w_term):
    """Simulate the noise-free field, with or without the w term; returns the store."""
    antenna_xds = read_telescope_layout(LAYOUT, telescope_name="ACA")
    ps_store = f"{work_dir}/a3_{'w_term' if w_term else 'no_w_term'}.ps.zarr"
    with dask.config.set(scheduler="synchronous"):
        distributed_applications.simulation.simulate_processing_set(
            ps_store=ps_store,
            antenna_xds=antenna_xds,
            time_params=TIME_PARAMS,
            frequency_params=FREQUENCY_PARAMS,
            polarization=["XX", "XY", "YX", "YY"],
            sky_components=sky["components"],
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[airy_disk_model(TELESCOPE)],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            uvw_params={"w_term": w_term},
            n_time_chunks=1,
            n_frequency_chunks=1,
            processing_function_threads=1,
            overwrite=True,
        )
    return ps_store


def oracle_model_update(true_model, captured):
    """A ``model_update_cube_single_field`` that leaves the true model.

    It runs the model update (which makes the clean mask), records the residual
    the first model update starts from (the dirty image), the PSF, the primary
    beam and the mask, and counts the model updates; then it replaces the model
    by ``true_model`` (Stokes, l, m), or with ``true_model`` None keeps the
    model the update made.
    """
    real = model_update.model_update_cube_single_field

    def model_update_with_the_true_model(
        img_xds, deconvolver, deconvolve_params, model_exists, **kwargs
    ):
        group_in = kwargs.get("image_data_group_in_name", "residual")
        group_out = kwargs.get("image_data_group_out_name", "model")
        dirty = img_xds[img_xds.attrs["data_groups"][group_in]["sky"]].values
        dirty = dirty[0, 0].astype(np.float64)
        result = real(img_xds, deconvolver, deconvolve_params, model_exists, **kwargs)
        roles = img_xds.attrs["data_groups"][group_in]
        labels = [str(label) for label in img_xds.polarization.values]
        assert labels == STOKES, labels

        def plane(role, dtype=np.float64):
            return img_xds[roles[role]].values[0, 0].astype(dtype)

        if "mask" not in captured:
            captured.update(
                dirty=dirty,
                point_spread_function=plane("point_spread_function"),
                primary_beam=np.nan_to_num(plane("primary_beam")),
                mask=plane("mask", bool),
            )
        captured["n_model_updates"] = captured.get("n_model_updates", 0) + 1
        if true_model is None:
            return result
        model_name = img_xds.attrs["data_groups"][group_out]["sky"]
        model = img_xds[model_name]
        assert model.dims == ("time", "frequency", "polarization", "l", "m"), model.dims
        data = np.empty(model.shape, dtype=model.dtype)
        data[...] = true_model
        img_xds[model_name] = model.copy(data=data)
        return result

    return model_update_with_the_true_model


def psf_weights_spy(captured):
    """``dft_image_uv_sampling`` that records the imaging weights the PSF is made of."""
    real = direct_fourier_transform_cpp.dft_image_uv_sampling

    def dft_image_uv_sampling(*args, **kwargs):
        weights = np.array(args[7], dtype=np.float64)
        captured.setdefault("psf_weights", []).append(weights)
        return real(*args, **kwargs)

    return dft_image_uv_sampling


def image(
    work_dir,
    ps_store,
    imager,
    sky,
    gridding_kernel_params,
    tag,
    controls=None,
    **overrides,
):
    """One cube or continuum run.

    With ``controls`` None its one model update leaves the true model; otherwise
    the imager deconvolves under the cycle controls ``controls`` and restores.
    Returns the products the first model update got (dirty image, PSF, primary
    beam, mask), the written residual, model, PSF, primary beam and fitted beam
    (and with ``controls`` the restored image), all as (Stokes, ...) arrays of
    the one channel or Taylor term, with the DFT the imaging weights the PSF is
    made of, and the number of model updates; with ``controls`` also, per
    Stokes plane, the imaging stop code and the iterations of every model
    update. ``overrides`` replace arguments of the imager call.
    """
    deconvolve = controls is not None
    controls = dict(controls if deconvolve else ORACLE_CONTROLS)
    image_store = f"{work_dir}/{tag}_{imager}.img.zarr"
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": PHASE_CENTER,
        "frequency_coords": [FREQUENCY],
        "polarization_coords": STOKES,
        "time_coords": [0],
        "fft_padding": FFT_PADDING,
    }
    arguments = {
        "ps_store": ps_store,
        "image_store": image_store,
        "gridder": "anti_alias",
        "deconvolver": "hogbom",
        "instrument_polarization_basis": "linear",
        "scan_intents": ["OBSERVE_TARGET#ON_SOURCE"],
        "image_data_variables_keep": KEEP + ["sky_restored"] if deconvolve else KEEP,
        "processing_set_data_group_name": "base",
        "single_precision_image": False,
        "processing_function_threads": 1,
        "n_mapping_parallelism": {"frequency": 1},
        "restore": deconvolve,
        "primary_beam_correction": False,
        "gridding_kernel_params": dict(gridding_kernel_params),
        "overwrite": True,
    }
    if imager == "cube":
        driver = image_cube_single_field
        arguments.update(
            image_params=image_params,
            imaging_weights_params=dict(IMAGING_WEIGHTS_PARAMS),
            cycle_control_params={**controls, "primary_beam_limit": PRIMARY_BEAM_LIMIT},
            primary_beam_params={"telescope": TELESCOPE},
        )
    else:
        driver = image_continuum_single_field
        arguments.update(
            image_params={
                **image_params,
                "n_taylor_terms": 1,
                "reference_frequency": FREQUENCY,
                "primary_beam_model": "airy",
                "list_dish_diameters": [DISH_DIAMETER],
                "list_blockage_diameters": [BLOCKAGE_DIAMETER],
            },
            imaging_weights_params={
                **IMAGING_WEIGHTS_PARAMS,
                "weighting_scope": "local",
            },
            cycle_control_params=controls,
            continuum_mode="mfs",
            primary_beam_limit=PRIMARY_BEAM_LIMIT,
            visibility_memory_mode="recompute",
        )
    arguments.update(overrides)
    captured = {}
    with (
        dask.config.set(scheduler="synchronous"),
        mock.patch.object(
            model_update,
            "model_update_cube_single_field",
            oracle_model_update(None if deconvolve else sky["true_model"], captured),
        ),
        mock.patch.object(
            direct_fourier_transform_cpp,
            "dft_image_uv_sampling",
            psf_weights_spy(captured),
        ),
    ):
        result = driver(**arguments)
    if not deconvolve:
        assert captured.get("n_model_updates") == 1, captured.get("n_model_updates")
    img = load_image(image_store)

    def planes(name):
        data = img[name]
        other = {dim: 0 for dim in data.dims if dim not in ("polarization", "l", "m")}
        return (
            data.isel(other)
            .transpose("polarization", "l", "m")
            .values.astype(np.float64)
        )

    beam = img["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]
    (label_dim,) = (dim for dim in beam.dims if dim.startswith("beam_param"))
    other = {dim: 0 for dim in beam.dims if dim not in ("polarization", label_dim)}
    captured.update(
        residual=planes("SKY_RESIDUAL"),
        model=planes("SKY_MODEL"),
        written_point_spread_function=planes("POINT_SPREAD_FUNCTION"),
        written_primary_beam=np.nan_to_num(planes("PRIMARY_BEAM")),
        beam=beam.isel(other).transpose("polarization", label_dim).values,
    )
    if not deconvolve:
        return captured
    # the cube's stop codes are in its convergence record, the continuum's in the
    # statistics of the residual its last stop test checked
    final = result["residual_statistics" if imager == "continuum" else "deconvolution"]
    n_model_updates = captured["n_model_updates"]
    captured.update(
        restored=planes("SKY_RESTORED"),
        stop_code={
            STOKES[key.pol]: int(fields["stop_code"].imaging)
            for key, fields in final.data.items()
        },
        iterations={
            STOKES[key.pol]: np.atleast_1d(fields["iter_done"])[
                :n_model_updates
            ].tolist()
            for key, fields in result["deconvolution"].data.items()
        },
    )
    return captured


def far_region(mask, pixels, beam_major):
    """The mask beyond ``FAR_RADIUS_BEAMS`` fitted major axes of every source."""
    radius = FAR_RADIUS_BEAMS * beam_major / abs(CELL_SIZE[0])
    ii, jj = np.indices(mask.shape)
    far = mask.copy()
    for i, j in pixels:
        far &= (ii - i) ** 2 + (jj - j) ** 2 > radius**2
    return far


def floors(residual, mask, far, f_ref):
    """Per Stokes plane: the RMS in the mask and in the far region over ``f_ref``
    of the plane, the far RMS and the peak in the mask over ``f_ref`` of Stokes I."""
    out = {}
    for k, name in enumerate(STOKES):
        plane = residual[k]
        out[name] = {
            "rms_mask": float(np.sqrt(np.mean(plane[mask[k]] ** 2)) / f_ref[name]),
            "rms_far": float(np.sqrt(np.mean(plane[far] ** 2)) / f_ref[name]),
            "rms_far_over_i": float(np.sqrt(np.mean(plane[far] ** 2)) / f_ref["I"]),
            "peak_mask_over_i": float(np.abs(plane[mask[k]]).max() / f_ref["I"]),
        }
    return out


def relative_difference(a, b):
    """The largest difference of two arrays over the largest magnitude of ``b``."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def reference_pixels(sky):
    """The source pixels and ``REFERENCE_PIXELS`` seeded others, ``[n, 2]``."""
    rng = np.random.default_rng(REFERENCE_SEED)
    seeded = rng.integers(0, IMAGE_SIZE, size=(REFERENCE_PIXELS, 2))
    return np.concatenate([sky["pixels"], seeded])


def reference_image(ps_store, weights, sky, pixels):
    """The dirty image and the PSF at ``pixels`` by a direct Fourier transform in NumPy.

    Per correlation ``n sum W V exp(-2 pi i (u l + v m + w (n - 1))) / sum W``
    over the samples of the store, with its ``VISIBILITY`` and ``UVW`` (``u,
    v, w`` in wavelengths) and ``W`` the imaging weights ``weights`` (time,
    baseline, frequency, correlation); the dirty image is converted to Stokes
    by the inverse of the correlations :func:`build_sky` simulates, and the
    PSF, the transform of unit visibilities, has one plane per correlation.
    So that the reference rounds far less than the kernel it checks, on any
    platform, ``n - 1`` is computed as ``-(l**2 + m**2) / (1 + n)``, which
    does not cancel, and every sum runs over the last, contiguous axis, where
    NumPy sums pairwise. Returns the dirty image and the PSF, each (plane,
    pixel).
    """
    (ms_xdt,) = load_processing_set(ps_store).values()
    ms_xds = ms_xdt.ds
    labels = [str(label) for label in ms_xds.polarization.values]
    assert labels == ["XX", "XY", "YX", "YY"], labels
    assert not ms_xds.FLAG.values.any()
    visibility = ms_xds.VISIBILITY.values
    assert visibility.shape == weights.shape, (visibility.shape, weights.shape)
    uvw = ms_xds.UVW.values[:, :, None, :] * (
        ms_xds.frequency.values[None, None, :, None] / SPEED_OF_LIGHT
    )
    u, v, w = uvw.reshape(-1, 3).T
    l, m = sky["l"][pixels[:, 0]], sky["m"][pixels[:, 1]]
    n = np.sqrt(1.0 - l**2 - m**2)
    n_minus_one = -(l**2 + m**2) / (1.0 + n)
    phase = 2 * np.pi * (np.outer(l, u) + np.outer(m, v) + np.outer(n_minus_one, w))
    transform = np.exp(-1j * phase)  # (pixel, sample)
    # (correlation, sample)
    weights = np.ascontiguousarray(weights.reshape(-1, 4).T)
    visibility = np.ascontiguousarray(visibility.reshape(-1, 4).T)
    sum_of_weights = weights.sum(axis=-1)
    xx, xy, yx, yy = (
        n * (weights[c] * visibility[c] * transform).sum(axis=-1) / sum_of_weights[c]
        for c in range(4)
    )
    point_spread_function = np.stack(
        [
            n * (weights[c] * transform).sum(axis=-1).real / sum_of_weights[c]
            for c in range(4)
        ]
    )
    # the inverse of XX = I + Q, XY = U + iV, YX = U - iV, YY = I - Q
    dirty = np.stack([(xx + yy) / 2, (xx - yy) / 2, (xy + yx) / 2, (xy - yx) / 2j])
    return dirty.real, point_spread_function


def clean_beam_truth(sky, beam):
    """The true model (Stokes, l, m) convolved with the clean beam of every plane.

    The clean beam of plane ``k`` is the elliptical Gaussian of peak 1 with the
    fitted beam ``beam[k]``, ``[major, minor, position angle]`` (FWHM and
    angle in radians, the angle from the m axis towards the l axis), computed
    here at the pixel offsets from every source, wrapped around the image as
    in the restore's circular convolution (offsets ``-n // 2`` to
    ``n // 2 - 1``).
    """
    out = np.zeros_like(sky["true_model"])
    for k, (major, minor, position_angle) in enumerate(beam):
        major, minor = major / abs(CELL_SIZE[0]), minor / abs(CELL_SIZE[1])
        for i, j in sky["pixels"]:
            di, dj = (
                (np.arange(n) - centre + n // 2) % n - n // 2
                for n, centre in zip(IMAGE_SIZE, (i, j), strict=True)
            )
            along = di[:, None] * np.sin(position_angle) - dj * np.cos(position_angle)
            across = di[:, None] * np.cos(position_angle) + dj * np.sin(position_angle)
            exponent = (along / major) ** 2 + (across / minor) ** 2
            out[k] += sky["true_model"][k, i, j] * np.exp(-4 * np.log(2) * exponent)
    return out


def deconvolution_errors(run, sky, threshold):
    """Per Stokes plane, the errors of a deconvolved run against the truth.

    ``peak_residual``: the peak of the written residual in the mask over the
    plane's threshold; ``model_flux``: the largest difference over the sources
    between the model in the 3 x 3 pixels of a source and its apparent flux;
    ``restored``: the largest difference in the mask between the restored image
    and :func:`clean_beam_truth`, both over the brightest apparent flux of the
    plane; ``spurious_flux``: the absolute model outside the sources' pixels
    over the absolute apparent flux of the sources.
    """
    pixels = sky["pixels"]
    boxes = [(slice(i - 1, i + 2), slice(j - 1, j + 2)) for i, j in pixels]
    outside = np.ones(IMAGE_SIZE, bool)
    for box in boxes:
        outside[box] = False
    restored_truth = clean_beam_truth(sky, run["beam"])
    out = {}
    for k, name in enumerate(STOKES):
        mask, model, f_ref = run["mask"][k], run["model"][k], sky["f_ref"][name]
        apparent = sky["true_model"][k][pixels[:, 0], pixels[:, 1]]
        model_flux = np.array([model[box].sum() for box in boxes])
        restored = run["restored"][k] - restored_truth[k]
        out[name] = {
            "peak_residual": float(
                np.abs(run["residual"][k][mask]).max() / threshold[k]
            ),
            "model_flux": float(np.abs(model_flux - apparent).max() / f_ref),
            "restored": float(np.abs(restored[mask]).max() / f_ref),
            "spurious_flux": float(
                np.abs(model[outside]).sum() / np.abs(apparent).sum()
            ),
        }
    return out


def run_pair(work_dir, field, w_term, gridding_kernel_params, tag):
    """The cube and the continuum run of one store and kernel, and the far region."""
    store, sky = field["stores"][w_term], field["sky"]
    runs = {
        imager: image(work_dir, store, imager, sky, gridding_kernel_params, tag)
        for imager in ("cube", "continuum")
    }
    cube = runs["cube"]
    far = far_region(cube["mask"][0], field["sky"]["pixels"], cube["beam"][0, 0])
    return runs, far


def assert_same_products(cube, continuum, sky):
    """The PSF, the mask, the dirty image, the fitted beam and the primary beam
    of the two imagers agree, and the true model went into both."""
    for name in ("point_spread_function", "written_point_spread_function", "dirty"):
        assert relative_difference(continuum[name], cube[name]) <= PRODUCT_RTOL, name
    assert np.array_equal(continuum["mask"], cube["mask"])
    assert relative_difference(continuum["beam"], cube["beam"]) <= PRODUCT_RTOL
    # the cube's primary beam is the simulator's; the continuum's is a table of it
    pixels = tuple(sky["pixels"].T)
    at_sources = cube["written_primary_beam"][0][pixels]
    difference = np.abs(at_sources - sky["primary_beam"]).max()
    assert difference <= EXACT_PRIMARY_BEAM_ATOL, difference
    for name in ("primary_beam", "written_primary_beam"):
        difference = np.abs(continuum[name] - cube[name]).max()
        assert difference <= CONTINUUM_PRIMARY_BEAM_ATOL, (name, difference)
    # the cube's model went through its basis round trip (measured: 2.8e-17)
    for run in (cube, continuum):
        difference = np.abs(run["model"] - sky["true_model"]).max()
        assert difference <= PRODUCT_RTOL * sky["f_ref"]["I"], difference


@pytest.fixture(scope="module")
def field(tmp_path_factory):
    """The sky and its two stores, with and without the w term."""
    work_dir = tmp_path_factory.mktemp("single_channel_a3")
    sky = build_sky()
    stores = {w_term: simulate(work_dir, sky, w_term) for w_term in (True, False)}
    return {"sky": sky, "stores": stores}


@pytest.fixture(scope="module", params=[True, False], ids=["w_term", "no_w_term"])
def dft_pair(request, field, tmp_path_factory):
    """Both imagers with the direct Fourier transform on one store, for both tests."""
    work_dir = str(tmp_path_factory.mktemp("dft"))
    runs, far = run_pair(work_dir, field, request.param, DFT, "dft")
    return {"w_term": request.param, "runs": runs, "far": far}


def test_direct_fourier_transform_truth(field, dft_pair):
    sky = field["sky"]
    cube, continuum = dft_pair["runs"]["cube"], dft_pair["runs"]["continuum"]
    # one call of the PSF transform per imager, the same imaging weights
    assert len(cube["psf_weights"]) == len(continuum["psf_weights"]) == 1
    psf_weights = (continuum["psf_weights"][0], cube["psf_weights"][0])
    assert relative_difference(*psf_weights) <= PRODUCT_RTOL
    assert_same_products(cube, continuum, sky)
    pixels = reference_pixels(sky)
    at_pixels = (slice(None), pixels[:, 0], pixels[:, 1])
    store = field["stores"][dft_pair["w_term"]]
    weights = cube["psf_weights"][0]
    dirty, point_spread_function = reference_image(store, weights, sky, pixels)
    for name, run in dft_pair["runs"].items():
        # the imaging: the dirty image and the PSF are the NumPy transform's
        difference = run["dirty"][at_pixels] - dirty
        errors = {"dirty": float(np.abs(difference).max() / sky["f_ref"]["I"])}
        for psf_name in ("point_spread_function", "written_point_spread_function"):
            difference = run[psf_name][at_pixels] - point_spread_function
            errors[psf_name] = float(np.abs(difference).max())
        # the prediction: the residual of the true model is at the rounding of
        # the input
        measured = floors(run["residual"], run["mask"], dft_pair["far"], sky["f_ref"])
        peaks = {p: measured[p]["peak_mask_over_i"] for p in STOKES}
        print(
            f"DFT, w term {dft_pair['w_term']}, {name}: against the NumPy transform "
            f"{errors}, residual peaks {peaks}"
        )
        assert max(errors.values()) <= REFERENCE_RTOL, (name, errors)
        assert max(peaks.values()) <= TRUTH_RESIDUAL_RTOL, (name, peaks)


def test_default_kernel_floor(tmp_path, field, dft_pair):
    sky, w_term = field["sky"], dft_pair["w_term"]
    runs, far = run_pair(
        str(tmp_path), field, w_term, DEFAULT_GRIDDING_KERNEL_PARAMS, "default"
    )
    assert_same_products(runs["cube"], runs["continuum"], sky)
    expected = KERNEL_FLOORS["w_term" if w_term else "no_w_term"]
    for name, run in runs.items():
        error = run["residual"] - dft_pair["runs"][name]["residual"]
        measured = floors(error, run["mask"], far, sky["f_ref"])
        print(f"default kernel, w term {w_term}, {name}: floors {measured}")
        for p in STOKES:
            for metric in ("rms_mask", "rms_far"):
                value = measured[p][metric]
                change = value / expected[p][metric] - 1.0
                assert abs(change) <= FLOOR_RTOL, (name, p, metric, value, change)
            gate = measured[p]["rms_far_over_i"]
            assert gate <= DYNAMIC_RANGE_GATE, (name, p, gate)


def test_deconvolution_cube_against_continuum(tmp_path, field):
    sky = field["sky"]
    threshold = [DECONVOLUTION_THRESHOLD * sky["f_ref"][p] for p in STOKES]
    controls = {**DECONVOLUTION_CONTROLS, "threshold": threshold}
    runs = {
        imager: image(
            str(tmp_path),
            field["stores"][False],
            imager,
            sky,
            DEFAULT_GRIDDING_KERNEL_PARAMS,
            "deconvolution",
            controls=controls,
        )
        for imager in ("cube", "continuum")
    }
    cube, continuum = runs["cube"], runs["continuum"]
    # the same cycles: the same iterations in every model update of every plane
    assert continuum["n_model_updates"] == cube["n_model_updates"]
    assert continuum["iterations"] == cube["iterations"]
    for name in ("model", "residual", "restored"):
        difference = np.abs(continuum[name] - cube[name]).max() / sky["f_ref"]["I"]
        assert difference <= DECONVOLUTION_RTOL, (name, difference)
    for name, run in runs.items():
        errors = deconvolution_errors(run, sky, threshold)
        print(
            f"deconvolution, {name}: {run['n_model_updates']} model updates, "
            f"stop codes {run['stop_code']}, errors {errors}"
        )
        # every plane stops on its threshold, and the residual written is below it
        assert run["stop_code"] == dict.fromkeys(STOKES, IMAGING_THRESHOLD), name
        for p in STOKES:
            assert errors[p]["peak_residual"] <= 1.0 + THRESHOLD_TOLERANCE, (name, p)
            for metric, margin in DECONVOLUTION_MARGINS.items():
                bound = margin * DECONVOLUTION_ERRORS[p][metric]
                assert errors[p][metric] <= bound, (name, p, metric, errors[p])
