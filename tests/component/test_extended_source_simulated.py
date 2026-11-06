"""Coral and Hogbom on a simulated extended source whose truth is known exactly,
each imaged by the cube imager and by the continuum imager (MFS with one
Taylor term), which must give the same results: the simulated visibilities, the
floor of the gridding kernel and the deconvolution. Coral is Asp with its
defaults ``asp_residual_components`` and ``asp_scale_significance``.

No downloads. A Cygnus A-like source (core, two lobes with hot spots, a jet and
a faint bridge, 19.8 Jy) is made on the 128 x 128 imaging grid of 6 arcsec
cells and simulated through the Airy beam of a 24.5 m dish
(``airy_disk_model("vla_nvss")``) for a 6 h VLA D track of 30 min
integrations at 3 GHz (``RR``, ``LL``) with the exact direct Fourier transform
of that image and no w term, so that the imagers can represent the truth
exactly: once without noise and once with thermal noise (seed 5; the noise of
the image, sigma, is 1.27e-4 Jy by the weights).

* ``test_simulated_visibilities``: 300 seeded visibilities without noise are
  ``sum M exp(+2 pi i (u l + v m)) / n`` of the sky times the beam ``M`` to
  1e-12 of its flux (measured 6e-15; the opposite sign of the phase is off by
  0.34, a missing ``1 / n`` by 3.5e-7), ``RR`` equals ``LL`` and w is 0. The
  noise of ``(RR - LL) / 2`` matches the weights to 3 percent (measured 0.998;
  0.991 to 1.010 over eight other seeds), with a mean within four standard
  errors of 0.
* ``test_gridding_kernel_floor``: without noise, the dirty image against that
  of the exact transform (``gridding_kernel_params={"type": "dft"}``) and the
  residual of the true model (the model update leaves the truth, so one
  residual update measures the kernel alone), over the peak of the exact
  dirty image, and the PSF against the exact one (its peak is 1), in both
  imagers, which make the same images bit for bit. ``KERNEL_GATES`` lie
  between the default kernel and the least misfit kernel of support 7, which
  is run as well and must exceed every gate: the test shows its own
  sensitivity. The deconvolution cannot show it: with the support-7 kernel
  the measures of the Coral tests stay inside their limits (the rms of F and
  of the residual change by less than 0.5 percent).
* ``test_recovers_the_simulated_source`` (with a box clean mask around the
  source) and ``test_without_a_clean_mask``: Coral and Hogbom clean the noisy
  data to ``THRESHOLD``, about three sigma, in both imagers. ``LIMITS`` bound
  the errors against the truth (``F`` is the model error at the clean beam
  over the peak of the truth there; the peak flux that of the restored image),
  the residual against sigma, and the negative model flux, the model outside
  the mask (without the mask: away from the source), the iterations and the
  model updates, which catch Asp without its residual components even where
  the fidelity changes little (it needs twice the iterations and three to
  four times the negative flux, while its rms F rises by 4 to 12 percent;
  ``test_coral_stages`` catches Asp without its significance stage). Hogbom's
  iterations have a lower limit too: a kernel that subtracts too much (a PSF
  5 percent too large) stops early. Coral's rms F must be below 0.6 of
  Hogbom's. Stokes I stops on its threshold, and every model update reports a
  stop on ``threshold_per_cycle``, never below the threshold, or on its limit
  of ``max_iter_per_cycle`` iterations. The restored image is the model at
  the clean beam plus the residual.
* ``test_coral_stages`` and ``test_hogbom_components``: every model update of
  Coral starts with its residual-shaped components (positive), and Coral has
  Asp Gaussians outside its significance stage, significance components (only
  with the mask: their noise is measured outside it) and Hogbom runs; its
  components sum to the model. Hogbom subtracts points only, one per
  iteration at a residual peak above ``threshold_per_cycle``, which make up
  the model pixel for pixel.
* ``test_cube_and_continuum_agree``: both imagers make the same model
  updates (iterations and, where recorded, components) and the same model,
  residual and restored image, bit for bit, in Stokes I and V. Stokes V is
  not cleaned: its model is 0.

Coral records its iteration history (``record_iteration_history``), Hogbom
only without the mask, so that its summary figures are drawn both with and
without it. With ``SAVE_PLOTS`` set (``1``, ``true``, ``yes`` or ``on``)
every test writes its summary figure (:mod:`astroviper.utils.report_plots`)
to ``PLOT_DIR`` (default ``plots``).

Every upper limit is above, and every lower limit below, the values of nine
noise seeds, seed 5 and eight others (1000 to 8000); the comments give them.
The noise depends on the seed and the time and channel indices only, not on
how the simulation is chunked. A change of the input at the rounding level
(the sky summed in another order) changes Coral's iterations by tens, its
model updates by one and its other measures by up to two thirds of the
spread of the seeds, and Hogbom's measures not in their first three digits.
"""

import os
import shutil
from unittest import mock

import dask
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from scipy.constants import speed_of_light
from scipy.stats import kurtosis, skew
from xradio.image import make_empty_sky_image, write_image
from xradio.measurement_set import load_processing_set, open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from astroviper.processing_functions.imaging import model_update
from astroviper.processing_functions.imaging.restore import (
    _clean_beam_kernel_ft,
    _convolve_with_clean_beam,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_THRESHOLD,
    MODEL_UPDATE_MAX_ITER_PER_CYCLE,
    MODEL_UPDATE_STOP_DESCRIPTIONS,
    MODEL_UPDATE_THRESHOLD_PER_CYCLE,
)
from astroviper.processing_functions.simulation.antenna_beams import (
    pack_beam_models,
    symmetric_beam_response,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.report_plots import (
    plot_plane_deconvolution_summary,
    plot_signed_images,
    plot_visibility_comparison,
)
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
ARCSEC = np.pi / (180 * 3600)
IMAGE_SIZE = [128, 128]
CELL_SIZE = np.array([-6.0, 6.0]) * ARCSEC
FREQUENCY = 3e9
BEAM_MODEL = airy_disk_model("vla_nvss")
# The continuum's default; the primary beam is above 0.32 over the image.
PRIMARY_BEAM_LIMIT = 0.2
THRESHOLD = 3.6e-4  # Jy, 2.8 times the thermal noise
IMAGERS = ("cube", "continuum")
DECONVOLVERS = {"coral": "asp", "hogbom": "hogbom"}
CONTROLS = {
    "max_iter": 20000,
    "max_cycles": 20,
    "max_iter_per_cycle": 2000,
    "threshold": [THRESHOLD, 1.0],  # Stokes I and V, which is noise: not cleaned
    "gain": 0.1,
    "psf_sidelobe_factor": 1.0,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}
# (deconvolver, masked) of the runs that record their iteration history
RECORD_HISTORY = {("coral", True), ("coral", False), ("hogbom", False)}
# one model update, which the kernel test replaces by the truth, and the
# residual update after it
ORACLE_CONTROLS = {
    **CONTROLS,
    "max_iter": 1,
    "max_cycles": 1,
    "max_iter_per_cycle": 1,
    "threshold": 0.0,
}
DIRTY_CONTROLS = {**ORACLE_CONTROLS, "max_iter": 0}
LEAST_MISFIT_7 = {
    "type": "least_misfit",
    "support": 7,
    "oversampling": 300,
    "interpolation": "linear",
}
KERNELS = {"default": None, "least_misfit_7": LEAST_MISFIT_7}
# Measured: default kernel; least misfit 7/300.
KERNEL_GATES = {
    "dirty image rms": 5e-7,  # 1.25e-7; 1.97e-6
    "dirty image max": 3e-6,  # 7.0e-7; 1.68e-5
    "PSF max": 1e-6,  # 2.27e-7; 4.68e-6
    "true model residual rms in mask": 3e-7,  # 9.7e-8; 8.6e-7
    "true model residual rms outside mask": 2e-7,  # 4.2e-8; 1.26e-6
}
# Upper limits per deconvolver, with the mask and without, and (lower, upper)
# for Hogbom's iterations (a PSF 5 percent too large stops early, 5 percent
# too small late). Measured with the cube imager: seed 5; eight other seeds.
# The continuum imager's are the same.
LIMITS = {
    ("coral", True): {
        "flux error": 1e-3,  # 1.00e-4; 7.87e-6 to 2.48e-4
        "peak flux error": 5e-4,  # 1.14e-4; 2.74e-5 to 2.10e-4
        "max |F|": 1e-3,  # 4.53e-4; 5.03e-4 to 6.24e-4
        "mean |F|": 1.6e-4,  # 1.16e-4; 1.10e-4 to 1.20e-4
        "median |F|": 1.5e-4,  # 9.74e-5; 9.22e-5 to 1.03e-4
        "rms F": 2e-4,  # 1.45e-4; 1.39e-4 to 1.49e-4
        "residual rms in mask / sigma": 1.7,  # 1.54; 1.53 to 1.54
        "residual MAD rms in mask / sigma": 2.0,  # 1.80; 1.79 to 1.83
        "residual rms outside mask / sigma": 1.5,  # 1.31; 1.30 to 1.40
        "residual MAD rms outside mask / sigma": 1.5,  # 1.32; 1.29 to 1.42
        "residual |skew| in mask": 0.25,  # 0.153; 0.0709 to 0.145
        "|model| outside mask / model flux": 1e-4,  # 6.60e-5; 6.45e-5 to 6.78e-5
        "negative model flux / model flux": 0.01,  # 8.01e-3; 7.78e-3 to 7.98e-3
        "iterations": 10000,  # 7958; 7848 to 7971
        "model updates": 11,  # 9; 8 to 9
    },
    ("coral", False): {
        "flux error": 1e-3,  # 3.03e-4; 2.72e-5 to 2.95e-4
        "peak flux error": 1e-3,  # 2.78e-4; 1.35e-4 to 3.41e-4
        "max |F|": 1e-3,  # 5.60e-4; 5.31e-4 to 6.18e-4
        "mean |F|": 1.6e-4,  # 1.29e-4; 1.22e-4 to 1.27e-4
        "median |F|": 1.5e-4,  # 1.11e-4; 9.88e-5 to 1.05e-4
        "rms F": 2e-4,  # 1.62e-4; 1.53e-4 to 1.60e-4
        "residual rms in mask / sigma": 1.7,  # 1.26; 1.25 to 1.27
        "residual MAD rms in mask / sigma": 2.0,  # 1.35; 1.33 to 1.38
        "residual excess kurtosis in mask": 0.5,  # -0.630; -0.650 to -0.585
        "residual |skew| in mask": 0.25,  # 0.0223; 7.19e-5 to 0.0191
        "|model| away from the source / model flux": 0.01,  # 7.11e-3; 7.26e-3 to 7.66e-3
        "negative model flux / model flux": 0.01,  # 8.07e-3; 8.08e-3 to 8.43e-3
        "iterations": 10000,  # 7399; 7264 to 7369
        "model updates": 12,  # 10; 9 to 10
    },
    ("hogbom", True): {
        "flux error": 1e-3,  # 1.20e-4; 1.38e-5 to 2.53e-4
        "peak flux error": 2e-3,  # 1.33e-3; 1.27e-3 to 1.52e-3
        "max |F|": 2.5e-3,  # 1.76e-3; 1.66e-3 to 1.81e-3
        "mean |F|": 3e-4,  # 2.12e-4; 2.06e-4 to 2.20e-4
        "median |F|": 2e-4,  # 1.41e-4; 1.31e-4 to 1.48e-4
        "rms F": 4e-4,  # 3.07e-4; 2.99e-4 to 3.22e-4
        "residual rms in mask / sigma": 1.8,  # 1.65; 1.63 to 1.65
        "residual MAD rms in mask / sigma": 2.2,  # 2.00; 1.89 to 2.00
        "residual rms outside mask / sigma": 2.0,  # 1.72; 1.70 to 1.74
        "residual MAD rms outside mask / sigma": 2.0,  # 1.70; 1.68 to 1.76
        "residual |skew| in mask": 0.25,  # 0.115; 0.136 to 0.216
        "|model| outside mask / model flux": 0.0,  # 0; 0 to 0
        "negative model flux / model flux": 0.025,  # 0.0172; 0.0162 to 0.0182
        "iterations": (13000, 18000),  # 14608; 13984 to 14965
        "model updates": 14,  # 11; 11 to 12
    },
    ("hogbom", False): {
        "flux error": 1e-3,  # 4.77e-4; 2.99e-4 to 5.94e-4
        "peak flux error": 2e-3,  # 1.38e-3; 1.33e-3 to 1.53e-3
        "max |F|": 2.5e-3,  # 1.94e-3; 1.78e-3 to 1.95e-3
        "mean |F|": 3e-4,  # 2.22e-4; 2.18e-4 to 2.30e-4
        "median |F|": 2e-4,  # 1.48e-4; 1.44e-4 to 1.54e-4
        "rms F": 4e-4,  # 3.21e-4; 3.13e-4 to 3.34e-4
        "residual rms in mask / sigma": 1.8,  # 1.44; 1.43 to 1.46
        "residual MAD rms in mask / sigma": 2.2,  # 1.62; 1.61 to 1.65
        "residual excess kurtosis in mask": 0.5,  # -0.865; -0.895 to -0.856
        "residual |skew| in mask": 0.25,  # 5.76e-3; 7.40e-4 to 0.0487
        "|model| away from the source / model flux": 0.03,  # 0.0209; 0.0189 to 0.0210
        "negative model flux / model flux": 0.03,  # 0.0217; 0.0201 to 0.0222
        "iterations": (14000, 18000),  # 15281; 14704 to 15539
        "model updates": 15,  # 12; 12 to 13
    },
}
SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "").lower() in ("1", "true", "yes", "on")
PLOT_DIR = os.environ.get("PLOT_DIR", "plots")


def gaussian(centre, sigma_major, sigma_minor, angle, flux):
    """Elliptical Gaussian of total flux ``flux`` on the pixel centres."""
    i, j = np.indices(IMAGE_SIZE, dtype=float)
    di, dj = i - centre[0], j - centre[1]
    u = np.cos(angle) * di + np.sin(angle) * dj
    v = np.cos(angle) * dj - np.sin(angle) * di
    g = np.exp(-0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2))
    return flux * g / g.sum()


def true_sky():
    """The Cygnus A-like source in Jy/pixel on the imaging grid."""
    c = IMAGE_SIZE[0] // 2
    sky = np.zeros(IMAGE_SIZE)
    sky[c, c] = 0.3  # core
    for centre, sigma_major, sigma_minor, angle, flux in (
        ((c + 28, c + 3), 6, 4, 0.1, 8.0),  # lobes
        ((c - 27, c - 2), 7, 4.5, -0.15, 6.0),
        ((c + 38, c + 4), 1, 1, 0, 2.0),  # hot spots
        ((c + 35, c + 8), 1.5, 1.2, 0.3, 0.8),
        ((c - 37, c - 4), 1, 1, 0, 1.5),
        ((c, c), 15, 3, 0.05, 1.0),  # bridge
    ):
        sky += gaussian(centre, sigma_major, sigma_minor, angle, flux)
    for t in np.linspace(0.15, 0.85, 8):  # jet
        sky += gaussian((c + 38 * t, c + 4 * t), 1, 1, 0, 0.025)
    return sky


def simulate_input(directory, noise_params):
    """The simulated processing set, the true sky, the sky times the beam of
    the simulator, the direction cosines of the pixels and the thermal noise of
    the Stokes I image with natural weights."""
    sky = true_sky()
    sky_store = str(directory / "sky.img.zarr")
    sky_image = make_empty_sky_image(
        phase_center=PC,
        image_size=IMAGE_SIZE,
        cell_size=list(CELL_SIZE),
        frequency_coords=[FREQUENCY],
        pol_coords=["I"],
        time_coords=[0],
    )
    sky_image["SKY"] = (
        ("time", "frequency", "polarization", "l", "m"),
        sky[None, None, None],
    )
    sky_image["SKY"].attrs["units"] = "Jy/pixel"
    write_image(sky_image, sky_store, out_format="zarr", overwrite=True)
    antenna_xds = read_telescope_layout("vla.d")
    # one thread: threads race in astropy's first sky offset frame (the uvw)
    with dask.config.set(scheduler="synchronous"):
        result = distributed_applications.simulation.simulate_processing_set(
            ps_store=str(directory / "input.ps.zarr"),
            antenna_xds=antenna_xds,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 1800.0,
                "n_samples": 12,
            },
            frequency_params={
                "freq_start": FREQUENCY,
                "freq_delta": 1e8,
                "n_channels": 1,
                "channel_width": 1e8,
            },
            polarization=["RR", "LL"],
            phase_center_ra_dec=PC[None, :],
            beam_models=[BEAM_MODEL],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            uvw_params={"w_term": False},
            noise_params=noise_params,
            sky_model_image=sky_store,
            sky_model_image_predict={"predict": "dft"},
            overwrite=True,
        )
    l, m = np.meshgrid(sky_image.l.values, sky_image.m.values, indexing="ij")
    beam = symmetric_beam_response(
        pack_beam_models([BEAM_MODEL])[0], l, m, np.float64(FREQUENCY), ipower=2
    )
    (ms_xdt,) = load_processing_set(result["ps_store"]).values()
    return {
        "ps_store": result["ps_store"],
        "sky": sky,
        "apparent": sky * beam,
        "l": l,
        "m": m,
        "sigma": 1 / np.sqrt(ms_xdt.ds.WEIGHT.values.sum()),
    }


@pytest.fixture(scope="module")
def noise_free(tmp_path_factory):
    """The simulation without noise."""
    return simulate_input(tmp_path_factory.mktemp("extended_source_noise_free"), None)


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """The simulation with thermal noise."""
    return simulate_input(
        tmp_path_factory.mktemp("extended_source"),
        {"t_receiver": 800.0, "random_seed": 5},
    )


@pytest.fixture(scope="module")
def clean_mask(tmp_path_factory):
    """A box around the source and its ``.npy`` path."""
    c = IMAGE_SIZE[0] // 2
    mask = np.zeros(IMAGE_SIZE, dtype=bool)
    mask[c - 52 : c + 53, c - 20 : c + 21] = True
    mask_file = tmp_path_factory.mktemp("extended_source_mask") / "mask.npy"
    np.save(mask_file, mask)
    return mask, str(mask_file)


def stokes_plane(image, name, polarization):
    """A Stokes plane of an image product (``polarization`` 0 for I, 1 for V;
    the first of its other axes)."""
    data = image[name]
    other = {dim: 0 for dim in data.dims if dim not in ("l", "m")}
    other["polarization"] = polarization
    return data.isel(other).values.astype(np.float64)


def run_image(
    ps_store,
    directory,
    imager,
    deconvolver,
    clean_mask,
    controls,
    kernel=None,
    restore=False,
):
    """Image a copy of ``ps_store`` with the cube or the continuum imager: the
    convergence record, imaging stop code and final residual entropy of
    Stokes I, its planes and its fitted beam, and the planes of Stokes V
    (``"model V"`` and so on)."""
    directory.mkdir(parents=True, exist_ok=True)
    copy = str(directory / "input.ps.zarr")
    shutil.copytree(ps_store, copy)
    image_store = str(directory / "image.img.zarr")
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": PC,
        "frequency_coords": open_processing_set(copy).xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "V"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }
    arguments = {
        "ps_store": copy,
        "image_store": image_store,
        "deconvolver": deconvolver,
        "clean_mask": clean_mask,
        "instrument_polarization_basis": "circular",
        "processing_set_data_group_name": "base",
        "image_data_variables_keep": [
            "sky_model",
            "sky_residual",
            "point_spread_function",
            "beam_fit_params_point_spread_function",
        ],
        "single_precision_image": False,
        "n_mapping_parallelism": {"frequency": 1},
        # one thread: more threads grid in a varying order, and the imagers
        # are compared bit for bit
        "processing_function_threads": 1,
        "gridding_kernel_params": kernel,
        "restore": restore,
        "overwrite": True,
    }
    with dask.config.set(scheduler="synchronous"):
        if imager == "cube":
            arguments["image_data_variables_keep"] += (
                ["sky_restored"] if restore else []
            )
            result = image_cube_single_field(
                image_params=image_params,
                imaging_weights_params={"weighting": "natural"},
                cycle_control_params={
                    **controls,
                    "primary_beam_limit": PRIMARY_BEAM_LIMIT,
                },
                primary_beam_params={"beam_model": BEAM_MODEL},
                **arguments,
            )
            stops = result["deconvolution"]
        else:
            arguments["image_data_variables_keep"] += (
                ["sky_restored_flat_noise"] if restore else []
            )
            result = image_continuum_single_field(
                image_params={
                    **image_params,
                    "n_taylor_terms": 1,
                    "reference_frequency": FREQUENCY,
                    "primary_beam_model": "airy",
                    "list_dish_diameters": [BEAM_MODEL["dish_diameter"]],
                    "list_blockage_diameters": [BEAM_MODEL["blockage_diameter"]],
                },
                imaging_weights_params={
                    "weighting": "natural",
                    "weighting_scope": "local",
                },
                cycle_control_params=controls,
                continuum_mode="mfs",
                primary_beam_limit=PRIMARY_BEAM_LIMIT,
                **arguments,
            )
            # with the statistics of the residual its last stop test checked
            stops = result["residual_statistics"]
    with xr.open_zarr(image_store) as stored:
        image = stored.load()
    beam = image["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"]
    (label,) = (dim for dim in beam.dims if dim.startswith("beam_param"))
    # a continuum run that makes the dirty image only has no record
    run = {
        "record": next(
            (
                fields
                for key, fields in result["deconvolution"].data.items()
                if key.pol == 0
            ),
            None,
        ),
        "stop_code": next(
            (
                int(fields["stop_code"].imaging)
                for key, fields in stops.data.items()
                if key.pol == 0
            ),
            None,
        ),
        "final entropy": next(
            (
                fields["entropy"][-1]
                for key, fields in stops.data.items()
                if key.pol == 0
            ),
            None,
        ),
        "beam": beam.isel({dim: 0 for dim in beam.dims if dim != label}).values,
    }
    restored_name = "SKY_RESTORED" if imager == "cube" else "SKY_RESTORED_FLAT_NOISE"
    for name, variable in (
        ("model", "SKY_MODEL"),
        ("residual", "SKY_RESIDUAL"),
        ("restored", restored_name),
        ("psf", "POINT_SPREAD_FUNCTION"),
    ):
        if variable in image:
            run[name] = stokes_plane(image, variable, 0)
            run[f"{name} V"] = stokes_plane(image, variable, 1)
    return run


def leave_the_truth(true_model):
    """A ``model_update_cube_single_field`` whose model is ``true_model``."""
    real = model_update.model_update_cube_single_field

    def model_update_with_the_true_model(img_xds, *args, **kwargs):
        result = real(img_xds, *args, **kwargs)
        group = kwargs.get("image_data_group_out_name", "model")
        name = img_xds.attrs["data_groups"][group]["sky"]
        model = np.zeros(img_xds[name].shape, dtype=img_xds[name].dtype)
        model[:, :, 0] = true_model  # Stokes I; V is 0
        img_xds[name] = img_xds[name].copy(data=model)
        return result

    return model_update_with_the_true_model


def rms(values):
    """The root mean square."""
    return np.sqrt(np.mean(values**2))


def mad_rms(values):
    """The rms of Gaussian noise of the same median absolute deviation."""
    return 1.4826 * np.median(np.abs(values - np.median(values)))


def save_figure(figure, name):
    """Write ``figure`` to ``PLOT_DIR`` as ``name``.png."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    figure.savefig(os.path.join(PLOT_DIR, f"{name}.png"), dpi=100)


def plot_name(request):
    """The file name of the summary figure of a test."""
    return request.node.name.replace("[", "_").rstrip("]")


@pytest.fixture(scope="module")
def kernel_floors(noise_free, clean_mask, tmp_path_factory):
    """The errors of a kernel in an imager, measured once: the dirty image
    against that of the exact transform and the residual of the true model,
    over the peak of the exact dirty image, and the PSF against the exact one
    (its peak is 1), with their images."""
    mask, mask_file = clean_mask
    runs = {}

    def run(imager, kernel):
        if (imager, kernel) in runs:
            return runs[imager, kernel]
        directory = tmp_path_factory.mktemp(f"kernel_{imager}_{kernel}")
        if (imager, "exact") not in runs:
            runs[imager, "exact"] = run_image(
                noise_free["ps_store"],
                directory / "exact",
                imager,
                "hogbom",
                mask_file,
                DIRTY_CONTROLS,
                {"type": "dft"},
            )
        dirty = run_image(
            noise_free["ps_store"],
            directory / "dirty",
            imager,
            "hogbom",
            mask_file,
            DIRTY_CONTROLS,
            KERNELS[kernel],
        )
        with mock.patch.object(
            model_update,
            "model_update_cube_single_field",
            leave_the_truth(noise_free["apparent"]),
        ):
            oracle = run_image(
                noise_free["ps_store"],
                directory / "oracle",
                imager,
                "hogbom",
                mask_file,
                ORACLE_CONTROLS,
                KERNELS[kernel],
            )
        np.testing.assert_array_equal(oracle["model"], noise_free["apparent"])
        exact = runs[imager, "exact"]
        peak = exact["residual"].max()
        dirty_error = dirty["residual"] - exact["residual"]
        psf_error = dirty["psf"] - exact["psf"]
        residual = oracle["residual"]
        runs[imager, kernel] = (
            {
                "dirty image rms": rms(dirty_error) / peak,
                "dirty image max": np.abs(dirty_error).max() / peak,
                "PSF max": np.abs(psf_error).max(),
                "true model residual rms in mask": rms(residual[mask]) / peak,
                "true model residual rms outside mask": rms(residual[~mask]) / peak,
            },
            (dirty_error / peak, psf_error, residual / peak),
        )
        return runs[imager, kernel]

    return run


@pytest.fixture(scope="module")
def deconvolution(simulated, clean_mask, tmp_path_factory):
    """The deconvolution runs, each made once:
    ``(deconvolver, imager, masked) -> run``."""
    runs = {}

    def run(deconvolver, imager, masked):
        if (deconvolver, imager, masked) not in runs:
            name = f"{deconvolver}_{imager}_{'mask' if masked else 'no_mask'}"
            runs[deconvolver, imager, masked] = run_image(
                simulated["ps_store"],
                tmp_path_factory.mktemp(name),
                imager,
                DECONVOLVERS[deconvolver],
                clean_mask[1] if masked else None,
                {
                    **CONTROLS,
                    "record_iteration_history": (deconvolver, masked) in RECORD_HISTORY,
                },
                restore=True,
            )
        return runs[deconvolver, imager, masked]

    return run


def truth_at_the_clean_beam(run, apparent):
    """The truth at the clean beam (Jy/beam), the model error there over its
    peak, and where that truth is above 1e-3 of its peak."""
    kernel_ft = _clean_beam_kernel_ft(run["beam"], *IMAGE_SIZE, CELL_SIZE[1], float, 1)
    smoothed = _convolve_with_clean_beam(apparent, kernel_ft, 1)
    peak = smoothed.max()
    error = _convolve_with_clean_beam(run["model"] - apparent, kernel_ft, 1) / peak
    return smoothed, error, smoothed > 1e-3 * peak


def measures(run, apparent, mask, sigma):
    """What ``LIMITS`` bound; ``mask`` None for the whole image, which the
    measures "in mask" then cover."""
    smoothed, error, source = truth_at_the_clean_beam(run, apparent)
    f = error[source]
    model, residual = run["model"], run["residual"]
    inside = residual.ravel() if mask is None else residual[mask]
    values = {
        "flux error": abs(model.sum() / apparent.sum() - 1),
        "peak flux error": abs(run["restored"].max() / smoothed.max() - 1),
        "max |F|": np.abs(f).max(),
        "mean |F|": np.abs(f).mean(),
        "median |F|": np.median(np.abs(f)),
        "rms F": rms(f),
        "residual rms in mask / sigma": rms(inside) / sigma,
        "residual MAD rms in mask / sigma": mad_rms(inside) / sigma,
        "residual excess kurtosis in mask": kurtosis(inside),
        "residual |skew| in mask": abs(skew(inside)),
        "|model| away from the source / model flux": np.abs(model[~source]).sum()
        / model.sum(),
        "negative model flux / model flux": -model[model < 0].sum() / model.sum(),
        "iterations": np.sum(run["record"]["iter_done"]),
        "model updates": len(run["record"]["model_update_stop_code"]),
    }
    if mask is not None:
        outside = residual[~mask]
        values |= {
            "residual rms outside mask / sigma": rms(outside) / sigma,
            "residual MAD rms outside mask / sigma": mad_rms(outside) / sigma,
            "|model| outside mask / model flux": np.abs(model[~mask]).sum()
            / model.sum(),
        }
    return values


def check_limits(values, limits):
    """Every measure at most its limit, or within its (lower, upper) limits."""
    exceeded = {}
    for name, limit in limits.items():
        lower, upper = limit if isinstance(limit, tuple) else (-np.inf, limit)
        if not lower <= values[name] <= upper:
            exceeded[name] = (float(values[name]), limit)
    assert not exceeded, exceeded


def check_stops(run):
    """Stokes I stops on its threshold; every model update reports a stop on
    ``threshold_per_cycle``, which is at least the threshold, or on its
    iteration limit, and makes at most ``max_iter_per_cycle`` iterations."""
    record = run["record"]
    n = len(record["model_update_stop_code"])
    assert run["stop_code"] == IMAGING_THRESHOLD
    assert set(record["model_update_stop_code"]) <= {
        MODEL_UPDATE_MAX_ITER_PER_CYCLE,
        MODEL_UPDATE_THRESHOLD_PER_CYCLE,
    }
    assert min(record["threshold_per_cycle"]) >= THRESHOLD
    assert max(np.atleast_1d(record["iter_done"])[:n]) <= CONTROLS["max_iter_per_cycle"]


def check_restored(run):
    """The restored image is the model at the clean beam plus the residual."""
    kernel_ft = _clean_beam_kernel_ft(run["beam"], *IMAGE_SIZE, CELL_SIZE[1], float, 1)
    restored = _convolve_with_clean_beam(run["model"], kernel_ft, 1) + run["residual"]
    assert np.abs(run["restored"] - restored).max() <= 1e-12 * run["restored"].max()


def check_component_flux(run):
    """The components of the iteration history sum to the model."""
    record = run["record"]
    n = len(record["model_update_stop_code"])
    flux = sum(np.sum(f) for f in record["iteration_component_flux"][:n])
    np.testing.assert_allclose(flux, run["model"].sum(), rtol=1e-12)


def save_summary(request, run, simulated, mask, title):
    """Write the summary figure of a deconvolution run."""
    smoothed, error, _ = truth_at_the_clean_beam(run, simulated["apparent"])
    figure = plot_plane_deconvolution_summary(
        run["record"],
        run["model"],
        run["residual"],
        restored=run["restored"],
        mask=mask,
        truth=smoothed,
        model_error=error * smoothed.max(),
        true_flux=simulated["apparent"].sum(),
        threshold=THRESHOLD,
        sigma=simulated["sigma"],
        stop_descriptions=MODEL_UPDATE_STOP_DESCRIPTIONS,
        final_entropy=run["final entropy"],
        title=f"{title} (images in Jy/beam, the model in Jy/pixel)",
    )
    save_figure(figure, plot_name(request))


def test_simulated_visibilities(noise_free, simulated, request):
    (ms_xdt,) = load_processing_set(noise_free["ps_store"]).values()
    clean = ms_xdt.ds
    visibility = clean.VISIBILITY.values  # time, baseline, frequency, RR and LL
    uvw = clean.UVW.values
    assert visibility.shape == (12, 351, 1, 2)
    np.testing.assert_array_equal(visibility[..., 0], visibility[..., 1])
    assert not uvw[..., 2].any()
    assert not clean.FLAG.values.any()
    rows = np.random.default_rng(0).choice(12 * 351, size=300, replace=False)
    time, baseline = np.unravel_index(rows, (12, 351))
    u, v = uvw[time, baseline, :2].T * clean.frequency.values[0] / speed_of_light
    l, m, flux = (noise_free[name].ravel() for name in ("l", "m", "apparent"))
    phase = 2 * np.pi * (np.outer(u, l) + np.outer(v, m))
    expected = np.exp(1j * phase) @ (flux / np.sqrt(1 - l**2 - m**2))
    error = np.abs(visibility[time, baseline, 0, 0] - expected).max()

    (ms_xdt,) = load_processing_set(simulated["ps_store"]).values()
    noisy = ms_xdt.ds
    stokes_v = (noisy.VISIBILITY.values[..., 0] - noisy.VISIBILITY.values[..., 1]) / 2
    normalized = stokes_v * np.sqrt(2 * noisy.WEIGHT.values[..., 0])
    normalized = np.concatenate([normalized.real.ravel(), normalized.imag.ravel()])
    if SAVE_PLOTS:
        figure = plot_visibility_comparison(
            np.hypot(u, v),
            visibility[time, baseline, 0, 0],
            expected,
            normalized,
            title=f"Simulated visibilities: largest difference {error:.1e} Jy "
            f"({error / flux.sum():.1e} of the flux)",
        )
        save_figure(figure, plot_name(request))
    assert error < 1e-12 * flux.sum()
    assert abs(normalized.std() - 1) < 0.03
    assert abs(normalized.mean()) < 4 / np.sqrt(normalized.size)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("imager", IMAGERS)
def test_gridding_kernel_floor(kernel_floors, clean_mask, request, imager, kernel):
    floors, images = kernel_floors(imager, kernel)
    if SAVE_PLOTS:
        figure = plot_signed_images(
            images,
            [
                f"dirty image error ({floors['dirty image rms']:.1e} rms)",
                f"PSF error ({floors['PSF max']:.1e} max; PSF peak 1)",
                "residual of the true model "
                f"({floors['true model residual rms in mask']:.1e} rms in mask)",
            ],
            masks=[clean_mask[0], None, clean_mask[0]],
            title=f"Kernel {kernel}, {imager} imager: the dirty image and the "
            "residual over the peak of the exact dirty image",
        )
        save_figure(figure, plot_name(request))
    accurate = {name: floors[name] < gate for name, gate in KERNEL_GATES.items()}
    if kernel == "default":
        assert all(accurate.values()), floors
    else:  # the test sees the less accurate kernel
        assert not any(accurate.values()), floors
    if imager == "continuum":
        _, cube = kernel_floors("cube", kernel)
        for plane, cube_plane in zip(images, cube, strict=True):
            np.testing.assert_array_equal(plane, cube_plane)


@pytest.mark.parametrize("imager", IMAGERS)
@pytest.mark.parametrize("deconvolver", DECONVOLVERS)
def test_recovers_the_simulated_source(
    deconvolution, simulated, clean_mask, request, deconvolver, imager
):
    mask, _ = clean_mask
    run = deconvolution(deconvolver, imager, True)
    values = measures(run, simulated["apparent"], mask, simulated["sigma"])
    if SAVE_PLOTS:
        save_summary(
            request,
            run,
            simulated,
            mask,
            f"{deconvolver} ({imager} imager) with the clean mask",
        )
    check_stops(run)
    check_restored(run)
    check_limits(values, LIMITS[deconvolver, True])
    if deconvolver == "coral":
        hogbom = measures(
            deconvolution("hogbom", imager, True),
            simulated["apparent"],
            mask,
            simulated["sigma"],
        )
        assert values["rms F"] < 0.6 * hogbom["rms F"]


@pytest.mark.parametrize("imager", IMAGERS)
@pytest.mark.parametrize("deconvolver", DECONVOLVERS)
def test_without_a_clean_mask(deconvolution, simulated, request, deconvolver, imager):
    run = deconvolution(deconvolver, imager, False)
    values = measures(run, simulated["apparent"], None, simulated["sigma"])
    if SAVE_PLOTS:
        save_summary(
            request,
            run,
            simulated,
            None,
            f"{deconvolver} ({imager} imager) without a clean mask",
        )
    check_stops(run)
    check_restored(run)
    check_limits(values, LIMITS[deconvolver, False])
    if deconvolver == "coral":
        check_component_flux(run)
        assert sum(run["record"]["n_residual_components"]) > 0
        assert sum(run["record"]["n_significance_components"]) == 0


@pytest.mark.parametrize("imager", IMAGERS)
def test_coral_stages(deconvolution, simulated, clean_mask, request, imager):
    run = deconvolution("coral", imager, True)
    if SAVE_PLOTS:
        save_summary(
            request,
            run,
            simulated,
            clean_mask[0],
            f"stages of coral ({imager} imager) with the clean mask",
        )
    check_component_flux(run)
    record = run["record"]
    scale = record["iteration_component_scale"]
    flux = record["iteration_component_flux"]
    # each model update starts with its residual-shaped components (the cube's
    # record ends with the empty history of the closing stop test)
    for s, n in zip(scale, record["n_residual_components"], strict=False):
        assert n > 0 and np.all(np.asarray(s)[:n] == -1)
    # every significance component is counted as an Asp Gaussian or a point too
    assert sum(record["n_scale_components"]) > sum(record["n_significance_components"])
    assert sum(record["n_significance_components"]) > 0
    assert sum(record["n_hogbom_runs"]) > 0
    assert all(np.all(f[s == -1] > 0) for s, f in zip(scale, flux, strict=True))


@pytest.mark.parametrize("imager", IMAGERS)
def test_hogbom_components(deconvolution, simulated, request, imager):
    run = deconvolution("hogbom", imager, False)
    if SAVE_PLOTS:
        save_summary(
            request,
            run,
            simulated,
            None,
            f"components of hogbom ({imager} imager, without a clean mask)",
        )
    record = run["record"]
    n = len(record["model_update_stop_code"])
    history = {
        name: record[f"iteration_{name}"][:n]
        for name in (
            "component_scale",
            "component_l",
            "component_m",
            "component_flux",
            "peak_residual",
        )
    }
    np.testing.assert_array_equal(
        [np.size(s) for s in history["component_scale"]],
        np.atleast_1d(record["iter_done"])[:n],
    )
    assert all(np.all(np.asarray(s) == 0) for s in history["component_scale"])
    # the kernel subtracts only at a residual peak above threshold_per_cycle
    for update, (peak, threshold) in enumerate(
        zip(history["peak_residual"], record["threshold_per_cycle"], strict=True)
    ):
        below = np.count_nonzero(np.abs(peak) <= threshold)
        assert not below, (update + 1, below, threshold)
    # the points make up the model
    model = np.zeros(IMAGE_SIZE)
    np.add.at(
        model,
        tuple(
            np.concatenate(history[name]).astype(int)
            for name in ("component_l", "component_m")
        ),
        np.concatenate(history["component_flux"]),
    )
    np.testing.assert_allclose(model, run["model"], rtol=0, atol=1e-12 * model.max())


@pytest.mark.parametrize("masked", [True, False], ids=["mask", "no_mask"])
@pytest.mark.parametrize("deconvolver", DECONVOLVERS)
def test_cube_and_continuum_agree(
    deconvolution, clean_mask, request, deconvolver, masked
):
    cube, continuum = (deconvolution(deconvolver, imager, masked) for imager in IMAGERS)
    names = ("model", "residual", "restored")
    if SAVE_PLOTS:
        figure = plot_signed_images(
            [cube[name] - continuum[name] for name in names],
            [
                f"{name}: cube - continuum ({unit})"
                for name, unit in zip(
                    names, ("Jy/pixel", "Jy/beam", "Jy/beam"), strict=True
                )
            ],
            masks=[clean_mask[0] if masked else None] * len(names),
            title=f"{deconvolver}, {'with' if masked else 'without'} the clean mask",
        )
        save_figure(figure, plot_name(request))
    n = len(continuum["record"]["model_update_stop_code"])
    assert len(cube["record"]["model_update_stop_code"]) == n
    # the cube's record ends with the closing stop test
    np.testing.assert_array_equal(
        np.atleast_1d(cube["record"]["iter_done"])[:n],
        continuum["record"]["iter_done"],
    )
    if (deconvolver, masked) in RECORD_HISTORY:
        for name in ("l", "m", "scale", "flux"):
            for update in range(n):
                np.testing.assert_array_equal(
                    cube["record"][f"iteration_component_{name}"][update],
                    continuum["record"][f"iteration_component_{name}"][update],
                    err_msg=f"{name} of model update {update + 1}",
                )
    for name in (*names, *(f"{name} V" for name in names)):
        np.testing.assert_array_equal(cube[name], continuum[name], err_msg=name)
    assert not cube["model V"].any()
