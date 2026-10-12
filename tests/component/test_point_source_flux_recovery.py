"""Component test: CLEAN flux recovery for a field of point sources (ALMA, compact array).

The scenario exercises the whole imaging cycle (residual updates in the
instrument basis, model updates in Stokes, cycle control, primary-beam
correction) on data whose truth is known exactly:

* ALMA in its most compact 12 m configuration (``alma.cycle8.1``: 43 antennas,
  baselines 15 to 161 m), a 2 h track near transit, 5 channels around 100 GHz,
  linear feeds, no noise.
* The imaging grid samples the synthesized beam with five pixels across its
  minor axis (0.7 arcsec cells).
* 42 point sources on pixel centres of the imaging grid, spread out to the
  radius where the primary-beam power falls to 0.3, at least two synthesized
  beams apart.  Fluxes are log-uniform over a chosen dynamic range; every
  source is polarised (Q, U and V each up to 30 percent of I, so the cross
  hands are complex: ``XY = U + iV``, ``YX = U - iV``); half are continuum
  sources, half are line sources whose flux follows a Gaussian profile across
  the channels.
* The simulator also writes the true sky on the imaging grid
  (``sky_image_params``), so every comparison is pixel by pixel.
* Imaging: natural weights, Hogbom CLEAN at double precision, restore and
  primary-beam correction, several residual update cycles.  Both polarization
  modes of the imager are run: the two parallel hands ``XX, YY`` give Stokes
  ``I, Q`` and all four correlations give ``I, Q, U, V``.
* The imaging parameters are the ones that reproduce the truth best in a scan
  of the cycle controls and the FFT padding (see
  :func:`cycle_control_params`): an FFT padding of 2, short model updates
  and a stopping threshold at the numerical floor of the imaging cycle, which
  is set by the gridding and lies near ``3e-6`` of the brightest source.

For every source, channel and Stokes parameter the CLEAN model flux (a 3 x 3
pixel box divided by the primary beam) must match the truth to 1 percent of
the source's Stokes I plus a floor set by what CLEAN leaves at source-free
positions (five times the rms of 3 x 3 box sums of model plus residual inside
the clean mask, plus the stopping threshold, both divided by the primary
beam; the residual alone understates the floor because CLEAN absorbs it into
the model).  The restored, primary-beam corrected image must match the truth
convolved with the imager's own clean beam to 2 percent.  Every plane must
cycle at least twice with a falling peak residual and, in the strict
variants, stop on the threshold.  The sources of the ``1e5`` variant reach
below the numerical floor; they are judged against the measured floor.

Near the numerical floor the peak residual still falls on the whole, but it
fluctuates from one cycle to the next, and differently on every platform:
differences of 1e-13 Jy in the residual (rounding in the transforms) decide
which pixel CLEAN picks, and from then on the peak residuals differ at the
percent level.  In a clean that runs into the floor, Linux x86-64 and macOS
arm64 agree to six digits for 14 cycles and part in cycle 15; rises of up to 5
percent above the lowest peak residual reached before have been seen.  A
falling peak residual therefore means that no cycle ends more than
``PEAK_RESIDUAL_RISE`` above the lowest peak residual reached before it; a
cycle that diverges rises without bound.

Imaging ``I, Q`` from the four-correlation processing set must load only the
parallel hands and reproduce the image of the two-hand data.

With thermal noise added to the visibilities the entropy stop of the
cycle control (``entropy_stop``) must end the clean of every plane where
the residual looks most like noise: with a residual at the level of the
thermal noise, and with a restored image as close to the truth as the noise
allows.

ALMA C8 (``alma.cycle8.8``, baselines to 8.5 km) at 950 GHz takes the field
to the primary beam power 0.2 radius (3276 pixels of 2.8 mas, five across
the minor axis of the beam), one channel, the two parallel hands of
unpolarized sources: the 1 Jy source on the lattice site farthest out along
an image axis, a 0.5 Jy source near the centre. There the w term, which the
two dimensional imager does not model, sets the floor of the imaging cycle,
and the gridding kernel (``C8_KERNEL``, the exponential of a semicircle of
support 7 with linear lookup in a table of 300 samples per cell) lies below
a third of it. The floor is measured, not assumed: the residual one residual
update leaves for the true model (the oracle). The controls come from
``alma_cycle_control_params`` (threshold 1.2 times the masked peak of the
oracle); every plane must stop on the threshold within the predicted number
of cycles plus two, with the residual and the error image at the oracle's
floor and nothing of it absorbed into the model. With typical Band 10 noise
the entropy stop must end the clean at the thermal noise. CI runs a 1000
pixel probe of the same field (where the kernel, not the w term, sets the
floor), with and without the w term and the noise; the full field runs with
``ASTROVIPER_FULL_C8_TEST=1``.

The scenario builder and the analysis are plain functions so that
``dev/imaging/alma_point_source_recovery`` can rerun them and plot the result.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import uniform_filter
from scipy.optimize import brentq
from xradio.image import load_image, make_empty_sky_image
from xradio.measurement_set import load_processing_set, open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging import image_cube_single_field
from astroviper.processing_functions.imaging.restore import (
    _elliptical_gaussian_kernel,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_ENTROPY,
    IMAGING_THRESHOLD,
)
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    alma_cycle_control_params,
    predicted_imaging_cycles,
    psf_fraction,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    imaging_dict_to_dataframe,
)
from astroviper.processing_functions.simulation.antenna_beams import (
    casa_airy_disk_response,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import inverse_sin_project
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180 * 3600)
LAYOUT = "alma.cycle8.1"  # most compact 12 m configuration
N_SOURCES = 42
PIXELS_PER_BEAM = 5  # across the minor axis of the synthesized beam
BEAM_MINOR_ARCSEC = 3.5  # natural weighting, in the highest channel
CELL_ARCSEC = BEAM_MINOR_ARCSEC / PIXELS_PER_BEAM  # 0.7 arcsec
IMAGE_SIZE = [288, 288]  # 202 arcsec: the primary beam down to the 1 percent level
CELL_SIZE = np.array([-CELL_ARCSEC, CELL_ARCSEC]) * ARCSEC
BEAM_ARCSEC = 4.0  # synthesized beam (major axis) estimate: lambda / 161 m at 100 GHz
MIN_SEPARATION_BEAMS = 2.0
PB_POWER_LIMIT = 0.3  # sources out to the radius where the beam power is this
LINE_WIDTH_CHANNELS = 0.7
POLARIZED_FRACTION = 0.3  # |Q|, |U| and |V| each up to this fraction of I
F_MAX = 1.0  # Jy, brightest source
TIME_PARAMS = {
    "time_start": "2019-10-03T19:00:00.000",
    "time_delta": 150.0,
    "n_samples": 48,  # a 2 h track
}
FREQUENCY_PARAMS = {
    "freq_start": 100e9,
    "freq_delta": 0.5e9,
    "n_channels": 5,
    "channel_width": 2e6,
    "spectral_window_name": "Band3",
}
STOKES = ["I", "Q", "U", "V"]
# the two polarization modes of the imager (linear feeds)
POLARIZATION_MODES = {
    "two_hands": {"polarization": ["XX", "YY"], "stokes": ["I", "Q"]},
    "four_correlations": {
        "polarization": ["XX", "XY", "YX", "YY"],
        "stokes": ["I", "Q", "U", "V"],
    },
}
# RA = local sidereal time at ALMA one hour into the track (16h18m), Dec at the zenith
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
# imaging parameters of the best deconvolution (see cycle_control_params)
FFT_PADDING = 2.0
THRESHOLD = 3.3e-6  # Jy, for F_MAX = 1 Jy: the numerical floor of the imaging cycle
MAX_ITER_PER_CYCLE = 300
FLUX_RTOL = 0.01
RESTORED_RTOL = 0.02
FLOOR_SIGMA = 5.0
FLOOR_GUARD_PIXELS = 2  # excluded around every source when measuring the floor
# largest peak residual allowed after a cycle, relative to the lowest one reached
# before it (fluctuations at the numerical floor: up to 1.05 seen)
PEAK_RESIDUAL_RISE = 1.25
IMAGING_WEIGHTS_PARAMS = {
    "weighting": "natural",
    "robust": 0.5,
    "casa_weighting_implementation": True,
}
CASA_KERNEL = {
    "type": "prolate_spheroidal_casa",
    "support": 7,
    "oversampling": 100,
    "interpolation": "nearest",
}
# typical ALMA noise (Cycle 13 sensitivity calculator at the zenith): T_sys reached
# through t_receiver with the default atmosphere term, the aperture efficiency of
# the Technical Handbook rescaled to the 10.7 m diameter of the beam model
BAND3_NOISE = {
    "t_receiver": 34.075,
    "ant_efficiency": 0.8957,
    "random_seed": 7,
}  # 74.3 K
BAND10_NOISE = {
    "t_receiver": 3169.775,
    "ant_efficiency": 0.3358,
    "random_seed": 7,
}  # 3210 K
# ALMA C8 at 950 GHz (see the module docstring)
C8_LAYOUT = "alma.cycle8.8"  # the first long baseline configuration
C8_FREQUENCY_PARAMS = {
    "freq_start": 950e9,
    "freq_delta": 0.5e9,
    "n_channels": 1,
    "channel_width": 2e6,
    "spectral_window_name": "Band10",
}
C8_CELL_ARCSEC = 2.8e-3  # a fifth of the fitted minor axis, 14.17 mas (natural weights)
C8_BEAM_ARCSEC = 16.4e-3  # fitted major axis
# image half widths in pixels: the CI probe, and the primary beam power 0.2 radius
C8_HALF_WIDTH = {"probe": 500, "full": 1638}
C8_FFT_PADDING = 1.2
C8_KERNEL = {
    "type": "exponential_semicircle",
    "support": 7,
    "oversampling": 300,
    "interpolation": "linear",
}
C8_FLUX_RTOL = 1e-3
C8_THRESHOLD_FACTOR = 1.2  # times the masked peak of the residual of the true model
# traced Python memory of a probe imaging run, in units of one image plane (32 to 34 measured)
C8_PROBE_MEMORY_PLANES = 34.0
FULL_C8 = os.environ.get("ASTROVIPER_FULL_C8_TEST") == "1"


def frequencies():
    return FREQUENCY_PARAMS["freq_start"] + FREQUENCY_PARAMS["freq_delta"] * np.arange(
        FREQUENCY_PARAMS["n_channels"]
    )


def primary_beam_power(radius, frequency):
    """ALMA 12 m (CASA Airy) primary-beam power at ``radius`` radians from the axis."""
    model = airy_disk_model("alma")
    voltage = casa_airy_disk_response(
        radius, 0.0, frequency, model["dish_diameter"], model["blockage_diameter"], model["max_rad_1GHz"]
    )  # fmt: skip
    return float(voltage) ** 2


def build_scenario(dynamic_range, seed=0):
    """Sources on pixel centres inside the 0.3-power radius, with fluxes and spectra.

    Returns a dict with the component list for the simulator, the truth Stokes
    fluxes ``[n_source, n_channel, 4]`` (I, Q, U, V), the per-source table and
    the grid.
    """
    rng = np.random.default_rng(seed)
    nu = frequencies()
    grid = make_empty_sky_image(
        PHASE_CENTER, IMAGE_SIZE, CELL_SIZE, nu, ["I"], [0.0], do_sky_coords=False
    )
    l_axis, m_axis = grid.l.values, grid.m.values
    # the smallest beam (highest channel) sets the radius where the power is 0.3
    r_max = brentq(lambda r: primary_beam_power(r, nu[-1]) - PB_POWER_LIMIT, 1e-9, 0.02)
    r_max_pix = r_max / (CELL_ARCSEC * ARCSEC)
    # hexagonal lattice of candidate pixel centres, spacing > 2 beams even after a +-1 pixel jitter
    spacing = int(np.ceil(MIN_SEPARATION_BEAMS * BEAM_ARCSEC / CELL_ARCSEC)) + 3
    row_step = spacing * np.sqrt(3) / 2
    n_rows = int(r_max_pix / row_step) + 2
    candidates = []
    for row in range(-n_rows, n_rows + 1):
        di = int(round(row * row_step))
        offset = spacing / 2 if row % 2 else 0.0
        for col in range(-n_rows * 2, n_rows * 2 + 1):
            dj = int(round(col * spacing + offset))
            if np.hypot(di, dj) <= r_max_pix - 1.5:
                candidates.append((di, dj))
    assert len(candidates) >= N_SOURCES, (
        f"only {len(candidates)} lattice sites for {N_SOURCES} sources"
    )
    chosen = rng.choice(len(candidates), N_SOURCES, replace=False)
    centre = np.array([IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2])
    pixels = (
        np.array([candidates[k] for k in chosen])
        + rng.integers(-1, 2, size=(N_SOURCES, 2))
        + centre
    )
    separation = np.hypot(*(pixels[:, None, :] - pixels[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(separation, np.inf)
    assert separation.min() * CELL_ARCSEC >= MIN_SEPARATION_BEAMS * BEAM_ARCSEC
    lm = np.stack([l_axis[pixels[:, 0]], m_axis[pixels[:, 1]]], axis=-1)
    ra_dec = inverse_sin_project(PHASE_CENTER, lm)
    # fluxes: log-uniform over the dynamic range, the two extremes always present
    peak_flux = F_MAX * 10 ** (-rng.uniform(0.0, np.log10(dynamic_range), N_SOURCES))
    peak_flux[0], peak_flux[1] = F_MAX, F_MAX / dynamic_range
    q_fraction = rng.uniform(-POLARIZED_FRACTION, POLARIZED_FRACTION, N_SOURCES)
    kind = np.array(
        ["continuum"] * (N_SOURCES // 2) + ["line"] * (N_SOURCES - N_SOURCES // 2)
    )
    rng.shuffle(kind)
    line_centre = rng.uniform(-0.5, len(nu) - 0.5, N_SOURCES)
    # drawn last so that the positions, fluxes and Q are those of the first version of the test
    u_fraction = rng.uniform(-POLARIZED_FRACTION, POLARIZED_FRACTION, N_SOURCES)
    v_fraction = rng.uniform(-POLARIZED_FRACTION, POLARIZED_FRACTION, N_SOURCES)
    channel = np.arange(len(nu))
    profile = np.where(
        kind[:, None] == "line",
        np.exp(
            -0.5
            * ((channel[None, :] - line_centre[:, None]) / LINE_WIDTH_CHANNELS) ** 2
        ),
        1.0,
    )
    stokes_i = peak_flux[:, None] * profile  # [n_source, n_channel]
    stokes = np.stack(
        [
            stokes_i,
            q_fraction[:, None] * stokes_i,
            u_fraction[:, None] * stokes_i,
            v_fraction[:, None] * stokes_i,
        ],
        axis=-1,
    )  # [n_source, n_channel, 4] = I, Q, U, V
    components = []
    for s in range(N_SOURCES):
        i_flux, q_flux, u_flux, v_flux = (stokes[s, :, k] for k in range(4))
        # linear feeds: XX = I + Q, XY = U + iV, YX = U - iV, YY = I - Q
        flux = np.stack(
            [
                i_flux + q_flux,
                u_flux + 1j * v_flux,
                u_flux - 1j * v_flux,
                i_flux - q_flux,
            ],
            axis=-1,
        )
        components.append(
            {
                "kind": "point",
                "flux": flux,
                "ra_dec": ra_dec[s],
                "name": f"source_{s:02d}",
            }
        )
    radius_arcsec = np.hypot(*lm.T) / ARCSEC
    sources = pd.DataFrame(
        {
            "pixel_l": pixels[:, 0],
            "pixel_m": pixels[:, 1],
            "ra": ra_dec[:, 0],
            "dec": ra_dec[:, 1],
            "radius_arcsec": radius_arcsec,
            "primary_beam_power": [
                primary_beam_power(r * ARCSEC, nu[-1]) for r in radius_arcsec
            ],
            "peak_flux": peak_flux,
            "q_fraction": q_fraction,
            "u_fraction": u_fraction,
            "v_fraction": v_fraction,
            "kind": kind,
            "line_centre_channel": np.where(kind == "line", line_centre, np.nan),
            "nearest_neighbour_beams": separation.min(axis=1)
            * CELL_ARCSEC
            / BEAM_ARCSEC,
        }
    )
    return {
        "dynamic_range": dynamic_range,
        "seed": seed,
        "components": components,
        "stokes": stokes,
        "sources": sources,
        "frequencies": nu,
        "l_axis": l_axis,
        "m_axis": m_axis,
        "r_max_arcsec": r_max / ARCSEC,
    }


def simulate(
    work_dir,
    scenario,
    tag,
    mode="four_correlations",
    noise_params=None,
    layout=LAYOUT,
    frequency_params=None,
    image_size=None,
    cell_size=None,
    w_term=True,
):
    """Simulate the field and write the truth image; returns the two store paths.

    ``noise_params`` adds thermal noise (``None``: no noise, unit weights);
    ``w_term=False`` drops the w term from the visibilities.
    """
    antenna_xds = read_telescope_layout(layout)
    ps_store = f"{work_dir}/{tag}.ps.zarr"
    truth_store = f"{work_dir}/{tag}_truth.img.zarr"
    distributed_applications.simulation.simulate_processing_set(
        ps_store=ps_store,
        antenna_xds=antenna_xds,
        time_params=TIME_PARAMS,
        frequency_params=frequency_params or FREQUENCY_PARAMS,
        polarization=POLARIZATION_MODES[mode]["polarization"],
        sky_components=scenario["components"],
        phase_center_ra_dec=PHASE_CENTER[None, :],
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
        noise_params=noise_params,
        uvw_params={"w_term": w_term},
        sky_image_params={
            "image_store": truth_store,
            "image_size": IMAGE_SIZE if image_size is None else image_size,
            "cell_size": CELL_SIZE if cell_size is None else cell_size,
            "polarization_coords": STOKES,
        },
        overwrite=True,
    )
    return ps_store, truth_store


def cycle_control_params(dynamic_range):
    """The cycle controls that reproduce the truth best.

    A scan of the controls on this field (restored image against the truth
    convolved with the clean beam, and the fluxes at the source positions)
    gave:

    - ``threshold``: the residual update is accurate to about ``3e-6`` of the
      brightest source (gridding), whatever the dynamic range of the field.
      Cleaning below that level adds spurious components and makes the
      fluxes of the sources worse, so the threshold sits at that floor and
      does not follow the faintest source.
    - ``max_iter_per_cycle``: short model updates, so that the residual is
      recomputed often.
    - ``gain``, ``psf_sidelobe_factor`` and the PSF fractions make no
      difference to the result on this field; a gain above 0.1 is worse
      once the clean reaches the floor.

    The FFT padding of 2 (``FFT_PADDING``) lowers the floor by a third
    against a padding of 1.2.
    """
    return {
        "max_iter": 200000,
        "max_cycles": 40,
        "threshold": THRESHOLD * F_MAX,
        "primary_beam_limit": 0.2,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.5,
        "max_iter_per_cycle": MAX_ITER_PER_CYCLE,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }


def image(
    work_dir,
    ps_store,
    tag,
    dynamic_range,
    mode="four_correlations",
    n_mapping_parallelism=5,
    polarization_coords=None,
    controls=None,
    image_size=None,
    cell_size=None,
    fft_padding=FFT_PADDING,
    gridding_kernel_params=CASA_KERNEL,
    restore=True,
):
    """Hogbom CLEAN at double precision with restore and PB correction, in the Stokes planes of ``mode``.

    ``controls`` replaces the cycle controls of :func:`cycle_control_params`.
    """
    ps_xdt = open_processing_set(ps_store)
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    phase_direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values
    if polarization_coords is None:
        polarization_coords = POLARIZATION_MODES[mode]["stokes"]
    image_params = {
        "image_size": IMAGE_SIZE if image_size is None else image_size,
        "cell_size": CELL_SIZE if cell_size is None else cell_size,
        "phase_direction": phase_direction,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": polarization_coords,
        "time_coords": [0],
        "fft_padding": fft_padding,
        "cpp_gridder": True,
    }
    image_store = f"{work_dir}/{tag}.img.zarr"
    if controls is None:
        controls = cycle_control_params(dynamic_range)
    result = image_cube_single_field(
        ps_store=ps_store,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params=IMAGING_WEIGHTS_PARAMS,
        cycle_control_params=controls,
        instrument_polarization_basis="linear",
        gridder="anti_alias",
        gridding_kernel_params=gridding_kernel_params,
        deconvolver="hogbom",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=[
            "sky_model",
            "sky_residual",
            "primary_beam",
            "point_spread_function",
            "beam_fit_params_point_spread_function",
            "mask",
        ],
        processing_set_data_group_name="base",
        single_precision_image=False,
        processing_function_threads=1,
        n_mapping_parallelism={"frequency": n_mapping_parallelism},
        restore=restore,
        primary_beam_correction=restore,
        overwrite=True,
    )
    return image_store, result["deconvolution"], controls


def restore_truth(truth_plane, beam_params, cell_size):
    """Convolve a truth plane (Jy/pixel) with the imager's unit-peak clean beam (Jy/beam)."""
    major, minor, pa = beam_params
    n_l, n_m = truth_plane.shape
    kernel = _elliptical_gaussian_kernel(
        n_l, n_m, major / abs(cell_size[0]), minor / abs(cell_size[1]), pa, np.float64
    )
    kernel = np.roll(np.roll(kernel, -(n_l // 2), axis=0), -(n_m // 2), axis=1)
    return np.fft.irfft2(
        np.fft.rfft2(truth_plane) * np.fft.rfft2(kernel), s=truth_plane.shape
    )


def analyse(
    image_store,
    truth_store,
    scenario,
    deconvolution,
    controls,
    cell_size=CELL_SIZE,
    flux_rtol=FLUX_RTOL,
):
    """Per source / channel / Stokes recovery table, per-plane convergence table and floors."""
    img = load_image(image_store)
    truth = load_image(truth_store)
    np.testing.assert_allclose(truth.l.values, img.l.values)
    np.testing.assert_allclose(truth.m.values, img.m.values)
    image_stokes = [str(label) for label in img.polarization.values]
    # the truth holds all four Stokes parameters: pick the imaged ones by label
    stokes_index = [STOKES.index(label) for label in image_stokes]
    assert [str(label) for label in truth.polarization.values] == STOKES
    truth_sky = truth.SKY.values[0][:, stokes_index]  # [channel, stokes, l, m]
    scenario_stokes = scenario["stokes"][:, :, stokes_index]
    model = img.SKY_MODEL.values[0]
    residual = img.SKY_RESIDUAL.values[0]
    restored = img.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.values[0]
    primary_beam = img.PRIMARY_BEAM.values[0]
    mask = img.MASK.values[0].astype(bool)
    beam = img.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0]  # [channel, stokes, 3]
    n_channel, n_stokes = model.shape[:2]
    sources = scenario["sources"]
    planes = [(c, k) for c in range(n_channel) for k in range(n_stokes)]
    shape = (n_channel, n_stokes)
    residual_rms = np.array(
        [np.sqrt(np.mean(residual[c, k][mask[c, k]] ** 2)) for c, k in planes]
    ).reshape(shape)
    residual_peak = np.array(
        [np.abs(residual[c, k][mask[c, k]]).max() for c, k in planes]
    ).reshape(shape)
    # what CLEAN leaves where there is no source: 3 x 3 box sums of model plus
    # residual inside the mask, away from every source (CLEAN absorbs the
    # numerical floor into the model, so the residual alone understates it)
    box_total = (
        uniform_filter(model + residual, size=(1, 1, 3, 3), mode="constant") * 9.0
    )
    source_free = mask.copy()
    guard = FLOOR_GUARD_PIXELS
    for i, j in zip(sources.pixel_l, sources.pixel_m, strict=True):
        source_free[:, :, i - guard : i + guard + 1, j - guard : j + guard + 1] = False
    floor_rms = np.array(
        [np.sqrt(np.mean(box_total[c, k][source_free[c, k]] ** 2)) for c, k in planes]
    ).reshape(shape)
    floor_peak = np.array(
        [np.abs(box_total[c, k][source_free[c, k]]).max() for c, k in planes]
    ).reshape(shape)
    restored_truth = np.zeros_like(truth_sky)
    for (
        c,
        k,
    ) in planes:  # the restore step uses the Stokes I beam for every polarization
        restored_truth[c, k] = np.where(
            mask[c, k], restore_truth(truth_sky[c, k], beam[c, 0], cell_size), 0.0
        )
    rows = []
    for s, (i, j) in enumerate(zip(sources.pixel_l, sources.pixel_m, strict=True)):
        for c, k in planes:
            true_flux = scenario_stokes[s, c, k]
            assert np.isclose(truth_sky[c, k, i, j], true_flux, rtol=1e-12, atol=0.0), (
                "truth image differs from the scenario"
            )
            pb = primary_beam[c, k, i, j]
            box = model[c, k, i - 1 : i + 2, j - 1 : j + 2].sum()
            recovered = box / pb
            allowance = (
                flux_rtol * scenario_stokes[s, c, 0]
                + (FLOOR_SIGMA * floor_rms[c, k] + controls["threshold"]) / pb
            )
            rows.append(
                {
                    "source": s,
                    "channel": c,
                    "stokes": image_stokes[k],
                    "kind": sources.kind[s],
                    "radius_arcsec": sources.radius_arcsec[s],
                    "primary_beam": pb,
                    "true_flux": true_flux,
                    "true_stokes_i": scenario_stokes[s, c, 0],
                    "apparent_flux": true_flux * pb,
                    "model_box_flux": box,
                    "model_pixel_flux": model[c, k, i, j],
                    "recovered_flux": recovered,
                    "error": recovered - true_flux,
                    "allowance": allowance,
                    "restored": restored[c, k, i, j],
                    "restored_truth": restored_truth[c, k, i, j],
                    "residual": residual[c, k, i, j],
                    "floor_rms": floor_rms[c, k],
                    "residual_rms": residual_rms[c, k],
                }
            )
    recovery = pd.DataFrame(rows)
    convergence = imaging_dict_to_dataframe(deconvolution)
    convergence["stokes"] = [image_stokes[int(p)] for p in convergence["pol"]]
    summary = {
        "stokes": image_stokes,
        "floor_rms": floor_rms,
        "floor_peak": floor_peak,
        "residual_rms": residual_rms,
        "residual_peak": residual_peak,
        "beam_arcsec": beam[:, 0, :2] / ARCSEC,
        "beam_pa_deg": np.rad2deg(beam[:, 0, 2]),
        "mask_pixels": mask[:, 0].sum(axis=(1, 2)),
        "clean_dynamic_range": F_MAX / floor_peak[:, 0],
        "threshold": controls["threshold"],
    }
    return recovery, convergence, summary


def check_recovery(recovery, convergence, summary, strict):
    """The assertions shared by the variants (``strict`` requires the threshold stop)."""
    # every source, channel and Stokes parameter within its allowance
    worst = (recovery.error.abs() / recovery.allowance).max()
    failures = recovery[recovery.error.abs() > recovery.allowance]
    assert failures.empty, (
        f"{len(failures)} flux errors beyond their allowance (worst {worst:.2f} x):\n"
        f"{failures.head(12).to_string()}"
    )
    # the restored, primary-beam corrected image agrees with the restored truth
    tolerance = (
        RESTORED_RTOL * recovery.true_stokes_i
        + (FLOOR_SIGMA * recovery.floor_rms + summary["threshold"])
        / recovery.primary_beam
    )
    bad = recovery[(recovery.restored - recovery.restored_truth).abs() > tolerance]
    assert bad.empty, (
        f"restored image differs from the restored truth:\n{bad.head(8).to_string()}"
    )
    # every imaged plane cycles and converges
    for _, plane in convergence.iterrows():
        peakres = np.abs(
            np.asarray(plane.peakres, dtype=float)
        )  # recorded with its sign
        label = f"channel {plane.chan} Stokes {plane.stokes}"
        assert plane.n_cycles >= 2, (
            f"{label}: only {plane.n_cycles} residual update cycle(s)"
        )
        assert peakres[-1] < 0.1 * peakres[0], (
            f"{label}: peak residual did not fall: {peakres}"
        )
        # at the numerical floor the peak residual fluctuates from cycle to cycle
        # (see the module docstring): compare with the lowest value reached before
        lowest_before = np.minimum.accumulate(peakres)[:-1]
        rise = peakres[1:] / lowest_before
        assert rise.max() <= PEAK_RESIDUAL_RISE, (
            f"{label}: peak residual of cycle {rise.argmax() + 2} is {rise.max():.3f} times "
            f"the lowest one reached before (limit {PEAK_RESIDUAL_RISE}): {peakres}"
        )
        if strict:
            assert plane.stop_code_imaging == IMAGING_THRESHOLD, (
                f"{label}: stopped for another reason than the threshold "
                f"(code {plane.stop_code_imaging})"
            )
    # the synthesized beam is what the geometry assumed, and it is sampled by at
    # least PIXELS_PER_BEAM pixels across its minor axis in every channel
    beam_arcsec = summary["beam_arcsec"]  # [channel, (major, minor)]
    assert beam_arcsec[:, 0].max() < 1.5 * BEAM_ARCSEC, beam_arcsec
    assert beam_arcsec[:, 1].min() >= PIXELS_PER_BEAM * CELL_ARCSEC, beam_arcsec


def run_scenario(
    work_dir, dynamic_range, mode="four_correlations", seed=0, n_mapping_parallelism=5
):
    """Simulate, image and analyse one variant; returns everything the slides need."""
    tag = f"alma_point_sources_{mode}_dr{int(dynamic_range):d}"
    scenario = build_scenario(dynamic_range, seed)
    start = time.time()
    ps_store, truth_store = simulate(work_dir, scenario, tag, mode)
    t_simulate = time.time() - start
    start = time.time()
    image_store, deconvolution, controls = image(
        work_dir, ps_store, tag, dynamic_range, mode, n_mapping_parallelism
    )
    t_image = time.time() - start
    recovery, convergence, summary = analyse(
        image_store, truth_store, scenario, deconvolution, controls
    )
    summary.update(
        {
            "mode": mode,
            "polarization": POLARIZATION_MODES[mode]["polarization"],
            "t_simulate": t_simulate,
            "t_image": t_image,
            "ps_store": ps_store,
            "truth_store": truth_store,
            "image_store": image_store,
        }
    )
    return scenario, recovery, convergence, summary


def print_summary(recovery, convergence, summary):
    print(
        f"\nmode {summary['mode']}: correlations {summary['polarization']}, Stokes {summary['stokes']}; "
        f"threshold {summary['threshold']:.3g} Jy"
    )
    for k, label in enumerate(summary["stokes"]):
        sub = recovery[recovery.stokes == label]
        print(
            f"Stokes {label}: max |error| / true I {(sub.error.abs() / sub.true_stokes_i).max():.3g}, "
            f"max |error| / allowance {(sub.error.abs() / sub.allowance).max():.3g}, "
            f"floor rms {np.median(summary['floor_rms'][:, k]):.3g}, "
            f"floor peak {summary['floor_peak'][:, k].max():.3g}, "
            f"residual rms {np.median(summary['residual_rms'][:, k]):.3g} Jy"
        )
    print(f"clean dynamic range per channel: {summary['clean_dynamic_range']}")
    print(f"beam FWHM (arcsec) per channel: {summary['beam_arcsec'][:, 0]}")
    print(
        convergence[
            ["chan", "stokes", "n_cycles", "iter_total", "peakres_start", "peakres_final", "stop_code_imaging"]
        ].to_string()
    )  # fmt: skip
    print(f"simulate {summary['t_simulate']:.1f} s, image {summary['t_image']:.1f} s")


@pytest.mark.parametrize(
    "dynamic_range, strict, mode",
    [
        (1e3, True, "four_correlations"),
        (1e3, True, "two_hands"),
        (1e5, False, "four_correlations"),
    ],
)
def test_point_source_flux_recovery(tmp_path, dynamic_range, strict, mode):
    scenario, recovery, convergence, summary = run_scenario(
        str(tmp_path), dynamic_range, mode
    )
    print_summary(recovery, convergence, summary)
    assert len(scenario["sources"]) == N_SOURCES
    assert (scenario["sources"].primary_beam_power >= PB_POWER_LIMIT - 0.02).all()
    assert (scenario["sources"].nearest_neighbour_beams >= MIN_SEPARATION_BEAMS).all()
    # every Stokes parameter of the mode is imaged and checked, and none of them is empty
    assert summary["stokes"] == POLARIZATION_MODES[mode]["stokes"]
    assert sorted(recovery.stokes.unique()) == sorted(summary["stokes"])
    for label in summary["stokes"]:
        assert recovery[recovery.stokes == label].true_flux.abs().max() > 0.01 * F_MAX
    check_recovery(recovery, convergence, summary, strict)


def test_parallel_hands_of_four_correlation_data(tmp_path):
    """Stokes I, Q from a four-correlation processing set: only XX and YY are loaded and
    the image is that of the two-hand data; unsupported requests are refused."""
    scenario = build_scenario(1e3)
    ps_four, _ = simulate(str(tmp_path), scenario, "four", "four_correlations")
    ps_two, _ = simulate(str(tmp_path), scenario, "two", "two_hands")
    four = load_processing_set(ps_four)
    two = load_processing_set(ps_two)
    (four_ms,) = list(four.values())
    (two_ms,) = list(two.values())
    assert list(four_ms.polarization.values) == ["XX", "XY", "YX", "YY"]
    np.testing.assert_array_equal(
        four_ms.VISIBILITY.sel(polarization=["XX", "YY"]).values,
        two_ms.VISIBILITY.values,
    )
    assert np.abs(four_ms.VISIBILITY.sel(polarization="XY").values.imag).max() > 0.01
    image_four, _, _ = image(str(tmp_path), ps_four, "four_as_iq", 1e3, "two_hands")
    image_two, _, _ = image(str(tmp_path), ps_two, "two_as_iq", 1e3, "two_hands")
    from_four, from_two = load_image(image_four), load_image(image_two)
    assert list(from_four.polarization.values) == ["I", "Q"]
    for variable in (
        "SKY_MODEL",
        "SKY_RESIDUAL",
        "SKY_RESTORED",
        "SKY_RESTORED_PRIMARY_BEAM_CORRECTED",
        "POINT_SPREAD_FUNCTION",
        "PRIMARY_BEAM",
    ):
        np.testing.assert_array_equal(
            from_four[variable].values, from_two[variable].values, err_msg=variable
        )
    # a single Stokes plane, a wrong order or planes the data cannot give are refused
    for polarization_coords in (["I"], ["Q", "I"], ["I", "V"]):
        with pytest.raises(ValueError, match="polarization_coords"):
            image(
                str(tmp_path),
                ps_four,
                "refused",
                1e3,
                polarization_coords=polarization_coords,
            )
    with pytest.raises(ValueError, match="needed for the requested Stokes planes"):
        image(str(tmp_path), ps_two, "refused", 1e3, "four_correlations")


def test_entropy_stop_with_thermal_noise(tmp_path):
    """With noise in the data the entropy stop ends every plane at a residual
    of the level of the thermal noise, close to the truth."""
    scenario = build_scenario(1e3)
    ps_store, truth_store = simulate(
        str(tmp_path), scenario, "noise", noise_params=BAND3_NOISE
    )
    (ms_xdt,) = list(load_processing_set(ps_store).values())
    # natural weights: the noise of a Stokes image is that of one visibility
    # over the root of the number of visibilities of the two hands it uses
    sigma = 1.0 / np.sqrt(np.nanmax(ms_xdt.WEIGHT.values))
    n_visibilities = ms_xdt.sizes["time"] * ms_xdt.sizes["baseline_id"]
    thermal_noise = sigma / np.sqrt(2 * n_visibilities)
    controls = cycle_control_params(1e3)
    controls.update(
        {
            "threshold": 0.0,
            "max_cycles": -1,
            "max_iter_per_cycle": 100,
            "entropy_stop": True,
        }
    )
    image_store, deconvolution, _ = image(
        str(tmp_path), ps_store, "noise", 1e3, controls=controls
    )
    convergence = imaging_dict_to_dataframe(deconvolution)
    assert len(convergence) == len(frequencies()) * len(STOKES)
    assert (convergence.stop_code_imaging == IMAGING_ENTROPY).all(), convergence[
        ["chan", "pol", "stop_code_imaging"]
    ].to_string()
    # the clean stopped long before its budget, and Stokes I, which holds the
    # flux, took more iterations than the other planes
    assert convergence.iter_total.max() < 3000
    stokes_i = convergence[convergence.pol == 0].iter_total
    assert stokes_i.min() > convergence[convergence.pol != 0].iter_total.max()
    for _, plane in convergence.iterrows():
        entropy = np.asarray(plane.entropy, dtype=float)
        followed = entropy[np.isfinite(entropy)]
        # one entropy per imaging cycle; the last one recorded for a cycle in
        # which the plane cleaned lies below the highest one
        assert len(entropy) == plane.n_cycles
        assert followed.max() > followed[-1]

    img = load_image(image_store)
    truth = load_image(truth_store)
    residual = img.SKY_RESIDUAL.values[0]
    restored = img.SKY_RESTORED.values[0]
    primary_beam = np.nan_to_num(img.PRIMARY_BEAM.values[0])
    mask = img.MASK.values[0].astype(bool)
    beam = img.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0]
    truth_sky = truth.SKY.values[0]
    for c in range(residual.shape[0]):
        for k in range(residual.shape[1]):
            inside = mask[c, k]
            residual_rms = np.sqrt(np.mean(residual[c, k][inside] ** 2))
            assert 0.6 * thermal_noise < residual_rms < 1.4 * thermal_noise, (
                c,
                k,
                residual_rms,
                thermal_noise,
            )
            # the restored image against the truth as the imager sees it
            # (attenuated by the primary beam), convolved with the clean beam
            convolved = restore_truth(
                truth_sky[c, k] * primary_beam[c, k], beam[c, 0], CELL_SIZE
            )
            error = np.sqrt(np.mean((convolved - restored[c, k])[inside] ** 2))
            assert error < 1.3 * thermal_noise, (c, k, error, thermal_noise)


def c8_geometry(size):
    """Image size and cell of the C8 field: ``"probe"`` (CI) or ``"full"``."""
    half_width = C8_HALF_WIDTH[size]
    return [2 * half_width, 2 * half_width], np.array(
        [-C8_CELL_ARCSEC, C8_CELL_ARCSEC]
    ) * ARCSEC


def build_c8_scenario(size, seed=0, dynamic_range=1e3):
    """Unpolarized point sources of one 950 GHz channel out to 0.97 of the half width.

    The 1 Jy source sits on the lattice site farthest out along an image axis
    (the worst position for the kernel and the w term), a 0.5 Jy source
    within 0.5 arcsec of the centre, the faint extreme and the others, log
    uniform over the dynamic range, on random lattice sites with a +-1 pixel
    jitter.
    """
    rng = np.random.default_rng(seed)
    half_width = C8_HALF_WIDTH[size]
    image_size, cell_size = c8_geometry(size)
    nu = np.array([C8_FREQUENCY_PARAMS["freq_start"]])
    grid = make_empty_sky_image(
        PHASE_CENTER, image_size, cell_size, nu, ["I"], [0.0], do_sky_coords=False
    )
    spacing = int(np.ceil(MIN_SEPARATION_BEAMS * C8_BEAM_ARCSEC / C8_CELL_ARCSEC)) + 3
    row_step = spacing * np.sqrt(3) / 2
    r_max_pix = 0.97 * half_width
    n_rows = int(r_max_pix / row_step) + 2
    sites = []
    for row in range(-n_rows, n_rows + 1):
        di = int(round(row * row_step))
        offset = spacing / 2 if row % 2 else 0.0
        for col in range(-n_rows * 2, n_rows * 2 + 1):
            dj = int(round(col * spacing + offset))
            if np.hypot(di, dj) <= r_max_pix - 1.5:
                sites.append((di, dj))
    sites = np.array(sites)
    edge = int(np.argmax(np.abs(sites).max(axis=1) * 1e6 - np.abs(sites).min(axis=1)))
    near = np.flatnonzero(np.hypot(*sites.T) * C8_CELL_ARCSEC <= 0.5)
    near = near[near != edge]
    centre_site = near[rng.integers(len(near))]
    others = np.setdiff1d(np.arange(len(sites)), [edge, centre_site])
    chosen = np.concatenate(
        [[edge, centre_site], rng.choice(others, N_SOURCES - 2, replace=False)]
    )
    jitter = rng.integers(-1, 2, size=(N_SOURCES, 2))
    jitter[:2] = 0
    pixels = sites[chosen] + jitter + half_width
    peak_flux = 10 ** (-rng.uniform(0.0, np.log10(dynamic_range), N_SOURCES))
    peak_flux[:3] = 1.0, 0.5, 1.0 / dynamic_range
    separation = np.hypot(*(pixels[:, None, :] - pixels[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(separation, np.inf)
    lm = np.stack([grid.l.values[pixels[:, 0]], grid.m.values[pixels[:, 1]]], axis=-1)
    ra_dec = inverse_sin_project(PHASE_CENTER, lm)
    stokes = np.zeros((N_SOURCES, 1, 4))
    stokes[:, 0, 0] = peak_flux
    components = [
        {
            "kind": "point",
            "flux": np.array([[f, 0.0, 0.0, f]]),
            "ra_dec": ra_dec[s],
            "name": f"source_{s:02d}",
        }
        for s, f in enumerate(peak_flux)
    ]
    radius_arcsec = np.hypot(*lm.T) / ARCSEC
    sources = pd.DataFrame(
        {
            "pixel_l": pixels[:, 0],
            "pixel_m": pixels[:, 1],
            "radius_arcsec": radius_arcsec,
            "primary_beam_power": [
                primary_beam_power(r * ARCSEC, nu[0]) for r in radius_arcsec
            ],
            "peak_flux": peak_flux,
            "kind": np.array(["continuum"] * N_SOURCES),
            "nearest_neighbour_beams": separation.min(axis=1)
            * C8_CELL_ARCSEC
            / C8_BEAM_ARCSEC,
        }
    )
    return {
        "components": components,
        "stokes": stokes,
        "sources": sources,
        "frequencies": nu,
    }


def oracle_residual(work_dir, ps_store, truth_store, tag, size, gridding_kernel_params):
    """The residual the imager leaves for the true model, the floor of its imaging cycle.

    One residual update with ``SKY_MODEL`` = truth x ``PRIMARY_BEAM`` after the
    setup, run through the distributed application with its processing
    function replaced, so the node task loads and selects the data as for a
    CLEAN run. Returns the Stokes planes of the residual of channel 0, the
    clean mask, the clean beam, the masked Stokes I peak of the dirty image
    and the recorded PSF sidelobe.
    """
    import astroviper.processing_functions.imaging as pf_imaging
    from astroviper.processing_functions.image_analysis.make_mask import make_mask
    from astroviper.processing_functions.imaging.image_cube_single_field import (
        imaging_preparation_single_field,
    )
    from astroviper.processing_functions.imaging.residual_update import (
        residual_update_cube_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        build_residual_imaging_dict,
    )
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    truth = load_image(truth_store).SKY.values
    captured = {}
    prep_keys = ("processing_set_data_group_name", "single_precision_image",
                 "processing_function_threads", "fft_backend", "image_data_variables_keep",
                 "psf_fitting_method", "gridding_kernel_params", "primary_beam_params")  # fmt: skip
    update_keys = ("processing_set_data_group_name", "instrument_polarization_basis",
                   "single_precision_image", "processing_function_threads", "fft_backend",
                   "image_data_variables_keep", "gridding_kernel_params")  # fmt: skip

    def processing_function(ps_xdt, img_xds, image_params, weights, controls, **kwargs):
        prep = {key: kwargs[key] for key in prep_keys if key in kwargs}
        update = {key: kwargs[key] for key in update_keys if key in kwargs}
        _, img_xds, _, _, _ = imaging_preparation_single_field(
            ps_xdt, img_xds, image_params, weights, controls, **prep
        )
        img_xds, _ = residual_update_cube_single_field(
            ps_xdt, img_xds, image_params, False, **update
        )
        make_mask(img_xds, primary_beam_limit=controls["primary_beam_limit"],
                  image_data_group_in_name="residual", image_data_group_out_name="residual",
                  combine_mask=False, overwrite=False)  # fmt: skip
        mask = img_xds.MASK.values[0, 0, 0].astype(bool)
        captured["dirty_peak"] = float(
            np.abs(img_xds.SKY_RESIDUAL.values[0, 0, 0][mask]).max()
        )
        captured["max_psf_sidelobe"] = float(
            img_xds.MAX_SIDELOBE_POINT_SPREAD_FUNCTION.values[0, 0, 0]
        )
        stokes_index = [
            STOKES.index(str(label)) for label in img_xds.polarization.values
        ]
        primary_beam = np.nan_to_num(img_xds.PRIMARY_BEAM.values)
        _, model_group = create_data_groups_in_and_out(
            img_xds, data_group_in_name="residual", data_group_out_name="model",
            data_group_out_modified={"sky": "SKY_MODEL"}, overwrite=True,
        )  # fmt: skip
        img_xds["SKY_MODEL"] = img_xds["SKY_RESIDUAL"].copy(
            data=truth[:, :, stokes_index] * primary_beam
        )
        modify_data_groups_xds(img_xds, "model", model_group, description="True model.")
        img_xds, _ = residual_update_cube_single_field(
            ps_xdt, img_xds, image_params, True, **update
        )
        captured.update(
            residual=img_xds.SKY_RESIDUAL.values[0, 0].astype(np.float64),
            mask=mask,
            primary_beam=primary_beam[0, 0, 0],
            beam=img_xds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0, 0, 0],
        )
        record = build_residual_imaging_dict(img_xds, "residual", controls)
        timing = pd.DataFrame({"T_prep": [0.0], "T_residual_update": [0.0], "T_model_update": [0.0],
                               "T_restore": [0.0], "task_id": [kwargs.get("task_id", 0)],
                               "n_channels": [img_xds.sizes["frequency"]], "n_cycles": [1]})  # fmt: skip
        return img_xds, timing, record

    image_size, cell_size = c8_geometry(size)
    controls = {"max_iter": 0, "max_cycles": 1, "threshold": 0.0, "primary_beam_limit": 0.2,
                "gain": 0.1, "psf_sidelobe_factor": 1.0, "max_iter_per_cycle": -1,
                "min_psf_fraction": 0.05, "max_psf_fraction": 0.8}  # fmt: skip
    with mock.patch.object(pf_imaging, "image_cube_single_field", processing_function):
        image(work_dir, ps_store, tag, 1e3, "two_hands", 1, controls=controls,
              image_size=image_size, cell_size=cell_size, fft_padding=C8_FFT_PADDING,
              gridding_kernel_params=gridding_kernel_params, restore=False)  # fmt: skip
    return captured


def floors(residual_plane, mask, sources, beam, cell_size):
    """RMS and peak in the mask, RMS away from the sources (3 beams), box floor of a plane."""
    far = mask.copy()
    radius = 3.0 * beam[0] / abs(cell_size[0])
    ii, jj = np.indices(mask.shape)
    for i, j in zip(sources.pixel_l, sources.pixel_m, strict=True):
        far &= (ii - i) ** 2 + (jj - j) ** 2 > radius**2
    box = uniform_filter(residual_plane, size=3, mode="constant") * 9.0
    free = mask.copy()
    guard = FLOOR_GUARD_PIXELS
    for i, j in zip(sources.pixel_l, sources.pixel_m, strict=True):
        free[i - guard : i + guard + 1, j - guard : j + guard + 1] = False
    return {
        "rms": float(np.sqrt(np.mean(residual_plane[mask] ** 2))),
        "peak": float(np.abs(residual_plane[mask]).max()),
        "rms_far": float(np.sqrt(np.mean(residual_plane[far] ** 2))),
        "box": float(np.sqrt(np.mean(box[free] ** 2))),
        "far": far,
    }


def run_c8(work_dir, size, w_term, noise):
    """Simulate the C8 field, find its floor (oracle) and image it with the heuristic controls."""
    image_size, cell_size = c8_geometry(size)
    scenario = build_c8_scenario(size)
    tag = f"c8_{size}_{'w' if w_term else 'no_w'}_{noise or 'no_noise'}"
    ps_store, truth_store = simulate(
        work_dir, scenario, tag, "two_hands", BAND10_NOISE if noise else None,
        layout=C8_LAYOUT, frequency_params=C8_FREQUENCY_PARAMS,
        image_size=image_size, cell_size=cell_size, w_term=w_term,
    )  # fmt: skip
    oracle = oracle_residual(
        work_dir, ps_store, truth_store, tag + "_oracle", size, C8_KERNEL
    )
    oracle_floors = floors(oracle["residual"][0], oracle["mask"], scenario["sources"],
                           oracle["beam"], cell_size)  # fmt: skip
    fraction = psf_fraction(oracle["max_psf_sidelobe"])
    if noise:
        (ms_xdt,) = list(load_processing_set(ps_store).values())
        sigma = 1.0 / np.sqrt(np.nanmax(ms_xdt.WEIGHT.values))
        reference = sigma / np.sqrt(
            2 * ms_xdt.sizes["time"] * ms_xdt.sizes["baseline_id"]
        )
        controls = alma_cycle_control_params(
            oracle["max_psf_sidelobe"], oracle["dirty_peak"], 0.0, noisy=True,
            thermal_noise=reference,
        )  # fmt: skip
    else:
        reference = C8_THRESHOLD_FACTOR * oracle_floors["peak"]
        controls = alma_cycle_control_params(
            oracle["max_psf_sidelobe"], oracle["dirty_peak"], reference
        )
    tracemalloc.start()
    start = time.time()
    image_store, deconvolution, controls = image(
        work_dir, ps_store, tag, 1e3, "two_hands", 1, controls=controls,
        image_size=image_size, cell_size=cell_size, fft_padding=C8_FFT_PADDING,
        gridding_kernel_params=C8_KERNEL,
    )  # fmt: skip
    t_image = time.time() - start
    traced_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    recovery, convergence, summary = analyse(
        image_store, truth_store, scenario, deconvolution, controls, cell_size=cell_size,
        flux_rtol=0.01 if noise else C8_FLUX_RTOL,
    )  # fmt: skip
    img = load_image(image_store)
    model, residual = img.SKY_MODEL.values[0, 0], img.SKY_RESIDUAL.values[0, 0]
    truth = load_image(truth_store).SKY.values[0, 0, :2] * np.nan_to_num(
        img.PRIMARY_BEAM.values[0, 0]
    )
    beam = img.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0, 0, 0]
    error_image = residual[0] + restore_truth(model[0] - truth[0], beam, cell_size)
    run_floors = floors(
        residual[0], oracle["mask"], scenario["sources"], beam, cell_size
    )
    far = run_floors["far"]
    return {
        "scenario": scenario, "recovery": recovery, "convergence": convergence,
        "summary": summary, "controls": controls, "oracle": oracle_floors,
        "run": run_floors, "error_rms_far": float(np.sqrt(np.mean(error_image[far] ** 2))),
        "error_peak_far": float(np.abs(error_image[far]).max()),
        "restored_error_rms": float(np.sqrt(np.mean(
            (img.SKY_RESTORED.values[0, 0, 0] - restore_truth(truth[0], beam, cell_size))[oracle["mask"]] ** 2))),
        "reference": reference, "predicted_cycles": predicted_imaging_cycles(
            oracle["dirty_peak"], reference, fraction), "t_image": t_image,
        "traced_planes": traced_peak / (8.0 * image_size[0] * image_size[1]),
        "oracle_residual": oracle["residual"][0], "far": far, "truth_store": truth_store,
    }  # fmt: skip


def check_c8(result, noise):
    """The C8 assertions: the floor of the imaging cycle, or the thermal noise, is reached."""
    convergence = result["convergence"]
    recovery = result["recovery"]
    stokes_i = convergence[convergence.stokes == "I"].iloc[0]
    if noise:
        assert (convergence.stop_code_imaging == IMAGING_ENTROPY).all(), convergence[
            ["stokes", "stop_code_imaging", "n_cycles"]
        ].to_string()
        assert 0.6 < result["run"]["rms"] / result["reference"] < 1.4, result["run"]
        assert result["restored_error_rms"] < 1.3 * result["reference"]
        detected = recovery[
            (recovery.stokes == "I")
            & (recovery.apparent_flux > 5 * result["reference"])
        ]
        print(
            f"sources above 5 sigma ({5 * result['reference']:.3g} Jy): {list(detected.source)}"
        )
    else:
        assert (convergence.stop_code_imaging == IMAGING_THRESHOLD).all()
        assert stokes_i.n_cycles <= result["predicted_cycles"] + 2, (
            stokes_i.n_cycles,
            result["predicted_cycles"],
        )
        oracle, run = result["oracle"], result["run"]
        assert 0.8 <= run["rms"] / oracle["rms"] <= 1.3, (run["rms"], oracle["rms"])
        assert 0.7 <= run["peak"] / oracle["peak"] <= 1.5, (run["peak"], oracle["peak"])
        assert result["error_rms_far"] <= 1.5 * oracle["rms_far"]
        assert result["error_peak_far"] <= 1.5 * oracle["peak"]
        # CLEAN absorbed nothing of the floor into the model
        assert result["summary"]["floor_rms"][0, 0] <= 1.5 * oracle["box"]
        peakres = np.abs(np.asarray(stokes_i.peakres, dtype=float))
        assert (
            peakres[1:] / np.minimum.accumulate(peakres)[:-1]
        ).max() <= PEAK_RESIDUAL_RISE
    failures = recovery[recovery.error.abs() > recovery.allowance]
    assert failures.empty, failures.head(12).to_string()  # noise aware with noise


@pytest.mark.parametrize("noise", [None, "band10"])
@pytest.mark.parametrize("w_term", [True, False], ids=["w_term", "no_w_term"])
def test_c8_950ghz_flux_recovery(tmp_path, w_term, noise):
    """ALMA C8 at 950 GHz, CI size: the imaging cycle reaches its floor or the noise."""
    result = run_c8(str(tmp_path), "probe", w_term, noise)
    print(f"C8 probe, w term {w_term}, noise {noise}: oracle {result['oracle']}, run "
          f"{ {k: v for k, v in result['run'].items() if k != 'far'} }, image {result['t_image']:.1f} s, "
          f"traced memory {result['traced_planes']:.1f} planes")  # fmt: skip
    check_c8(result, noise)
    assert result["traced_planes"] <= 1.5 * C8_PROBE_MEMORY_PLANES


def test_c8_kernel_lowers_the_floor(tmp_path):
    """Without the w term the chosen kernel's floor is far below the CASA kernel's."""
    scenario = build_c8_scenario("probe")
    image_size, cell_size = c8_geometry("probe")
    ps_store, truth_store = simulate(
        str(tmp_path), scenario, "kernels", "two_hands", layout=C8_LAYOUT,
        frequency_params=C8_FREQUENCY_PARAMS, image_size=image_size, cell_size=cell_size,
        w_term=False,
    )  # fmt: skip
    rms = {}
    for name, kernel in (("casa", CASA_KERNEL), ("chosen", C8_KERNEL)):
        oracle = oracle_residual(
            str(tmp_path), ps_store, truth_store, name, "probe", kernel
        )
        rms[name] = floors(oracle["residual"][0], oracle["mask"], scenario["sources"],
                           oracle["beam"], cell_size)["rms_far"]  # fmt: skip
    assert rms["chosen"] < rms["casa"] / 20, rms


@pytest.mark.skipif(
    not FULL_C8, reason="3276 pixel images: set ASTROVIPER_FULL_C8_TEST=1"
)
@pytest.mark.parametrize("noise", [None, "band10"])
@pytest.mark.parametrize("w_term", [True, False], ids=["w_term", "no_w_term"])
def test_c8_950ghz_flux_recovery_full(tmp_path, w_term, noise):
    """The full C8 field out to primary beam power 0.2; without noise the kernel's floor
    lies below a third of the w term's (from the w term reference image)."""
    result = run_c8(str(tmp_path), "full", w_term, noise)
    check_c8(result, noise)
    if w_term and not noise:
        scenario = build_c8_scenario("full")
        image_size, cell_size = c8_geometry("full")
        ps_store, _ = simulate(
            str(tmp_path), scenario, "full_no_w", "two_hands", layout=C8_LAYOUT,
            frequency_params=C8_FREQUENCY_PARAMS, image_size=image_size,
            cell_size=cell_size, w_term=False,
        )  # fmt: skip
        no_w = oracle_residual(str(tmp_path), ps_store, result["truth_store"], "full_no_w",
                               "full", C8_KERNEL)["residual"][0]  # fmt: skip
        # the w term reference image: the residuals of the true model with and without w
        far = result["far"]
        w_rms_far = np.sqrt(np.mean((result["oracle_residual"] - no_w)[far] ** 2))
        kernel_rms_far = np.sqrt(np.mean(no_w[far] ** 2))
        assert kernel_rms_far <= w_rms_far / 3, (kernel_rms_far, w_rms_far)
