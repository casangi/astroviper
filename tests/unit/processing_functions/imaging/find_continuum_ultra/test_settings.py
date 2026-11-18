"""Tests of find_continuum_ultra's settings against the vendored findContinuum
on the simple port's metadata."""

import inspect
import itertools
import os

import numpy as np
import pytest
import xarray as xr
from xradio.image import make_empty_sky_image

from astroviper.distributed_applications.imaging.find_continuum import (
    casa_image_tasks,
)
from astroviper.distributed_applications.imaging.find_continuum.find_continuum import (
    _cube_metadata,
)
from astroviper.processing_functions.imaging.find_continuum import (
    linear_frequency_axis,
    moment_velocity_widths,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    continuum_n_bin,
    find_continuum_settings,
    initial_moment_difference_level,
    is_bad_atmosphere,
    is_time_division_mode,
    linear_spectral_axis,
    max_baseline_from_beam_fits,
    moment_channel_weights,
    n_baseline_channels,
    primary_beam_mean_channels,
    rounds_after_original,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    FindContinuumSettings,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    RAD_PER_ARCSEC,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    CHANNEL_WIDTHS,
    N_CHANNELS,
    findContinuum,
    pick,
    quiet,
    vendored_block,
)

BEAM = "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
BEAM_DIMS = ["time", "frequency", "polarization", "beam_params_label"]
# findContinuum's defaults of the arguments the pipeline does not pass.
DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(
        findContinuum.findContinuum
    ).parameters.items()
}
# The two middle values, over the 480 channels, of the float32 major and minor
# axes of the imaging case's beam fits (rad, as float32 bits): their medians
# give its max baseline, 92.46267128374839 m (92.462671 m in CASA's log).
IMAGING_MAJOR = np.array([934912507, 934912512], dtype=np.uint32).view(np.float32)
IMAGING_MINOR = np.array([924232151, 924232157], dtype=np.uint32).view(np.float32)

MEAN_FREQUENCY = vendored_block(
    959,
    "meanFreqHz = np.mean([firstFreq,lastFreq])",
    959,
    "meanFreqHz = np.mean([firstFreq,lastFreq])",
    ["firstFreq", "lastFreq"],
)
IMAGE_INFO_CONSTANTS = vendored_block(
    7368,
    "ARCSEC_PER_RAD = 206264.80624709636",
    7369,
    "c_mks = 2.99792458e8",
    [],
)()
MAX_BASELINE = vendored_block(
    7430,
    "if bmaj > 0:",
    7433,
    "maxBaseline = 0",
    ["bmaj", "bmin", "frequencyGHz", "c_mks", "ARCSEC_PER_RAD"],
)
N_BIN = vendored_block(
    989,
    "bandwidth = np.abs(nchan*channelWidth) # of the cube",
    1068,
    "casalogPost('nchan/nbin = %d/%d = %f >= %d=minRegions' % "
    "(nchan,nbin,nchan/nbin,minRegions))",
    [
        "nchan",
        "channelWidth",
        "nbin",
        "spectralDynamicRangeBandWidth",
        "vis",
        "img",
        "verbose",
        "maxBaseline",
    ],
)
ROUNDS = vendored_block(
    1070,
    "if amendMaskIterations == 'auto':",
    1080,
    "casalogPost('FDM spectrum detected: setting amendMaskIterations=%d' % "
    "(amendMaskIterations))",
    ["amendMaskIterations", "channelWidth", "nchan", "telescope", "maxBaseline"],
)
BAD_ATMOSPHERE = vendored_block(
    1244,
    "if (b > skyTransmissionThreshold or e > skyTempThreshold):",
    1256,
    "casalogPost(meanSpectrumMethodMessage)",
    ["b", "e", "skyTransmissionThreshold", "skyTempThreshold"],
)
ROUNDS_ENABLED = vendored_block(
    1302,
    "if amendMaskIterations != 0:",
    1309,
    "amendMask = False",
    ["amendMaskIterations"],
)
MOMENT_DIFFERENCE_LEVEL = vendored_block(
    1364,
    "if (amendMask or checkIfMaskWouldHaveBeenAmended):",
    1370,
    "casalogPost('Raising momDiffLevel by 3.5 to %.1f because this is 7m "
    "singleContinuum' % (momDiffLevel))",
    [
        "amendMask",
        "checkIfMaskWouldHaveBeenAmended",
        "maxBaseline",
        "singleContinuum",
        "momDiffLevel",
    ],
)
N_BASELINE_CHANNELS = vendored_block(
    5092,
    "if ((type(nBaselineChannels) == float or type(nBaselineChannels) == "
    "np.float64) and not fitsTable):",
    5097,
    "nBaselineChannels = int(round_half_up(nBaselineChannels*nchan))",
    ["nBaselineChannels", "fitsTable", "chanInfo"],
)
PRIMARY_BEAM_MEAN_CHANNELS = vendored_block(
    7793,
    "if nchan < 16:  # PIPE-1657",
    7796,
    "pbmomChans = '%d,%d,%d,%d,%d' % "
    "(nchan//8, nchan//4, nchan//2, 3*nchan//4, 7*nchan//8)",
    ["nchan"],
)


def _beam_fits(beams, polarizations=("I",)):
    """Beam fits of one time from ``beams`` ``(n_channels, n_polarizations,
    3)`` (rad)."""
    return xr.DataArray(
        np.asarray(beams)[None],
        dims=BEAM_DIMS,
        coords={"polarization": list(polarizations)},
        attrs={"units": "rad"},
    )


def _window(
    frequency,
    beams,
    transmission,
    sky_temperature,
    polarizations=("I",),
    shape=(5, 4),
    cell_arcsec=0.2,
    telescope_name="ALMA",
    single_continuum=False,
    n_bin=1,
    spectral_dynamic_range_bandwidth=None,
):
    """A spectral window: an image dataset with the metadata findContinuum
    reads and the arguments of its settings."""
    xds = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=list(shape),
        cell_size=[cell_arcsec * RAD_PER_ARCSEC] * 2,
        frequency_coords=frequency,
        pol_coords=list(polarizations),
        time_coords=[0.0],
    )
    xds["SKY_RESIDUAL"] = xr.DataArray(
        np.zeros((1, len(frequency), len(polarizations), *shape), np.float32),
        dims=["time", "frequency", "polarization", "l", "m"],
    )
    xds[BEAM] = _beam_fits(beams, polarizations).drop_vars("polarization")
    return dict(
        xds=xds,
        transmission=transmission,
        sky_temperature=sky_temperature,
        telescope_name=telescope_name,
        single_continuum=single_continuum,
        n_bin=n_bin,
        spectral_dynamic_range_bandwidth=spectral_dynamic_range_bandwidth,
    )


def _atmosphere(rng, n_channels, bad):
    """Transmission and sky temperature with a slope and absorption lines; a
    bad atmosphere has deep lines and a 60 K line in the sky temperature at the
    centre of the window."""
    x = np.linspace(-1.0, 1.0, n_channels)
    transmission = rng.uniform(0.6, 0.97) + rng.uniform(-0.05, 0.05) * x
    sky_temperature = rng.uniform(10.0, 80.0) + rng.uniform(-3.0, 3.0) * x
    for _ in range(int(rng.integers(0, 3))):
        line = np.exp(-0.5 * ((x - rng.uniform(-1, 1)) / rng.uniform(0.02, 0.5)) ** 2)
        depth = rng.uniform(0.0, 0.5 if bad else 0.03)
        transmission = transmission * (1.0 - depth * line)
        sky_temperature = sky_temperature + 250.0 * depth * line
    if bad:
        channel = np.arange(n_channels)
        sky_temperature = sky_temperature + 60.0 * np.exp(
            -0.5 * ((channel - n_channels / 2) / (n_channels / 40)) ** 2
        )
    return transmission, sky_temperature


def _random_window(rng):
    n_channels = pick(rng, N_CHANNELS)
    increment = pick(rng, CHANNEL_WIDTHS) * pick(rng, [1.0, 1.0, 1.0, -1.0])
    first = pick(rng, [1.5e9, 22.2e9, 86e9, 183e9, 230e9, 345e9, 650e9])
    first += pick(rng, [0.0, float(rng.uniform(0, 1e8))])
    first -= min(increment, 0.0) * (n_channels - 1)
    if rng.random() < 0.5:
        frequency = first + increment * np.arange(n_channels)
    else:
        last = first + (n_channels - 1) * increment * (1 + rng.uniform(-1e-9, 1e-9))
        frequency = np.linspace(first, last, n_channels)
    polarizations = pick(rng, [("I",), ("I", "Q")])
    major = (
        10 ** rng.uniform(-1.3, 0.9)
        * (frequency[0] / frequency)
        * (1 + 1e-3 * rng.standard_normal(n_channels))
    )
    beam = np.stack(
        [
            major * RAD_PER_ARCSEC,
            major * rng.uniform(0.3, 1.0) * RAD_PER_ARCSEC,
            np.full(n_channels, rng.uniform(-1.5, 1.5)),
        ],
        axis=-1,
    )
    beams = np.stack([beam * (1.0 + 0.3 * k) for k in range(len(polarizations))], 1)
    if rng.random() < 0.5:
        beams = beams.astype(np.float32)
    return _window(
        frequency,
        beams,
        *_atmosphere(rng, n_channels, bad=bool(rng.random() < 0.3)),
        polarizations=polarizations,
        shape=(int(rng.integers(2, 7)), int(rng.integers(2, 7))),
        cell_arcsec=pick(rng, [0.02, 0.06, 0.2, 0.3, 1.0, 3.0]),
        telescope_name=pick(rng, ["ALMA", "VLA"]),
        single_continuum=bool(rng.random() < 0.3),
        n_bin=pick(rng, [1, 1, 2, 4, 8, 40]),
        spectral_dynamic_range_bandwidth=pick(
            rng,
            [
                None,
                None,
                (int(rng.integers(0, 40)) + 0.5) * abs(increment),
                float(rng.uniform(1e5, 1e9)),
                0.5,
            ],
        ),
    )


def _reference_window(
    n_channels,
    first_frequency,
    channel_width,
    cell_arcsec,
    beam_arcsec,
    telescope_name="ALMA",
    bad_atmosphere=False,
):
    """The metadata of one of findContinuum's reference cubes (S1 §11.2): the
    beam constant over the channels (arcsec), or the imaging case's."""
    frequency = first_frequency + channel_width * np.arange(n_channels)
    if beam_arcsec == "imaging":
        beams = np.zeros((n_channels, 1, 3), np.float32)
        beams[:, 0, 0] = np.resize(IMAGING_MAJOR, n_channels)
        beams[:, 0, 1] = np.resize(IMAGING_MINOR, n_channels)
    else:
        beams = np.zeros((n_channels, 1, 3))
        beams[..., :2] = beam_arcsec * RAD_PER_ARCSEC
    rng = np.random.default_rng(n_channels)
    return _window(
        frequency,
        beams,
        *_atmosphere(rng, n_channels, bad_atmosphere),
        cell_arcsec=cell_arcsec,
        telescope_name=telescope_name,
    )


# Name: the window, then the max baseline, baseline channels, rounds after the
# original round and n_bin of the port and of segment_1_check.py.
REFERENCE_WINDOWS = {
    "line_set": ((480, 230e9, 976562.5, 0.2, 1.0), (268.8549272335251, 91, 4, 1)),
    "line_set:VLA": (
        (480, 230e9, 976562.5, 0.2, 1.0, "VLA"),
        (268.8549272335251, 91, 4, 1),
    ),
    "line_set_badatm": (
        (480, 230e9, 976562.5, 0.2, 1.0, "ALMA", True),
        (268.8549272335251, 91, 4, 1),
    ),
    "atmosphere": ((480, 183e9, 976562.5, 0.2, 1.0), (337.9050998016982, 91, 4, 1)),
    "tdm": ((120, 230e9, 15.625e6, 0.06, 0.3), (896.183090778417, 23, 0, 1)),
    "tdm:VLA": ((120, 230e9, 15.625e6, 0.06, 0.3, "VLA"), (896.183090778417, 23, 0, 1)),
    "eight_channels": ((8, 230e9, 976562.5, 0.2, 1.0), (268.8549272335251, 2, 4, 0)),
    "large_32": ((32, 230e9, 976562.5, 0.2, 1.0), (268.8549272335251, 6, 4, 1)),
    "imaging": (
        (480, 230e9, 976562.5, 0.3, "imaging"),
        (92.46267128374839, 91, 4, 1),
    ),
}


def _settings(window):
    xds = window["xds"]
    return find_continuum_settings(
        xds.frequency.values,
        xds.l.values,
        xds.m.values,
        float(xds.frequency.attrs["rest_frequency"]["data"]),
        xds[BEAM],
        window["transmission"],
        window["sky_temperature"],
        window["telescope_name"],
        window["single_continuum"],
        window["n_bin"],
        window["spectral_dynamic_range_bandwidth"],
    )


def _no_reduction(requests):
    raise AssertionError(f"the settings asked for cube reductions: {requests}")


def _port_settings(window, directory):
    """The settings of ``window`` as the simple port's findContinuum computes
    them, by its functions and lines on the port's metadata of the window."""
    xds = window["xds"]
    metadata = _cube_metadata(xds, "SKY_RESIDUAL", BEAM, window["telescope_name"])
    stem = os.path.join(directory, "window")
    image, psf = stem + ".residual", stem + ".psf"
    cubes = {image: {"metadata": metadata}, psf: {"metadata": metadata}}
    for path in cubes:
        os.makedirs(path)
    cubes[image]["atmosphere"] = {
        "transmission": np.asarray(window["transmission"], dtype=np.float64),
        "tsky": np.asarray(window["sky_temperature"], dtype=np.float64),
    }
    with casa_image_tasks.cube_session(cubes, _no_reduction), quiet():
        image_info = findContinuum.getImageInfo(image)
        chan_info = findContinuum.numberOfChannelsInCube(
            image, returnFreqs=True, returnChannelWidth=True
        )
        psf_info = findContinuum.getImageInfo(psf)
        variation = findContinuum.atmosphereVariation(
            image,
            image_info,
            chan_info,
            airmass=DEFAULTS["airmass"],
            pwv=DEFAULTS["pwv"],
        )
    nchan, first, last, increment = chan_info
    width = abs(increment)
    telescope, max_baseline = image_info[12], psf_info[11]
    with quiet():
        rounds = ROUNDS(
            DEFAULTS["amendMaskIterations"], width, nchan, telescope, max_baseline
        )["amendMaskIterations"]
        rounds_enabled = ROUNDS_ENABLED(rounds)["amendMask"]
        channels = casa_image_tasks.parse_channel_selection(
            PRIMARY_BEAM_MEAN_CHANNELS(nchan)["pbmomChans"], nchan
        )
        return {
            "n_channels": nchan,
            "first_frequency": first,
            "last_frequency": last,
            "channel_increment": increment,
            "channel_width": width,
            "mean_frequency": MEAN_FREQUENCY(first, last)["meanFreqHz"],
            "rest_frequency": metadata["rest_frequency"],
            "moment_channel_weights": moment_velocity_widths(
                linear_frequency_axis(xds.frequency.values),
                metadata["rest_frequency"],
            ),
            "n_l": image_info[5],
            "n_m": image_info[6],
            "cell_l_arcsec": image_info[3],
            "cell_m_arcsec": image_info[4],
            "beam_major_arcsec": psf_info[0],
            "beam_minor_arcsec": psf_info[1],
            "max_baseline": max_baseline,
            "is_time_division_mode": findContinuum.tdmSpectrum(width, nchan),
            "is_time_division_mode_for_telescope": findContinuum.tdmSpectrum(
                width, nchan, telescope
            ),
            "n_bin": N_BIN(
                nchan,
                width,
                window["n_bin"],
                window["spectral_dynamic_range_bandwidth"],
                DEFAULTS["vis"],
                image,
                DEFAULTS["verbose"],
                max_baseline,
            )["nbin"],
            "n_rounds_after_original": rounds,
            "rounds_enabled": rounds_enabled,
            "transmission_variation": variation[1],
            "sky_temperature_std": variation[4],
            "bad_atmosphere": BAD_ATMOSPHERE(
                variation[1],
                variation[4],
                DEFAULTS["skyTransmissionThreshold"],
                DEFAULTS["skyTempThreshold"],
            )["badAtmosphere"],
            "moment_difference_level": MOMENT_DIFFERENCE_LEVEL(
                rounds_enabled,
                DEFAULTS["checkIfMaskWouldHaveBeenAmended"],
                max_baseline,
                window["single_continuum"],
                DEFAULTS["momDiffLevel"],
            )["momDiffLevel"],
            "n_baseline_channels": N_BASELINE_CHANNELS(
                DEFAULTS["nBaselineChannels"], False, chan_info
            )["nBaselineChannels"],
            "single_continuum": window["single_continuum"],
            "primary_beam_mean_channels": tuple(range(nchan))
            if channels is None
            else tuple(channels.tolist()),
        }


def _assert_equal_settings(settings, expected):
    """Every field equal bit for bit, of the field's type."""
    for field, annotation in FindContinuumSettings.__annotations__.items():
        value = getattr(settings, field)
        assert type(value) is annotation, field
        if field == "primary_beam_mean_channels":
            assert value == expected[field]
        else:
            assert np.array_equal(
                np.float64(value), np.float64(expected[field]), equal_nan=True
            ), field


@pytest.mark.parametrize("name", REFERENCE_WINDOWS)
def test_settings_of_the_reference_windows_equal_the_ports(name, tmp_path):
    arguments, known = REFERENCE_WINDOWS[name]
    window = _reference_window(*arguments)
    settings = _settings(window)
    _assert_equal_settings(settings, _port_settings(window, tmp_path))
    assert (
        settings.max_baseline,
        settings.n_baseline_channels,
        settings.n_rounds_after_original,
        settings.n_bin,
    ) == known
    assert settings.bad_atmosphere == name.endswith("badatm")


@pytest.mark.parametrize(
    "n_channels, channel_width, is_time_division_mode",
    [(480, 976562.5, False), (120, 15.625e6, True)],
)
def test_settings_of_7m_single_continuum_windows_equal_the_ports(
    n_channels, channel_width, is_time_division_mode, tmp_path
):
    """8 arcsec beams at 230 GHz: baselines up to 60 m, and up to 400 m in
    time-division mode, keep the rounds; single continuum raises the level."""
    window = _reference_window(n_channels, 230e9, channel_width, 1.0, 8.0)
    window["single_continuum"] = True
    settings = _settings(window)
    _assert_equal_settings(settings, _port_settings(window, tmp_path))
    assert settings.is_time_division_mode_for_telescope is is_time_division_mode
    assert settings.max_baseline <= 60
    assert (settings.n_rounds_after_original, settings.moment_difference_level) == (
        4,
        11.5,
    )


def test_settings_of_random_windows_equal_the_ports(tmp_path):
    rng = np.random.default_rng(11)
    for index in range(40):
        window = _random_window(rng)
        _assert_equal_settings(
            _settings(window), _port_settings(window, tmp_path / str(index))
        )


def test_linear_spectral_axis_extrapolates_the_first_increment():
    frequency = np.linspace(230e9, 231.234567e9, 480)
    n_channels, first, last, increment = linear_spectral_axis(frequency)
    assert (n_channels, first, increment) == (480, 230e9, frequency[1] - frequency[0])
    assert last == first + 479 * increment != frequency[-1]
    # A window of one channel, as the imager's last window can be.
    assert linear_spectral_axis(frequency[3:4]) == (1, frequency[3], frequency[3], 1.0)
    assert linear_frequency_axis(frequency[3:4])[:2] == (frequency[3], 1.0)


def test_moment_channel_weights_equal_the_ports():
    rng = np.random.default_rng(12)
    for _ in range(300):
        n_channels = pick(rng, [1, 2, 3, *N_CHANNELS])
        first = rng.uniform(1e9, 900e9)
        increment = pick(rng, CHANNEL_WIDTHS) * pick(rng, [1.0, -1.0])
        increment *= 1 + rng.uniform(-1e-3, 1e-3)
        rest_frequency = first * rng.uniform(0.98, 1.02)
        weights = moment_channel_weights(first, increment, n_channels, rest_frequency)
        assert weights.dtype == np.float64 and weights.shape == (n_channels,)
        np.testing.assert_array_equal(
            weights,
            moment_velocity_widths((first, increment, 0.0, n_channels), rest_frequency),
        )


def test_max_baseline_equals_findcontinuums_arithmetic():
    """getImageInfo's arithmetic on the port's beam in arcsec: the root is the
    C pow of a NumPy scalar, which differs from sqrt in some of these cases on
    platforms whose pow is not correctly rounded (3 of them on macOS arm64)."""
    rng = np.random.default_rng(15)
    for _ in range(5000):
        major, minor = sorted(10 ** rng.uniform(-7.5, -4.5, 2), reverse=True)
        first_frequency = rng.uniform(1e9, 950e9)
        expected = MAX_BASELINE(
            np.float64(np.degrees(major) * 3600.0),
            np.float64(np.degrees(minor) * 3600.0),
            np.float64(first_frequency) * 1e-9,
            IMAGE_INFO_CONSTANTS["c_mks"],
            IMAGE_INFO_CONSTANTS["ARCSEC_PER_RAD"],
        )["maxBaseline"]
        fits = _beam_fits([[[major, minor, 0.0]]])
        assert max_baseline_from_beam_fits(fits, first_frequency)[0] == expected


def test_max_baseline_widens_float32_beam_fits_first():
    beams = np.zeros((2, 1, 3), np.float32)
    beams[:, 0, 0], beams[:, 0, 1] = IMAGING_MAJOR, IMAGING_MINOR
    expected = (92.46267128374839, 4.5644650739476855, 1.8523081577192826)
    assert max_baseline_from_beam_fits(_beam_fits(beams), 230e9) == expected
    assert max_baseline_from_beam_fits(_beam_fits(beams.astype(float)), 230e9) == (
        expected
    )
    # Converted in float32, the major axis would be another value.
    assert float(np.median(np.degrees(IMAGING_MAJOR) * 3600.0)) != expected[1]


@pytest.mark.parametrize(
    "major, minor, max_baseline",
    [(1.0, 0.0, np.inf), (0.0, 1.0, 0.0), (np.nan, np.nan, 0.0)],
)
def test_max_baseline_of_degenerate_beams_equals_the_ports(
    major, minor, max_baseline, tmp_path
):
    """findContinuum's arithmetic on beams the input checks refuse."""
    beams = np.zeros((8, 1, 3))
    beams[..., 0], beams[..., 1] = major * RAD_PER_ARCSEC, minor * RAD_PER_ARCSEC
    window = _reference_window(8, 230e9, 976562.5, 0.2, 1.0)
    window["xds"][BEAM] = _beam_fits(beams).drop_vars("polarization")
    with np.errstate(divide="ignore", invalid="ignore"):
        settings = _settings(window)
        _assert_equal_settings(settings, _port_settings(window, tmp_path))
    assert settings.max_baseline == max_baseline


def test_max_baseline_refuses_other_times_and_no_stokes_i():
    beams = np.full((2, 8, 1, 3), RAD_PER_ARCSEC)
    two_times = xr.DataArray(beams, dims=BEAM_DIMS, coords={"polarization": ["I"]})
    with pytest.raises(ValueError, match="one time"):
        max_baseline_from_beam_fits(two_times, 230e9)
    with pytest.raises(ValueError, match="Stokes I"):
        max_baseline_from_beam_fits(_beam_fits(beams[0], ("XX",)), 230e9)


def test_time_division_mode_equals_findcontinuum():
    widths = sorted(
        {
            *CHANNEL_WIDTHS,
            1e6,
            0.97e6,
            np.nextafter(0.97e6, 0),
            7.5e6,
            np.nextafter(7.5e6, 0),
            15e6,
            np.nextafter(15e6, 0),
        }
    )
    with quiet():
        for width, n_channels in itertools.product(
            widths, [8, 61, 62, 63, 120, 128, 239, 240, 241, 480, 3840]
        ):
            for telescope_name in ("ALMA", "VLA"):
                expected = findContinuum.tdmSpectrum(width, n_channels, telescope_name)
                value = is_time_division_mode(width, n_channels, telescope_name)
                assert value is expected, (width, n_channels, telescope_name)
            expected = findContinuum.tdmSpectrum(width, n_channels)
            assert is_time_division_mode(width, n_channels) is expected


def test_continuum_n_bin_equals_findcontinuum():
    """The half-channel ties of 14 channel widths, random bandwidths, values up
    to 1000 (taken as GHz) and n_bin without a bandwidth."""
    rng = np.random.default_rng(13)
    widths = {15.625e6 / 2**k for k in range(8)} | {
        30517.578125,
        61035.15625,
        122070.3125,
        564453.125,
        282226.5625,
        141113.28125,
        70556.640625,
    }
    ties = [
        (3840, width, 1, (k + 0.5) * width)
        for width in sorted(widths)
        for k in range(60)
    ]
    # The bandwidth's trip from Hz to GHz and back flips 74 of the 840 ties
    # (S1 §9 trap 10).
    assert sum(continuum_n_bin(*tie) != round(tie[3] / tie[1]) for tie in ties) == 74
    cases = list(ties)
    for _ in range(400):
        cases.append(
            (
                pick(rng, N_CHANNELS),
                pick(rng, CHANNEL_WIDTHS),
                pick(rng, [1, 2, 3, 4, 8, 13, 40, 300]),
                pick(
                    rng,
                    [
                        None,
                        None,
                        float(rng.uniform(1e4, 2e9)),
                        float(rng.uniform(1e-3, 1000.0)),
                        int(rng.integers(1001, 10**9)),
                    ],
                ),
            )
        )
    with quiet():
        for n_channels, width, n_bin, bandwidth in cases:
            value = continuum_n_bin(n_channels, width, n_bin, bandwidth)
            expected = N_BIN(
                n_channels,
                width,
                n_bin,
                bandwidth,
                DEFAULTS["vis"],
                "",
                DEFAULTS["verbose"],
                0.0,
            )
            assert type(value) is int and value == expected["nbin"]
    assert continuum_n_bin(240, 15.625e6, 1, 226.5625e6) == 15
    assert round(226.5625e6 / 15.625e6) == 14
    assert continuum_n_bin(480, 976562.5, 1, 2.5 * 976562.5) == 2
    assert continuum_n_bin(480, 976562.5, 1, 0.4 * 976562.5) == 0
    assert continuum_n_bin(480, 976562.5, 1, 5.1e6) == 5
    assert continuum_n_bin(12, 976562.5, 1, None) == 0
    assert continuum_n_bin(480, 976562.5, 4, None) == 4


def test_rounds_and_moment_difference_level_equal_findcontinuum():
    max_baselines = [0, 33.2, 59.999999, 60, 60.000001, 268.85, 400, 400.0000001]
    windows = [(976562.5, 480), (15.625e6, 120), (7.8125e6, 241), (7.8125e6, 240)]
    cases = itertools.product(
        windows, ("ALMA", "VLA"), [*max_baselines, np.nan], (False, True)
    )
    with quiet():
        for (width, n_channels), telescope_name, max_baseline, single in cases:
            expected = ROUNDS(
                DEFAULTS["amendMaskIterations"],
                width,
                n_channels,
                telescope_name,
                max_baseline,
            )
            rounds = rounds_after_original(
                is_time_division_mode(width, n_channels, telescope_name), max_baseline
            )
            assert rounds == expected["amendMaskIterations"]
            enabled = ROUNDS_ENABLED(rounds)["amendMask"]
            assert enabled == (rounds != 0)
            expected = MOMENT_DIFFERENCE_LEVEL(
                enabled,
                DEFAULTS["checkIfMaskWouldHaveBeenAmended"],
                max_baseline,
                single,
                DEFAULTS["momDiffLevel"],
            )
            level = initial_moment_difference_level(enabled, max_baseline, single)
            assert level == expected["momDiffLevel"]


def test_is_bad_atmosphere_equals_findcontinuum():
    rng = np.random.default_rng(14)
    variations = [0.0, 0.4, np.nextafter(0.4, 0), np.nextafter(0.4, 1), np.nan]
    deviations = [0.0, 5.0, np.nextafter(5.0, 0), np.nextafter(5.0, 9), np.nan]
    with quiet():
        for variation, deviation in itertools.product(
            [*variations, *rng.uniform(0, 1, 10)], [*deviations, *rng.uniform(0, 9, 10)]
        ):
            expected = BAD_ATMOSPHERE(
                variation,
                deviation,
                DEFAULTS["skyTransmissionThreshold"],
                DEFAULTS["skyTempThreshold"],
            )
            assert is_bad_atmosphere(variation, deviation) is expected["badAtmosphere"]


def test_n_baseline_channels_equal_findcontinuum():
    with quiet():
        for n_channels in range(1, 4001):
            chan_info = (n_channels, 230e9, 231e9, 976562.5)
            expected = N_BASELINE_CHANNELS(
                DEFAULTS["nBaselineChannels"], False, chan_info
            )
            assert n_baseline_channels(n_channels) == expected["nBaselineChannels"]
    # 0.19 n is k + 1/2 for n = 50 (mod 100): rounded half up, not to even.
    assert [n_baseline_channels(n) for n in (8, 150, 350, 480)] == [2, 29, 67, 91]


def test_primary_beam_mean_channels_equal_the_ports():
    for n_channels in [*range(1, 200), 480, 3840, 15360]:
        channels = casa_image_tasks.parse_channel_selection(
            PRIMARY_BEAM_MEAN_CHANNELS(n_channels)["pbmomChans"], n_channels
        )
        assert primary_beam_mean_channels(n_channels) == (
            tuple(range(n_channels)) if channels is None else tuple(channels.tolist())
        )
