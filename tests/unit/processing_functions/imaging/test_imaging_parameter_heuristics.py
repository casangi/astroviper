"""Unit tests for the image geometry and polarization heuristics of a single field cube."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    DEFAULT_GRIDDING_KERNEL_PARAMS,
    make_gridding_kernel,
)
from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
    airy_disk_response,
    casa_airy_disk_response,
)
from astroviper.processing_functions.imaging.utils.fft_sizing import (
    next_fft_friendly_size,
)
from astroviper.processing_functions.imaging.utils.imaging_parameter_heuristics import (
    BEAM_FIT_AGREEMENT,
    FFT_PADDING,
    FIELD_OF_VIEW_MARGIN,
    GRIDDING_KERNEL_PARAMS,
    REFIT_PIXELS_PER_BEAM,
    REFIT_SAMPLING_RATIO,
    SPEED_OF_LIGHT,
    UNIFORM_DISK_HALF_POWER_WIDTH,
    beam_refit_cells,
    cell_size_from_beam,
    cell_size_guess,
    default_robust,
    field_of_view,
    image_size_for_field_of_view,
    instrument_polarization_basis_from_correlations,
    longest_antenna_separation,
    max_cell_size_for_uv_coverage,
    plausible_beam_fits,
    primary_beam_radius,
    primary_beam_telescope,
    refitted_minor_axis,
    stokes_for_correlations,
)
from astroviper.processing_functions.simulation.antenna_beams import (
    polynomial_beam_response,
)
from astroviper.utils.beam_models import (
    EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ,
    airy_disk_model,
    read_beam_polynomial_coefficients,
)


@pytest.mark.parametrize(
    "telescope_name, dish_diameters, expected",
    [
        ("ALMA", [12.0, 12.0], "alma"),
        ("ALMA", [12.0, 7.0], "aca"),  # mixed: the smallest dish
        ("ALMA", None, "alma"),
        ("ACA", None, "aca"),
        ("alma", [7.0], "aca"),
        ("VLA", [25.0], "vla"),
        ("EVLA", None, "vla"),
        ("ngVLA", [18.0], "ngvla"),
    ],
)
def test_primary_beam_telescope(telescope_name, dish_diameters, expected):
    assert primary_beam_telescope(telescope_name, dish_diameters) == expected


def test_primary_beam_telescope_unknown():
    with pytest.raises(ValueError, match="no primary beam model"):
        primary_beam_telescope("MeerKAT")


def test_default_robust():
    assert default_robust("alma") == 0.5
    assert default_robust("aca") == 0.5
    assert default_robust("vla") == 2.0
    with pytest.raises(ValueError, match="pass robust"):
        default_robust("ngvla")


@pytest.mark.parametrize(
    "correlations, basis, stokes",
    [
        (["XX", "YY"], "linear", ["I", "Q"]),
        (["YY", "XX"], "linear", ["I", "Q"]),
        (["XX", "XY", "YX", "YY"], "linear", ["I", "Q", "U", "V"]),
        (["RR", "LL"], "circular", ["I", "V"]),
        (["RR", "RL", "LR", "LL"], "circular", ["I", "Q", "U", "V"]),
    ],
)
def test_stokes_for_correlations(correlations, basis, stokes):
    assert instrument_polarization_basis_from_correlations(correlations) == basis
    assert stokes_for_correlations(correlations) == (basis, stokes)


@pytest.mark.parametrize("correlations", [["XX", "RR"], ["I"], ["XX"], ["RR", "RL"]])
def test_stokes_for_correlations_invalid(correlations):
    with pytest.raises(ValueError):
        stokes_for_correlations(correlations)


@pytest.mark.parametrize("telescope", ["alma", "aca", "ngvla"])
@pytest.mark.parametrize("level", [0.5, 0.2, 0.05])
def test_primary_beam_radius_is_the_level_of_the_imager_beam(telescope, level):
    """The imager's power pattern equals ``level`` at the radius and lies above
    it inside."""
    model = airy_disk_model(telescope)
    frequency = 100e9
    radius = primary_beam_radius(frequency, level, {"telescope": telescope})

    def beam(r):
        return float(
            airy_disk_response(
                r,
                0.0,
                frequency,
                model["dish_diameter"],
                model["blockage_diameter"],
                ipower=2,
            )
        )

    assert beam(radius * 0.999) >= level > beam(radius * 1.001)
    assert np.all([beam(r) > level for r in np.linspace(0.0, radius * 0.999, 50)])
    # Scales with the wavelength.
    assert primary_beam_radius(
        2 * frequency, level, {"telescope": telescope}
    ) == pytest.approx(radius / 2, rel=2e-3)


def _uncut_airy_disk_radius(frequency, level, model, n_scan=4096):
    """The radius of CASA's Airy power pattern without the cut at max_rad_1GHz,
    on the scan of primary_beam_radius."""
    from scipy.optimize import brentq

    def excess(radius):
        return (
            float(
                casa_airy_disk_response(
                    radius,
                    0.0,
                    frequency,
                    model["dish_diameter"],
                    model["blockage_diameter"],
                    model["max_rad_1GHz"],
                    ipower=2,
                )
            )
            - level
        )

    radii = np.linspace(
        0.0, 1.5 * SPEED_OF_LIGHT / frequency / model["dish_diameter"], n_scan
    )
    below = np.flatnonzero([excess(radius) < 0.0 for radius in radii])
    return float(brentq(excess, radii[below[0] - 1], radii[below[0]]))


@pytest.mark.parametrize("telescope", ["alma", "aca", "ngvla"])
@pytest.mark.parametrize("level", [0.5, 0.2, 0.05])
@pytest.mark.parametrize("frequency", [100e9, 230e9])
def test_primary_beam_radius_of_an_airy_disk_is_that_without_the_cut(
    telescope, level, frequency
):
    """For CASA's Airy disk the main lobe crosses the level inside max_rad_1GHz,
    so the zero beyond it leaves the radius unchanged to the bit."""
    casa = airy_disk_model(telescope, func="casa_airy")
    assert primary_beam_radius(
        frequency, level, {"beam_model": casa}
    ) == _uncut_airy_disk_radius(frequency, level, casa)


@pytest.mark.parametrize("level", [0.5, 0.2, 0.05])
def test_primary_beam_radius_is_the_level_of_the_imager_vla_beam(level):
    """For "vla" the imager's beam is the EVLA polynomial at the exact radius;
    100 and 200 GHz lie in no band, so both take the last entry (43948 MHz)."""
    frequency = 100e9
    radius = primary_beam_radius(frequency, level, {"telescope": "vla"})
    coefficients = (
        read_beam_polynomial_coefficients("EVLA_").BPC.sel(frequency=43948e6).values[0]
    )

    def beam(r):
        return float(
            polynomial_beam_response(
                r,
                0.0,
                frequency,
                coefficients,
                EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ,
                2,
                n_sample=None,
            )
        )

    assert beam(radius * 0.999) >= level > beam(radius * 1.001)
    assert np.all([beam(r) > level for r in np.linspace(0.0, radius * 0.999, 50)])
    assert primary_beam_radius(
        2 * frequency, level, {"telescope": "vla"}
    ) == pytest.approx(radius / 2, rel=2e-3)


@pytest.mark.parametrize(
    "frequency, radius_tclean",
    [(1.5e9, 0.7405), (6e9, 0.7395), (22e9, 0.7484), (33e9, 0.7410), (45e9, 0.7355)],
)
def test_primary_beam_radius_vla_is_that_of_tclean(frequency, radius_tclean):
    """The 0.2 radius in lambda / 25 m of tclean's .pb for an EVLA measurement
    set (CASA 6.7.5), for CASA's lookup table."""
    radius = primary_beam_radius(
        frequency,
        0.2,
        {"beam_model": read_beam_polynomial_coefficients("EVLA_", n_sample=10000)},
    )
    assert radius / (SPEED_OF_LIGHT / frequency / 25.0) == pytest.approx(
        radius_tclean, rel=1e-3
    )


def test_primary_beam_radius_vla_below_the_beam_at_its_cut():
    """At 350 MHz the P-band polynomial is still 0.137 at the table radius, where
    the imager's beam is cut to zero, so the 0.1 radius is that cut."""
    frequency = 350e6
    radius = primary_beam_radius(frequency, 0.1, {"telescope": "vla"})
    assert radius == pytest.approx(
        EVLA_BEAM_POLYNOMIAL_MAX_RAD_1GHZ / (frequency / 1e9), rel=1e-9
    )


def test_primary_beam_radius_alma_half_power_width_and_default():
    """ALMA 12 m half power width at 100 GHz is about 1.13 lambda / 12 m (60 arcsec)."""
    width = 2 * primary_beam_radius(100e9, 0.5) * 180 / np.pi * 3600
    assert width == pytest.approx(60.0, abs=1.0)
    assert primary_beam_radius(100e9, 0.5) == primary_beam_radius(
        100e9, 0.5, {"telescope": "alma"}
    )
    with pytest.raises(ValueError):
        primary_beam_radius(100e9, 1.0)


def test_field_of_view():
    radius = primary_beam_radius(230e9, 0.2, {"telescope": "aca"})
    assert field_of_view(230e9, 0.2, {"telescope": "aca"}) == pytest.approx(
        FIELD_OF_VIEW_MARGIN * 2 * radius
    )


def test_longest_antenna_separation_and_cell_size_guess():
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    assert longest_antenna_separation(positions) == pytest.approx(np.sqrt(26.0))
    u_max = 1000.0 * 3e9 / SPEED_OF_LIGHT
    assert cell_size_guess(1000.0, 3e9) == pytest.approx(1.0 / (3.0 * u_max))


@pytest.mark.parametrize("fov, cell", [(1.0, 0.0013), (1e-3, 1e-6), (0.5, 0.5)])
def test_image_size_for_field_of_view(fov, cell):
    size = image_size_for_field_of_view(fov, cell)
    assert size % 2 == 0
    assert size * cell >= fov
    assert size == next_fft_friendly_size(int(np.ceil(fov / cell)), even=True)


def test_cell_size_from_beam():
    assert cell_size_from_beam(1e-5, 5.0) == pytest.approx(2e-6)
    with pytest.raises(ValueError):
        cell_size_from_beam(1e-5, 0.0)


#: Longest baseline (wavelengths) and image of the stand-in PSF passes: the minor
#: axis is at least 1 / (3 u_max) = 3.33e-5 rad and the major axis shorter than
#: 200 cells of 1e-6 rad.
U_MAX = 1e4
IMAGE_SIZE = 200
CELL = 1e-6


def _point_spread_function_pass(
    stokes_i_beams, frequency, uv_distance_max, image_size=IMAGE_SIZE, cell=CELL
):
    """The variables and attributes of a PSF pass that the beam fit heuristics
    read; Stokes V has a beam of twice the Stokes I one."""
    beams = np.asarray(stokes_i_beams, dtype=float).reshape(-1, 3)
    return xr.Dataset(
        {
            "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": (
                ("frequency", "polarization", "beam_params_label"),
                np.stack([beams, 2.0 * beams], axis=1),
            ),
            "UV_DISTANCE_MAX": ("frequency", np.asarray(uv_distance_max, float)),
        },
        coords={
            "frequency": np.atleast_1d(frequency),
            "polarization": ["I", "V"],
            "beam_params_label": ["major", "minor", "pa"],
        },
        attrs={"image_size": [image_size, image_size], "cell_size": [-cell, cell]},
    )


def test_plausible_beam_fits():
    """Finite, 0 < minor <= major, the minor axis at least 1 / (3 u_max) and the
    major axis shorter than the image; by label, whatever the order of the dims."""
    limit = 1.0 / (3.0 * U_MAX)
    extent = IMAGE_SIZE * CELL
    cases = [
        ([1e-4, 5e-5, 0.3], U_MAX, True),
        ([1e-4, limit, 0.3], U_MAX, True),
        ([1e-4, 0.99 * limit, 0.3], U_MAX, False),
        ([0.99 * extent, 5e-5, 0.3], U_MAX, True),
        ([extent, 5e-5, np.pi / 2], U_MAX, False),
        ([1e-4, np.nan, 0.3], U_MAX, False),
        ([1e-4, 5e-5, np.inf], U_MAX, False),
        ([0.0, 0.0, 0.0], U_MAX, False),
        ([5e-5, 1e-4, 0.3], U_MAX, False),
        ([1e-4, 5e-5, 0.3], np.nan, False),
    ]
    xds = _point_spread_function_pass(
        [beam for beam, _, _ in cases],
        3e9 + 1e6 * np.arange(len(cases)),
        [u_max for _, u_max, _ in cases],
    )
    expected = [plausible for _, _, plausible in cases]
    plausible = plausible_beam_fits(xds)
    np.testing.assert_array_equal(plausible.sel(polarization="I").values, expected)
    np.testing.assert_array_equal(
        plausible_beam_fits(xds.transpose()).sel(polarization="I").values, expected
    )
    # Stokes V: twice the Stokes I beams, so the beam at 0.99 of the image is not.
    assert not plausible.sel(polarization="V").values[3]


def _beams(minor_axes, pa=0.3):
    minor_axes = np.asarray(minor_axes, dtype=float)
    return np.stack([2.0 * minor_axes, minor_axes, np.full(minor_axes.size, pa)], 1)


def test_beam_refit_cells_from_the_plausible_fits():
    """The highest channel with data is refitted, here one whose fit is junk
    (longer than the image); the cells put REFIT_PIXELS_PER_BEAM and
    REFIT_SAMPLING_RATIO times as many pixels across the median of the
    plausible minor axes scaled to its frequency, which an off fit does not move."""
    frequency = np.array([3.0e9, 3.1e9, 3.2e9, 3.3e9, 3.4e9, 3.5e9])
    minor_axis = 5e-5  # at 3.4 GHz
    beams = _beams(minor_axis * 3.4e9 / frequency)
    beams[2, 1] *= 1.1
    beams[4] = [1.0, 3e-4, np.pi / 2]
    beams[5] = 0.0
    uv_distance_max = np.full(frequency.size, U_MAX)
    uv_distance_max[5] = np.nan
    refit_frequency, estimate, cells = beam_refit_cells(
        _point_spread_function_pass(beams, frequency, uv_distance_max)
    )
    assert refit_frequency == 3.4e9
    assert estimate == pytest.approx(minor_axis, rel=1e-12)
    np.testing.assert_allclose(
        cells,
        [
            estimate / REFIT_PIXELS_PER_BEAM,
            estimate / (REFIT_PIXELS_PER_BEAM * REFIT_SAMPLING_RATIO),
        ],
        rtol=1e-15,
    )


def test_beam_refit_cells_without_a_plausible_fit():
    """Without a plausible fit the cells sample the half power width of a
    uniformly weighted disk out to the longest baseline of the channel."""
    frequency = np.array([3.0e9, 3.4e9])
    beams = [[1.0, 3e-4, np.pi / 2], [np.nan, np.nan, np.nan]]
    refit_frequency, estimate, cells = beam_refit_cells(
        _point_spread_function_pass(beams, frequency, [0.9 * U_MAX, U_MAX])
    )
    width = UNIFORM_DISK_HALF_POWER_WIDTH / U_MAX
    assert (refit_frequency, estimate) == (3.4e9, None)
    np.testing.assert_allclose(
        cells,
        [
            width / REFIT_PIXELS_PER_BEAM,
            width / (REFIT_PIXELS_PER_BEAM * REFIT_SAMPLING_RATIO),
        ],
        rtol=1e-15,
    )
    with pytest.raises(ValueError, match="No channel has data"):
        beam_refit_cells(
            _point_spread_function_pass(beams, frequency, [np.nan, np.nan])
        )


def _refit(beam, image_size=IMAGE_SIZE, cell=CELL):
    return _point_spread_function_pass([beam], 3.4e9, [U_MAX], image_size, cell)


def test_refitted_minor_axis_of_two_refits_that_agree():
    """Two plausible refits within a factor BEAM_FIT_AGREEMENT give the smaller
    minor axis, whatever the estimate."""
    minor_axis = 2.0**-14
    first = _refit([1.5e-4, minor_axis, 0.3])
    agreeing = _refit([1.5e-4, BEAM_FIT_AGREEMENT * minor_axis, 0.3], 250, 0.8e-6)
    assert refitted_minor_axis([first, agreeing], None) == minor_axis
    assert refitted_minor_axis([agreeing, first], 3.0 * minor_axis) == minor_axis
    too_far = np.nextafter(BEAM_FIT_AGREEMENT * minor_axis, np.inf)
    with pytest.raises(ValueError, match="give no plausible minor axis"):
        refitted_minor_axis([first, _refit([1.5e-4, too_far, 0.3])], None)


@pytest.mark.parametrize(
    "junk",
    [
        # longer than the image, the position angle at the bound of the fit
        [1.0, 2.0e-4, np.pi / 2],
        # plausible, but an axis-aligned fit about three times too wide
        [1.4e-4, 1.2e-4, np.pi / 2],
    ],
)
def test_refitted_minor_axis_of_a_junk_and_a_good_refit(junk):
    """A refit that is not plausible, or that agrees with nothing, does not
    count: the good refit counts through the estimate of the PSF pass, and
    without the estimate nothing counts."""
    good = _refit([1.0e-4, 4.0e-5, 0.96])
    for refits in ([_refit(junk), good], [good, _refit(junk)]):
        assert refitted_minor_axis(refits, 4.6e-5) == 4.0e-5
        for estimate in (None, 2.0e-4):
            with pytest.raises(ValueError, match="give no plausible minor axis"):
                refitted_minor_axis(refits, estimate)


def test_refitted_minor_axis_of_refits_that_agree_only_with_the_estimate():
    """Refits that disagree with each other count when each agrees with the
    estimate; the smaller minor axis is taken. Two refits that are not
    plausible give nothing, whatever the estimate."""
    estimate = 5.0e-5
    low = _refit([1.5e-4, 0.82 * estimate, 0.3])
    high = _refit([1.5e-4, 1.2 * estimate, 0.3])
    assert refitted_minor_axis([high, low], estimate) == 0.82 * estimate
    with pytest.raises(ValueError, match="give no plausible minor axis"):
        refitted_minor_axis([high, low], None)
    junk = [_refit([1.0, estimate, np.pi / 2]), _refit([np.nan, np.nan, np.nan])]
    with pytest.raises(ValueError, match="give no plausible minor axis"):
        refitted_minor_axis(junk, estimate)


def test_max_cell_size_for_uv_coverage():
    """1 / (2 u_max) with u_max the longest baseline of the cell guess."""
    guess = cell_size_guess(1000.0, 3e9)
    u_max = 1000.0 * 3e9 / SPEED_OF_LIGHT
    assert max_cell_size_for_uv_coverage(guess) == pytest.approx(1.0 / (2.0 * u_max))


def test_gridding_kernel_is_made_for_its_padding():
    """Least misfit 9/300/linear designed for padding 1.2, the imager's default;
    its correction at the edge of the image kept is above the floor the imager
    refuses."""
    kernel = make_gridding_kernel(GRIDDING_KERNEL_PARAMS, FFT_PADDING)
    attrs = kernel["ANTI_ALIAS"].attrs
    assert GRIDDING_KERNEL_PARAMS == DEFAULT_GRIDDING_KERNEL_PARAMS
    assert (
        attrs["kernel_type"],
        attrs["support"],
        attrs["oversampling"],
        attrs["interpolation"],
    ) == ("least_misfit", 9, 300, "linear")
    assert attrs["shape"]["fft_padding"] == FFT_PADDING == 1.2


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, "3", [3]])
def test_sampling_rejects_malformed_original_argument(value):
    with pytest.raises((TypeError, ValueError), match="pixels_per_beam"):
        cell_size_from_beam(1.0, value)


@pytest.mark.parametrize("value", [0, 1, np.nan, np.inf, True, "0.2", [0.2]])
def test_beam_boundary_rejects_malformed_original_argument(value):
    with pytest.raises((TypeError, ValueError), match="level"):
        field_of_view(1e11, value)


def test_checked_controls_skip_validation_but_measured_beam_checks_remain(monkeypatch):
    from astroviper.processing_functions.imaging.config_check import (
        param_check_residual_update as checks,
    )

    def unexpected(*args, **kwargs):
        pytest.fail("already checked sampling was validated again")

    monkeypatch.setattr(checks, "check_pixels_per_beam", unexpected)
    assert cell_size_from_beam(1.0, 5, input_checked=True) == 0.2
