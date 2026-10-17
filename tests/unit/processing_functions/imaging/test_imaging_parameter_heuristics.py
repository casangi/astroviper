"""Unit tests for the image geometry and polarization heuristics of a single field cube."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
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
    FFT_PADDING,
    FIELD_OF_VIEW_MARGIN,
    GRIDDING_KERNEL_PARAMS,
    SPEED_OF_LIGHT,
    cell_size_from_beam,
    cell_size_from_beam_fit,
    cell_size_guess,
    default_robust,
    field_of_view,
    image_size_for_field_of_view,
    instrument_polarization_basis_from_correlations,
    longest_antenna_separation,
    max_cell_size_for_uv_coverage,
    primary_beam_radius,
    primary_beam_telescope,
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


def test_cell_size_from_beam_fit():
    """The Stokes I minor axis of the highest frequency channel with a valid
    fit, by label whatever the order of the dims."""
    import xarray as xr

    beam = np.zeros((3, 2, 3))  # frequency, polarization, beam_params_label
    beam[:, 0, 1] = [3e-5, 2e-5, 0.0]  # Stokes I minor axes; the top channel flagged
    beam[:, 1, 1] = 1e-5  # Stokes V
    fit = xr.DataArray(
        beam,
        dims=("frequency", "polarization", "beam_params_label"),
        coords={
            "frequency": [3e9, 3.4e9, 3.8e9],
            "polarization": ["I", "V"],
            "beam_params_label": ["major", "minor", "pa"],
        },
    )
    assert cell_size_from_beam_fit(fit, 5.0) == pytest.approx(4e-6)
    assert cell_size_from_beam_fit(fit.transpose(), 4.0) == pytest.approx(5e-6)
    fit[1, 0, 1] = np.nan
    assert cell_size_from_beam_fit(fit, 5.0) == pytest.approx(6e-6)
    with pytest.raises(ValueError, match="No channel has a valid beam fit"):
        cell_size_from_beam_fit(fit.where(fit.frequency < 3e9, 0.0), 5.0)


def test_max_cell_size_for_uv_coverage():
    """1 / (2 u_max) with u_max the longest baseline of the cell guess."""
    guess = cell_size_guess(1000.0, 3e9)
    u_max = 1000.0 * 3e9 / SPEED_OF_LIGHT
    assert max_cell_size_for_uv_coverage(guess) == pytest.approx(1.0 / (2.0 * u_max))


def test_gridding_kernel_is_made_for_its_padding():
    """Prolate spheroidal 9/300/linear designed for padding 1.2; its
    correction at the edge of the image kept is above the floor the imager
    refuses."""
    kernel = make_gridding_kernel(GRIDDING_KERNEL_PARAMS, FFT_PADDING)
    assert kernel.params == {
        "type": "prolate_spheroidal",
        "support": 9,
        "oversampling": 300,
        "interpolation": "linear",
    }
    assert kernel.shape["fft_padding"] == FFT_PADDING == 1.2
