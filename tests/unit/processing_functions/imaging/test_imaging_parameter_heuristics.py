"""Unit tests for the image geometry and polarization heuristics of a single field cube."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
    casa_airy_disk_response,
)
from astroviper.processing_functions.imaging.utils.fft_sizing import (
    next_fft_friendly_size,
)
from astroviper.processing_functions.imaging.utils.imaging_parameter_heuristics import (
    FIELD_OF_VIEW_MARGIN,
    SPEED_OF_LIGHT,
    cell_size_from_beam,
    cell_size_guess,
    default_robust,
    field_of_view,
    image_size_for_field_of_view,
    instrument_polarization_basis_from_correlations,
    longest_antenna_separation,
    primary_beam_radius,
    primary_beam_telescope,
    stokes_for_correlations,
)
from astroviper.utils.beam_models import airy_disk_model


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


@pytest.mark.parametrize("telescope", ["alma", "aca", "vla", "ngvla"])
@pytest.mark.parametrize("level", [0.5, 0.2, 0.05])
def test_primary_beam_radius_is_the_level_of_the_imager_beam(telescope, level):
    """The imager's power pattern equals ``level`` at the radius (to the CASA
    table quantisation) and lies above it inside."""
    model = airy_disk_model(telescope)
    frequency = 100e9
    radius = primary_beam_radius(frequency, level, {"telescope": telescope})

    def beam(r):
        return float(
            casa_airy_disk_response(
                r,
                0.0,
                frequency,
                model["dish_diameter"],
                model["blockage_diameter"],
                model["max_rad_1GHz"],
                ipower=2,
            )
        )

    assert beam(radius * 0.999) >= level > beam(radius * 1.001)
    assert np.all([beam(r) > level for r in np.linspace(0.0, radius * 0.999, 50)])
    # Scales with the wavelength.
    assert primary_beam_radius(
        2 * frequency, level, {"telescope": telescope}
    ) == pytest.approx(radius / 2, rel=2e-3)


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
