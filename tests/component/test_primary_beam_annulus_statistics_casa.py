"""Component test: primary beam annulus statistics against CASA ``ia.statistics``.

The reference, ``data/casa_primary_beam_annulus_reference.npz``, was made with
``data/generate_casa_primary_beam_annulus_reference.py`` (casatools 6.7.6-14,
CASA pipeline 0091d599c); the test suite does not import CASA. It holds three
synthetic images, a Stokes I cube with the nominal levels, an I, Q, U, V cube
and a small image whose levels come from its primary beam, with what the
pipeline measures on them: the levels of ``pblimits``, the per-plane
statistics in the whole annulus, outside the clean mask of each plane and
outside the union of the clean masks, and the median, minimum and maximum of
the RMS over the continuum channels.

The numbers of kept pixels must be equal; the RMS, standard deviation, median
and median absolute deviation must agree to a relative 1e-12, since the same
pixels are kept and only the order of the sums differs. CASA reports 0 for a
plane without pixels, where AstroVIPER gives NaN.
"""

import os

import numpy as np
import pytest
from xradio.image import make_empty_sky_image

from astroviper.processing_functions.image_analysis.primary_beam_annulus_statistics import (
    calculate_primary_beam_annulus_statistics,
    primary_beam_annulus_limits,
    summarize_primary_beam_annulus_rms,
)

REFERENCE_FILE = os.path.join(
    os.path.dirname(__file__), "data", "casa_primary_beam_annulus_reference.npz"
)
SCENARIOS = ("nominal", "stokes", "edge")
RTOL = 1e-12
CASA_NAMES = {
    "rms": "rms",
    "standard_deviation": "sigma",
    "median": "median",
    "median_absolute_deviation": "medabsdevmed",
}


@pytest.fixture(scope="module")
def reference():
    return dict(np.load(REFERENCE_FILE))


def _image(reference, scenario):
    sky = reference[f"{scenario}_sky"]
    n_chan, n_pol, n_l, n_m = sky.shape
    img_xds = make_empty_sky_image(
        phase_center=[0.6, -0.2],
        image_size=[n_l, n_m],
        cell_size=[1e-5, 1e-5],
        frequency_coords=1e11 + 1e6 * np.arange(n_chan),
        pol_coords=["I", "Q", "U", "V"][:n_pol],
        time_coords=[0],
    )
    dims = ("time", "frequency", "polarization", "l", "m")
    for name, key in (
        ("SKY_RESTORED", "sky"),
        ("PRIMARY_BEAM", "primary_beam"),
        ("MASK", "mask"),
    ):
        img_xds[name] = (dims, reference[f"{scenario}_{key}"][np.newaxis])
    img_xds.attrs["data_groups"] = {
        "restored": {
            "sky": "SKY_RESTORED",
            "primary_beam": "PRIMARY_BEAM",
            "mask": "MASK",
        }
    }
    return img_xds


def _pipeline_levels(reference, scenario):
    """The levels as the pipeline writes them into its mask expressions."""
    return [float(f"{level:f}") for level in reference[f"{scenario}_annulus_limits"]]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_levels_match_pblimits(reference, scenario):
    limits = primary_beam_annulus_limits(_image(reference, scenario), (0.2, 0.3))
    assert limits == tuple(reference[f"{scenario}_annulus_limits"])


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize(
    ("variant", "mask_union", "suffix"),
    [
        ("whole", False, ""),
        ("plane_mask", False, "_outside_mask"),
        ("union_mask", True, "_outside_mask"),
    ],
)
def test_statistics_match_casa(reference, scenario, variant, mask_union, suffix):
    statistics = calculate_primary_beam_annulus_statistics(
        _image(reference, scenario),
        _pipeline_levels(reference, scenario),
        mask_union=mask_union,
    ).isel(time=0)
    casa = f"{scenario}_{variant}_"
    n_pixels = statistics["n_pixels" + suffix].values
    np.testing.assert_array_equal(n_pixels, reference[casa + "npts"])
    measured = n_pixels > 0
    for name, casa_name in CASA_NAMES.items():
        values = statistics[name + suffix].values
        np.testing.assert_allclose(
            values[measured], reference[casa + casa_name][measured], rtol=RTOL, atol=0
        )
        assert np.isnan(values[~measured]).all()
        assert (reference[casa + casa_name][~measured] == 0).all()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_summary_matches_the_pipeline(reference, scenario):
    statistics = calculate_primary_beam_annulus_statistics(
        _image(reference, scenario),
        _pipeline_levels(reference, scenario),
        mask_union=True,
    )
    summary = summarize_primary_beam_annulus_rms(
        statistics, reference[f"{scenario}_channel_indices"]
    ).isel(time=0)
    for name in ("rms_median", "rms_min", "rms_max"):
        np.testing.assert_allclose(
            summary[name].values,
            reference[f"{scenario}_summary_{name}"],
            rtol=RTOL,
            atol=0,
        )
    assert (
        summary["full_annulus"].values == reference[f"{scenario}_summary_full_annulus"]
    ).all()


def test_the_pipeline_levels_are_rounded_to_six_decimals(reference):
    img_xds = _image(reference, "edge")
    unrounded = calculate_primary_beam_annulus_statistics(
        img_xds, reference["edge_annulus_limits"]
    )
    middle_channel = unrounded["n_pixels"].isel(time=0, frequency=1).values
    assert (middle_channel != reference["edge_whole_npts"][1]).all()
