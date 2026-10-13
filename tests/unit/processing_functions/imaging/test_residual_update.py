"""Unit tests for :mod:`astroviper.processing_functions.imaging.residual_update`.

The residual update must work for whichever measurement-set data group is
imaged (``processing_set_data_group_name``), not only ``"base"``: the loaders
hand the node task a processing set that holds only the requested group, and
the imaging driver's default group is ``"corrected"``. A residual update with a
sky model degrids that model; the degrid has to read the ``uvw`` of the imaged
group, and the ``model`` visibility group it registers has to inherit the imaged
group's roles.

All data are small and synthetic (no downloads).
"""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
)
from astroviper.processing_functions.imaging.residual_update import (
    imaging_setup_single_field,
    make_visibility_model_single_field,
    residual_update_cube_single_field,
)
from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from tests.unit.processing_functions.imaging.degrid_test_datasets import (
    OVERSAMPLING,
    SUPPORT,
)
from tests.unit.processing_functions.imaging.degrid_test_datasets import (
    build_datasets as _build_degrid_datasets,
)

# Variables of the two conventional MSv4 groups (as written by the converter).
MS_DATA_GROUPS = {
    "base": {
        "correlated_data": "VISIBILITY",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "uvw": "UVW",
    },
    "corrected": {
        "correlated_data": "VISIBILITY_CORRECTED",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "uvw": "UVW",
    },
}

FREQUENCY = 1.4e9
IMAGE_SIZE = [32, 32]
CELL_SIZE = np.array([-1.0e-4, 1.0e-4])  # rad
UV_EXTENT = 150.0  # metres; well inside the padded uv grid for this cell size


def _point_source_processing_set(data_group_name, n_time=8, n_antenna=4):
    """A one-MSv4 processing set observing a 1 Jy unpolarized point source.

    The source sits at the phase centre, so every ``XX``/``YY`` visibility is
    ``1 + 0j``. As a loader returns it, the measurement set holds only the
    variables and the data group named ``data_group_name``.
    """
    rng = np.random.default_rng(0)
    antennas = [f"a{i}" for i in range(n_antenna)]
    pairs = [(a, b) for i, a in enumerate(antennas) for b in antennas[i + 1 :]]
    n_baseline = len(pairs)
    shape = (n_time, n_baseline, 1, 2)

    uvw = np.zeros((n_time, n_baseline, 3))
    uvw[..., :2] = rng.uniform(-UV_EXTENT, UV_EXTENT, (n_time, n_baseline, 2))

    data_group = MS_DATA_GROUPS[data_group_name]
    vis_dims = ("time", "baseline_id", "frequency", "polarization")
    ms_xds = xr.Dataset(
        data_vars={
            data_group["correlated_data"]: (
                vis_dims,
                np.ones(shape, dtype=np.complex128),
            ),
            data_group["flag"]: (vis_dims, np.zeros(shape, dtype=bool)),
            data_group["weight"]: (vis_dims, np.ones(shape)),
            data_group["uvw"]: (("time", "baseline_id", "uvw_label"), uvw),
        },
        coords={
            "time": np.arange(n_time, dtype=float),
            "baseline_id": np.arange(n_baseline),
            "baseline_antenna1_name": ("baseline_id", [a for a, _ in pairs]),
            "baseline_antenna2_name": ("baseline_id", [b for _, b in pairs]),
            "frequency": [FREQUENCY],
            "polarization": ["XX", "YY"],
            "uvw_label": ["u", "v", "w"],
        },
        attrs={"data_groups": {data_group_name: dict(data_group)}},
    )
    return xr.DataTree.from_dict({"ms_0": ms_xds})


def _empty_image():
    from xradio.image import make_empty_sky_image

    return make_empty_sky_image(
        phase_center=np.array([1.0, 0.5]),
        image_size=IMAGE_SIZE,
        cell_size=CELL_SIZE,
        frequency_coords=[FREQUENCY],
        pol_coords=["XX", "YY"],
        time_coords=[0],
        do_sky_coords=False,
    )


def _add_point_source_model(img_xds, flux=1.0):
    """Register a ``model`` image group holding ``flux`` Jy (Stokes I) at the centre.

    Creates the group the way the model update does: in = ``residual``,
    out = ``model`` with the ``sky`` role renamed to ``SKY_MODEL``.
    """
    _, data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name="residual",
        data_group_out_name="model",
        data_group_out_modified={"sky": "SKY_MODEL"},
        overwrite=True,
    )
    model = xr.zeros_like(img_xds["SKY_RESIDUAL"])
    centre = {
        "polarization": "I",
        "l": img_xds.l.values[np.argmin(np.abs(img_xds.l.values))],
        "m": img_xds.m.values[np.argmin(np.abs(img_xds.m.values))],
    }
    model.loc[centre] = flux
    img_xds[data_group_out["sky"]] = model
    modify_data_groups_xds(
        img_xds,
        data_group_out_name="model",
        data_group_out=data_group_out,
        description="Point-source test model.",
    )
    return img_xds


@pytest.mark.parametrize("data_group_name", ["base", "corrected"])
def test_residual_update_with_model_uses_the_imaged_data_group(data_group_name):
    """A cycle with a sky model degrids against the imaged group, not ``base``.

    Regression: ``make_visibility_model_single_field`` used to drop the group
    name, so the degridder fell back to ``"base"`` and every deconvolution
    with another group (the driver's default ``"corrected"`` included) failed
    with ``Data group base not found``.
    """
    ps_xdt = _point_source_processing_set(data_group_name)
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": np.array([1.0, 0.5]),
        "time_coords": [0],
        "polarization_coords": ["I", "Q"],
        "fft_padding": 1.2,
    }
    common = dict(
        processing_set_data_group_name=data_group_name,
        single_precision_image=False,
        fft_backend="scipy",
    )

    img_xds, _ = imaging_setup_single_field(
        ps_xdt, _empty_image(), image_params, {"weighting": "natural"}, **common
    )
    img_xds, _ = residual_update_cube_single_field(
        ps_xdt, img_xds, image_params, model_exists=False, **common
    )
    dirty_peak = float(np.abs(img_xds["SKY_RESIDUAL"].sel(polarization="I")).max())
    assert dirty_peak > 0.5

    # One model update's worth of model: the true source.
    img_xds = _add_point_source_model(img_xds)
    img_xds, _ = residual_update_cube_single_field(
        ps_xdt, img_xds, image_params, model_exists=True, **common
    )

    ms_xdt = ps_xdt["ms_0"]
    imaged_group = MS_DATA_GROUPS[data_group_name]
    ms_data_groups = ms_xdt.attrs["data_groups"]
    # The model and residual groups inherit the imaged group's roles.
    for role in ("flag", "weight", "uvw"):
        assert ms_data_groups["model"][role] == imaged_group[role]
        assert ms_data_groups["residual"][role] == imaged_group[role]
    assert ms_data_groups["model"]["correlated_data"] == "VISIBILITY_MODEL"
    assert ms_data_groups["residual"]["correlated_data"] == "VISIBILITY_RESIDUAL"

    # Degridding the true source predicts the observed visibilities, so the
    # residual visibilities and the residual image are (close to) zero.
    observed = ms_xdt[imaged_group["correlated_data"]].values
    np.testing.assert_allclose(
        ms_xdt["VISIBILITY_MODEL"].values, observed, rtol=0, atol=1e-2
    )
    residual_peak = float(np.abs(img_xds["SKY_RESIDUAL"]).max())
    assert residual_peak < 1e-2 * dirty_peak


@pytest.mark.parametrize("data_group_name", ["base", "corrected"])
def test_make_visibility_model_reads_the_input_data_group(data_group_name):
    """Every MS is degridded against ``ms_data_group_in_name``."""
    ms_list = []
    for seed in range(2):
        ms_xds, img_xds, _ = _build_degrid_datasets(seed=seed)
        # Rename the helper's "base" group to the group under test.
        ms_xds.attrs["data_groups"] = {
            data_group_name: ms_xds.attrs["data_groups"]["base"]
        }
        ms_list.append(ms_xds)
    ps_xdt = xr.DataTree.from_dict({f"ms_{i}": ms for i, ms in enumerate(ms_list)})
    cgk_1D = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

    make_visibility_model_single_field(
        ps_xdt, img_xds, cgk_1D, ms_data_group_in_name=data_group_name
    )

    for ms_xdt in ps_xdt.values():
        model_group = ms_xdt.attrs["data_groups"]["model"]
        assert model_group["correlated_data"] == "VISIBILITY_MODEL"
        assert model_group["uvw"] == "UVW"
        assert model_group["weight_imaging"] == "WEIGHT_IMAGING"
        # The helper's uniform 2+0j UV model degrids to a constant.
        np.testing.assert_allclose(
            ms_xdt["VISIBILITY_MODEL"].values, 2.0 + 0.0j, rtol=0, atol=1e-5
        )


def test_make_visibility_model_missing_input_data_group_raises():
    """An input group the measurement set does not hold is reported, not guessed."""
    ms_xds, img_xds, _ = _build_degrid_datasets()
    ps_xdt = xr.DataTree.from_dict({"ms_0": ms_xds})
    cgk_1D = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT)

    with pytest.raises(AssertionError, match="Data group corrected not found"):
        make_visibility_model_single_field(
            ps_xdt, img_xds, cgk_1D, ms_data_group_in_name="corrected"
        )
