"""GridderAntiAlias obeys the interface rules, and its methods give main's arrays.

The code paths of main that the methods replace are the C++ calls of
``add_visibility_grid_single_field`` and ``add_uv_sampling_grid_single_field``
(written out in ``_main_kernel_call``), their allocation of the uv grid, and
the functions the thin methods call: ``degrid_visibility_grid_single_field``,
``ifft_norm_img_xds``, ``fft_norm_img_xds`` and ``fft_norm_continuum_img_xds``.
Every comparison is bit for bit, with one thread.
"""

import copy

import numpy as np
import pytest
import xarray as xr
import xradio.image.image_xds  # noqa: F401  (registers xr_img)

from astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder import (
    fft_norm_continuum_img_xds,
    fft_norm_img_xds,
    ifft_norm_img_xds,
)
from astroviper.processing_functions.imaging.gridders.anti_alias.anti_alias_grid_cpp import (
    anti_alias_grid,
    anti_alias_grid_uv_sampling,
)
from astroviper.processing_functions.imaging.gridders.anti_alias.degrid_visibility_grid import (
    degrid_visibility_grid_single_field,
)
from astroviper.processing_functions.imaging.gridders.anti_alias.gridder_anti_alias import (
    GridderAntiAlias,
)
from astroviper.processing_functions.imaging.gridders.gridder_base import Gridder
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    anti_alias_kernel_arguments,
    make_gridding_kernel,
)
from astroviper.processing_functions.imaging.utils.fft_sizing import padded_grid_size
from astroviper.processing_functions.imaging.utils.frequency_mapping import (
    map_visibility_frequencies_to_image,
)

METHODS = (
    "new_grid",
    "grid",
    "grid_uv_sampling",
    "degrid",
    "uv_to_image",
    "image_to_uv",
)
CONSTANTS = (
    "name",
    "normalization",
    "gcf_terms",
    "grid_roles",
    "grid_uv_sampling_roles",
    "degrid_role",
)

N_PIXELS = 24
FFT_PADDING = 1.2
IMAGE_FREQUENCIES = np.array([1.00e9, 1.05e9, 1.10e9])
MS_FREQUENCIES = np.array([1.00e9, 1.01e9, 1.05e9, 1.10e9])
REFERENCE_FREQUENCY = 1.04e9
CORRELATIONS = {2: ["XX", "YY"], 4: ["XX", "XY", "YX", "YY"]}
KERNELS = {"default": None, "casa": {"type": "prolate_spheroidal_casa"}}
VISIBILITY_DIMS = ("time", "baseline_id", "frequency", "polarization")
GRID_DIMS = ("time", "frequency", "polarization", "u", "v")

CASES = pytest.mark.parametrize(
    "kernel_name, n_pol, complex_dtype",
    [
        (kernel_name, n_pol, complex_dtype)
        for kernel_name in KERNELS
        for n_pol in (2, 4)
        for complex_dtype in (np.complex64, np.complex128)
    ],
)
GRID_METHODS = pytest.mark.parametrize(
    "method, kernel_function",
    [("grid", anti_alias_grid), ("grid_uv_sampling", anti_alias_grid_uv_sampling)],
)


def _measurement_set(n_pol, seed):
    """Random samples with a zero weight, NaN weights, a NaN visibility, a NaN
    uvw and a sample beyond the uv grid."""
    rng = np.random.default_rng(seed)
    shape = (6, 40, MS_FREQUENCIES.size, n_pol)
    uvw = rng.uniform(-5000.0, 5000.0, (6, 40, 3))
    uvw[..., 2] *= 0.05
    uvw[0, 0] = [9500.0, 100.0, 0.0]
    uvw[1, 1] = np.nan
    visibility = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    visibility[4, 4, 1] = np.nan
    weight = rng.random(shape) + 0.1
    weight[2, 2] = 0.0
    weight[3, 3, 0] = np.nan
    ms_xds = xr.Dataset(
        {
            "VISIBILITY": (VISIBILITY_DIMS, visibility),
            "UVW": (("time", "baseline_id", "uvw_label"), uvw),
            "WEIGHT_IMAGING": (VISIBILITY_DIMS, weight),
        },
        coords={"frequency": MS_FREQUENCIES, "polarization": CORRELATIONS[n_pol]},
    )
    ms_xds.attrs["data_groups"] = {
        "corrected": {
            "correlated_data": "VISIBILITY",
            "uvw": "UVW",
            "weight_imaging": "WEIGHT_IMAGING",
        }
    }
    return ms_xds


def _image(polarizations, plane_dim="frequency"):
    coords = {
        "time": [0.0],
        "polarization": polarizations,
        "l": (np.arange(N_PIXELS) - N_PIXELS // 2) * -2.0e-5,
        "m": (np.arange(N_PIXELS) - N_PIXELS // 2) * 2.0e-5,
    }
    if plane_dim == "frequency":
        coords["frequency"] = IMAGE_FREQUENCIES
    else:
        coords[plane_dim] = np.arange(2, dtype=np.int32)
    image = xr.Dataset(coords=coords)
    image.attrs = {"type": "image_dataset", "data_groups": {"residual": {}}}
    return image


def _model_image(polarizations, plane_dim, complex_dtype, seed=7):
    """A model sky, complex in a four-correlation basis and real otherwise."""
    rng = np.random.default_rng(seed)
    image = _image(polarizations, plane_dim)
    complex_sky = polarizations == CORRELATIONS[4]
    float_dtype = np.float32 if complex_dtype == np.complex64 else np.float64
    shape = (1, image.sizes[plane_dim], len(polarizations), N_PIXELS, N_PIXELS)
    values = np.zeros(shape, dtype=complex_dtype if complex_sky else float_dtype)
    values[..., 4:20, 4:20] = rng.standard_normal((*shape[:3], 16, 16))
    if complex_sky:
        values[..., 4:20, 4:20] += 1j * rng.standard_normal((*shape[:3], 16, 16))
    image["SKY_MODEL"] = (("time", plane_dim, "polarization", "l", "m"), values)
    image.attrs["data_groups"] = {"model_in": {"sky": "SKY_MODEL"}}
    return image


def _uv_grids(n_pol, complex_dtype, seed=3):
    """Random uv grids and normalizations of both roles in the group "residual"."""
    rng = np.random.default_rng(seed)
    image = _image(CORRELATIONS[n_pol])
    n_uv = padded_grid_size([N_PIXELS, N_PIXELS], FFT_PADDING)
    shape = (1, IMAGE_FREQUENCIES.size, n_pol, n_uv[0], n_uv[1])
    roles = {**GridderAntiAlias.grid_roles, **GridderAntiAlias.grid_uv_sampling_roles}
    for name in ("VISIBILITY", "UV_SAMPLING"):
        grid = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        image[name] = (GRID_DIMS, grid.astype(complex_dtype))
        image[f"{name}_NORMALIZATION"] = (GRID_DIMS[:3], rng.random(shape[:3]) + 0.5)
    image.attrs["data_groups"]["residual"] = roles
    return image


def _main_kernel_call(
    kernel_function,
    ms_xds,
    img_xds,
    gcf_xds,
    weights,
    frequency_map,
    grid,
    normalization,
):
    """The C++ call of main's add_visibility_grid_single_field (anti_alias_grid)
    or add_uv_sampling_grid_single_field (anti_alias_grid_uv_sampling)."""
    time_map = (np.zeros(ms_xds.sizes["time"])).astype(int)
    pol_map = (np.arange(0, weights.shape[3])).astype(int)
    n_uv = padded_grid_size([img_xds.sizes["l"], img_xds.sizes["m"]], FFT_PADDING)
    delta_lm = img_xds.xr_img.get_lm_cell_size()
    table, support, oversampling, interpolation = anti_alias_kernel_arguments(gcf_xds)
    visibilities = (
        (ms_xds.VISIBILITY.values,) if kernel_function is anti_alias_grid else ()
    )
    kernel_function(
        grid,
        normalization,
        *visibilities,
        ms_xds.UVW.values,
        ms_xds.frequency.values,
        frequency_map,
        time_map,
        pol_map,
        weights,
        table,
        n_uv,
        delta_lm,
        support=support,
        oversampling=oversampling,
        interpolation=interpolation,
        processing_function_threads=1,
    )


def _assert_same_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert (
        np.ascontiguousarray(actual).tobytes()
        == np.ascontiguousarray(expected).tobytes()
    )


def _without_dates(xds):
    """The dataset with the time stamps of its data groups removed."""
    xds = xds.copy()
    xds.attrs = copy.deepcopy(xds.attrs)
    for group in xds.attrs.get("data_groups", {}).values():
        group.pop("date", None)
    return xds


def _assert_same_dataset(actual, expected):
    """Identical datasets, data-group dates aside, with the same bits in every
    data variable."""
    xr.testing.assert_identical(_without_dates(actual), _without_dates(expected))
    for name in expected.data_vars:
        _assert_same_bits(actual[name].values, expected[name].values)


# --------------------------------------------------------------------------
# Interface rules
# --------------------------------------------------------------------------


def test_anti_alias_gridder_is_a_gridder_that_cannot_be_instantiated():
    assert issubclass(GridderAntiAlias, Gridder)
    with pytest.raises(TypeError, match="GridderAntiAlias cannot be instantiated"):
        GridderAntiAlias()


@pytest.mark.parametrize("attribute", CONSTANTS + METHODS)
def test_anti_alias_gridder_attributes_cannot_be_reassigned_or_deleted(attribute):
    value = getattr(GridderAntiAlias, attribute)
    with pytest.raises(
        AttributeError, match=f"GridderAntiAlias.{attribute} cannot be reassigned"
    ):
        setattr(GridderAntiAlias, attribute, None)
    with pytest.raises(
        AttributeError, match=f"GridderAntiAlias.{attribute} cannot be deleted"
    ):
        delattr(GridderAntiAlias, attribute)
    assert getattr(GridderAntiAlias, attribute) is value


def test_anti_alias_gridder_constants():
    assert GridderAntiAlias.name == "anti_alias"
    assert GridderAntiAlias.normalization == "sum_of_weights"
    assert GridderAntiAlias.gcf_terms == ("anti_alias",)
    assert list(GridderAntiAlias.grid_roles.items()) == [
        ("visibility", "VISIBILITY"),
        ("visibility_normalization", "VISIBILITY_NORMALIZATION"),
    ]
    assert list(GridderAntiAlias.grid_uv_sampling_roles.items()) == [
        ("uv_sampling", "UV_SAMPLING"),
        ("uv_sampling_normalization", "UV_SAMPLING_NORMALIZATION"),
    ]
    assert GridderAntiAlias.degrid_role == "visibility"


@pytest.mark.parametrize("roles", ["grid_roles", "grid_uv_sampling_roles"])
def test_anti_alias_gridder_role_maps_are_read_only(roles):
    role_map = getattr(GridderAntiAlias, roles)
    before = dict(role_map)
    first_role = next(iter(role_map))
    with pytest.raises(TypeError):
        role_map[first_role] = "OTHER"
    with pytest.raises(TypeError):
        del role_map[first_role]
    assert dict(role_map) == before


@pytest.mark.parametrize("method", METHODS)
def test_anti_alias_gridder_implements_every_method_as_a_static_method(method):
    assert isinstance(GridderAntiAlias.__dict__[method], staticmethod)


# --------------------------------------------------------------------------
# The methods give the arrays of main's code paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("uv_sampling", [False, True])
@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
def test_new_grid_is_the_allocation_of_main(uv_sampling, complex_dtype):
    image = _image(CORRELATIONS[4])
    values, spatial_dims, variable_type = GridderAntiAlias.new_grid(
        image, (1, IMAGE_FREQUENCIES.size, 4), uv_sampling, FFT_PADDING, complex_dtype
    )
    n_uv = padded_grid_size([image.sizes["l"], image.sizes["m"]], FFT_PADDING)
    expected = np.zeros(
        (1, IMAGE_FREQUENCIES.size, 4, n_uv[0], n_uv[1]), dtype=complex_dtype
    )
    _assert_same_bits(values, expected)
    assert spatial_dims == ("u", "v")
    assert variable_type is None


@GRID_METHODS
@CASES
def test_grid_methods_are_the_kernel_calls_of_main(
    method, kernel_function, kernel_name, n_pol, complex_dtype
):
    """Two measurement sets accumulated into the planes of a cube."""
    gcf_xds = make_gridding_kernel(KERNELS[kernel_name], FFT_PADDING)
    image = _image(CORRELATIONS[n_pol])
    leading_shape = (1, IMAGE_FREQUENCIES.size, n_pol)
    grid, _, _ = GridderAntiAlias.new_grid(
        image, leading_shape, method == "grid_uv_sampling", FFT_PADDING, complex_dtype
    )
    normalization = np.zeros(leading_shape)
    expected_grid = grid.copy()
    expected_normalization = normalization.copy()
    for seed in (0, 1):
        ms_xds = _measurement_set(n_pol, seed)
        weights = ms_xds.WEIGHT_IMAGING.values
        frequency_map = map_visibility_frequencies_to_image(
            ms_xds.frequency.values, image.frequency.values
        )
        getattr(GridderAntiAlias, method)(
            ms_xds,
            image,
            gcf_xds,
            weights,
            frequency_map,
            grid,
            normalization,
            "corrected",
            FFT_PADDING,
            1,
        )
        _main_kernel_call(
            kernel_function,
            ms_xds,
            image,
            gcf_xds,
            weights,
            frequency_map,
            expected_grid,
            expected_normalization,
        )
    assert np.count_nonzero(normalization) == normalization.size
    _assert_same_bits(grid, expected_grid)
    _assert_same_bits(normalization, expected_normalization)


@GRID_METHODS
def test_grid_methods_accumulate_into_one_plane_of_a_grid(method, kernel_function):
    """As the MFS functions grid a Taylor term: Taylor-weighted weights, all
    channels to one plane, a view of the grid and a temporary normalization."""
    gcf_xds = make_gridding_kernel(None, FFT_PADDING)
    image = _image(CORRELATIONS[2])
    ms_xds = _measurement_set(2, 0)
    x = (ms_xds.frequency.values - REFERENCE_FREQUENCY) / REFERENCE_FREQUENCY
    weights = np.ascontiguousarray(
        ms_xds.WEIGHT_IMAGING.values * x[None, None, :, None]
    )
    frequency_map = np.zeros(MS_FREQUENCIES.size, dtype=np.int64)
    grid, _, _ = GridderAntiAlias.new_grid(
        image, (1, 3, 2), method == "grid_uv_sampling", FFT_PADDING, np.complex128
    )
    normalization = np.zeros((1, 1, 2))
    expected_grid = grid.copy()
    expected_normalization = normalization.copy()
    getattr(GridderAntiAlias, method)(
        ms_xds,
        image,
        gcf_xds,
        weights,
        frequency_map,
        grid[:, 1:2],
        normalization,
        "corrected",
        FFT_PADDING,
        1,
    )
    _main_kernel_call(
        kernel_function,
        ms_xds,
        image,
        gcf_xds,
        weights,
        frequency_map,
        expected_grid[:, 1:2],
        expected_normalization,
    )
    assert np.count_nonzero(grid[:, 1]) > 0
    assert not np.any(grid[:, [0, 2]])
    _assert_same_bits(grid, expected_grid)
    _assert_same_bits(normalization, expected_normalization)


@CASES
def test_degrid_is_degrid_visibility_grid_single_field(
    kernel_name, n_pol, complex_dtype
):
    """Two predictions: the second writes into the array the first allocated."""
    gcf_xds = make_gridding_kernel(KERNELS[kernel_name], FFT_PADDING)
    image = _image(CORRELATIONS[n_pol])
    rng = np.random.default_rng(5)
    n_uv = padded_grid_size([N_PIXELS, N_PIXELS], FFT_PADDING)
    shape = (1, IMAGE_FREQUENCIES.size, n_pol, n_uv[0], n_uv[1])
    ms_xds = _measurement_set(n_pol, 0)
    expected = copy.deepcopy(ms_xds)
    frequency_map = map_visibility_frequencies_to_image(
        ms_xds.frequency.values, image.frequency.values
    )
    arguments = {
        "ms_data_group_in_name": "corrected",
        "ms_data_group_out_name": "predicted",
        "ms_data_group_out_modified": {"correlated_data": "VISIBILITY_PREDICTED"},
        "overwrite": True,
        "fft_padding": FFT_PADDING,
        "processing_function_threads": 1,
        "description": "Model visibilities.",
    }
    for _ in range(2):
        model = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        model = model.astype(complex_dtype)
        returned = GridderAntiAlias.degrid(
            ms_xds, image, gcf_xds, model, frequency_map, **arguments
        )
        assert returned is None
        degrid_visibility_grid_single_field(
            expected, gcf_xds, image, model, frequency_map, **arguments
        )
    assert np.count_nonzero(ms_xds.VISIBILITY_PREDICTED.values) > 0
    _assert_same_dataset(ms_xds, expected)


@CASES
def test_uv_to_image_is_ifft_norm_img_xds(kernel_name, n_pol, complex_dtype):
    """Sky and point spread function, one uv grid kept and one deleted."""
    gcf_xds = make_gridding_kernel(KERNELS[kernel_name], FFT_PADDING)
    image = _uv_grids(n_pol, complex_dtype)
    expected = copy.deepcopy(image)
    arguments = {
        "image_params": {"image_size": [N_PIXELS, N_PIXELS]},
        "image_data_group_in_name": "residual",
        "image_data_group_out_name": "residual",
        "image_data_group_out_modified": {
            "sky": "SKY_DIRTY",
            "point_spread_function": "PSF_DIRTY",
        },
        "overwrite": True,
        "image_data_variables_keep": ["uv_sampling"],
        "processing_function_threads": 1,
        "fft_backend": "scipy",
        "complex_dtype": complex_dtype,
    }
    result = GridderAntiAlias.uv_to_image(image, gcf_xds=gcf_xds, **arguments)
    expected = ifft_norm_img_xds(expected, gcf_xds=gcf_xds, **arguments)
    assert result is image
    assert {"SKY_DIRTY", "PSF_DIRTY", "UV_SAMPLING"} <= set(result)
    assert "VISIBILITY" not in result
    _assert_same_dataset(result, expected)


IMAGE_TO_UV_CASES = {
    "frequency planes": ("frequency", CORRELATIONS[2], None, fft_norm_img_xds),
    "four correlations": ("frequency", CORRELATIONS[4], None, fft_norm_img_xds),
    "stokes planes combined": ("frequency", ["I", "Q"], "linear", fft_norm_img_xds),
    "taylor planes": (
        "taylor_term",
        CORRELATIONS[2],
        None,
        fft_norm_continuum_img_xds,
    ),
}


@pytest.mark.parametrize("case", IMAGE_TO_UV_CASES)
@pytest.mark.parametrize("kernel_name", KERNELS)
@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
def test_image_to_uv_is_the_forward_transform_of_main(case, kernel_name, complex_dtype):
    plane_dim, polarizations, new_polarization_basis, function = IMAGE_TO_UV_CASES[case]
    gcf_xds = make_gridding_kernel(KERNELS[kernel_name], FFT_PADDING)
    image = _model_image(polarizations, plane_dim, complex_dtype)
    expected = copy.deepcopy(image)
    arguments = {
        "image_params": {
            "image_size": [N_PIXELS, N_PIXELS],
            "fft_padding": FFT_PADDING,
        },
        "image_data_group_in_name": "model_in",
        "image_data_group_out_name": "model_uv",
        "image_data_group_out_modified": {"visibility": "VISIBILITY_MODEL_UV"},
        "overwrite": True,
        "image_data_variables_keep": ["sky"],
        "processing_function_threads": 1,
        "fft_backend": "scipy",
        "data_variables_to_process": None,
        "complex_dtype": complex_dtype,
    }
    result = GridderAntiAlias.image_to_uv(
        image,
        gcf_xds=gcf_xds,
        plane_dim=plane_dim,
        new_polarization_basis=new_polarization_basis,
        **arguments,
    )
    if function is fft_norm_img_xds:
        arguments["new_polarization_basis"] = new_polarization_basis
    expected = function(expected, gcf_xds=gcf_xds, **arguments)
    assert result is image
    assert "VISIBILITY_MODEL_UV" in result
    _assert_same_dataset(result, expected)


@pytest.mark.parametrize(
    "plane_dim, new_polarization_basis",
    [("channel", None), ("taylor_term", "linear")],
)
def test_image_to_uv_refuses_another_plane_dimension_and_combined_taylor_planes(
    plane_dim, new_polarization_basis
):
    image = _model_image(CORRELATIONS[2], "taylor_term", np.complex128)
    original = copy.deepcopy(image)
    with pytest.raises(ValueError, match="plane_dim must be 'frequency', or"):
        GridderAntiAlias.image_to_uv(
            image,
            {"image_size": [N_PIXELS, N_PIXELS], "fft_padding": FFT_PADDING},
            make_gridding_kernel(None, FFT_PADDING),
            plane_dim,
            "model",
            "model",
            {"visibility": "VISIBILITY_MODEL"},
            True,
            None,
            1,
            "scipy",
            None,
            np.complex128,
            new_polarization_basis,
        )
    xr.testing.assert_identical(image, original)
