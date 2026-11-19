"""GridderDirectFourierTransform obeys the interface rules, and its methods give main's arrays.

The code paths of main that the methods replace are the allocation and the C++
calls of ``add_dft_image_single_field`` (written out in ``_main_kernel_call``),
the body of ``predict_dft_visibilities_single_field`` (``_main_predict``),
``normalize_dft_image`` and the registration of the model group by
``prepare_model_uv_continuum_single_field``. Every comparison is bit for bit,
with one thread.
"""

import copy

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.direct_fourier_transform import (
    _model_visibilities,
    image_direction_cosines,
    normalize_dft_image,
)
from astroviper.processing_functions.imaging.gridders.direct_fourier_transform.gridder_direct_fourier_transform import (
    GridderDirectFourierTransform,
)
from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
    dft_image,
    dft_image_uv_sampling,
    dft_predict,
)
from astroviper.processing_functions.imaging.gridders.gridder_base import Gridder
from astroviper.processing_functions.imaging.utils.frequency_mapping import (
    map_visibility_frequencies_to_image,
)
from astroviper.processing_functions.imaging.utils.imaging_polarization import (
    is_four_correlation_basis,
)
from astroviper.utils.data_group_tools import create_data_groups_in_and_out

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
IMAGE_FREQUENCIES = np.array([1.00e9, 1.05e9, 1.10e9])
MS_FREQUENCIES = np.array([1.00e9, 1.01e9, 1.05e9, 1.10e9])
REFERENCE_FREQUENCY = 1.04e9
CORRELATIONS = {2: ["XX", "YY"], 4: ["XX", "XY", "YX", "YY"]}
VISIBILITY_DIMS = ("time", "baseline_id", "frequency", "polarization")
IMAGE_DIMS = ("time", "frequency", "polarization", "l", "m")

CASES = pytest.mark.parametrize(
    "n_pol, complex_dtype",
    [
        (n_pol, complex_dtype)
        for n_pol in (2, 4)
        for complex_dtype in (np.complex64, np.complex128)
    ],
)
GRID_METHODS = pytest.mark.parametrize(
    "method, kernel_function",
    [("grid", dft_image), ("grid_uv_sampling", dft_image_uv_sampling)],
)


def _measurement_set(n_pol, seed):
    """Random samples with a zero weight, NaN weights, a NaN visibility and a
    NaN uvw."""
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


def _image(polarizations):
    image = xr.Dataset(
        coords={
            "time": [0.0],
            "frequency": IMAGE_FREQUENCIES,
            "polarization": polarizations,
            "l": (np.arange(N_PIXELS) - N_PIXELS // 2) * -2.0e-5,
            "m": (np.arange(N_PIXELS) - N_PIXELS // 2) * 2.0e-5,
        }
    )
    image.attrs = {"type": "image_dataset", "data_groups": {"residual": {}}}
    return image


def _main_allocation(img_xds, uv_sampling, complex_dtype):
    """The image that main's add_dft_image_single_field allocated."""
    float_dtype = np.float32 if np.dtype(complex_dtype) == np.complex64 else np.float64
    image_dtype = (
        np.dtype(complex_dtype)
        if not uv_sampling and is_four_correlation_basis(img_xds.polarization.values)
        else float_dtype
    )
    return np.zeros(tuple(img_xds.sizes[dim] for dim in IMAGE_DIMS), dtype=image_dtype)


def _main_kernel_call(
    kernel_function, ms_xds, img_xds, weights, frequency_map, grid, normalization
):
    """The C++ call of main's add_dft_image_single_field (dft_image, or with
    uv_sampling dft_image_uv_sampling), with its casts and maps."""
    time_map = np.zeros(ms_xds.sizes["time"], dtype=np.int64)
    pol_map = np.arange(ms_xds.sizes["polarization"], dtype=np.int64)
    visibilities = (
        (np.ascontiguousarray(ms_xds.VISIBILITY.values, dtype=np.complex128),)
        if kernel_function is dft_image
        else ()
    )
    kernel_function(
        grid,
        normalization,
        *visibilities,
        np.ascontiguousarray(ms_xds.UVW.values, dtype=np.float64),
        np.ascontiguousarray(ms_xds.frequency.values, dtype=np.float64),
        np.ascontiguousarray(frequency_map, dtype=np.int64),
        time_map,
        pol_map,
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(img_xds.l.values, dtype=np.float64),
        np.ascontiguousarray(img_xds.m.values, dtype=np.float64),
        processing_function_threads=1,
    )


def _main_predict(ms_xds, img_xds, model, frequency_map, arguments):
    """The body of main's predict_dft_visibilities_single_field."""
    ms_data_group_in, ms_data_group_out = create_data_groups_in_and_out(
        ms_xds,
        data_group_in_name=arguments["ms_data_group_in_name"],
        data_group_out_name=arguments["ms_data_group_out_name"],
        data_group_out_modified=dict(arguments["ms_data_group_out_modified"]),
        overwrite=arguments["overwrite"],
    )
    time_map = np.zeros(ms_xds.sizes["time"], dtype=np.int64)
    pol_map = np.arange(ms_xds.sizes["polarization"], dtype=np.int64)
    model_visibilities = _model_visibilities(
        ms_xds,
        ms_data_group_in,
        ms_data_group_out,
        arguments["ms_data_group_out_name"],
        description=arguments["description"],
    )
    l, m = image_direction_cosines(img_xds)
    dft_predict(
        np.ascontiguousarray(model),
        model_visibilities.values,
        np.ascontiguousarray(ms_xds.UVW.values, dtype=np.float64),
        np.ascontiguousarray(ms_xds.frequency.values, dtype=np.float64),
        np.ascontiguousarray(frequency_map, dtype=np.int64),
        time_map,
        pol_map,
        l,
        m,
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


def test_direct_fourier_transform_is_a_gridder_that_cannot_be_instantiated():
    assert issubclass(GridderDirectFourierTransform, Gridder)
    with pytest.raises(
        TypeError, match="GridderDirectFourierTransform cannot be instantiated"
    ):
        GridderDirectFourierTransform()


@pytest.mark.parametrize("attribute", CONSTANTS + METHODS)
def test_direct_fourier_transform_attributes_cannot_be_reassigned_or_deleted(
    attribute,
):
    value = getattr(GridderDirectFourierTransform, attribute)
    with pytest.raises(
        AttributeError,
        match=f"GridderDirectFourierTransform.{attribute} cannot be reassigned",
    ):
        setattr(GridderDirectFourierTransform, attribute, None)
    with pytest.raises(
        AttributeError,
        match=f"GridderDirectFourierTransform.{attribute} cannot be deleted",
    ):
        delattr(GridderDirectFourierTransform, attribute)
    assert getattr(GridderDirectFourierTransform, attribute) is value


def test_direct_fourier_transform_constants():
    assert GridderDirectFourierTransform.name == "direct_fourier_transform"
    assert GridderDirectFourierTransform.normalization == "sum_of_weights"
    assert GridderDirectFourierTransform.gcf_terms == ()
    assert list(GridderDirectFourierTransform.grid_roles.items()) == [
        ("sky", "SKY_RESIDUAL"),
        ("visibility_normalization", "VISIBILITY_NORMALIZATION"),
    ]
    assert list(GridderDirectFourierTransform.grid_uv_sampling_roles.items()) == [
        ("point_spread_function", "POINT_SPREAD_FUNCTION"),
        ("uv_sampling_normalization", "UV_SAMPLING_NORMALIZATION"),
    ]
    assert GridderDirectFourierTransform.degrid_role == "sky"


@pytest.mark.parametrize("roles", ["grid_roles", "grid_uv_sampling_roles"])
def test_direct_fourier_transform_role_maps_are_read_only(roles):
    role_map = getattr(GridderDirectFourierTransform, roles)
    before = dict(role_map)
    first_role = next(iter(role_map))
    with pytest.raises(TypeError):
        role_map[first_role] = "OTHER"
    with pytest.raises(TypeError):
        del role_map[first_role]
    assert dict(role_map) == before


@pytest.mark.parametrize("method", METHODS)
def test_direct_fourier_transform_implements_every_method_as_a_static_method(method):
    assert isinstance(GridderDirectFourierTransform.__dict__[method], staticmethod)


# --------------------------------------------------------------------------
# The methods give the arrays of main's code paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("uv_sampling", [False, True])
@CASES
def test_new_grid_is_the_allocation_of_main(uv_sampling, n_pol, complex_dtype):
    image = _image(CORRELATIONS[n_pol])
    values, spatial_dims, variable_type = GridderDirectFourierTransform.new_grid(
        image, (1, IMAGE_FREQUENCIES.size, n_pol), uv_sampling, None, complex_dtype
    )
    _assert_same_bits(values, _main_allocation(image, uv_sampling, complex_dtype))
    assert spatial_dims == ("l", "m")
    assert variable_type == ("point_spread_function" if uv_sampling else "sky")


@GRID_METHODS
@CASES
def test_grid_methods_are_the_kernel_calls_of_main(
    method, kernel_function, n_pol, complex_dtype
):
    """Two measurement sets accumulated into the planes of a cube."""
    image = _image(CORRELATIONS[n_pol])
    grid = _main_allocation(image, method == "grid_uv_sampling", complex_dtype)
    normalization = np.zeros(grid.shape[:3])
    expected_grid = grid.copy()
    expected_normalization = normalization.copy()
    for seed in (0, 1):
        ms_xds = _measurement_set(n_pol, seed)
        weights = ms_xds.WEIGHT_IMAGING.values
        frequency_map = map_visibility_frequencies_to_image(
            ms_xds.frequency.values, image.frequency.values
        )
        getattr(GridderDirectFourierTransform, method)(
            ms_xds,
            image,
            None,
            weights,
            frequency_map,
            grid,
            normalization,
            "corrected",
            None,
            1,
        )
        _main_kernel_call(
            kernel_function,
            ms_xds,
            image,
            weights,
            frequency_map,
            expected_grid,
            expected_normalization,
        )
    assert np.count_nonzero(normalization) == normalization.size
    _assert_same_bits(grid, expected_grid)
    _assert_same_bits(normalization, expected_normalization)


@GRID_METHODS
def test_grid_methods_accumulate_into_one_plane_of_an_image(method, kernel_function):
    """As the MFS functions image a Taylor term: Taylor-weighted weights, all
    channels to one plane, a view of the image and a temporary normalization."""
    image = _image(CORRELATIONS[2])
    ms_xds = _measurement_set(2, 0)
    x = (ms_xds.frequency.values - REFERENCE_FREQUENCY) / REFERENCE_FREQUENCY
    weights = np.ascontiguousarray(
        ms_xds.WEIGHT_IMAGING.values * x[None, None, :, None]
    )
    frequency_map = np.zeros(MS_FREQUENCIES.size, dtype=np.int64)
    grid = _main_allocation(image, method == "grid_uv_sampling", np.complex128)
    normalization = np.zeros((1, 1, 2))
    expected_grid = grid.copy()
    expected_normalization = normalization.copy()
    getattr(GridderDirectFourierTransform, method)(
        ms_xds,
        image,
        None,
        weights,
        frequency_map,
        grid[:, 1:2],
        normalization,
        "corrected",
        None,
        1,
    )
    _main_kernel_call(
        kernel_function,
        ms_xds,
        image,
        weights,
        frequency_map,
        expected_grid[:, 1:2],
        expected_normalization,
    )
    assert np.count_nonzero(grid[:, 1]) > 0
    assert not np.any(grid[:, [0, 2]])
    _assert_same_bits(grid, expected_grid)
    _assert_same_bits(normalization, expected_normalization)


@pytest.mark.parametrize("method", ["new_grid", "grid"])
def test_an_image_of_other_polarizations_is_refused_as_on_main(method):
    image = _image(CORRELATIONS[4])
    ms_xds = _measurement_set(2, 0)
    message = (
        "the image polarization axis must match the measurement set: 4 image "
        "polarizations for 2 visibility polarizations."
    )
    with pytest.raises(ValueError, match=message):
        if method == "new_grid":
            GridderDirectFourierTransform.new_grid(
                image, (1, IMAGE_FREQUENCIES.size, 2), False, None, np.complex128
            )
        else:
            GridderDirectFourierTransform.grid(
                ms_xds,
                image,
                None,
                ms_xds.WEIGHT_IMAGING.values,
                np.zeros(MS_FREQUENCIES.size, dtype=np.int64),
                np.zeros((1, 3, 4, N_PIXELS, N_PIXELS), dtype=np.complex128),
                np.zeros((1, 3, 4)),
                "corrected",
                None,
                1,
            )


@CASES
def test_degrid_is_the_prediction_of_main(n_pol, complex_dtype):
    """Two predictions: the second writes into the array the first allocated."""
    image = _image(CORRELATIONS[n_pol])
    rng = np.random.default_rng(5)
    float_dtype = np.float32 if complex_dtype == np.complex64 else np.float64
    model_dtype = complex_dtype if n_pol == 4 else float_dtype
    shape = (1, IMAGE_FREQUENCIES.size, n_pol, N_PIXELS, N_PIXELS)
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
        "description": "Model visibilities.",
    }
    for _ in range(2):
        model = np.zeros(shape, dtype=model_dtype)
        model[..., 6:18, 6:18] = rng.standard_normal((*shape[:3], 12, 12))
        if n_pol == 4:
            model[..., 6:18, 6:18] += 1j * rng.standard_normal((*shape[:3], 12, 12))
        returned = GridderDirectFourierTransform.degrid(
            ms_xds,
            image,
            None,
            model,
            frequency_map,
            fft_padding=None,
            processing_function_threads=1,
            **arguments,
        )
        assert returned is None
        _main_predict(expected, image, model, frequency_map, arguments)
    assert np.count_nonzero(ms_xds.VISIBILITY_PREDICTED.values) > 0
    _assert_same_dataset(ms_xds, expected)


@CASES
def test_uv_to_image_is_normalize_dft_image(n_pol, complex_dtype):
    """Sky and point spread function, divided in place by their normalizations."""
    rng = np.random.default_rng(3)
    image = _image(CORRELATIONS[n_pol])
    sky = _main_allocation(image, False, complex_dtype)
    sky[...] = rng.standard_normal(sky.shape)
    psf = _main_allocation(image, True, complex_dtype)
    psf[...] = rng.standard_normal(psf.shape)
    image["SKY_DIRTY"] = (IMAGE_DIMS, sky)
    image["PSF_DIRTY"] = (IMAGE_DIMS, psf)
    normalization = rng.random(sky.shape[:3]) + 0.5
    normalization[0, 1, 0] = 0.0
    image["SKY_DIRTY_NORMALIZATION"] = (IMAGE_DIMS[:3], normalization)
    image["PSF_DIRTY_NORMALIZATION"] = (IMAGE_DIMS[:3], normalization[:, ::-1])
    image.attrs["data_groups"]["dirty"] = {
        "sky": "SKY_DIRTY",
        "visibility_normalization": "SKY_DIRTY_NORMALIZATION",
        "point_spread_function": "PSF_DIRTY",
        "uv_sampling_normalization": "PSF_DIRTY_NORMALIZATION",
    }
    expected = copy.deepcopy(image)
    unnormalized = expected["SKY_DIRTY"].values.copy()
    result = GridderDirectFourierTransform.uv_to_image(
        image,
        image_params={},
        gcf_xds=None,
        image_data_group_in_name="residual",
        image_data_group_out_name="dirty",
        image_data_group_out_modified={
            "sky": "SKY_DIRTY",
            "point_spread_function": "PSF_DIRTY",
        },
        overwrite=True,
        image_data_variables_keep=[],
        processing_function_threads=1,
        fft_backend="scipy",
        complex_dtype=complex_dtype,
    )
    normalize_dft_image(expected, "dirty", uv_sampling=False)
    normalize_dft_image(expected, "dirty", uv_sampling=True)
    assert result is image
    assert not np.array_equal(result["SKY_DIRTY"].values, unnormalized)
    _assert_same_dataset(result, expected)


def _model_image():
    image = _image(["I", "Q"])
    image["SKY_MODEL"] = (
        IMAGE_DIMS,
        np.random.default_rng(7).standard_normal((1, 3, 2, N_PIXELS, N_PIXELS)),
    )
    image.attrs["data_groups"]["model_in"] = {"sky": "SKY_MODEL"}
    return image


def _image_to_uv(image, image_data_group_out_name):
    return GridderDirectFourierTransform.image_to_uv(
        image,
        image_params={},
        gcf_xds=None,
        plane_dim="taylor_term",
        image_data_group_in_name="model_in",
        image_data_group_out_name=image_data_group_out_name,
        image_data_group_out_modified={"visibility": "VISIBILITY_MODEL"},
        overwrite=True,
        image_data_variables_keep=[],
        processing_function_threads=1,
        fft_backend="scipy",
        data_variables_to_process=None,
        complex_dtype=np.complex128,
        new_polarization_basis=None,
    )


def test_image_to_uv_leaves_the_model_images_as_they_are():
    image = _model_image()
    expected = copy.deepcopy(image)
    result = _image_to_uv(image, "model_in")
    assert result is image
    xr.testing.assert_identical(result, expected)


def test_image_to_uv_registers_another_output_group_as_main():
    """The group main's prepare_model_uv_continuum_single_field registered for
    the direct Fourier transform: the input group's sky image, nothing else."""
    image = _model_image()
    model = image["SKY_MODEL"].values
    result = _image_to_uv(image, "model_out")
    assert result is image
    assert result["SKY_MODEL"].values is model
    group = dict(result.attrs["data_groups"]["model_out"])
    assert group.pop("description") == "Model images for the direct Fourier transform."
    group.pop("date")
    assert group == {"sky": "SKY_MODEL"}
    assert result.attrs["data_groups"]["model_in"] == {"sky": "SKY_MODEL"}
