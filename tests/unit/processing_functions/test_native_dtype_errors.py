"""Native diagnostics identify Python arguments and distinguish swapped dtypes."""

import copy
import sys

import numpy as np
import pytest


def _case(kind):
    from astroviper.processing_functions.imaging.model_update.deconvolvers import (
        aspclean,
        hogbom,
    )
    from astroviper.processing_functions.imaging.model_update.deconvolvers.mtmfs import (
        _mtmfs_ext,
    )
    from tests.unit.processing_functions.test_serialized_buffers import (
        _case as serialized_case,
    )

    point = np.zeros((32, 32))
    point[16, 16] = 1.0
    if kind == "hogbom_maximg":
        return hogbom.maximg, dict(image=point, mask=np.ones_like(point, dtype=bool))
    if kind == "hogbom":
        return hogbom.clean, dict(
            dirty_image=point,
            psf=point.copy(),
            model=point * 0,
            mask=np.ones_like(point, dtype=bool),
        )
    if kind in ("hogbom_cube", "hogbom_many", "asp_cube"):
        cube = point.reshape((1, 1, 1, 32, 32))
        kwargs = dict(
            residual_cube=cube,
            psf_cube=cube.copy(),
            model_cube=cube * 0,
            mask_cube=np.ones_like(cube, dtype=bool),
            threshold_per_cycle=np.zeros((1, 1, 1)),
            max_iter_per_cycle_updated=np.ones((1, 1, 1), dtype=np.int32),
        )
        if kind == "asp_cube":
            for name in ("residual", "psf", "model", "mask"):
                kwargs[name] = kwargs.pop(name + "_cube")
            kwargs["mask"] = kwargs["mask"].astype(float)
            return aspclean.clean_cube, kwargs
        return (
            hogbom.clean_cube_many_threads
            if kind == "hogbom_many"
            else hogbom.clean_cube
        ), kwargs
    if kind == "asp":
        return aspclean.clean, dict(
            residual=point, psf=point.copy(), model=point * 0, mask=np.ones_like(point)
        )
    if kind == "mtmfs":
        return _mtmfs_ext.clean, dict(
            residual=point[None],
            psf=point[None].copy(),
            model=point[None] * 0,
            mask=np.ones_like(point[None]),
        )
    if kind == "hessian":
        return _mtmfs_ext.hessian, dict(psf=point[None])
    if kind == "principal":
        return _mtmfs_ext.principal_solution, dict(
            residual=point[None], inverse_hessian=np.ones((1, 1))
        )
    if kind == "simulation":
        from astroviper.processing_functions.simulation.visibility_kernel_cpp import (
            calculate_visibilities,
        )

        return calculate_visibilities, serialized_case("simulation")
    if kind == "psf_fit":
        from astroviper.processing_functions.image_analysis.psf_gaussian_fit_cpp._psf_gaussian_fit_ext import (
            fit_psf_beam,
        )

        return fit_psf_beam, dict(
            psf=point[None],
            delta_x=1.0e-4,
            delta_y=1.0e-4,
            psfcutoff=0.35,
            beams=np.zeros((1, 3)),
        )
    if kind.startswith("weight_"):
        from tests.unit.processing_functions.imaging.imaging_setup.imaging_weighting.test_grid_imaging_weights_cpp import (
            _binding_validation_case,
        )

        return _binding_validation_case(kind == "weight_degrid", np.float64)
    if kind.startswith("aa_"):
        from astroviper.processing_functions.imaging.gridders.anti_alias.anti_alias_grid_cpp import (
            anti_alias_degrid,
            anti_alias_grid,
            anti_alias_grid_uv_sampling,
        )

        kwargs = serialized_case("degrid64" if kind == "aa_degrid" else "grid64")
        if kind == "aa_sampling":
            kwargs.pop("vis_data")
        return {
            "aa_grid": anti_alias_grid,
            "aa_sampling": anti_alias_grid_uv_sampling,
            "aa_degrid": anti_alias_degrid,
        }[kind], kwargs
    if kind.startswith("dft_"):
        from astroviper.processing_functions.imaging.gridders.direct_fourier_transform.direct_fourier_transform_cpp import (
            dft_image,
            dft_image_uv_sampling,
            dft_predict,
        )

        kwargs = serialized_case("dft64")
        if kind != "dft_predict":
            kwargs["image"] = kwargs.pop("model")
            kwargs["normalization"] = np.zeros((1, 1, 1))
            kwargs["weight"] = np.ones((1, 1, 1, 1))
            if kind == "dft_sampling":
                kwargs.pop("vis_data")
        return {
            "dft_image": dft_image,
            "dft_sampling": dft_image_uv_sampling,
            "dft_predict": dft_predict,
        }[kind], kwargs
    raise AssertionError(kind)


_ARGUMENTS = {
    "hogbom_maximg": ("image", "mask"),
    "hogbom": ("dirty_image", "psf", "model", "mask"),
    "hogbom_cube": (
        "residual_cube",
        "psf_cube",
        "model_cube",
        "mask_cube",
        "threshold_per_cycle",
        "max_iter_per_cycle_updated",
    ),
    "hogbom_many": (
        "residual_cube",
        "psf_cube",
        "model_cube",
        "mask_cube",
        "threshold_per_cycle",
        "max_iter_per_cycle_updated",
    ),
    "asp": ("residual", "psf", "model", "mask"),
    "asp_cube": (
        "residual",
        "psf",
        "model",
        "mask",
        "threshold_per_cycle",
        "max_iter_per_cycle_updated",
    ),
    "mtmfs": ("residual", "psf", "model", "mask"),
    "hessian": ("psf",),
    "principal": ("residual", "inverse_hessian"),
    "simulation": (
        "visibility",
        "uvw",
        "antenna1",
        "antenna2",
        "frequency",
        "polarization_index",
        "point_source_flux",
        "k_vector",
        "inverse_n",
        "point_source_ra_dec",
        "pointing_ra_dec",
        "beam_model_map",
        "parallactic_angle",
        "mueller_selection",
    ),
    "psf_fit": ("beams",),
    "weight_grid": ("grid", "sum_weight"),
    "weight_degrid": ("imaging_weight", "grid_imaging_weight"),
    "aa_grid": ("grid", "normalization"),
    "aa_sampling": ("grid", "normalization"),
    "aa_degrid": ("grid", "vis_data"),
    "dft_image": ("image", "normalization"),
    "dft_sampling": ("image", "normalization"),
    "dft_predict": ("model", "vis_data"),
}


@pytest.mark.parametrize(
    "kind,name", [(kind, name) for kind, names in _ARGUMENTS.items() for name in names]
)
@pytest.mark.parametrize("failure", ["dtype", "byte_order"])
def test_native_dtype_errors_use_python_names_and_explicit_byte_order(
    kind, name, failure, unaligned_array
):
    kernel, original = _case(kind)
    value = original[name]
    if failure == "byte_order" and value.dtype.byteorder == "|":
        pytest.skip("Single-byte arrays have no byte order")
    bad_dtype = (
        np.dtype(np.int16) if failure == "dtype" else value.dtype.newbyteorder("S")
    )
    bad = (
        np.zeros(value.shape, dtype=bad_dtype)
        if failure == "dtype"
        else value.astype(bad_dtype)
    )
    messages = []
    for unaligned in (False, True):
        kwargs = copy.deepcopy(original)
        kwargs[name] = unaligned_array(bad, 1) if unaligned else bad.copy()
        before = {
            key: value.copy()
            for key, value in kwargs.items()
            if isinstance(value, np.ndarray)
        }
        with pytest.raises(ValueError if kind == "psf_fit" else RuntimeError) as error:
            kernel(**kwargs)
        message = str(error.value)
        messages.append(message)
        assert message.startswith(name + " ")
        assert (
            "wrong dtype" if failure == "dtype" else "non-native byte order"
        ) in message
        assert "expected native " in message
        assert value.dtype.name in message
        assert f"got {bad.dtype.name} ({bad.dtype.str}, " in message
        expected_order = (
            sys.byteorder
            if failure == "dtype"
            else ("big" if sys.byteorder == "little" else "little")
        )
        assert f"{expected_order}-endian)" in message
        for key, snapshot in before.items():
            np.testing.assert_array_equal(kwargs[key], snapshot, err_msg=key)
        if unaligned:
            assert kwargs[name].base[0] == 0xA5
            assert kwargs[name].base[-1] == 0xA5
    assert messages[0] == messages[1]


@pytest.mark.parametrize(
    "beam_kind,key",
    [
        ("polynomial", "frequency"),
        ("polynomial", "coefficients"),
        ("jones_image", "jones"),
        ("jones_image", "parallactic_angle"),
        ("jones_image", "frequency"),
        ("jones_image", "polarization_index"),
    ],
)
@pytest.mark.parametrize("failure", ["dtype", "byte_order"])
def test_beam_dtype_errors_identify_the_python_dictionary_entry(
    beam_kind, key, failure, unaligned_array
):
    kernel, kwargs = _case("simulation")
    model = dict(kind=beam_kind, max_rad_1GHz=1.0, frequency=np.array([1.0e9]))
    if beam_kind == "polynomial":
        model.update(coefficients=np.array([[1.0, 0.0]]), n_sample=0)
    else:
        model.update(
            jones=np.ones((1, 1, 4, 4, 4), dtype=np.complex128),
            parallactic_angle=np.zeros(1),
            polarization_index=np.arange(4, dtype=np.int64),
            cell_size_l=1.0e-3,
            cell_size_m=1.0e-3,
        )
    value = model[key]
    dtype = np.dtype(np.int16) if failure == "dtype" else value.dtype.newbyteorder("S")
    bad = (
        np.zeros(value.shape, dtype=dtype)
        if failure == "dtype"
        else value.astype(dtype)
    )
    messages = []
    visibility = kwargs["visibility"].copy()
    for unaligned in (False, True):
        trial_model = copy.deepcopy(model)
        trial_model[key] = unaligned_array(bad, 1) if unaligned else bad.copy()
        kwargs["beam_models"] = [trial_model]
        with pytest.raises(RuntimeError) as error:
            kernel(**kwargs)
        message = str(error.value)
        messages.append(message)
        assert message.startswith(f"beam_models[0]['{key}'] ")
        assert (
            "wrong dtype" if failure == "dtype" else "non-native byte order"
        ) in message
        assert value.dtype.name in message
        assert f"got {bad.dtype.name} ({bad.dtype.str}, " in message
        np.testing.assert_array_equal(kwargs["visibility"], visibility)
        np.testing.assert_array_equal(trial_model[key], bad)
        if unaligned:
            assert trial_model[key].base[0] == 0xA5
            assert trial_model[key].base[-1] == 0xA5
    assert messages[0] == messages[1]
