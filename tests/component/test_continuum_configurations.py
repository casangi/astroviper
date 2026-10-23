"""Small continuum software regressions, using downloaded reference data and no CASA dependency.

Every MFS/MVC x local/global x CASA/native weighting configuration runs through
multiple imaging cycles. Changing reduction topology and all cache modes must
preserve its products. Different algorithm configurations need not agree.
"""

from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging import image_continuum_single_field
from astroviper.processing_functions.imaging.restore import (
    _elliptical_gaussian_kernel,
)


@pytest.fixture
def continuum_input(tmp_path, tw_hydra_source_store):
    with TemporaryDirectory(prefix="continuum_input_", dir=tmp_path) as directory:
        input_directory = Path(directory)
        store = input_directory / "twhya_selfcal_lsrk_5chans.ps.zarr"
        copytree(tw_hydra_source_store, store)
        processing_set = open_processing_set(str(store))
        fields = processing_set.xr_ps.get_combined_field_and_source_xds()
        frequency = processing_set.xr_ps.get_freq_axis().values
        params = dict(
            image_size=[64, 64],
            cell_size=np.array([-0.2, 0.2]) * np.pi / (180 * 3600),
            phase_direction=fields.FIELD_PHASE_CENTER_DIRECTION.sel(
                field_name=fields.attrs["center_field_name"]
            ).values,
            frequency_coords=frequency,
            polarization_coords=["I", "Q"],
            time_coords=[0],
            fft_padding=1.2,
            cpp_gridder=True,
            n_taylor_terms=2,
            reference_frequency=float(np.mean(frequency)),
        )
        mask = np.zeros((64, 64), dtype=bool)
        mask[18:43, 21:47] = True
        mask[18:23, 21:29] = False
        mask_path = input_directory / "mask.npy"
        np.save(mask_path, mask)
        yield store, params, mask_path, mask


def _run(
    store,
    params,
    mask_path,
    output,
    continuum_mode,
    scope,
    casa,
    cache_mode,
    reduce_mode,
    read_output=None,
):
    with dask.config.set(scheduler="synchronous"):
        result = image_continuum_single_field(
            ps_store=str(store),
            image_store=str(output),
            image_params=params,
            imaging_weights_params=dict(
                weighting="briggs",
                robust=0.5,
                weighting_scope=scope,
                casa_weighting_implementation=casa,
            ),
            cycle_control_params=dict(
                max_iter=8,
                max_cycles=2,
                max_iter_per_cycle=4,
                threshold=0.0,
                gain=0.1,
                psf_sidelobe_factor=1.5,
                min_psf_fraction=0.05,
                max_psf_fraction=0.8,
            ),
            gridder="anti_alias",
            deconvolver="mtmfs",
            continuum_mode=continuum_mode,
            restore=True,
            primary_beam_correction=True,
            primary_beam_limit=0.2,
            clean_mask=str(mask_path),
            scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
            processing_set_data_group_name="base",
            image_data_variables_keep=[
                "sky_model",
                "sky_residual",
                "point_spread_function",
                "primary_beam",
                "beam_fit_params_point_spread_function",
            ],
            single_precision_image=False,
            processing_function_threads=1,
            n_mapping_parallelism={"frequency": 3},
            memory_mode="in_memory",
            weight_memory_mode="in_memory" if cache_mode == "recompute" else cache_mode,
            visibility_memory_mode=cache_mode,
            wideband_primary_beam_memory_mode=cache_mode,
            overwrite=True,
            vizualize_graph=False,
            compute_backend="dask",
            reduce_mode=reduce_mode,
        )
    with xr.open_zarr(output if read_output is None else read_output) as stored:
        image = stored.load()
    return result, image


def _circular_clean_beam_convolution(plane, beam_parameters, cell_size):
    """Independently apply the fitted unit-peak beam on the periodic image grid."""
    major, minor, position_angle = np.asarray(beam_parameters, dtype=float)
    kernel = _elliptical_gaussian_kernel(
        *plane.shape,
        major / cell_size,
        minor / cell_size,
        position_angle,
        plane.dtype,
    )
    return np.fft.irfft2(
        np.fft.rfft2(plane) * np.fft.rfft2(np.fft.ifftshift(kernel)),
        s=plane.shape,
    )


def _assert_restoration_values(image, primary_beam_limit=0.2):
    """Check ``M * B + H**(-1) R`` and model-first PB correction."""
    model = image.SKY_MODEL.isel(taylor_term=0, drop=True)
    primary_beam = image.PRIMARY_BEAM.isel(frequency=0, drop=True)
    fitted_beam = image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION
    for dimension in ("frequency", "taylor_term", "psf_taylor_order"):
        if dimension in fitted_beam.dims:
            fitted_beam = fitted_beam.isel({dimension: 0}, drop=True)
    cell_size = abs(float(image.l.values[1] - image.l.values[0]))

    for time in range(image.sizes["time"]):
        for polarization in range(image.sizes["polarization"]):
            index = {"time": time, "polarization": polarization}
            model_plane = model.isel(index).values
            # MT-MFS restores the principal-solution residual. Build the
            # Hankel matrix directly from the PSF peak for an independent
            # NumPy oracle rather than reusing the native restoration helper.
            residual_terms = (
                image.SKY_RESIDUAL.isel(index).transpose("taylor_term", "l", "m").values
            )
            psf_terms = (
                image.POINT_SPREAD_FUNCTION.isel(index)
                .transpose("psf_taylor_order", "l", "m")
                .values
            )
            peak = np.unravel_index(np.argmax(np.abs(psf_terms[0])), psf_terms[0].shape)
            nterms = residual_terms.shape[0]
            orders = np.add.outer(np.arange(nterms), np.arange(nterms))
            hessian = psf_terms[:, peak[0], peak[1]][orders]
            residual_plane = np.linalg.solve(
                hessian, residual_terms.reshape(nterms, -1)
            )[0].reshape(model_plane.shape)
            beam_plane = primary_beam.isel(index).values
            beam_parameters = fitted_beam.isel(index).values

            convolved_model = _circular_clean_beam_convolution(
                model_plane, beam_parameters, cell_size
            )
            expected_restored = convolved_model + residual_plane
            np.testing.assert_allclose(
                image.SKY_RESTORED.isel(index),
                expected_restored,
                rtol=1e-6,
                atol=max(float(np.max(np.abs(expected_restored))) * 1e-8, 1e-14),
            )

            inside = beam_plane >= primary_beam_limit
            divided_model = np.zeros_like(model_plane)
            np.divide(model_plane, beam_plane, out=divided_model, where=inside)
            expected_corrected = np.full_like(residual_plane, np.nan)
            expected_corrected[inside] = residual_plane[inside] / beam_plane[inside]
            expected_corrected[inside] += _circular_clean_beam_convolution(
                divided_model, beam_parameters, cell_size
            )[inside]
            np.testing.assert_allclose(
                image.SKY_RESTORED_PRIMARY_BEAM_CORRECTED.isel(index),
                expected_corrected,
                rtol=1e-6,
                atol=max(float(np.nanmax(np.abs(expected_corrected))) * 1e-8, 1e-14),
                equal_nan=True,
            )


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
@pytest.mark.parametrize("scope", ["local", "global"])
@pytest.mark.parametrize(
    "casa", [False, True], ids=["native-weighting", "casa-weighting"]
)
def test_continuum_configuration_cache_and_reduction_regression(
    tmp_path, continuum_input, continuum_mode, scope, casa
):
    store, params, mask_path, mask = continuum_input
    reference_result, reference = _run(
        store,
        params,
        mask_path,
        tmp_path / "reference.img.zarr",
        continuum_mode,
        scope,
        casa,
        "recompute",
        "tree",
    )
    result, actual = _run(
        store,
        params,
        mask_path,
        tmp_path / "cached.img.zarr",
        continuum_mode,
        scope,
        casa,
        "in_place",
        "single_node",
    )
    products = [
        "SKY_MODEL",
        "SKY_RESIDUAL",
        "POINT_SPREAD_FUNCTION",
        "PRIMARY_BEAM",
        "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        "SKY_RESTORED",
        "SKY_RESTORED_PRIMARY_BEAM_CORRECTED",
    ]
    assert result["n_cycles"] == reference_result["n_cycles"] == 2
    # max_iter is per image plane; the reported total sums polarizations.
    assert 0 < result["controller"].total_iter_done <= 8 * actual.sizes["polarization"]
    assert (
        result["controller"].total_iter_done
        == reference_result["controller"].total_iter_done
    )
    assert actual.sizes["taylor_term"] == 2
    assert actual.sizes["psf_taylor_order"] == 3
    assert actual.sizes["frequency"] == 1
    for name in products:
        assert name in actual and name in reference
        assert actual[name].dims == reference[name].dims
        # A peak-scaled floor handles near-zero pixels after different sum order.
        scale = float(np.nanmax(np.abs(reference[name].values)))
        np.testing.assert_allclose(
            actual[name],
            reference[name],
            rtol=1e-6,
            atol=max(scale * 1e-8, 1e-14),
            equal_nan=True,
            err_msg=name,
        )
        xr.testing.assert_allclose(actual[name], result["image"][name])
    assert np.isfinite(actual.SKY_RESIDUAL).all()
    assert np.isfinite(actual.SKY_MODEL).all()
    assert np.max(np.abs(actual.SKY_MODEL.values)) > 0
    outside_mask = ~xr.DataArray(mask, dims=("l", "m"))
    assert (actual.SKY_MODEL.where(outside_mask, 0) == 0).all()
    # Channel caches must not leak into the final public image store.
    assert not any("CACHE" in name or "MVC_" in name for name in actual.data_vars)
    beam = actual.PRIMARY_BEAM.isel(frequency=0, drop=True)
    assert actual.SKY_RESTORED.dims == ("time", "polarization", "l", "m")
    corrected = actual.SKY_RESTORED_PRIMARY_BEAM_CORRECTED
    assert corrected.dims == ("time", "polarization", "l", "m")
    np.testing.assert_array_equal(
        np.isfinite(corrected),
        np.isfinite(beam) & (beam >= 0.2),
    )
    _assert_restoration_values(actual)


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_continuum_uses_writer_resolved_path(
    tmp_path, continuum_input, monkeypatch, continuum_mode
):
    """Follow XRADIO's returned path for allocation, caches, and final output."""
    import xradio.image

    store, params, mask_path, _ = continuum_input
    requested = tmp_path / "requested.zarr"
    resolved = tmp_path / "requested.img.zarr"
    real_write = xradio.image.write_image
    calls = []

    def write_with_resolved_path(image, imagename, **kwargs):
        calls.append(str(imagename))
        destination = str(resolved) if str(imagename) == str(requested) else imagename
        real_write(image, imagename=destination, **kwargs)
        return [str(destination)]

    # Emulate the modern writer contract even with older supported XRADIO.
    monkeypatch.setattr(xradio.image, "write_image", write_with_resolved_path)
    result, image = _run(
        store,
        params,
        mask_path,
        requested,
        continuum_mode,
        "global",
        True,
        "in_place",
        "tree",
        read_output=resolved,
    )
    assert calls == [str(requested), str(resolved)]
    assert not requested.exists()
    assert resolved.is_dir()
    assert result["n_cycles"] == 2
    assert np.isfinite(image.SKY_RESIDUAL).all()
    assert "SKY_RESTORED_PRIMARY_BEAM_CORRECTED" in image
    assert not any("CACHE" in name or "MVC_" in name for name in image.data_vars)
