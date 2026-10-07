"""Small continuum software regressions, using downloaded reference data and no CASA dependency.

Every MFS/MVC x local/global x CASA/native weighting configuration runs through
multiple imaging cycles. Changing reduction topology and all cache modes must
preserve its products. Different algorithm configurations need not agree.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging import image_continuum_single_field


@pytest.fixture
def continuum_input(tmp_path, tw_hydra_archive):
    with TemporaryDirectory(prefix="continuum_input_", dir=tmp_path) as directory:
        input_directory = Path(directory)
        with ZipFile(tw_hydra_archive) as archive:
            archive.extractall(input_directory)
        store = input_directory / "twhya_selfcal_lsrk_5chans.ps.zarr"
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
            nterms=2,
            reference_frequency=float(np.mean(frequency)),
            reference_frequency_hz=float(np.mean(frequency)),
        )
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:44, 20:44] = True
        mask_path = input_directory / "mask.npy"
        np.save(mask_path, mask)
        yield store, params, mask_path, mask


def _run(
    store,
    params,
    mask_path,
    output,
    specmode,
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
            iteration_control_params=dict(
                max_iter=8,
                max_cycles=2,
                max_iter_per_cycle=4,
                threshold=0.0,
                gain=0.1,
                psf_sidelobe_factor=1.5,
                min_psf_fraction=0.05,
                max_psf_fraction=0.8,
            ),
            gridder="prolate_spheroidal",
            deconvolver="hogbom",
            specmode=specmode,
            restore=True,
            pbcor=True,
            pblimit=0.2,
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
            n_chunks=3,
            memory_mode="in_memory",
            weight_memory_mode="in_memory" if cache_mode == "recompute" else cache_mode,
            visibility_memory_mode=cache_mode,
            widebandpb_memory_mode=cache_mode,
            overwrite=True,
            vizualize_graph=False,
            compute_backend="dask",
            reduce_mode=reduce_mode,
        )
    with xr.open_zarr(output if read_output is None else read_output) as stored:
        image = stored.load()
    return result, image


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
@pytest.mark.parametrize("scope", ["local", "global"])
@pytest.mark.parametrize(
    "casa", [False, True], ids=["native-weighting", "casa-weighting"]
)
def test_continuum_configuration_cache_and_reduction_regression(
    tmp_path, continuum_input, specmode, scope, casa
):
    store, params, mask_path, mask = continuum_input
    reference_result, reference = _run(
        store,
        params,
        mask_path,
        tmp_path / "reference.img.zarr",
        specmode,
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
        specmode,
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
        "SKY_RESTORED_PBCOR",
    ]
    assert result["n_major_cycles"] == reference_result["n_major_cycles"] == 2
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
    restored = actual.SKY_RESTORED.isel(taylor_term=0, drop=True)
    corrected = actual.SKY_RESTORED_PBCOR
    assert corrected.dims == ("time", "polarization", "l", "m")
    valid = beam > 0.2
    xr.testing.assert_allclose(corrected.where(valid), (restored / beam).where(valid))


@pytest.mark.parametrize("specmode", ["mfs", "mvc"])
def test_continuum_uses_writer_resolved_path(
    tmp_path, continuum_input, monkeypatch, specmode
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
        specmode,
        "global",
        True,
        "in_place",
        "tree",
        read_output=resolved,
    )
    assert calls == [str(requested), str(resolved)]
    assert not requested.exists()
    assert resolved.is_dir()
    assert result["n_major_cycles"] == 2
    assert np.isfinite(image.SKY_RESIDUAL).all()
    assert "SKY_RESTORED_PBCOR" in image
    assert not any("CACHE" in name or "MVC_" in name for name in image.data_vars)
