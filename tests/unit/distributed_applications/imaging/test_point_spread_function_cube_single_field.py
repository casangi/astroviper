"""Tests of the point_spread_function_cube_single_field driver: the reduce,
the channels in the image's order whatever order the reduce leaves them in,
and the refusal of a Measurement Set v2."""

import importlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord

import astroviper.distributed_applications as distributed_applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
N_ANTENNA = 6
# The module itself (the package re-exports the function under its name).
psf_module = importlib.import_module(
    "astroviper.distributed_applications.imaging.point_spread_function_cube_single_field"
)


def _image_params(frequency_coords):
    return {
        "image_size": [32, 32],
        "cell_size": np.array([-8.0, 8.0]) * np.pi / (180 * 3600),
        "phase_direction": PC,
        "frequency_coords": np.asarray(frequency_coords),
        "polarization_coords": ["I", "V"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }


def _chunk_result(task_id, frequency):
    return {
        "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
        "point_spread_function_properties": xr.Dataset(
            {"SENSITIVITY": ("frequency", np.asarray(frequency) / 1e9)},
            coords={"frequency": frequency},
        ),
    }


def test_combine_point_spread_function_results_composes():
    """Concatenates in the order given and composes under tree reduction."""
    combine = psf_module.combine_point_spread_function_results
    pair = combine([_chunk_result(2, [5e9, 6e9]), _chunk_result(0, [1e9, 2e9])], {})
    result = combine([pair, _chunk_result(1, [3e9])], {})
    assert list(result["timing_node_tasks"]["task_id"]) == [2, 0, 1]
    properties = result["point_spread_function_properties"]
    np.testing.assert_array_equal(properties.frequency, [5e9, 6e9, 1e9, 2e9, 3e9])
    np.testing.assert_array_equal(properties.SENSITIVITY, [5, 6, 1, 2, 3])


@pytest.fixture(scope="module")
def ps_store(tmp_path_factory):
    """No source, 6 VLA antennas, 3 channels."""
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(tmp_path_factory.mktemp("psf_driver") / "empty.ps.zarr"),
        antenna_xds=read_telescope_layout(
            "vla.d", antenna_selection=list(range(N_ANTENNA))
        ),
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1800.0,
            "n_samples": 2,
        },
        frequency_params={
            "freq_start": 3e9,
            "freq_delta": 0.2e9,
            "n_channels": 3,
            "channel_width": 1e7,
        },
        polarization=["RR", "LL"],
        point_source_flux=np.zeros((1, 1, 1, 4)),
        point_source_ra_dec=PC[None, None, :],
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("vla_nvss")],
        beam_model_map=np.zeros(N_ANTENNA, int),
        n_time_chunks=1,
        n_frequency_chunks=3,
        overwrite=True,
    )
    return result["ps_store"]


def test_channels_in_the_image_order(ps_store, monkeypatch):
    """A reduce that reverses its inputs: the channels still come back in the
    order of the image's frequency coordinate."""
    from xradio.measurement_set import open_processing_set

    combine = psf_module.combine_point_spread_function_results
    monkeypatch.setattr(
        psf_module,
        "combine_point_spread_function_results",
        lambda input_data, input_params: combine(input_data[::-1], input_params),
    )
    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    properties = psf_module.point_spread_function_cube_single_field(
        ps_store,
        _image_params(frequency_coords),
        {"weighting": "natural"},
        instrument_polarization_basis="circular",
        processing_set_data_group_name="base",
        n_mapping_parallelism={"frequency": 3},
    )
    np.testing.assert_array_equal(properties.frequency.values, frequency_coords)
    # The uv extent in wavelengths grows with the channel frequency.
    assert np.all(np.diff(properties.UV_DISTANCE_MAX.values) > 0)


def test_measurement_set_v2_is_refused(tmp_path):
    ms_v2 = tmp_path / "data.ms"
    ms_v2.mkdir()
    (ms_v2 / "table.dat").write_bytes(b"")
    with pytest.raises(ValueError, match="is a Measurement Set v2"):
        psf_module.point_spread_function_cube_single_field(
            str(ms_v2),
            _image_params([3e9]),
            {"weighting": "natural"},
            instrument_polarization_basis="circular",
        )


def test_the_direct_fourier_transform_point_spread_function(ps_store):
    """The DFT gridder reaches the node tasks and the record holds no kernel
    parameters for it. The beam fit stays within 1 percent of the gridder's: the DFT
    includes the w term, which the gridder leaves out, and the fits of this
    PSF, wider than the image, differ by 1.1e-3."""
    from xradio.measurement_set import open_processing_set

    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    common = dict(
        instrument_polarization_basis="circular",
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 3},
    )
    dft = psf_module.point_spread_function_cube_single_field(
        ps_store,
        _image_params(frequency_coords),
        {"weighting": "natural"},
        gridder="direct_fourier_transform",
        **common,
    )
    gridder = psf_module.point_spread_function_cube_single_field(
        ps_store, _image_params(frequency_coords), {"weighting": "natural"}, **common
    )
    assert dft.attrs["gridding_kernel_params"] is None
    np.testing.assert_allclose(
        dft.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
        gridder.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
        rtol=1e-2,
    )
    xr.testing.assert_identical(dft.SENSITIVITY, gridder.SENSITIVITY)


def test_the_gridder_reaches_the_node_tasks(ps_store, monkeypatch):
    """The driver hands the gridder to its node tasks, which declare it:
    GraphVIPER gives a node task only the keys of input_params it declares, and
    one without the gridder would grid with the anti-alias gridder."""
    import inspect

    import graphviper.graph_tools
    from xradio.measurement_set import open_processing_set

    class MapReached(Exception):
        pass

    def first_map(**kwargs):
        raise MapReached(kwargs["node_task"], kwargs["input_params"])

    monkeypatch.setattr(graphviper.graph_tools, "map", first_map)
    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    with pytest.raises(MapReached) as reached:
        psf_module.point_spread_function_cube_single_field(
            ps_store,
            _image_params(frequency_coords),
            {"weighting": "natural"},
            instrument_polarization_basis="circular",
            processing_set_data_group_name="base",
            gridder="direct_fourier_transform",
        )
    node_task, input_params = reached.value.args
    assert input_params["gridder"] == "direct_fourier_transform"
    assert "gridder" in inspect.signature(node_task).parameters


def test_psf_driver_checks_nested_configuration_before_io(tmp_path):
    from astroviper.distributed_applications.imaging.point_spread_function_cube_single_field import (
        point_spread_function_cube_single_field,
    )

    image_params = {
        "image_size": [0, 16],
        "cell_size": [-1e-5, 1e-5],
        "phase_direction": [0, 0],
        "frequency_coords": [1e9],
        "time_coords": [0],
        "polarization_coords": ["I", "Q"],
        "fft_padding": 1.2,
    }
    output = tmp_path / "never-created.zarr"
    with pytest.raises(ValueError, match="image_size"):
        point_spread_function_cube_single_field(
            "missing.ps.zarr",
            image_params,
            {"weighting": "natural"},
            image_store=str(output),
        )
    assert not output.exists()


def test_both_io_engines_and_fits_write_the_same_cube(tmp_path):
    """The xradio and chunk engines write the same point spread function cube,
    and the chunk engine's FITS files hold the same values."""
    from xradio.image import open_image
    from xradio.measurement_set import open_processing_set

    from tests.unit.distributed_applications.simulation.test_simulate_processing_set import (
        run,
    )

    simulated, _ = run(tmp_path, polarization=["XX", "YY"], noise_params=None)
    ps_store = simulated["ps_store"]
    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    common = dict(
        instrument_polarization_basis="linear",
        processing_set_data_group_name="base",
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 3},
    )
    stores = {}
    for name, options in {
        "xradio": dict(io_engine="xradio"),
        "chunk": dict(io_engine="chunk"),
        "fits": dict(io_engine="chunk", output_image_format="fits"),
    }.items():
        stores[name] = str(tmp_path / f"psf_{name}")
        psf_module.point_spread_function_cube_single_field(
            ps_store,
            {**_image_params(frequency_coords), "polarization_coords": ["I", "Q"]},
            {"weighting": "natural"},
            image_store=stores[name],
            **options,
            **common,
        )
    xradio = xr.open_zarr(stores["xradio"] + ".img.zarr").load()
    chunk = xr.open_zarr(stores["chunk"] + ".img.zarr").load()
    np.testing.assert_array_equal(
        chunk.POINT_SPREAD_FUNCTION.values, xradio.POINT_SPREAD_FUNCTION.values
    )
    fits = open_image(
        {"point_spread_function": stores["fits"] + "/POINT_SPREAD_FUNCTION.fits"}
    )
    np.testing.assert_allclose(
        fits.POINT_SPREAD_FUNCTION.values,
        xradio.POINT_SPREAD_FUNCTION.values,
        rtol=0,
        atol=1e-12,
    )


def _spy_fft_backends(monkeypatch):
    """The fft_backend of every transform of the FFT module, in call order."""
    from astroviper.processing_functions import fft

    seen = []
    for name in ("fft2", "ifft2"):
        real = getattr(fft, name)

        def spy(*args, _real=real, **kwargs):
            seen.append(kwargs["fft_backend"])
            return _real(*args, **kwargs)

        monkeypatch.setattr(fft, name, spy)
    return seen


@pytest.mark.parametrize("fft_backend", ["numpy", "scipy", "pyfftw"])
def test_the_fft_backend_reaches_the_transforms(ps_store, fft_backend, monkeypatch):
    """The application hands fft_backend through its node tasks to the gridder
    normalization: every transform of the FFT module runs in that library."""
    from xradio.measurement_set import open_processing_set

    seen = _spy_fft_backends(monkeypatch)
    frequency_coords = open_processing_set(ps_store).xr_ps.get_freq_axis().values
    psf_module.point_spread_function_cube_single_field(
        ps_store,
        _image_params(frequency_coords),
        {"weighting": "natural"},
        instrument_polarization_basis="circular",
        processing_set_data_group_name="base",
        fft_backend=fft_backend,
    )
    assert seen and set(seen) == {fft_backend}


@pytest.mark.parametrize("fft_backend", ["SciPy", "fftw", "NumPy"])
def test_an_unknown_fft_backend_is_refused(ps_store, fft_backend):
    """The schema refuses a misspelt library before any work."""
    with pytest.raises(AssertionError, match="fft_backend"):  # the param.json schema
        psf_module.point_spread_function_cube_single_field(
            ps_store,
            _image_params([3e9]),
            {"weighting": "natural"},
            instrument_polarization_basis="circular",
            fft_backend=fft_backend,
        )
