"""Output representation, disk layout, and codec validation."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_driver_workflow,
)
from astroviper.processing_functions.imaging.config_check._output import (
    check_output_layout,
)


def fits_config(**image):
    return {"output_image_format": "fits", "skunk_works": True, "image_params": image}


@pytest.mark.parametrize(
    "image",
    [
        {"time_coords": [0.0, 1.0]},
        {"image_size": [1, 4]},
        {"polarization_coords": ["I", "V"]},
        {"polarization_coords": ["bad"]},
        {"frequency_coords": [1e9, 1.3e9, 1.4e9]},
        {"frequency_coords": [1e9, 1e9]},
        {"direction_reference": "GALACTIC"},
    ],
)
def test_fits_coordinates_rejected_without_approximation(image):
    with pytest.raises(ValueError, match="FITS"):
        check_driver_workflow(fits_config(**image), mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize("frequency", [[1e9], [1.2e9, 1.1e9, 1e9]])
def test_fits_single_and_descending_uniform_channels(frequency):
    config = fits_config(
        time_coords=[0.0],
        image_size=[4, 4],
        frequency_coords=frequency,
        polarization_coords=["I", "Q"],
        direction_reference="ICRS",
    )
    config["image_data_variables_keep"] = ["sky_residual"]
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize(
    "kept", [["typo"], ["visibility"], ["beam_fit_params_point_spread_function"], []]
)
def test_fits_kept_variables_rejected(kept):
    config = fits_config()
    config["image_data_variables_keep"] = kept
    with pytest.raises(ValueError):
        check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)


def test_restore_effective_output_does_not_mutate_keep_list():
    config = fits_config()
    config.update(restore=True, image_data_variables_keep=[])
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)
    assert config["image_data_variables_keep"] == []


def test_layout_uses_task_boundaries_when_available():
    config = {
        "image_params": {"frequency_coords": list(range(8))},
        "image_chunking": {"frequency": 3},
    }
    check_output_layout(config)
    with pytest.raises(ValueError, match="divide every node task"):
        check_output_layout(
            config,
            parallel_coords={
                "frequency": {"data_chunks": {0: np.arange(4), 1: np.arange(4, 8)}}
            },
        )


@pytest.mark.parametrize(
    "sharding",
    [
        {"index_codecs": [{"name": "crc32c"}], "codecs": [{"name": "bytes"}]},
        {"codecs": [{"name": "bytes"}, {"name": "zstd"}, {"name": "blosc"}]},
        {"codecs": [{"name": "bytes"}, {"name": "unknown"}]},
    ],
)
def test_planned_sharded_codec_chain_rejected(sharding):
    config = {"skunk_works": True, "zarr_meta": {"SKY": {"sharding": sharding}}}
    with pytest.raises(ValueError):
        check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)


def test_planned_sharded_codec_chain_is_read_only():
    config = {
        "skunk_works": True,
        "zarr_meta": {
            "SKY": {
                "sharding": {
                    "index_codecs": [{"name": "bytes"}],
                    "codecs": [{"name": "bytes"}, {"name": "blosc"}],
                }
            }
        },
    }
    snapshot = deepcopy(config)
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)
    assert config == snapshot


def test_unsupported_codec_rejected_only_for_direct_writer(monkeypatch):
    from zarr.codecs import ZstdCodec

    # Third-party BytesBytesCodec implementations are valid for Zarr, but their
    # serialized names must also be implemented by the direct writer.
    monkeypatch.setattr(
        ZstdCodec, "to_dict", lambda self: {"name": "third_party_codec"}
    )
    config = {"compressor": ZstdCodec()}
    check_driver_workflow(config, mode=CUBE_SINGLE_FIELD)
    with pytest.raises(ValueError, match="direct writer"):
        check_driver_workflow({**config, "skunk_works": True}, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize("coordinates", [[], "I", [["I", "Q"]]])
def test_fits_requires_a_polarization_vector(coordinates):
    from astroviper.processing_functions.imaging.config_check._output import (
        check_fits_coordinates,
    )

    with pytest.raises(ValueError, match="polarization_coords"):
        check_fits_coordinates({"polarization_coords": coordinates})


def test_fits_requires_a_frequency_vector():
    from astroviper.processing_functions.imaging.config_check._output import (
        check_fits_coordinates,
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        check_fits_coordinates({"frequency_coords": [[1e9, 1.1e9]]})


@pytest.mark.parametrize(
    "kept",
    [
        "sky_residual",
        {"sky_residual": True},
        np.array([["sky_residual"]]),
        np.array("sky_residual"),
    ],
)
def test_output_variable_names_require_a_vector(kept):
    from astroviper.processing_functions.imaging.config_check._output import (
        check_kept_variables,
    )

    with pytest.raises(TypeError, match="sequence of variable names"):
        check_kept_variables({"image_data_variables_keep": kept})


@pytest.mark.parametrize(
    "mode,continuum", [(CUBE_SINGLE_FIELD, "mfs"), (CONTINUUM_SINGLE_FIELD, "mvc")]
)
@pytest.mark.parametrize("key", ["visibility", "uv_sampling", "aperture"])
def test_distributed_uv_outputs_are_refused_for_workflows_without_them(
    mode, continuum, key
):
    from astroviper.processing_functions.imaging.config_check._output import (
        check_kept_variables,
    )

    config = {"image_data_variables_keep": [key], "continuum_mode": continuum}
    with pytest.raises(ValueError, match="UV-grid outputs"):
        check_kept_variables(config, mode=mode, distributed=True)
    check_kept_variables(config, mode=mode, distributed=False)


def test_zarr_v2_codec_contract_rejects_unknown_objects(monkeypatch):
    import zarr
    from numcodecs import Blosc

    from astroviper.processing_functions.imaging.config_check._output import (
        check_output_codec,
    )

    monkeypatch.setattr(zarr, "__version__", "2.18.7")
    with pytest.raises(TypeError, match="Zarr v2 compressor"):
        check_output_codec({"compressor": object()})
    check_output_codec({"compressor": Blosc()})
