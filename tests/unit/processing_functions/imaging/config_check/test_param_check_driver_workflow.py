"""Workflow configuration and partition preflight before graph construction."""

from copy import deepcopy

import pytest

from astroviper.processing_functions.imaging.config_check import (
    CUBE_SINGLE_FIELD,
    check_driver_workflow,
)
from tests.unit.distributed_applications.imaging.test_continuum import (
    wideband_store as wideband_store,
)


@pytest.mark.parametrize(
    "options",
    [
        {"image_chunking": []},
        {"image_chunking": {"bogus": 1}},
        {"image_chunking": {"frequency": True}},
        {"image_chunking": {"frequency": 0}},
        {"msv2_open_options": []},
        {"msv2_open_options": {"array_backend": "numpy"}},
        {"msv2_open_options": {"scan_intents": None}},
        {"compressor": object()},
    ],
)
def test_workflow_schema_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        check_driver_workflow(options, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize(
    "options",
    [
        {"skunk_works": False},
        {"skunk_works": True, "write_visibility_model_to_ps": True},
        {"skunk_works": True, "write_imaging_weights_to_ps": True},
    ],
)
def test_msv2_storage_restrictions(options):
    from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
        check_input_storage,
    )

    original = options.copy()
    with pytest.raises(ValueError, match="Measurement Set"):
        check_input_storage(options, input_is_msv2=True)
    check_input_storage(options, input_is_msv2=False)
    assert options == original


def test_msv2_read_only_supported_path():
    from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
        check_input_storage,
    )

    check_input_storage({"skunk_works": True}, input_is_msv2=True)
    with pytest.raises(ValueError, match="convert"):
        check_input_storage({"skunk_works": True}, input_is_msv2=True, allow_msv2=False)


@pytest.mark.parametrize("kind", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize("fault", ["missing", "repeated", "unknown", "taylor_basis"])
@pytest.mark.parametrize("input_checked", [False, True])
def test_drivers_validate_actual_partitions_before_graphs_or_writes(
    monkeypatch, tmp_path, wideband_store, kind, fault, input_checked
):
    import importlib

    import graphviper.graph_tools
    import graphviper.graph_tools.coordinate_utils as coordinate_utils
    import xradio.measurement_set

    from tests.unit.distributed_applications.imaging.test_continuum import (
        _tw_hydra_image_params,
    )

    source = xradio.measurement_set.open_processing_set(str(wideband_store))
    image = _tw_hydra_image_params(source, {"n_taylor_terms": 1})
    controls = {
        "max_iter": 0,
        "max_cycles": 0,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.0,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_iter_per_cycle": -1,
    }
    original = coordinate_utils.interpolate_data_coords_onto_parallel_coords

    def corrupt_mapping(mapping):
        mapping = deepcopy(mapping)
        first = next(iter(mapping))
        if fault == "missing":
            mapping.clear()
        elif fault == "repeated":
            mapping[max(mapping) + 1] = deepcopy(mapping[first])
        elif fault == "unknown":
            mapping[first]["data_selection"]["unknown-ms"] = {"frequency": slice(None)}
        else:
            mapping[first]["image_params"] = {
                "n_taylor_terms": 2,
                "reference_frequency": image["reference_frequency"],
            }
        return mapping

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Invalid channel partitions reached a later preflight, graph, or write"
        )

    module = importlib.import_module(
        "astroviper.distributed_applications.imaging.image_cube_single_field"
        if kind == "cube"
        else "astroviper.distributed_applications.imaging.image_continuum_single_field"
    )
    if kind == "cube":
        monkeypatch.setattr(
            coordinate_utils,
            "interpolate_data_coords_onto_parallel_coords",
            lambda *args, **kwargs: corrupt_mapping(original(*args, **kwargs)),
        )
    else:
        # Corrupt the final mapping after exact-frequency selection, so the
        # test exercises the shared partition guard rather than an earlier one.
        select = module._apply_exact_frequency_selection_to_continuum_mapping
        monkeypatch.setattr(
            module,
            "_apply_exact_frequency_selection_to_continuum_mapping",
            lambda *args, **kwargs: corrupt_mapping(select(*args, **kwargs)),
        )
    monkeypatch.setattr(
        "astroviper.node_tasks.imaging.config_check.preflight_imaging_inputs", forbidden
    )
    monkeypatch.setattr(graphviper.graph_tools, "generate_dask_workflow", forbidden)
    monkeypatch.setattr("xradio.image.write_image", forbidden)
    function = (
        module.image_cube_single_field
        if kind == "cube"
        else module.image_continuum_single_field
    )
    options = {} if kind == "cube" else {"continuum_mode": kind}
    with pytest.raises(ValueError, match="Channel partition|Workflow configuration"):
        function(
            str(wideband_store),
            str(tmp_path / "refused.img.zarr"),
            image,
            {"weighting": "natural"},
            controls,
            deconvolver="hogbom",
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 2},
            input_checked=input_checked,
            image_data_variables_keep=["sky_residual"],
            **options,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "missing", ["image_params", "imaging_weights_params", "cycle_control_params"]
)
def test_complete_workflow_requires_each_configuration_group(missing):
    config = {
        name: {}
        for name in ("image_params", "imaging_weights_params", "cycle_control_params")
    }
    del config[missing]
    with pytest.raises(KeyError, match=missing):
        check_driver_workflow(config, mode=CUBE_SINGLE_FIELD, require_complete=True)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"compute_backend": "gpu"}, "compute_backend"),
        ({"write_visibility_model_to_ps": True}, "write_visibility_model_to_ps"),
        ({"write_imaging_weights_to_ps": True}, "write_imaging_weights_to_ps"),
        ({"image_sharding": {"frequency": 1}}, "skunk_works"),
        ({"output_image_format": "hdf5"}, "output_image_format"),
        ({"output_image_format": "fits"}, "FITS.*skunk_works"),
        (
            {
                "output_image_format": "fits",
                "skunk_works": True,
                "image_chunking": {"frequency": 1},
            },
            "FITS.*image_chunking",
        ),
        ({"reduce_mode": "bad"}, "reduce_mode"),
        (
            {
                "primary_beam_limit": 0.2,
                "cycle_control_params": {"primary_beam_limit": 0.3},
            },
            "conflicts",
        ),
    ],
)
def test_invalid_workflow_options_rejected_without_fallback(options, message):
    original = deepcopy(options)
    with pytest.raises(ValueError, match=message):
        check_driver_workflow(options, mode=CUBE_SINGLE_FIELD)
    assert options == original


def test_partition_unknown_source_and_configuration_presence_are_refused():
    import xarray as xr

    from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
        check_channel_partitions,
        check_configuration_consistency,
    )

    with pytest.raises(ValueError, match="unknown Measurement Set"):
        check_channel_partitions(
            {0: {"data_selection": {"missing": {}}}},
            {"ms": xr.Dataset(coords={"frequency": [1e9]})},
            image_params={"frequency_coords": [1e9]},
            mode=CUBE_SINGLE_FIELD,
        )
    with pytest.raises(ValueError, match="changed presence"):
        check_configuration_consistency(
            {"n_taylor_terms": 2}, {}, names=["n_taylor_terms"]
        )


@pytest.mark.parametrize(
    "node,params,message",
    [
        (None, {}, "without an append_node"),
        (lambda: None, None, "without append_input_params"),
    ],
)
def test_append_pair_requires_both_arguments_at_the_shared_boundary(
    node, params, message
):
    from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
        check_continuum_graph_arguments,
    )

    with pytest.raises(ValueError, match=message):
        check_continuum_graph_arguments(node, params)


@pytest.mark.parametrize("node,params", [(None, None), (lambda: None, {})])
def test_append_pair_accepts_absent_or_complete_arguments(node, params):
    from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
        check_continuum_graph_arguments,
    )

    check_continuum_graph_arguments(node, params)
