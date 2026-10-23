"""Check requested output representations before creating any files."""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
    require_mapping,
)


def check_fits_coordinates(image_params):
    """Require image coordinates exactly representable by the cube FITS writer.

    Parameters
    ----------
    image_params : mapping
        Supplied image coordinates. Only available keys are checked.
    """
    # --- Number of axes ---
    # The implemented four-axis FITS image has no independent time axis.
    if "time_coords" in image_params and np.size(image_params["time_coords"]) != 1:
        raise ValueError("FITS output supports a single time plane.")
    if "image_size" in image_params and np.any(
        np.asarray(image_params["image_size"]) < 2
    ):
        raise ValueError("FITS output needs at least 2 pixels per direction axis.")

    # --- Polarization axis ---
    # A linear FITS axis can encode consecutive Stokes/correlation codes only.
    if "polarization_coords" in image_params:
        codes_by_label = {
            "I": 1,
            "Q": 2,
            "U": 3,
            "V": 4,
            "RR": -1,
            "LL": -2,
            "RL": -3,
            "LR": -4,
            "XX": -5,
            "YY": -6,
            "XY": -7,
            "YX": -8,
        }
        labels = image_params["polarization_coords"]
        if (
            not isinstance(labels, list | tuple | np.ndarray)
            or np.ndim(labels) != 1
            or not len(labels)
        ):
            raise ValueError("FITS polarization_coords must be a nonempty vector.")
        if any(label not in codes_by_label for label in labels):
            raise ValueError("FITS polarization_coords contains an unknown label.")
        codes = [codes_by_label[label] for label in labels]
        if codes != list(range(codes[0], codes[0] + len(codes))):
            raise ValueError("FITS output requires consecutive polarization codes.")

    # --- Spectral axis ---
    # Do not substitute the first-to-last linear approximation for real centres.
    # Use the writer's tolerance (one millionth of a channel) for round-off.
    if "frequency_coords" in image_params:
        frequencies = require_finite_real(
            "FITS frequency_coords", image_params["frequency_coords"], positive=True
        )
        if frequencies.ndim != 1:
            raise ValueError("FITS frequency_coords must be one-dimensional.")
        if frequencies.size > 1:
            increment = (frequencies[-1] - frequencies[0]) / (frequencies.size - 1)
            if increment == 0:
                raise ValueError("FITS frequency coordinates have zero overall span.")
            linear = frequencies[0] + increment * np.arange(frequencies.size)
            if np.max(np.abs(frequencies - linear)) > 1e-6 * abs(increment):
                raise ValueError(
                    "FITS output requires uniformly spaced frequency_coords; use Zarr to preserve the channel centres."
                )

    # --- Direction frame ---
    # The writer/reader pair supports RA/DEC in these two reference frames.
    if "direction_reference" in image_params:
        frame = image_params["direction_reference"]
        if not isinstance(frame, str) or frame.upper() not in ("FK5", "ICRS"):
            raise ValueError("FITS direction_reference must be FK5 or ICRS.")


def check_kept_variables(config, *, fits=False, require_image=True):
    """Check the requested variable registry and the FITS representation.

    Parameters
    ----------
    config : mapping
        Output configuration, including restoration switches.
    fits : bool
        Whether to require the fixed FITS writer's image/beam dimensions.
    """
    from astroviper.utils.io import imaging_data_variables_and_dims_double_precision

    # --- Registry keys ---
    # Reject a typo before the output code indexes its variable definitions.
    kept = config.get("image_data_variables_keep")
    if kept is None:
        return
    if (
        isinstance(kept, str)
        or not isinstance(kept, list | tuple | np.ndarray)
        or np.ndim(kept) != 1
    ):
        raise TypeError(
            "image_data_variables_keep must be a sequence of variable names."
        )
    definitions = imaging_data_variables_and_dims_double_precision
    if any(not isinstance(name, str) or name not in definitions for name in kept):
        raise ValueError(
            "image_data_variables_keep contains an unknown imaging variable."
        )
    if not fits:
        return

    # --- Effective output set ---
    # Restoration adds mandatory outputs locally; never append to the caller list.
    effective = set(kept)
    if config.get("restore", False):
        effective.add("sky_restored")
    if config.get("primary_beam_correction", False):
        effective.add("sky_restored_primary_beam_corrected")
    image_dims = ("time", "frequency", "polarization", "l", "m")
    beam_dims = ("time", "frequency", "polarization", "beam_params_label")
    unsupported = [
        name
        for name in effective
        if tuple(definitions[name]["dims"]) not in (image_dims, beam_dims)
    ]
    if unsupported:
        raise ValueError(
            f"Variables {sorted(unsupported)} have no FITS representation."
        )
    if require_image and not any(
        tuple(definitions[name]["dims"]) == image_dims for name in effective
    ):
        raise ValueError(
            "image_data_variables_keep contains no image-domain variables for FITS."
        )


def check_output_layout(config, *, parallel_coords=None):
    """Check disk layout using parameters and, when available, task boundaries.

    Parameters
    ----------
    config : mapping
        Image coordinates and optional image_chunking/image_sharding mappings.
    parallel_coords : mapping or None
        Generated task coordinates. Absent during the first parameter check.
    """
    # --- Layout schema ---
    # A positive count and a known dimension are decidable before partitioning.
    from astroviper.utils.io import validate_image_chunking_and_sharding

    # Before mapping, only image extents and explicit disk chunks are known.
    # Once task coordinates exist, the same validator checks write alignment.
    image = config.get("image_params", {})
    shape = {
        dim: len(image[f"{dim}_coords"])
        for dim in ("time", "frequency", "polarization")
        if f"{dim}_coords" in image
    }
    if "image_size" in image:
        shape.update(zip(("l", "m"), image["image_size"], strict=True))
    validate_image_chunking_and_sharding(
        config.get("image_chunking"),
        config.get("image_sharding"),
        shape,
        parallel_coords,
        extra_dims=("time", "frequency", "polarization", "l", "m", "u", "v"),
        parallelism_pending=parallel_coords is None,
    )


def check_output_codec(config):
    """Check the selected compressor against the writer without file access.

    Parameters
    ----------
    config : mapping
        Output settings with an optional compressor instance.
    """
    # --- Supplied/planned direct-writer metadata ---
    # Validate known codec chains before dispatch; actual disk metadata is still
    # checked by the original reader/writer guards at the point of use.
    if config.get("skunk_works", False) and config.get("zarr_meta"):
        metadata = require_mapping("zarr_meta", config["zarr_meta"])
        for name, meta in metadata.items():
            require_mapping(f"zarr_meta[{name!r}]", meta)
            sharding = meta.get("sharding")
            if sharding:
                if any(
                    codec.get("name") == "crc32c"
                    for codec in sharding.get("index_codecs", [])
                ):
                    raise ValueError(
                        "The concurrent sharded writer does not support index crc32c."
                    )
                codecs = sharding.get("codecs", [])
            else:
                codecs = meta.get("codecs", [])
            transformed = False
            for codec in codecs:
                codec_name = codec.get("name")
                if codec_name in ("bytes", "sharding_indexed"):
                    continue
                if codec_name not in ("blosc", "zstd", "gzip", "gz", "crc32c"):
                    raise ValueError(
                        f"The direct writer does not support codec {codec_name!r}."
                    )
                if sharding and codec_name == "blosc" and transformed:
                    raise ValueError(
                        "blosc must be the first transform codec in a shard."
                    )
                transformed = True

    compressor = config.get("compressor")
    if compressor is None or config.get("output_image_format", "zarr") == "fits":
        return
    import zarr

    # --- Zarr compressor interface ---
    # The standard v3 writer needs a BytesBytesCodec or a supported translation.
    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.abc.codec import BytesBytesCodec

        from astroviper.utils.io import _to_zarr_v3_codec

        if not isinstance(compressor, BytesBytesCodec) and not callable(
            getattr(compressor, "get_config", None)
        ):
            raise TypeError("compressor must be a supported Zarr/numcodecs codec.")
        codec = _to_zarr_v3_codec(compressor)
        description = codec.to_dict()
        codec_name = description["name"]
    else:
        from numcodecs.abc import Codec

        if not isinstance(compressor, Codec):
            raise TypeError("Zarr v2 compressor must be a numcodecs codec.")
        codec_name = compressor.get_config()["id"]

    # --- Direct writer codec support ---
    # That writer implements a smaller codec set than the Zarr library itself.
    if config.get("skunk_works", False) and codec_name not in (
        "blosc",
        "zstd",
        "gzip",
        "crc32c",
        "bytes",
    ):
        raise ValueError(
            f"The direct writer does not support compressor {codec_name!r}."
        )
