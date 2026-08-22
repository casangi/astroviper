"""Selection, loading, and local execution for image statistics.

This module is the node-task layer of ``imstatistics``. It chooses the image
variable, translates CASA-like selectors into source-image positions,
intersects the logical user selection with a GraphVIPER node partition, loads
only the required pixels and masks, and calls the pure processing functions.

Selection and reduction are deliberately separate. For example,
``chans="2~5"`` first restricts the input to four channels. If ``frequency`` is
in ``axes``, those channels are reduced; otherwise they remain as a four-element
output dimension. Unselected coordinates are not restored or zero-filled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_IMAGE_DIMS = ("time", "frequency", "polarization", "l", "m")
_DEFAULT_STATISTICS = ("max", "min", "sum", "mean", "npts")
ImageIndexer = slice | np.ndarray


@dataclass(frozen=True)
class ImageSelection:
    """Positional user, execution, and effective image selections.

    ``user_indexers`` describes the logical request. ``partition_indexers``
    describes the part assigned to this node, in source-image positions.  Only
    their intersection is exposed to storage through ``effective_indexers``.
    Keeping the two inputs separate prevents an execution partition from
    replacing or expanding the user's request.

    Attributes
    ----------
    user_indexers : dict[str, slice or numpy.ndarray]
        Absolute zero-based source selection from user-facing parameters.
        Regular selections remain compact slices; only irregular selections
        use integer arrays. Every image dimension is present.
    partition_indexers : dict[str, slice or numpy.ndarray]
        Absolute source positions assigned to this node. Relative GraphVIPER
        positions are converted to source positions during construction.
    effective_indexers : dict[str, slice or numpy.ndarray]
        Ordered user/partition intersection passed directly to ``isel`` or
        XRADIO storage loading.
    boxes : tuple
        Inclusive ``(l0, m0, l1, m1)`` boxes in source pixel coordinates.
    reduction_dims : tuple[str, ...]
        Dimensions removed by the requested reduction.
    """

    user_indexers: dict[str, ImageIndexer]
    partition_indexers: dict[str, ImageIndexer]
    effective_indexers: dict[str, ImageIndexer]
    boxes: tuple[tuple[int, int, int, int], ...]
    reduction_dims: tuple[str, ...]

    @property
    def indexers(self) -> dict[str, ImageIndexer]:
        """Storage-ready indexers (compatibility alias)."""
        return self.effective_indexers


# Compatibility for callers which used the implementation name during the
# initial image-statistics work.  New code should use ImageSelection.
ImageSlicer = ImageSelection


def _compact_positions(positions: np.ndarray) -> ImageIndexer:
    """Use the smallest supported representation for absolute positions.

    Parameters
    ----------
    positions : numpy.ndarray
        One-dimensional, ordered absolute pixel positions.

    Returns
    -------
    slice or numpy.ndarray
        A slice for a singleton or constant-positive-step progression. Empty,
        descending, duplicated, and irregular positions remain arrays.

    Notes
    -----
    This helper is primarily a fallback for selectors that already arrived as
    arrays. Parsers should produce slices directly where possible so a large
    temporary array is never allocated merely to compress it afterward.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("Positional index arrays must be one-dimensional")
    if positions.size == 0:
        return positions
    if positions.size == 1:
        position = int(positions[0])
        return slice(position, position + 1, 1)
    steps = np.diff(positions)
    if np.all(steps == steps[0]) and steps[0] > 0:
        return slice(int(positions[0]), int(positions[-1]) + 1, int(steps[0]))
    return positions


def _positions(selector, size: int) -> ImageIndexer:
    """Normalize a selector relative to an axis without expanding slices.

    Parameters
    ----------
    selector : int, slice, or array-like
        Relative positional selector. Negative integer positions follow NumPy
        semantics; a Boolean array must have length ``size``.
    size : int
        Length of the axis against which the selector is normalized.

    Returns
    -------
    slice or numpy.ndarray
        A concrete slice when the normalized positions are regular, otherwise
        a one-dimensional integer array.

    Raises
    ------
    ValueError
        If an array selector is not one-dimensional or a Boolean selector has
        the wrong length.
    IndexError
        If an integer position lies outside the axis.
    """
    if isinstance(selector, slice):
        start, stop, step = selector.indices(size)
        return slice(start, stop, step)
    if isinstance(selector, int | np.integer):
        position = int(selector)
        position = position + size if position < 0 else position
        if position < 0 or position >= size:
            raise IndexError(f"Selection is outside axis length {size}")
        return slice(position, position + 1, 1)
    selected = np.asarray(selector)
    if selected.dtype == bool:
        if selected.ndim != 1 or selected.size != size:
            raise ValueError("Boolean positional selector has the wrong length")
        selected = np.flatnonzero(selected)
    elif selected.ndim != 1:
        raise ValueError("Positional index arrays must be one-dimensional")
    selected = selected.astype(np.int64, copy=False)
    selected = np.where(selected < 0, selected + size, selected)
    if np.any((selected < 0) | (selected >= size)):
        raise IndexError(f"Selection is outside axis length {size}")
    return _compact_positions(selected)


def _indexer_length(indexer: ImageIndexer) -> int:
    """Return selected axis length without materializing a slice.

    Parameters
    ----------
    indexer : slice or numpy.ndarray
        Concrete compact image indexer.

    Returns
    -------
    int
        Number of selected positions.
    """
    if isinstance(indexer, slice):
        return len(range(indexer.start, indexer.stop, indexer.step))
    return len(indexer)


def _indexer_positions(indexer: ImageIndexer) -> np.ndarray:
    """Materialize absolute positions for coordinate-dependent operations.

    Parameters
    ----------
    indexer : slice or numpy.ndarray
        Concrete compact image indexer.

    Returns
    -------
    numpy.ndarray
        Absolute integer pixel positions.

    Notes
    -----
    Storage selection must use the compact indexer directly. This conversion is
    reserved for operations such as constructing a disjoint-box union mask.
    """
    if isinstance(indexer, slice):
        return np.arange(indexer.start, indexer.stop, indexer.step, dtype=np.int64)
    return np.asarray(indexer, dtype=np.int64)


def _take_indexer(source: ImageIndexer, relative_selector) -> ImageIndexer:
    """Compose a worker-relative selector with a user image selection.

    Parameters
    ----------
    source : slice or numpy.ndarray
        Absolute positions selected by the user.
    relative_selector : int, slice, or array-like
        Worker selection expressed relative to ``source`` rather than the
        original image axis.

    Returns
    -------
    slice or numpy.ndarray
        Absolute source-image selection for the worker. Slice-with-slice
        composition remains constant-memory; irregular results remain arrays.

    Examples
    --------
    ``slice(100, 200)`` with relative ``slice(20, 40)`` becomes absolute
    ``slice(120, 140)``.
    """
    relative = _positions(relative_selector, _indexer_length(source))
    if isinstance(source, slice):
        source_range = range(source.start, source.stop, source.step)
        if isinstance(relative, slice):
            selected = source_range[relative]
            return slice(selected.start, selected.stop, selected.step)
        return _compact_positions(
            source.start + np.asarray(relative, dtype=np.int64) * source.step
        )
    if isinstance(relative, slice):
        return _compact_positions(source[relative])
    return _compact_positions(source[relative])


def _index_expression(expression: str, size: int, name: str) -> ImageIndexer:
    """Parse a CASA-like integer-axis expression into a compact indexer.

    Parameters
    ----------
    expression : str
        Comma/semicolon-separated scalars, inclusive ``a~b`` ranges, stepped
        ``a~b^step`` ranges, or ``<``, ``<=``, ``>``, and ``>=`` comparisons.
        An empty expression selects the full axis.
    size : int
        Source-axis length.
    name : str
        User-facing selector name included in error messages, such as
        ``"chans"`` or ``"timerange"``.

    Returns
    -------
    slice or numpy.ndarray
        A slice for a regular selection or regular union. A sorted, unique
        integer array is returned only when the selected positions are
        genuinely irregular.

    Raises
    ------
    ValueError
        If the syntax is unsupported, the step is invalid, or no positions are
        specified.
    IndexError
        If a scalar or range endpoint is outside the axis.

    Examples
    --------
    ``"2~20^2"`` becomes ``slice(2, 21, 2)`` while ``"1,4,8~10"`` becomes
    ``array([1, 4, 8, 9, 10])``.
    """
    # An empty expression selects the full axis with a constant-memory slice.
    if expression is None or str(expression).strip() == "":
        return slice(0, size, 1)
    selected: list[range] = []
    # Parse every comma- or semicolon-separated union member independently.
    for raw_token in re.split(r"[;,]", str(expression)):
        token = raw_token.strip().replace(" ", "")
        if not token:
            continue
        # Convert an inclusive, optionally stepped interval to a compact range.
        match = re.fullmatch(r"(-?\d+)~(-?\d+)(?:\^(\d+))?", token)
        if match:
            start, stop = int(match.group(1)), int(match.group(2))
            step = int(match.group(3) or 1)
            if step < 1:
                raise ValueError(f"{name} range step must be positive")
            direction = 1 if stop >= start else -1
            if start < -size or start >= size or stop < -size or stop >= size:
                raise IndexError(f"{name} selection is outside axis length {size}")
            values = range(start, stop + direction, direction * step)
            crosses_zero = (start < 0 <= stop) or (stop < 0 <= start)
            if not crosses_zero:
                first = values[0] + size if values[0] < 0 else values[0]
                last_value = values[-1]
                last = last_value + size if last_value < 0 else last_value
                selected.append(range(min(first, last), max(first, last) + 1, step))
            else:
                normalized = [value + size if value < 0 else value for value in values]
                selected.extend(range(value, value + 1) for value in normalized)
            continue
        # Convert a comparison to its equivalent bounded interval on the axis.
        match = re.fullmatch(r"(<=|>=|<|>)(-?\d+)", token)
        if match:
            op, boundary_text = match.groups()
            boundary = int(boundary_text)
            if op == "<":
                selected.append(range(0, min(max(boundary, 0), size)))
            elif op == "<=":
                selected.append(range(0, min(max(boundary + 1, 0), size)))
            elif op == ">":
                selected.append(range(min(max(boundary + 1, 0), size), size))
            else:
                selected.append(range(min(max(boundary, 0), size), size))
            continue
        # Represent a scalar as a one-position interval to retain its dimension.
        if re.fullmatch(r"-?\d+", token):
            position = int(token)
            position = position + size if position < 0 else position
            if position < 0 or position >= size:
                raise IndexError(f"{name} selection is outside axis length {size}")
            selected.append(range(position, position + 1))
            continue
        raise ValueError(f"Unsupported {name} selector token {raw_token!r}")
    if not selected:
        raise ValueError(f"{name} selection is empty")
    # A single parsed interval maps directly to an isel-compatible slice.
    if len(selected) == 1:
        values = selected[0]
        return slice(values.start, values.stop, values.step)
    # Combine a regular union into one slice without expanding its positions.
    nonempty = sorted((values for values in selected if values), key=lambda x: x.start)
    if nonempty:
        common_step = nonempty[0].step
        first = nonempty[0].start
        last = nonempty[0][-1]
        regular_union = True
        for values in nonempty[1:]:
            if (
                values.step != common_step
                or (values.start - first) % common_step
                or values.start > last + common_step
            ):
                regular_union = False
                break
            last = max(last, values[-1])
        if regular_union:
            return slice(first, last + 1, common_step)
    # Materialize only a genuinely irregular union as sorted unique positions.
    indices = np.unique(
        np.fromiter((value for item in selected for value in item), int)
    )
    return _compact_positions(indices)


def _parse_boxes(box: str | None, l_size: int, m_size: int):
    """Parse inclusive pixel-box text into validated spatial bounds.

    Parameters
    ----------
    box : str, optional
        Comma-separated groups of ``l0,m0,l1,m1`` pixel coordinates.
    l_size, m_size : int
        Direction-plane axis lengths used for bounds validation.

    Returns
    -------
    tuple of tuple of int
        Inclusive ``(l0, m0, l1, m1)`` boxes.
    """
    if box is None or str(box).strip() == "":
        return ()
    values = [int(value.strip()) for value in str(box).split(",") if value.strip()]
    if len(values) % 4:
        raise ValueError("box must contain groups of x0,y0,x1,y1")
    boxes = []
    for offset in range(0, len(values), 4):
        l0, m0, l1, m1 = values[offset : offset + 4]
        if l0 > l1 or m0 > m1:
            raise ValueError("box lower-left coordinate must precede upper-right")
        if l0 < 0 or m0 < 0 or l1 >= l_size or m1 >= m_size:
            raise IndexError("box lies outside the image direction plane")
        boxes.append((l0, m0, l1, m1))
    return tuple(boxes)


def _region_boxes(region: Any, l_size: int, m_size: int):
    """Extract pixel boxes from a region record, CRTF text, or CRTF file.

    Parameters
    ----------
    region : dict, str, pathlib.Path, or None
        Region record with ``blc``/``trc``, CRTF pixel-box text, plain box
        coordinates, or a path containing one of those textual forms.
    l_size, m_size : int
        Direction-plane axis lengths used for bounds validation.

    Returns
    -------
    tuple of tuple of int
        Inclusive, validated pixel boxes.

    Raises
    ------
    ValueError
        If the region is not one of the supported pixel-box forms.
    """
    if region is None or region == "":
        return ()
    if isinstance(region, dict):
        if not {"blc", "trc"} <= set(region):
            raise ValueError("A region record must contain 'blc' and 'trc'")
        blc, trc = region["blc"], region["trc"]
        return _parse_boxes(
            f"{int(blc[0])},{int(blc[1])},{int(trc[0])},{int(trc[1])}",
            l_size,
            m_size,
        )
    text = Path(region).read_text() if isinstance(region, Path) else str(region)
    possible_path = Path(text.strip("`"))
    try:
        is_file = "\n" not in text and possible_path.is_file()
    except OSError:
        is_file = False
    if is_file:
        text = possible_path.read_text()
    matches = re.findall(
        r"box\s*\[\s*\[\s*(-?\d+)\s*pix\s*,\s*(-?\d+)\s*pix\s*\]\s*,"
        r"\s*\[\s*(-?\d+)\s*pix\s*,\s*(-?\d+)\s*pix\s*\]\s*\]",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return _parse_boxes(
            ",".join(value for match in matches for value in match), l_size, m_size
        )
    if re.fullmatch(r"[\d\s,\-]+", text):
        return _parse_boxes(text, l_size, m_size)
    raise ValueError("Only pixel-coordinate box regions are currently supported")


def _polarization_indices(expression: str, coordinate: xr.DataArray) -> ImageIndexer:
    """Resolve polarization labels to a compact positional indexer.

    Parameters
    ----------
    expression : str
        One label, concatenated single-character labels, or comma-separated
        labels. An empty expression selects every polarization.
    coordinate : xarray.DataArray
        Polarization coordinate whose values define valid labels and order.

    Returns
    -------
    slice or numpy.ndarray
        A slice when requested labels occupy a regular progression, otherwise
        their integer positions.
    """
    labels = [str(value) for value in coordinate.values]
    if expression is None or str(expression).strip() == "":
        return slice(0, len(labels), 1)
    text = str(expression).replace(" ", "")
    if "," in text:
        requested = [value for value in text.split(",") if value]
    elif text in labels:
        requested = [text]
    elif all(len(label) == 1 for label in labels):
        requested = list(text)
    else:
        requested = []
        remaining = text
        candidates = sorted(labels, key=len, reverse=True)
        while remaining:
            match = next(
                (label for label in candidates if remaining.startswith(label)), None
            )
            if match is None:
                raise ValueError(f"Cannot parse stokes selection {expression!r}")
            requested.append(match)
            remaining = remaining[len(match) :]
    unknown = set(requested) - set(labels)
    if unknown:
        raise ValueError(f"Unknown polarization labels: {sorted(unknown)}")
    return _compact_positions(
        np.asarray([labels.index(label) for label in requested], dtype=np.int64)
    )


def _reduction_dims(axes, dims: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize named or integer statistic axes to unique dimension names.

    Parameters
    ----------
    axes : int, str, sequence, or None
        Axes requested for reduction. ``-1`` and ``None`` select all axes.
    dims : tuple of str
        Image dimension order used to resolve integer axes.

    Returns
    -------
    tuple of str
        Unique dimensions in requested order.
    """
    if axes == -1 or axes is None:
        return dims
    values = [axes] if isinstance(axes, str | int | np.integer) else list(axes)
    result = []
    for value in values:
        if isinstance(value, int | np.integer):
            value = dims[int(value)]
        if value not in dims:
            raise ValueError(f"Unknown statistics axis {value!r}")
        if value not in result:
            result.append(value)
    return tuple(result)


def _choose_data_array(
    image: xr.DataArray | xr.Dataset, data_variable: str | None
) -> tuple[xr.Dataset, str]:
    """Normalize an image object and resolve the analyzed data variable.

    Parameters
    ----------
    image : xarray.DataArray or xarray.Dataset
        Image metadata or in-memory image.
    data_variable : str, optional
        Explicit variable name. If omitted from a dataset, exactly one variable
        must contain all canonical image dimensions.

    Returns
    -------
    xarray.Dataset
        Dataset representation of ``image``.
    str
        Resolved image-variable name.
    """
    if isinstance(image, xr.DataArray):
        name = data_variable or image.name or "IMAGE"
        return image.rename(name).to_dataset(), name
    if not isinstance(image, xr.Dataset):
        raise TypeError("image must be an xarray DataArray, Dataset, or on-disk path")
    if data_variable is not None:
        if data_variable not in image:
            raise KeyError(f"Image has no data variable {data_variable!r}")
        return image, data_variable
    candidates = [
        name
        for name, value in image.data_vars.items()
        if set(_IMAGE_DIMS) <= set(value.dims)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "data_variable is required when the image does not have exactly one five-axis variable"
        )
    return image, candidates[0]


def _open_metadata(image, data_variable):
    """Open lazy Zarr metadata or normalize an in-memory image.

    Parameters
    ----------
    image : str, pathlib.Path, xarray.DataArray, or xarray.Dataset
        On-disk Zarr image or in-memory image object.
    data_variable : str, optional
        Explicit image-variable name.

    Returns
    -------
    xarray.Dataset
        Dataset whose on-disk variables remain lazy.
    str
        Resolved image-variable name.
    """
    if isinstance(image, str | Path):
        dataset = xr.open_zarr(str(image))
    else:
        dataset = image
    return _choose_data_array(dataset, data_variable)


def _required_variables(metadata: xr.Dataset, variable: str, mask) -> list[str]:
    """Identify the minimal set of data variables a node must load.

    Parameters
    ----------
    metadata : xarray.Dataset
        Source image metadata.
    variable : str
        Image data variable being analyzed.
    mask : str or object
        Named mask or an external mask. Only a named mask adds an on-disk
        variable to the result.

    Returns
    -------
    list of str
        Image variable followed by the optional named mask variable.
    """
    required = [variable]
    if isinstance(mask, str) and mask:
        if mask not in metadata:
            raise KeyError(f"Image has no mask variable {mask!r}")
        required.append(mask)
    return required


def build_image_selection(
    data: xr.DataArray,
    *,
    axes=-1,
    region="",
    box="",
    chans="",
    stokes="",
    timerange="",
    partition: dict[str, slice | np.ndarray] | None = None,
) -> ImageSelection:
    """Build one storage-ready selection from user and execution selectors.

    Parameters
    ----------
    data : xarray.DataArray
        Source image metadata. Pixel values need not be loaded.
    axes : int, str, or sequence, default -1
        Dimensions to reduce. ``-1`` and ``None`` mean every dimension.
    region, box, chans, stokes, timerange
        User-facing selectors accepted by :func:`image_statistics`.
    partition : mapping, optional
        Positional selector for one node, relative to the user selection.

    Returns
    -------
    ImageSelection
        User, partition, and effective positions in a common absolute
        positional language.

    Notes
    -----
    The order is ``user selection -> partition normalization -> intersection``.
    A partition can therefore restrict but never expand the user request.

    With five channels, ``chans="1~4"`` and partition
    ``{"frequency": slice(1, 3)}`` produce the compact user indexer
    ``slice(1, 5, 1)`` and effective source indexer ``slice(2, 4, 1)``.
    """
    dims = tuple(data.dims)
    user_indexers = {dim: slice(0, data.sizes[dim], 1) for dim in dims}
    if "frequency" in dims:
        user_indexers["frequency"] = _index_expression(
            chans, data.sizes["frequency"], "chans"
        )
    if "time" in dims:
        user_indexers["time"] = _index_expression(
            timerange, data.sizes["time"], "timerange"
        )
    if "polarization" in dims:
        user_indexers["polarization"] = _polarization_indices(
            stokes, data["polarization"]
        )
    if box and region:
        raise ValueError("Specify either region or box, not both")
    boxes = ()
    if box or region:
        if not {"l", "m"} <= set(dims):
            raise ValueError("Pixel regions require both 'l' and 'm' dimensions")
        boxes = _parse_boxes(box, data.sizes["l"], data.sizes["m"])
        if region:
            boxes = _region_boxes(region, data.sizes["l"], data.sizes["m"])
    if boxes:
        user_indexers["l"] = slice(
            min(item[0] for item in boxes), max(item[2] for item in boxes) + 1, 1
        )
        user_indexers["m"] = slice(
            min(item[1] for item in boxes), max(item[3] for item in boxes) + 1, 1
        )

    partition_indexers = dict(user_indexers)
    for dim, relative_selector in (partition or {}).items():
        if dim not in dims:
            raise ValueError(f"Partition dimension {dim!r} is not present")
        partition_indexers[dim] = _take_indexer(user_indexers[dim], relative_selector)

    # Partitions are relative subsets of the user selection, so their normalized
    # source indexers are already the effective intersection.
    effective_indexers = partition_indexers
    if any(_indexer_length(value) == 0 for value in effective_indexers.values()):
        raise ValueError("The image selection is empty")
    return ImageSelection(
        user_indexers=user_indexers,
        partition_indexers=partition_indexers,
        effective_indexers=effective_indexers,
        boxes=boxes,
        reduction_dims=_reduction_dims(axes, dims),
    )


build_image_slicer = build_image_selection


def _box_mask(selected: xr.DataArray, selection: ImageSelection):
    """Construct a Boolean union mask for multiple selected pixel boxes.

    A single box is already represented exactly by the storage slices and does
    not require a mask. Multiple boxes share a bounding storage rectangle; this
    mask removes pixels in the rectangle that belong to none of the boxes.
    """
    if len(selection.boxes) <= 1:
        return None
    l_indices = _indexer_positions(selection.effective_indexers["l"])
    m_indices = _indexer_positions(selection.effective_indexers["m"])
    li, mi = np.meshgrid(l_indices, m_indices, indexing="ij")
    mask = np.zeros((selected.sizes["l"], selected.sizes["m"]), dtype=bool)
    for l0, m0, l1, m1 in selection.boxes:
        mask |= (li >= l0) & (li <= l1) & (mi >= m0) & (mi <= m1)
    return xr.DataArray(mask, dims=("l", "m"))


def _select_external_mask(
    mask: xr.DataArray, selection: ImageSelection
) -> xr.DataArray:
    """Apply effective indexers to dimensions present in an external mask."""
    applicable = {
        dim: value
        for dim, value in selection.effective_indexers.items()
        if dim in mask.dims
    }
    return mask.isel(applicable)


def _apply_masks(
    data: xr.DataArray,
    dataset: xr.Dataset,
    selection: ImageSelection,
    mask,
    stretch: bool,
    includepix,
    excludepix,
) -> xr.DataArray:
    """Apply region, Boolean-mask, and value-range exclusions in order.

    Parameters
    ----------
    data : xarray.DataArray
        Loaded, selected image pixels.
    dataset : xarray.Dataset
        Loaded variables containing ``data`` and any named mask.
    selection : ImageSelection
        Effective selection and optional pixel boxes.
    mask : str, xarray.DataArray, array-like, or None
        Named or external Boolean mask. ``True`` pixels are retained.
    stretch : bool
        Whether a lower-dimensional mask may broadcast over the image.
    includepix, excludepix : pair of float, optional
        Inclusive value ranges to retain or exclude.

    Returns
    -------
    xarray.DataArray
        Image data with excluded samples represented by NaN.
    """
    spatial_mask = _box_mask(data, selection)
    if spatial_mask is not None:
        data = data.where(spatial_mask)
    if mask is not None and not (isinstance(mask, str) and mask == ""):
        if isinstance(mask, str):
            if mask not in dataset:
                raise KeyError(f"Image has no mask variable {mask!r}")
            mask_array = dataset[mask]
        elif isinstance(mask, xr.DataArray):
            mask_array = _select_external_mask(mask, selection)
        else:
            values = np.asarray(mask, dtype=bool)
            if values.ndim == data.ndim:
                mask_dims = data.dims
            elif (
                stretch
                and values.ndim == 2
                and values.shape
                == (
                    data.sizes["l"],
                    data.sizes["m"],
                )
            ):
                mask_dims = ("l", "m")
            else:
                raise ValueError(
                    "An array mask must match the selected image, or be a "
                    "two-dimensional l/m mask with stretch=True"
                )
            mask_array = xr.DataArray(values, dims=mask_dims)
        if not stretch and (
            mask_array.dims != data.dims or mask_array.sizes != data.sizes
        ):
            raise ValueError(
                "Mask does not match the selected image; use stretch=True to broadcast degenerate dimensions"
            )
        mask_array, _ = xr.broadcast(mask_array.astype(bool), data)
        data = data.where(mask_array)
    if includepix is not None:
        low, high = includepix
        data = data.where((data >= low) & (data <= high))
    if excludepix is not None:
        low, high = excludepix
        data = data.where((data < low) | (data > high))
    return data


def image_statistics(
    image: xr.DataArray | xr.Dataset | str | Path,
    *,
    data_variable: str | None = None,
    axes=-1,
    region="",
    box="",
    chans="",
    stokes="",
    timerange="",
    mask=None,
    stretch: bool = False,
    includepix=None,
    excludepix=None,
    statistics=_DEFAULT_STATISTICS,
    partition: dict[str, slice | np.ndarray] | None = None,
    data_selection: dict | None = None,
    finalize: bool = True,
) -> xr.Dataset:
    """Load one selected image block and calculate local image statistics.

    Parameters
    ----------
    image : xarray.DataArray, xarray.Dataset, str, or pathlib.Path
        An in-memory image or an on-disk XRADIO/Zarr image. On-disk pixels are
        not loaded until all selectors and the optional graph partition have
        been merged.
    data_variable : str, optional
        Image variable to analyze. Required when a Dataset contains more than
        one five-axis image variable.
    axes : int, sequence of int, str, or sequence of str, default -1
        Dimensions reduced by the statistics. ``-1`` reduces all dimensions.
        Named dimensions are preferred; integers follow ``DataArray.dims``.
    region : str, pathlib.Path, or dict, optional
        Pixel-box region. Supported forms are a CASA record containing ``blc``
        and ``trc``, CRTF ``box[[...pix],[...pix]]`` text, or a CRTF file
        containing pixel boxes. Other CRTF shapes are rejected.
    box : str, optional
        CASA inclusive pixel-box syntax ``x0,y0,x1,y1``. Additional groups of
        four integers form a union of disjoint boxes.
    chans, timerange : str, optional
        Zero-based index expressions supporting scalars, inclusive ``a~b``
        ranges, ``a~b^step``, comma/semicolon unions, and ``< <= > >=``.
    stokes : str, optional
        CASA polarization labels, either concatenated or comma-separated.
    mask : str, xarray.DataArray, or array-like, optional
        Named image mask or an aligned/broadcastable Boolean mask.
    stretch : bool, default False
        Allow lower-dimensional or degenerate masks to broadcast.
    includepix, excludepix : pair of float, optional
        Inclusive value range to retain or exclude, respectively.
    statistics : sequence of str
        Any of ``min``, ``minpos``, ``max``, ``maxpos``, ``sum``, ``sumsq``,
        ``mean``, ``median``, ``medabsdevmed`` (or ``mad``), ``rms``,
        ``sigma``, and ``npts``.
    partition : mapping, optional
        Structured application-generated selector relative to the user
        selection. It is intersected with, never substituted for, user input.
    data_selection : mapping, optional
        GraphVIPER-injected equivalent of ``partition``. It may be nested under
        the input name ``"image"``. Do not supply it together with
        ``partition``.
    finalize : bool, default True
        Return public statistics when true, otherwise a mergeable partial state.

    Returns
    -------
    xarray.Dataset
        One variable per requested statistic. Reduced dimensions are absent;
        every other selected dimension, including singleton dimensions, is
        retained with its coordinates.

    Notes
    -----
    Selection precedes masking, and masking precedes reduction. Value filters
    therefore operate only on selected pixels. NaN and masked samples do not
    contribute to ``npts`` or numerical statistics.

    Only the requested image variable and optional named mask are loaded. With
    ``finalize=False``, the result is a mergeable numerical state; derived
    statistics are calculated only after distributed states are merged.

    Examples
    --------
    Sum selected channels while retaining the other dimensions::

        image_statistics(
            image, chans="2~5", axes="frequency", statistics=("sum",)
        )

    Calculate spatial statistics separately for each selected channel::

        image_statistics(
            image,
            chans="2~5",
            axes=("time", "polarization", "l", "m"),
        )
    """
    from astroviper.processing_functions.image_analysis.statistics import (
        create_statistics_state,
        finalize_statistics_state,
    )

    metadata, variable = _open_metadata(image, data_variable)
    data = metadata[variable]
    graph_partition = None
    if data_selection:
        graph_partition = data_selection.get("image", data_selection)
    if partition and graph_partition:
        raise ValueError("partition and graph data_selection cannot both be specified")
    selection = build_image_selection(
        data,
        axes=axes,
        region=region,
        box=box,
        chans=chans,
        stokes=stokes,
        timerange=timerange,
        partition=partition or graph_partition,
    )

    required = _required_variables(metadata, variable, mask)

    if isinstance(image, str | Path):
        from xradio.image import load_image

        lazy_selection = load_image(str(image), block_des=selection.effective_indexers)
        missing = set(required) - set(lazy_selection.data_vars)
        if missing:
            raise KeyError(f"Loaded image is missing variables: {sorted(missing)}")
        selected_dataset = lazy_selection[required].load()
    else:
        selected_dataset = metadata[required].isel(selection.effective_indexers).load()
    selected = selected_dataset[variable]
    selected = _apply_masks(
        selected,
        selected_dataset,
        selection,
        mask,
        stretch,
        includepix,
        excludepix,
    )
    state = create_statistics_state(
        selected,
        selection.reduction_dims,
        statistics=statistics,
        positions=selection.effective_indexers,
    )
    return finalize_statistics_state(state, statistics) if finalize else state


imstatistics = image_statistics
