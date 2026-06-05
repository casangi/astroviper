# AGENTS.md — AstroVIPER

> Guidance for AI coding agents working in the
> **AstroVIPER** repository. Read this fully before editing. The conventions in
> [Coding Conventions](#coding-conventions), [Data Groups](#data-groups-the-core-bookkeeping-concept),
> and [The Python ↔ C++ Memory Contract](#the-python--c-memory-contract) are
> **not optional** — breaking them produces silently wrong science or large
> performance/memory regressions. When in doubt, mirror the patterns already in
> `src/astroviper/processing_functions/imaging/`.

---

## 1. Project Overview

**AstroVIPER** (Astro **V**isibility and **I**mage **P**arallel **E**xecution
**R**eduction) is a radio-astronomy package for **interferometric imaging**
(turning measured visibilities into sky images). It is the *domain/science*
layer of a three-package stack:

| Package | Role | Docs |
| --- | --- | --- |
| **AstroVIPER** (this repo) | Science: gridding, weighting, FFT, deconvolution (CLEAN), imaging workflows | — |
| **XRADIO** | I/O + data model: Processing Sets, MS v4, image datasets, schema, accessors | <https://xradio.readthedocs.io/en/latest/> |
| **GraphVIPER** | Concurrency: builds & runs Dask MapReduce graphs | <https://graphviper.readthedocs.io/en/latest/> |

A hard design rule of the stack: **a clean separation between the concurrency
layer (GraphVIPER) and the domain layer (AstroVIPER science code).** Keep Dask /
graph concerns out of `processing_functions/`, and keep science out of the graph
builders.

- **Language**: Python `>=3.11,<3.14`, with performance-critical kernels in
  **Numba** or **C++20** (via pybind11).
- **Data model**: everything is `xarray` (`DataArray` / `Dataset` /
  `DataTree`), persisted as **Zarr**.
- **Core deps** (`pyproject.toml`): `graphviper`, `xradio[zarr]`, `numpy`,
  `numba`, `opt_einsum`. `toolviper` (logging, Dask client, memory helpers)
  comes in transitively via `graphviper` and is used directly throughout.

### Scope for agents — what to IGNORE
- **Do not work on calibration.** Treat these as out of scope unless the user
  explicitly asks: `src/astroviper/calibration/`,
  `src/astroviper/distributed_graphs/calibration/`,
  `src/astroviper/processing_functions/calibration/`,
  `tests/calibration/`.
- **Pay special attention to imaging** — it is the most developed and most
  important subsystem. See [Imaging Deep-Dive](#6-imaging-deep-dive).

---

## 2. Setup, Build & Test Commands

This package ships **compiled C++ extensions**, so it is **not** pure Python —
the build system is `scikit-build-core` + `pybind11` + CMake.

### Install (developer)
```bash
git clone git@github.com:casangi/astroviper.git
cd astroviper
pip install -e '.[all]'     # builds the C++ extensions; '[all]' adds test/docs/interactive deps
pre-commit install          # installs black + nbstripout git hooks
```
- **macOS**: install casacore via conda *first*, because `pip install
  python-casacore` does not work on macOS:
  ```bash
  conda install -c conda-forge python-casacore
  ```
- **After editing any `.cpp` / `.hpp` / `CMakeLists.txt`** you must rebuild for
  the change to take effect (the C++ is compiled at install time):
  ```bash
  pip install -e '.[all]'   # or: pip install .
  ```
  `scikit-build-core` caches in `build/{wheel_tag}/`; if a build behaves oddly,
  delete `build/` and reinstall.

### C++ extensions (built from `CMakeLists.txt` at repo root)
There are three pybind11 modules; all require **C++20** (for `std::atomic_ref`):
- `processing_functions/imaging/deconvolvers/hogbom` → `_hogbom_ext`
- `processing_functions/imaging/deconvolvers/aspclean` → ASP CLEAN
- `processing_functions/imaging/gridders/prolate_spheroidal_grid_cpp` →
  `_prolate_spheroidal_grid_ext`

### Tests (pytest)
```bash
pytest tests/unit             # unit tests (fast; mirror src/ layout)
pytest tests/stakeholder      # end-to-end science validation (slower, downloads data)
pytest tests                  # everything
pytest --cov=astroviper tests # with coverage
```
- Test layout **mirrors the source tree**: a function in
  `src/astroviper/processing_functions/imaging/foo.py` is tested under
  `tests/unit/processing_functions/imaging/`.
- Shared fixtures live in `tests/utils/conftest.py` (auto-discovered, no import
  needed). A canonical example/template is `tests/utils/__template__.py`.
- Data is fetched with `toolviper.utils.data.download(...)`; MS v2 → MS v4
  conversion uses `xradio...convert_msv2_to_processing_set`.
- **Add or update tests for any code you change**, even if not asked.

### Formatting / pre-commit (required before committing)
- **Black** is the formatter. CI fails on unformatted code.
- `nbstripout` strips notebook outputs (keeps diffs small, repo light).
- If pre-commit rewrites files, **re-stage** them and commit again.

---

## 3. Architecture: the Four Layers

AstroVIPER is organized into four layers. **Calls only ever go downward**
(graphs → node_tasks → processing_functions → utils); never call upward.

```
src/astroviper/
├── distributed_graphs/   (1) Build + compute Dask graphs (GraphVIPER map/reduce).
│                             Nodes are node_tasks. NO science here.
├── node_tasks/           (2) Thin functions: ONE dict arg in, ONE value out.
│                             Do I/O (load_processing_set, write Zarr) + call processing_functions.
├── processing_functions/ (3) The science. Pure Python / Numba / C++(pybind11).
│                             Python owns all large-array memory. NO graph/Dask code.
└── utils/                (4) Helpers: data_group_tools, io, data_partitioning, check_params, …
```

Each of layers 1–3 is further split by **subdomain** (mirrored across the
layers): `imaging` (focus), `image_analysis`, `flagging`,
`visibility_manipulation`, `calibration` (ignored — see scope).

### Layer responsibilities & boundaries
1. **`distributed_graphs/`** — Constructs `parallel_coords`, maps a `node_task`
   over data chunks via `graphviper...map`, reduces results
   (`graphviper...reduce`), then `generate_dask_workflow` + `dask.compute`. This
   is the *only* layer that knows about Dask, chunking, and clusters.
2. **`node_tasks/`** — A node task has the signature `func(input_params: dict,
   graph_mode: bool = True)` and returns a single object (often a small pandas
   `DataFrame` of timings/stats). It loads its data slice (or uses pre-loaded
   data the framework injected), calls into `processing_functions`, and writes
   results to the Zarr store. Keep these thin.
3. **`processing_functions/`** — Stateless science functions operating on
   in-memory `xarray`/NumPy objects. **No I/O of the processing set, no Dask.**
   This is where gridders, weighting, FFT normalization, primary beams, and
   deconvolvers live. Numba/C++ kernels live in sub-packages here.
4. **`utils/`** — Cross-cutting helpers. Most important:
   `utils/data_group_tools.py` (data-group bookkeeping — read it before touching
   data groups) and `utils/io.py` (Zarr variable definitions + chunk writers).

### Public API surface
Each layer/subdomain re-exports its entry points via `__init__.py`. The imaging
entry point is exposed at all three layers as `image_cube_single_field`:
```python
import astroviper.distributed_graphs as distributed_graphs
distributed_graphs.imaging.image_cube_single_field(...)   # user-facing driver
```

---

## 4. Core Domain Concepts (from XRADIO)

You must understand these data structures; they appear in nearly every function.

- **Processing Set** (`ps_xdt`): an `xarray.DataTree` whose child nodes are
  individual **Measurement Sets**. Iterate as `for ms_name, ms_xdt in
  ps_xdt.items(): ...`. All MS v4s in a PS share the same data-variable layout.
- **Measurement Set v4** (`ms_xdt` / `ms_xds`): self-describing correlated data
  for a single observation/SPW/pol-setup. The main dataset holds `VISIBILITY`
  (interferometer) or `SPECTRUM` (single dish), plus `UVW`, `WEIGHT`, `FLAG`,
  with sub-datasets (`antenna_xds`, `field_and_source_*`, `weather_xds`, …) in
  attributes. Dimensions: `time × baseline_id × frequency × polarization`.
- **Image dataset** (`img_xds`): a sky-image cube. Two coordinate spaces:
  - **image domain** dims `(time, frequency, polarization, l, m)`
  - **uv / grid domain** dims `(time, frequency, polarization, u, v)`
  Created via `xradio.image.make_empty_sky_image(...)`.

### Lazy vs. eager (naming tells you which)
- `open_*` → **lazy**: loads only metadata; data variables are Dask arrays.
  (`open_processing_set`)
- `load_*` → **eager**: loads everything into memory now.
  (`load_processing_set`)

### XRADIO accessors (used heavily — prefer these over hand-rolled logic)
- Processing set: `ps_xdt.xr_ps.summary()`, `.get_freq_axis()`,
  `.get_combined_field_and_source_xds()`, `.get_combined_antenna_xds()`,
  `.sizes`.
- Measurement set: `ms_xds.xr_ms.sel(data_group_name='imaging')`,
  `.get_field_and_source_xds()`, `.get_partition_info()`.
- Image: `img_xds.xr_img.add_data_group(...)`, `.delete_data_variables(...)`,
  `.get_lm_cell_size()`.

---

## 5. Data Groups: the core bookkeeping concept

> Authoritative refs: XRADIO §Data Groups
> <https://xradio.readthedocs.io/en/latest/measurement_set/overview.html#data-groups>
> and `src/astroviper/utils/data_group_tools.py`.

A **data group** lets one dataset hold **multiple versions of the same logical
variable** (e.g. raw vs. corrected vs. residual visibilities) without
overwriting. It is a dict stored at `xds.attrs["data_groups"]`:

```python
ms_xds.attrs["data_groups"] = {
  "base":    {"correlated_data": "VISIBILITY",           "flag": "FLAG", "weight": "WEIGHT",         "uvw": "UVW"},
  "imaging": {"correlated_data": "VISIBILITY_CORRECTED", "flag": "FLAG", "weight": "WEIGHT_IMAGING", "uvw": "UVW"},
}
```
- **Keys = logical roles** (lowercase): `correlated_data`, `flag`, `weight`,
  `uvw` (visibility side); `sky`, `visibility`, `point_spread_function`,
  `primary_beam`, `mask`, `uv_sampling`, … (image side). Plus metadata keys
  `date` and `description`.
- **Values = data-variable names** (UPPERCASE). Groups may share variables
  (above, `base` and `imaging` share `FLAG`/`UVW`).
- A new version of a variable is the standard name + `_` + a descriptor, e.g.
  `VISIBILITY` → `VISIBILITY_PHASE_SHIFTED`, `VISIBILITY_MODEL`,
  `VISIBILITY_RESIDUAL`; `SKY` → `SKY_RESIDUAL`, `SKY_MODEL`.

### Always use the tooling in `utils/data_group_tools.py`
Do **not** mutate `attrs["data_groups"]` by hand. The standard two-step pattern
used everywhere in imaging:

```python
from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out, modify_data_groups_xds,
)

# 1) Resolve in/out groups (validates input exists; guards against accidental
#    overwrite unless overwrite=True). out = {**in, **out_modified}.
data_group_in, data_group_out = create_data_groups_in_and_out(
    img_xds,
    data_group_in_name="residual",
    data_group_out_name="model",
    data_group_out_modified={"sky": "SKY_MODEL"},
    overwrite=False,
)

# 2) ... compute and write the new data variable(s) into the dataset ...

# 3) Register the new group (in place) + stamp date/description audit trail.
modify_data_groups_xds(
    img_xds,
    data_group_out_name="model",
    data_group_out=data_group_out,
    description="Deconvolved model image.",
)
```
- For **processing-set-wide** changes use the `*_ps_xdt` variants
  (`create_ps_xdt_data_groups_in_and_out`, `modify_data_groups_ps_xdt`) — they
  iterate all MSes and **assert the layout is identical across MSes** (ignoring
  `date`/`description`).
- `date`/`description` are an append-only audit trail (`"; "`-joined). Write a
  meaningful `description` for every new group.
- Conventional group names in the imaging loop: `base` → `corrected` →
  (`residual`, `model`).

---

## 6. Imaging Deep-Dive

This is the subsystem to know best. The cube single-field imager is implemented
across all three layers, each named `image_cube_single_field`:

### 6.1 Driver — `distributed_graphs/imaging/image_cube_single_field.py`
The user-facing function. Sequence:
1. `make_empty_sky_image(...)` → `write_image(..., out_format="zarr")` (creates
   the image store with correct coords/dims).
2. `calculate_number_of_chunks_for_cube_imaging(...)` → decide frequency chunk
   count from per-channel memory estimate + available threads.
3. `make_parallel_coord(coord=img_xds.frequency, n_chunks=...)` → defines
   parallelism (imaging is **parallelized over frequency** for cubes).
4. `create_empty_data_variables_on_disk(...)` → pre-allocate Zarr arrays
   (NaN-filled) for the variables in `image_data_variables_keep`, so map tasks
   can **lazily write their own slice**.
5. `open_processing_set(...)` (lazy) →
   `interpolate_data_coords_onto_parallel_coords(...)` → `node_task_data_mapping`.
6. `map(input_data=ps_xdt, node_task=node_tasks.imaging.image_cube_single_field,
   data_loading_task=_load_processing_set_chunk, disk_chunk_sizes=..., ...)` →
   `reduce(..., mode="tree")` → `generate_dask_workflow` → `dask.compute` →
   `zarr.consolidate_metadata`.

Notable parameters (see the function's NumPy docstring for the full list):
`image_params`, `imaging_weights_params`, `iteration_control_params`, `gridder`
(`"prolate_spheroidal"`), `deconvolver` (`"hogbom"` / `"asp"`),
`processing_set_data_group_name`, `double_precision`,
`processing_function_threads`, `n_chunks`, `disk_chunk_sizes`,
`fft_backend`, `memory_mode`.

- **`disk_chunk_sizes`** (e.g. `{"frequency": 200}` or `"Auto"`) adds a
  *data-loading layer*: one load node per on-disk chunk reads the native chunk
  once; map tasks then sub-select their slice from the pre-loaded data
  (`input_params["input_data"]`), avoiding redundant disk reads.
- **`memory_mode`**: only `"in_memory"` is currently implemented (there are
  `assert memory_mode == "in_memory"` guards; `"in_place"` / `"cache"` are
  stubs). Do not assume the others work.

### 6.2 Node task — `node_tasks/imaging/image_cube_single_field.py`
Signature `image_cube_single_field(input_params, graph_mode=True)`:
1. `memory_setup(131072)` **first** (pins the malloc mmap threshold so big
   allocations are released to the OS on free — must run before any large
   allocation).
2. Build the empty per-chunk `img_xds`.
3. Get data: use `input_params["input_data"]` if the loading layer pre-loaded it,
   else `load_processing_set(...)` (eager) for this chunk's `data_selection`.
4. Call `pf.imaging.image_cube_single_field(input_params, ps_xdt, img_xds)`.
5. Write the result slice to Zarr via
   `astroviper.utils.io.write_result_chunk_to_disk_using_zarr(...)`.
6. `free_memory()` and return a small timing `DataFrame`.

### 6.3 Science — `processing_functions/imaging/image_cube_single_field.py`
Runs the **major/minor cycle** CLEAN loop via `IterationController`:
- `residual_cycle_cube_single_field(...)` — degrid model → compute residual
  visibilities → grid → FFT-normalize → form residual image (+ PSF on the first
  iteration), primary beam, and imaging weights (first iteration only).
- `model_update_cycle_cube_single_field(...)` → `deconvolve(...)` (the minor
  cycle: Hogbom or ASP CLEAN in C++) updates the model image, with a mask from
  `make_mask`.
- Accumulate per-plane stats into a `ReturnDict`; check convergence; iterate.
- A final residual cycle produces the last residual image.

### 6.4 Key processing functions & gridders
- Weighting: `calculate_imaging_weights.py` (`"natural"`, `"briggs"`/robust).
- Gridding (vis → uv grid): `add_visibility_grid.py`
  (`add_visibility_grid_single_field`), sampling/PSF grid:
  `add_uv_sampling_grid.py`.
- Degridding (model uv grid → vis): `get_visibility_grid.py`.
- FFT + normalization: `fft_normalize_prolate_spheriodal_gridder.py`.
- Primary beam: `make_pb_symmetric.py` (airy disk).
- Polarization: `image_analysis/transform_polarization_basis.py`
  (stokes ↔ linear).
- PSF fit: `image_analysis/point_spread_function_gaussian_fit.py`.
- Deconvolvers: `deconvolution.py` dispatch → `deconvolvers/hogbom` (C++),
  `deconvolvers/aspclean` (C++).
- The standard gridder kernel exists in **two interchangeable** forms (the
  Python wrapper picks via a `cpp_gridder` flag): Numba
  (`gridders/prolate_spheroidal_grid.py::prolate_spheroidal_grid_jit`) and C++
  (`gridders/prolate_spheroidal_grid_cpp`). **Keep both in sync** if you change
  the gridding math.

### 6.5 Imaging I/O conventions — `utils/io.py`
- `imaging_data_variables_and_dims_double_precision` /
  `..._single_precision` map each logical role → `{"dims", "dtype", "name"}`.
  This is the single source of truth for image variable names, dims, and dtypes.
- Standard dim tuples: `full_dims_lm = ["time","frequency","polarization","l","m"]`,
  `full_dims_uv = [...,"u","v"]`, `norm_dims = ["time","frequency","polarization"]`.
- `create_empty_data_variables_on_disk(...)` writes NaN-filled (or
  fill-as-appropriate) Zarr arrays chunked along the parallel dim; handles Zarr
  v2 and v3 (`_to_zarr_v3_codec`).
- `write_result_chunk_to_disk_using_zarr(...)` writes each kept variable's slice
  using the `task_coords` slices.

---

## 7. The Python ↔ C++ Memory Contract

> **This is the most important performance/correctness rule in the repo.** It
> applies to all pybind11 kernels (gridder, degridder, Hogbom, ASP).

**Python always owns and controls large-array memory. C++ is given pointers to
read/write Python-owned NumPy buffers in place. No unnecessary copies are ever
made.** Concretely, the bindings (`.../python/bindings.cpp`) follow these rules —
match them in any new kernel:

1. **Large arrays: typed without `forcecast`.** Declared as
   `py::array_t<T, py::array::c_style>` (e.g. `complex128`, `float64`,
   `int64`). With no `forcecast`, a wrong dtype/layout raises immediately
   instead of silently allocating a converted copy. **Never add `forcecast` to a
   large array.**
2. **Tiny arrays may keep `forcecast`.** Only genuinely small ones (`n_uv`,
   `delta_lm` — 2 elements) use `c_style | forcecast` so callers aren't forced
   to match an exact int width.
3. **Explicit checks before compute**: verify `ndim`/shape, require
   C-contiguity (`require_c_contiguous(...)` — tells the caller to use
   `np.ascontiguousarray()` if needed), and for outputs require writeability
   (raise if `info.readonly` / `!arr.writeable()` — message: "modified in
   place").
4. **In-place output.** Accumulators (`grid`, `normalization`) and CLEAN buffers
   (residual/model cubes) are written through `mutable_data()` pointers; they
   are *not* returned. Repeated calls accumulate (e.g. summing over MSes).
5. **Release the GIL during compute**: capture raw pointers, then
   `py::gil_scoped_release release;` around the heavy loop. The NumPy buffers
   stay valid because Python owns them.
6. **No implicit complex narrowing.** The degridder takes an untyped `py::array`
   precisely so pybind11 cannot safe-cast a `complex64` buffer into a
   `complex128` temporary (which would silently drop the caller's writes); it
   inspects the dtype and writes `complex64` *or* `complex128` in place.
7. **Threading inside the kernel** is via a `num_threads` argument →
   `std::thread` workers; `num_threads <= 0` falls back to
   `std::thread::hardware_concurrency()`, `1` runs serial. Concurrent writes to
   shared grid cells use lock-free `std::atomic_ref<double>` adds (C++20) —
   `complex<double>` is added as two `double` lanes.

### Numba equivalent
Numba kernels follow the same memory philosophy: decorate with
`@jit(nopython=True, cache=True, nogil=True)` and **modify arrays in place**
(`grid`, `normalization`, …). `nogil=True` lets them run under Dask threads.

### Two levels of parallelism (don't conflate them)
- **Across chunks**: the Dask graph (GraphVIPER) — one task per frequency chunk.
- **Within a task**: `processing_function_threads` → passed as `num_threads` to
  the C++/Numba kernels.

---

## 8. Parallelism Model (GraphVIPER)

> Ref: <https://graphviper.readthedocs.io/en/latest/graph_building_tutorial.html>

GraphVIPER is a Dask MapReduce layer. The flow used by AstroVIPER:

1. **`parallel_coords`** — dict keyed by the dimension(s) to parallelize over
   (for cube imaging: `frequency`). Build with
   `make_parallel_coord(coord=..., n_chunks=...)`. Chunks may overlap.
2. **`node_task_data_mapping`** — from
   `interpolate_data_coords_onto_parallel_coords(parallel_coords, ps_xdt)`.
   Maps each graph node → `{chunk_indices, parallel_dims, data_selection,
   task_coords}`.
3. **`map(input_data, node_task_data_mapping, node_task, input_params, ...)`** —
   builds the map stage. A `node_task` must take **a single dict** and return
   **a single value**.
4. **`reduce(viper_graph, reduce_fn, input_params, mode="tree"|"single_node")`** —
   `reduce_fn(input_data, input_params)`.
5. `generate_dask_workflow(viper_graph)` → `dask.compute(...)`.

**The framework injects these keys into every `node_task`'s `input_params`**
(do not set them yourself; do read them): `chunk_indices`, `parallel_dims`,
`data_selection`, `task_coords`, `task_id`, `input_data` (pre-loaded data if a
`data_loading_task`/`disk_chunk_sizes` was used, else `None`), `date_time`
(+ `viper_local_dir` when local caching is enabled). AstroVIPER adds the rest
(`image_params`, `iteration_control_params`, `image_store`, `double_precision`,
`deconvolver`, `fft_backend`, etc.).

### Why this design (do not regress it)
GraphVIPER deliberately **loads data inside compute nodes** rather than as
separate Dask array nodes. This (a) fixes Dask "memory backpressure" for large
in-memory image cubes Dask can't see, and (b) keeps the graph small (Xarray-
backed Dask datasets create a node per data variable — fatal for MS-style data
with many variables). Keep I/O inside node tasks; don't expand the graph with
per-variable nodes.

---

## 9. Coding Conventions

> AstroVIPER follows XRADIO's conventions:
> <https://xradio.readthedocs.io/en/latest/development.html#coding-conventions>

- **Naming — the critical rule** (mismatches break schema checks and data-group
  lookups):
  - **Coordinates**: lowercase `snake_case`, eagerly loaded — e.g. `frequency`,
    `time`, `polarization`, `antenna_name`, `baseline_id`, `l`, `m`, `u`, `v`.
  - **Data variables**: UPPERCASE `SNAKE_CASE`, lazily loaded — e.g.
    `VISIBILITY`, `WEIGHT`, `UVW`, `FLAG`, `SKY_RESIDUAL`,
    `POINT_SPREAD_FUNCTION`, `PRIMARY_BEAM`, `WEIGHT_IMAGING`.
  - **New versions**: `STANDARD_NAME` + `_` + descriptor (`VISIBILITY_MODEL`,
    `SKY_MODEL`).
  - **Functions / variables**: `snake_case`, **descriptive** (`image_size`, not
    `imsize`).
  - **Classes**: `CamelCase` (`IterationController`, `ReturnDict`).
- **Formatting**: **Black** (enforced by CI + pre-commit). Don't hand-format.
- **Imports**: prefer **absolute** imports (`from
  astroviper.processing_functions.imaging.residual_cycle import ...`). Relative
  imports appear only as short re-exports in `__init__.py` files. Heavy/optional
  deps (dask, zarr, matplotlib, the C++ ext, even numpy in some hot node-task
  paths) are frequently imported **inside functions** to keep worker import time
  and graph-serialization cost low — follow the local pattern in the file you're
  editing.
- **Docstrings**: **NumPy-style** for all public functions/classes (see
  `data_group_tools.py` and `add_visibility_grid.py` for exemplars — Parameters,
  Returns, Raises, See Also).
- **Logging**: use the **toolviper** logger, not `print`:
  ```python
  import toolviper.utils.logger as logger
  logger.info(...); logger.debug(...)
  ```
  (Some legacy `print(...)` calls exist in hot loops — don't add new ones.)
- **Performance**: vectorize with NumPy; if not feasible, use **Numba**
  (`@jit(nopython=True, cache=True, nogil=True)`) or C++; verify with timing.
  Large arrays are timed via `T_*` keys collected into per-chunk `DataFrame`s /
  `ReturnDict`s — keep that bookkeeping when adding stages.
- **Parameter validation**: reuse `astroviper.utils.check_params.check_params`
  (type/range/allowed-value checks with defaults) rather than ad-hoc validation.

---

## 10. Do / Don't (gotchas)

**Do**
- Mirror existing patterns in `processing_functions/imaging/` for any new
  science function (data-group in/out resolution → compute in place → register
  group → return stats/timings).
- Keep the gridder's Numba and C++ implementations consistent.
- Pass `num_threads` through to kernels and respect `processing_function_threads`.
- Keep the layering: graph code in `distributed_graphs/`, I/O in `node_tasks/`,
  science in `processing_functions/`.
- Run `black`, `pytest`, and rebuild after C++ edits before finishing.

**Don't**
- ❌ Add `forcecast` to (or otherwise allow silent copies / dtype conversions of)
  large arrays crossing the Python↔C++ boundary.
- ❌ Mutate `xds.attrs["data_groups"]` directly — use `data_group_tools`.
- ❌ Lowercase a data-variable name or uppercase a coordinate name.
- ❌ Put Dask/graph logic into `processing_functions/`, or science into
  `distributed_graphs/`.
- ❌ Assume `memory_mode` other than `"in_memory"` works.
- ❌ Touch calibration code (see [scope](#scope-for-agents--what-to-ignore)).
- ❌ Commit notebook outputs (pre-commit's `nbstripout` enforces this) or
  unformatted code.
- ❌ Introduce relative imports outside `__init__.py` re-exports.

---

## 11. Where to make a change (task → location)

| Task | Where |
| --- | --- |
| New/changed imaging **workflow / parallelism** | `distributed_graphs/imaging/` |
| New **node task** (I/O + orchestration of a chunk) | `node_tasks/imaging/` |
| New **science kernel** (gridding, weighting, deconvolution, FFT, PB) | `processing_functions/imaging/` |
| New **C++ kernel** | a sub-package under `processing_functions/.../{gridders,deconvolvers}/` with `include/`, `src/`, `python/bindings.cpp`, `CMakeLists.txt`; register it in the root `CMakeLists.txt` |
| Data-group helpers | `utils/data_group_tools.py` |
| Zarr variable defs / chunk writers | `utils/io.py` |
| Chunk-count / thread heuristics | `utils/data_partitioning.py` |
| Parameter validation helpers | `utils/check_params.py` |
| Tests | `tests/unit/<mirror of src path>/`; end-to-end in `tests/stakeholder/` |

---

## 12. Reference Links
- AstroVIPER repo: <https://github.com/casangi/astroviper>
- XRADIO docs: <https://xradio.readthedocs.io/en/latest/>
  - Data groups: <https://xradio.readthedocs.io/en/latest/measurement_set/overview.html#data-groups>
  - Coding conventions: <https://xradio.readthedocs.io/en/latest/development.html#coding-conventions>
  - Lazy vs eager (`open_`/`load_`): <https://xradio.readthedocs.io/en/latest/development.html#lazy-and-eager-functions>
- GraphVIPER docs: <https://graphviper.readthedocs.io/en/latest/>
  - Graph building tutorial (map/reduce, `parallel_coords`): <https://graphviper.readthedocs.io/en/latest/graph_building_tutorial.html>
- pybind11: <https://pybind11.readthedocs.io/> · scikit-build-core: <https://scikit-build-core.readthedocs.io/> · Numba: <https://numba.readthedocs.io/>
