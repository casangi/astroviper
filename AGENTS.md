# AGENTS.md — AstroVIPER

> A guide for developers and AI agents for developing code in AstroVIPER.

---

## 1. Project Overview

**AstroVIPER** (Astro **V**isibility and **I**mage **P**arallel **E**xecution
**R**eduction) is part of the VIPER ecosystem that does radio astronomy data processing.

| Package | Role | Repo |
| --- | --- | --- |
| **ToolVIPER** | Logging, parameter checking, and Dask cluster tools | https://github.com/casangi/toolviper |
| **XRADIO** | I/O + data model: Processing Sets (MS v4), image datasets, calibration datasets, schema, accessors | https://github.com/casangi/xradio |
| **GraphVIPER** | Concurrency: builds & runs Dask MapReduce graphs | https://github.com/casangi/graphviper |
| **AstroVIPER** (this repo) | Science: flagging, calibration, imaging, image analysis, visibility manipulation, simulation | https://github.com/casangi/astroviper |
| **FlowVIPER** | Data-processing workflows built from AstroVIPER's distributed applications | https://github.com/casangi/flowviper |

Note that all the packages are still under development and functionality is not complete.

- **Language**: Python (supported versions: `requires-python` in
  [`pyproject.toml`](pyproject.toml)), with performance-critical kernels in C++
  (standard & compiler floor: [`cmake/AstroviperPybind.cmake`](cmake/AstroviperPybind.cmake)).
- **Data model**: everything is `xarray` (`DataArray` / `Dataset` /`DataTree`).

## 2. Setup, Build & Test Commands

This package ships **compiled C++ extensions**, so it is **not** pure Python —
the build system is `scikit-build-core` + `pybind11` + CMake.

### Install (developer)
See [README.md](README.md) for instructions.

### C++ build configuration (documented at the source — not duplicated here)
- **Compiler flags, C++ standard, toolchain floor, per-module build**: all
  shared compiler settings for every kernel live in one file,
  [`cmake/AstroviperPybind.cmake`](cmake/AstroviperPybind.cmake), and its
  comments are the authoritative documentation (the `astroviper_cpp_flags`
  target, the `astroviper_add_pybind_module(...)` helper, the standalone
  single-kernel build, and the **`-ffp-contract=off`** reproducibility flag —
  see [the memory contract](#6-the-python--c-memory-contract); do not remove
  it). To change a flag for every kernel, edit that file — and only that file.
- **Build type (Release / Debug / …)**: set solely via `build-type` in
  `[tool.scikit-build.cmake]` of [`pyproject.toml`](pyproject.toml). The
  comment block above that key is the authoritative documentation: the flags
  each build type implies, how to override for a single build without editing
  the file, and the stale-cache fix. Optimization level / `-fPIC` are left to
  CMake's build type; don't hand-roll `-O3` in cmake.

### Tests (pytest)
```bash
pytest tests/unit             # unit tests (fast; mirror src/ layout)
pytest tests/component        # end-to-end science validation (slower, downloads data)
pytest tests                  # everything
pytest --cov=astroviper tests # with coverage
```
- Test layout **mirrors the source tree**: a function in
  `src/astroviper/processing_functions/imaging/foo.py` is tested under
  `tests/unit/processing_functions/imaging/`.
- Shared fixtures live in `tests/utils/conftest.py` (auto-discovered, no import
  needed). A canonical example/template is `tests/utils/__template__.py`.
- Data is fetched with `toolviper.utils.data.download(...)`.
- **Add or update tests for any code you change**, even if not asked.

### Formatting / pre-commit (required before committing)
- **Ruff** is the formatter (`ruff format`, black-compatible) and the linter
  (`ruff check --fix`): import sorting plus pyflakes (undefined names, unused
  imports/variables), pycodestyle errors, bugbear, and pyupgrade rules — the
  authoritative rule list lives in `[tool.ruff.lint]` of
  [`pyproject.toml`](pyproject.toml). CI fails on unformatted code and lint
  errors. `pyupgrade` (py311+) and `absolufy-imports` also run.
- `nbstripout` strips notebook outputs (keeps diffs small, repo light).
- If pre-commit rewrites files, **re-stage** them and commit again.

### Documentation notebooks (`docs/`)
Tutorials live under `docs/`, organised to **mirror the source module tree** —
put a new notebook in the folder matching the layer + subdomain of the function
it demonstrates:
- `distributed_applications_tutorials/{imaging,image_analysis,model,simulation}/`
- `processing_functions_tutorials/{imaging,image_analysis,visibility_manipulation,simulation}/`

Notebooks must stay **output-stripped** (`nbstripout`) and
**headless-executable** — they are run non-interactively (`nbconvert` /
`nbclient`, and the CI "run ipynb" job). To keep them from hanging a headless run:
- **Never activate an interactive Matplotlib backend** — no `%matplotlib widget`,
  `%matplotlib notebook`, or `ipympl`. Those open a Jupyter *comm* to a browser
  frontend that never answers under `nbclient`, so the cell blocks until the
  execution timeout — a hard hang in CI. Use **`%matplotlib inline`**. (A
  *commented* `# %matplotlib widget` hint for interactive users is fine.)
- **ipywidgets is OK headless.** `interact(...)`, `interactive(...)` and
  `display(widget)` all execute fine under `nbclient` (the widget just renders no
  live output) — keep them for human readers. Don't design a cell so its result
  *depends on* live widget interaction; compute the result eagerly and let the
  widget only re-render it. When a cell is purely an interactive explorer, a
  static fallback (e.g. a small multi-panel montage) is the most robust choice for
  the executed copy — see
  `processing_functions_tutorials/imaging/demo_standard_grid.ipynb`.
- Always execute notebooks with a **per-cell timeout** so a genuinely stuck cell
  fails loudly instead of hanging the whole run.
- **Heavy notebooks are slow, not hung.** Run the notebook suite **sequentially**
  (or with a small concurrency cap): launching many gridding/imaging notebooks at
  once oversubscribes CPU/RAM and can make an ordinary cell look like a deadlock.

---

## 3. Architecture: the Four Layers

AstroVIPER is organized into four layers. **Calls only ever go downward or horizontally**
(distributed_applications → node_tasks → processing_functions → utils); never call upward. Note that
a layer can call any layer lower than itself for example distributed_applications can call processing_functions
and a layer can call horizontally for example a processing_function can call another processing_function.

```
src/astroviper/
├── distributed_applications/
│                         (1) Build + compute Dask graphs (GraphVIPER map/reduce).
│                             Nodes are node_tasks. NO science here. Several graphs and compute calls can be done.
├── node_tasks/           (2) The functions that run as graph nodes.
│                             Do I/O (load_processing_set, write Zarr) + call processing_functions.
├── processing_functions/ (3) The science. Pure Python / C++(pybind11).
│                             Python owns all large-array memory. NO graph/Dask code. Xarray can still be used without Dask arrays and native lazy Xarray's arrays are also allowed
└── utils/                (4) Helpers: data_group_tools, io, data_partitioning, check_params, …
```

Each of layers 1–3 is further split by **subdomain**, mirrored across the
layers where implemented (not every subdomain exists in every layer yet):
`imaging`, `image_analysis`, `flagging`, `visibility_manipulation`,
`calibration`, `model`, `simulation`. Additional **subdomains** can be added in
the future.

### Layer responsibilities & boundaries
1. **`distributed_applications/`** — Constructs `parallel_coords`, maps a `node_task`
   over data chunks via `graphviper...map`, reduces results
   (`graphviper...reduce`), then `generate_dask_workflow` + `dask.compute`. This
   is the *only* layer that knows about Dask, chunking, and clusters. It also
   creates the (empty) on-disk output structures that node tasks write into.
2. **`node_tasks/`** — The functions that get delayed in `distributed_applications/`.
   They may load chunks of input data, write chunks of output data, and call
   `processing_functions`. The structures they write into are created by
   `distributed_applications/`.
3. **`processing_functions/`** — Stateless science functions operating on
   in-memory `xarray`/NumPy objects. **No I/O on the input or output data.**
   This is where gridders, weighting, FFT normalization, primary beams, and
   deconvolvers live. C++ kernels live in sub-packages here.
   > **Open question:** should temporary/cache data (e.g. a cfcache) be allowed
   > to be written to disk from this layer? Undecided — raise it before relying
   > on either answer.
4. **`utils/`** — Cross-cutting helpers. Most important:
   `utils/data_group_tools.py` (data-group bookkeeping — read it before touching
   data groups) and `utils/io.py` (Zarr variable definitions + chunk writers).
   The io.py will eventually move to XRADIO.

### Public API surface
Each layer/subdomain re-exports its entry points via `__init__.py`. The imaging
entry point is exposed at all three layers as `image_cube_single_field`, the
simulation entry point as `simulate_processing_set`:
```python
import astroviper.distributed_applications as distributed_applications
distributed_applications.imaging.image_cube_single_field(...)
distributed_applications.simulation.simulate_processing_set(...)
```

### The `simulation` subdomain (port of SIRIUS)
`simulate_processing_set` is a **pure generator**: there is no input processing
set, so the distributed application maps the node task over
`parallel_coords = {"time", "frequency"}` with `map(input_data={}, ...)` and
each task writes its `(time, frequency)` block of `VISIBILITY/UVW/WEIGHT/FLAG`
into an MSv4 processing set that the driver created beforehand
(`utils/measurement_set_tools.py`: empty MSv4 skeleton + region writes;
`utils/telescope_layout.py`: CASA `.cfg` layouts → `antenna_xds`;
`utils/beam_models.py`: shipped Airy / beam-polynomial / Zernike models). The
science lives in `processing_functions/simulation/` (`calculate_uvw`,
`calculate_parallactic_angles`, `antenna_beams`, `calculate_visibilities` with
a NumPy reference and the C++ `visibility_kernel_cpp`, `calculate_noise`).
Conventions: `uvw = antenna2 - antenna1` (MSv4); beam-image datasets are
`JONES[parallactic_angle, frequency, polarization, l, m]`; polarizations are MSv4
strings (`"RR"`, `"XX"`, ...). Data shipped in `src/astroviper/data/simulation/`.
Legacy SIRIUS reference fixtures live in
`tests/unit/processing_functions/simulation/data/` (see
`generate_legacy_fixtures.py` there).

---

## 4. Core Domain Concepts (from XRADIO)

You must understand these data structures; they appear in nearly every function.

- **Processing Set** (`ps_xdt`): an `xarray.DataTree` whose child nodes are
  individual **Measurement Sets**. Iterate as `for ms_name, ms_xdt in
  ps_xdt.items(): ...`.
- **Measurement Set v4** (`ms_xdt` / `ms_xds`): self-describing correlated data
  for a single observation/SPW/pol-setup. The main dataset holds `VISIBILITY`
  (interferometer) or `SPECTRUM` (single dish), plus `UVW`, `WEIGHT`, `FLAG`,
  with sub-datasets (`antenna_xds`, `field_and_source_*`, `weather_xds`, …) in
  attributes. Dimensions: `time × baseline_id × frequency × polarization` for Visibility correlated datasets (interferometers) and `time × antenna_name × frequency × polarization` for Spectrum correlated datasets (single dish).
- **Image dataset** (`img_xds`): Two coordinate spaces:
  - **image domain** dims `(time, frequency, polarization, l, m)`
  - **uv / grid domain** dims `(time, frequency, polarization, u, v)`
  Created via `xradio.image.make_empty_sky_image(...)`.

### Lazy vs. eager (naming tells you which)
- `open_*` → **lazy**: loads only metadata; data variables are Dask arrays.
  (`open_processing_set`). Used in the distributed_applications layer.
- `load_*` → **eager**: loads everything into memory now.
  (`load_processing_set`). Used in node_tasks layer.

### XRADIO accessors (used heavily — prefer these over hand-rolled logic)
- Processing set: `ps_xdt.xr_ps.summary()`, `.get_freq_axis()`,
  `.get_combined_field_and_source_xds()`, `.get_combined_antenna_xds()`,
  `.sizes`: https://xradio.readthedocs.io/en/latest/measurement_set/api.html#processingsetxdt-api
- Measurement set: `ms_xds.xr_ms.sel(data_group_name='imaging')`,
  `.get_field_and_source_xds()`, `.get_partition_info()`: https://xradio.readthedocs.io/en/latest/measurement_set/api.html#measurementsetxdt-api
- Image: `img_xds.xr_img.add_data_group(...)`, `.delete_data_variables(...)`,
  `.get_lm_cell_size()`

---

## 5. Data Groups: the core bookkeeping concept

> Authoritative refs: XRADIO §Data Groups
> <https://xradio.readthedocs.io/en/latest/measurement_set/overview.html#data-groups>
> and `src/astroviper/utils/data_group_tools.py`.

A **data group** lets one dataset hold **multiple versions of the same logical
variable** (e.g. raw vs. corrected vs. residual visibilities) without
overwriting. It is a dict stored at `xds.attrs["data_groups"]` and here is an MSv4 example:

```python
ms_xds.attrs["data_groups"] = {
  "base":    {"correlated_data": "VISIBILITY",           "flag": "FLAG", "weight": "WEIGHT",         "uvw": "UVW"},
  "imaging": {"correlated_data": "VISIBILITY_CORRECTED", "flag": "FLAG", "weight": "WEIGHT_IMAGING", "uvw": "UVW"},
  "model": {"correlated_data": "VISIBILITY_MODEL", "uvw": "UVW"},
  "calibrated": {"correlated_data": "VISIBILITY_CORRECTED", "flag": "FLAG_CORRECTED", "weight": "WEIGHT_CORRECTED", "uvw": "UVW"}
}
```

Here is an image example (image-side roles):
```python
img_xds.attrs["data_groups"] = {
  "residual": {"sky": "SKY_RESIDUAL", "point_spread_function": "POINT_SPREAD_FUNCTION",
               "primary_beam": "PRIMARY_BEAM", "mask": "MASK", "beam_fit_params_point_spread_function":"BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"},
  "model":    {"sky": "SKY_MODEL", "mask": "MASK"},
  "restored": {"sky": "SKY_RESTORED","point_spread_function": "POINT_SPREAD_FUNCTION",
               "primary_beam": "PRIMARY_BEAM", "mask": "MASK",
               "beam_fit_params_point_spread_function":"BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"},
}
```

- **Keys = logical roles examples** (lowercase): `correlated_data`, `flag`, `weight`,
  `uvw`; `sky`, `visibility`, `point_spread_function`,
  `primary_beam`, `mask`, `uv_sampling`, … (image side). Plus metadata keys
  `date` and `description`.
- **Values = data-variable names** (UPPERCASE). Groups may share variables
  (above, `base` and `imaging` share `FLAG`/`UVW`).
- A new version of a variable is the standard name + `_` + a descriptor, e.g.
  `VISIBILITY` → `VISIBILITY_PHASE_SHIFTED`, `VISIBILITY_MODEL`,
  `VISIBILITY_RESIDUAL`; `SKY` → `SKY_RESIDUAL`, `SKY_MODEL`.

Not all keys need to be present for a given data group, and additional keys can be added as needed, but must be added to the schema in XRADIO. For example, the MSv4 data group model  does not have the key weight.

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

### Data-group parameter naming (imaging processing functions)
Every imaging processing function names its data-group parameters from a fixed
vocabulary so call sites read consistently:

| Parameter | Side | Role |
| --- | --- | --- |
| `ms_data_group_in_name` / `ms_data_group_out_name` | measurement set | input / output group name |
| `image_data_group_in_name` / `image_data_group_out_name` | image dataset | input / output group name |
| `ms_data_group_out_modified` / `image_data_group_out_modified` | — | dict of role → new data-variable name for the output group |
| `processing_set_data_group_name` | processing set | the (input) group name applied across **every** MS in a processing set — used by the node tasks that operate on a whole `ps_xdt` |

- Use `image_data_group_*`; use the explicit `_in_`/`_out_` split
  (not a single bare `..._data_group_name`).
- **Not all four are needed** — a function uses only the ones it touches (e.g. a
  gridder reads an MS and writes an image, so it has `ms_data_group_in_name` +
  `image_data_group_in_name`/`image_data_group_out_name`).
- When a function genuinely consumes/produces **more than one group of the same
  side+direction**, append the role to the canonical name rather than inventing
  a new prefix. Example —
  `calculate_residual_visibilities` forms `residual = observed − model` from two
  MS inputs and one MS output: `ms_data_group_in_observed`,
  `ms_data_group_in_model`, `ms_data_group_out_residual`.

---

## 6. The Python ↔ C++ Memory Contract

> **This is the most important performance/correctness rule in the repo.** It
> applies to all pybind11 kernels (gridder, degridder, imaging weights, Hogbom,
> ASP).

**Python always owns and controls large-array memory. C++ is given pointers to
read/write Python-owned NumPy buffers in place. No unnecessary copies are ever
made.** Concretely, the bindings (`.../python/bindings.cpp`) follow these rules —
match them in any new kernel:

1. **Arrays: typed without `forcecast`.** Declared as
   `py::array_t<T, py::array::c_style>` (e.g. `complex128`, `float64`,
   `int64`). With no `forcecast`, a wrong dtype/layout raises immediately
   instead of silently allocating a converted copy. **Never add `forcecast` to a
   large array.**
2. **Explicit checks before compute**: verify `ndim`/shape, require
   C-contiguity (`require_c_contiguous(...)` — tells the caller to use
   `np.ascontiguousarray()` if needed), and for outputs require writeability
   (raise if `info.readonly` / `!arr.writeable()` — message: "modified in
   place").
3. **In-place output.** Accumulators (`grid`, `normalization`) and CLEAN buffers
   (residual/model cubes) are written through `mutable_data()` pointers; they
   are *not* returned. Repeated calls accumulate (e.g. summing over MSes).
4. **Release the GIL during compute**: capture raw pointers, then
   `py::gil_scoped_release release;` around the heavy loop. The NumPy buffers
   stay valid because Python owns them.
5. **No implicit complex narrowing.** The gridder's `grid`/`normalization` and
   the degridder's `grid`/`vis_data` are taken as untyped `py::array` precisely
   so pybind11 cannot safe-cast a `complex64` buffer into a `complex128`
   temporary (which would silently drop the caller's writes). The bindings
   inspect each array's dtype and dispatch to the templated kernel: the gridder
   accumulates into a `complex64` *or* `complex128` grid (templated on the grid
   float type; visibilities stay `complex128`), and the degridder reads a
   `complex64`/`complex128` grid and writes `complex64`/`complex128`
   visibilities in place — all four combinations are explicitly instantiated.
6. **Threading inside the kernel** is via a `processing_function_threads` argument →
   `std::thread` workers; `processing_function_threads <= 0` falls back to
   `std::thread::hardware_concurrency()`, `1` runs serial. Concurrent writes to
   shared grid cells use lock-free `std::atomic_ref<double>` adds (C++20) —
   `complex<double>` is added as two `double` lanes.

---

## 7. Parallelism Model (GraphVIPER)

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
   builds the map stage.
4. **`reduce(viper_graph, reduce_fn, input_params,
   mode="tree"|"single_node"|"tree_n")`** — `reduce_fn(input_data,
   input_params)`. `"tree"` is a binary tree, `"single_node"` reduces
   everything in one task, and `"tree_n"` is the variable-arity tree
   (`n_batch` inputs per reduce task).
5. `generate_dask_workflow(viper_graph)` → `dask.compute(...)`.

When applicable multiple instances of map-reduce can be done in a single distributed_application.

### Two levels of parallelism (don't conflate them)
- **GraphVIPER Map-Reduce**: Distributed parallelism using Dask or MPI
- **Within a task**: `processing_function_threads`, passed down unchanged from
  distributed_applications to the C++ kernels (the same parameter name at every layer of the
  stack).

### Why this design (do not regress it)
GraphVIPER deliberately **loads data inside compute nodes** rather than as
separate Dask array nodes. This (a) fixes Dask "memory backpressure" for large
in-memory image cubes Dask can't see, and (b) keeps the graph small (Xarray-
backed Dask datasets create a node per data variable — fatal for MS-style data
with many variables). Keep I/O inside node tasks; don't expand the graph with
per-variable nodes.

---

## 8. Coding Conventions

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
  - **Consistency**: Parameter names should be consistent throughout the stack.
  - **CLEAN terminology**: say **residual update cycle** and **model update
    cycle** (not CASA's "major cycle" / "minor cycle") in code, docstrings, and
    docs — matching `residual_cycle_cube_single_field` /
    `model_update_cycle_cube_single_field`. A one-time "(CASA's major/minor
    cycle)" parenthetical for orientation is fine.
- **Formatting**: **Ruff** (enforced by CI + pre-commit). Don't hand-format.
- **Imports**: prefer **absolute** imports (`from
  astroviper.processing_functions.imaging.residual_cycle import ...`). Relative
  imports appear only as short re-exports in `__init__.py` files. Heavy/optional
  deps (dask, zarr, matplotlib, the C++ ext, even numpy in some hot node-task
  paths) are frequently imported **inside functions** to keep worker import time
  and graph-serialization cost low — follow the local pattern in the file you're
  editing. No * imports.
- **Docstrings**: **NumPy-style** for all functions/classes (see
  `data_group_tools.py` and `add_visibility_grid.py` for exemplars — Parameters,
  Returns, Raises, See Also).
- **Shared parameter docs (`astroviper.utils.param_docs`) — the rule for
  repeated APIs**: when the same parameter is spelled out in more than one
  public function (the common case: the distributed applications, node task, and processing
  function of one feature all declaring `image_params`,
  `processing_function_threads`, …), its description must **not** be
  hand-copied between docstrings. Instead:
  1. Put the canonical description in the subdomain's registry — for imaging,
     `processing_functions/imaging/_param_docs.py` (`IMAGING_PARAM_DOCS`,
     `{param_name: description}`).
  2. Decorate every function that shares it with `@shares_param_docs`
     (`from astroviper.utils.param_docs import shares_param_docs`) and make
     sure the file is listed in `_target_files()` in `utils/param_docs.py`.
  3. Run `python -m astroviper.utils.param_docs sync` to rewrite the source
     docstrings, then re-run `ruff format`. `... param_docs check` runs in
     pre-commit and CI and fails on drift.
  Only the *description* is shared — each function keeps its own
  `name : type, default ...` line, so defaults may differ per layer. This
  applies to **all future work with a repeated API**, not just imaging: a new
  subdomain gets its own `_param_docs.py` registry following the same pattern.
  See [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md) for the worked imaging example.
- **Logging**: use the **toolviper** logger, not `print`:
  ```python
  import toolviper.utils.logger as logger
  logger.info(...); logger.debug(...)
  ```
  The log level in node_tasks and processing_functions should always be debug or higher because there will be many instances of these functions.
- **Performance**: vectorize with NumPy; if not feasible, use C++; verify with timing.
  Large arrays are timed via `T_*` keys collected into per-chunk `DataFrame`s /
  `ReturnDict`s — keep that bookkeeping when adding stages.
- **Parameter validation**: the user-facing distributed-graph entry points use
  **toolviper's schema validation** — decorate the function with
  `@toolviper.utils.parameter.validate(config_dir=...)`. **Rule: the
  `<module_name>.param.json` schema file lives right next to the module it
  validates** (not in a central `config/` directory), and the decorator points
  the validator at the module's own directory, e.g.
  `validate(config_dir=os.path.dirname(os.path.abspath(__file__)))`. The schema
  key is the function name. Validate **once, at the graph layer** — node tasks
  and processing functions trust the already-checked `input_params`. For
  internal/leaf-level checks reuse
  `astroviper.utils.check_params.check_params` (type/range/allowed-value checks
  with defaults) rather than ad-hoc validation.

---

## 9. Do / Don't (gotchas)

**Do**
- Mirror existing patterns in `processing_functions/imaging/` for any new
  science function (data-group in/out resolution → compute in place → register
  group → return stats/timings).
- Keep new C++ kernels consistent with the §6 memory contract and with the
  binding patterns of the existing kernels.
- Pass `processing_function_threads` through unchanged from distributed applications to processing function kernels —
  the same parameter name is used at every layer.
- Keep the layering: graph code in `distributed_applications/`, I/O in `node_tasks/`,
  science in `processing_functions/`.
- Run `ruff format`, `pytest`, and rebuild after C++ edits before finishing.

**Don't**
- ❌ Add `forcecast` to (or otherwise allow silent copies / dtype conversions of)
  large arrays crossing the Python↔C++ boundary.
- ❌ Mutate `xds.attrs["data_groups"]` directly — use `data_group_tools`.
- ❌ Lowercase a data-variable name or uppercase a coordinate name.
- ❌ Put Dask/graph logic into `processing_functions/`, or science into
  `distributed_applications/`.
- ❌ Commit notebook outputs (pre-commit's `nbstripout` enforces this) or
  unformatted code.
- ❌ Introduce relative imports outside `__init__.py` re-exports.

## 10. Definition of Done for New Features
- Tests written with at least 80% coverage.
- Example notebook created (start from `docs/notebook_template.ipynb`).
- NumPy-style docstrings on all public functions.
- Initial performance testing done using larger datasets and relevant timing comparison with CASA.

---

## 11. Reference Links
- AstroVIPER repo: <https://github.com/casangi/astroviper>
- XRADIO docs: <https://xradio.readthedocs.io/en/latest/>
  - Data groups: <https://xradio.readthedocs.io/en/latest/measurement_set/overview.html#data-groups>
  - Coding conventions: <https://xradio.readthedocs.io/en/latest/development.html#coding-conventions>
  - Lazy vs eager (`open_`/`load_`): <https://xradio.readthedocs.io/en/latest/development.html#lazy-and-eager-functions>
- GraphVIPER docs: <https://graphviper.readthedocs.io/en/latest/>
  - Graph building tutorial (map/reduce, `parallel_coords`): <https://graphviper.readthedocs.io/en/latest/graph_building_tutorial.html>
- pybind11: <https://pybind11.readthedocs.io/> · scikit-build-core: <https://scikit-build-core.readthedocs.io/>

---
