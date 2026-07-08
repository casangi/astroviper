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
  `src/astroviper/distributed_applications/calibration/`,
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

### C++ extensions — one place for all build flags
There are three pybind11 modules, all built at **C++20** (the gridder needs
`std::atomic_ref`; the deconvolvers use the same standard for uniformity):
- `processing_functions/imaging/deconvolvers/hogbom` → `_hogbom_ext`
- `processing_functions/imaging/deconvolvers/aspclean` → `_aspclean_ext`
- `processing_functions/imaging/gridders/prolate_spheroidal_grid_cpp` →
  `_prolate_spheroidal_grid_ext`

**All shared compiler settings live in one file: [`cmake/AstroviperPybind.cmake`](cmake/AstroviperPybind.cmake).**
It finds `pybind11`/`Threads` once and defines:
- an `INTERFACE` target `astroviper_cpp_flags` carrying the C++ standard
  (`cxx_std_20`, strict — extensions off), the **`-ffp-contract=off`**
  reproducibility flag (see [the memory contract](#7-the-python--c-memory-contract);
  bit-identical IEEE-754 across compilers/arches — do not remove), `-stdlib=libc++`
  for Clang, the `VERSION_INFO` define, and a `Threads::Threads` link; and
- a helper `astroviper_add_pybind_module(NAME … SOURCES … [INCLUDE_DIRS …]
  [DESTINATION …])` that creates the module, wires `include/`, applies the shared
  flags, and installs it (`INCLUDE_DIRS` defaults to `include`; `DESTINATION`
  defaults to the module's path relative to `src/`).

To **change a flag for every kernel**, edit `cmake/AstroviperPybind.cmake` — and
only that file. Optimization level / `-fPIC` are left to CMake's build type
(`build-type=Release` is pinned in `pyproject.toml`); don't hand-roll `-O3`.

Each module's `CMakeLists.txt` is ~3 lines: an `astroviper_add_pybind_module(...)`
call, preceded by a small standalone-build bootstrap block. The root
`CMakeLists.txt` only `include()`s the shared module and `add_subdirectory()`s
each one. A single kernel can also be **built on its own** for fast iteration —
`cmake -S <module_dir> -B build && cmake --build build` — and it picks up the
exact same flags (the module finds and includes the shared file itself).

### Build type (Release / Debug)
`CMAKE_BUILD_TYPE` is set via `build-type` in `[tool.scikit-build.cmake]` of
`pyproject.toml` (pinned to `Release`). It is the **only** lever for the
optimization/debug flags — the shared cmake file deliberately leaves opt-level to
CMake. The reproducibility/standard flags (`-std=c++20`, `-ffp-contract`,
`-stdlib=libc++`, threads, `VERSION_INFO`) apply for **every** build type.

| `build-type` | Flags | Notes |
| --- | --- | --- |
| `Release` | `-O3 -DNDEBUG` | default; asserts **off** |
| `Debug` | `-g` (i.e. `-O0`) | asserts + pybind11 bounds checks **on** |
| `RelWithDebInfo` | `-O2 -g -DNDEBUG` | profiling with symbols |
| `MinSizeRel` | `-Os -DNDEBUG` | smallest binary |

Override for a single build without editing `pyproject.toml` (keeps `Release` as
the committed default):
```bash
pip install -e . --no-build-isolation --config-settings=cmake.build-type=Debug
# or, equivalently, via the SKBUILD_<dotted-key-uppercased> env var:
SKBUILD_CMAKE_BUILD_TYPE=Debug pip install -e . --no-build-isolation
```
If a switch behaves oddly (stale cache), `rm -rf build` first —
`CMAKE_BUILD_TYPE` is cached per `build/{wheel_tag}/`.

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

### Documentation notebooks (`docs/`)
Tutorials live under `docs/`, organised to **mirror the source module tree** —
put a new notebook in the folder matching the layer + subdomain of the function
it demonstrates:
- `distributed_applications_tutorials/{imaging,image_analysis,model}/`
- `processing_functions_tutorials/{imaging,image_analysis,visibility_manipulation}/`
- `calibration_tutorials/`, plus `theory/` and `utils/`.

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

AstroVIPER is organized into four layers. **Calls only ever go downward**
(graphs → node_tasks → processing_functions → utils); never call upward.

```
src/astroviper/
├── distributed_applications/   (1) Build + compute Dask graphs (GraphVIPER map/reduce).
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
1. **`distributed_applications/`** — Constructs `parallel_coords`, maps a `node_task`
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
import astroviper.distributed_applications as distributed_applications
distributed_applications.imaging.image_cube_single_field(...)   # user-facing driver
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

### Data-group parameter naming (imaging processing functions)
Every imaging processing function names its data-group parameters from a fixed
vocabulary so call sites read consistently:

| Parameter | Side | Role |
| --- | --- | --- |
| `ms_data_group_in_name` / `ms_data_group_out_name` | measurement set | input / output group name |
| `image_data_group_in_name` / `image_data_group_out_name` | image dataset | input / output group name |
| `ms_data_group_out_modified` / `image_data_group_out_modified` | — | dict of role → new data-variable name for the output group |

- Use `image_data_group_*` (not `img_*`); use the explicit `_in_`/`_out_` split
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

## 6. Imaging Deep-Dive

This is the subsystem to know best. The cube single-field imager is implemented
across all three layers, each named `image_cube_single_field`:

### 6.1 Driver — `distributed_applications/imaging/image_cube_single_field.py`
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
`instrument_polarization_basis` (`"linear"` / `"circular"`),
`processing_set_data_group_name`, `single_precision_image`,
`processing_function_threads`, `n_chunks`, `disk_chunk_sizes`,
`fft_backend`, `memory_mode`.

- **`instrument_polarization_basis`** (`"linear"` → feeds `XX`/`YY`,
  `"circular"` → `RR`/`LL`). The per-chunk gridding is done in this correlation
  basis: the node task builds the empty image with these correlation pol labels,
  and the residual cycle transforms the model image into this basis before
  degridding. The output image is always produced in the Stokes basis given by
  `image_params["polarization_coords"]`.
- **`single_precision_image`** (default `True`) sets the precision of the
  **image-domain** arrays. When `True`, the gridded visibility/uv-sampling grids
  and every sky/PSF/model image are single precision (`complex64` / `float32`),
  so the minor-cycle deconvolution runs in single precision and the image-cube
  memory footprint is roughly halved. The **visibilities stay double precision**
  (`complex128`) and the residual visibility is computed in double precision —
  only the image-domain arrays (and the FFTs over them) are cast to single
  precision after gridding. When `False` the image-domain arrays are double
  precision. See [the precision model](#65-imaging-io-conventions--utilsio).
- **Parameter validation happens only here.** The distributed-graph
  `image_cube_single_field` is decorated with
  `@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)`; the full
  parameter schema lives in `image_cube_single_field.param.json` **next to the
  module** (`distributed_applications/imaging/`), and the decorator points the
  validator at the module's own directory via `config_dir`. The node-task and
  processing-function layers do **not** re-validate — they trust the
  already-checked `input_params`. (See the rule in
  [Coding Conventions](#9-coding-conventions): the `*.param.json` schema always
  sits beside the function it validates, not in a central `config/` directory.)

- **`disk_chunk_sizes`** (e.g. `{"frequency": 200}` or `"Auto"`) adds a
  *data-loading layer*: one load node per on-disk chunk reads the native chunk
  once; map tasks then sub-select their slice from the pre-loaded data
  (`input_params["input_data"]`), avoiding redundant disk reads.
- **`memory_mode`**: only `"in_memory"` is currently implemented (there are
  `assert memory_mode == "in_memory"` guards; `"in_place"` / `"cache"` are
  stubs). Do not assume the others work.

### 6.2 Node task — `node_tasks/imaging/image_cube_single_field.py`
The node task has a **fully explicit, NumPy-documented, standalone-callable
signature** (`image_cube_single_field(image_params, imaging_weights_params, ...,
task_coords, data_selection, image_store, input_data_store, ..., graph_mode=True)`)
— it does *not* take an opaque `input_params` dict. **`graphviper.graph_tools.map`
adapts it automatically** to the single-`input_params`-dict calling convention,
so the driver passes the explicit node task to `map(...)` directly (see
[6.6](#66-layer-interfaces--parameter-doc-codegen)).
1. `memory_setup(131072)` **first** (pins the malloc mmap threshold so big
   allocations are released to the OS on free — must run before any large
   allocation).
2. Build the empty per-chunk `img_xds` (correlation pol labels derived from
   `instrument_polarization_basis`).
3. Get data: use `input_data` if the loading layer pre-loaded it, else
   `load_processing_set(...)` (eager) for this chunk's `data_selection`.
4. Call `pf.imaging.image_cube_single_field(ps_xdt, img_xds, image_params, ...)`
   with explicit keyword arguments.
5. Write the result slice to Zarr via
   `astroviper.utils.io.write_result_chunk_to_disk_using_zarr(...)`.
6. `free_memory()` and return a small timing `DataFrame`.

### 6.3 Science — `processing_functions/imaging/image_cube_single_field.py`
The **iteration-control and bookkeeping helpers** (not science kernels) live in
the `processing_functions/imaging/utils/` subpackage: `iteration_control.py`
(`IterationController`, stop codes, `merge_return_dicts`,
`get_calculate_cycle_controls`, the `get_*_from_returndict` extractors,
`format_/print_deconvolve_dict`), `return_dict.py` (`ReturnDict`, `Key`),
`timing.py` (`accumulate_timing`), and `visibility.py`
(`drop_auto_correlations` — the shared cross-correlation filter used by the PSF
and undeconvolved-image gridders; do not re-inline it). Import them from
`astroviper.processing_functions.imaging.utils` (the package re-exports the
public symbols).

Runs the **major/minor cycle** CLEAN loop via `IterationController`:
- `residual_cycle_cube_single_field(...)` — degrid model → compute residual
  visibilities → grid → FFT-normalize → form residual image (+ PSF on the first
  iteration), primary beam, and imaging weights (first iteration only).
- `model_update_cycle_cube_single_field(...)` → `deconvolve(...)` (the minor
  cycle: Hogbom or ASP CLEAN in C++) updates the model image, with a mask from
  `make_mask`.
- Accumulate per-plane stats into a `ReturnDict`; check convergence; iterate.
- A final residual cycle produces the last residual image.
- When `restore=True` (off by default), a final `restore_image(...)` step
  (`processing_functions/imaging/restore.py`) convolves the model with the clean
  beam (the per-frequency Gaussian fit to the PSF, in the `residual` data group)
  and adds the residual, writing `SKY_RESTORED`. The driver auto-adds
  `"sky_restored"` to `image_data_variables_keep` so it is created/written.

### 6.4 Key processing functions & gridders
> Naming convention: single-field imaging functions put the `single_field`
> qualifier **at the end** (`make_point_spread_function_single_field`,
> `make_primary_beam_single_field`, `make_undeconvolved_image_single_field`,
> `imaging_setup_single_field`, …).

- Weighting: `calculate_imaging_weights.py` (`"natural"`, `"briggs"`/robust).
- Gridding (vis → uv grid): `add_visibility_grid.py`
  (`add_visibility_grid_single_field`).
- PSF / UV-sampling grid: `make_point_spread_function.py`
  (`make_point_spread_function_single_field`, plus the
  `add_uv_sampling_grid_single_field` / `add_uv_sampling_grid_mosaic` gridders
  that build the PSF numerator — gridding the imaging weights *is* the first
  step of forming the PSF).
- Undeconvolved (dirty/residual) image gridding: `make_undeconvolved_image.py`
  (`make_undeconvolved_image_single_field`).
- Degridding (model uv grid → vis): `get_visibility_grid.py`.
- FFT + normalization: `fft_normalize_prolate_spheriodal_gridder.py`.
- Primary beam: `make_pb_symmetric.py` (airy disk).
- Polarization: `image_analysis/transform_polarization_basis.py`
  (stokes ↔ linear).
- PSF fit: `image_analysis/point_spread_function_gaussian_fit.py`.
- Restore: `restore.py` (`restore_image`) — clean-beam-convolved model plus
  residual (FFT convolution, clean beam built from the residual group's
  `beam_fit_params_point_spread_function`).
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

#### Precision model (`single_precision_image`)
The driver's `single_precision_image` (default `True`) controls **image-domain**
precision only; visibilities are **always** `complex128`:

| Stage | `single_precision_image=True` | `=False` |
| --- | --- | --- |
| Observed/model/residual **visibilities** | `complex128` | `complex128` |
| Gridded UV / UV-sampling grid | `complex64` | `complex128` |
| Grid **normalization** accumulator | `float64` (always) | `float64` |
| Sky / PSF / model **images** | `float32` | `float64` |
| Model→vis **uv grid** (`fft_norm_img_xds`) | `complex64` | `complex128` |
| Minor-cycle **deconvolution** | `float32` | `float64` |

Casting happens **after gridding, before the FFT**: the C++ gridder accumulates
directly into a `complex64` grid (no extra full-resolution copy), the iFFT/FFT
run at the grid precision, and the resulting images are `float32`. The
degridder widens each (possibly `complex64`) model-grid cell to `complex128` and
writes `complex128` model visibilities, so `residual = observed − model` is
formed in double precision. Threading the precision: the driver sets
`input_params["single_precision_image"]`; `residual_cycle` derives
`complex_dtype`/`float_dtype` from it and forwards `complex_dtype` to the
gridders (`add_visibility_grid_single_field`, `add_uv_sampling_grid_single_field`),
to `fft_norm_img_xds`/`ifft_norm_img_xds`, and to the PSF builder. On-disk Zarr
dtypes follow via `create_empty_data_variables_on_disk(double_precision=not
single_precision_image)`.

### 6.6 Layer interfaces & parameter-doc codegen
Every imaging layer (driver, node task, science processing functions) exposes a
**fully explicit, NumPy-documented, standalone-callable** signature — none of
them take an opaque `input_params` dict. Two pieces of infrastructure keep that
clean:

- **Explicit node tasks (auto-adapted by graphviper)** — graphviper's `map`
  invokes each node task with a single `input_params` dict (into which it injects
  per-node keys: `task_id`, `task_coords`, `data_selection`, `input_data`, ...).
  A node task may instead have a **fully explicit signature**: `map` adapts it
  automatically via `graphviper.graph_tools.map.make_graph_node_task`, which
  returns a thin `<name>_wrap(input_params)` adapter that expands the dict into
  the function's declared keyword arguments, **forwarding only the keys it
  declares** (extra keys the driver/graphviper add are dropped). Legacy node
  tasks that take a single `input_params` dict are passed through unchanged. So
  the driver passes the explicit `node_tasks.imaging.image_cube_single_field`
  straight to `map` — **no astroviper-side wrapper** — and the node task stays a
  real, documented, directly-callable function. The adapter lives in
  **graphviper**, not astroviper, so graphviper users never see this detail.

- **Parameter-doc codegen (`astroviper.utils.param_docs`)** — a parameter such as
  `image_params` is spelled out in all three layers. Its canonical description
  lives in **one** registry,
  `processing_functions/imaging/_param_docs.py` (`IMAGING_PARAM_DOCS`,
  `{param_name: description}`). Functions that share these descriptions are
  marked with `@shares_param_docs`
  (`from astroviper.utils.param_docs import shares_param_docs`). The codegen
  rewrites the matching `Parameters` *descriptions* in the source files —
  preserving each function's own `name : type` line (so functions can give a
  parameter different defaults) and every other docstring section.

  **Workflow:** edit a description in `_param_docs.py`, then
  ```bash
  python -m astroviper.utils.param_docs sync     # rewrite the source docstrings
  python -m astroviper.utils.param_docs check    # verify in sync (CI / pre-commit)
  ```
  (`python src/astroviper/utils/param_docs.py sync|check` works too — the tool is
  standalone and needs only `libcst`, no package build.) The
  [`imaging-param-docs` pre-commit hook](.pre-commit-config.yaml) and the
  [`param-docs` CI workflow](.github/workflows/param-docs.yml) run `check`. After
  running `sync`, re-run **black** and commit. To bring a **new** function into
  the system: decorate it `@shares_param_docs`, add its file to `_target_files()`
  in `utils/param_docs.py` (if not already listed), and run `sync`.

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
6. **No implicit complex narrowing.** The gridder's `grid`/`normalization` and
   the degridder's `grid`/`vis_data` are taken as untyped `py::array` precisely
   so pybind11 cannot safe-cast a `complex64` buffer into a `complex128`
   temporary (which would silently drop the caller's writes). The bindings
   inspect each array's dtype and dispatch to the templated kernel: the gridder
   accumulates into a `complex64` *or* `complex128` grid (templated on the grid
   float type; visibilities stay `complex128`), and the degridder reads a
   `complex64`/`complex128` grid and writes `complex64`/`complex128`
   visibilities in place — all four combinations are explicitly instantiated.
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

## 10. Do / Don't (gotchas)

**Do**
- Mirror existing patterns in `processing_functions/imaging/` for any new
  science function (data-group in/out resolution → compute in place → register
  group → return stats/timings).
- Keep the gridder's Numba and C++ implementations consistent.
- Pass `num_threads` through to kernels and respect `processing_function_threads`.
- Keep the layering: graph code in `distributed_applications/`, I/O in `node_tasks/`,
  science in `processing_functions/`.
- Run `black`, `pytest`, and rebuild after C++ edits before finishing.

**Don't**
- ❌ Add `forcecast` to (or otherwise allow silent copies / dtype conversions of)
  large arrays crossing the Python↔C++ boundary.
- ❌ Mutate `xds.attrs["data_groups"]` directly — use `data_group_tools`.
- ❌ Lowercase a data-variable name or uppercase a coordinate name.
- ❌ Put Dask/graph logic into `processing_functions/`, or science into
  `distributed_applications/`.
- ❌ Assume `memory_mode` other than `"in_memory"` works.
- ❌ Touch calibration code (see [scope](#scope-for-agents--what-to-ignore)).
- ❌ Commit notebook outputs (pre-commit's `nbstripout` enforces this) or
  unformatted code.
- ❌ Introduce relative imports outside `__init__.py` re-exports.

---

## 11. Where to make a change (task → location)

| Task | Where |
| --- | --- |
| New/changed imaging **workflow / parallelism** | `distributed_applications/imaging/` |
| New **node task** (I/O + orchestration of a chunk) | `node_tasks/imaging/` — give it an explicit signature and pass it straight to `map` (graphviper auto-adapts it; see [6.6](#66-layer-interfaces--parameter-doc-codegen)) |
| New **science kernel** (gridding, weighting, deconvolution, FFT, PB) | `processing_functions/imaging/` |
| graphviper single-dict → explicit node-task adapter | **graphviper** `graph_tools/map.py` (`make_graph_node_task`, applied automatically by `map`) |
| Shared **parameter docstring** (appears in >1 layer) | edit `processing_functions/imaging/_param_docs.py`, mark functions `@shares_param_docs`, run `python -m astroviper.utils.param_docs sync` (tool: `utils/param_docs.py`) |
| New **C++ kernel** | a sub-package under `processing_functions/.../{gridders,deconvolvers}/` with `include/`, `src/`, `python/bindings.cpp`, and a ~3-line `CMakeLists.txt` calling `astroviper_add_pybind_module(NAME … SOURCES …)` (copy an existing module's, incl. its standalone bootstrap block); add one `add_subdirectory(...)` line to the root `CMakeLists.txt`. Shared flags come from `cmake/AstroviperPybind.cmake` — don't re-declare them. |
| Iteration control / `ReturnDict` / timing bookkeeping | `processing_functions/imaging/utils/` |
| Data-group helpers | `utils/data_group_tools.py` |
| Zarr variable defs / chunk writers | `utils/io.py` |
| Chunk-count / thread heuristics | `utils/data_partitioning.py` |
| Driver-level parameter schema (toolviper `@validate`) | `<module>.param.json` **next to the module** (e.g. `distributed_applications/imaging/`) |
| Ad-hoc parameter validation helpers | `utils/check_params.py` |
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
