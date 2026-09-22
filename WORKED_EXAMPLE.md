## How-To Example: Imaging Deep-Dive

This is the subsystem to know best — use it as the worked example of how the
four layers fit together. The cube single-field imager is implemented across all
three code layers, each named `image_cube_single_field`. A typical user-facing
call (see the tutorial notebook
`docs/distributed_applications_tutorials/imaging/single_field_cube.ipynb` for a
runnable end-to-end version):

```python
from astroviper.distributed_applications.imaging import image_cube_single_field

return_dict = image_cube_single_field(
    ps_store="twhya_selfcal_lsrk_5chans.ps.zarr",   # input processing set (Zarr)
    image_store="twhya_clean.img.zarr",             # output image dataset (Zarr)
    image_params=image_params,                       # image geometry + output coords
    imaging_weights_params={"weighting": "briggs", "robust": 0.5},
    iteration_control_params={"niter": 300, "nmajor": 3, "threshold": 0.001, "gain": 0.1},
    gridder="prolate_spheroidal",
    deconvolver="hogbom_many_threads",
    image_data_variables_keep=["sky_residual", "sky_model", "mask",
                               "point_spread_function", "primary_beam",
                               "beam_fit_params_point_spread_function"],
    processing_set_data_group_name="base",
    single_precision_image=True,
    processing_function_threads=1,
    n_mapping_parallelism={"frequency": 5},
    restore=True,
    overwrite=True,
)
```

The rest of this section walks down the layers behind that call:

### 12.1 Driver — `distributed_applications/imaging/image_cube_single_field.py`
The user-facing function. Sequence:
1. `make_empty_sky_image(...)` → `write_image(..., out_format="zarr")` (creates
   the image store with correct coords/dims).
2. `calculate_mapping_parallelism_for_cube_imaging(...)` → decide the mapping
   parallelism (frequency chunk count) from per-channel memory estimate +
   available threads.
3. `make_parallel_coord(coord=img_xds.frequency, n_chunks=...)` → defines
   parallelism (imaging is **parallelized over frequency** for cubes).
4. `create_empty_data_variables_on_disk(...)` → pre-allocate Zarr arrays
   (NaN-filled) for the variables in `image_data_variables_keep`, so map tasks
   can **lazily write their own slice**.
5. `open_processing_set(...)` (lazy) →
   `interpolate_data_coords_onto_parallel_coords(...)` → `node_task_data_mapping`.
6. `map(input_data=ps_xdt, node_task=node_tasks.imaging.image_cube_single_field,
   ...)` → `reduce(..., mode="tree")` → `generate_dask_workflow` →
   `dask.compute` → `zarr.consolidate_metadata`.


### 12.2 Node task — `node_tasks/imaging/image_cube_single_field.py`
The node task has a **fully explicit, NumPy-documented, standalone-callable
signature** (`image_cube_single_field(image_params, imaging_weights_params, ...,
task_coords, data_selection, image_store, input_data_store, ..., graph_mode=True)`).
1. Build the empty per-chunk `img_xds` (correlation pol labels derived from
   `instrument_polarization_basis`).
2. Get data: use `input_data` if the loading layer pre-loaded it, else
   `load_processing_set(...)` (eager) for this chunk's `data_selection`.
3. Call `pf.imaging.image_cube_single_field(ps_xdt, img_xds, image_params, ...)`
   with explicit keyword arguments.
4. Write the result slice to Zarr via
   `astroviper.utils.io.write_result_chunk_to_disk_using_zarr(...)`.

### 12.3 Science — `processing_functions/imaging/image_cube_single_field.py`
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

Runs the CLEAN loop of **residual update cycles** and **model update cycles**
(CASA's *major* and *minor* cycles — use the update-cycle names in AstroVIPER)
via `IterationController`:
- **Residual update cycle** — `residual_cycle_cube_single_field(...)`: degrid
  model → compute residual visibilities → grid → FFT-normalize → form residual
  image (+ PSF on the first iteration), primary beam, and imaging weights
  (first iteration only).
- **Model update cycle** — `model_update_cycle_cube_single_field(...)` →
  `deconvolve(...)` (Hogbom or ASP CLEAN in C++) updates the model image, with
  a mask from `make_mask`.
- Accumulate per-plane stats into a `ReturnDict`; check convergence; iterate.
- A final residual update cycle produces the last residual image.
- When `restore=True` (off by default), a final `restore_image(...)` step
  (`processing_functions/imaging/restore.py`) convolves the model with the clean
  beam (the per-frequency Gaussian fit to the PSF, in the `residual` data group)
  and adds the residual, writing `SKY_RESTORED`. The distributed application auto-adds
  `"sky_restored"` to `image_data_variables_keep` so it is created/written.

### 12.4 Key processing functions & gridders
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
- The standard gridder kernel is implemented in C++
  (`gridders/prolate_spheroidal_grid_cpp`); pure-Python reference copies used
  as test oracles live in
  `tests/unit/processing_functions/imaging/gridders/reference_gridders.py`.
  **Keep the references in sync** if you change the gridding math.

### 12.5 Imaging I/O conventions — `utils/io.py`
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
The distributed application `single_precision_image` (default `True`) controls **image-domain**
precision only; visibilities are `complex128`:

| Stage | `single_precision_image=True` | `=False` |
| --- | --- | --- |
| Observed/model/residual **visibilities** | `complex128` | `complex128` |
| Gridded UV / UV-sampling grid | `complex64` | `complex128` |
| Grid **normalization** accumulator | `float64` (always) | `float64` |
| Sky / PSF / model **images** | `float32` | `float64` |
| Model→vis **uv grid** (`fft_norm_img_xds`) | `complex64` | `complex128` |
| Model-update-cycle **deconvolution** | `float32` | `float64` |

Casting happens **after gridding, before the FFT**: the C++ gridder accumulates
directly into a `complex64` grid (no extra full-resolution copy), the iFFT/FFT
run at the grid precision, and the resulting images are `float32`. The
degridder widens each (possibly `complex64`) model-grid cell to `complex128` and
writes `complex128` model visibilities, so `residual = observed − model` is
formed in double precision. Threading the precision: the distributed applications sets
`input_params["single_precision_image"]`; `residual_cycle` derives
`complex_dtype`/`float_dtype` from it and forwards `complex_dtype` to the
gridders (`add_visibility_grid_single_field`, `add_uv_sampling_grid_single_field`),
to `fft_norm_img_xds`/`ifft_norm_img_xds`, and to the PSF builder. On-disk Zarr
dtypes follow via `create_empty_data_variables_on_disk(double_precision=not
single_precision_image)`.

### 12.6 Layer interfaces & parameter-doc codegen
Every imaging layer (distributed applications, node task, processing functions) exposes a
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
  declares** (extra keys the distributed_applications/graphviper add are dropped). Legacy node
  tasks that take a single `input_params` dict are passed through unchanged. So
  the distributed application passes the explicit `node_tasks.imaging.image_cube_single_field`
  straight to `map` — **no astroviper-side wrapper** — and the node task stays a
  real, documented, directly-callable function. The adapter lives in
  **graphviper**, not astroviper, so astroviper users never see this detail.

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
  running `sync`, re-run **ruff format** and commit. To bring a **new** function into
  the system: decorate it `@shares_param_docs`, add its file to `_target_files()`
  in `utils/param_docs.py` (if not already listed), and run `sync`.
