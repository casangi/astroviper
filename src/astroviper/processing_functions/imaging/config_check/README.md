# Imaging configuration validation

Three independent validators own residual-update, model-update, and workflow
checks. `check_config` calls all three with the same configuration and mode.
A validator returns `None` on success and raises at the first invalid setting.
Checks are grouped and described in their owning `param_check_*` module.

| Imaging kind | Field layout | Status |
| --- | --- | --- |
| `cube` | `single_field` | Supported |
| `continuum` | `single_field` | Supported |
| `cube` | `multifield` | Raises `NotImplementedError` |
| `continuum` | `multifield` | Raises `NotImplementedError` |

Every validator takes `ImagingMode(imaging_kind, field_layout)`. Continuum
`mfs`/`mvc` specifies how continuum products are formed, independently of these
axes. Future multifield requirements belong in each of the three validators.

## Configuration boundaries

Public distributed, node, and processing entry points accept keyword-only
`input_checked=False`. Direct calls import and run their applicable validators
inside `if not input_checked`. Checked callers pass `input_checked=True` down
the stack; graph nodes also receive it through `input_params`. Only reuse that
flag with unchanged configuration in the same imaging mode.

Distributed drivers use `require_complete=True` to require image coordinates
and cycle controls before I/O, output creation, or graph construction. Lower
entry points validate the subset of settings their interfaces receive. Validation
does not insert defaults, rewrite aliases, clamp values, sort scales, or select
a different algorithm. Optional omitted settings use the implementation's
existing defaults without modifying the caller's dictionary. Explicit invalid
values, including `None` thresholds, raise.

`reference_frequency` is the canonical configuration key, in Hz.
`reference_frequency_hz` and `nterms` are rejected configuration aliases.
The historical `reference_frequency_hz` name in persisted continuum metadata
remains readable. Högbom continuum requires exactly one Taylor term; MT-MFS
requires two to four. Cube planes are independent single-term images.

JSON schemas check basic structure, dtypes, and allowed strings. Python owns
numerical bounds and conditional relationships, including checks on direct
node/processing calls. The three validators cover geometry, frequencies,
weighting, algorithm compatibility, scales, gain, stopping controls, memory
policies, backend capabilities, restoration, and output compatibility.
Unsupported requested features raise, including multifield execution,
`field_name`, continuum MPI, continuum graph visualization, and cube global
weighting. The cube Dask graph visualization and cube MPI setup remain supported.

## Units and numerical policies

Pass plain numerical values in the API's units: frequencies in Hz; cell sizes
and phase direction in radians; UVW in meters; times in the XRADIO time
coordinate convention; image sizes, CLEAN boxes, and scales in pixels. Flux
thresholds and absolute-Briggs noise use Jy, matching the imaging kernels.
Quantity objects are rejected instead of implicitly converting units. Stored
frequency units, when supplied, must be Hz.

Fixed gain must satisfy `0 < gain <= 1`. Adaptive gain is available only through
the separately documented native MT-MFS API, whose safety checks are preserved.
Scales must start at zero, increase strictly at native float32 scale precision,
and fit within half the smallest image axis. No scale is dropped. Briggs robust
is in `[-2, 2]`; absolute-Briggs noise must be positive. `uniform` remains an
unchanged public weighting value and uses its production kernel recipe.

## Data-dependent preflight

After opening input metadata and making channel partitions, both distributed
drivers check partition exclusivity, complete selected-channel coverage, and a
common Taylor basis. A node-layer helper then reads selected weights, flags, and
UVW in time blocks; it never reads visibility arrays or writes input data. It
uses the production correlation equalization and density/degridding kernels,
including local or global weighting, and excludes flagged samples and
self-correlations from spectral support. This additional metadata pass occurs
before output creation and graph construction. Local density uses one frequency
at a time; global density uses two metadata passes.

`check_spectral_sampling` constructs `sqrt(weight) * x**order`, with
`x=(frequency-reference_frequency)/reference_frequency`, and normalizes by a
common weight scale before SVD. It requires at least `nterms` distinct usable
frequencies and full numerical rank. The rank tolerance is
`max(design.shape) * eps * largest_singular_value` for the configured working
precision. A full-rank design warns when its reciprocal condition number is
below `sqrt(eps)`. A reference outside the usable frequency range also warns.
Errors include field, spectral window, selected channels, polarization, Taylor
order, reference frequency, usable counts, rank, and conditioning context.
Cube preflight requires positive effective weight in every requested channel.

## Product checks and diagnostics

New products are checked before first use even when `input_checked=True`, and
before a stopping decision can bypass CLEAN. Residual/model Taylor counts are
exactly `nterms`; PSF counts are `2*nterms-1`. Geometry, coordinates, masks, working
precision, and derived per-plane controls must agree. Every time/polarization
plane's per-scale Taylor Hessians are checked before any model plane is changed.
They must be positive definite and numerically invertible at working precision;
poor conditioning warns. Native array and Hessian boundary guards remain active.

Workflow results and persisted image metadata retain `input_validation`:

- `spectral_design` records SVD rank, singular values, precision, usable selection,
  `spectral_design_condition_number` (2-norm), and
  `spectral_design_reciprocal_condition_number`.
- `model_products` records the actual geometry, precision, reference, effective
  scales, controls, and `taylor_hessian_condition_number` per plane and scale
  (the native infinity-norm estimate). It warns above `1/sqrt(eps)`; eigenvalues
  at or below `nterms * eps * largest_absolute_eigenvalue` fail.

These metrics describe different matrices and norms. They are deliberately
stored under separate names. Derived controller limits, dataset selections, and
automatically evaluated beam recipes remain runtime state. Caller configuration
is preserved, and shared configuration is checked for consistency across stages.

Sparse XRADIO padding is excluded only when the absent time-centroid, all-NaN
UVW, and all-NaN weights agree. Real rows with invalid weights still raise.
Channel matching uses a sorted private lookup with linear memory, and cube
sampling diagnostics retain only each channel's own selection.
