# Continuum software test coverage

These tests protect the continuum implementation during development and PR
review. Generated known-truth point sources check continuum flux, spectral-index
and primary-beam recovery. Scientific equivalence to CASA, QA2 acceptance, and
deployment-cluster performance are outside the repository test suite's scope.

## Coverage map

Paths below are relative to `tests/`.

| Area | Existing coverage | Added coverage |
| --- | --- | --- |
| Processing mathematics | MFS gridding and prediction; MVC Taylor equations and PB cutoff; model accumulation; restoration/PB correction | Independent scalar moment calculations for 1–3 Taylor terms, float32/64, multiple times and polarizations; invalid/empty weights; coordinate alignment; residual-only updates; invalid final PB products |
| Persistent state | Initial/later model state; observed grids; static products; weight and PB caches | Complex subtraction with independent array/metadata ownership; mismatched cache rejection; Zarr v2/v3 PB round trips at both precisions; reordered, disjoint channel writes; missing-cache failures |
| Distributed reduction | MFS/MVC reduction; static PB consistency; global weight preparation; timing aggregation | Flat/tree associativity with read-only inputs; incompatible Taylor metadata, dimensions, shapes and coordinates; task-local observed-grid preservation and duplicate ownership rejection |
| Complete continuum workflow | TW Hydra imaging, partition changes, natural/uniform/Briggs paths, Taylor orders, precision, restoration, PB correction, later-cycle prediction | All eight MFS/MVC × local/global × native/CASA weighting configurations, each with masked cleaning and restored output; recomputed grids/tree reduction compared with disk caches/single-node reduction |
| Memory and scheduling | Individual cache-mode comparisons; process-worker result transfer; direct I/O and sharding | Simultaneous disk-backed weights, observed grids and MVC PBs; reopened output compared with returned arrays; temporary channel products excluded from public output |
| Stopping | Dirty-only imaging; iteration/cycle controls; refreshed-residual verification before restoration | Complete runs enforce two imaging cycles, nonzero model, matching iteration totals across cache/reduction variants, and model support inside the supplied mask |
| Shared imaging functions | Weighting, FFTs, deconvolver, PSF fitting, ALMA/VLA PB prescriptions | Reused by the continuum tests; existing shared tests remain necessary |

## New test modules

- `unit/processing_functions/imaging/test_continuum_contracts.py`: numerical
  oracles and model/cache/PB input contracts.
- `unit/node_tasks/imaging/test_continuum_cache_contracts.py`: real Zarr cache
  operations and channel ownership.
- `unit/distributed_applications/imaging/test_continuum_reduction_contracts.py`:
  reducer consistency, validation and task-local cache ownership.
- `component/test_continuum_configurations.py`: eight configuration regressions
  using the five-channel TW Hydra fixture obtained through ToolVIPER's managed
  data service. No CASA installation is needed. Each case extracts its own
  writable Processing Set.
- `component/test_continuum_known_truth.py`: compact generated phase-centre
  spectral-index and single-field off-axis point-source cases. MFS and MVC must
  recover the two Taylor coefficients and the primary-beam-corrected one-Jy
  flux without CASA or stored visibility data. MVC additionally verifies that
  `local_beam_moments` recovers the off-axis spectral index; the processing-level
  regression contrasts this with the documented `channel_psf`/CASA bias at the
  same realistic Taylor-zero stopping threshold.

The component comparisons are within each algorithm configuration. They do
**not** require MFS and MVC, local and global weighting, or native and CASA
weighting to produce identical images. They protect equivalence of execution
and storage choices. Independent numerical unit-test oracles complement these
comparisons, which alone could miss a defect shared by both execution paths.

### Taylor-coefficient convention

The MFS and MVC tests intentionally use different primary-beam conventions.
MFS fits the apparent sky, `A_nu * I_nu`, so `SKY_MODEL` `tt1 / tt0` contains
the intrinsic source spectral index plus the primary-beam spectral index. MVC
fits the intrinsic spectrum under its common-beam convention by predicting
`M_nu * A_nu / Abar`, so its `tt1 / tt0` excludes the primary-beam spectral
index. This matches CASA's standard-gridder MT-MFS and MVC behavior. Tests that
compare recovered Taylor coefficients must therefore use the expected
convention for the selected mode rather than requiring off-axis MFS and MVC
coefficients to be equal.

The component matrix uses 64×64 images, three frequency tasks, one kernel
thread and two imaging cycles. The controller's iteration total sums the two
polarization planes; `max_iter` is a per-plane budget. Product comparisons use
`rtol=1e-6` with a peak-scaled `1e-8` absolute floor (minimum `1e-14`) to allow
floating-point reduction order changes near zero. Reopened products must also
match the in-memory return. The restored reference-frequency image is a single
`(time, polarization, l, m)` product without a Taylor axis: higher Taylor terms
describe spectral coefficients, not additional restored images. Its values are
checked against the Taylor-zero model convolved with the fitted clean beam plus
the Taylor-zero residual. PB correction uses the model-first convention
`convolve(model / PB, clean_beam) + residual / PB`. Unit tests verify these
values; component tests verify that pixels below the PB cutoff are NaN.

## Additional boundary coverage

The three `test_continuum_edge_cases.py` modules (processing functions,
node tasks and distributed applications) cover:

- Reference-frequency precedence and fallback; scalar, empty, nonfinite and
  nonpositive inputs; invalid Taylor orders and incomplete model registrations.
- Empty Processing Sets and partially available weights.
- MVC grid allocation and reuse at complex64/128; malformed visibility arrays,
  missing axes, incompatible preallocated buffers and invalid execution options.
  These wrapper tests replace only the compiled gridding call with a small
  accumulator; native numerical gridding remains covered by the existing kernel
  and end-to-end tests.
- Direct and wrapped append inputs; explicit precision/thread/backend controls;
  task-local beam ownership and cached-weight shape validation.
- Invalid or duplicate distributed weighting results; empty, nonfinite and
  duplicate density-grid frequency coordinates.
- Watchdog diagnostics on writable and unavailable paths: logging failure must
  not replace the original task failure.

## Exception contracts without stored test data

The three `test_continuum_exceptions.py` modules use only small in-memory
NumPy/xarray objects. They create no files, add no fixture archives and require
no downloads. Each case checks the exception type and an informative part of
its message.

- Processing functions: incomplete restoration/model/PSF registrations,
  missing Taylor dimensions and empty axes, unsupported deconvolution,
  malformed per-plane controls, invalid prediction models/frequencies,
  incompatible PB dimensions/coordinates, missing accumulation state and
  malformed MVC normal-equation inputs.
- Node tasks: absent observed-grid caches or output-store parameters, missing
  static/model state at finalization, malformed model-update envelopes,
  incompatible clean masks and unreduced reference-PB state.
- Distributed layer: incomplete map results, invalid cache mappings, duplicate
  weight/PB task ownership, and absent or ambiguous partition frequencies.

Restoration, model-update and cached-MFS rejection tests also verify that the
input image remains unchanged on those early failure paths. The tests use
ordinary xarray objects; they do not fabricate impossible dimension states to
force otherwise unreachable guards. Existing complete imaging regressions use
the same managed TW Hydra fixture, unchanged.

## Execution variants and lifecycle regressions

Additional modules cover execution variants and lifecycle contracts:

| Module | Contracts |
| --- | --- |
| `unit/processing_functions/imaging/test_continuum_variants.py` | PSF/beam/legacy-metadata layouts through the real restoration backend; missing backend output; invalid layouts; exact PB cutoffs; zero/extreme weights and single-channel/single-Taylor inputs |
| `unit/node_tasks/imaging/test_continuum_lifecycle.py` | Direct versus already-prepared finalization, exactly-once restoration, immutable accumulated model, cache replacement and repeated cleanup in Zarr v2/v3 |
| `unit/distributed_applications/imaging/test_continuum_variants.py` | Uneven frequency partitions, reordered inputs, unbalanced trees, zero-contribution leaves, corrupt upstream graph results, remaining geometry/metadata exceptions |

`tests/utils/continuum_images.py` generates the small images in memory. Disk
cache cases use only pytest temporary directories. No stored data fixtures or downloads
were added. Finalization unit tests stub image preparation to count its calls,
but execute real restoration and PB correction; full imaging graphs remain
covered by the component matrix. Upstream-failure tests substitute graph
results intentionally to check that the driver reports the specific failure.

### Storage defects found and repaired

The lifecycle tests exposed two defects with xarray 2026.7.0 and Zarr 3.3.0.
Their four former expected-failure cases now run as ordinary regression tests.

1. **MFS cache format preservation (two precision cases).** The writer now
   passes the existing root's Zarr format explicitly to `Dataset.to_zarr`.
   Creating or replacing a cache subgroup therefore cannot introduce v3 root
   metadata into a v2 store and hide existing public arrays. The tests check
   the root format and public array after each write, then repeat cleanup.
2. **Imaging-weight cache metadata (fresh/interrupted-run cases).** Cache
   creation, writing, activation and removal open each processing-set child
   directly with consolidation disabled, avoiding stale child metadata even
   when root consolidation is disabled. The driver refreshes child metadata
   before root metadata after each structural or registration change. Workers
   only write disjoint array regions; they do not consolidate metadata.
   Tests verify cache replacement, activation, repeated cleanup and preserved
   input weights through both direct and consolidated root/child reads, in
   Zarr v2 and v3.

The changes are confined to continuum cache I/O. They do not change imaging
mathematics or the cube workflow, and require no new stored data or downloads.

## Running the tests

Run from the repository root using an environment containing this checkout:

```bash
python -m pytest tests/unit/processing_functions/imaging/test_continuum*.py \
  tests/unit/node_tasks/imaging/test_continuum*.py \
  tests/unit/distributed_applications/imaging/test_continuum*.py \
  tests/component/test_continuum_configurations.py -q

python -m pytest tests/unit tests/component/test_continuum_configurations.py \
  --cov=astroviper --cov-branch --cov-report=term-missing
```

The new files follow existing pytest discovery conventions; Linux CI already
runs `tests/`, so no separate workflow is required. Existing cube references
are unchanged.

## Scope and limits

Coverage is not exhaustive. The eight-case matrix uses linear parallel hands,
double precision and the current Högbom continuum path. Existing tests cover
additional precision and scheduling paths separately; this is not the full
Cartesian product of telescope, polarization, precision, cache mode, scheduler,
weighting and Taylor order. No distributed multi-host or MPI guarantee follows
from local Dask tests. New supported functionality should add focused tests
rather than merely increasing the number of equivalent configurations.

## Coverage reporting

The coverage command above measures the current Python statement and branch
coverage for the continuum suites. Native C++ kernels and scientific
equivalence to external imaging packages require separate validation; Python
coverage alone is not evidence of scientific correctness.

## External input and primary-beam references

TW Hydra is downloaded once per session by `tw_hydra_source_store` through
ToolVIPER's managed data manifest and Cloudflare-backed data service. Continuum
tests create separate writable directory copies of the unpacked processing set.
The shared download is deleted at session teardown, and copied fixture inputs
are deleted at their test/module teardown, including on failures. The cube
component tests use the same ToolVIPER download path. Generated test outputs
remain subject to pytest's normal temporary-directory retention.

Primary-beam tests embed small tables of independent CASA output values;
no PB data downloads are required.
Large-data image-level comparisons are outside the repository test suite.
