# Continuum software test coverage

These tests protect the continuum implementation during development and PR
review. They do not establish scientific equivalence to CASA, QA2 acceptance,
or performance on a deployment cluster. Those remain harness/demonstrator work.

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
  using the bundled five-channel TW Hydra fixture. No CASA installation,
  external harness checkout, Google Drive access or new reference archive is
  needed. Each case extracts its own writable Processing Set.

The component comparisons are within each algorithm configuration. They do
**not** require MFS and MVC, local and global weighting, or native and CASA
weighting to produce identical images. They protect equivalence of execution
and storage choices. Independent numerical unit-test oracles complement these
comparisons, which alone could miss a defect shared by both execution paths.

The component matrix uses 64×64 images, three frequency tasks, one kernel
thread and two imaging cycles. The controller's iteration total sums the two
polarization planes; `max_iter` is a per-plane budget. Product comparisons use
`rtol=1e-6` with a peak-scaled `1e-8` absolute floor (minimum `1e-14`) to allow
floating-point reduction order changes near zero. Reopened products must also
match the in-memory return, and PB-corrected intensity must equal restored
Taylor zero divided by the valid reference/effective PB.

## Additional boundary coverage

The three `test_continuum_edge_cases.py` modules (processing functions,
node tasks and distributed applications) add 96 cases:

- Reference-frequency precedence and fallback; scalar, empty, nonfinite and
  nonpositive inputs; invalid Taylor orders and incomplete model registrations.
- Empty Processing Sets and partially available weights; positional copies
  preserve destination coordinates without silently reindexing values.
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

The three `test_continuum_exceptions.py` modules add 107 cases using only
small in-memory NumPy/xarray objects. They create no files, add no fixture
archives and require no downloads. Each case checks the exception type and
an informative part of its message.

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
force otherwise unreachable guards. Existing complete imaging regressions
continue to use the already bundled fixture, unchanged.

## Execution variants and lifecycle regressions

An additional 102 cases cover all proposed follow-up areas:

| Module | Cases | Contracts |
| --- | ---: | --- |
| `unit/processing_functions/imaging/test_continuum_variants.py` | 45 | 32 PSF/beam/legacy-metadata layouts through the real restoration backend; missing backend output; invalid layouts; exact PB cutoffs; zero/extreme weights and single-channel/single-Taylor inputs |
| `unit/node_tasks/imaging/test_continuum_lifecycle.py` | 22 | Direct versus already-prepared finalization, exactly-once restoration, immutable accumulated model, cache replacement and repeated cleanup in Zarr v2/v3 |
| `unit/distributed_applications/imaging/test_continuum_variants.py` | 35 | Uneven frequency partitions, reordered inputs, unbalanced trees, zero-contribution leaves, corrupt upstream graph results, remaining geometry/metadata exceptions |

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
runs `tests/`, so no separate workflow is required. Existing Google Drive cube
references are unchanged.

## Harness audit and limits

The light harness supplied useful cases for unreduced PB rejection and
preserving task-local observed grids through a reduction tree; these now live
in repository tests. Other light-harness contracts were already represented,
and have been extended with failure and ownership checks. Medium-harness
storage/scheduler and persisted-image comparisons informed the component
matrix; its CASA assertions were deliberately not imported. Large datasets,
source-specific imaging behavior, scientific flux/noise/beam tolerances,
storage-volume benchmarks and runtime scaling remain outside this suite.

Coverage is not exhaustive. The eight-case matrix uses linear parallel hands,
double precision and the current Högbom continuum path. Existing tests cover
additional precision and scheduling paths separately; this is not the full
Cartesian product of telescope, polarization, precision, cache mode, scheduler,
weighting and Taylor order. No distributed multi-host or MPI guarantee follows
from local Dask tests. New supported functionality should add focused tests
rather than merely increasing the number of equivalent configurations.

## Measured coverage (2026-10-06)

On `RADPS-roadmap-194` at source commit `eca8472`, the original 135 dedicated
continuum tests passed. The 76 added cases also pass (68 unit, eight component).
The following comparison uses the same 11 continuum-named Python implementation
modules and the original suite plus these additions; it excludes shared modules
and does not measure native C++ coverage. Installed implementation files were
verified byte-for-byte against the checkout.

| Suite | Statement coverage | Branch coverage |
| --- | --- | --- |
| Original 135 cases | 2898/3452 (84.0%) | 1007/1562 (64.5%) |
| Original suite + 76 additions | 2937/3452 (85.1%) | 1039/1562 (66.5%) |
| Including 96 additional boundary cases (307 total) | 3040/3452 (88.1%) | 1128/1562 (72.2%) |
| Including 107 exception cases (414 total) | 3142/3452 (91.0%) | 1229/1562 (78.7%) |
| Including 102 execution/lifecycle cases (516 total) | 3190/3452 (92.4%) | 1275/1562 (81.6%) |

These percentages describe executed code, not exhaustive correctness. Several
new tests strengthen assertions on paths that were already executed, so their
value is not captured by the coverage increment alone. Remaining branches
include validation/error paths and less-used execution options.

Validation also ran the full repository unit suite plus the new component matrix:
1,721 passed, six skipped, 35 subtests passed. Eight final harness-derived
contract cases were added after that run collected its tests and are included
in the separately verified 76-case additions run. Ruff lint and format checks
pass. No imaging source code or existing reference data was changed.

The final combined continuum run, including the 96 boundary cases, passed all
307 tests in 221 seconds on smurf. Ruff lint and formatting checks passed.
The coverage row above comes from a fresh run of the entire dedicated suite,
using the same 11 implementation modules as the earlier measurements.

The subsequent combined run including the 107 exception cases passed all 414
tests in 221 seconds. These additions exercise 95 previously uncovered explicit
`raise` statements (194 remain uncovered, down from 289). They require no new
stored data or downloads. Ruff lint/format checks pass; imaging code and
existing data fixtures remain unchanged.

Before the storage fixes, the combined run passed 512 cases with four strict expected failures
in 231 seconds. The expected failures reproduce the two storage defects
described above; at that stage they were not passing correctness checks. Coverage includes
code executed by those cases before their expected failure. Ruff lint and
format checks pass for all four newly added Python files. No production code,
stored data fixtures or download requirements were changed.

After the storage fixes, the full dedicated continuum suite passes all 516
cases in 171 seconds, with no skips or expected failures. The 22-case lifecycle
module also passes separately. Both changed implementation modules were
verified byte-for-byte against the GCC-14-built installation in `viper`. Ruff
lint/format, parameter documentation sync and `git diff --check` pass.
The coverage table above records the pre-fix implementation and was not
remeasured for this storage patch.
