# Merging main into the continuum branch — 2026-09-23

## Scope

Merge `origin/main` at `88758c3` into `CASA_niter_convention`, whose pre-merge
head is `565b314`. The latter already contains the sidelobe Gaussian
position-angle correction and its 30 regression cases. This merge preserves
continuum-specific behavior while integrating the newer cube implementation.

Eight files had textual conflicts. Resolutions were reviewed individually;
neither branch was selected wholesale. Related call sites and regression tests
were updated where an interface needed to support both workflows.

## Conflict resolutions

| File | Resolution and rationale |
| --- | --- |
| `src/astroviper/processing_functions/imaging/calculate_imaging_weights.py` | Pass both `truncate_uv_cells` and `frequency_map` to weight gridding and degridding. Retain local imports and add the frequency mapper there. Introduce `frequency_matching` to select exact or nearest matching independently of the UV-cell convention. |
| `src/astroviper/processing_functions/imaging/degrid_visibility_grid.py` | Combine main's polarization-axis validation with continuum's allocation fallback. If observed visibilities were deliberately not loaded for a cached-grid cycle, registered imaging weights provide the shape/dimension template for allocating complex128 model visibilities. Preserve main's fuller documentation and shared-parameter decorator. |
| `src/astroviper/processing_functions/imaging/fft_normalize_prolate_spheriodal_gridder.py` | Both conflict blocks contained only comments/whitespace. Keep main's fuller explanation of in-place division, precision preservation, and avoiding large temporary arrays. An AST comparison verified that choosing this documentation did not change executable statements. |
| `src/astroviper/processing_functions/imaging/get_visibility_grid.py` | Keep the fuller accurate documentation, including the correct `visibility` data-group role. Retain the continuum wrapper's explicit contiguous-grid preparation and delegate the default output mapping to the shared degridder. Pass the resulting `grid` to that primitive. Its additional contiguity conversion is a no-copy operation when the grid is already contiguous. |
| `src/astroviper/processing_functions/imaging/imaging_weighting/grid_imaging_weights.py` | Preserve UV-cell truncation, continuum's `channel_map` gridding argument, main's `frequency_map` in both operations, and native-byte-order/contiguous floating-point normalization before weight degridding. Both map names use the same validation. Reject gridding calls that provide both names, even if the arrays agree. Preserve existing continuum positional arguments; the added `frequency_map` argument is keyword-only. |
| `src/astroviper/processing_functions/imaging/utils/frequency_mapping.py` | Provide explicit `matching="exact"` and `matching="nearest"` policies. Preserve the former continuum one-to-one algorithm and main's nearest-channel algorithm, including finite/nonempty input validation. Audit and update callers so each workflow selects the intended policy. |
| `tests/unit/processing_functions/imaging/test_fft_ifft.py` | Keep both independently added test groups: no interactive plotting during FFT processing, and precision/in-place/input-preservation checks. No test group was discarded. |
| `tests/unit/processing_functions/imaging/test_imaging_weights.py` | Keep all tests from both parents. Adapt the degridding mock to accept `processing_function_threads`, `truncate_uv_cells`, and `frequency_map`. |

For documentation-only conflicts, the rule was to preserve the fullest
accurate explanation. For test conflicts, preserve both branches' coverage.

## Frequency matching policy

### Continuum

`matching="exact"` requires each visibility frequency to match exactly one
image frequency within the existing numerical tolerances (`rtol=1e-12`,
`atol=0`). No two visibility channels may map to the same image plane.
Sparse and reordered subsets are supported. Missing, ambiguous, shifted
beyond tolerance, and many-to-one matches are rejected.

This is numerical equality within tolerance, not a bitwise floating-point
comparison. It does not prevent intentional collapse into a continuum
weight-density plane: that separate operation explicitly supplies an all-zero
channel map to the weight gridder.

MVC visibility gridding explicitly requests exact matching. Continuum setup
passes `frequency_matching="exact"` when calculating local imaging weights;
that setting is used for both weight gridding and degridding. Existing global
continuum weighting paths retain their own channel-selection behavior.

### Cube

`matching="nearest"` preserves main's nearest-channel mapping, normally
limited to half an image-channel width. It permits several visibility
channels to map to one image plane. Main's single-channel conventions are
preserved: visibility spacing supplies the tolerance for a one-channel image,
and when both axes contain one channel, that sole plane is selected.

Cube visibility gridding, PSF gridding, and model prediction select nearest
matching explicitly. Cube setup passes `frequency_matching="nearest"` for
imaging weights. The helper and shared weighting function also default to
nearest matching for compatibility with main's callers.

## Relationship to the separate PRs

At the time of this merge:

- PR #276 (shared degridding primitive) had already merged into main. Its
  incoming changes include the frequency-map support reconciled here.
- PR #277 (contiguous model grids) remained open. Its proposed wrapper
  conversion overlaps the protection now also provided by the shared
  primitive. We retain the continuum wrapper conversion for now; future
  integration can reconcile the duplication without losing the regression.
- The other open PRs did not modify the weighting/frequency-mapping files.
  Their eventual merge therefore would not, by itself, resolve those conflicts.

The workflow remains to review the extracted cube-relevant PRs individually,
merge them into main, and subsequently merge main into this continuum branch
again. This merge does not assume that open PR changes are already present.

## Validation performed during resolution

These are separate focused runs, with overlapping coverage; their counts
must not be added together as one full-suite result.

| Focused run | Result |
| --- | --- |
| Resolved shared degridder plus existing continuum allocation tests | 8 passed, 2 subtests passed |
| Resolved wrapper/shared degridder plus continuum allocation tests | 18 passed, 2 subtests passed |
| Weight wrappers and C++ weighting kernel, including new interface regressions | 32 passed, 3 subtests passed |
| Frequency policies, cube wrapper/PSF/gridding, MVC and weighting integration | 34 passed, 12 subtests passed |
| Both resolved FFT and imaging-weight test files | 42 passed, 2 subtests passed |

New weighting regressions cover both map names, channel collapse, ambiguous
argument rejection, invalid maps, simultaneous frequency mapping and UV-cell
truncation, and strided non-native-byte-order density grids. Frequency tests
cover exact tolerances, unique/reordered selections, duplicate and many-to-one
rejection, shifted cube acceptance, shifted MVC rejection, and weighting's
selected policy. All test function names from both parents of the final two
test-conflict files were retained.

Because tests were run while other files remained conflicted, validation
loaded the relevant working-tree modules with isolated dependencies. The
installed weighting extension lacked `truncate_uv_cells`, so the current
branch's extension was built separately in `/tmp` using GCC 14 and loaded
under an isolated name. The shared viper installation was not modified.
Pytest emitted import-rewrite warnings for already imported plugins.

Targeted Ruff, formatting, whitespace, and shared-parameter documentation
checks passed during resolution. These focused runs do not replace a full
build and test campaign against the completely merged package.

## Remaining work

1. Build/install the complete merged implementation in the intended test
   environment and run the broader unit/component and continuum harness tests.
2. Reassess numerical expectations and regenerate reference images for this
   branch's actual iteration behavior where justified. The previous PSF PR's
   references and deep-CLEAN acceptance changes were not copied wholesale.
3. Publish any newly validated reference assets separately; local ignored
   Zarr stores are not distributed by a Git commit.

Completing this merge commit does not imply that those follow-up tests or
reference updates have already been performed.
