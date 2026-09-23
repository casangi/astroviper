# Cube reference images

The four TW Hydra truth stores were regenerated on 2026-09-22 from merged
commit `0180d98` plus the sidelobe position-angle correction. They include the
corrected angular beam, Gaussian-subtracted sidelobe estimate and resulting
CLEAN trajectories. `cube_reference_manifest.json` records the fitter source
hash, dependency versions, generation settings and archive SHA-256 checksums.

Each reference was generated with one processing thread and
`n_mapping_parallelism=1`, using the existing `_regenerate_truth_images()`
procedure. Dirty and niter=100 references use float64. The multi-cycle cases
have separate float64 and float32 references. Deep float32 comparisons use
the observable acceptance criteria described below.

## Local use

Use an environment installed from the current checkout with compatible
GraphVIPER/ToolVIPER versions (see the manifest). Regeneration and validation
here used isolated source/dependency overrides; the existing viper installation
was not reinstalled.

The regenerated `*truth.img.zarr` directories live beside
`test_single_field_imaging.py`. Run the component test from this directory so
its relative paths resolve. The input processing set is also present locally.
The `.zarr` stores and archives are ignored by Git; committing the manifest
alone does not distribute them.

## Publication pending

The Google Drive IDs in `_TRUTH_IMAGE_DRIVE_IDS` still point to the previous
reference release. The new archives have not been uploaded. A fresh checkout
without local stores will therefore still obtain the old images until those
published assets are updated.

Publication-ready archives and a copy of the manifest are staged on smurf at:

`/export/home/smurf/RADPS_test/psf_beam_pr270_proposal/cube_validation/reference_release/`

Upload those four archives to the project's reference-data host. If new Drive
file IDs are assigned, update `_TRUTH_IMAGE_DRIVE_IDS`; if the existing files
are replaced in place, their IDs can stay unchanged. Then verify a download
into an empty directory against the archive hashes and update the publication
status in the manifest. Existing local caches will need explicit replacement:
`_download_zarr()` intentionally does not overwrite an existing store.

## Deep-CLEAN precision acceptance

The corrected sidelobe estimator allows substantially longer model update
cycles (CASA minor cycles). Tiny PSF differences can then select different
near-tied CLEAN peaks. A pixelwise component-model comparison is consequently
not a reliable precision-equivalence criterion for this fixture. The image
and residual difference plots are retained for diagnosis.

For comparisons involving float32, every Stokes-I channel now requires:

- Full-image restored peak difference below the existing 15% of reference peak.
- Restored-image L2 difference within the common reference CLEAN mask below 15%
  of the reference L2 norm, guarding spatial structure as well as peak values.
- Integrated restored-image flux in that same mask within 15% of reference.
  Matching pixel and restoring-beam geometry make image sums proportional to
  the same flux units; beam variation is separately bounded at 1e-4 relative.
- Residual RMS in that same mask within 5% of reference. This is RMS about zero,
  not standard deviation after subtracting the mean and not a noise-only RMS.

Coordinates and masks must agree exactly. Primary beams remain bounded at
1e-6; PSFs at 1e-4; restoring-beam parameters at 1e-4 relative plus 1e-10
radians absolute. All model, residual and restored arrays must be finite.
Fresh deep runs must execute all four configured update cycles, spend the
10000-iteration budget for every plane, and return the iteration-limit stop
code with finite histories. The particular per-cycle allocation is not pinned
for float32. Stokes Q remains diagnostic in the image plots; the observable
image criteria retain the existing Stokes-I assertion scope.

These are fixture-specific regression bounds, not universal scientific or
QA2 acceptance criteria. They preserve the previous 15% restored-image limit
and add independent flux, morphology, residual-RMS and invariant checks.
Float64 comparisons still pin the full images at 1e-6 and the complete
reference deconvolution histories. Dirty and shallow tests are unchanged.

## Validation on the PSF PR

With fresh independent reconstructions and the regenerated local references,
the full cube component suite passes: **15 passed**. Ruff and diff whitespace
checks also pass. Additional diagnostic checks reject ten deliberate defects:
missing or rescaled restoration, image displacement, inflated residual RMS,
nonfinite model values, changed beam or mask, wrong iteration total, wrong
stop code, and nonfinite convergence history.

The reference publication step described above remains outstanding. A clean
checkout still downloads the old reference release until those assets are
published.

## Transport to the continuum branch (2026-09-23)

The component tests and four reference archives from PR commit `cd0d2e5`
were transported to `CASA_niter_convention` at `f036acc`. All archive SHA-256
checksums matched this manifest before extraction. Previous local reference
stores were retained as backups outside the repository. No reference values
or acceptance limits were changed during this transfer.

Against the current installed continuum implementation, fresh cube component
reconstructions give **9 passed and 6 failed**. The dirty-beam checks and all
deep-CLEAN precision comparisons now pass. The remaining failures are the two
`niter=100` model-history checks and four double-precision multi-cycle history
checks. The first channel's cycle counts are `[3, 84, 2818, 7095]`, compared
with the PR expectation `[3, 84, 2819, 7094]`.

The continuum branch retains CASA-inclusive Högbom iteration limits, unlike
the implementation used to generate these PR references. The remaining
trajectory differences are consistent with that difference; this transfer
has not established that the strict reference histories and images are valid
for the continuum branch. Its numerical references need separate validation
before replacing them. The earlier 15-pass result above applies to the PR,
not to this continuum validation.

Validation logs, XML results, verified archive extraction and backups are in
`/export/home/smurf/RADPS_test/cube_pr_transport_f036acc/`.
Reference publication remains pending.
