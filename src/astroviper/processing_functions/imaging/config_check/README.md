# Imaging configuration validation

Three independent validators own residual-update, model-update, and workflow
checks. `check_config` calls all three with the same configuration and mode.
A validator returns `None` on success and raises at the first invalid setting.
Checks are grouped and described in their owning `param_check_*` module.

Shared imaging rules live in `_predicates.py`: primary-beam cutoffs, Briggs
robustness, positive integer Taylor counts, the restoration selector and the
restoring beam. The three parameter-check modules
call these utilities independently. `_common.py` retains the basic type/numerical
checks and imaging-mode definitions.

`_spectral.py` validates selected spectral arrays, identifies positive-weight
channels, and builds the common weighted Taylor design. Spectral SVD and
approximate Hessian diagnostics retain their numerical acceptance tests and
error context in their owning checkers. Neither utility imports a parameter-check
module, inserts defaults, or modifies supplied configuration or arrays.

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

Continuum append preparation honors either the keyword flag or the dictionary
flag, without inserting a flag into the caller's mapping. For unchecked calls,
the append entry checks model/workflow settings and preparation checks residual
settings once; checks on actual reduced products always remain active.

Distributed drivers use `require_complete=True` to require image coordinates
and cycle controls before I/O, output creation, or graph construction. Lower
entry points validate the subset of settings their interfaces receive. Validation
does not insert defaults, rewrite aliases, clamp values, sort scales, or select
a different algorithm. Only settings declared optional use their documented
defaults without modifying the caller's dictionary. Explicit invalid
values, including `None` thresholds, raise.

`reference_frequency`, in Hz, is used consistently in configuration and continuum
metadata. `nterms` is a rejected configuration alias. Högbom continuum requires
exactly one Taylor term; MT-MFS
requires two to four. Cube planes are independent single-term images.

JSON schemas check basic structure, dtypes, and allowed strings. Python owns
numerical bounds and conditional relationships, including checks on direct
node/processing calls. Shared allowed-value rules intentionally also protect
Python entry points that have no schema decorator. These non-mutating checks
use `require_*` predicates; the legacy `check_params` helper inserts defaults
and is not interchangeable with them. The three validators cover geometry, frequencies,
weighting, algorithm compatibility, scales, gain, stopping controls, memory
policies, backend capabilities, restoration, and output compatibility.
Unsupported requested features raise, including multifield execution,
continuum MPI, continuum graph visualization, and cube global
weighting. The cube Dask graph visualization and cube MPI setup remain supported.
`field_name` is retained as a string or `None` in all driver interfaces. Delphi
uses it to determine the phase centre; the standard cube and continuum drivers
reserve it for future field selection. Input must already be single-field.
`mpi_cluster_setup` is retained on all drivers and ignored outside the cube MPI
backend. `primary_beam_limit` accepts the inclusive interval `[0.0, 1.0]`.

Cube and continuum restoration use the same `restoration_output` selector:

| Selector | Cube product | Continuum product |
| --- | --- | --- |
| `apparent` | `SKY_RESTORED` | `SKY_RESTORED_FLAT_NOISE` |
| `primary_beam_corrected` | `SKY_RESTORED_PRIMARY_BEAM_CORRECTED` | `SKY_RESTORED_FLAT_SKY` |
| `both` | Both cube products | Both Taylor stacks |

Cube workflows and direct restoration default to apparent output; the
continuum distributed driver defaults to corrected output. Imaging workflows
compute restoration products only when `restore=True`; the selector is still
validated when restoration is disabled. Neither workflow accepts the retired
`primary_beam_correction` Boolean. Cube correction divides model and residual
by the channel primary beam before convolution. MFS divides their Taylor
polynomials, while MVC uses its common beam. Corrected-only output does not
allocate an apparent restored cube. The selector controls intensity products;
the separate spectral-product convention is described by the restoration API.

`restoring_beam` is `None` (each plane keeps its own PSF fit) or a target
`[major, minor, pa]` in radians with `major >= minor > 0` and `|pa| <= pi` (a
larger position angle is in degrees). A string, tclean's
`'common'` among them, is refused with a pointer to `common_beam`. A target
requires `restore=True`, and its major axis must be smaller than the image,
`min(image_size) * min(|cell_size|)`: a larger one is arcseconds or degrees
given as radians, or a common beam inflated by a failed PSF fit. The drivers
check it against `image_params`, and direct calls of `restore_image` and
`restore_continuum_image` against the image they restore.
`beam_fit_params_sky_restored` can be kept only with `restore=True` and a
target. Whether the target contains a plane's PSF beam is known only after the
PSF fit: the cube restore keeps such a plane's residual unscaled and warns, and
the continuum append node refuses the target right after its PSF fit.

Output requests must match the selected workflow and restoration products.
Restored images and spectral products require `restore=True`; alpha products
need at least two Taylor terms and beta at least three. Continuum-only keys are
refused by cube imaging, and a continuum keep list cannot override
`restoration_output`. All these relationships are checked before graph or output
creation without changing the supplied configuration.

Numerical spectral and Hessian diagnostics use warning-level ToolVIPER log
messages, so worker diagnostics follow the software stack's logging controls.

The workflow checker also logs the MVC/Asp significance warning when significance
is enabled without a clean mask or an automatic mask. The diagnostic runs at the
unchecked entry boundary; checked downstream calls do not repeat it. Actual noise estimation
remains part of deconvolution.

The model-update checker owns `auto_mask_params`: known keys, values in range, and
no `clean_mask` beside it. Keys left out take the defaults of `make_auto_mask`, which
merges them without changing the caller's dictionary.

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
and fit within half the smallest image axis. No scale is dropped. Both `briggs`
and `briggs_abs` require `imaging_weights_params["robust"]` explicitly, as a finite
real number in `[-2, 2]`. An omitted value raises `KeyError`; no robustness value
is inferred or inserted into the caller's mapping. Absolute-Briggs noise must
be positive. `uniform` remains an
unchanged public weighting value and uses its production kernel recipe.

Natural weighting is independent of density scope and always uses the local
execution path. The scope resolver returns `local` for natural weighting even
when the supplied scope is `global`; it does not rewrite the caller's mapping.
Supplied scope values still undergo enum validation. No global weight-density
or Briggs-factor graph is constructed for natural weighting.

`clean_box=(xmin, xmax, ymin, ymax)` uses half-open pixel bounds, with x along
`m` and y along `l`. For `image_size=[n_l, n_m]`, require
`0 <= xmin < xmax <= n_m` and `0 <= ymin < ymax <= n_l`. Bounds must be integers;
invalid rectangles are rejected rather than clipped. `None` or the integer
tuple `(-1, -1, -1, -1)` selects the whole image.

## Coordinate-only spectral preflight

Polarization coordinates must be a nonempty one-dimensional vector of supported
Stokes labels. Lists and NumPy arrays are accepted; an xarray DataArray must
have exactly `dims=("polarization",)`. Cube and continuum apply this contract
at the shared configuration boundary before opening input data or creating
output. The checker neither renames dimensions nor changes caller coordinates.

After opening input metadata and making channel partitions, both distributed
imaging drivers check partition exclusivity, selected-channel coverage, and a
common Taylor basis. The node-layer metadata helper inspects coordinates,
attributes, shapes, and dtypes. It does not load visibility, statistical-weight,
flag, time-centroid, or UVW arrays, calculate imaging weights, or construct PSFs.

Preflight approximates every selected frequency sample with `q_nu = 1`. For
`x_nu = (frequency_nu - reference_frequency) / reference_frequency`, the Taylor
design is `A_nu,k = x_nu**k`. Its support is nominal: an available coordinate is
not evidence of usable data after flags and weighting. Diagnostics record
`weight_assumption="equal"` and
`support_assumption="selected_frequency_coordinates"` explicitly.

At least `n_taylor_terms` distinct selected frequencies are required. Rank is
assessed after combining repeated frequencies and scaling every design column
to unit Euclidean norm. With `D_kk = norm(A[:, k])` and `B = A D^-1`, the systems
`A c = b` and `B z = b`, `z = D c`, have the same exact rank. Normalization tests
column independence without treating a small high-order Taylor column as noise.
A zero column cannot constrain its coefficient and is rejected.

The SVD of `B` is computed in float64. Its numerical-rank threshold is
`max(M, n_taylor_terms) * eps64 * largest_singular_value(B)`, with `M` distinct
frequencies. The smallest singular value is the spectral-norm distance to loss
of full column rank. The threshold is a conventional approximation to SVD
roundoff resolution, not an uncertainty bound for the imaging pipeline, and is
independent of image storage precision. Repeated observations add weight but no
independent constraints: combining proportional rows preserves `A.T @ A` and
makes duplication of an entire input leave this rank decision unchanged.

`check_spectral_sampling` also accepts caller-supplied weights when those are
already available; it performs no I/O or weight preparation. Its original-basis
singular values and condition number describe physical Taylor coefficients and
are diagnostic, not rejection thresholds. Rank singular values, normalization,
float64 precision, and tolerance are recorded separately. Error context includes
field, spectral window, selected channels, polarization, Taylor order, and
reference frequency. Cube preflight checks nominal source-coordinate coverage
for each requested output channel using the production nearest-channel mapping.
Neither cube nor continuum metadata preflight determines positive-weight support.

### Order-zero spectral noise estimate

For independent equal-variance channel noise, let `h` be the first row of the
pseudoinverse of `A`. Under equal weights, the order-zero variance is
`sigma**2 * sum(h**2)`; a constant-only average has variance `sigma**2 / N`.
Their standard-deviation ratio is `G0 = sqrt(N) * norm(h)`.

More generally, for supplied `q = weight / max(weight)` and `W = sum(q)`, the
same independent unweighted-channel noise assumption gives
`G0 = W * sqrt(sum(q * h**2)) / sqrt(sum(q**2))`. Every original sample is
retained for this propagation; grouping frequencies for rank does not alter
the independent-noise model. With `A = B D`, use `A^+ = D^-1 B^+` to recover
`h` from a column-normalized SVD without forming normal equations.

Absolute flux and noise cancel. `G0 > 3` produces a warning, not a rejection.
Three is an explicit noise-amplification policy motivated by a three-sigma
residual target; it does not predict the final residual or CLEAN stopping depth.
With preflight unit weights, this is an equal-weight estimate only. Actual
weighting, flags, unequal noise, correlations, spectral model bias, and spatial
imaging effects are outside that early estimate. A reference outside the selected
frequency range also warns.

### Hessian diagnostics from supplied arrays

`check_taylor_hessian_approximation` is a standalone diagnostic for callers that
already have spectral weights and projected-baseline estimates. The distributed
coordinate-only preflight does not invoke it or collect those arrays. It checks
the column-equilibrated spectral Gram matrix and Gaussian spatial blocks using
its supplied inputs. Its local pivot/eigenvalue thresholds use the requested
working precision; noise amplification assumes covariance proportional to the
approximate Hessian. These are different assumptions from spectral `G0`.

Actual-product validation checks the measured PSF-derived Hessians once the
imaging tasks have formed them. Neither the coordinate rank test nor an optional
Gaussian diagnostic certifies those later products.

### Shared spatial-block Hessian diagnostic

For delta-component amplitudes, keep the spatially coupled operator
`H_(k,a),(l,b) = P_(k+l)(theta_a-theta_b)`. Its Taylor blocks are entire PSF
convolutions, not their central pixels. Under a periodic, shift-invariant boundary
model, the Fourier transform diagonalizes the spatial convolutions. The small
matrix at each spatial frequency is `H_kl(u) = Fourier(P_(k+l))(u)`.
`_spatial_hessian.py` applies the same diagnostic to both representations:

- **Gaussian preflight:** for channel `i`, take a peak-one circular Gaussian with
  `FWHM_i = c / (frequency_i * Bmax_i)` radians. `Bmax_i` is the longest projected
  baseline in metres with positive effective weight after flags and correlation
  selection. The caller supplies this information from already available data;
  obtaining it is outside coordinate-only preflight. Convert FWHM to `sigma_i`, and form
  the Taylor Fourier PSFs from
  `sum_i q_i * x_i**order * sigma_i**2 * exp(-2*pi**2*sigma_i**2*|u|**2)`.
  Each channel contributes only where `|u| <= Bmax_i * frequency_i / c`,
  including the boundary. Gaussian tails beyond that channel's longest
  baseline cannot represent measured information and are excluded before
  forming the Taylor moments. Modes outside all channel envelopes are
  unsupported, not conditioning failures. Within the combined envelope, a
  mode supported by too few channels can still have deficient Taylor rank.
  This circular outer envelope does not model UV holes or directional coverage.
  The Gaussian area factor is retained; only a common normalization cancels.
  No Gaussian cleaning components are introduced. Evaluate at most 33 by 33
  coordinates sampled from the image's half-Fourier grid. This is a coarse
  spatial approximation and may miss problems between sampled modes.
- **Actual PSFs:** remove index zero along each even spatial axis, leaving
  paired negative/positive lags about the peak at `size // 2`. Odd axes are
  retained unchanged. Center-to-origin shift and transform every Taylor PSF, then
  examine every stored half-Fourier mode for each time/polarization plane.
  This uses the retained PSF structure, including interactions between different
  delta-component positions. It assumes periodic convolution and does not model
  a restricted CLEAN mask or the exact finite-boundary subtraction operator.
  The imaging PSF is unchanged; only the diagnostic uses the paired-lag patch.

The common checks are:

1. **Spatial support.** Modes with `abs(H00(u)) <= nterms * eps * max(abs(H00))`
   are reported as unsupported and are not inverted. This is a numerical
   working-precision floor, not a measured sensitivity/noise threshold. Missing
   spatial modes are expected; the full pixel-space Hessian need not be
   invertible. Negative order-zero transfers above this floor remain visible
   to the curvature checks rather than being hidden as missing support.
2. **Self-adjointness.** Supported modes whose Taylor transfers have significant
   imaginary parts are reported as non-Hermitian. For actual PSFs, the absolute
   allowance per order is `eps * max(1, ceil(log2(N_pixels))) * sum(abs(PSF_order))`
   on the retained patch. The FFT has logarithmically many stages, and the
   unnormalized transform obeys `abs(FFT(E)) <= sum(abs(E))`. This gives an
   absolute scale even at Fourier cancellations; it is a roundoff estimate,
   not a rigorous bound on upstream gridding errors. Use the larger epsilon
   of the input and working precisions. Analytic transfers retain
   `nterms * eps * max(abs(transfer_order))`. Support and Taylor-rank thresholds
   remain independent. PSFs are not silently
   symmetrized; these modes are excluded from the symmetric-eigenvalue analysis.
3. **Taylor curvature/rank within supported modes.** Equilibrate each real block
   by its diagonal, then distinguish negative curvature from eigenvalues
   unresolved at `nterms * eps * largest_absolute_eigenvalue`. Record supported
   modes with deficient Taylor constraints separately from unsupported modes.
4. **Order-zero amplification.** On positive, full-rank blocks, assume
   `C_R(u) = tau**2 * H(u)` and calculate
   `G0(u) = sqrt(H00(u) * inverse(H(u))00)` from the equilibrated eigensystem.
   Record modes above three and the maximum with its spatial frequency and
   reciprocal spatial wavelength. This is an assumption-based per-mode noise
   ratio, not a prediction of integrated image noise or CLEAN stopping depth.

Spatial issues produce a diagnostic warning, not configuration rejection or
changes to the model/PSFs. This avoids treating the global pixel-space inverse
as the CLEAN algorithm. The local native inversion guards still reject unsafe
component updates independently. Gaussian models can be optimistic about missing
short spacings and sidelobes; actual PSFs can differ substantially.
The longest-baseline recipe is a resolution approximation, not a fitted restoring
beam: circular Gaussians omit beam ellipticity and weighting-dependent broadening.
The analytic Gaussian Fourier model also omits pixel-sampling alias corrections.

Preflight stores the result under
`spectral_design.taylor_hessian_approximation.spatial_blocks`. Actual-PSF results
are under `model_products.taylor_hessian_spatial_blocks`, with time/polarization
identifiers. Summaries contain counts, at most four example locations per issue,
and the worst amplification; no full spatial Hessian is stored. Frequencies are
in cycles per radian when angular cells are known, otherwise explicitly in
cycles per pixel. The zero-frequency mode has no finite reciprocal wavelength.

The actual check requires one real FFT per Taylor PSF, stores a half-Fourier
stack, and assesses at most 4096 small Taylor blocks at once. It never allocates
the square matrix over all image pixels and Taylor terms. Existing native
deconvolution kernels and the local cleaning update are unchanged.

## Product checks and diagnostics

New products are checked before first use even when `input_checked=True`, and
before a stopping decision can bypass CLEAN. Residual/model Taylor counts are
exactly `nterms`; PSF counts are `2*nterms-1`. Geometry, coordinates, masks, working
precision, and derived per-plane controls must agree. Every time/polarization
plane's per-scale Taylor Hessians are checked before any model plane is changed.
They must be positive definite and numerically invertible at working precision;
order-zero noise amplification above three warns. Native array and Hessian
boundary guards remain active.

### Three local Hessian checks

1. **Native inversion safety.** MT-MFS solves `H * delta_c = R` on every scale.
   The same native setup used by CLEAN attempts Cholesky factorization before
   any model plane changes. It rejects pivots at or below `100 * eps * H_ii`,
   relative to each original diagonal so small higher-order moments do not fail
   solely because of their magnitude. The factor 100 is the existing native
   round-off safety margin, not a scientific accuracy target. An unsafe inverse
   is unacceptable even if order zero would otherwise be well constrained.
2. **Positive definiteness and numerical rank.** With `D = diag(H)`, check the
   eigenvalues of `D**(-1/2) H D**(-1/2)`. This congruence preserves positive
   definiteness while removing differences in Taylor coefficient scales. Every
   coefficient direction must have positive curvature; eigenvalues at or below
   `nterms * eps * largest_absolute_eigenvalue` reject the full Taylor solve.
   This threshold is a dimension-scaled round-off heuristic on the equilibrated
   matrix, not an accuracy guarantee for the unscaled native solve.
3. **Order-zero noise amplification.** A large full-Hessian condition number can
   reflect higher Taylor terms with little effect on reference intensity. For
   residual noise covariance `C`, the order-zero variance is
   `(H**(-1) C H**(-T))_00`. A constant-only update has variance
   `C_00 / H_00**2`. Under the explicit assumption `C = tau**2 * H`, their
   standard-deviation ratio simplifies to `G_H0 = sqrt(H_00 * (H**(-1))_00)`.
   Both entries are already available: no extra image pass or inversion is
   needed. Warn when `G_H0 > 3`. As in spectral preflight, three is a permitted
   noise-amplification factor motivated by a three-sigma residual target, not
   an equivalence to CLEAN stopping depth or a prediction of final residuals.
   The covariance proportionality is assumed, not measured: arbitrary imaging
   weights and scale filtering need not satisfy it. The metadata and warning
   identify this assumption explicitly.

Workflow results and persisted image metadata retain `input_validation`:

- `spectral_design` records SVD rank, singular values, precision, usable selection,
  `spectral_design_condition_number` (2-norm), and
  `spectral_design_reciprocal_condition_number`, together with
  `taylor_zero_noise_amplification` and its
  `taylor_zero_noise_amplification_limit` (3.0).
- `model_products` records the actual geometry, precision, reference, effective
  scales, controls, and `taylor_hessian_condition_number` per plane and scale
  (the native infinity-norm estimate, retained for inspection). Warnings now use
  `taylor_hessian_order_zero_noise_amplification`, which records per-scale
  `G_H0`, its limit (3.0), and `covariance_assumption=proportional_to_hessian`.

These metrics describe different matrices and norms. They are deliberately
stored under separate names. Derived controller limits, dataset selections, and
automatically evaluated beam recipes remain runtime state. Caller configuration
is preserved, and shared configuration is checked for consistency across stages.

Sparse XRADIO padding is excluded only when the absent time-centroid, all-NaN
UVW, and all-NaN weights agree. Real rows with invalid weights still raise.
Channel matching uses a sorted private lookup with linear memory, and cube
sampling diagnostics retain only each channel's own selection.


## Configuration checks and runtime guards

Static configuration predicates have one implementation in this package.
Imaging entry points validate direct calls and propagate `input_checked=True`
after validation. Legacy controller, beam, and gridding helpers delegate to the
same predicates rather than keeping independent copies. The standalone
controller retains its documented acceptance of integral float cycle counts;
distributed configuration still requires integers.

Output-layout validation uses the shared generic implementation in `utils.io`,
which also serves direct writers without introducing an upward layer dependency.
Before mapping, it checks known image extents and explicit chunks; after mapping,
it additionally checks task alignment and inferred chunk sizes. FITS direct
calls enforce the same representable-coordinate policy as preflight: irregular
channels are rejected rather than approximated.

Checks of actual products are still required: source roles and units, cache
contents, residual/model/PSF geometry and precision, normalization arrays,
spatial Hessians, and native memory layout. A valid configuration cannot prove
that these runtime objects satisfy their contracts. Metadata-driven beam
inference and the cached numerical kernel-edge test are single shared
implementations called during preflight, not duplicate configuration predicates.

## Retired legacy checkers

`processing_functions.imaging.check_imaging_parameters` retains only
`check_imaging_weights_params`, the active compatibility wrapper delegating to
`check_residual_update`. Its parameters are validated without inserting defaults.

The following functions had no repository callers, tests, documentation examples,
or exports from the imaging package, and have been removed:

- `check_grid_params`
- `check_gcf_params`
- `check_mosaic_pb_params`
- `check_rotation_params`
- `check_norm_params`
- `check_pb_params`

Direct imports of these legacy functions must migrate to the appropriate
residual-update, model-update, and workflow validators (or `check_config` for
all three). Their old dictionaries, default insertion, and unit conversions are
not a compatibility interface: callers must supply the canonical imaging
configuration. This retirement does not remove the general-purpose
`astroviper.utils.check_params` utility, which still has active callers.

## Feather

Feather has a separate validation entry point, without a cube/continuum mode.
`check_feather_config` checks input paths and output policy before image access;
`check_feather_inputs` checks the shapes and dtypes of opened `SKY` products
without loading pixels. The driver retains file access, existence checks, and
chunking. A missing overwrite key is read as `False` without changing `outim`.

Output layout is checked from supplied coordinates at entry and again with the
constructed image dimensions and task boundaries when these become available.
Both stages use the same layout validator. Continuum beam metadata validation
returns a resolved copy for reuse across tasks and cycles; resolving a beam on
a direct helper call still applies its own applicable checks.

## Runtime Taylor-Hessian reuse

The model-update processing function checks current products once per update.
The append node checks products itself when convergence skips that update.
Dimension/basis/precision, model, mask and derived-control checks always run;
`input_checked` never skips these runtime checks.

Expensive PSF/Hessian diagnostics may reuse an explicit workflow-owned
`hessian_cache`. A content digest and coordinate labels identify the actual PSF;
Taylor count, precision, image geometry, reference frequency, scales, scale bias
and cell sampling also participate in the key. Changed inputs force fresh
checks, including in-place PSF edits and replacement products after worker
transfer. A linear PSF read avoids repeating scale convolutions and spatial
Fourier diagnostics. Only successful checks populate the cache. Cached
statistics are copied into output diagnostics; the cache remains separate from
image metadata, and native cleaner safety checks remain active on every update.
Direct calls without a cache recompute all diagnostics.

After configuration validation, cube and continuum drivers check destination
existence before opening the input or running metadata preflight. This read-only
filesystem check lives in `utils.io`, follows XRADIO's actual Zarr naming policy,
and checks the cube FITS output directory as supplied. It leaves existing files
untouched, including when overwrite is enabled. Writers retain their final
existence checks to catch destinations created after preflight.

Distributed single-field drivers require explicit FFT padding and exactly one
image time coordinate. The default designed kernel needs padding >= 1.05;
CASA's fixed kernel also permits 1.0. These checks precede input I/O. Standalone
processing functions retain their supported multi-time array contracts.
MFS can publish its Taylor UV grids; an explicitly requested sampling grid
remains in the static cache for final publication after cleaning. Ordinary runs
retain no extra grid buffers. Cube and MVC reject UV-grid disk requests before
creating output. MVC does not silently omit requested channel grids.

Continuum automatic cycle limits are resolved from the first dirty Taylor-zero
image before convergence is checked. Positive thresholds require no extra
weight-statistic work. For zero thresholds, first-cycle map tasks accumulate
small additive correlation-weight sums from their loaded weights and flags.
After global reduction, the existing sensitivity formula supplies the reference
noise: correlation variance is sum(W**2/w)/sum(W)**2 and Stokes variances follow
the instrument-to-Stokes transform. Combining sums before the ratio makes the
result independent of frequency partitioning. No visibility/UVW reads or driver
sample scan are added; later cycles reuse the resolved controller limit.

`continuum_finalize_node` validates supplied model and workflow settings on
direct calls, including the deconvolver, restoration selector and preparation
switch. Unprepared images receive residual-configuration checks during actual
preparation. A checked caller may supply the keyword or dictionary trust flag;
finalization forwards it without changing the caller's mapping. Accumulated
model/static-state checks and restoration's actual-product checks always run,
including when the configuration was already validated.
