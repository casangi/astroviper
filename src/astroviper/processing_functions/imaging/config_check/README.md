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
precision. A full-rank design warns when fitting the additional Taylor terms
amplifies order-zero noise by more than three relative to a constant-only fit.
A reference outside the usable frequency range also warns.
Errors include field, spectral window, selected channels, polarization, Taylor
order, reference frequency, usable counts, rank, and conditioning context.
Cube preflight requires positive effective weight in every requested channel.

### Order-zero spectral noise diagnostic

The smallest singular value can describe a poorly constrained slope or curvature
without implying a poorly constrained order-zero intensity. We therefore target
order zero directly. Let `q = weight / max(weight)`, `W = sum(q)`, and let `h` be
the first row of the pseudoinverse of the weighted Taylor design `A`. Then
`delta_c0 = h @ delta_b`. For independent unweighted channel noise with equal
standard deviation `sigma`, the order-zero variance is
`sigma**2 * sum(q * h**2)`. The constant-only weighted average has variance
`sigma**2 * sum(q**2) / W**2`. Their standard-deviation ratio is

`G0 = W * sqrt(sum(q * h**2)) / sqrt(sum(q**2))`.

Absolute flux and noise cancel. The first pseudoinverse row is calculated from
the thin SVD already used for rank checking; no visibility reads or extra
weight/flag passes are required. We avoid normal equations, which square the
condition number. The threshold `G0 > 3` produces a warning, not a rejection.
Three is an explicit noise-amplification policy motivated by a three-sigma
residual target; it does not establish equivalence to CLEAN stopping depth or
predict the final image residual. Unequal channel noise, correlations, spectral
model bias, and spatial imaging effects are outside this diagnostic. Numerical
rank deficiency still rejects the requested full Taylor fit separately.

### PSF-free Hessian approximation at the distributed entry point

Before output creation and imaging-graph execution, MT-MFS preflight also calls
`check_taylor_hessian_approximation` using the selected frequencies and effective
weights already collected above. Local guards use the point-scale spectral
estimate `H_hat = A.T @ A`, or `H_hat_kl = sum(q_i * x_i**(k+l))`. Spatial
interactions between delta components are assessed separately with Gaussian
channel PSF blocks, as described below. There are no visibility reads, additional
metadata passes, or image-sized preflight allocations.

The early checks mirror the later Hessian checks:

1. Reject an approximate Cholesky failure or relative pivots at or below
   `100 * eps`, matching the native diagonal-relative safety margin.
2. Reject eigenvalues of the equilibrated approximation at or below
   `nterms * eps * largest_absolute_eigenvalue`, because the approximate full
   Taylor solve is numerically unresolved at the requested working precision.
3. Warn when `sqrt(H_hat_00 * inverse(H_hat)_00) > 3`, assuming residual
   covariance proportional to `H_hat`. Flux and absolute noise cancel.

To avoid squaring the raw Taylor dynamic range, normalize the columns of `A`
before forming the equilibrated Gram matrix. Compute order-zero amplification
with a Cholesky solve rather than an explicit inverse. These local checks do not
describe spatial coupling; the additional Gaussian block diagnostic does.
Passing preflight does not certify a later actual Hessian: Gaussian PSFs omit
sidelobes, sampling gaps, gridding effects, and per-plane differences hidden by
weight aggregation. All actual-PSF checks remain active in the append node.

The result is stored as `spectral_design.taylor_hessian_approximation`, including
the approximation name, equilibrated matrix, pivots, eigenvalues, thresholds,
noise amplification, and its explicit covariance assumption. This approximation
uses a different noise assumption from spectral `G0`, which assumes equal noise
in the unweighted channels. Cube and single-term Högbom workflows do not request
this additional MT-MFS approximation.

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
  selection. It is collected in the existing metadata pass and combined across
  Measurement Sets at the same frequency. Convert FWHM to `sigma_i`, and form
  the Taylor Fourier PSFs from
  `sum_i q_i * x_i**order * sigma_i**2 * exp(-2*pi**2*sigma_i**2*|u|**2)`.
  The Gaussian area factor is retained; only a common normalization cancels.
  No Gaussian cleaning components are introduced. Evaluate at most 33 by 33
  coordinates sampled from the image's half-Fourier grid. This is a coarse
  spatial approximation and may miss problems between sampled modes.
- **Actual PSFs:** center-to-origin shift and transform every Taylor PSF, then
  examine every stored half-Fourier mode for each time/polarization plane.
  This uses the full PSF structure, including interactions between different
  delta-component positions. It assumes periodic convolution and does not model
  a restricted CLEAN mask or the exact finite-boundary subtraction operator.

The common checks are:

1. **Spatial support.** Modes with `abs(H00(u)) <= nterms * eps * max(abs(H00))`
   are reported as unsupported and are not inverted. This is a numerical
   working-precision floor, not a measured sensitivity/noise threshold. Missing
   spatial modes are expected; the full pixel-space Hessian need not be
   invertible. Negative order-zero transfers above this floor remain visible
   to the curvature checks rather than being hidden as missing support.
2. **Self-adjointness.** Supported modes whose Taylor transfers have significant
   imaginary parts are reported as non-Hermitian. The tolerance is
   `nterms * eps * max(abs(transfer_order))` for each order. PSFs are not silently
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
