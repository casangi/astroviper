# Single-field continuum-imaging algorithm

This note describes the scientific conventions and distributed execution of
`image_continuum_single_field`. It complements the runnable
`single_field_continuum.ipynb` tutorial and defines the meaning of the Taylor,
primary-beam, and restored-image products produced by the application.

## Notation

Let

$$
x_\nu = \frac{\nu-\nu_0}{\nu_0},
\qquad
I_\nu(\boldsymbol\ell) \simeq
\sum_{t=0}^{N-1} I_t(\boldsymbol\ell)x_\nu^t,
$$

where $I_\nu$ is the intrinsic sky, $I_t$ are its Taylor coefficients,
$\nu_0$ is the reference frequency, and $N$ is `n_taylor_terms`. We use
$A_\nu$ for the frequency-dependent power primary beam, $F$ for the Fourier
transform, $S_\nu$ for visibility sampling, and $W_\nu$ for imaging weights.
The physical channel measurement equation is

$$
d_\nu = S_\nu F\left[A_\nu I_\nu\right] + n_\nu.
$$

For channel weight sum $s_\nu=\sum_j W_{j\nu}$, define the normalized channel
residual image and normalized channel PSF operator by

$$
R_\nu = \frac{1}{s_\nu}F^H S_\nu^H W_\nu r_\nu,
\qquad
B_\nu* = \frac{1}{s_\nu}F^H S_\nu^H W_\nu S_\nu F.
$$

All primary-beam divisions and pixel-local Taylor solves below are restricted
to valid primary-beam and spectral support.

## Distributed execution

Each imaging cycle has three scientific stages:

1. **Map:** frequency partitions independently load their visibility samples,
   apply the selected imaging weights, and grid either MFS Taylor UV products
   or channel-resolved MVC products. MVC map tasks also inverse transform their
   channel grids and construct additive Taylor contributions.
2. **Reduce:** additive products from all frequency partitions are summed.
   MFS reduces Taylor UV grids; MVC reduces Taylor residual, PSF, beam, and
   normalization contributions that no longer carry a frequency dimension.
3. **Append:** MFS reduced UV grids are inverse transformed and normalized;
   MVC reduced normal-equation contributions are finalized. The append node
   then performs one global Taylor-model update, prepares prediction state for
   the next cycle, or restores the final model after convergence.

The distributed driver forwards state between graphs but does not perform
image arithmetic. The model update, FFTs, primary-beam transformations, and
restoration remain in node tasks or processing functions.

## MFS convention

MFS models the apparent sky

$$
J_\nu = A_\nu I_\nu
\simeq \sum_t J_t x_\nu^t.
$$

Map tasks grid Taylor-weighted visibility and PSF contributions. Reduction
sums those contributions before one global inverse FFT and normalization. The
minor-cycle model coefficients `SKY_MODEL` are therefore the apparent-sky
coefficients $J_t$, and the MFS predictor evaluates their Taylor polynomial at
each channel before degridding it.

Consequently, $J_1/J_0$ generally contains both the intrinsic source spectrum
and primary-beam chromaticity. Intrinsic Taylor coefficients are recovered
during restoration by fitting

$$
A_\nu \simeq \sum_p A_p x_\nu^p
$$

and recursively dividing the apparent-sky polynomial by the beam polynomial:

$$
I_t = \frac{J_t-\sum_{p=1}^{t}A_pI_{t-p}}{A_0}.
$$

This division is applied separately to the model and principal-solution
residual coefficients before clean-beam convolution.

## MVC convention and exact major-cycle prediction

MVC stores the Taylor model in a common effective-primary-beam convention.
With

$$
\bar A = \frac{1}{S}\sum_\nu s_\nu A_\nu,
\qquad
S=\sum_\nu s_\nu,
\qquad
J_t=\bar A I_t,
$$

the MVC predictor is

$$
v_\nu^{\mathrm{MVC}} =
S_\nu F\left[
\frac{A_\nu}{\bar A}\sum_t J_tx_\nu^t
\right].
$$

Substitution of $J_t=\bar A I_t$ recovers the physical predictor
$S_\nu F[A_\nu\sum_t I_tx_\nu^t]$. Thus the MVC major cycle uses the physical
measurement operator independently of the selected minor-cycle approximation.

The exact negative gradient and Hessian in the common-beam variables are

$$
g_{J,t} = \sum_\nu s_\nu x_\nu^t
\frac{A_\nu}{\bar A}R_\nu,
$$

$$
[H_J\Delta J]_t =
\sum_{q,\nu}s_\nu x_\nu^{t+q}
\frac{A_\nu}{\bar A}
\left[B_\nu*\left(
\frac{A_\nu}{\bar A}\Delta J_q
\right)\right].
$$

This Hessian is direction dependent and is not an ordinary convolution. MVC
therefore provides two convolutional minor-cycle approximations.

### `channel_psf`

This is the Memo I/CASA-style approximation. It assumes that
$A_\nu/\bar A\simeq1$ over the useful beam support. The normalized Taylor
residuals are

$$
b_t = \frac{1}{S}\sum_\nu s_\nu x_\nu^tR_\nu,
$$

while the Taylor PSFs retain the primary-beam-weighted channel PSFs,

$$
B_k^{\mathrm{MVC}}* =
\frac{1}{S\bar A}
\sum_\nu s_\nu x_\nu^k A_\nu(B_\nu*).
$$

This approximation retains channel-dependent PSF structure. It is strongest
near the pointing centre, where the beam ratio is close to one, and is usually
the safer approximation for diffuse emission whose response includes extended
or frequency-dependent sidelobes. It can bias the spectrum of compact sources
towards the primary-beam edge, where the beam ratio is chromatic.

### `local_beam_moments`

This approximation retains local frequency-dependent primary-beam moments. It
uses the exact intrinsic-sky gradient numerator

$$
u_t=\sum_\nu s_\nu x_\nu^tA_\nu R_\nu,
$$

the pixel-local squared-beam matrix

$$
G_{tq}(\boldsymbol\ell)=
\sum_\nu s_\nu x_\nu^{t+q}A_\nu^2(\boldsymbol\ell),
$$

the beam-free spectral matrix

$$
C_{tq}=\frac{1}{S}\sum_\nu s_\nu x_\nu^{t+q},
$$

and the beam-free Taylor PSF matrix

$$
K_{tq}=\frac{1}{S}\sum_\nu s_\nu x_\nu^{t+q}(B_\nu*).
$$

It assumes that the primary beam is broad compared with the important PSF
structure,

$$
B_\nu*(A_\nu f)\simeq A_\nu(B_\nu*f),
$$

and that the normalized PSF response dominating the component update changes
slowly with frequency. These assumptions give

$$
H_I\simeq G C^{-1}K
$$

and the preconditioned minor-cycle system

$$
K\Delta J\simeq \bar A C G^{-1}u.
$$

The right-hand side supplied to MT-MFS is therefore
$\bar A C G^{-1}u$, paired with the beam-free Taylor PSF matrix $K$. Pixels
with too few supported channels retain only a Taylor-zero solution.

This approximation is intended primarily for compact emission, including
compact sources away from the pointing centre. It is less reliable for diffuse
emission, distant PSF sidelobes, strongly frequency-dependent PSFs, or poorly
conditioned spectral coverage near the primary-beam cutoff. Neither MVC
approximation is guaranteed for diffuse emission near the beam edge.

## Model update and residual restoration

Högbom updates Taylor term zero. MT-MFS jointly updates all Taylor terms using
the central Taylor-PSF Hessian. Its stored Taylor residuals are
normal-equation right-hand sides. Before restoration they are converted to
coefficient-space residuals by the principal solution

$$
R_{\mathrm{principal}}(\boldsymbol\ell)
=H_{\mathrm{centre}}^{-1}R(\boldsymbol\ell).
$$

The stored residual used for cycle control is not modified.

For MFS, the apparent model and principal residual Taylor polynomials are first
divided by `PRIMARY_BEAM_TAYLOR`. For MVC, the common-beam model and residual
coefficients are divided by $\bar A$. In both modes correction precedes
clean-beam convolution because spatially varying primary-beam multiplication
does not commute with that convolution. The distributed application writes
the intrinsic result as `SKY_RESTORED_FLAT_SKY`.

For restored intrinsic coefficients, spectral products are

$$
\alpha=\frac{I_1}{I_0},
\qquad
\beta=\frac{I_2}{I_0}-\frac{1}{2}\alpha(\alpha-1),
$$

and the propagated spectral-index error is

$$
\sigma_\alpha=|\alpha|
\sqrt{\left(\frac{R_0}{I_0}\right)^2+
      \left(\frac{R_1}{I_1}\right)^2}.
$$

These products are defined only where restored Taylor zero is finite, nonzero,
and exceeds one tenth of the peak Taylor-zero principal residual. Invalid
pixels are represented by NaN.

## Output products

Unless noted otherwise, image products use dimensions
`(time, taylor_term, polarization, l, m)`.

| Variable | Dimensions | Meaning |
|---|---|---|
| `SKY_MODEL` | Taylor image | Accumulated model. Apparent-sky coefficients for MFS; common-beam coefficients $\bar A I_t$ for MVC. |
| `SKY_RESIDUAL` | Taylor image | Taylor normal-equation residuals used for cycle control and model updating. MT-MFS restoration privately applies the principal solution. |
| `POINT_SPREAD_FUNCTION` | `(time, psf_taylor_order, polarization, l, m)` | Taylor PSF/Hessian planes; there are `2 * n_taylor_terms - 1` orders. |
| `PRIMARY_BEAM` | Common image plane | Reference-frequency beam for MFS or effective common beam $\bar A$ for MVC, according to its registered data-group role. |
| `PRIMARY_BEAM_TAYLOR` | Taylor image | Polynomial fit to the frequency-dependent beam. Used for MFS polynomial division; diagnostic for MVC because its channel beams already entered the MVC equations. |
| `BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION` | Beam-parameter planes | Fitted clean-beam major axis, minor axis, and position angle. |
| `SKY_RESTORED_FLAT_SKY` | Taylor image | Intrinsic, primary-beam-corrected restored Taylor coefficients. This is the restored product written by the distributed application. |
| `SKY_RESTORED_FLAT_NOISE` | Taylor image | Apparent/flat-noise restored coefficients available from the processing function for diagnostics; not written by the distributed application. |
| `ALPHA` | `(time, polarization, l, m)` | Dimensionless spectral index from corrected restored Taylor terms. |
| `ALPHA_ERROR` | `(time, polarization, l, m)` | Dimensionless propagated spectral-index error. |
| `BETA` | `(time, polarization, l, m)` | Dimensionless spectral curvature; available with at least three Taylor terms. |

Image brightness units are inherited from the image data model. Restored model
components are convolved with the fitted unit-peak clean beam, giving the
usual restored-image brightness convention.
