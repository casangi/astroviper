# Image quality metrics

AstroVIPER computes the image quality metrics behind the CASA pipeline's
imaging quality assessment: measurements made on the images while they exist.
References are to the CASA pipeline at commit 0091d599c with CASA 6.7.6, with
paths relative to the pipeline repository root.

The metrics are telescope neutral. A number that defines how something is
measured, such as the primary beam levels of the annulus or the continuum
channels, is passed in by the caller and has no telescope default. AstroVIPER
computes no scores: turning metrics into scores, thresholds and messages is the
caller's job.

## Metrics

| AstroVIPER | Pipeline | Pipeline code |
| --- | --- | --- |
| `primary_beam_annulus_limits` | `pblimit_image`, `pblimit_cleanmask` (`pblimits`) | `pipeline/hif/heuristics/imageparams_base.py:1533-1569`, called at `pipeline/hif/tasks/tclean/tclean.py:1204-1206` |
| `calculate_primary_beam_annulus_statistics` | the per channel `ia.statistics` of the image without primary beam correction in the annulus | `pipeline/hif/heuristics/cleanbox.py:219-332` |
| `summarize_primary_beam_annulus_rms` | `image_rms`, `image_rms_min`, `image_rms_max` and the per Stokes `image_rms_iquv` | `cleanbox.py:339-375`; the ALMA image score divides `image_rms` (Stokes I images) or `image_rms_iquv[0]` (other images) by `dr_corrected_sensitivity` (`pipeline/hif/tasks/tclean/qa.py:113-120`), the VLA continuum score is a linear score of the maximum of the corrected image over `image_rms` (`qa.py:77-79`), and VLASS images and VLA cubes score 1.0 (`qa.py:64-75`) |
| `chauvenet_sample` | casacore `ChauvenetCriterionStatistics` with `zscore=-1` and `maxiter=5` | casacore 3.6 `scimath/StatsFramework/ChauvenetCriterionStatistics.tcc:99-130`, `ZScoreCalculator.cc:38-117` |

The first three functions are in
`src/astroviper/processing_functions/image_analysis/primary_beam_annulus_statistics.py`,
the last in `src/astroviper/processing_functions/image_analysis/statistics/chauvenet.py`,
which the automatic mask and the continuum finders share.

## Definitions

### Annulus levels

`primary_beam_annulus_limits(img_xds, nominal_annulus_limits)` reads the
primary beam at time 0, frequency index `n_f // 2`, polarization 0 and `l`
index `n_l // 2`, along `m` from index 0. XRADIO orders pixels as CASA does
(the `l` index is CASA's x, and `m` index 0 is CASA's y = 0), so this is the
pipeline's sample `pb[nx // 2, i, 0, nf // 2]`. When the value at `m` index 0
is above the nominal lower level, compared in double precision, the levels are
the values `n_m // 20` and `2 * (n_m // 20)` pixels from the edge (the
pipeline's `int(0.05 * ny)`); otherwise they are the nominal levels. The
pipeline passes `(0.2, 0.3)` and tests the edge against 0.2. When the edge is
above the lower level, an image with fewer than 20 pixels along `m` gets equal
levels, an empty annulus, as in the pipeline.

The pipeline also keeps its nominal levels when the edge pixel is masked, and
otherwise walks past zero pixels to the first nonzero one. `tclean` masks and
zeroes the primary beam at and below `|pblimit|`, which is at most 0.2 in every
pipeline imaging mode, so neither branch changes the result; here a zero or NaN
edge keeps the nominal levels. Some VLASS modes replace the lower level by a
constant (`pipeline/hif/heuristics/imageparams_vlass_single_epoch_continuum.py:305-312`),
which is the caller's choice.

The function reads the middle channel of the dataset it is given. A block of
channels has its own middle channel, so callers pass the whole cube or its
middle channel.

### Annulus statistics

`calculate_primary_beam_annulus_statistics(img_xds, annulus_limits, mask_union)`
measures every `(time, frequency, polarization)` plane of the image of the data
group (default `restored`) in the pixels where the plane's own primary beam
lies strictly between the two levels, compared in double precision as in
CASA's mask expressions, and where the image is finite. The pipeline measures
the restored image of its last `tclean` run without primary beam correction
(`.image`, or `.image.tt0` for multi-term imaging; `cleanbox.py:33-36, 155-162`).

In memory, the cube imager's `restored` group names that image, its primary
beam and its mask, unless `restoration_output="primary_beam_corrected"`, which
keeps only the corrected image. In an image store the `restored` group names
only the restored image, the primary beam is in the `residual` group, and the
mask is written only when `image_data_variables_keep` asks for it; to measure a
stored image, a caller adds a data group that names the restored image, the
primary beam and the mask.

The selected pixels are clipped with Chauvenet's criterion and reduced to
CASA's statistics: `n_pixels` [`npts`]; `rms`, the square root of the mean
square [`rms`]; `standard_deviation` with `n - 1` degrees of freedom [`sigma`];
`median`; and `median_absolute_deviation`, the unscaled median of
`|x - median|` [`medabsdevmed`]. The pipeline's robust RMS is
`1.4826 * medabsdevmed` (`cleanbox.py:355-356`).

Each plane is measured twice: in the whole annulus, and outside the clean mask
(suffix `_outside_mask`). With `mask_union=False` each plane excludes its own
mask, which is what a node task holding a block of channels can compute. With
`mask_union=True` every plane excludes the union of the masks of all planes:
the pipeline's clean mask collapsed with `max` over Stokes and channels
(`cleanbox.py:222-237`), outside where it is below 0.1. Without a mask both
variants are those of the whole annulus, which the pipeline then measures
(`cleanbox.py:312-317`). A plane without kept pixels has `n_pixels` 0 and NaN
statistics.

The mask must be a clean mask. Without a user or automatic clean mask the cube
imager's `mask` role names a primary beam mask, true wherever the primary beam
is above `primary_beam_limit` times its peak, while the pipeline measures the
whole annulus of an image cleaned without a clean mask (`cleanmask` `''` or
`'pb'`); callers leave the role out for such an image.

### Summary over channels

`summarize_primary_beam_annulus_rms(annulus_statistics, channel_indices)` takes,
for each time and polarization, the RMS outside the clean mask, or that of the
whole annulus when the median over all channels of the pixels kept outside the
mask is below 10, as when no channel keeps one (PIPE-541,
`cleanbox.py:297-310`). As in the pipeline, statistics without a mask never
switch (`full_annulus` is `False`). It returns the median, minimum and maximum of that RMS
over the channels of `channel_indices` (`cleanbox.py:339-362`). A repeated index
counts repeatedly, as the pipeline's `np.hstack` of its channel ranges keeps
repeats. A channel without kept pixels counts as an RMS of 0, as CASA reports
it, and the result is NaN when no channel keeps a pixel even in the whole
annulus. To reproduce the pipeline on a cube, the statistics of every channel
are computed with `mask_union=True`.

The caller maps the pipeline's continuum frequency ranges to channel indices
(`pipeline/infrastructure/utils/imaging.py:111-170`, used at
`cleanbox.py:339-353`), and passes every channel when the ranges are `ALL`,
`ALLCONT` or `NONE` and for a continuum image, which has one channel.

### Chauvenet's criterion

`chauvenet_sample(values)` makes up to six passes (`maxiter=5`). Each removes
the values farther from the mean than `z` standard deviations (`n - 1` degrees
of freedom), where `n * erfc(z / sqrt(2)) = 0.5` for the `n` values the pass
starts from. The passes stop when one removes nothing or fewer than two values
are left. The arithmetic is done in the dtype of the values; the annulus
statistics pass float64.

## Departures from the pipeline

- **z-score.** `chauvenet_sample` uses the root `z(n)`. casacore finds `z` by
  bisection from a table and caches it for the whole process, so its `z` is a
  value in `[z(n), z(n + 1))`, which reaches 0.087 above `z(n)` for 5 values,
  0.040 for 10 and 0.0016 for 185 (for 10 values casacore uses 2.0, where
  `z(10) = 1.960`). A value between the two ranges is removed here and kept by
  CASA, and the later passes then start from a different mean and standard
  deviation. On 1200 CASA planes of Student t noise, 877 of them with 1 to 830
  annulus pixels, 5 planes, all of at most 84 pixels, keep 1 to 5 pixels fewer
  than CASA, and their RMS differs by 4 to 85 percent. In random samples of 1000
  values the kept values differ by at most one and the RMS by less than 1
  percent. Fewer pixels kept outside the mask can bring their median count
  below 10, so that the summary takes the whole annulus where the pipeline
  does not.
- **Nested passes.** Each pass filters the values the previous one kept, where
  casacore applies each range to all values. The two differ only when a later
  range reaches a value an earlier one removed; no difference occurs in 200000
  skewed random samples of 6 to 44 values.
- **Polarizations pooled.** For an image with several Stokes planes, the
  pipeline's `image_rms` and its switch to the whole annulus come from
  statistics pooled over all Stokes planes of a channel (`axes=[0, 1, 2]`,
  `cleanbox.py:294`). AstroVIPER measures each plane on its own, which is the
  same for an image with one polarization, and the summary makes the switch for
  each polarization. The per Stokes values the score uses for such images,
  `image_rms_iquv`, are per plane in the pipeline as well (`axes=[0, 1]`,
  `cleanbox.py:296`), but the pipeline switches them with the pooled count
  (`cleanbox.py:297-304, 330-332`), about `n_pol` times a plane's count. For
  I, Q, U, V images the summary therefore takes the whole annulus at median
  plane counts of about 3 to 9, where the pipeline does not.
- **Single Stokes images other than I.** For such a cube the pipeline's
  `image_rms_iquv[0]`, which the score uses, is the RMS of channel 0
  (`cleanbox.py:347-350, 364-373`); the summary gives the median over the
  channels.
- **Rounded levels.** The pipeline writes the levels into its mask expressions
  with `%f`, so it uses them rounded to six decimals. The functions use the
  levels as given, and callers pass `round(level, 6)` to reproduce the
  pipeline. For levels read from the primary beam the rounding moves the pixels
  at the sampled radii into or out of the annulus (16 pixels of the middle
  channel of the small image of the CASA fixture).
- **Empty planes.** CASA reports 0 for every statistic of a plane without
  pixels inside a cube; the per plane statistics are NaN, and the summary counts
  such a channel as 0, as the pipeline does.
- **No pixel in any channel.** When no channel keeps a pixel even in the whole
  annulus, the pipeline's fallback to -999 raises a TypeError
  (`cleanbox.py:388-401`); the summary returns NaN.
- **Masked pixels.** CASA leaves out the pixels the image's own pixel mask
  masks but keeps unmasked non-finite pixels. One such pixel in a plane's
  annulus gives that plane `npts` 0 and statistics 0, which the pipeline's
  summary counts as an RMS of 0. AstroVIPER leaves out non-finite pixels and
  measures the rest, so only planes whose annulus is partly non-finite differ.
  In both, only the values of the primary beam and of the clean mask count,
  not their pixel masks.
- **Clean mask.** The pipeline's clean mask is a float image, outside where it
  is below 0.1; AstroVIPER's is bool, `True` inside.
- **Continuum images.** The functions take the image and its primary beam on
  `frequency`. The continuum imager puts its restored planes on `taylor_term`,
  also for one term, and its primary beam on one `frequency` plane, so a
  continuum image is measured from a dataset that puts its first term on that
  `frequency` plane.

## Validation

`tests/component/test_primary_beam_annulus_statistics_casa.py` compares the
three annulus functions with CASA without importing CASA. Its reference,
`tests/component/data/casa_primary_beam_annulus_reference.npz` (159 KiB), is
made by hand with `tests/component/data/generate_casa_primary_beam_annulus_reference.py`
and casatools 6.7.6-14, which runs the pipeline's mask expressions and
`ia.statistics` calls in CASA and reproduces its `pblimits` and its median over
the continuum channels after the switch to the whole annulus. It holds three
images:

- a Stokes I cube of 48 x 48 pixels and 8 channels with the nominal levels, a
  channel whose primary beam is 0, a masked region of the annulus in one
  channel only, and a primary beam pixel of `float32(0.2)`, which the double
  precision comparison keeps;
- an I, Q, U, V cube of 32 x 32 pixels and 3 channels whose clean masks differ
  between polarizations (I and Q share one) and, for U and V, between channels;
- a 24 x 24 image of 3 channels whose levels come from its primary beam and
  whose union of clean masks leaves 5 annulus pixels per channel, so that the
  summary switches to the whole annulus.

Every Chauvenet pass of the generator's data keeps the same pixels for any `z`
in `[z(n), z(n + 1)]`, so the reference does not depend on the `z` CASA finds,
and the generator checks that its NumPy copy of casacore's clipping keeps as
many pixels as CASA.

Tolerances: the levels and the numbers of kept pixels are equal; the RMS,
standard deviation, median, median absolute deviation and summary values agree
to a relative 1e-12. The largest difference is 7e-16, and medians are equal.
The unit tests are in
`tests/unit/processing_functions/image_analysis/test_primary_beam_annulus_statistics.py`
and `test_chauvenet.py`.

Performance: on a 2048 x 2048 Stokes I cube of 32 channels with about 445000
annulus pixels per channel, `calculate_primary_beam_annulus_statistics` measures
both variants of every channel in 1.3 s on one thread of an Apple M3 Pro, with
the cube in memory. CASA 6.7.6 `ia.statistics` with the pipeline's arguments
takes 3.5 to 3.6 s for both variants, with its images in the file cache, also
on one thread: the casatools build for macOS has no OpenMP. The numbers of kept
pixels are equal in every channel and the RMS agrees to a relative 9e-14.

## Planned metrics

- **Sensitivity and dirty maximum from the imagers.** Both imagers return
  `SENSITIVITY` (`processing_functions/imaging/weight_statistics.py`), compared
  with CASA `apparentsens` and the pipeline's sensitivity
  (`imageparams_base.py:2009-2361`), and the signed Stokes I maximum of the
  first residual where the primary beam is above the lower level, behind the
  dirty dynamic range (`tclean.py:1296-1305`, `cleanbox.py:101-122`).
- **Fused statistics in the cube imager.** The imaging node task computes both
  annulus variants of the planes of its block of channels while they are in
  memory, the reduce forms the union of the clean masks, and the imager returns
  the metrics and writes them next to the image. Clipped statistics do not
  merge, so the statistics outside the union need every plane's annulus pixels
  after the reduce: the node task keeps each plane's annulus values and mask
  flags, or a second pass reads the annulus of the restored cube. The levels
  come from the primary beam of the middle channel of the whole cube, which a
  node task holding another block gets from the driver.
- **Image extrema.** Minimum and maximum where the primary beam is above the
  upper level, of the restored image and of the restored image divided by the
  primary beam, which is CASA's corrected image `(M * B + R) / P`; AstroVIPER's
  corrected restore is `(M / P) * B + R / P` (`cleanbox.py:153-217`). For an
  image whose intent is not TARGET the pipeline also restricts them to the box
  from `0.375 N - 1` to `0.625 N - 1` pixels on both axes, `N = max(nx, ny)`
  (PIPE-611, `cleanbox.py:177-191`), which the caller passes.
- **Continuum channel moments and residual line metrics.** Moments 0, 8 and 10
  of the restored cube over the continuum channels, accumulated while imaging,
  then the peak signal to noise of moment 8 against the per channel Chauvenet
  scaled MAD of the cube in `f * lower < PB < upper` outside the clean mask,
  with `f = min(1.05, round_down(1 + (upper / lower - 1) / 2, 2))`
  (`tclean.py:1745, 1781-1787`), its largest connected segment in beams and its
  histogram asymmetry (`tclean.py:1630-1905`).
- **Beam outliers.** The channels whose PSF beam is an outlier, as
  `check_psf` and `find_good_commonbeam` find them
  (`imageparams_base.py:2644-2824`).
- **Gaussian fit metrics.** The offset of a check source in beams and its peak
  over integrated flux (`pipeline/qa/checksource.py:10-153`), and the I, Q and U
  fits of polarization calibrators (`qa.py:250-332`).
