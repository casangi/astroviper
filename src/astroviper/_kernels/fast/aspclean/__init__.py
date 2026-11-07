"""
Fast Asp (Coral) model update, selected by ``ASTROVIPER_KERNELS=asp=fast``.

``clean`` and ``clean_cube`` are those of the reference
(``deconvolvers.aspclean``, PR #410 semantics with ``asp_min_mask_coverage``):
the same parameters, defaults, checks, exceptions and results. The residual,
the model and every value of the returned dict (the components, the
residual-shaped fits, the iteration history and every count) have the
reference's bits, at any number of threads; only the wall times in ``timing``
differ. The algorithm, its control flow and every floating-point expression
are the reference's; what changes is how the work is done:

* the 2-D transforms run pocketfft's own 1-D plans line by line, as the
  reference's calls do, but only on the rows that are not zero on input or
  are read on output, fused with the products of the spectra and with the
  passes that use the rows of the inverse transforms (peak searches, residual
  updates, noise samples), and threaded over blocks of rows and columns;
* the smoothed residuals of the search are never stored: each row is searched
  as it is made;
* elementwise passes run on all threads, and sums stay serial, in the
  reference's order.

A call whose PSF or model holds a value that is not finite or of magnitude
2^64 or more, or whose residual holds one of magnitude 2^64 or more or one that
is not finite where the reference searches (the mask above 0.99, or anywhere
without a mask), is declined after the checks: the reference runs it with the
same arguments, so its bits are the reference's, NaNs included, at the call's
thread count, and the kernel switch traces it with path ``"declined"``. On
such values pocketfft's SIMD lanes and its scalar path (the reference's on
some lines) can give a NaN another sign bit or payload, as can two compilations
of one sum.

The PSF and the mask are read in place at any address; a residual or model at
an address that is not a multiple of its item size (Dask may hand a worker
one) is updated through an aligned copy of its bytes, written back in place.

The module is built for the baseline instruction set only (``ISA`` is
``"generic"``).
"""

from astroviper._kernels.fast.aspclean._aspclean_fast_ext import ISA, clean, clean_cube

__all__ = ["ISA", "clean", "clean_cube"]
