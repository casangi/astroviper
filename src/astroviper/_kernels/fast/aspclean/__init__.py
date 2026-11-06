"""
Fast Asp (Coral) model update, selected by ``ASTROVIPER_KERNELS=asp=fast``.

``clean`` and ``clean_cube`` are those of the reference
(``deconvolvers.aspclean``, PR #410 semantics with ``asp_min_mask_coverage``):
the same parameters, defaults, checks, exceptions and results. The residual,
the model and every value of the returned dict (the components, the
residual-shaped fits, the iteration history and every count) have the
reference's bits, NaNs included (on x86-64), at any number of threads; only
the wall times in ``timing`` differ. The algorithm, its control flow and every
floating-point expression are the reference's; what changes is how the work is
done:

* the 2-D transforms run pocketfft's own 1-D plans line by line, as the
  reference's calls do, but only on the rows that are not zero on input or
  are read on output, fused with the products of the spectra and with the
  passes that use the rows of the inverse transforms (peak searches, residual
  updates, noise samples), and threaded over blocks of rows and columns;
* the smoothed residuals of the search are never stored: each row is searched
  as it is made;
* elementwise passes run on all threads, and sums stay serial, in the
  reference's order.

The module is built for the baseline instruction set only (``ISA`` is
``"generic"``): pocketfft's AVX2 lanes give some NaNs another sign bit than
the reference's.
"""

from astroviper._kernels.fast.aspclean._aspclean_fast_ext import ISA, clean, clean_cube

__all__ = ["ISA", "clean", "clean_cube"]
