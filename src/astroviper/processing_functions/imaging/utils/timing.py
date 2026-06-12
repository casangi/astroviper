"""Timing-bookkeeping helpers for the imaging cycle (non-science utilities)."""


def accumulate_timing(timing, return_df):
    """Accumulate the columns of a one-row timing frame into ``timing`` in place.

    Each processing-function timing is summed across every call (setup plus all
    residual and model-update cycles) so the final per-chunk timing frame reports
    the total wall time spent in each processing function.

    Parameters
    ----------
    timing : dict
        Accumulator mapping timing-column name to a running total (seconds).
    return_df : pandas.DataFrame or None
        One-row timing frame whose columns are added into ``timing``.  ``None``
        is ignored.
    """
    if return_df is None:
        return
    for column in return_df.columns:
        timing[column] = timing.get(column, 0.0) + float(return_df[column].iloc[0])
