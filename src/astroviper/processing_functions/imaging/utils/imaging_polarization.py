"""Polarization modes of the imager: which correlations make which Stokes planes.

The residual update grids and degrids in the instrument (correlation) basis
and the model update deconvolves in the Stokes basis.  Two modes are
supported:

========================  ================  ================
correlations gridded      linear feeds      circular feeds
========================  ================  ================
the two parallel hands    ``I, Q``          ``I, V``
all four correlations     ``I, Q, U, V``    ``I, Q, U, V``
========================  ================  ================

The Stokes planes requested for the image therefore fix the correlations that
are loaded and gridded; nothing else is read from the processing set.
"""

import numpy as np

CORRELATIONS = {
    "linear": {2: ["XX", "YY"], 4: ["XX", "XY", "YX", "YY"]},
    "circular": {2: ["RR", "LL"], 4: ["RR", "RL", "LR", "LL"]},
}
STOKES = {
    "linear": {2: ["I", "Q"], 4: ["I", "Q", "U", "V"]},
    "circular": {2: ["I", "V"], 4: ["I", "Q", "U", "V"]},
}


def correlations_for_stokes(polarization_coords, instrument_polarization_basis):
    """Correlations that must be gridded to make the requested Stokes planes.

    Parameters
    ----------
    polarization_coords : list of str
        Stokes planes of the image, ``image_params["polarization_coords"]``.
    instrument_polarization_basis : str
        ``"linear"`` or ``"circular"``.

    Returns
    -------
    list of str
        ``["XX", "YY"]`` / ``["RR", "LL"]`` for the two-plane request and the
        four correlations of the basis for ``["I", "Q", "U", "V"]``.

    Raises
    ------
    ValueError
        For an unknown basis or a request that is not one of the two supported
        sets of the basis (in that order).
    """
    if instrument_polarization_basis not in CORRELATIONS:
        raise ValueError(
            f"instrument_polarization_basis must be 'linear' or 'circular'; "
            f"got {instrument_polarization_basis!r}."
        )
    requested = [str(label) for label in np.atleast_1d(polarization_coords)]
    for n_correlation, stokes in STOKES[instrument_polarization_basis].items():
        if requested == stokes:
            return list(CORRELATIONS[instrument_polarization_basis][n_correlation])
    supported = " or ".join(
        str(stokes) for stokes in STOKES[instrument_polarization_basis].values()
    )
    raise ValueError(
        f"polarization_coords {requested} is not supported for the "
        f"{instrument_polarization_basis} instrument_polarization_basis: request "
        f"{supported} (the two parallel hands or all four correlations are gridded)."
    )


def correlation_selection(data_polarization, correlations, ms_name=""):
    """Index positions of the needed correlations in a measurement set.

    Parameters
    ----------
    data_polarization : array_like of str
        The ``polarization`` coordinate of the measurement set.
    correlations : list of str
        Correlations to grid (:func:`correlations_for_stokes`).
    ms_name : str
        Name used in the error message.

    Returns
    -------
    list of int or None
        Positions to select along ``polarization``; ``None`` when the
        measurement set holds exactly these correlations in this order, so
        that nothing has to be selected.

    Raises
    ------
    ValueError
        If a needed correlation is not in the measurement set.
    """
    available = [str(label) for label in np.atleast_1d(data_polarization)]
    missing = [label for label in correlations if label not in available]
    if missing:
        raise ValueError(
            f"measurement set {ms_name!r} holds the correlations {available} but "
            f"{list(correlations)} are needed for the requested Stokes planes "
            f"(missing {missing}); check polarization_coords and "
            "instrument_polarization_basis."
        )
    if available == list(correlations):
        return None
    return [available.index(label) for label in correlations]


def is_four_correlation_basis(polarization_labels):
    """``True`` for the four correlations of the linear or the circular basis."""
    labels = {str(label) for label in np.atleast_1d(polarization_labels)}
    return any(labels == set(basis[4]) for basis in CORRELATIONS.values())
