"""Reduction backend of ``find_continuum_fused``: fused passes over the cube.

findContinuum asks for one reduction of the cube at a time, and
``find_continuum`` makes a pass over the image store for each.
:class:`FusedReductions` answers the requests from a cache instead. A request
that misses the cache runs one pass that also computes the reductions
findContinuum asks for next, as far as they are known by then:

- the first pass adds the first-round reductions that need no earlier result
  and were not computed by the imager while it made the cube: the moment 0
  and 8 images of the cube, the statistics of the cube and of the primary
  beam, and the mean primary beam;
- a pass for the moment 0 and 8 images of a channel selection adds the noise
  of the cube outside the joint mask in those channels (``cubeNoiseLevel``,
  with the arguments findContinuum has computed by then), the cube peak in
  those channels and the moment images of the intersections with earlier
  selections;
- a pass for the median absolute deviation spectrum outside the joint mask
  (``xmadm``) adds the mean spectrum with the same mask.

The cache answers only a request identical to one already computed, with the
small images of its mask unchanged, so the results are those of
``find_continuum`` whatever is predicted; a prediction findContinuum does not
ask for costs computation, not a read. Moment maps are answered as they are,
or read afresh from their store, since ``immoments`` only reads them; every
other result is a copy, since findContinuum changes them.
"""

import contextlib
import copy
import hashlib
import io
import sys

import numpy as np
import toolviper.utils.logger as logger

from astroviper.distributed_applications.imaging.find_continuum import (
    casa_image_tasks,
    findContinuum,
)
from astroviper.distributed_applications.imaging.find_continuum.ultra_reductions import (
    StoredMap,
)
from astroviper.node_tasks.imaging.find_continuum import small_image_attributes
from astroviper.processing_functions.imaging.find_continuum import (
    CHANNEL_STATISTICS,
    find_continuum_first_round_requests,
)


def _frozen(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _frozen(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return value


def _find_continuum_locals():
    """Local variables of the running findContinuum call, or ``None``."""
    frame = sys._getframe()
    while frame is not None:
        code = frame.f_code
        if (
            code.co_name == "findContinuum"
            and code.co_filename == findContinuum.__file__
        ):
            return frame.f_locals
        frame = frame.f_back
    return None


class FusedReductions:
    """Answers findContinuum's reduction requests from fused passes.

    Parameters
    ----------
    run_pass : callable
        ``run_pass(requests, n_requested)`` makes one pass over the image store
        for the requests, the ``n_requested`` requested ones first and the
        predicted ones after them, and returns their finalized results; moment
        maps may be
        :class:`~astroviper.distributed_applications.imaging.find_continuum.ultra_reductions.StoredMap`,
        and a prediction left out is ``None``, which is not cached.
    cube, primary_beam : str
        Paths of the registered dirty cube and primary beam, as findContinuum
        names them.
    first_round : list, optional
        Results of the four requests of
        :func:`~astroviper.processing_functions.imaging.find_continuum.find_continuum_first_round_requests`
        already computed (from the imager's first-round store), ``None`` for
        one that was not; they fill the cache, and the first pass computes the
        others.
    """

    def __init__(self, run_pass, cube, primary_beam, first_round=None):
        self.run_pass = run_pass
        self.cube = cube
        self.primary_beam = primary_beam
        self.cache = {}
        # Content digest of every small image a key names, per path and write
        # id: an image findContinuum writes again with the same pixels keeps it.
        self.digests = {}
        self.selections = []
        self.captured = None
        self.first_round = first_round
        self._first_pass = True

    def __call__(self, requests):
        if self.captured is not None:
            self.captured.extend(requests)
            return [self._placeholder(request) for request in requests]
        if self.first_round is not None:
            keys = map(self._key, self._first_round())
            self.cache.update(
                (key, result)
                for key, result in zip(keys, self.first_round, strict=True)
                if result is not None
            )
            self.first_round = None
        keys = [self._key(request) for request in requests]
        batch = {
            key: request
            for key, request in zip(keys, requests, strict=True)
            if key not in self.cache
        }
        if batch:
            n_requested = len(batch)
            for request in self._predict(requests):
                batch.setdefault(self._key(request), request)
            batch = {
                key: request for key, request in batch.items() if key not in self.cache
            }
            logger.debug(
                f"find_continuum_fused pass: {n_requested} requested, {len(batch) - n_requested} predicted"
            )
            results = self.run_pass(list(batch.values()), n_requested)
            self.cache.update(
                (key, result)
                for key, result in zip(batch, results, strict=True)
                if result is not None
            )
        return [
            _answer(request, self.cache[key])
            for request, key in zip(requests, keys, strict=True)
        ]

    def _key(self, request):
        small_images, _ = casa_image_tasks._mask_references(request["mask"])
        fields = {
            key: value
            for key, value in request.items()
            if key not in ("frequency_axis", "rest_frequency")
        }
        return _frozen(fields), tuple(self._fingerprint(path) for path in small_images)

    def _fingerprint(self, path):
        write_id = small_image_attributes(path)["write_id"]
        if (path, write_id) not in self.digests:
            values, mask, _ = casa_image_tasks._read(path)
            digest = hashlib.sha1(np.ascontiguousarray(values))
            digest.update(np.ascontiguousarray(mask))
            self.digests[path, write_id] = digest.hexdigest()
        return path, self.digests[path, write_id]

    def _placeholder(self, request):
        shape = casa_image_tasks._cube(request["image"])["metadata"]["shape"]
        n_channels = (
            shape[3] if request["channels"] is None else len(request["channels"])
        )
        if request["kind"] == "moments":
            return {moment: np.full(shape[:2], np.nan) for moment in request["moments"]}
        if request["kind"] == "profile":
            return np.full(n_channels, np.nan)
        if request["per_channel"]:
            return {key: np.full(n_channels, np.nan) for key in CHANNEL_STATISTICS}
        return {key: np.nan for key in CHANNEL_STATISTICS}

    def _capture(self, function, *args, **kwargs):
        """The requests ``function`` makes, answered with NaN placeholders."""
        self.captured = []
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                function(*args, **kwargs)
            return self.captured
        except Exception as error:
            # A prediction never stops the run: the request then misses the cache.
            logger.debug(
                f"find_continuum_fused could not predict {function.__name__}: {error!r}"
            )
            return []
        finally:
            self.captured = None

    def _predict(self, requests):
        predicted = []
        if self._first_pass:
            self._first_pass = False
            predicted += self._first_round()
        for request in requests:
            if (
                request["kind"] == "moments"
                and request["image"] == self.cube
                and request["mask"] == ""
                and request["moments"] == [0, 8]
                and request["channels"] is not None
            ):
                predicted += self._selection_round(request)
            if request["kind"] == "profile" and request["function"] == "xmadm":
                # computeStatisticalSpectrumFromMask asks for the mean profile
                # with the same mask next.
                predicted.append(dict(request, function="mean"))
        return predicted

    def _first_round(self):
        metadata = casa_image_tasks._cube(self.cube)["metadata"]
        return find_continuum_first_round_requests(
            (
                metadata["reference_value"][3],
                metadata["increment"][3],
                metadata["reference_pixel"][3],
                metadata["shape"][3],
            ),
            metadata["rest_frequency"],
            self.cube,
            self.primary_beam,
        )

    def _selection_round(self, request):
        channels = set(request["channels"])
        predicted = [dict(request, kind="statistics", per_channel=False)]
        for key in ("moments", "frequency_axis", "rest_frequency"):
            predicted[0].pop(key)
        for earlier in self.selections:
            common = channels & earlier
            if common and common != channels and common != earlier:
                predicted.append(dict(request, channels=sorted(common)))
        self.selections.append(channels)
        names = _find_continuum_locals()
        if names is not None and not names.get("skipCubeNoise", False):
            predicted += self._capture(
                findContinuum.cubeNoiseLevel,
                self.cube,
                pbcube=names.get("pbmom"),
                mask=names.get("jointMaskTest"),
                chans=names.get("selection"),
                subimage=names.get("subimage"),
                imageInfo=names.get("imageInfo"),
            )
        return predicted


def _answer(request, result):
    """A cached result as findContinuum gets it: moment maps as they are or
    loaded fresh from their store, every other result a copy."""
    if request["kind"] == "moments":
        return {
            moment: values.load() if isinstance(values, StoredMap) else values
            for moment, values in result.items()
        }
    return copy.deepcopy(result)
