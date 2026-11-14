"""Zarr store that reads exactly the requested byte ranges."""

import asyncio
import os

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore


def _pread(path, prototype, byte_range):
    fd = os.open(path, os.O_RDONLY)
    try:
        if isinstance(byte_range, RangeByteRequest):
            start, stop = byte_range.start, byte_range.end
        else:
            size = os.fstat(fd).st_size
            if byte_range is None:
                start, stop = 0, size
            elif isinstance(byte_range, OffsetByteRequest):
                start, stop = byte_range.offset, size
            elif isinstance(byte_range, SuffixByteRequest):
                start, stop = max(0, size - byte_range.suffix), size
            else:
                raise TypeError(f"Unexpected byte_range, got {byte_range}.")
        parts = []
        while start < stop:
            part = os.pread(fd, stop - start, start)
            if not part:
                break
            parts.append(part)
            start += len(part)
    finally:
        os.close(fd)
    return prototype.buffer.from_bytes(b"".join(parts))


class PreadLocalStore(LocalStore):
    """:class:`zarr.storage.LocalStore` whose reads use ``os.pread``.

    ``LocalStore`` reads through a buffered ``open()``, whose buffer is the
    file's ``st_blksize``, and a read smaller than the buffer fills the whole
    buffer. On Lustre (``st_blksize`` 4 MiB) every inner chunk read from a
    shard then costs 4 MiB. This store reads only the requested bytes.
    """

    def get_sync(self, key, *, prototype=None, byte_range=None):
        if prototype is None:
            prototype = default_buffer_prototype()
        self._ensure_open_sync()
        try:
            return _pread(self.root / key, prototype, byte_range)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def get(self, key, prototype=None, byte_range=None):
        if prototype is None:
            prototype = default_buffer_prototype()
        await self._ensure_open()
        try:
            return await asyncio.to_thread(
                _pread, self.root / key, prototype, byte_range
            )
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def get_partial_values(self, prototype, key_ranges):
        return list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(_pread, self.root / key, prototype, byte_range)
                    for key, byte_range in key_ranges
                )
            )
        )
