"""Shared-memory image transport between parent and workers (§8.8).

The parent copies each source image once into named ``SharedMemory`` blocks
(RGB, plus a second block when an alpha plane is present); workers attach by
name, copy into private arrays (the block API exposes read-only arrays, and
detaching with live views is unsafe), then close their handles. The parent
keeps its handles alive until the pool has shut down and then unlinks —
workers only ever attach and close, never unlink.

Shared memory is the only large-data IPC. If a block cannot be created, the
failure surfaces as ``RunError("shm.unavailable")`` — there is no silent
pickle fallback (§8.8).
"""

from __future__ import annotations

import contextlib
import itertools
from multiprocessing import shared_memory

import numpy as np

from pic_utils.core.errors import SHM_UNAVAILABLE, RunError
from pic_utils.core.image import Image
from pic_utils.runner.protocol import ImageHandle

__all__ = ["SharedImageStore", "attach_image"]


class SharedImageStore:
    """Parent-side owner of the run's shared-memory image blocks."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._counter = itertools.count()
        self._blocks: dict[
            str, tuple[shared_memory.SharedMemory, shared_memory.SharedMemory | None]
        ] = {}

    @property
    def total_bytes(self) -> int:
        """Bytes currently held in shared blocks (memory accounting, §8.6)."""
        return sum(
            rgb.size + (0 if alpha is None else alpha.size) for rgb, alpha in self._blocks.values()
        )

    def put(self, image: Image) -> ImageHandle:
        """Copy ``image`` into shared blocks and return a handle for workers.

        On any failure the blocks created for this image are released again:
        a failed ``put`` never leaks.
        """
        index = next(self._counter)
        base_name = f"picutils_{self._run_id}_{index}"
        rgb_block: shared_memory.SharedMemory | None = None
        alpha_block: shared_memory.SharedMemory | None = None
        try:
            rgb_block = self._create(f"{base_name}_rgb", image.rgb.nbytes)
            _copy_into(rgb_block, image.rgb)
            if image.alpha is not None:
                alpha_block = self._create(f"{base_name}_alpha", image.alpha.nbytes)
                _copy_into(alpha_block, image.alpha)
        except BaseException:
            for block in (rgb_block, alpha_block):
                if block is not None:
                    _close_and_unlink(block)
            raise
        handle = ImageHandle(
            name=f"{base_name}_rgb",
            shape=(image.height, image.width, 3),
            dtype=str(image.rgb.dtype),
            alpha_name=None if alpha_block is None else f"{base_name}_alpha",
            alpha_shape=None if image.alpha is None else tuple(image.alpha.shape),
        )
        self._blocks[handle.name] = (rgb_block, alpha_block)
        return handle

    def release(self, handle: ImageHandle) -> None:
        """Close and unlink the blocks behind ``handle``."""
        blocks = self._blocks.pop(handle.name)
        _close_and_unlink(blocks[0])
        if blocks[1] is not None:
            _close_and_unlink(blocks[1])

    def close(self) -> None:
        """Release every block still held (called after pool shutdown, §8.8)."""
        for handle_name in list(self._blocks):
            handle = ImageHandle(name=handle_name, shape=(0, 0, 0))
            self.release(handle)

    def _create(self, name: str, size: int) -> shared_memory.SharedMemory:
        try:
            return shared_memory.SharedMemory(name=name, create=True, size=size)
        except OSError as error:
            raise RunError(code=SHM_UNAVAILABLE) from error


def attach_image(handle: ImageHandle) -> Image:
    """Attach the blocks behind ``handle`` and copy them into private arrays.

    Worker-side entry point (§8.3 step 1): the copies are writable-private,
    the shared blocks are only closed here, never unlinked — the parent owns
    their lifetime.
    """
    rgb = _attach_plane(handle.name, handle.shape, handle.dtype)
    alpha = None
    if handle.alpha_name is not None:
        if handle.alpha_shape is None:
            raise ValueError(f"handle {handle.name!r} has an alpha block but no alpha shape")
        alpha = _attach_plane(handle.alpha_name, handle.alpha_shape, handle.dtype)
    return Image(rgb, alpha)


def _attach_plane(name: str, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    try:
        block = shared_memory.SharedMemory(name=name)
    except OSError as error:
        raise RunError(code=SHM_UNAVAILABLE) from error
    try:
        return np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf).copy()
    finally:
        block.close()


def _copy_into(block: shared_memory.SharedMemory, array: np.ndarray) -> None:
    view = np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)
    view[...] = array


def _close_and_unlink(block: shared_memory.SharedMemory) -> None:
    block.close()
    # Another owner may have unlinked already (e.g. a crashed parent cleaned
    # up); the segment is going away either way.
    with contextlib.suppress(OSError):
        block.unlink()
