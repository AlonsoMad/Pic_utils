"""Shared-memory lifecycle tests (architecture §8.8)."""

from __future__ import annotations

import multiprocessing
import pickle
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
from fixtures import rgb_gradient, rgba_gradient

from pic_utils.core.errors import SHM_UNAVAILABLE, RunError
from pic_utils.core.image import Image
from pic_utils.runner import shm as shm_module
from pic_utils.runner.protocol import ImageHandle
from pic_utils.runner.shm import SharedImageStore, attach_image

_POSIX = hasattr(Path("/dev/shm"), "is_dir") and Path("/dev/shm").is_dir()


def live_segments() -> set[str]:
    """Names of the store's shared-memory segments visible on the OS."""
    return {path.name for path in Path("/dev/shm").glob("picutils_*")}


def test_rgb_round_trip() -> None:
    image = Image(rgb_gradient(16))
    store = SharedImageStore("runrt")

    handle = store.put(image)

    attached = attach_image(handle)
    np.testing.assert_array_equal(attached.rgb, image.rgb)
    assert attached.alpha is None
    store.close()


def test_alpha_round_trips_in_its_own_block() -> None:
    image = Image.from_rgba(rgba_gradient(16))
    store = SharedImageStore("runalpha")

    handle = store.put(image)

    assert handle.alpha_name is not None
    assert handle.alpha_name != handle.name
    assert handle.alpha_shape == (16, 16)
    attached = attach_image(handle)
    np.testing.assert_array_equal(attached.rgb, image.rgb)
    np.testing.assert_array_equal(attached.alpha, image.alpha)
    store.close()


def test_opaque_image_has_no_alpha_block() -> None:
    store = SharedImageStore("runopaque")

    handle = store.put(Image(rgb_gradient()))

    assert handle.alpha_name is None
    assert handle.alpha_shape is None
    assert handle.nbytes == 8 * 8 * 3
    store.close()


def test_handle_names_are_unique_per_store() -> None:
    store = SharedImageStore("runnames")

    first = store.put(Image(rgb_gradient()))
    second = store.put(Image(rgb_gradient()))

    assert first.name != second.name
    assert first.name.startswith("picutils_runnames_")
    store.close()


def test_handle_nbytes_covers_both_planes() -> None:
    store = SharedImageStore("runbytes")
    image = Image.from_rgba(rgba_gradient(8))

    handle = store.put(image)

    assert handle.nbytes == image.nbytes == 8 * 8 * 4
    assert store.total_bytes == image.nbytes
    store.release(handle)
    assert store.total_bytes == 0
    store.close()


def test_handle_is_picklable() -> None:
    store = SharedImageStore("runpickle")
    handle = store.put(Image(rgb_gradient()))

    clone = pickle.loads(pickle.dumps(handle))

    assert clone == handle
    store.close()


def test_release_unlinks_the_blocks() -> None:
    if not _POSIX:
        pytest.skip("/dev/shm enumeration is POSIX-only")
    store = SharedImageStore("runfree")
    image = Image.from_rgba(rgba_gradient(8))

    handle = store.put(image)
    assert live_segments() >= {handle.name, handle.alpha_name}

    store.release(handle)

    assert not (live_segments() & {handle.name, handle.alpha_name})


def test_close_releases_every_remaining_block() -> None:
    if not _POSIX:
        pytest.skip("/dev/shm enumeration is POSIX-only")
    store = SharedImageStore("runclose")
    before = live_segments()
    handles = [store.put(Image(rgb_gradient())) for _ in range(3)]

    store.close()
    store.close()  # idempotent

    assert live_segments() == before
    assert all(handle.name not in live_segments() for handle in handles)


def test_put_failure_cleans_up_partial_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _POSIX:
        pytest.skip("/dev/shm enumeration is POSIX-only")
    store = SharedImageStore("runfail")
    before = live_segments()
    real_shared_memory = shared_memory.SharedMemory

    class Flaky(real_shared_memory):  # type: ignore[misc, valid-type]
        def __init__(self, *, name: str | None = None, create: bool = False, size: int = 0) -> None:
            if create and name is not None and name.endswith("_alpha"):
                raise OSError("simulated allocation failure")
            super().__init__(name=name, create=create, size=size)

    monkeypatch.setattr(shm_module.shared_memory, "SharedMemory", Flaky)
    with pytest.raises(RunError) as excinfo:
        store.put(Image.from_rgba(rgba_gradient(8)))

    assert excinfo.value.code == SHM_UNAVAILABLE
    assert live_segments() == before


def test_attach_missing_block_raises_shm_unavailable() -> None:
    handle = ImageHandle(name="picutils_missing_0_rgb", shape=(8, 8, 3))
    with pytest.raises(RunError) as excinfo:
        attach_image(handle)
    assert excinfo.value.code == SHM_UNAVAILABLE


def test_attach_copies_into_private_arrays() -> None:
    store = SharedImageStore("runprivate")
    image = Image(rgb_gradient(8))
    handle = store.put(image)
    attached = attach_image(handle)

    # Overwrite the shared block; the attached image must not change, and the
    # arrays must be read-only (the Image invariant).
    block = shared_memory.SharedMemory(name=handle.name)
    try:
        np.ndarray(handle.shape, dtype=np.uint8, buffer=block.buf)[...] = 0
    finally:
        block.close()

    np.testing.assert_array_equal(attached.rgb, image.rgb)
    with pytest.raises(ValueError, match="read-only"):
        attached.rgb[0, 0, 0] = 255
    store.close()


def test_attach_from_spawned_process() -> None:
    """A spawned worker can attach by name (the §1.2 concurrency boundary)."""
    store = SharedImageStore("runspawn")
    handle = store.put(Image(rgb_gradient(8)))

    context = multiprocessing.get_context("spawn")
    process = context.Process(target=attach_image, args=(handle,))
    process.start()
    process.join(timeout=60)

    store.release(handle)
    assert process.exitcode == 0
