"""Content-addressed canvas assets (architecture §10.2).

Assets live in a sibling folder next to the canvas file
(``<name>.assets/``). File names are the first 16 hex digits of the SHA-256
of the stored bytes, so saving only ever adds new files and never
invalidates an existing reference. Writes are atomic (temp + ``os.replace``).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
from pathlib import Path

from PIL import Image as PILImage

from pic_utils.core.errors import CanvasFileError
from pic_utils.core.image import Image, Mask
from pic_utils.io.loader import load_image

__all__ = ["AssetStore"]

_HASH_LENGTH = 16
_TEMP_PREFIX = ".pic_utils-"
_TEMP_SUFFIX = ".tmp"


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically, creating the parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise


def _content_name(data: bytes, suffix: str) -> str:
    digest = hashlib.sha256(data).hexdigest()[:_HASH_LENGTH]
    return f"{digest}{suffix}"


def _png_bytes(image: Image) -> bytes:
    buffer = io.BytesIO()
    if image.alpha is None:
        pil_image = PILImage.fromarray(image.rgb, "RGB")
    else:
        pil_image = PILImage.fromarray(image.to_rgba(), "RGBA")
    pil_image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


def _mask_png_bytes(mask: Mask) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(mask.data, "L").save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()


class AssetStore:
    """One canvas's assets folder; content-addressed and append-only."""

    def __init__(self, assets_dir: str | Path) -> None:
        self.assets_dir = Path(assets_dir)

    # --- writing ------------------------------------------------------------

    def put_png_bytes(self, data: bytes) -> str:
        """Store PNG bytes and return the relative reference."""
        reference = _content_name(data, ".png")
        target = self.assets_dir / reference
        if not target.is_file():
            _write_bytes_atomic(target, data)
        return reference

    def put_image(self, image: Image) -> str:
        """Encode ``image`` as a lossless PNG and return its reference."""
        return self.put_png_bytes(_png_bytes(image))

    def put_mask(self, mask: Mask) -> str:
        """Encode a mask as a grayscale PNG and return its reference."""
        data = _mask_png_bytes(mask)
        reference = _content_name(data, ".mask.png")
        target = self.assets_dir / reference
        if not target.is_file():
            _write_bytes_atomic(target, data)
        return reference

    def put_file(self, path: str | Path) -> str:
        """Load an image file, normalize it, and store it as a PNG."""
        return self.put_image(load_image(path))

    def copy_from(self, other: AssetStore, reference: str) -> None:
        """Copy one referenced asset from ``other`` into this store."""
        data = other.resolve(reference).read_bytes()
        target = self.assets_dir / reference
        if not target.is_file():
            _write_bytes_atomic(target, data)

    # --- reading ------------------------------------------------------------

    def exists(self, reference: str) -> bool:
        """Whether ``reference`` resolves to a file inside this store."""
        try:
            return self.resolve(reference).is_file()
        except CanvasFileError:
            return False

    def resolve(self, reference: str) -> Path:
        """Return the absolute path for ``reference``.

        Raises:
            CanvasFileError: if the reference is missing or escapes the store.
        """
        root = self.assets_dir.resolve()
        target = (self.assets_dir / reference).resolve()
        if target != root and not target.is_relative_to(root):
            raise CanvasFileError(f"asset reference escapes the store: {reference!r}")
        if not target.is_file():
            raise CanvasFileError(f"missing asset: {reference}")
        return target

    # --- maintenance --------------------------------------------------------

    def gc(self, referenced: set[str]) -> None:
        """Best-effort removal of files not in ``referenced`` (§10.2 step 3)."""
        if not self.assets_dir.is_dir():
            return
        for child in self.assets_dir.iterdir():
            if not child.is_file() or child.name in referenced:
                continue
            with contextlib.suppress(OSError):
                child.unlink()
