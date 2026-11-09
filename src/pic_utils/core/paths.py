"""Pure path helpers: content hashing, safe filenames, directory creation.

No Qt and no ``QStandardPaths`` here — standard locations are wired by the
settings layer (P5-01, architecture §16).
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path

__all__ = ["content_hash", "content_hash_short", "ensure_dir", "safe_name"]

_HASH_CHUNK_BYTES = 1024 * 1024
# Anything that is unsafe in a filename on Linux or Windows, as a run.
_UNSAFE_RUN = re.compile(r'[\s\x00-\x1f\x7f/\\:*?"<>|]+')
_EDGE_JUNK = " ._"
_FALLBACK_NAME = "untitled"


def content_hash(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 hex digest of the file's bytes.

    The file is streamed in 1 MiB chunks; it is never loaded whole.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash_short(path: str | os.PathLike[str], length: int = 16) -> str:
    """Return the first ``length`` hex characters of :func:`content_hash`.

    Canvas asset names use the default of 16 characters (architecture §10.2).
    """
    return content_hash(path)[:length]


def safe_name(name: str, max_length: int = 80) -> str:
    """Return a deterministic, filesystem-safe filename derived from ``name``.

    The result is ASCII-only: text is folded with NFKD and non-ASCII
    characters dropped (``café`` -> ``cafe``). Runs of whitespace and
    filesystem-unsafe characters (``/ \\ : * ? " < > |`` and control
    characters) collapse to a single ``_``. Leading/trailing dots, spaces and
    underscores are stripped (avoids hidden files, ``..`` and Windows
    trailing-dot problems), and the result is bounded to ``max_length``
    characters. A name that sanitizes to nothing falls back to
    ``untitled``.

    The mapping is not injective (``a/b`` and ``a_b`` both become ``a_b``);
    callers that need uniqueness append a :func:`content_hash_short`.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    cleaned = _UNSAFE_RUN.sub("_", folded).strip(_EDGE_JUNK)
    if not cleaned:
        return _FALLBACK_NAME
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].strip(_EDGE_JUNK)
    return cleaned or _FALLBACK_NAME


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` and any missing parents; return it as a ``Path``.

    Idempotent: an existing directory is left untouched. Raises
    ``FileExistsError`` if ``path`` exists but is a file.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
