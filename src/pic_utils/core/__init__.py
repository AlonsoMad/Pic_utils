"""Core utilities for Pic Utils.

Typed exceptions and runtime error codes (:mod:`pic_utils.core.errors`),
JSON-safe conversion and atomic writes (:mod:`pic_utils.core.jsonio`), and
pure path helpers (:mod:`pic_utils.core.paths`).

``core/`` is pure Python + numpy: it imports nothing from the application
layers (architecture §1.3).
"""

from pic_utils.core.errors import (
    BLOCK_INVALID_OUTPUT,
    BLOCK_RAISED,
    BLOCK_SNAPSHOT_MISMATCH,
    BLOCK_TIMEOUT,
    BLOCK_TOO_MANY_OUTPUTS,
    CANCELLED,
    IO_READ,
    IO_WRITE,
    MEMORY_LIMIT,
    POOL_UNSTABLE,
    SHM_UNAVAILABLE,
    WORKER_CRASHED,
    BlockError,
    CanvasFileError,
    ExportError,
    ImageLoadError,
    ImageTooLargeError,
    InvalidBlockOutput,
    MemoryLimitError,
    PicUtilsError,
    ProjectFileError,
    RunError,
    UnsupportedImageError,
    ValidationError,
)
from pic_utils.core.jsonio import jsonable, read_json, write_json_atomic
from pic_utils.core.paths import content_hash, content_hash_short, ensure_dir, safe_name

__all__ = [
    "BLOCK_INVALID_OUTPUT",
    "BLOCK_RAISED",
    "BLOCK_SNAPSHOT_MISMATCH",
    "BLOCK_TIMEOUT",
    "BLOCK_TOO_MANY_OUTPUTS",
    "CANCELLED",
    "IO_READ",
    "IO_WRITE",
    "MEMORY_LIMIT",
    "POOL_UNSTABLE",
    "SHM_UNAVAILABLE",
    "WORKER_CRASHED",
    "BlockError",
    "CanvasFileError",
    "ExportError",
    "ImageLoadError",
    "ImageTooLargeError",
    "InvalidBlockOutput",
    "MemoryLimitError",
    "PicUtilsError",
    "ProjectFileError",
    "RunError",
    "UnsupportedImageError",
    "ValidationError",
    "content_hash",
    "content_hash_short",
    "ensure_dir",
    "jsonable",
    "read_json",
    "safe_name",
    "write_json_atomic",
]
