"""Typed exception hierarchy and stable runtime error codes (architecture §18).

Every error raised by Pic Utils code is one of the classes below — never a
bare ``Exception`` (§2). The §18.2 runtime codes are stable strings that the
UI maps to translated messages; adding a code is a contract change (CC task).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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
]

# --- Runtime error codes (architecture §18.2) -------------------------------

BLOCK_RAISED: Final[str] = "block.raised"
BLOCK_INVALID_OUTPUT: Final[str] = "block.invalid_output"
BLOCK_SNAPSHOT_MISMATCH: Final[str] = "block.snapshot_mismatch"
BLOCK_TIMEOUT: Final[str] = "block.timeout"
BLOCK_TOO_MANY_OUTPUTS: Final[str] = "block.too_many_outputs"
MEMORY_LIMIT: Final[str] = "memory.limit"
WORKER_CRASHED: Final[str] = "worker.crashed"
POOL_UNSTABLE: Final[str] = "pool.unstable"
SHM_UNAVAILABLE: Final[str] = "shm.unavailable"
IO_READ: Final[str] = "io.read"
IO_WRITE: Final[str] = "io.write"
CANCELLED: Final[str] = "cancelled"

# --- Exception hierarchy (architecture §18.1) -------------------------------


class PicUtilsError(Exception):
    """Base class for every error raised by Pic Utils code."""


@dataclass(frozen=True, eq=False)
class ValidationError(PicUtilsError):
    """A pipeline or run configuration failed validation.

    ``issues`` holds the concrete ``Issue`` dataclasses of the pipeline layer
    (architecture §6.3: ``severity``, ``code``, ``message``, ``step_id``,
    ``param``). It is typed ``Sequence[Any]`` because ``core/`` MUST NOT
    import ``pipeline/`` (§1.3); the structural check happens at the
    pipeline/pipeline boundary.
    """

    issues: Sequence[Any]

    def __post_init__(self) -> None:
        messages = [str(getattr(issue, "message", issue)) for issue in self.issues]
        super().__init__("; ".join(messages) if messages else "validation failed")


@dataclass(frozen=True, eq=False)
class BlockError(PicUtilsError):
    """A block failed while an experiment ran it (code ``block.raised``).

    Attributes:
        block: Registry name of the block that failed.
        step_id: Pipeline step that ran the block, when known.
    """

    block: str
    step_id: str | None = None

    def _where(self) -> str:
        return f" in step {self.step_id!r}" if self.step_id is not None else ""

    def _message(self) -> str:
        return f"block {self.block!r} failed{self._where()}"

    def __post_init__(self) -> None:
        super().__init__(self._message())


@dataclass(frozen=True, eq=False)
class InvalidBlockOutput(BlockError):
    """A block returned output that failed runtime validation.

    Code: ``block.invalid_output``.
    """

    def _message(self) -> str:
        return f"block {self.block!r} produced invalid output{self._where()}"


@dataclass(frozen=True, eq=False)
class ImageLoadError(PicUtilsError):
    """An image could not be loaded.

    Attributes:
        path: File that failed to load.
        reason: Human-readable failure description.
    """

    path: Path
    reason: str

    def _message(self) -> str:
        return f"cannot load {self.path}: {self.reason}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        super().__init__(self._message())


@dataclass(frozen=True, eq=False)
class ImageTooLargeError(ImageLoadError):
    """An image exceeds the maximum supported resolution (§4.2).

    Attributes:
        width: Pixel width of the rejected image.
        height: Pixel height of the rejected image.
    """

    width: int
    height: int

    def _message(self) -> str:
        return (
            f"cannot load {self.path}: image is {self.width}x{self.height}, "
            "above the maximum supported size"
        )


@dataclass(frozen=True, eq=False)
class UnsupportedImageError(ImageLoadError):
    """The file is an image but its format is not supported (§4.2)."""


class MemoryLimitError(PicUtilsError):
    """An experiment was deferred or failed: it exceeded the memory budget.

    Code: ``memory.limit``.
    """


class ExportError(PicUtilsError):
    """An export failed, e.g. the target file could not be written (§11)."""


class CanvasFileError(PicUtilsError):
    """A canvas document could not be saved or loaded (§10.8)."""


class ProjectFileError(PicUtilsError):
    """A project file could not be saved, loaded, or migrated (§17.1)."""


@dataclass(frozen=True, eq=False)
class RunError(PicUtilsError):
    """A batch run failed with a stable runtime error code (§18.2).

    Attributes:
        code: One of the runtime error codes defined above.
    """

    code: str

    def __post_init__(self) -> None:
        super().__init__(f"run failed: {self.code}")
