"""Qt-free block snapshot and snapshot hashing (architecture §5.4, §8.2).

``BlockSnapshot`` and :func:`snapshot_hash` live here — not in
``registry.py`` — because spawned workers must build and verify snapshots
without importing Qt (§1.3: worker code must not import Qt). The registry
imports this module and re-exports ``BlockSnapshot``, so every existing
import path keeps working; the worker imports it from here directly.

:func:`snapshot_hash` is THE hash function: the registry and the worker
both call it, so the "mirror the registry's algorithm" duplication (and
its drift risk) is gone.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pic_utils.blocks.api import BlockDef

__all__ = ["BlockSnapshot", "snapshot_hash"]


@dataclass(frozen=True, slots=True)
class BlockSnapshot:
    """An immutable view of the blocks a run binds to."""

    defs: Mapping[str, BlockDef]
    hash: str

    def get(self, name: str) -> BlockDef:
        return self.defs[name]

    def __contains__(self, name: object) -> bool:
        return name in self.defs

    def __iter__(self) -> Iterator[str]:
        return iter(self.defs)

    def __len__(self) -> int:
        return len(self.defs)


def snapshot_hash(
    definitions: Mapping[str, BlockDef],
    file_hashes: Mapping[Path, str],
    *,
    user_blocks_dir: Path | None,
) -> str:
    """Hash the snapshot: builtin entries plus user-file digests (§5.4).

    ``definitions`` maps block name -> definition (any mapping); entries
    whose definition has ``source is None`` are built-ins. ``file_hashes``
    maps user-block file path -> content digest. ``user_blocks_dir`` is the
    folder the paths are relative to (``None`` means no user blocks).

    The registry and the spawned worker call this same function, so a run's
    ``block_snapshot_hash`` is reproducible on both sides by construction.
    """
    builtin_entries = sorted(
        (name, definition.content_hash, definition.api_version)
        for name, definition in definitions.items()
        if definition.source is None
    )
    if user_blocks_dir is None:
        user_entries: list[tuple[str, str]] = []
    else:
        user_entries = sorted(
            (str(path.relative_to(user_blocks_dir)), digest) for path, digest in file_hashes.items()
        )
    payload = json.dumps(
        {"builtins": builtin_entries, "user": user_entries},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
