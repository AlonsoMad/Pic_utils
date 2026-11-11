"""Block registry and immutable snapshots (architecture §5.4-§5.5).

Built-ins are discovered from ``pic_utils.blocks.builtin`` with
``pkgutil.iter_modules``. User blocks are imported from a configured folder;
a file that fails to import keeps its last-good definitions. Every reload
swaps an immutable snapshot atomically and emits ``changed``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pkgutil
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from PySide6.QtCore import QObject, Signal

from pic_utils.blocks import builtin as builtin_package
from pic_utils.blocks import userblocks
from pic_utils.blocks.api import BlockDef, BlockLoadError

__all__ = ["BlockRegistry", "BlockSnapshot", "ReloadReport"]


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


@dataclass(frozen=True, slots=True)
class ReloadReport:
    """The diff and errors produced by one reload."""

    loaded: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...]
    errors: tuple[BlockLoadError, ...]
    duration_ms: float


class BlockRegistry(QObject):
    """Discover built-ins and user blocks; expose immutable snapshots."""

    changed = Signal(object)

    def __init__(
        self,
        user_blocks_dir: Path | None = None,
        *,
        include_builtins: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._user_blocks_dir = Path(user_blocks_dir) if user_blocks_dir is not None else None
        self._include_builtins = include_builtins
        self._lock = threading.RLock()
        self._snapshot = BlockSnapshot(MappingProxyType({}), "")
        self._file_defs: dict[Path, tuple[BlockDef, ...]] = {}
        self._file_hashes: dict[Path, str] = {}
        self.reload()

    @property
    def user_blocks_dir(self) -> Path | None:
        return self._user_blocks_dir

    def set_user_blocks_dir(self, path: Path | None) -> ReloadReport:
        """Change the user blocks folder and reload."""
        with self._lock:
            self._user_blocks_dir = Path(path) if path is not None else None
            self._file_defs = {}
            self._file_hashes = {}
        return self.reload()

    def snapshot(self) -> BlockSnapshot:
        """Return the current immutable snapshot."""
        with self._lock:
            return self._snapshot

    def reload(self) -> ReloadReport:
        """Rebuild the snapshot, keeping each file's last-good blocks on error."""
        started = time.perf_counter()
        with self._lock:
            previous = self._snapshot
            errors: list[BlockLoadError] = []
            builtins = self._load_builtins(errors) if self._include_builtins else {}
            user_defs, file_hashes = self._load_user_blocks(errors)

            definitions: dict[str, BlockDef] = dict(builtins)
            for path in sorted(user_defs):
                for definition in user_defs[path]:
                    if definition.name in definitions:
                        errors.append(
                            BlockLoadError(
                                reason=f"duplicate block name {definition.name!r}",
                                path=path,
                                block=definition.name,
                            )
                        )
                        continue
                    definitions[definition.name] = definition

            snapshot = BlockSnapshot(
                MappingProxyType(definitions),
                self._snapshot_hash(definitions, file_hashes),
            )
            self._snapshot = snapshot
            self._file_defs = user_defs
            self._file_hashes = file_hashes

            report = ReloadReport(
                loaded=tuple(sorted(set(definitions) - set(previous.defs))),
                updated=tuple(
                    sorted(
                        name
                        for name in definitions
                        if name in previous.defs
                        and definitions[name].content_hash != previous.defs[name].content_hash
                    )
                ),
                removed=tuple(sorted(set(previous.defs) - set(definitions))),
                errors=tuple(errors),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
        self.changed.emit(report)
        return report

    def _load_builtins(self, errors: list[BlockLoadError]) -> dict[str, BlockDef]:
        definitions: dict[str, BlockDef] = {}
        for module_info in pkgutil.iter_modules(builtin_package.__path__):
            if module_info.name.startswith("_"):
                continue
            module_name = f"{builtin_package.__name__}.{module_info.name}"
            try:
                module = importlib.import_module(module_name)
            except Exception as error:
                errors.append(
                    BlockLoadError(
                        reason=f"built-in import failed: {error}",
                        block=module_name,
                    )
                )
                continue
            for definition in userblocks.collect_block_defs(
                module,
                source=None,
                content_hash="",
            ):
                if definition.name in definitions:
                    errors.append(
                        BlockLoadError(
                            reason=f"duplicate built-in block {definition.name!r}",
                            block=definition.name,
                        )
                    )
                    continue
                definitions[definition.name] = definition
        return definitions

    def _load_user_blocks(
        self,
        errors: list[BlockLoadError],
    ) -> tuple[dict[Path, tuple[BlockDef, ...]], dict[Path, str]]:
        folder = self._user_blocks_dir
        if folder is None or not folder.is_dir():
            return {}, {}

        try:
            current_hashes = userblocks.hash_user_files(folder)
        except OSError as error:
            errors.append(BlockLoadError(reason=f"cannot hash user blocks: {error}", path=folder))
            return dict(self._file_defs), dict(self._file_hashes)

        userblocks.purge_user_modules(folder)
        definitions: dict[Path, tuple[BlockDef, ...]] = {}
        hashes: dict[Path, str] = {}

        for path in userblocks.iter_block_files(folder):
            relative = path.relative_to(folder)
            current_hash = current_hashes.get(relative, "")
            try:
                module = userblocks.load_user_module(path, folder)
                file_defs = userblocks.collect_block_defs(
                    module,
                    source=path,
                    content_hash=current_hash,
                )
            except Exception as error:
                errors.append(
                    BlockLoadError(
                        reason=f"user block import failed: {error}",
                        path=path,
                    )
                )
                previous = self._file_defs.get(path)
                if previous is not None:
                    definitions[path] = previous
                    if path in self._file_hashes:
                        hashes[path] = self._file_hashes[path]
                continue
            definitions[path] = file_defs
            hashes[path] = current_hash

        for relative, digest in current_hashes.items():
            absolute = folder / relative
            if absolute not in hashes:
                hashes[absolute] = digest

        return definitions, hashes

    def _snapshot_hash(
        self,
        definitions: Mapping[str, BlockDef],
        file_hashes: Mapping[Path, str],
    ) -> str:
        builtin_entries = sorted(
            (definition.name, definition.content_hash, definition.api_version)
            for definition in definitions.values()
            if definition.source is None
        )
        folder = self._user_blocks_dir
        if folder is None:
            user_entries: list[tuple[str, str]] = []
        else:
            user_entries = sorted(
                (str(path.relative_to(folder)), digest) for path, digest in file_hashes.items()
            )
        payload = json.dumps(
            {"builtins": builtin_entries, "user": user_entries},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
