"""User block discovery and import helpers (architecture §5.5)."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import shutil
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from pic_utils.blocks.api import BlockDef

__all__ = [
    "USERBLOCKS_PACKAGE",
    "collect_block_defs",
    "hash_user_files",
    "iter_block_files",
    "load_user_module",
    "purge_user_modules",
]

USERBLOCKS_PACKAGE = "pic_utils_userblocks"


def iter_block_files(folder: Path) -> list[Path]:
    """Return the top-level user block files, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )


def hash_user_files(folder: Path) -> dict[Path, str]:
    """Hash every ``.py`` file under ``folder``, keyed by relative path."""
    hashes: dict[Path, str] = {}
    if not folder.is_dir():
        return hashes
    for path in sorted(folder.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hashes[path.relative_to(folder)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def purge_user_modules(folder: Path) -> None:
    """Remove the user-block package and every module loaded from ``folder``."""
    if not folder.is_dir():
        return
    resolved = folder.resolve()
    for cache_dir in folder.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
    for name, module in list(sys.modules.items()):
        if name == USERBLOCKS_PACKAGE or name.startswith(f"{USERBLOCKS_PACKAGE}."):
            sys.modules.pop(name, None)
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            path = Path(module_file).resolve()
        except OSError:
            continue
        if path.is_relative_to(resolved):
            sys.modules.pop(name, None)


def load_user_module(path: Path, folder: Path) -> ModuleType:
    """Import one top-level user block file as ``pic_utils_userblocks.<stem>``."""
    module_name = f"{USERBLOCKS_PACKAGE}.{path.stem}"
    package = sys.modules.get(USERBLOCKS_PACKAGE)
    if package is None:
        package = types.ModuleType(USERBLOCKS_PACKAGE)
        package.__path__ = [str(folder)]
        sys.modules[USERBLOCKS_PACKAGE] = package

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    folder_text = str(folder)
    inserted = folder_text not in sys.path
    if inserted:
        sys.path.insert(0, folder_text)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(folder_text)
    return module


def collect_block_defs(
    module: ModuleType,
    *,
    source: Path | None,
    content_hash: str,
) -> tuple[BlockDef, ...]:
    """Collect the blocks defined in ``module`` with registry metadata."""
    definitions: list[BlockDef] = []
    for _, member in inspect.getmembers(module, inspect.isfunction):
        definition = getattr(member, "__block_def__", None)
        if not isinstance(definition, BlockDef):
            continue
        if definition.fn.__module__ != module.__name__:
            continue
        definitions.append(
            replace(
                definition,
                module=module.__name__,
                source=source,
                content_hash=content_hash,
            )
        )
    return tuple(sorted(definitions, key=lambda item: item.name))
