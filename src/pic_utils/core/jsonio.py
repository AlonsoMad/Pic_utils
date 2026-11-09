"""JSON-safe conversion and atomic JSON writes (architecture §2, §9.2).

``write_json_atomic`` is the atomic-save primitive required by NFR-REL-03:
a failed write leaves the target file and its directory untouched — no
partial files, no leftover temp files.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["jsonable", "read_json", "write_json_atomic"]

_JSON_INDENT = 2
_TEMP_PREFIX = ".pic_utils-"
_TEMP_SUFFIX = ".tmp"


def jsonable(value: Any) -> Any:
    """Convert ``value`` to JSON-safe Python types, recursing into containers.

    numpy scalars become Python scalars (``np.float32(1.5)`` -> ``1.5``),
    numpy arrays become nested lists, tuples become lists. Dict keys are
    converted the same way: numpy scalar keys become Python scalars and any
    other non-JSON key type raises ``TypeError``. Values that are already
    JSON-safe (``bool``, ``int``, ``float``, ``str``, ``None``) pass through
    unchanged. Any other object raises ``TypeError``.

    Raises:
        TypeError: If a value (or dict key) cannot be represented as JSON.
    """
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, np.generic):
                key = key.item()
            if not isinstance(key, str | int | float) and key is not None:
                msg = f"JSON object keys must be str, int, float or None, got {type(key).__name__}"
                raise TypeError(msg)
            converted[key] = jsonable(item)
        return converted
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    msg = f"object of type {type(value).__name__} is not JSON-serializable"
    raise TypeError(msg)


def write_json_atomic(path: str | os.PathLike[str], data: Any) -> None:
    """Serialize ``data`` and atomically replace the JSON file at ``path``.

    The payload is converted with :func:`jsonable` and serialized to a string
    BEFORE any file is touched, so a non-serializable payload raises
    ``TypeError`` without creating anything. The string is then written to a
    unique temp file in the same directory and moved into place with
    ``os.replace`` — readers see either the old or the new file, never a
    partial one (NFR-REL-03). A failed write leaves the original file intact
    and removes the temp file.

    Raises:
        TypeError: If ``data`` is not JSON-serializable (nothing is written).
        OSError: If the file cannot be written or replaced.
    """
    payload = json.dumps(jsonable(data), ensure_ascii=False, indent=_JSON_INDENT) + "\n"
    target = Path(path)
    handle_fd, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read and parse the JSON file at ``path``.

    Raises:
        FileNotFoundError: If ``path`` does not exist (stdlib error, includes
            the path).
        ValueError: If the file exists but is not valid UTF-8 text or not
            valid JSON; the message names the file and the failure position.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        msg = f"{source}: not valid UTF-8 text: {error.reason}"
        raise ValueError(msg) from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"{source}: invalid JSON: {error.msg} (line {error.lineno}, column {error.colno})"
        raise ValueError(msg) from error
