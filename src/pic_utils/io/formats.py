"""Supported image formats and the resolution limit (architecture §4).

This is the only place that maps a filename extension to a Pillow format
name; the loader trusts the format Pillow detects in the file itself.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "MAX_PIXELS",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FORMATS",
    "format_for_path",
    "is_supported_path",
]

MAX_PIXELS = 48_000_000

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}

SUPPORTED_FORMATS: frozenset[str] = frozenset(SUPPORTED_EXTENSIONS.values())


def format_for_path(path: str | Path) -> str:
    """Return the Pillow format for ``path``'s extension.

    Raises ``ValueError`` when the extension is not one of PNG/JPEG/TIFF.
    """
    suffix = Path(path).suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[suffix]
    except KeyError:
        raise ValueError(f"unsupported image extension: {suffix or '<none>'}") from None


def is_supported_path(path: str | Path) -> bool:
    """Return whether ``path`` has a supported output extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
