"""Typed application settings over ``QSettings`` (architecture §16).

The backing store is an INI file at ``QStandardPaths.AppConfigLocation /
settings.ini``. Every key from the §16 table has a typed accessor with a
default and, where the table specifies a range, clamping on write. Setting a
value emits :attr:`changed` with the key so live consumers (language, theme,
auto-reload) can react without a restart.

Standard locations (§16) are resolved here — ``core/paths.py`` stays Qt-free
by design. Paths are created on demand; failures are logged, never fatal.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QObject, QSettings, QStandardPaths, Signal

from pic_utils.runner.service import RunnerSettings

__all__ = [
    "LANGUAGE_OPTIONS",
    "THEME_OPTIONS",
    "Settings",
    "app_config_dir",
    "app_data_dir",
    "default_blocks_folder",
    "default_results_root",
    "logs_dir",
]

logger = logging.getLogger(__name__)

LANGUAGE_OPTIONS = ("system", "en", "es")
THEME_OPTIONS = ("system", "light", "dark")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

_ORGANIZATION = "PicUtils"
_APPLICATION = "PicUtils"

_EXAMPLE_BLOCK_NAME = "example_blocks.py"

_EXAMPLE_BLOCK = '''"""Example user blocks: edit or delete this file.

Reload blocks (Tools menu) or save the file to hot-reload. A file that fails
to import keeps its last-good definitions and reports the error non-modally.
"""

from typing import Annotated

import numpy as np

from pic_utils.blocks import IntRange, OutputKind, Seed, block
from pic_utils.core.image import Image


@block(
    name="posterize",
    category="examples",
    description="Reduce each channel to N evenly spaced levels.",
    output=OutputKind.IMAGE,
)
def posterize(
    image: Image,
    levels: Annotated[int, IntRange(2, 32, step=1)] = 6,
    seed: Annotated[int | None, Seed()] = None,
) -> Image:
    rgb = image.rgb.astype(np.float32) / 255.0
    quantized = np.round(rgb * (levels - 1)) / (levels - 1)
    out = np.clip(quantized * 255.0, 0, 255).astype(np.uint8)
    if seed is not None:
        noise = np.random.default_rng(seed).uniform(-0.5, 0.5, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.from_rgb(out)
'''


def _ensure_app_identity() -> None:
    """Make standard locations deterministic even before ``QApplication``."""
    if not QCoreApplication.organizationName():
        QCoreApplication.setOrganizationName(_ORGANIZATION)
    if not QCoreApplication.applicationName():
        QCoreApplication.setApplicationName(_APPLICATION)


def app_config_dir() -> Path:
    """``AppConfigLocation`` (settings, user blocks)."""
    _ensure_app_identity()
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation))


def app_data_dir() -> Path:
    """``AppDataLocation`` (logs, presets, results fallback)."""
    _ensure_app_identity()
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))


def logs_dir() -> Path:
    """``AppDataLocation/logs`` (§16)."""
    return app_data_dir() / "logs"


def default_blocks_folder() -> Path:
    """``AppConfigLocation/blocks`` — created on first run with an example."""
    return app_config_dir() / "blocks"


def default_results_root() -> Path:
    """``PicturesLocation/PicUtils``, falling back to ``AppDataLocation/results``."""
    pictures = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
    if pictures:
        return Path(pictures) / "PicUtils"
    return app_data_dir() / "results"


class Settings(QObject):
    """Typed, validated view of the ``settings.ini`` keys (§16)."""

    #: Emitted with the changed key whenever a value is written.
    changed = Signal(str)

    def __init__(self, path: str | Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        _ensure_app_identity()
        self._path = Path(path) if path is not None else app_config_dir() / "settings.ini"
        self._settings = QSettings(str(self._path), QSettings.Format.IniFormat)

    @property
    def path(self) -> Path:
        return self._path

    # --- raw access ---------------------------------------------------------

    def _value(self, key: str, default: object) -> object:
        value = self._settings.value(key, default)
        return default if value is None else value

    def _set(self, key: str, value: object) -> None:
        if self._settings.value(key) == value:
            return
        self._settings.setValue(key, value)
        self._settings.sync()
        self.changed.emit(key)

    # --- ui -----------------------------------------------------------------

    @property
    def language(self) -> str:
        value = str(self._value("ui/language", LANGUAGE_OPTIONS[0]))
        return value if value in LANGUAGE_OPTIONS else LANGUAGE_OPTIONS[0]

    def set_language(self, value: str) -> None:
        if value not in LANGUAGE_OPTIONS:
            raise ValueError(f"language must be one of {LANGUAGE_OPTIONS}, got {value!r}")
        self._set("ui/language", value)

    @property
    def theme(self) -> str:
        value = str(self._value("ui/theme", THEME_OPTIONS[0]))
        return value if value in THEME_OPTIONS else THEME_OPTIONS[0]

    def set_theme(self, value: str) -> None:
        if value not in THEME_OPTIONS:
            raise ValueError(f"theme must be one of {THEME_OPTIONS}, got {value!r}")
        self._set("ui/theme", value)

    # --- paths ----------------------------------------------------------------

    @property
    def blocks_folder(self) -> Path:
        return Path(str(self._value("paths/blocks_folder", str(default_blocks_folder()))))

    def set_blocks_folder(self, value: Path) -> None:
        self._set("paths/blocks_folder", str(value))

    @property
    def results_root(self) -> Path:
        return Path(str(self._value("paths/results_root", str(default_results_root()))))

    def set_results_root(self, value: Path) -> None:
        self._set("paths/results_root", str(value))

    @property
    def trash_root(self) -> Path:
        """``<results_root>/trash`` (§16); derived, not stored."""
        return self.results_root / "trash"

    # --- runner -----------------------------------------------------------------

    def _int(self, key: str, default: int, low: int, high: int) -> int:
        try:
            value = int(self._value(key, default))  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    @staticmethod
    def _clamp(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))

    @property
    def workers(self) -> int:
        return self._int("runner/workers", 2, 1, 8)

    def set_workers(self, value: int) -> None:
        self._set("runner/workers", self._clamp(value, 1, 8))

    @property
    def memory_ceiling_mb(self) -> int:
        return self._int("runner/memory_ceiling_mb", 8192, 512, 8192)

    def set_memory_ceiling_mb(self, value: int) -> None:
        clamped = self._clamp(value, 512, 8192)
        self._set("runner/memory_ceiling_mb", clamped)

    @property
    def experiment_timeout_s(self) -> int:
        return self._int("runner/experiment_timeout_s", 300, 1, 86_400)

    def set_experiment_timeout_s(self, value: int) -> None:
        clamped = self._clamp(value, 1, 86_400)
        self._set("runner/experiment_timeout_s", clamped)

    @property
    def auto_reload_blocks(self) -> bool:
        return bool(self._value("runner/auto_reload_blocks", True))

    def set_auto_reload_blocks(self, value: bool) -> None:
        self._set("runner/auto_reload_blocks", bool(value))

    def runner_settings(self) -> RunnerSettings:
        """The runner-facing slice (workers, ceiling, timeout)."""
        return RunnerSettings(
            workers=self.workers,
            memory_ceiling_mb=self.memory_ceiling_mb,
            experiment_timeout_s=float(self.experiment_timeout_s),
        )

    # --- preview ----------------------------------------------------------------

    @property
    def proxy_long_side(self) -> int:
        return self._int("preview/proxy_long_side", 1024, 128, 4096)

    def set_proxy_long_side(self, value: int) -> None:
        self._set("preview/proxy_long_side", self._clamp(value, 128, 4096))

    @property
    def debounce_ms(self) -> int:
        return self._int("preview/debounce_ms", 150, 0, 5_000)

    def set_debounce_ms(self, value: int) -> None:
        self._set("preview/debounce_ms", self._clamp(value, 0, 5_000))

    # --- export -----------------------------------------------------------------

    @property
    def jpeg_quality(self) -> int:
        return self._int("export/jpeg_quality", 92, 1, 100)

    def set_jpeg_quality(self, value: int) -> None:
        self._set("export/jpeg_quality", self._clamp(value, 1, 100))

    @property
    def tiff_compression(self) -> str:
        return str(self._value("export/tiff_compression", "tiff_lzw"))

    def set_tiff_compression(self, value: str) -> None:
        self._set("export/tiff_compression", value)

    # --- undo / logging -----------------------------------------------------------

    @property
    def undo_limit(self) -> int:
        return self._int("undo/limit", 100, 1, 10_000)

    def set_undo_limit(self, value: int) -> None:
        self._set("undo/limit", self._clamp(value, 1, 10_000))

    @property
    def log_level(self) -> str:
        value = str(self._value("logging/level", "INFO"))
        return value if value in LOG_LEVELS else "INFO"

    def set_log_level(self, value: str) -> None:
        if value not in LOG_LEVELS:
            raise ValueError(f"log level must be one of {LOG_LEVELS}, got {value!r}")
        self._set("logging/level", value)

    # --- paths on demand ---------------------------------------------------------

    def ensure_paths(self) -> None:
        """Create the configured paths (§16); failures are logged, not fatal.

        The blocks folder is created on first run with an example block so a
        fresh install has something to reload.
        """
        created_blocks = not self.blocks_folder.is_dir()
        for directory in (self.blocks_folder, self.results_root):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                logger.warning("could not create %s: %s", directory, error)
        if created_blocks:
            example = self.blocks_folder / _EXAMPLE_BLOCK_NAME
            try:
                example.write_text(_EXAMPLE_BLOCK, encoding="utf-8")
            except OSError as error:
                logger.warning("could not write the example block %s: %s", example, error)
