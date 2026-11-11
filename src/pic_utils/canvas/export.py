"""Full-resolution canvas export (architecture §10.7, §11).

``render_canvas`` is the only full-resolution compositor. It loads one
layer at a time through a small LRU (size 2), applies the layer mask, and
draws with the same transform and blend mapping as the screen. ``export_canvas``
writes the canvas plus optional result files into a collision-safe export
folder, with an atomic write per file and an sRGB profile embedded by
:func:`pic_utils.io.saver.save_image`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QTransform

from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.model import CanvasDoc, CanvasLayer
from pic_utils.canvas.render import (
    BLEND_TO_QT,
    background_color,
    effective_opacity,
    layer_transform,
    layer_visible,
    qimage_from_image,
    qimage_to_image,
)
from pic_utils.core.errors import CanvasFileError, ExportError
from pic_utils.core.jsonio import write_json_atomic
from pic_utils.io.loader import load_image, load_mask
from pic_utils.io.saver import save_image

__all__ = [
    "EXPORT_FORMAT",
    "EXPORT_VERSION",
    "ExportOptions",
    "ExportReport",
    "export_canvas",
    "render_canvas",
]

EXPORT_FORMAT = "pic_utils.export"
EXPORT_VERSION = 1

ExportFormat = Literal["png", "jpeg", "tiff"]
ExportScope = Literal["canvas", "canvas_and_accepted", "everything"]

_EXTENSIONS: dict[str, str] = {"png": ".png", "jpeg": ".jpg", "tiff": ".tiff"}
_FORMATS = frozenset(_EXTENSIONS)
_SCOPES = frozenset({"canvas", "canvas_and_accepted", "everything"})


@dataclass(frozen=True)
class ExportOptions:
    """One export request from the export dialog (§11.1-§11.3)."""

    target_dir: Path
    fmt: str = "png"
    jpeg_quality: int = 92
    scope: str = "canvas"


@dataclass(frozen=True)
class ExportReport:
    """The files written by one export."""

    folder: Path
    files: tuple[Path, ...]
    scope: str
    format: str
    jpeg_quality: int
    timestamp: str


class _LayerLoader:
    """Load full-resolution layers one at a time, LRU size 2 (§10.7)."""

    def __init__(self, doc: CanvasDoc, max_cached: int = 2) -> None:
        self._store = AssetStore(doc.assets_dir) if doc.assets_dir is not None else None
        self._max_cached = max(1, max_cached)
        self._cache: OrderedDict[tuple[str, str | None], QImage] = OrderedDict()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def qimage(self, layer: CanvasLayer) -> QImage:
        key = (layer.image_ref, layer.mask_ref)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if self._store is None:
            raise CanvasFileError("canvas has no assets folder")
        image = load_image(self._store.resolve(layer.image_ref))
        mask = (
            load_mask(self._store.resolve(layer.mask_ref)) if layer.mask_ref is not None else None
        )
        qimage = qimage_from_image(image, mask)
        self._cache[key] = qimage
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return qimage


def render_canvas(doc: CanvasDoc, *, scale: float = 1.0) -> QImage:
    """Render the full-resolution composite; QImage painting is thread-safe."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    width = max(1, round(doc.width * scale))
    height = max(1, round(doc.height * scale))
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        background = background_color(doc.background)
        if background is not None:
            painter.fillRect(image.rect(), background)
        loader = _LayerLoader(doc)
        for layer in doc.layers:
            if not layer_visible(doc, layer):
                continue
            qimage = loader.qimage(layer)
            painter.save()
            painter.setOpacity(effective_opacity(doc, layer))
            painter.setCompositionMode(BLEND_TO_QT[layer.blend])
            transform = layer_transform(layer, qimage.width(), qimage.height())
            if scale != 1.0:
                transform = transform * QTransform().scale(scale, scale)
            painter.setTransform(transform)
            painter.drawImage(0, 0, qimage)
            painter.restore()
    finally:
        painter.end()
    return image


def _unique_export_dir(target_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = target_dir / f"export_{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = target_dir / f"export_{stamp}-{suffix}"
        suffix += 1
    return candidate


def _unique_result_name(stem: str, extension: str, used: set[str], folder: Path) -> str:
    candidate = f"{stem}{extension}"
    suffix = 2
    while candidate in used or (folder / candidate).exists():
        candidate = f"{stem}-{suffix}{extension}"
        suffix += 1
    used.add(candidate)
    return candidate


def export_canvas(
    doc: CanvasDoc,
    target_dir: str | Path,
    *,
    fmt: str = "png",
    jpeg_quality: int = 92,
    scope: str = "canvas",
    accepted: Sequence[Path] = (),
    run_results: Sequence[Path] = (),
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> ExportReport:
    """Render and write one export folder (§11.2); atomic per file."""
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}")
    if scope not in _SCOPES:
        raise ValueError(f"unsupported export scope {scope!r}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg quality must be in 1..100, got {jpeg_quality}")
    if cancel is not None and cancel():
        raise ExportError("export cancelled")

    parent = Path(target_dir)
    try:
        parent.mkdir(parents=True, exist_ok=True)
        folder = _unique_export_dir(parent)
        folder.mkdir()
    except OSError as error:
        raise ExportError(f"cannot create the export folder in {parent}: {error}") from error

    extension = _EXTENSIONS[fmt]
    timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    if scope == "canvas":
        result_paths: list[Path] = []
    elif scope == "canvas_and_accepted":
        result_paths = list(accepted)
    else:
        result_paths = list(run_results)
    total = 1 + len(result_paths)
    files: list[Path] = []

    canvas_name = f"canvas{extension}"
    canvas_path = folder / canvas_name
    try:
        save_image(
            qimage_to_image(render_canvas(doc)),
            canvas_path,
            jpeg_quality=jpeg_quality,
            embed_srgb=True,
        )
    except (OSError, ValueError) as error:
        raise ExportError(f"cannot write {canvas_path}: {error}") from error
    files.append(canvas_path)
    if progress is not None:
        progress(1, total)

    result_names: list[str] = []
    if result_paths:
        results_folder = folder / "results"
        results_folder.mkdir()
        used: set[str] = set()
        for done, source in enumerate(result_paths, start=2):
            if cancel is not None and cancel():
                raise ExportError("export cancelled")
            source_path = Path(source)
            name = _unique_result_name(source_path.stem, extension, used, results_folder)
            target = results_folder / name
            try:
                image = load_image(source_path)
                save_image(
                    image,
                    target,
                    jpeg_quality=jpeg_quality,
                    embed_srgb=True,
                )
            except (OSError, ValueError) as error:
                raise ExportError(f"cannot export {source_path}: {error}") from error
            result_names.append(name)
            files.append(target)
            if progress is not None:
                progress(done, total)

    manifest = {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "scope": scope,
        "image_format": fmt,
        "jpeg_quality": jpeg_quality,
        "timestamp": timestamp,
        "canvas": canvas_name,
        "results": result_names,
        "files": [canvas_name, *(f"results/{name}" for name in result_names)],
    }
    try:
        write_json_atomic(folder / "manifest.json", manifest)
    except OSError as error:
        raise ExportError(f"cannot write the export manifest: {error}") from error
    return ExportReport(
        folder=folder,
        files=tuple(files),
        scope=scope,
        format=fmt,
        jpeg_quality=jpeg_quality,
        timestamp=timestamp,
    )
