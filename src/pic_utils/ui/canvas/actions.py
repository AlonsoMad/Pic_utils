"""Canvas save/load/export actions and the export worker thread (§10.2, §11).

Save and load are synchronous (the page shows a busy cursor); export runs in
a dedicated ``QThread`` so rendering and file writes never block the GUI
(§11.4). Failures are reported through ``notify`` without losing canvas
state.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog, QWidget

from pic_utils.canvas.export import ExportOptions, ExportReport, export_canvas
from pic_utils.canvas.model import CanvasDoc, load_canvas, save_canvas
from pic_utils.core.errors import PicUtilsError
from pic_utils.settings import Settings

__all__ = ["CanvasActions", "ExportTask"]

_CANVAS_FILTER = "Pic Utils canvas (*.pucanvas *.canvas)"
_CANVAS_SUFFIXES = (".pucanvas", ".canvas")


class ExportTask(QThread):
    """Render and write one export off the GUI thread (§11.4)."""

    finished_ok = Signal(object)  # ExportReport
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(
        self,
        doc: CanvasDoc,
        options: ExportOptions,
        *,
        accepted: tuple[Path, ...] = (),
        run_results: tuple[Path, ...] = (),
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._doc = doc
        self._options = options
        self._accepted = accepted
        self._run_results = run_results

    def run(self) -> None:
        try:
            report = export_canvas(
                self._doc,
                self._options.target_dir,
                fmt=self._options.fmt,
                jpeg_quality=self._options.jpeg_quality,
                scope=self._options.scope,
                accepted=self._accepted,
                run_results=self._run_results,
                progress=lambda done, total: self.progress.emit(done, total),
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished_ok.emit(report)


class CanvasActions(QObject):
    """Save/open dialogs plus synchronous and threaded export."""

    notify = Signal(str)
    saved = Signal(Path)
    loaded = Signal(object)  # CanvasDoc
    exported = Signal(object)  # ExportReport
    export_failed = Signal(str)
    export_progress = Signal(int, int)

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._path: Path | None = None
        self._task: ExportTask | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    # --- save / load ---------------------------------------------------------

    def save(self, doc: CanvasDoc, parent: QWidget | None = None) -> Path | None:
        """Save to the current path, or ask for one; returns the path."""
        target = self._path
        if target is None:
            chosen, _ = QFileDialog.getSaveFileName(
                parent,
                parent.tr("Save canvas") if parent else "Save canvas",
                "",
                _CANVAS_FILTER,
            )
            if not chosen:
                return None
            target = Path(chosen)
            if target.suffix.lower() not in _CANVAS_SUFFIXES:
                target = target.with_suffix(".pucanvas")
        try:
            save_canvas(target, doc)
        except (PicUtilsError, OSError) as error:
            self.notify.emit(str(error))
            return None
        self._path = target
        self.saved.emit(target)
        return target

    def save_as(self, doc: CanvasDoc, parent: QWidget | None = None) -> Path | None:
        """Save under a new path; subsequent saves go there."""
        self._path = None
        return self.save(doc, parent)

    def open(self, path: str | Path, parent: QWidget | None = None) -> CanvasDoc | None:
        """Load a canvas file; missing assets are reported without opening."""
        try:
            doc = load_canvas(path)
        except (PicUtilsError, OSError) as error:
            self.notify.emit(str(error))
            return None
        self._path = Path(path)
        self.loaded.emit(doc)
        return doc

    def open_dialog(self, parent: QWidget | None = None) -> CanvasDoc | None:
        chosen, _ = QFileDialog.getOpenFileName(
            parent,
            parent.tr("Open canvas") if parent else "Open canvas",
            "",
            _CANVAS_FILTER,
        )
        if not chosen:
            return None
        return self.open(chosen, parent)

    # --- export --------------------------------------------------------------

    def export(
        self,
        doc: CanvasDoc,
        options: ExportOptions,
        *,
        accepted: tuple[Path, ...] = (),
        run_results: tuple[Path, ...] = (),
    ) -> ExportReport | None:
        """Synchronous export (tests and scripts)."""
        try:
            report = export_canvas(
                doc,
                options.target_dir,
                fmt=options.fmt,
                jpeg_quality=options.jpeg_quality,
                scope=options.scope,
                accepted=accepted,
                run_results=run_results,
            )
        except (PicUtilsError, OSError) as error:
            self.notify.emit(str(error))
            return None
        self.exported.emit(report)
        return report

    def export_async(
        self,
        doc: CanvasDoc,
        options: ExportOptions,
        *,
        accepted: tuple[Path, ...] = (),
        run_results: tuple[Path, ...] = (),
    ) -> ExportTask:
        """Start an export on a worker thread; returns the task."""
        task = ExportTask(
            doc,
            options,
            accepted=accepted,
            run_results=run_results,
            parent=self,
        )
        task.finished_ok.connect(self._on_export_finished)
        task.failed.connect(self._on_export_failed)
        task.progress.connect(self.export_progress.emit)
        task.finished.connect(task.deleteLater)
        self._task = task
        task.start()
        return task

    def _on_export_finished(self, report: object) -> None:
        self._task = None
        if isinstance(report, ExportReport):
            self.exported.emit(report)

    def _on_export_failed(self, message: str) -> None:
        self._task = None
        self.export_failed.emit(message)
        self.notify.emit(message)

    def wait_for_export(self, timeout_ms: int = 30_000) -> bool:
        """Wait for the active export task (tests only)."""
        task = self._task
        return True if task is None else task.wait(timeout_ms)
