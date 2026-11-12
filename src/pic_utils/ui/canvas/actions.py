"""Canvas save/load/export actions and worker threads (§10.2, §11).

Save and export run on dedicated ``QThread`` objects so asset copies,
rendering, and file writes never block the GUI (§10.2, §11.4). Failures are
reported through ``notify`` without losing canvas state.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog, QWidget

from pic_utils.canvas.export import ExportOptions, ExportReport, export_canvas
from pic_utils.canvas.model import CanvasDoc, load_canvas, save_canvas
from pic_utils.core.errors import PicUtilsError
from pic_utils.settings import Settings

__all__ = ["CanvasActions", "ExportTask", "SaveTask"]

_CANVAS_FILTER = "Pic Utils canvas (*.pucanvas *.canvas)"
_CANVAS_SUFFIXES = (".pucanvas", ".canvas")


class SaveTask(QThread):
    """Copy assets and write the canvas JSON off the GUI thread (§10.2)."""

    finished_ok = Signal(object)  # Path
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(
        self,
        doc: CanvasDoc,
        path: Path,
        *,
        keep: set[str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._doc = doc
        self._path = path
        self._keep = keep

    def run(self) -> None:
        try:
            save_canvas(
                self._path,
                self._doc,
                progress=lambda done, total: self.progress.emit(done, total),
                keep=self._keep,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished_ok.emit(self._path)


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
                cancel=self.isInterruptionRequested,
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished_ok.emit(report)


class CanvasActions(QObject):
    """Save/open dialogs plus synchronous and threaded save/export."""

    notify = Signal(str)
    saved = Signal(Path)
    loaded = Signal(object)  # CanvasDoc
    exported = Signal(object)  # ExportReport
    export_failed = Signal(str)
    export_progress = Signal(int, int)
    save_progress = Signal(int, int)

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._path: Path | None = None
        self._export_task: ExportTask | None = None
        self._save_task: SaveTask | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    # --- save / load ---------------------------------------------------------

    def _choose_save_path(self, parent: QWidget | None) -> Path | None:
        chosen, _ = QFileDialog.getSaveFileName(
            parent,
            self.tr("Save canvas"),
            "",
            _CANVAS_FILTER,
        )
        if not chosen:
            return None
        target = Path(chosen)
        if target.suffix.lower() not in _CANVAS_SUFFIXES:
            target = target.with_suffix(".pucanvas")
        return target

    def save(
        self,
        doc: CanvasDoc,
        parent: QWidget | None = None,
        *,
        keep: set[str] | None = None,
    ) -> Path | None:
        """Save synchronously (tests, scripts, and the close prompt)."""
        target = self._path
        if target is None:
            target = self._choose_save_path(parent)
            if target is None:
                return None
        try:
            save_canvas(target, doc, keep=keep)
        except (PicUtilsError, OSError) as error:
            self.notify.emit(str(error))
            return None
        self._path = target
        self.saved.emit(target)
        return target

    def save_as(
        self,
        doc: CanvasDoc,
        parent: QWidget | None = None,
        *,
        keep: set[str] | None = None,
    ) -> Path | None:
        """Save under a new path; subsequent saves go there."""
        self._path = None
        return self.save(doc, parent, keep=keep)

    def save_async(
        self,
        doc: CanvasDoc,
        parent: QWidget | None = None,
        *,
        save_as: bool = False,
        keep: set[str] | None = None,
    ) -> SaveTask | None:
        """Start a worker-thread save; returns the task or ``None`` on cancel."""
        target = None if save_as else self._path
        if target is None:
            target = self._choose_save_path(parent)
            if target is None:
                return None
        task = SaveTask(doc, target, keep=keep, parent=self)
        task.finished_ok.connect(self._on_save_finished)
        task.failed.connect(self._on_save_failed)
        task.progress.connect(self.save_progress.emit)
        task.finished.connect(task.deleteLater)
        self._save_task = task
        task.start()
        return task

    def _on_save_finished(self, path: object) -> None:
        self._save_task = None
        if isinstance(path, Path):
            self._path = path
            self.saved.emit(path)

    def _on_save_failed(self, message: str) -> None:
        self._save_task = None
        self.notify.emit(message)

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
            self.tr("Open canvas"),
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
        self._export_task = task
        task.start()
        return task

    def _on_export_finished(self, report: object) -> None:
        self._export_task = None
        if isinstance(report, ExportReport):
            self.exported.emit(report)

    def _on_export_failed(self, message: str) -> None:
        self._export_task = None
        self.export_failed.emit(message)
        self.notify.emit(message)

    # --- lifecycle -----------------------------------------------------------

    def wait_for_export(self, timeout_ms: int = 30_000) -> bool:
        """Wait for the active export task (tests only)."""
        task = self._export_task
        return True if task is None else task.wait(timeout_ms)

    def wait_for_save(self, timeout_ms: int = 30_000) -> bool:
        """Wait for the active save task (tests only)."""
        task = self._save_task
        return True if task is None else task.wait(timeout_ms)

    def shutdown(self) -> None:
        """Stop worker tasks before the owning window is destroyed."""
        export = self._export_task
        if export is not None and export.isRunning():
            export.requestInterruption()
            export.wait()
        self._export_task = None
        save = self._save_task
        if save is not None and save.isRunning():
            save.wait()
        self._save_task = None
