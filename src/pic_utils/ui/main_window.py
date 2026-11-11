"""Application shell: mode switch, top bar, menus, status bar (§13.1).

The window hosts the Lab and Canvas pages in a ``QStackedWidget``. Run
controls (Lab mode only) and the progress bar live in the top bar; drag and
drop on the window adds image sources and opens project/canvas files.
"""

from __future__ import annotations

import platform
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QActionGroup, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pic_utils import __version__
from pic_utils.settings import Settings, logs_dir
from pic_utils.ui.canvas.page import CanvasPage
from pic_utils.ui.dialogs.settings_dialog import SettingsDialog
from pic_utils.ui.i18n import LanguageManager, retranslate_tree
from pic_utils.ui.lab.page import LabPage
from pic_utils.ui.lab.project_actions import CANVAS_SUFFIX, IMAGE_SUFFIXES, PROJECT_SUFFIX

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Single window with a Lab mode and a Canvas mode (FR-UI-01)."""

    sources_dropped = Signal(list)  # list[Path]
    project_open_requested = Signal(Path)
    canvas_open_requested = Signal(Path)
    blocks_reload_requested = Signal()

    def __init__(
        self,
        settings: Settings,
        languages: LanguageManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._languages = languages

        self._lab_page = LabPage(settings)
        self._canvas_page = CanvasPage(settings)
        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._lab_page)
        self._stack.addWidget(self._canvas_page)

        # Lab glue: project open, blocks reload, run progress (P5-10).
        self.project_open_requested.connect(self._lab_page.open_project)
        self.blocks_reload_requested.connect(self._lab_page.library().reload_now)
        self._lab_page.notify.connect(self.statusBar().showMessage)
        self._lab_page.launch_requested.connect(self._on_launch_requested)

        # Canvas glue: open, save, export, and status (P6-07).
        self.canvas_open_requested.connect(self._canvas_page.open_canvas)
        self._canvas_page.notify.connect(self.statusBar().showMessage)
        self._canvas_page.canvas_status.connect(self.statusBar().showMessage)

        self._progress = QProgressBar(self)
        self._progress.hide()
        self._run_status = QLabel(self)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_top_bar(central))
        layout.addWidget(self._stack)
        self.set_run_controls(self._lab_page.run_controls())

        self.setCentralWidget(central)
        self.setAcceptDrops(True)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().addWidget(self._run_status)
        self.statusBar().addPermanentWidget(self._memory_label())

        self._build_menus()
        self.retranslate()

    # --- construction helpers ------------------------------------------------

    def _build_top_bar(self, parent: QWidget) -> QToolBar:
        bar = QToolBar(parent)
        bar.setMovable(False)
        self._mode_lab = QAction(parent)
        self._mode_lab.setCheckable(True)
        self._mode_canvas = QAction(parent)
        self._mode_canvas.setCheckable(True)
        self._run_controls_host = QWidget(parent)
        host_layout = QHBoxLayout(self._run_controls_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self._run_controls_host)
        bar.addSeparator()
        bar.addWidget(self._progress)
        return bar

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        self._file_menu = QMenu(menu_bar)
        self._tools_menu = QMenu(menu_bar)
        self._help_menu = QMenu(menu_bar)
        menu_bar.addMenu(self._file_menu)
        menu_bar.addMenu(self._tools_menu)
        menu_bar.addMenu(self._help_menu)

        self._new_canvas = QAction(self._file_menu)
        self._new_canvas.triggered.connect(self._canvas_page.new_document)
        self._file_menu.addAction(self._new_canvas)

        self._open_project = QAction(self._file_menu)
        self._open_project.setShortcut(QKeySequence.StandardKey.Open)
        self._open_project.triggered.connect(self._open_project_dialog)
        self._file_menu.addAction(self._open_project)

        self._open_canvas = QAction(self._file_menu)
        self._open_canvas.triggered.connect(self._open_canvas_dialog)
        self._file_menu.addAction(self._open_canvas)

        self._save_canvas = QAction(self._file_menu)
        self._save_canvas.setShortcut(QKeySequence.StandardKey.Save)
        self._save_canvas.triggered.connect(lambda: self._canvas_page.save_canvas(self))
        self._file_menu.addAction(self._save_canvas)

        self._save_canvas_as = QAction(self._file_menu)
        self._save_canvas_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._save_canvas_as.triggered.connect(lambda: self._canvas_page.save_canvas_as(self))
        self._file_menu.addAction(self._save_canvas_as)

        self._export_canvas = QAction(self._file_menu)
        self._export_canvas.setShortcut(QKeySequence("Ctrl+E"))
        self._export_canvas.triggered.connect(lambda: self._canvas_page.export_canvas(self))
        self._file_menu.addAction(self._export_canvas)

        self._reload_blocks = QAction(self._tools_menu)
        self._reload_blocks.triggered.connect(self._reload_blocks_slot)
        self._tools_menu.addAction(self._reload_blocks)

        self._settings_action = QAction(self._tools_menu)
        self._settings_action.setShortcut(QKeySequence("Ctrl+,"))
        self._settings_action.triggered.connect(self._open_settings)
        self._tools_menu.addAction(self._settings_action)

        self._copy_diagnostics = QAction(self._help_menu)
        self._copy_diagnostics.triggered.connect(self._copy_diagnostics_slot)
        self._help_menu.addAction(self._copy_diagnostics)

        mode_group = QActionGroup(self)
        mode_group.addAction(self._mode_lab)
        mode_group.addAction(self._mode_canvas)
        mode_group.setExclusive(True)
        self._mode_lab.setChecked(True)
        mode_group.triggered.connect(self._switch_mode)

    def _memory_label(self) -> QLabel:
        return QLabel(self)

    # --- mode switching -------------------------------------------------------

    def _switch_mode(self, action: QAction) -> None:
        """Switch the stacked page; the mode switch is the source of truth."""
        if action is self._mode_lab:
            self._stack.setCurrentWidget(self._lab_page)
        else:
            self._stack.setCurrentWidget(self._canvas_page)
            self._canvas_page.refresh_status()

    # --- run progress (FR-UI-05) ----------------------------------------------

    def _on_launch_requested(self, request: object) -> None:
        """Show progress while a batch runs (FR-UI-05)."""
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.show()

    # --- injected panels (stable layout API) -----------------------------------

    def set_run_controls(self, widget: QWidget) -> None:
        """Place the Lab run-controls strip in the top bar (P5-10)."""
        layout = self._run_controls_host.layout()
        assert layout is not None
        layout.addWidget(widget)

    def canvas_page(self) -> CanvasPage:
        """The Canvas page (tests and integrations)."""
        return self._canvas_page

    # --- dialogs and actions -----------------------------------------------------

    def _open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open project"),
            "",
            self.tr("Pic Utils project (*.puproject)"),
        )
        if path:
            self.project_open_requested.emit(Path(path))

    def _open_canvas_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open canvas"),
            "",
            self.tr("Pic Utils canvas (*.pucanvas *.canvas)"),
        )
        if path:
            self.canvas_open_requested.emit(Path(path))

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self._languages, self)
        dialog.exec()

    def _reload_blocks_slot(self) -> None:
        self.blocks_reload_requested.emit()

    def _copy_diagnostics_slot(self) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.diagnostics_text())

    def diagnostics_text(self) -> str:
        """App version, platform, and dependency versions for bug reports."""
        instance = QApplication.instance()
        qt_version = instance.property("qt_version") if instance is not None else "?"
        return "\n".join(
            [
                f"Pic Utils {__version__}",
                f"Python {platform.python_version()} on {platform.platform()}",
                f"Qt {qt_version}",
                f"Log: {logs_dir() / 'app.log'}",
            ]
        )

    # --- status and progress ---------------------------------------------------------

    def show_progress(self, done: int, total: int) -> None:
        """Update the run progress bar; hidden when idle."""
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(done)
        self._progress.show()

    def clear_progress(self) -> None:
        self._progress.hide()

    def set_run_status(self, text: str) -> None:
        self._run_status.setText(text)

    # --- drag and drop -----------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_SUFFIXES:
                self.sources_dropped.emit([path])
            elif path.suffix == PROJECT_SUFFIX:
                self.project_open_requested.emit(path)
            elif path.suffix == CANVAS_SUFFIX:
                self.canvas_open_requested.emit(path)
        event.acceptProposedAction()

    # --- retranslation ------------------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self.setWindowTitle(self.tr("Pic Utils"))
        self._mode_lab.setText(self.tr("Lab"))
        self._mode_canvas.setText(self.tr("Canvas"))
        self._file_menu.setTitle(self.tr("&File"))
        self._tools_menu.setTitle(self.tr("&Tools"))
        self._help_menu.setTitle(self.tr("&Help"))
        self._new_canvas.setText(self.tr("&New canvas"))
        self._open_project.setText(self.tr("&Open project…"))
        self._open_canvas.setText(self.tr("Open &canvas…"))
        self._save_canvas.setText(self.tr("&Save canvas"))
        self._save_canvas_as.setText(self.tr("Save canvas &as…"))
        self._export_canvas.setText(self.tr("&Export canvas…"))
        self._reload_blocks.setText(self.tr("&Reload blocks"))
        self._settings_action.setText(self.tr("Se&ttings…"))
        self._copy_diagnostics.setText(self.tr("&Copy diagnostics"))
        retranslate_tree(self._lab_page)
        retranslate_tree(self._canvas_page)
