"""Run controls: sources, run configuration, launch/cancel (§13.1, FR-UI-05).

The panel owns the source list (add/remove, per-source checkboxes), the
second-image selector, the run mode, batch size, and master seed. The plan
count updates on every relevant change via the injected
``plan_count_provider`` (None = invalid configuration). Launch emits the
built :class:`RunConfig`; while a run is active the panel shows progress,
memory, and a summary (FR-RUN-09).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pic_utils.pipeline.model import RunConfig

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["RunControls", "SourceEntry"]

_MODES = ("grid", "random", "hybrid")


class SourceEntry:
    """One source image with its per-source enable checkbox (§13.1)."""

    __slots__ = ("enabled", "path")

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled


class RunControls(QWidget):
    """Source list, run configuration, and launch/cancel controls."""

    #: The user pressed Launch with a valid configuration.
    launch_requested = Signal(object)  # RunConfig
    #: The user pressed Cancel.
    cancel_requested = Signal()
    #: The source list or second image changed (preview + plan refresh).
    sources_changed = Signal()

    def __init__(
        self,
        plan_count_provider: Callable[[RunConfig], int | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plan_count_provider = plan_count_provider
        self._sources: list[SourceEntry] = []
        self._second: Path | None = None
        self._running = False

        # Source list -------------------------------------------------------
        self._source_list = QListWidget(self)
        self._add_source = QPushButton(self)
        self._add_source.clicked.connect(self._add_source_dialog)
        self._remove_source = QPushButton(self)
        self._remove_source.clicked.connect(self._remove_selected_source)
        self._source_list.itemChanged.connect(self._on_item_changed)

        # Second image ------------------------------------------------------
        self._second_label = QLabel(self)
        self._second_button = QPushButton(self)
        self._second_button.clicked.connect(self._pick_second)
        self._second_clear = QPushButton(self)
        self._second_clear.clicked.connect(self._clear_second)

        # Run configuration ---------------------------------------------------
        self._mode = QComboBox(self)
        self._mode.addItems(list(_MODES))
        self._batch_size = QSpinBox(self)
        self._batch_size.setRange(1, 10_000)
        self._batch_size.setValue(32)
        self._random_samples = QSpinBox(self)
        self._random_samples.setRange(0, 100)
        self._master_seed = QSpinBox(self)
        self._master_seed.setRange(0, 0x7FFFFFFF)

        self._batch_size.valueChanged.connect(self._refresh_plan_count)
        self._random_samples.valueChanged.connect(self._refresh_plan_count)
        self._master_seed.valueChanged.connect(self._refresh_plan_count)
        self._mode.currentIndexChanged.connect(self._refresh_plan_count)

        # Launch / status -----------------------------------------------------
        self._plan_count = QLabel(self)
        self._launch = QPushButton(self)
        self._launch.clicked.connect(self._on_launch)
        self._cancel = QPushButton(self)
        self._cancel.clicked.connect(self.cancel_requested.emit)
        self._cancel.setEnabled(False)
        self._summary = QLabel(self)
        self._memory = QLabel(self)

        layout = QVBoxLayout(self)
        layout.addWidget(self._source_list)
        source_buttons = QHBoxLayout()
        source_buttons.addWidget(self._add_source)
        source_buttons.addWidget(self._remove_source)
        layout.addLayout(source_buttons)

        second_row = QHBoxLayout()
        second_row.addWidget(self._second_label, 1)
        second_row.addWidget(self._second_button)
        second_row.addWidget(self._second_clear)
        layout.addLayout(second_row)

        config_row = QHBoxLayout()
        config_row.addWidget(self._mode)
        config_row.addWidget(self._batch_size)
        config_row.addWidget(self._random_samples)
        config_row.addWidget(self._master_seed)
        layout.addLayout(config_row)

        launch_row = QHBoxLayout()
        launch_row.addWidget(self._plan_count, 1)
        launch_row.addWidget(self._launch)
        launch_row.addWidget(self._cancel)
        layout.addLayout(launch_row)

        status_row = QHBoxLayout()
        status_row.addWidget(self._summary, 1)
        status_row.addWidget(self._memory)
        layout.addLayout(status_row)

        self.retranslate()
        self._refresh_plan_count()

    # --- public API ---------------------------------------------------------

    def sources(self) -> list[Path]:
        """The enabled source paths, in list order."""
        return [entry.path for entry in self._sources if entry.enabled]

    def all_sources(self) -> list[Path]:
        return [entry.path for entry in self._sources]

    def second_image(self) -> Path | None:
        return self._second

    def run_config(self) -> RunConfig:
        """The current run configuration (§7.1); axes come from the panel."""
        return RunConfig(
            mode=self._mode.currentText(),
            batch_size=self._batch_size.value(),
            random_samples=self._random_samples.value(),
            master_seed=self._master_seed.value(),
        )

    def set_running(self, running: bool) -> None:
        """Swap Launch/Cancel enablement while a run is active."""
        self._running = running
        self._launch.setEnabled(not running)
        self._cancel.setEnabled(running)

    def is_running(self) -> bool:
        return self._running

    def show_progress(self, done: int, total: int) -> None:
        self._summary.setText(f"{done} / {total}")

    def show_summary(self, summary: object) -> None:
        """Render the end-of-run summary (FR-RUN-09)."""
        status = getattr(summary, "status", "?")
        completed = getattr(summary, "completed", 0)
        failed = getattr(summary, "failed", 0)
        cancelled = getattr(summary, "cancelled", 0)
        elapsed = getattr(summary, "elapsed_ms", 0)
        self._summary.setText(
            self.tr("%n completed, %1 failed, %2 cancelled (%3 s)")
            .replace("%n", str(completed))
            .replace("%1", str(failed))
            .replace("%2", str(cancelled))
            .replace("%3", f"{elapsed / 1000.0:.1f}")
        )
        self.set_running(False)
        del status

    def set_memory_mb(self, total_mb: int) -> None:
        self._memory.setText(f"{total_mb} MiB")

    def add_sources(self, paths: list) -> None:
        """Add sources (drag-and-drop or dialog); skips duplicates."""
        changed = False
        for path in paths:
            path = Path(path)
            if any(entry.path == path for entry in self._sources):
                continue
            self._sources.append(SourceEntry(path))
            item = QListWidgetItem(str(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._source_list.addItem(item)
            changed = True
        if changed:
            self._refresh_plan_count()
            self.sources_changed.emit()

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._add_source.setText(self.tr("Add images…"))
        self._remove_source.setText(self.tr("Remove"))
        self._second_button.setText(self.tr("Second image…"))
        self._second_clear.setText(self.tr("Clear"))
        self._launch.setText(self.tr("Launch"))
        self._cancel.setText(self.tr("Cancel"))
        self._update_second_label()
        self._refresh_plan_count()

    # --- internals ------------------------------------------------------------

    def _update_second_label(self) -> None:
        self._second_label.setText(
            str(self._second) if self._second is not None else self.tr("no second image")
        )

    def _add_source_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Add images"),
            "",
            self.tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"),
        )
        if paths:
            self.add_sources([Path(p) for p in paths])

    def _remove_selected_source(self) -> None:
        row = self._source_list.currentRow()
        if row < 0:
            return
        del self._sources[row]
        self._source_list.takeItem(row)
        self._refresh_plan_count()
        self.sources_changed.emit()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        row = self._source_list.row(item)
        if 0 <= row < len(self._sources):
            self._sources[row].enabled = item.checkState() == Qt.CheckState.Checked
            self._refresh_plan_count()
            self.sources_changed.emit()

    def _pick_second(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Second image"),
            "",
            self.tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"),
        )
        if path:
            self._second = Path(path)
            self._update_second_label()
            self._refresh_plan_count()
            self.sources_changed.emit()

    def _clear_second(self) -> None:
        self._second = None
        self._update_second_label()
        self._refresh_plan_count()
        self.sources_changed.emit()

    def _on_launch(self) -> None:
        if self._plan_count.text().startswith("—"):
            return
        self.launch_requested.emit(self.run_config())

    def _refresh_plan_count(self) -> None:
        count = self._plan_count_provider(self.run_config())
        if count is None:
            self._plan_count.setText(self.tr("— invalid configuration —"))
            self._launch.setEnabled(False)
        else:
            self._plan_count.setText(self.tr("%n experiment(s)", "", count))
            self._launch.setEnabled(not self._running)
