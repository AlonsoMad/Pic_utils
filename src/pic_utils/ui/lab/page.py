"""Lab mode composition root (architecture §13.1) — final version.

Composes the panels into the §13.1 splitter layout and wires their signals:
library activation adds steps; the editor's selection drives the parameter
panel and the preview; runs feed the results grid. Owns the block registry,
the run service, the session trash, and the project actions; the run
controls' plan count reads the editor's pipeline through the live snapshot.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.results.trash import prepare_trash
from pic_utils.runner.manifest import RunStore
from pic_utils.runner.service import RunnerSettings, RunRequest, RunService, RunSummary
from pic_utils.settings import Settings
from pic_utils.ui.lab.block_library import BlockLibraryPanel
from pic_utils.ui.lab.parameter_panel import ParameterPanel
from pic_utils.ui.lab.pipeline_editor import PipelineEditorPanel, PipelineModel
from pic_utils.ui.lab.preview_pane import PreviewPane
from pic_utils.ui.lab.project_actions import ProjectActions, ProjectData
from pic_utils.ui.lab.results_grid import ResultsGrid
from pic_utils.ui.lab.run_controls import RunControls

__all__ = ["LabPage"]


class LabPage(QWidget):
    """Left library; center editor + preview; right parameters; bottom results."""

    #: The user asked to launch a batch (the shell hosts the progress bar).
    launch_requested = Signal(object)  # RunRequest
    #: A non-modal notification for the status bar (§13.6).
    notify = Signal(str)

    def __init__(self, settings: Settings | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings if settings is not None else Settings()
        self._trash = prepare_trash(self._settings.results_root)
        self._registry = BlockRegistry(self._settings.blocks_folder, parent=self)
        self._project_actions = ProjectActions(self)

        self._library = BlockLibraryPanel(self._registry, self)
        self._editor = PipelineEditorPanel(PipelineModel(self._registry.snapshot, self), self)
        self._preview = PreviewPane(
            lambda: (
                str(folder)
                if (folder := self._registry.user_blocks_dir) is not None and folder.is_dir()
                else None
            ),
            debounce_ms=self._settings.debounce_ms,
            proxy_long_side=self._settings.proxy_long_side,
            parent=self,
        )
        self._parameters = ParameterPanel(self._editor.model(), self._editor.undo_stack(), self)
        self._results = ResultsGrid(
            RunStore(self._settings.results_root, "pending"), self._trash, self
        )
        self._run_controls = RunControls(self._plan_count_provider, self)

        self._build_layout()
        self._service = RunService(self)
        self._wire()
        self.retranslate()

    # --- layout (§13.1 splitters) --------------------------------------------

    def _build_layout(self) -> None:
        center = QSplitter(Qt.Orientation.Vertical)
        center.addWidget(self._editor)
        center.addWidget(self._preview)

        columns = QSplitter(Qt.Orientation.Horizontal)
        columns.addWidget(self._library)
        columns.addWidget(center)
        columns.addWidget(self._parameters)
        columns.setStretchFactor(0, 1)
        columns.setStretchFactor(1, 3)
        columns.setStretchFactor(2, 1)

        rows = QSplitter(Qt.Orientation.Vertical)
        rows.addWidget(columns)
        rows.addWidget(self._results)
        rows.setStretchFactor(0, 3)
        rows.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(rows)

    # --- panel accessors (stable composition API) ------------------------------

    def run_controls(self) -> RunControls:
        """The run-controls strip; the shell hosts it in the top bar."""
        return self._run_controls

    def editor(self) -> PipelineEditorPanel:
        return self._editor

    def library(self) -> BlockLibraryPanel:
        return self._library

    def parameters(self) -> ParameterPanel:
        return self._parameters

    def results_grid(self) -> ResultsGrid:
        return self._results

    def preview_pane(self) -> PreviewPane:
        return self._preview

    # --- wiring ---------------------------------------------------------------

    def _wire(self) -> None:
        self._library.block_activated.connect(self._editor.add_block)
        self._library.reload_errors.connect(self._on_reload_errors)
        self._editor.step_selected.connect(self._parameters.show_step)
        self._editor.model().pipeline_changed.connect(lambda _p: self._on_pipeline_changed())

        self._parameters.param_edited.connect(lambda *_: self._on_pipeline_changed())
        self._parameters.mask_binding_changed.connect(lambda *_: self._on_pipeline_changed())

        self._run_controls.launch_requested.connect(self._launch)
        self._run_controls.cancel_requested.connect(self._service.cancel)
        self._run_controls.sources_changed.connect(self._on_pipeline_changed)

        self._service.run_started.connect(self._on_run_started)
        self._service.progress.connect(self._on_progress)
        self._service.result_ready.connect(self._on_result_ready)
        self._service.run_finished.connect(self._on_run_finished)
        self._service.run_failed.connect(
            lambda code, message: self.notify.emit(f"{code}: {message}")
        )
        self._service.memory_updated.connect(self._run_controls.set_memory_mb)

        self._registry.changed.connect(lambda _snapshot: self._preview.restart())
        self._settings.changed.connect(self._on_setting_changed)

    # --- slots ----------------------------------------------------------------

    def _on_pipeline_changed(self) -> None:
        """Pipeline, parameters, or sources changed: preview + plan refresh."""
        self._request_preview()
        self._run_controls.refresh_plan_count()

    def _request_preview(self) -> None:
        sources = self._run_controls.sources()
        second = self._run_controls.second_image()
        self._preview.request(
            self._editor.model().pipeline.to_dict(),
            str(sources[0]) if sources else None,
            str(second) if second is not None else None,
        )

    def _plan_count_provider(self, config: object) -> int | None:
        from pic_utils.pipeline.plan import build_plan
        from pic_utils.pipeline.validate import is_blocking, validate_run_config

        controls = getattr(self, "_run_controls", None)
        if controls is None:
            return None  # still constructing the run controls
        pipeline = self._editor.model().pipeline
        snapshot = self._registry.snapshot()
        sources = controls.sources()
        issues = validate_run_config(
            config,  # type: ignore[arg-type]
            pipeline,
            snapshot,
            source_count=len(sources),
        )
        if is_blocking(issues) or not sources:
            return None
        return len(build_plan(config, pipeline, snapshot, len(sources)))  # type: ignore[arg-type]

    def _launch(self, config: object) -> None:
        from pic_utils.pipeline.validate import is_blocking, validate_run_config

        sources = self._run_controls.sources()
        issues = validate_run_config(
            config,  # type: ignore[arg-type]
            self._editor.model().pipeline,
            self._registry.snapshot(),
            source_count=len(sources),
        )
        if is_blocking(issues):
            self.notify.emit(issues[0].message)
            return
        request = RunRequest(
            pipeline=self._editor.model().pipeline,
            snapshot=self._registry.snapshot(),
            sources=tuple(sources),
            second=self._run_controls.second_image(),
            run_config=config,  # type: ignore[arg-type]
            results_root=self._settings.results_root,
            settings=RunnerSettings(
                workers=self._settings.workers,
                memory_ceiling_mb=self._settings.memory_ceiling_mb,
                experiment_timeout_s=float(self._settings.experiment_timeout_s),
            ),
            user_blocks_dir=self._registry.user_blocks_dir,
        )
        self._service.start(request)
        self.launch_requested.emit(request)

    def _on_run_started(self, run_id: str, _total: int) -> None:
        self._run_controls.set_running(True)
        self._results.bind_run(RunStore(self._settings.results_root, run_id))

    def _on_progress(self, done: int, total: int) -> None:
        self._run_controls.show_progress(done, total)

    def _on_result_ready(self, record: object) -> None:
        self._results.model().append_result(record)  # type: ignore[arg-type]

    def _on_run_finished(self, summary: object) -> None:
        assert isinstance(summary, RunSummary)
        self._run_controls.show_summary(summary)
        self._results.refresh_runs()
        self._results.model().reload()
        self.notify.emit(
            self.tr("Run finished: %1 completed, %2 failed")
            .replace("%1", str(summary.completed))
            .replace("%2", str(summary.failed))
        )

    def _on_reload_errors(self, errors: list) -> None:
        if errors:
            self.notify.emit(self.tr("Block reload failed: %1").replace("%1", errors[0].reason))

    def _on_setting_changed(self, key: str) -> None:
        if key == "runner/auto_reload_blocks":
            self._library.set_auto_reload(self._settings.auto_reload_blocks)

    # --- project glue (§17) ---------------------------------------------------

    def project_data(self) -> ProjectData:
        """The current project (§17.1) from the live panels."""
        return ProjectData(
            sources=tuple(self._run_controls.all_sources()),
            second_image=self._run_controls.second_image(),
            pipeline=self._editor.model().pipeline,
            run=self._run_controls.run_config(),
        )

    def open_project(self, path: Path) -> None:
        """Load a project file; missing files open the relink dialog."""
        from pic_utils.ui.lab.project_actions import ProjectFileError

        try:
            data = self._project_actions.load(path, self)
        except ProjectFileError as error:
            self.notify.emit(str(error))
            return
        if data is not None:
            self.load_project_data(data)

    def save_project(self, parent: QWidget | None = None) -> None:
        """Save to the current path, or ask for one (FR-PERS-01)."""
        self._project_actions.save(self.project_data(), parent or self)

    def save_project_as(self, parent: QWidget | None = None) -> None:
        self._project_actions.save_as(self.project_data(), parent or self)

    def load_project_data(self, data: ProjectData) -> None:
        """Apply a loaded project (§17.1); the undo stack clears."""
        self._editor.load_pipeline(data.pipeline)
        self._run_controls.clear_sources()
        self._run_controls.add_sources(list(data.sources))
        self._run_controls.set_second_image(data.second_image)
        vary: dict[tuple[str, str], tuple[str, Any]] = {}
        for axis in data.run.axes:
            vary[(axis.step_id, axis.param)] = ("sweep", tuple(axis.values))
        for step_id, param in data.run.sampled:
            vary[(step_id, param)] = ("sample", None)
        self._parameters.set_vary_state(vary)
        self._run_controls.apply_run_config(data.run)

    # --- retranslation -----------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        from pic_utils.ui.i18n import retranslate_tree

        retranslate_tree(self)
