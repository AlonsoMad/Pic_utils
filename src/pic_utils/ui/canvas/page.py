"""Canvas mode composition root (architecture §13.1) — final version.

Left asset tray; center canvas view; right layer list + layer properties.
The page owns the document, the mode's undo stack, and the save/export
actions; every mutation goes through a ``QUndoCommand`` and the scene is
re-synced from the result.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QUndoStack
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.model import CanvasDoc, CanvasLayer
from pic_utils.core.errors import CanvasFileError, PicUtilsError
from pic_utils.io.loader import load_image
from pic_utils.results.store import AcceptedIndex, list_runs, load_run
from pic_utils.settings import Settings
from pic_utils.ui.canvas.actions import CanvasActions
from pic_utils.ui.canvas.asset_tray import AssetTray
from pic_utils.ui.canvas.canvas_view import CanvasView
from pic_utils.ui.canvas.commands import (
    AddLayerCommand,
    ApplyMaskCommand,
    CropCanvasCommand,
    GroupCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
    ResizeCanvasCommand,
    SetGroupPropertyCommand,
    SetLayerGroupCommand,
    SetLayerPropertyCommand,
    TransformLayerCommand,
    UngroupCommand,
)
from pic_utils.ui.canvas.export_dialog import ExportDialog
from pic_utils.ui.canvas.layer_list import LayerListPanel
from pic_utils.ui.canvas.layer_properties import LayerPropertiesPanel
from pic_utils.ui.dialogs.mask_dialog import MaskDialog
from pic_utils.ui.i18n import retranslate_tree

__all__ = ["CanvasPage"]


class CanvasPage(QWidget):
    """Canvas layout with asset tray, viewport, layer list, and properties."""

    #: A non-modal notification for the status bar (§13.6).
    notify = Signal(str)
    #: Canvas dimensions or export progress for the status bar.
    canvas_status = Signal(str)

    def __init__(
        self,
        settings: Settings | None = None,
        registry: BlockRegistry | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings if settings is not None else Settings()
        self._registry = (
            registry
            if registry is not None
            else BlockRegistry(self._settings.blocks_folder, parent=self)
        )
        self._doc = CanvasDoc()
        self._undo = QUndoStack(self)
        self._undo.setUndoLimit(self._settings.undo_limit)
        self._undo_action = QAction(self)
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_action.triggered.connect(self._undo.undo)
        self._undo.canUndoChanged.connect(self._undo_action.setEnabled)
        self._undo_action.setEnabled(False)
        self.addAction(self._undo_action)
        self._redo_action = QAction(self)
        self._redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_action.triggered.connect(self._undo.redo)
        self._undo.canRedoChanged.connect(self._redo_action.setEnabled)
        self._redo_action.setEnabled(False)
        self.addAction(self._redo_action)
        self._current_layer_id: str | None = None
        self._syncing_selection = False

        self._view = CanvasView(self)
        self._asset_tray = AssetTray(AcceptedIndex(self._settings.results_root), self)
        self._layer_list = LayerListPanel(self)
        self._properties = LayerPropertiesPanel(self)
        self._actions = CanvasActions(self._settings, self)

        self._build_layout()
        self._wire()
        self._view.set_doc(self._doc)
        self._layer_list.set_doc(self._doc)
        self._properties.set_layer(None, {})
        self._update_status()
        self.retranslate()

    # --- layout --------------------------------------------------------------

    def _build_layout(self) -> None:
        self._crop_button = QPushButton(self)
        self._crop_button.clicked.connect(self._view.begin_crop)
        self._resize_button = QPushButton(self)
        self._resize_button.clicked.connect(self._resize_canvas_dialog)
        self._fit_button = QPushButton(self)
        self._fit_button.clicked.connect(self._view.fit_canvas)
        self._zoom_in_button = QPushButton(self)
        self._zoom_in_button.clicked.connect(self._view.zoom_in)
        self._zoom_out_button = QPushButton(self)
        self._zoom_out_button.clicked.connect(self._view.zoom_out)
        self._save_button = QPushButton(self)
        self._save_button.clicked.connect(self.save_canvas)
        self._export_button = QPushButton(self)
        self._export_button.clicked.connect(self.export_canvas)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        for button in (
            self._crop_button,
            self._resize_button,
            self._fit_button,
            self._zoom_out_button,
            self._zoom_in_button,
            self._save_button,
            self._export_button,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch(1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._layer_list)
        right.addWidget(self._properties)

        columns = QSplitter(Qt.Orientation.Horizontal)
        columns.addWidget(self._asset_tray)
        columns.addWidget(self._view)
        columns.addWidget(right)
        columns.setStretchFactor(0, 1)
        columns.setStretchFactor(1, 4)
        columns.setStretchFactor(2, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(columns)

    # --- accessors -----------------------------------------------------------

    def doc(self) -> CanvasDoc:
        return self._doc

    def undo_stack(self) -> QUndoStack:
        return self._undo

    def canvas_actions(self) -> CanvasActions:
        return self._actions

    def canvas_view(self) -> CanvasView:
        return self._view

    def asset_tray(self) -> AssetTray:
        return self._asset_tray

    def layer_list(self) -> LayerListPanel:
        return self._layer_list

    def layer_properties(self) -> LayerPropertiesPanel:
        return self._properties

    # --- wiring --------------------------------------------------------------

    def _wire(self) -> None:
        self._asset_tray.add_requested.connect(self._add_layer_from_path)
        self._layer_list.layer_selected.connect(self._on_layer_selected)
        self._layer_list.visibility_toggled.connect(self._on_visibility_toggled)
        self._layer_list.move_requested.connect(self._on_move_requested)
        self._layer_list.group_requested.connect(self._on_group_requested)
        self._layer_list.ungroup_requested.connect(self._on_ungroup_requested)
        self._layer_list.delete_requested.connect(self._delete_selected)

        self._properties.property_changed.connect(self._on_property_changed)
        self._properties.group_property_changed.connect(self._on_group_property_changed)
        self._properties.group_membership_changed.connect(self._on_group_membership_changed)
        self._properties.mask_requested.connect(self._on_mask_requested)
        self._properties.mask_removed.connect(self._on_mask_removed)

        self._view.crop_requested.connect(self._on_crop_requested)
        self._view.delete_requested.connect(self._delete_selected)
        self._view.nudge_requested.connect(self._on_nudge_requested)
        self._view.selection_changed.connect(self._on_scene_selection)
        self._view.canvas_scene().move_finished.connect(self._on_move_finished)
        self._view.canvas_scene().transform_finished.connect(self._on_transform_finished)

        self._undo.indexChanged.connect(
            self._on_undo_index_changed, Qt.ConnectionType.DirectConnection
        )
        self._actions.notify.connect(self.notify.emit)
        self._actions.exported.connect(self._on_exported)
        self._actions.export_failed.connect(self.notify.emit)
        self._actions.export_progress.connect(self._on_export_progress)

    # --- document lifecycle --------------------------------------------------

    def maybe_save(self, parent: QWidget | None = None) -> bool:
        """Ask to save unsaved changes; returns False when the user cancels."""
        if not self._doc.dirty:
            return True
        answer = QMessageBox.question(
            parent or self,
            self.tr("Unsaved canvas"),
            self.tr("Save changes to the current canvas?"),
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self._save_now(parent) is not None
        return True

    def new_document(self) -> None:
        """Start an empty canvas (one canvas open at a time, FR-CAN-09)."""
        if not self.maybe_save(self):
            return
        self._doc.discard_temporary_assets()
        self._doc = CanvasDoc()
        self._undo.clear()
        self._current_layer_id = None
        self._view.set_doc(self._doc)
        self._layer_list.set_doc(self._doc)
        self._properties.set_layer(None, {})
        self._update_status()

    def open_canvas(self, path: str | Path) -> None:
        """Open a canvas file; missing assets are reported without opening."""
        if not self.maybe_save(self):
            return
        doc = self._actions.open(path, self)
        if doc is not None:
            self._set_doc(doc)

    def _set_doc(self, doc: CanvasDoc) -> None:
        self._doc.discard_temporary_assets()
        self._doc = doc
        self._undo.clear()
        self._current_layer_id = None
        self._view.set_doc(doc)
        self._layer_list.set_doc(doc)
        self._properties.set_layer(None, {})
        self._asset_tray.refresh()
        self._update_status()

    def save_canvas(self, parent: QWidget | None = None) -> None:
        """Save to the current path, or ask for one, on a worker thread."""
        self._actions.save_async(self._doc, parent or self, keep=self._undo_asset_refs())

    def save_canvas_as(self, parent: QWidget | None = None) -> None:
        """Save under a new path on a worker thread."""
        self._actions.save_async(
            self._doc, parent or self, save_as=True, keep=self._undo_asset_refs()
        )

    def _save_now(self, parent: QWidget | None = None) -> Path | None:
        """Synchronous save used by the unsaved-changes prompt."""
        return self._actions.save(self._doc, parent or self, keep=self._undo_asset_refs())

    def _undo_asset_refs(self) -> set[str]:
        """Assets any command on the undo stack can restore to the document."""
        references: set[str] = set()
        for index in range(self._undo.count()):
            command = self._undo.command(index)
            getter = getattr(command, "referenced_assets", None)
            if callable(getter):
                references.update(getter())
        return references

    def shutdown(self) -> None:
        """Stop worker tasks and discard temporary assets."""
        self._actions.shutdown()
        self._doc.discard_temporary_assets()

    # --- layer creation ------------------------------------------------------

    def _add_layer_from_path(self, path: object) -> None:
        source = Path(str(path))
        try:
            image = load_image(source)
        except PicUtilsError as error:
            self.notify.emit(str(error))
            return
        store = AssetStore(self._doc.ensure_assets_dir())
        try:
            reference = store.put_image(image)
        except OSError as error:
            self.notify.emit(str(error))
            return
        layer = CanvasLayer(
            id=uuid.uuid4().hex,
            name=source.stem,
            image_ref=reference,
        )
        self._undo.push(AddLayerCommand(self._doc, layer, natural_size=image.size))
        self._select_layer(layer.id)

    # --- selection -----------------------------------------------------------

    def _select_layer(self, layer_id: str | None) -> None:
        self._current_layer_id = layer_id
        self._syncing_selection = True
        try:
            self._view.select_layer(layer_id)
            self._layer_list.select_layer(layer_id)
            self._properties.set_layer(
                self._doc.find_layer(layer_id) if layer_id else None,
                self._doc.groups,
            )
        finally:
            self._syncing_selection = False

    def _on_layer_selected(self, layer_id: object) -> None:
        if self._syncing_selection:
            return
        self._select_layer(str(layer_id) if layer_id else None)

    def _on_scene_selection(self, layer_ids: list) -> None:
        if self._syncing_selection:
            return
        self._select_layer(str(layer_ids[0]) if layer_ids else None)

    # --- commands ------------------------------------------------------------

    def _on_visibility_toggled(self, layer_id: str, visible: bool) -> None:
        self._undo.push(SetLayerPropertyCommand(self._doc, layer_id, "visible", visible))

    def _on_move_requested(self, from_index: int, to_index: int) -> None:
        if 0 <= from_index < len(self._doc.layers):
            layer = self._doc.layers[from_index]
            if layer.locked:
                self.notify.emit(self.tr("Unlock the layer before reordering it"))
                return
        self._undo.push(MoveLayerCommand(self._doc, from_index, to_index))

    def _on_group_requested(self, layer_ids: list) -> None:
        selected = [str(layer_id) for layer_id in layer_ids]
        if any(
            (layer := self._doc.find_layer(layer_id)) is not None and layer.locked
            for layer_id in selected
        ):
            self.notify.emit(self.tr("Unlock the selected layers before grouping"))
            return
        self._undo.push(GroupCommand(self._doc, selected))

    def _on_ungroup_requested(self, group_id: str) -> None:
        if any(layer.locked for layer in self._doc.layers if layer.group_id == group_id):
            self.notify.emit(self.tr("Unlock the grouped layers before ungrouping"))
            return
        self._undo.push(UngroupCommand(self._doc, group_id))

    def _delete_selected(self) -> None:
        selected = self._view.selected_layer_ids() or self._layer_list.selected_layer_ids()
        for layer_id in selected:
            layer = self._doc.find_layer(layer_id)
            if layer is None or layer.locked:
                continue
            self._undo.push(RemoveLayerCommand(self._doc, layer_id))
        self._select_layer(None)

    def _on_property_changed(self, layer_id: str, prop: str, value: object) -> None:
        layer = self._doc.find_layer(layer_id)
        if layer is not None and layer.locked and prop != "locked":
            self.notify.emit(self.tr("Unlock the layer before editing it"))
            return
        self._undo.push(SetLayerPropertyCommand(self._doc, layer_id, prop, value))

    def _on_group_property_changed(self, group_id: str, prop: str, value: object) -> None:
        if any(layer.locked for layer in self._doc.layers if layer.group_id == group_id):
            self.notify.emit(self.tr("Unlock the group before editing it"))
            return
        self._undo.push(SetGroupPropertyCommand(self._doc, group_id, prop, value))

    def _on_group_membership_changed(self, layer_id: str, group_id: object) -> None:
        layer = self._doc.find_layer(layer_id)
        if layer is not None and layer.locked:
            self.notify.emit(self.tr("Unlock the layer before changing its group"))
            return
        self._undo.push(
            SetLayerGroupCommand(
                self._doc,
                layer_id,
                str(group_id) if group_id is not None else None,
            )
        )

    def _on_move_finished(self, layer_id: str, x: float, y: float) -> None:
        layer = self._doc.find_layer(layer_id)
        if layer is None or layer.locked:
            return
        self._undo.push(
            TransformLayerCommand(self._doc, layer_id, x, y, layer.scale, layer.rotation)
        )

    def _on_transform_finished(self, layer_id: str, scale: float, rotation: float) -> None:
        layer = self._doc.find_layer(layer_id)
        if layer is None or layer.locked:
            return
        self._undo.push(
            TransformLayerCommand(self._doc, layer_id, layer.x, layer.y, scale, rotation)
        )

    def _on_nudge_requested(self, dx: float, dy: float) -> None:
        for layer_id in self._view.selected_layer_ids():
            layer = self._doc.find_layer(layer_id)
            if layer is None or layer.locked:
                continue
            self._undo.push(
                TransformLayerCommand(
                    self._doc,
                    layer_id,
                    layer.x + dx,
                    layer.y + dy,
                    layer.scale,
                    layer.rotation,
                )
            )

    def _on_crop_requested(self, rect) -> None:
        self._undo.push(
            CropCanvasCommand(self._doc, rect.x(), rect.y(), rect.width(), rect.height())
        )

    def _resize_canvas_dialog(self) -> None:
        width, accepted = QInputDialog.getInt(
            self,
            self.tr("Resize canvas"),
            self.tr("Width"),
            self._doc.width,
            1,
            100_000,
        )
        if not accepted:
            return
        height, accepted = QInputDialog.getInt(
            self,
            self.tr("Resize canvas"),
            self.tr("Height"),
            self._doc.height,
            1,
            100_000,
        )
        if not accepted:
            return
        self._undo.push(ResizeCanvasCommand(self._doc, width, height))

    # --- masks ---------------------------------------------------------------

    def _on_mask_requested(self, layer_id: str) -> None:
        layer = self._doc.find_layer(layer_id)
        if layer is None:
            return
        if layer.locked:
            self.notify.emit(self.tr("Unlock the layer before editing its mask"))
            return
        store = AssetStore(self._doc.ensure_assets_dir())
        try:
            image = load_image(store.resolve(layer.image_ref))
        except PicUtilsError as error:
            self.notify.emit(str(error))
            return
        dialog = MaskDialog(image, self._registry.snapshot(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        mask = dialog.result_mask()
        if mask is None:
            return
        try:
            reference = store.put_mask(mask)
        except OSError as error:
            self.notify.emit(str(error))
            return
        self._undo.push(ApplyMaskCommand(self._doc, layer_id, reference))

    def _on_mask_removed(self, layer_id: str) -> None:
        layer = self._doc.find_layer(layer_id)
        if layer is None:
            return
        if layer.locked:
            self.notify.emit(self.tr("Unlock the layer before editing its mask"))
            return
        self._undo.push(ApplyMaskCommand(self._doc, layer_id, None))

    # --- export --------------------------------------------------------------

    def export_canvas(self, parent: QWidget | None = None) -> None:
        """Ask for export options and start a threaded export (§11)."""
        dialog = ExportDialog(self._settings, parent or self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        self.canvas_status.emit(self.tr("Exporting…"))
        self._actions.export_async(
            self._doc,
            options,
            accepted=self._accepted_paths(),
            run_results=self._run_result_paths(),
        )

    def _accepted_paths(self) -> tuple[Path, ...]:
        index = AcceptedIndex(self._settings.results_root)
        index.refresh()
        return tuple(
            index.results_root / accepted.run_id / accepted.record.path
            for accepted in index.results()
        )

    def _run_result_paths(self) -> tuple[Path, ...]:
        runs = list_runs(self._settings.results_root)
        if not runs:
            return ()
        latest = runs[0]
        try:
            data = load_run(self._settings.results_root, latest.run_id)
        except OSError:
            return ()
        return tuple(
            latest.path / record.path
            for record in data.results
            if data.curation.get(record.id) != "discarded"
        )

    def _on_exported(self, report: object) -> None:
        folder = getattr(report, "folder", "")
        self.notify.emit(self.tr("Exported to %1").replace("%1", str(folder)))

    def _on_export_progress(self, done: int, total: int) -> None:
        self.canvas_status.emit(
            self.tr("Exporting %1/%2").replace("%1", str(done)).replace("%2", str(total))
        )

    # --- sync ----------------------------------------------------------------

    def _on_undo_index_changed(self, _index: int) -> None:
        self._sync()

    def _sync(self) -> None:
        try:
            self._view.sync()
        except CanvasFileError as error:
            self.notify.emit(str(error))
            return
        current = self._current_layer_id
        self._syncing_selection = True
        try:
            self._layer_list.reload()
        finally:
            self._syncing_selection = False
        layer = self._doc.find_layer(current) if current is not None else None
        self._syncing_selection = True
        try:
            if layer is None:
                self._current_layer_id = None
                self._layer_list.select_layer(None)
                self._properties.set_layer(None, {})
            else:
                self._layer_list.select_layer(layer.id)
                self._properties.set_layer(layer, self._doc.groups)
        finally:
            self._syncing_selection = False
        self._update_status()

    def refresh_status(self) -> None:
        """Re-emit the canvas dimensions (mode switch, shell status bar)."""
        self._update_status()

    def _update_status(self) -> None:
        self.canvas_status.emit(f"{self._doc.width} × {self._doc.height}")  # noqa: RUF001

    # --- events and i18n -----------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._asset_tray.refresh(rescan=False)

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._crop_button.setText(self.tr("Crop"))
        self._resize_button.setText(self.tr("Resize…"))
        self._fit_button.setText(self.tr("Fit"))
        self._zoom_in_button.setText(self.tr("Zoom in"))
        self._zoom_out_button.setText(self.tr("Zoom out"))
        self._save_button.setText(self.tr("Save"))
        self._export_button.setText(self.tr("Export…"))
        self._undo_action.setText(self.tr("Undo"))
        self._redo_action.setText(self.tr("Redo"))
        retranslate_tree(self)
