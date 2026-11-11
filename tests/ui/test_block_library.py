"""Block library tests: model, proxy filters, reload, auto-reload (P5-02)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.ui.lab.block_library import (
    ROLE_BLOCK_DEF,
    BlockLibraryModel,
    BlockLibraryPanel,
    BlockLibraryProxy,
)

USER_BLOCK_OK = '''
from typing import Annotated

from pic_utils.blocks import IntRange, OutputKind, block
from pic_utils.core.image import Image


@block(name="user_sharpen", category="sharpen", description="Unsharp mask.", output=OutputKind.IMAGE)
def user_sharpen(
    image: Image,
    amount: Annotated[int, IntRange(0, 100, step=1)] = 50,
) -> Image:
    return image
'''

USER_BLOCK_BROKEN = "this is not python ("


@pytest.fixture
def registry(tmp_path: Path, qtbot: object) -> BlockRegistry:
    folder = tmp_path / "blocks"
    folder.mkdir()
    (folder / "good.py").write_text(USER_BLOCK_OK, encoding="utf-8")
    return BlockRegistry(folder)


@pytest.fixture
def panel(qtbot: object, registry: BlockRegistry) -> BlockLibraryPanel:
    widget = BlockLibraryPanel(registry)
    qtbot.addWidget(widget)  # type: ignore[attr-defined]
    return widget


class TestModel:
    def test_lists_all_blocks_sorted(self, registry: BlockRegistry) -> None:
        model = BlockLibraryModel(registry)
        names = [
            model.data(model.index(row, 0), Qt.ItemDataRole.DisplayRole) for row in range(model.rowCount())
        ]
        assert names == sorted(names)
        assert "user_sharpen" in names

    def test_roles_expose_definition_and_category(self, registry: BlockRegistry) -> None:
        model = BlockLibraryModel(registry)
        row = next(
            row
            for row in range(model.rowCount())
            if model.data(model.index(row, 0), Qt.ItemDataRole.DisplayRole) == "user_sharpen"
        )
        index = model.index(row, 0)
        definition = model.data(index, ROLE_BLOCK_DEF)
        assert definition is not None
        assert definition.name == "user_sharpen"
        assert model.data(index, Qt.ItemDataRole.DisplayRole) == "user_sharpen"

    def test_categories_include_user_category(self, registry: BlockRegistry) -> None:
        model = BlockLibraryModel(registry)
        assert "sharpen" in model.categories()


class TestProxy:
    def test_search_filters_by_name_and_description(self, registry: BlockRegistry) -> None:
        model = BlockLibraryModel(registry)
        proxy = BlockLibraryProxy()
        proxy.setSourceModel(model)

        proxy.set_search("sharpen")
        assert proxy.rowCount() == 1

        proxy.set_search("unsharp")
        assert proxy.rowCount() == 1

        proxy.set_search("zzz-no-match")
        assert proxy.rowCount() == 0

    def test_category_filter(self, registry: BlockRegistry) -> None:
        model = BlockLibraryModel(registry)
        proxy = BlockLibraryProxy()
        proxy.setSourceModel(model)
        proxy.set_category_label("sharpen")
        assert proxy.rowCount() == 1

    def test_clearing_filters_restores_rows(self, registry: BlockRegistry) -> None:
        model = BlockLibraryModel(registry)
        proxy = BlockLibraryProxy()
        proxy.setSourceModel(model)
        total = proxy.rowCount()
        proxy.set_search("sharpen")
        proxy.set_search("")
        assert proxy.rowCount() == total


class TestPanel:
    def test_activation_emits_block_name(
        self, panel: BlockLibraryPanel, qtbot: object
    ) -> None:
        seen: list[str] = []
        panel.block_activated.connect(seen.append)
        proxy = panel._proxy
        index = proxy.index(0, 0)
        panel._on_activated(index)
        assert len(seen) == 1
        assert isinstance(seen[0], str)

    def test_manual_reload_reflects_new_file(
        self, panel: BlockLibraryPanel, registry: BlockRegistry, tmp_path: Path
    ) -> None:
        (registry.user_blocks_dir / "second.py").write_text(
            USER_BLOCK_OK.replace("user_sharpen", "user_second"), encoding="utf-8"
        )
        panel.reload_now()
        names = [
            panel._model.data(panel._model.index(row, 0), Qt.ItemDataRole.DisplayRole)
            for row in range(panel._model.rowCount())
        ]
        assert "user_second" in names

    def test_broken_file_reports_errors_deduplicated(
        self, panel: BlockLibraryPanel, registry: BlockRegistry, tmp_path: Path
    ) -> None:
        reports: list[list[object]] = []
        panel.reload_errors.connect(reports.append)
        broken = registry.user_blocks_dir / "broken.py"
        broken.write_text(USER_BLOCK_BROKEN, encoding="utf-8")
        panel.reload_now()
        panel.reload_now()
        # deduplicated: the same error is not reported twice
        assert len(reports) == 1

    def test_successful_reload_clears_errors(
        self, panel: BlockLibraryPanel, registry: BlockRegistry
    ) -> None:
        reports: list[list[object]] = []
        panel.reload_errors.connect(reports.append)
        broken = registry.user_blocks_dir / "broken.py"
        broken.write_text(USER_BLOCK_BROKEN, encoding="utf-8")
        panel.reload_now()
        broken.unlink()
        panel.reload_now()
        assert len(reports) == 1  # still only the original report

    def test_auto_reload_watches_user_dir(self, panel: BlockLibraryPanel, registry: BlockRegistry) -> None:
        assert panel._watcher.directories() == [str(registry.user_blocks_dir)]
        panel.set_auto_reload(False)
        assert panel._watcher.directories() == []
        panel.set_auto_reload(True)
        assert panel._watcher.directories() == [str(registry.user_blocks_dir)]

    def test_registry_change_resets_model(self, panel: BlockLibraryPanel) -> None:
        before = panel._model.rowCount()
        panel._registry.reload()
        assert panel._model.rowCount() == before

    def test_retranslate_sets_strings(self, panel: BlockLibraryPanel) -> None:
        panel.retranslate()
        assert panel._search.placeholderText()
        assert panel._reload.text()
