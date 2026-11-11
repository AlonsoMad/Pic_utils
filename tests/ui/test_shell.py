"""Shell tests: settings, language manager, main window (P5-01)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QMimeData, QPointF, QUrl
from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QStackedWidget

from pic_utils import __version__
from pic_utils.settings import Settings
from pic_utils.ui.i18n import LanguageManager
from pic_utils.ui.main_window import MainWindow


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(path=tmp_path / "settings.ini")


@pytest.fixture
def languages(settings: Settings) -> LanguageManager:
    return LanguageManager(settings)


@pytest.fixture
def window(qtbot: object, settings: Settings, languages: LanguageManager) -> MainWindow:
    main = MainWindow(settings=settings, languages=languages)
    qtbot.addWidget(main)  # type: ignore[attr-defined]
    return main


class TestDefaults:
    def test_defaults_match_architecture(self, settings: Settings) -> None:
        assert settings.language == "system"
        assert settings.theme == "system"
        assert settings.workers == 2
        assert settings.memory_ceiling_mb == 8192
        assert settings.experiment_timeout_s == 300
        assert settings.auto_reload_blocks is True
        assert settings.proxy_long_side == 1024
        assert settings.debounce_ms == 150
        assert settings.undo_limit == 100
        assert settings.jpeg_quality == 92
        assert settings.tiff_compression == "tiff_lzw"

    def test_results_root_default(self, tmp_path: Path, settings: Settings) -> None:
        assert settings.results_root.name == "PicUtils"

    def test_trash_root_derived(self, tmp_path: Path, settings: Settings) -> None:
        assert settings.trash_root == settings.results_root / "trash"


class TestClamping:
    @pytest.mark.parametrize(
        ("setter", "low", "high", "below", "above"),
        [
            ("set_workers", 1, 8, 0, 99),
            ("set_memory_ceiling_mb", 512, 8192, 1, 999_999),
            ("set_experiment_timeout_s", 1, 86_400, 0, 999_999),
            ("set_proxy_long_side", 128, 4096, 1, 999_999),
            ("set_debounce_ms", 0, 5_000, -5, 9_999),
            ("set_undo_limit", 1, 10_000, 0, 99_999),
            ("set_jpeg_quality", 1, 100, 0, 101),
        ],
    )
    def test_numeric_ranges_clamp(
        self, settings: Settings, setter: str, low: int, high: int, below: int, above: int
    ) -> None:
        getattr(settings, setter)(below)
        assert getattr(settings, setter.removeprefix("set_")) == low
        getattr(settings, setter)(above)
        assert getattr(settings, setter.removeprefix("set_")) == high

    def test_language_rejects_unknown(self, settings: Settings) -> None:
        with pytest.raises(ValueError):
            settings.set_language("fr")

    def test_theme_rejects_unknown(self, settings: Settings) -> None:
        with pytest.raises(ValueError):
            settings.set_theme("sepia")


class TestPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.ini"
        first = Settings(path=path)
        first.set_language("es")
        first.set_theme("dark")
        first.set_workers(4)
        first.set_memory_ceiling_mb(4096)
        first.set_experiment_timeout_s(600)
        first.set_auto_reload_blocks(False)
        first.set_proxy_long_side(2048)
        first.set_debounce_ms(250)
        first.set_undo_limit(50)
        first.set_jpeg_quality(80)
        first.set_tiff_compression("tiff_deflate")

        second = Settings(path=path)
        assert second.language == "es"
        assert second.theme == "dark"
        assert second.workers == 4
        assert second.memory_ceiling_mb == 4096
        assert second.experiment_timeout_s == 600
        assert second.auto_reload_blocks is False
        assert second.proxy_long_side == 2048
        assert second.debounce_ms == 250
        assert second.undo_limit == 50
        assert second.jpeg_quality == 80
        assert second.tiff_compression == "tiff_deflate"

    def test_changed_signal_emits_key(self, settings: Settings) -> None:
        seen: list[str] = []
        settings.changed.connect(seen.append)
        settings.set_language("es")
        settings.set_theme("dark")
        settings.set_workers(3)
        assert seen == ["ui/language", "ui/theme", "runner/workers"]

    def test_changed_not_emitted_for_same_value(self, settings: Settings) -> None:
        seen: list[str] = []
        settings.changed.connect(seen.append)
        settings.set_theme("system")
        settings.set_theme("system")
        assert seen == ["ui/theme"]

    def test_runner_settings_slice(self, settings: Settings) -> None:
        from pic_utils.runner.service import RunnerSettings

        settings.set_workers(3)
        settings.set_memory_ceiling_mb(4096)
        settings.set_experiment_timeout_s(120)
        runner = settings.runner_settings()
        assert isinstance(runner, RunnerSettings)
        assert runner.workers == 3
        assert runner.memory_ceiling_mb == 4096
        assert runner.experiment_timeout_s == 120.0


class TestPaths:
    def test_ensure_paths_creates_blocks_folder_with_example(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        settings.set_blocks_folder(tmp_path / "blocks")
        settings.ensure_paths()
        assert (tmp_path / "blocks").is_dir()
        example = tmp_path / "blocks" / "example_blocks.py"
        assert example.is_file()
        assert "posterize" in example.read_text(encoding="utf-8")


class TestMainWindow:
    def test_mode_switch_changes_stack(
        self, qtbot: object, settings: Settings, languages: LanguageManager
    ) -> None:
        window = MainWindow(settings=settings, languages=languages)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(window)
        stack = window.findChild(QStackedWidget)
        assert stack is not None
        assert stack.currentWidget() is window._lab_page
        window._mode_canvas.trigger()
        assert stack.currentWidget() is window._canvas_page
        window._mode_lab.trigger()
        assert stack.currentWidget() is window._lab_page
        assert window._mode_lab.isChecked() and not window._mode_canvas.isChecked()

    def test_progress_bar_updates(
        self, qtbot: object, settings: Settings, languages: LanguageManager
    ) -> None:
        window = MainWindow(settings=settings, languages=languages)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(window)
        window.show_progress(3, 10)
        assert window._progress.value() == 3
        assert window._progress.maximum() == 10
        assert not window._progress.isHidden()
        window.clear_progress()
        assert window._progress.isHidden()

    def test_run_status_label(
        self, qtbot: object, settings: Settings, languages: LanguageManager
    ) -> None:
        window = MainWindow(settings=settings, languages=languages)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(window)
        window.set_run_status("running")
        assert window._run_status.text() == "running"

    def test_retranslate_uses_translations(
        self, qtbot: object, settings: Settings, languages: LanguageManager
    ) -> None:
        languages.set_language("es")
        window = MainWindow(settings=settings, languages=languages)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(window)
        assert window._mode_lab.text() == "Laboratorio"

    def test_settings_changes_propagate(
        self, qtbot: object, settings: Settings, languages: LanguageManager
    ) -> None:
        window = MainWindow(settings=settings, languages=languages)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(window)
        settings.set_theme("dark")
        assert window._settings.theme == "dark"

    def test_diagnostics_text(
        self, qtbot: object, settings: Settings, languages: LanguageManager
    ) -> None:
        window = MainWindow(settings=settings, languages=languages)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(window)
        text = window.diagnostics_text()
        assert "Pic Utils" in text
        assert __version__ in text
        assert "Python" in text

    def test_drop_image_paths_signal(
        self,
        qtbot: object,
        settings: Settings,
        languages: LanguageManager,
        tmp_path: Path,
    ) -> None:
        window = MainWindow(settings=settings, languages=languages)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(window)
        seen: list[list[Path]] = []
        window.sources_dropped.connect(seen.append)
        image = tmp_path / "photo.png"
        image.write_bytes(b"")
        project = tmp_path / "exp.puproject"
        project.write_bytes(b"")

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(image)), QUrl.fromLocalFile(str(project))])
        enter = QDragEnterEvent(
            QPointF(1, 1).toPoint(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        drop = QDropEvent(
            enter.positionF() if hasattr(enter, "positionF") else enter.pos(),
            enter.possibleActions(),
            mime,
            enter.mouseButtons(),
            enter.keyboardModifiers(),
            enter.type(),
        )
        window.dropEvent(drop)
        assert seen == [[image]]
