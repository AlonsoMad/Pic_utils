"""Project save/load, relink, and preset tests (P5-08)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QDialog, QFileDialog

from pic_utils.pipeline.model import Pipeline, RunConfig
from pic_utils.ui.dialogs.relink_dialog import RelinkDialog
from pic_utils.ui.lab.project_actions import (
    PROJECT_FORMAT,
    ProjectData,
    ProjectFileError,
    load_preset,
    load_preset_pipeline,
    load_project,
    missing_files,
    save_preset,
    save_project,
    shipped_preset_paths,
)


@pytest.fixture
def source_png(tmp_path: Path) -> Path:
    path = tmp_path / "images" / "photo.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    return path


@pytest.fixture
def project_data(source_png: Path) -> ProjectData:
    pipeline = Pipeline.from_dict(
        {"steps": [{"id": "a", "block": "channel_permute", "params": {"order": "bgr"}}]}
    )
    return ProjectData(
        sources=(source_png,),
        second_image=None,
        pipeline=pipeline,
        run=RunConfig(mode="grid", batch_size=8, master_seed=7),
    )


class TestProjectIO:
    def test_round_trip(self, tmp_path: Path, project_data: ProjectData) -> None:
        path = tmp_path / "project.puproject"
        save_project(path, project_data)
        loaded = load_project(path)
        assert loaded.sources == project_data.sources
        assert loaded.second_image is None
        assert loaded.pipeline == project_data.pipeline
        assert loaded.run.mode == "grid"
        assert loaded.run.batch_size == 8
        assert loaded.run.master_seed == 7

    def test_paths_relative_inside_project_dir(
        self, tmp_path: Path, project_data: ProjectData
    ) -> None:
        path = tmp_path / "project.puproject"
        save_project(path, project_data)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["sources"] == ["images/photo.png"]
        loaded = load_project(path)
        assert loaded.sources == (tmp_path / "images" / "photo.png",)

    def test_paths_absolute_outside_project_dir(
        self, tmp_path: Path, source_png: Path, project_data: ProjectData
    ) -> None:
        other = tmp_path / "project" / "p.puproject"
        save_project(other, project_data)
        payload = json.loads(other.read_text(encoding="utf-8"))
        assert payload["sources"] == [str(source_png)]

    def test_round_trip_with_second_image(
        self, tmp_path: Path, project_data: ProjectData, source_png: Path
    ) -> None:
        second = source_png.parent / "second.png"
        cv2.imwrite(str(second), np.zeros((8, 8, 3), dtype=np.uint8))
        data = ProjectData(
            sources=project_data.sources,
            second_image=second,
            pipeline=project_data.pipeline,
            run=project_data.run,
        )
        path = tmp_path / "p.puproject"
        save_project(path, data)
        loaded = load_project(path)
        assert loaded.second_image == second

    def test_header_fields(self, tmp_path: Path, project_data: ProjectData) -> None:
        path = tmp_path / "p.puproject"
        save_project(path, project_data)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["format"] == PROJECT_FORMAT
        assert payload["version"] == 1
        assert payload["planner_version"] == 1
        assert "app_version" in payload

    def test_wrong_format_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.puproject"
        path.write_text(json.dumps({"format": "something.else"}), encoding="utf-8")
        with pytest.raises(ProjectFileError, match="format"):
            load_project(path)

    def test_newer_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "future.puproject"
        path.write_text(
            json.dumps({"format": PROJECT_FORMAT, "version": 99}), encoding="utf-8"
        )
        with pytest.raises(ProjectFileError, match="version"):
            load_project(path)

    def test_malformed_json_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.puproject"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProjectFileError, match="cannot read"):
            load_project(path)

    def test_missing_files_listed(
        self, tmp_path: Path, project_data: ProjectData, source_png: Path
    ) -> None:
        data = ProjectData(
            sources=(source_png, source_png.parent / "gone.png"),
            second_image=None,
            pipeline=project_data.pipeline,
            run=project_data.run,
        )
        missing = missing_files(data)
        assert missing == [source_png.parent / "gone.png"]


class TestProjectActions:
    def test_load_emits_project(
        self, tmp_path: Path, project_data: ProjectData
    ) -> None:
        from pic_utils.ui.lab.project_actions import ProjectActions

        path = tmp_path / "p.puproject"
        save_project(path, project_data)
        actions = ProjectActions()
        loaded: list[object] = []
        actions.project_loaded.connect(loaded.append)
        result = actions.load(path)
        assert result is not None
        assert loaded == [result]
        assert actions.path == path

    def test_load_with_missing_files_opens_relink(
        self, tmp_path: Path, project_data: ProjectData, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pic_utils.ui.lab.project_actions import ProjectActions

        missing = tmp_path / "images" / "gone.png"
        data = ProjectData(
            sources=(missing,),
            second_image=None,
            pipeline=project_data.pipeline,
            run=project_data.run,
        )
        path = tmp_path / "p.puproject"
        save_project(path, data)

        replacement = tmp_path / "images" / "replacement.png"
        cv2.imwrite(str(replacement), np.zeros((8, 8, 3), dtype=np.uint8))

        class FakeDialog:
            DialogCode = QDialog.DialogCode

            def __init__(self, missing_files: list, parent=None):
                pass

            def exec(self):
                return QDialog.DialogCode.Accepted

            def remapping(self):
                return {missing: replacement}

        import pic_utils.ui.lab.project_actions as pa

        monkeypatch.setattr(pa, "RelinkDialog", FakeDialog)
        actions = ProjectActions()
        loaded: list[object] = []
        actions.project_loaded.connect(loaded.append)
        result = actions.load(path)
        assert result is not None
        assert result.sources == (replacement,)

    def test_load_cancelled_when_relink_rejected(
        self, tmp_path: Path, project_data: ProjectData, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pic_utils.ui.lab.project_actions import ProjectActions

        data = ProjectData(
            sources=(tmp_path / "gone.png",),
            second_image=None,
            pipeline=project_data.pipeline,
            run=project_data.run,
        )
        path = tmp_path / "p.puproject"
        save_project(path, data)

        class RejectedDialog:
            DialogCode = QDialog.DialogCode

            def __init__(self, missing_files: list, parent=None):
                pass

            def exec(self):
                return QDialog.DialogCode.Rejected

            def remapping(self):
                return {}

        import pic_utils.ui.lab.project_actions as pa

        monkeypatch.setattr(pa, "RelinkDialog", RejectedDialog)
        actions = ProjectActions()
        loaded: list[object] = []
        actions.project_loaded.connect(loaded.append)
        assert actions.load(path) is None
        assert loaded == []


class TestRelinkDialog:
    def test_browse_records_choice(
        self, qtbot: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        replacement = tmp_path / "new.png"
        cv2.imwrite(str(replacement), np.zeros((4, 4, 3), dtype=np.uint8))
        dialog = RelinkDialog([tmp_path / "old.png"])
        qtbot.addWidget(dialog)  # type: ignore[attr-defined]
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *args, **kwargs: (str(replacement), "")),
        )
        dialog._browse(dialog._rows[0][0], dialog._rows[0][2])
        dialog.accept()
        assert dialog.remapping() == {tmp_path / "old.png": replacement}

    def test_cancel_returns_empty(self, qtbot: object, tmp_path: Path) -> None:
        dialog = RelinkDialog([tmp_path / "old.png"])
        qtbot.addWidget(dialog)  # type: ignore[attr-defined]
        dialog.reject()
        assert dialog.remapping() == {}


class TestPresets:
    def test_preset_round_trip(self, tmp_path: Path) -> None:
        payload = {"pipeline": {"steps": [{"id": "a", "block": "segment"}]}}
        path = tmp_path / "presets" / "mine.json"
        save_preset(path, "pipeline", "Mine", payload)
        document = load_preset(path)
        assert document["kind"] == "pipeline"
        assert document["name"] == "Mine"
        assert document["format"] == "pic_utils.preset"

    def test_preset_bad_format_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"format": "x"}), encoding="utf-8")
        with pytest.raises(ProjectFileError, match="format"):
            load_preset(path)

    def test_preset_pipeline_requires_kind(self, tmp_path: Path) -> None:
        path = tmp_path / "params.json"
        save_preset(path, "params", "P", {"block": "segment", "params": {}})
        with pytest.raises(ProjectFileError, match="kind"):
            load_preset_pipeline(path)

    def test_preset_pipeline_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        save_preset(
            path, "pipeline", "P", {"pipeline": {"steps": [{"id": "a", "block": "segment"}]}}
        )
        pipeline = load_preset_pipeline(path)
        assert pipeline.steps[0].block == "segment"

    def test_four_shipped_examples_exist(self) -> None:
        paths = shipped_preset_paths()
        assert len(paths) == 4
        for path in paths:
            document = load_preset(path)
            assert document["format"] == "pic_utils.preset"

    def test_shipped_examples_load(self) -> None:
        from pic_utils.blocks.registry import BlockRegistry
        from pic_utils.pipeline.validate import validate_pipeline

        registry = BlockRegistry(None)
        for path in shipped_preset_paths():
            document = load_preset(path)
            if document["kind"] == "pipeline":
                pipeline = load_preset_pipeline(path)
                issues = validate_pipeline(
                    pipeline,
                    registry.snapshot(),
                    second_image_available=False,
                    sources_available=True,
                )
                errors = [i for i in issues if i.severity == "error"]
                assert not errors, f"{path.name}: {[i.code for i in errors]}"
