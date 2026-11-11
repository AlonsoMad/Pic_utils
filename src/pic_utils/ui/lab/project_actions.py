"""Project save/load, relink flow, and preset management (§17, FR-PERS-01..06).

Project files (``*.puproject``) embed the full pipeline and run config;
paths are relative to the project file when inside its directory, absolute
otherwise (§17.1). Loading resolves relative paths against the project
directory; missing files trigger the relink dialog (FR-PERS-05). Presets
(``pic_utils.preset``) are a separate library; the shipped examples are
read-only resources surfaced under "Examples" (§17.2).
"""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QWidget

from pic_utils import __version__
from pic_utils.pipeline.model import Pipeline, RunConfig
from pic_utils.ui.dialogs.relink_dialog import RelinkDialog

__all__ = [
    "CANVAS_SUFFIX",
    "IMAGE_SUFFIXES",
    "PROJECT_FORMAT",
    "PROJECT_SUFFIX",
    "PRESET_FORMAT",
    "ProjectData",
    "ProjectActions",
    "load_preset",
    "missing_files",
    "preset_names",
    "save_preset",
    "shipped_preset_paths",
]

PROJECT_FORMAT = "pic_utils.project"
PROJECT_VERSION = 1
PRESET_FORMAT = "pic_utils.preset"
PRESET_VERSION = 1
PLANNER_VERSION = 1

PROJECT_SUFFIX = ".puproject"
CANVAS_SUFFIX = ".pucanvas"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class ProjectData:
    """The in-memory project (§17.1)."""

    sources: tuple[Path, ...]
    second_image: Path | None
    pipeline: Pipeline
    run: RunConfig


class ProjectFileError(Exception):
    """A project or preset file could not be read (§18.1)."""


# --- project I/O -------------------------------------------------------------


def _relative(path: Path, project_dir: Path) -> str:
    """Relative to the project dir when inside it, absolute otherwise (§17.1)."""
    try:
        return str(path.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        return str(path)


def _resolve(raw: str, project_dir: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else project_dir / path


def save_project(path: Path, data: ProjectData) -> None:
    """Write the project file (§17.1 schema); creates the parent folder."""
    payload = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "planner_version": PLANNER_VERSION,
        "app_version": __version__,
        "sources": [_relative(source, path.parent) for source in data.sources],
        "second_image": None
        if data.second_image is None
        else _relative(data.second_image, path.parent),
        "pipeline": data.pipeline.to_dict(),
        "run": {
            "mode": data.run.mode,
            "batch_size": data.run.batch_size,
            "random_samples": data.run.random_samples,
            "master_seed": data.run.master_seed,
            "axes": [
                {"step_id": axis.step_id, "param": axis.param, "values": list(axis.values)}
                for axis in data.run.axes
            ],
            "sampled": [
                {"step_id": step_id, "param": param} for step_id, param in data.run.sampled
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_project(path: Path) -> ProjectData:
    """Read a project file (§17.1); paths resolve against its directory.

    Raises:
        ProjectFileError: wrong format, unsupported version, or bad shape.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectFileError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProjectFileError(f"{path}: the project file must be a JSON object")
    if payload.get("format") != PROJECT_FORMAT:
        raise ProjectFileError(f"{path}: format must be {PROJECT_FORMAT!r}")
    version = payload.get("version")
    if not isinstance(version, int) or version > PROJECT_VERSION:
        raise ProjectFileError(f"{path}: unsupported project version {version!r}")
    project_dir = Path(path).parent

    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list) or not all(isinstance(s, str) for s in raw_sources):
        raise ProjectFileError(f"{path}: 'sources' must be a list of paths")
    second = payload.get("second_image")
    if second is not None and not isinstance(second, str):
        raise ProjectFileError(f"{path}: 'second_image' must be a string or null")

    try:
        pipeline = Pipeline.from_dict(payload.get("pipeline", {}))
    except ValueError as error:
        raise ProjectFileError(f"{path}: {error}") from error

    run_raw = payload.get("run", {})
    if not isinstance(run_raw, dict):
        raise ProjectFileError(f"{path}: 'run' must be an object")
    axes = []
    for axis in run_raw.get("axes", []):
        if isinstance(axis, dict) and isinstance(axis.get("values"), list):
            axes.append(
                (
                    axis.get("step_id"),
                    axis.get("param"),
                    tuple(axis.get("values", [])),
                )
            )
    run = RunConfig(
        mode=str(run_raw.get("mode", "grid")),
        batch_size=int(run_raw.get("batch_size", 32)),
        random_samples=int(run_raw.get("random_samples", 0)),
        master_seed=int(run_raw.get("master_seed", 0)),
        axes=axes,  # type: ignore[arg-type]
        sampled=tuple(
            (s.get("step_id"), s.get("param"))
            for s in run_raw.get("sampled", [])
            if isinstance(s, dict)
        ),
    )
    return ProjectData(
        sources=tuple(_resolve(s, project_dir) for s in raw_sources),
        second_image=None if second is None else _resolve(second, project_dir),
        pipeline=pipeline,
        run=run,
    )


def missing_files(data: ProjectData) -> list[Path]:
    """Project sources (and the second image) that do not exist on disk."""
    paths = list(data.sources)
    if data.second_image is not None:
        paths.append(data.second_image)
    return [path for path in paths if not path.is_file()]


# --- presets (§17.2) -----------------------------------------------------------


def save_preset(path: Path, kind: str, name: str, payload: dict[str, Any]) -> None:
    """Write a preset file; ``kind`` is ``"pipeline"`` or ``"params"``."""
    document = {
        "format": PRESET_FORMAT,
        "version": PRESET_VERSION,
        "kind": kind,
        "name": name,
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


def load_preset(path: Path) -> dict[str, Any]:
    """Read a preset file; validates format and version.

    Raises:
        ProjectFileError: wrong format or unsupported version.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectFileError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict) or document.get("format") != PRESET_FORMAT:
        raise ProjectFileError(f"{path}: format must be {PRESET_FORMAT!r}")
    version = document.get("version")
    if not isinstance(version, int) or version > PRESET_VERSION:
        raise ProjectFileError(f"{path}: unsupported preset version {version!r}")
    return document


def preset_names(folder: Path, kind: str) -> list[str]:
    """Preset display names of one kind in a user preset folder."""
    names: list[str] = []
    if not folder.is_dir():
        return names
    for path in sorted(folder.glob("*.json")):
        try:
            document = load_preset(path)
        except ProjectFileError:
            continue
        if document.get("kind") == kind:
            names.append(str(document.get("name", path.stem)))
    return names


def shipped_preset_paths() -> list[Path]:
    """The read-only example presets shipped with the app (§17.2)."""
    source = importlib.resources.files("pic_utils.resources.presets")
    with importlib.resources.as_file(source) as folder:
        return sorted(folder.glob("*.json"))


def load_preset_pipeline(path: Path) -> Pipeline:
    """Load a ``kind: pipeline`` preset as a :class:`Pipeline`.

    Raises:
        ProjectFileError: wrong kind or malformed pipeline.
    """
    document = load_preset(path)
    if document.get("kind") != "pipeline":
        raise ProjectFileError(f"{path}: preset kind must be 'pipeline'")
    try:
        return Pipeline.from_dict(document.get("pipeline", {}))
    except ValueError as error:
        raise ProjectFileError(f"{path}: {error}") from error


# --- controller ------------------------------------------------------------------


class ProjectActions(QObject):
    """Save/load with dialogs and the relink flow; wired by the Lab page."""

    project_loaded = Signal(object)  # ProjectData
    project_saved = Signal(Path)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    def save(self, data: ProjectData, parent: QWidget | None = None) -> Path | None:
        """Save to the current path, or ask for one; returns the path."""
        target = self._path
        if target is None:
            chosen, _ = QFileDialog.getSaveFileName(
                parent,
                parent.tr("Save project") if parent else "Save project",
                "",
                "Pic Utils project (*.puproject)",
            )
            if not chosen:
                return None
            target = Path(chosen)
            if not target.suffix:
                target = target.with_suffix(PROJECT_SUFFIX)
        save_project(target, data)
        self._path = target
        self.project_saved.emit(target)
        return target

    def save_as(self, data: ProjectData, parent: QWidget | None = None) -> Path | None:
        """Save under a new path; subsequent saves go there."""
        self._path = None
        return self.save(data, parent)

    def load(self, path: Path, parent: QWidget | None = None) -> ProjectData | None:
        """Load a project; missing files open the relink dialog first."""
        data = load_project(path)
        missing = missing_files(data)
        if missing:
            dialog = RelinkDialog(missing, parent)
            if dialog.exec() != RelinkDialog.DialogCode.Accepted or not dialog.remapping():
                return None
            remapping = dialog.remapping()
            data = ProjectData(
                sources=tuple(remapping.get(s, s) for s in data.sources),
                second_image=None
                if data.second_image is None
                else remapping.get(data.second_image, data.second_image),
                pipeline=data.pipeline,
                run=data.run,
            )
        self._path = path
        self.project_loaded.emit(data)
        return data
