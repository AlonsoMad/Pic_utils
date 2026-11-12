"""End-to-end acceptance suite (P7-01).

One test per acceptance criterion (PRD §14.1, AC-01…AC-12). The tests drive
the real components — the block registry, the spawned worker pool, the run
store, the results grid, the canvas document, and the export renderer — on
synthetic images. AC-11 (the 48 MP memory ceiling) is measured by
``tests/perf/test_perf.py`` and re-asserted here with the same helper.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from perf_support import (
    RssSampler,
    build_request,
    make_48mp_source,
    process_tree_rss_mb,
    run_to_completion,
)
from pytestqt.qtbot import QtBot

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.export import export_canvas, render_canvas
from pic_utils.canvas.model import CanvasDoc, CanvasLayer, assets_dir_for
from pic_utils.core.image import Image, Mask
from pic_utils.core.jsonio import read_json
from pic_utils.io.loader import load_image, probe_image
from pic_utils.io.saver import save_image
from pic_utils.pipeline.model import Pipeline, RunConfig, SweepAxis
from pic_utils.results.trash import SessionTrash
from pic_utils.runner.manifest import ResultRecord, RunStore
from pic_utils.runner.service import RunService, RunSummary
from pic_utils.settings import Settings
from pic_utils.ui.canvas.actions import CanvasActions
from pic_utils.ui.canvas.commands import ApplyMaskCommand
from pic_utils.ui.canvas.page import CanvasPage
from pic_utils.ui.lab.page import LabPage
from pic_utils.ui.lab.project_actions import ProjectData, load_project, save_project

pytestmark = pytest.mark.usefixtures("qapp")


def solid(color: tuple[int, int, int], size: int = 64) -> Image:
    return Image.from_rgb(np.full((size, size, 3), color, dtype=np.uint8))


def write_solid(path: Path, color: tuple[int, int, int], size: int = 64) -> Path:
    save_image(solid(color, size), path)
    return path


def make_source(tmp_path: Path) -> Path:
    return write_solid(tmp_path / "source.png", (10, 20, 30))


def make_request(
    results_root: Path, blocks_dir: Path, sources: list[Path], pipeline: Pipeline, config: RunConfig
) -> object:
    registry = BlockRegistry(blocks_dir)
    return replace(
        build_request(results_root, sources[0]),  # type: ignore[arg-type]
        pipeline=pipeline,
        snapshot=registry.snapshot(),
        sources=tuple(sources),
        run_config=config,
        user_blocks_dir=blocks_dir,
    )


def run_batch(
    qtbot: QtBot, service: RunService, request: object, timeout_ms: int = 300_000
) -> RunSummary:
    return run_to_completion(service, request, qtbot.waitUntil, timeout_ms=timeout_ms)  # type: ignore[arg-type]


# --- AC-01 -------------------------------------------------------------------


def test_ac01_loads_a_48mp_source_within_the_memory_ceiling(
    settings: Settings, tmp_path: Path
) -> None:
    source = make_48mp_source(tmp_path / "big.png")
    info = probe_image(source)
    assert info.width * info.height == 48_000_000

    page = LabPage(settings)
    page.run_controls().add_sources([source])

    assert page.run_controls().sources() == [source]
    image = load_image(source)
    assert (image.width, image.height) == (info.width, info.height)
    assert process_tree_rss_mb() < 8192
    page.deleteLater()


# --- AC-02 -------------------------------------------------------------------


def test_ac02_hot_reload_adds_a_user_block(qtbot: QtBot, settings: Settings) -> None:
    page = LabPage(settings)
    blocks = settings.blocks_folder
    blocks.mkdir(parents=True, exist_ok=True)
    (blocks / "my_blocks.py").write_text(
        '''"""AC-02 user block."""

from typing import Annotated

import numpy as np

from pic_utils.blocks import IntRange, OutputKind, block
from pic_utils.core.image import Image


@block(name="my_posterize", category="e2e", description="posterize",
       output=OutputKind.IMAGE)
def my_posterize(
    image: Image, levels: Annotated[int, IntRange(2, 32, step=1)] = 4
) -> Image:
    rgb = image.rgb.astype(np.float32) / 255.0
    quantized = np.round(rgb * (levels - 1)) / (levels - 1)
    return Image.from_rgb(np.clip(quantized * 255.0, 0, 255).astype(np.uint8))
''',
        encoding="utf-8",
    )

    report = page.library().reload_now()
    assert "my_posterize" in report.loaded

    names = {
        page.library()._model.data(page.library()._model.index(row, 0))
        for row in range(page.library()._model.rowCount())
    }
    assert "my_posterize" in names

    # auto-generated controls: the block's parameters become widgets
    page.editor().add_block("my_posterize")
    step_id = page.editor().model().step_id_at(0)
    assert step_id is not None
    page.parameters().show_step(step_id)
    assert page.parameters().widget("levels") is not None


# --- AC-03 -------------------------------------------------------------------


def test_ac03_sweep_of_20_values_lands_in_one_run_folder(
    qtbot: QtBot, settings: Settings, blocks_dir: Path, tmp_path: Path
) -> None:
    source = make_source(tmp_path)
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "e2e_sweep"}]})
    config = RunConfig(
        mode="grid",
        master_seed=3,
        axes=(SweepAxis("s1", "amount", tuple(range(20))),),
    )
    service = RunService()
    summary = run_batch(
        qtbot,
        service,
        make_request(settings.results_root, blocks_dir, [source], pipeline, config),
    )

    assert summary.status == "completed"
    assert summary.completed == 20
    store = RunStore(settings.results_root, summary.run_id)
    assert (store.run_dir / "manifest.json").is_file()
    records = store.read().results
    assert len(records) == 20
    assert all((store.run_dir / record.path).is_file() for record in records)
    assert len({record.experiment for record in records}) == 20
    service.deleteLater()


# --- AC-04 -------------------------------------------------------------------


def test_ac04_cancel_keeps_completed_results(
    qtbot: QtBot, settings: Settings, blocks_dir: Path, tmp_path: Path
) -> None:
    source = make_source(tmp_path)
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "e2e_sleep"}]})
    config = RunConfig(
        mode="grid",
        master_seed=5,
        axes=(SweepAxis("s1", "seconds", (1.0,) * 6),),
    )
    request = make_request(settings.results_root, blocks_dir, [source], pipeline, config)
    request = replace(request, settings=replace(request.settings, workers=1))  # type: ignore[attr-defined]
    service = RunService()

    records: list[object] = []
    service.result_ready.connect(records.append)
    service.start(request)  # type: ignore[arg-type]
    qtbot.waitUntil(lambda: len(records) >= 1, timeout=60_000)
    service.cancel()
    summary: RunSummary | None = None
    service.run_finished.connect(lambda s: records.append(s) or None)
    qtbot.waitUntil(lambda: any(isinstance(item, RunSummary) for item in records), timeout=60_000)
    summary = next(item for item in records if isinstance(item, RunSummary))
    qtbot.waitUntil(lambda: not service.is_running, timeout=10_000)

    assert summary.status == "cancelled"
    assert summary.completed >= 1
    assert summary.cancelled >= 1
    disk = RunStore(settings.results_root, summary.run_id)
    completed = disk.read().results
    assert completed
    assert all((disk.run_dir / record.path).is_file() for record in completed)
    service.deleteLater()


# --- AC-05 -------------------------------------------------------------------


def test_ac05_raising_block_skips_one_experiment_and_the_batch_completes(
    qtbot: QtBot, settings: Settings, blocks_dir: Path, tmp_path: Path
) -> None:
    source = make_source(tmp_path)
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "e2e_raise"}]})
    config = RunConfig(
        mode="grid",
        master_seed=7,
        axes=(SweepAxis("s1", "fail", (True, False)),),
    )
    service = RunService()
    summary = run_batch(
        qtbot,
        service,
        make_request(settings.results_root, blocks_dir, [source], pipeline, config),
    )

    assert summary.status == "completed"
    assert summary.completed == 1
    assert summary.failed == 1
    disk = RunStore(settings.results_root, summary.run_id)
    assert [record.experiment for record in disk.read().results] == [1]
    run_log = (disk.run_dir / "run.log").read_text(encoding="utf-8")
    assert "block.raised" in run_log
    assert "e2e boom" in run_log
    service.deleteLater()


# --- AC-06 -------------------------------------------------------------------


def test_ac06_discard_moves_to_session_trash_and_restores(
    settings: Settings,
) -> None:
    run_root = settings.results_root
    store = RunStore(run_root, "20260101-000000-000")
    store.create({"run_id": store.run_id})
    record_path = store.run_dir / "result_0000_00.png"
    thumb_path = store.run_dir / "thumb_0000_00.png"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(solid((1, 2, 3), 8), record_path)
    save_image(solid((1, 2, 3), 8), thumb_path)
    store.append_result(
        _record(
            id="0000:00",
            path=record_path.name,
            thumb=thumb_path.name,
        )
    )

    trash = SessionTrash(run_root, session_id="test-session")
    model = _grid_model(store, trash)
    record = model._record_at(0)
    assert record is not None

    model.discard(0)
    assert not record_path.exists()
    assert not thumb_path.exists()
    assert (trash.session_dir / store.run_id / record.path).is_file()
    assert store.read_curation() == {"0000:00": "discarded"}

    model.set_show_discarded(True)
    model.restore(0)
    assert record_path.is_file()
    assert thumb_path.is_file()
    assert store.read_curation() == {}


def _record(id: str, path: str, thumb: str) -> ResultRecord:
    return ResultRecord(
        id=id,
        experiment=0,
        layer=0,
        source=0,
        kind="image",
        path=path,
        thumb=thumb,
        width=8,
        height=8,
        seed=7,
        params={},
        elapsed_ms=0,
        accepted=None,
    )


def _grid_model(store: RunStore, trash: SessionTrash) -> object:
    from pic_utils.ui.lab.results_grid import ResultsGridModel

    return ResultsGridModel(store, trash)


# --- AC-07 -------------------------------------------------------------------


def test_ac07_canvas_compose_transforms_zorder_opacity_blend_mask_crop_group(
    settings: Settings, tmp_path: Path
) -> None:
    red = write_solid(tmp_path / "red.png", (255, 0, 0))
    blue = write_solid(tmp_path / "blue.png", (0, 0, 255))
    page = CanvasPage(settings)
    doc = page.doc()

    page._add_layer_from_path(red)
    page._add_layer_from_path(blue)
    assert [layer.name for layer in doc.layers] == ["red", "blue"]
    red_id, blue_id = doc.layers[0].id, doc.layers[1].id

    # transform, opacity, blend, z-order
    page._on_property_changed(blue_id, "x", 10.0)
    page._on_property_changed(blue_id, "y", 5.0)
    page._on_property_changed(blue_id, "scale", 0.5)
    page._on_property_changed(blue_id, "rotation", 15.0)
    page._on_property_changed(blue_id, "opacity", 0.75)
    page._on_property_changed(blue_id, "blend", "multiply")
    assert doc.layer(blue_id).blend == "multiply"

    # group and mask
    page._on_group_requested([red_id, blue_id])
    group_id = next(iter(doc.groups))
    assert doc.layer(red_id).group_id == group_id
    store = AssetStore(doc.ensure_assets_dir())
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, 32:] = 255
    mask_ref = store.put_mask(Mask(mask))
    page.undo_stack().push(ApplyMaskCommand(doc, blue_id, mask_ref))
    assert doc.layer(blue_id).mask_ref == mask_ref

    # crop
    page._on_crop_requested(_rect(4, 4, 40, 40))
    assert (doc.width, doc.height) == (40, 40)
    assert doc.layer(blue_id).x == 6.0

    # the composite renders with every property
    rendered = render_canvas(doc)
    assert (rendered.width(), rendered.height()) == (40, 40)
    assert page.undo_stack().canUndo()
    page.shutdown()


def _rect(x: float, y: float, width: float, height: float) -> object:
    from PySide6.QtCore import QRectF

    return QRectF(x, y, width, height)


# --- AC-08 -------------------------------------------------------------------


def test_ac08_save_and_reopen_restores_layers_transforms_and_masks(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    red = write_solid(tmp_path / "red.png", (255, 0, 0))
    page = CanvasPage(settings)
    page._add_layer_from_path(red)
    layer_id = page.doc().layers[0].id
    page._on_property_changed(layer_id, "x", 12.0)
    page._on_property_changed(layer_id, "opacity", 0.5)
    store = AssetStore(page.doc().ensure_assets_dir())
    mask = np.full((64, 64), 128, dtype=np.uint8)
    mask_ref = store.put_mask(Mask(mask))
    page.undo_stack().push(ApplyMaskCommand(page.doc(), layer_id, mask_ref))

    path = tmp_path / "collage.pucanvas"
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(path), ""),
    )
    actions = page.canvas_actions()
    saved = actions.save(page.doc())
    assert saved == path
    assert actions.wait_for_save()

    reopened = CanvasActions(settings)
    loaded = reopened.open(path)
    assert loaded is not None
    assert [layer.id for layer in loaded.layers] == [layer_id]
    assert loaded.layer(layer_id).x == 12.0
    assert loaded.layer(layer_id).opacity == 0.5
    assert loaded.layer(layer_id).mask_ref == mask_ref
    assert assets_dir_for(path).is_dir()
    assert (assets_dir_for(path) / mask_ref).is_file()
    page.shutdown()


# --- AC-09 -------------------------------------------------------------------


def test_ac09_saved_project_rerun_matches_the_original_run(
    qtbot: QtBot, settings: Settings, blocks_dir: Path, tmp_path: Path
) -> None:
    source = make_source(tmp_path)
    pipeline = Pipeline.from_dict(
        {
            "steps": [
                {"id": "s1", "block": "e2e_sweep", "params": {}},
                {"id": "s2", "block": "e2e_noise", "params": {}},
            ]
        }
    )
    config = RunConfig(
        mode="grid",
        master_seed=42,
        axes=(SweepAxis("s1", "amount", (0, 10, 20, 30)),),
    )
    project_path = tmp_path / "project.puproject"
    save_project(
        project_path,
        ProjectData(sources=(source,), second_image=None, pipeline=pipeline, run=config),
    )
    loaded = load_project(project_path)
    assert loaded.sources == (source,)
    assert loaded.pipeline.to_dict() == pipeline.to_dict()
    assert loaded.run.master_seed == 42
    # load_project returns raw axis tuples; normalize them to SweepAxis values
    run = replace(
        loaded.run,
        axes=tuple(
            SweepAxis(step_id, param, tuple(values)) for step_id, param, values in loaded.run.axes
        ),
    )

    service = RunService()
    first = run_batch(
        qtbot,
        service,
        make_request(settings.results_root, blocks_dir, [source], loaded.pipeline, run),
    )
    second = run_batch(
        qtbot,
        service,
        make_request(settings.results_root, blocks_dir, [source], loaded.pipeline, run),
    )

    assert first.completed == second.completed == 4
    first_store = RunStore(settings.results_root, first.run_id)
    second_store = RunStore(settings.results_root, second.run_id)
    first_header = read_json(first_store.manifest_path)
    second_header = read_json(second_store.manifest_path)
    assert [experiment["seed"] for experiment in first_header["experiments"]] == [
        experiment["seed"] for experiment in second_header["experiments"]
    ]
    first_pixels = [
        _pixels(first_store.run_dir / record.path)
        for record in sorted(first_store.read().results, key=lambda r: (r.experiment, r.layer))
    ]
    second_pixels = [
        _pixels(second_store.run_dir / record.path)
        for record in sorted(second_store.read().results, key=lambda r: (r.experiment, r.layer))
    ]
    assert first_pixels == second_pixels
    service.deleteLater()


def _pixels(path: Path) -> list[int]:
    return [int(value) for value in np.asarray(load_image(path).rgb).ravel()[:64]]


# --- AC-10 -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "extension", "mime"),
    [("png", ".png", "image/png"), ("jpeg", ".jpg", "image/jpeg"), ("tiff", ".tiff", "image/tiff")],
)
def test_ac10_exports_full_resolution_in_each_format(
    settings: Settings, tmp_path: Path, fmt: str, extension: str, mime: str
) -> None:
    from PIL import Image as PILImage

    red = write_solid(tmp_path / "red.png", (255, 0, 0), size=64)
    accepted = write_solid(tmp_path / "accepted.png", (10, 20, 30), size=48)
    store = AssetStore(tmp_path / "assets")
    doc = CanvasDoc(size=(64, 64), assets_dir=store.assets_dir)
    doc.add_layer(CanvasLayer(id="a", name="red", image_ref=store.put_image(load_image(red))))
    report = export_canvas(
        doc,
        tmp_path / "exports",
        fmt=fmt,
        jpeg_quality=92,
        scope="canvas_and_accepted",
        accepted=(accepted,),
    )
    canvas_file = report.folder / f"canvas{extension}"
    result_file = report.folder / "results" / f"accepted{extension}"
    assert canvas_file.is_file()
    assert result_file.is_file()
    with PILImage.open(canvas_file) as image:
        assert image.size == (64, 64)
        assert image.format == mime.split("/")[1].upper()
    with PILImage.open(result_file) as image:
        assert image.size == (48, 48)


def test_ac10_export_scopes_select_what_is_written(settings: Settings, tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "assets")
    doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
    doc.add_layer(CanvasLayer(id="a", name="a", image_ref=store.put_image(solid((1, 1, 1), 8))))
    accepted = write_solid(tmp_path / "accepted.png", (2, 2, 2), size=8)
    other = write_solid(tmp_path / "other.png", (3, 3, 3), size=8)

    only_canvas = export_canvas(doc, tmp_path / "scope-canvas", fmt="png", scope="canvas")
    assert [path.name for path in only_canvas.files] == ["canvas.png"]

    with_accepted = export_canvas(
        doc,
        tmp_path / "scope-accepted",
        fmt="png",
        scope="canvas_and_accepted",
        accepted=(accepted,),
    )
    assert sorted(path.name for path in with_accepted.files) == ["accepted.png", "canvas.png"]
    assert (with_accepted.folder / "results" / "accepted.png").is_file()
    assert not (with_accepted.folder / "results" / "other.png").exists()

    everything = export_canvas(
        doc,
        tmp_path / "scope-everything",
        fmt="png",
        scope="everything",
        run_results=(accepted, other),
    )
    assert sorted(path.name for path in everything.files) == [
        "accepted.png",
        "canvas.png",
        "other.png",
    ]


# --- AC-11 -------------------------------------------------------------------


@pytest.mark.slow
def test_ac11_default_batch_of_32_at_48mp_stays_under_the_ceiling(
    qtbot: QtBot, tmp_path: Path
) -> None:
    source = make_48mp_source(tmp_path / "source.png")
    request = build_request(tmp_path / "results", source, experiments=32)
    service = RunService()
    with RssSampler() as sampler:
        summary = run_to_completion(service, request, qtbot.waitUntil, timeout_ms=900_000)
    assert summary.status == "completed"
    assert summary.completed == 32
    assert sampler.peak_mb < 8192
    service.deleteLater()


# --- AC-12 -------------------------------------------------------------------


def test_ac12_switching_language_updates_the_ui_without_restart(
    qtbot: QtBot, settings: Settings
) -> None:
    from pic_utils.ui.i18n import LanguageManager
    from pic_utils.ui.main_window import MainWindow

    languages = LanguageManager(settings)
    window = MainWindow(settings=settings, languages=languages)
    qtbot.addWidget(window)

    assert languages.language == "en"
    assert window._mode_lab.text() == "Lab"

    languages.set_language("es")
    qtbot.waitUntil(lambda: window._mode_lab.text() == "Laboratorio", timeout=5_000)

    # the whole window retranslates: menus, Lab, and Canvas
    assert window._mode_canvas.text() == "Lienzo"
    assert window._file_menu.title() == "&Archivo"
    assert window._lab_page.run_controls()._launch.text() == "Lanzar"
    assert window._lab_page.library()._reload.text() == "Recargar bloques"
    assert window.canvas_page()._crop_button.text() == "Recortar"
    assert window.canvas_page()._export_button.text() == "Exportar…"

    # the settings dialog updates live as well
    from pic_utils.ui.dialogs.settings_dialog import SettingsDialog

    dialog = SettingsDialog(settings, languages, window)
    qtbot.addWidget(dialog)
    dialog._apply_language("es")
    assert dialog._labels["language"].text() == "&Idioma"
    assert dialog._labels["workers"].text() == "&Trabajadores"
    assert dialog._runner_box.title() == "Ejecutor"
    assert dialog.windowTitle() == "Ajustes"

    # switching back restores the English strings on the same instances
    languages.set_language("en")
    assert window._mode_lab.text() == "Lab"
    assert dialog.windowTitle() == "Settings"
