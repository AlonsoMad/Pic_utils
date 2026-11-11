"""Run layout, manifest, and results.jsonl tests (architecture §9.1-§9.2)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pic_utils.core.jsonio import read_json
from pic_utils.runner.manifest import (
    RUN_FORMAT,
    ResultRecord,
    RunStore,
    new_run_id,
    result_id,
)


def make_record(experiment: int = 3, layer: int = 0, **overrides: object) -> ResultRecord:
    values: dict[str, object] = {
        "id": result_id(experiment, layer),
        "experiment": experiment,
        "layer": layer,
        "source": 0,
        "kind": "image",
        "path": f"results/result_{experiment:04d}_{layer:02d}.png",
        "thumb": f"thumbs/result_{experiment:04d}_{layer:02d}.png",
        "width": 8,
        "height": 8,
        "seed": 123456789,
        "params": {"step": {"threshold": 128}},
        "elapsed_ms": 812,
    }
    values.update(overrides)
    return ResultRecord(**values)  # type: ignore[arg-type]


def test_new_run_id_format() -> None:
    run_id = new_run_id()
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", run_id)


def test_new_run_id_is_unique() -> None:
    assert len({new_run_id() for _ in range(50)}) == 50


def test_result_id_is_zero_padded() -> None:
    assert result_id(3, 0) == "0003:00"
    assert result_id(12, 7) == "0012:07"


def test_create_builds_layout_and_running_manifest(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "20261111-100000-abc123")
    run_dir = store.create({"pipeline": {"steps": []}})
    assert run_dir == tmp_path / "20261111-100000-abc123"
    for name in ("results", "thumbs", "blocks"):
        assert (run_dir / name).is_dir()
    header = read_json(store.manifest_path)
    assert header["format"] == RUN_FORMAT
    assert header["version"] == 1
    assert header["run_id"] == "20261111-100000-abc123"
    assert header["status"] == "running"
    assert header["pipeline"] == {"steps": []}


def test_write_manifest_is_atomic_and_rewritable(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    store.create({})
    store.write_manifest({"status": "completed", "summary": {"completed": 2}})
    header = read_json(store.manifest_path)
    assert header["status"] == "completed"
    assert header["run_id"] == "r1"
    # No temp files left behind.
    assert [path.name for path in tmp_path.glob("r1/.*")] == []


def test_append_and_read_results_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    store.create({})
    first = make_record(experiment=0, layer=0)
    second = make_record(experiment=1, layer=0, seed=42)
    store.append_result(first)
    store.append_result(second)

    lines = store.results_jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "0000:00"

    run_data = store.read()
    assert run_data.results == (first, second)


def test_read_tolerates_torn_final_line(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    store.create({})
    store.append_result(make_record(experiment=0))
    good = store.results_jsonl_path.read_text(encoding="utf-8")
    torn = good + '{"id": "0001:00", "experiment": 1, "lay'
    store.results_jsonl_path.write_text(torn, encoding="utf-8")

    run_data = store.read()
    assert [record.id for record in run_data.results] == ["0000:00"]


def test_read_rejects_malformed_middle_line(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    store.create({})
    store.append_result(make_record(experiment=0))
    store.append_result(make_record(experiment=1))
    good = store.results_jsonl_path.read_text(encoding="utf-8")
    lines = good.splitlines()
    lines[0] = '{"nonsense": true}'
    store.results_jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        store.read()


def test_curation_defaults_to_empty_and_round_trips(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    assert store.read_curation() == {}
    store.create({})
    store.set_curation("0003:00", "accepted")
    store.set_curation("0004:00", "discarded")
    assert store.read_curation() == {"0003:00": "accepted", "0004:00": "discarded"}
    store.clear_curation("0003:00")
    assert store.read_curation() == {"0004:00": "discarded"}
    # Clearing an absent id is a no-op.
    store.clear_curation("0003:00")
    assert store.read_curation() == {"0004:00": "discarded"}


def test_set_curation_rejects_unknown_state(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    store.create({})
    with pytest.raises(ValueError, match="accepted"):
        store.set_curation("0003:00", "maybe")


def test_result_and_thumb_paths(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    assert store.result_file(3, 0) == tmp_path / "r1" / "results" / "result_0003_00.png"
    assert store.thumb_file(12, 5) == tmp_path / "r1" / "thumbs" / "result_0012_05.png"
    assert store.blocks_dir() == tmp_path / "r1" / "blocks"


def test_result_record_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        ResultRecord.from_dict({"id": "0000:00"})


def test_result_record_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        ResultRecord.from_dict(["0000:00"])


def test_read_returns_empty_results_before_any_append(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "r1")
    store.create({})
    run_data = store.read()
    assert run_data.results == ()
    assert run_data.curation == {}
