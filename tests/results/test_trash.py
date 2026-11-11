"""Session trash lifecycle and curation ordering tests (architecture §9.3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pic_utils.core.jsonio import write_json_atomic
from pic_utils.results.store import (
    AcceptedIndex,
    discard_result,
    list_runs,
    load_run,
    restore_result,
)
from pic_utils.results.trash import SessionTrash, new_session_id, prepare_trash
from pic_utils.runner.manifest import ResultRecord, RunStore


def make_record(experiment: int = 3, layer: int = 0, **overrides: object) -> ResultRecord:
    values: dict[str, object] = {
        "id": f"{experiment:04d}:{layer:02d}",
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


def seed_run(results_root: Path, run_id: str, records: list[ResultRecord]) -> RunStore:
    store = RunStore(results_root, run_id)
    store.create({})
    for record in records:
        store.result_file(record.experiment, record.layer).write_bytes(b"png-bytes")
        store.thumb_file(record.experiment, record.layer).write_bytes(b"thumb-bytes")
        store.append_result(record)
    return store


def test_session_id_format() -> None:
    assert re.fullmatch(r"\d{8}-\d{6}-\d+", new_session_id())


def test_discard_moves_files_preserving_layout(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s1")
    record = make_record(experiment=3)
    store = seed_run(tmp_path, "r1", [record])

    destinations = trash.discard("r1", [record.path, record.thumb])

    assert destinations == [
        tmp_path / "trash" / "s1" / "r1" / "results" / "result_0003_00.png",
        tmp_path / "trash" / "s1" / "r1" / "thumbs" / "result_0003_00.png",
    ]
    assert not store.result_file(3, 0).exists()
    assert not store.thumb_file(3, 0).exists()
    assert destinations[0].read_bytes() == b"png-bytes"


def test_discard_skips_missing_sources(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s1")
    assert trash.discard("r1", ["results/never-written.png"]) == []


def test_restore_moves_files_back(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s1")
    record = make_record(experiment=3)
    store = seed_run(tmp_path, "r1", [record])
    trash.discard("r1", [record.path, record.thumb])

    restored = trash.restore("r1", [record.path, record.thumb])

    assert restored == [store.result_file(3, 0), store.thumb_file(3, 0)]
    assert store.result_file(3, 0).read_bytes() == b"png-bytes"
    assert not (trash.session_dir / "r1" / record.path).exists()


def test_restore_skips_missing_trash_files(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s1")
    assert trash.restore("r1", ["results/never-trashed.png"]) == []


def test_cleanup_other_sessions_keeps_current(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s2")
    (trash.trash_root / "s1" / "r1").mkdir(parents=True)
    (trash.trash_root / "s2" / "r1").mkdir(parents=True)
    (trash.trash_root / "s3").mkdir(parents=True)

    trash.cleanup_other_sessions()

    assert (trash.trash_root / "s2").is_dir()
    assert not (trash.trash_root / "s1").exists()
    assert not (trash.trash_root / "s3").exists()


def test_cleanup_other_sessions_without_trash_root(tmp_path: Path) -> None:
    SessionTrash(tmp_path, session_id="s1").cleanup_other_sessions()
    assert not (tmp_path / "trash").exists()


def test_cleanup_current_removes_session_dir(tmp_path: Path) -> None:
    trash = prepare_trash(tmp_path, session_id="s1")
    (trash.session_dir / "r1").mkdir()
    trash.cleanup_current()
    assert not trash.session_dir.exists()
    assert trash.trash_root.is_dir()


def test_prepare_trash_creates_session_and_prunes_others(tmp_path: Path) -> None:
    stale = tmp_path / "trash" / "old-session"
    stale.mkdir(parents=True)
    trash = prepare_trash(tmp_path, session_id="s1")
    assert trash.session_dir.is_dir()
    assert not stale.exists()


def test_discard_result_moves_first_then_records(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s1")
    record = make_record(experiment=3)
    store = seed_run(tmp_path, "r1", [record])

    discard_result(store, trash, record)

    assert store.read_curation() == {"0003:00": "discarded"}
    assert not store.result_file(3, 0).exists()
    assert (trash.session_dir / "r1" / record.path).is_file()


def test_restore_result_clears_state(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s1")
    record = make_record(experiment=3)
    store = seed_run(tmp_path, "r1", [record])
    discard_result(store, trash, record)

    restore_result(store, trash, record)

    assert store.read_curation() == {}
    assert store.result_file(3, 0).is_file()


def test_list_runs_newest_first_and_ignores_trash(tmp_path: Path) -> None:
    seed_run(tmp_path, "20261111-100000-aaaaaa", [make_record(experiment=0)])
    seed_run(tmp_path, "20261111-090000-bbbbbb", [make_record(experiment=0)])
    (tmp_path / "trash" / "s1").mkdir(parents=True)
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / "loose.txt").write_text("x", encoding="utf-8")

    runs = list_runs(tmp_path)
    assert [run.run_id for run in runs] == ["20261111-100000-aaaaaa", "20261111-090000-bbbbbb"]
    assert all(run.status == "running" for run in runs)


def test_list_runs_ignores_foreign_manifest(tmp_path: Path) -> None:
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    write_json_atomic(stranger / "manifest.json", {"format": "something.else"})

    assert list_runs(tmp_path) == ()


def test_list_runs_empty_root(tmp_path: Path) -> None:
    assert list_runs(tmp_path) == ()
    assert list_runs(tmp_path / "missing") == ()


def test_load_run_returns_manifest_results_and_curation(tmp_path: Path) -> None:
    record = make_record(experiment=3)
    store = seed_run(tmp_path, "r1", [record])
    store.set_curation(record.id, "accepted")

    run_data = load_run(tmp_path, "r1")

    assert run_data.header["run_id"] == "r1"
    assert run_data.results == (record,)
    assert run_data.curation == {record.id: "accepted"}


def test_accepted_index_refresh_and_events(tmp_path: Path) -> None:
    accepted = make_record(experiment=0)
    discarded = make_record(experiment=1)
    pending = make_record(experiment=2)
    store = seed_run(tmp_path, "r1", [accepted, discarded, pending])
    store.set_curation(accepted.id, "accepted")
    store.set_curation(discarded.id, "discarded")

    index = AcceptedIndex(tmp_path)
    index.refresh()

    assert [item.record.id for item in index.results()] == [accepted.id]

    # A discard event removes it from the index without a rescan.
    index.apply("r1", accepted.id, "discarded")
    assert index.results() == ()

    # An accept event adds it back.
    index.apply("r1", accepted.id, "accepted")
    assert [item.record.id for item in index.results()] == [accepted.id]
    assert index.results()[0].run_id == "r1"

    # Clearing to pending removes it again.
    index.apply("r1", accepted.id, None)
    assert index.results() == ()

    # A refresh reconciles with disk.
    index.refresh()
    assert [item.record.id for item in index.results()] == [accepted.id]


def test_accepted_index_apply_unknown_run_is_ignored_until_refresh(tmp_path: Path) -> None:
    index = AcceptedIndex(tmp_path)
    index.apply("missing-run", "0000:00", "accepted")
    assert index.results() == ()
    assert not (tmp_path / "missing-run").exists()


def test_accepted_index_accept_unknown_id_does_not_leave_empty_state(
    tmp_path: Path,
) -> None:
    seed_run(tmp_path, "r1", [make_record(experiment=0)])
    index = AcceptedIndex(tmp_path)
    index.refresh()
    index.apply("r1", "0099:09", "accepted")
    assert index.results() == ()


def test_accepted_index_orders_runs_newest_first(tmp_path: Path) -> None:
    older = make_record(experiment=0)
    newer = make_record(experiment=0)
    seed_run(tmp_path, "20261111-090000-aaaaaa", [older]).set_curation(older.id, "accepted")
    seed_run(tmp_path, "20261111-100000-bbbbbb", [newer]).set_curation(newer.id, "accepted")

    index = AcceptedIndex(tmp_path)
    index.refresh()

    assert [item.run_id for item in index.results()] == [
        "20261111-100000-bbbbbb",
        "20261111-090000-aaaaaa",
    ]


def test_discard_without_trash_file_still_records(tmp_path: Path) -> None:
    # A result whose files were never written (worker crashed after the
    # record) can still be discarded: the move skips, the state is recorded.
    trash = SessionTrash(tmp_path, session_id="s1")
    record = make_record(experiment=3)
    store = RunStore(tmp_path, "r1")
    store.create({})
    store.append_result(record)

    discard_result(store, trash, record)

    assert store.read_curation() == {record.id: "discarded"}


def test_trash_survives_across_session_objects(tmp_path: Path) -> None:
    record = make_record(experiment=3)
    seed_run(tmp_path, "r1", [record])
    SessionTrash(tmp_path, session_id="s1").discard("r1", [record.path])

    restored = SessionTrash(tmp_path, session_id="s1").restore("r1", [record.path])

    assert restored == [tmp_path / "r1" / record.path]


def test_discard_rejects_paths_outside_the_run(tmp_path: Path) -> None:
    trash = SessionTrash(tmp_path, session_id="s1")
    (tmp_path / "r1" / "a").mkdir(parents=True)
    escape = tmp_path / "r1" / "a" / ".." / "escape.png"
    escape.write_bytes(b"x")

    for relative in ("../escape.png", "/etc/passwd", ""):
        with pytest.raises(ValueError, match="relative to the run directory"):
            trash.discard("r1", [relative])
    with pytest.raises(ValueError, match="'/' separators"):
        trash.discard("r1", ["..\\escape.png"])
    with pytest.raises(ValueError, match="relative to the run directory"):
        trash.restore("r1", ["../../escape.png"])

    # Nothing was moved out of the run.
    assert escape.is_file()
