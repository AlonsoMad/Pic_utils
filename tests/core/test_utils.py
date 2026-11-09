"""Behavior tests for the core utilities and logging setup (P0-03).

Covers the atomic-write guarantee (NFR-REL-03), the ``jsonable()`` numpy
conversion, the full error hierarchy and §18.2 code table, the pure path
helpers, and the logging setup idempotency rules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import logging.handlers
import os
from pathlib import Path
from types import SimpleNamespace

import fixtures
import numpy as np
import pytest

from pic_utils.core import (
    BLOCK_INVALID_OUTPUT,
    BLOCK_RAISED,
    BLOCK_SNAPSHOT_MISMATCH,
    BLOCK_TIMEOUT,
    BLOCK_TOO_MANY_OUTPUTS,
    CANCELLED,
    IO_READ,
    IO_WRITE,
    MEMORY_LIMIT,
    POOL_UNSTABLE,
    SHM_UNAVAILABLE,
    WORKER_CRASHED,
    BlockError,
    CanvasFileError,
    ExportError,
    ImageLoadError,
    ImageTooLargeError,
    InvalidBlockOutput,
    MemoryLimitError,
    PicUtilsError,
    ProjectFileError,
    RunError,
    UnsupportedImageError,
    ValidationError,
    content_hash,
    content_hash_short,
    ensure_dir,
    jsonable,
    read_json,
    safe_name,
    write_json_atomic,
)
from pic_utils.logging_setup import run_logger, setup_logging

# --- Errors (architecture §18.1, §18.2) -------------------------------------

ALL_ERRORS = [
    ValidationError(issues=[]),
    BlockError(block="segment"),
    InvalidBlockOutput(block="overlay", step_id="step-2"),
    ImageLoadError(path=Path("missing.png"), reason="no such file"),
    ImageTooLargeError(path=Path("huge.png"), reason="too large", width=9000, height=9000),
    UnsupportedImageError(path=Path("weird.png"), reason="16-bit samples"),
    MemoryLimitError("experiment needs 9.1 GB"),
    ExportError("target is not writable"),
    CanvasFileError("unsupported canvas version"),
    ProjectFileError("corrupt project file"),
    RunError(code=WORKER_CRASHED),
]


@pytest.mark.parametrize("error", ALL_ERRORS, ids=[type(e).__name__ for e in ALL_ERRORS])
def test_every_error_is_catchable_as_pic_utils_error(error: Exception) -> None:
    with pytest.raises(PicUtilsError):
        raise error


def test_error_hierarchy_relations() -> None:
    assert issubclass(PicUtilsError, Exception)
    assert issubclass(InvalidBlockOutput, BlockError)
    assert issubclass(ImageTooLargeError, ImageLoadError)
    assert issubclass(UnsupportedImageError, ImageLoadError)
    for cls in (type(e) for e in ALL_ERRORS):
        assert issubclass(cls, PicUtilsError)


def test_errors_are_frozen_dataclasses_where_attributes_are_named() -> None:
    error = BlockError(block="segment", step_id="step-2")
    assert error.block == "segment"
    assert error.step_id == "step-2"
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.block = "other"  # type: ignore[misc]


def test_image_load_error_attributes() -> None:
    error = ImageLoadError(path="missing.png", reason="no such file")
    assert error.path == Path("missing.png")
    assert error.reason == "no such file"
    assert "missing.png" in str(error)
    assert "no such file" in str(error)


def test_image_too_large_error_attributes() -> None:
    error = ImageTooLargeError(path=Path("huge.png"), reason="too large", width=9000, height=6000)
    assert isinstance(error, ImageLoadError)
    assert error.width == 9000
    assert error.height == 6000
    assert "9000x6000" in str(error)
    assert "huge.png" in str(error)


def test_invalid_block_output_message() -> None:
    error = InvalidBlockOutput(block="overlay", step_id="step-3")
    assert error.block == "overlay"
    assert error.step_id == "step-3"
    assert "overlay" in str(error)
    assert "invalid output" in str(error)
    assert "step-3" in str(error)


def test_validation_error_carries_issues() -> None:
    issue = SimpleNamespace(
        severity="error",
        code="param.missing",
        message="missing value for strength",
        step_id="step-1",
        param="strength",
    )
    error = ValidationError(issues=[issue])
    assert list(error.issues) == [issue]
    assert "missing value for strength" in str(error)
    assert "validation failed" in str(ValidationError(issues=[]))


def test_run_error_carries_code() -> None:
    error = RunError(code=WORKER_CRASHED)
    assert error.code == "worker.crashed"
    assert "worker.crashed" in str(error)


def test_runtime_error_codes_match_architecture_18_2() -> None:
    assert BLOCK_RAISED == "block.raised"
    assert BLOCK_INVALID_OUTPUT == "block.invalid_output"
    assert BLOCK_SNAPSHOT_MISMATCH == "block.snapshot_mismatch"
    assert BLOCK_TIMEOUT == "block.timeout"
    assert BLOCK_TOO_MANY_OUTPUTS == "block.too_many_outputs"
    assert MEMORY_LIMIT == "memory.limit"
    assert WORKER_CRASHED == "worker.crashed"
    assert POOL_UNSTABLE == "pool.unstable"
    assert SHM_UNAVAILABLE == "shm.unavailable"
    assert IO_READ == "io.read"
    assert IO_WRITE == "io.write"
    assert CANCELLED == "cancelled"


# --- jsonable / atomic JSON I/O ----------------------------------------------


def test_jsonable_converts_numpy_scalars() -> None:
    assert jsonable(np.int64(7)) == 7
    assert type(jsonable(np.int64(7))) is int
    assert jsonable(np.float32(1.5)) == 1.5
    assert type(jsonable(np.float32(1.5))) is float
    assert jsonable(np.bool_(True)) is True
    assert type(jsonable(np.bool_(False))) is bool
    assert jsonable(np.str_("text")) == "text"


def test_jsonable_preserves_float32_values() -> None:
    # Conversion must keep the float32 value, not round it to a decimal literal.
    assert jsonable(np.float32(0.1)) == 0.10000000149011612
    assert jsonable(np.array([0.5, 2.25], dtype=np.float16)) == [0.5, 2.25]


def test_jsonable_converts_arrays_to_nested_lists() -> None:
    converted = jsonable(fixtures.rgb_gradient())
    assert isinstance(converted, list)
    assert converted[0][0] == [0, 0, 0]
    assert converted[7][7] == [255, 255, 255]
    assert jsonable(np.arange(4, dtype=np.int32)) == [0, 1, 2, 3]


def test_jsonable_walks_nested_containers() -> None:
    value = {
        "scalar": np.float64(2.5),
        "tuple": (np.int32(1), np.bool_(True), "text"),
        "nested": {"array": np.arange(3, dtype=np.int64), "plain": [1, "x"]},
    }
    assert jsonable(value) == {
        "scalar": 2.5,
        "tuple": [1, True, "text"],
        "nested": {"array": [0, 1, 2], "plain": [1, "x"]},
    }


def test_jsonable_converts_numpy_dict_keys() -> None:
    assert jsonable({np.int64(3): "x", np.str_("k"): np.float64(0.5)}) == {3: "x", "k": 0.5}


def test_jsonable_passes_json_safe_values_through() -> None:
    for value in (None, True, False, 3, 2.5, "text"):
        assert jsonable(value) == value


def test_jsonable_rejects_non_convertible_objects() -> None:
    with pytest.raises(TypeError, match="not JSON-serializable"):
        jsonable(object())
    with pytest.raises(TypeError, match="not JSON-serializable"):
        jsonable({"nested": {1, 2}})
    with pytest.raises(TypeError, match="not JSON-serializable"):
        jsonable(b"bytes")
    with pytest.raises(TypeError, match="keys must be"):
        jsonable({("tuple",): "x"})


def test_write_json_atomic_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text('{"old": true}\n', encoding="utf-8")
    write_json_atomic(target, {"new": [1, 2], "value": np.float32(1.5)})
    assert read_json(target) == {"new": [1, 2], "value": 1.5}
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": [1, 2], "value": 1.5}


def test_failed_serialization_leaves_original_intact_and_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text('{"original": true}\n', encoding="utf-8")
    with pytest.raises(TypeError, match="not JSON-serializable"):
        write_json_atomic(target, {"bad": object()})
    assert target.read_text(encoding="utf-8") == '{"original": true}\n'
    assert [entry.name for entry in tmp_path.iterdir()] == ["data.json"]


def test_failed_replace_leaves_original_intact_and_no_temp_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.json"
    target.write_text("keep\n", encoding="utf-8")

    def failing_replace(src: object, dst: object, **kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_json_atomic(target, {"a": 1})
    assert target.read_text(encoding="utf-8") == "keep\n"
    assert [entry.name for entry in tmp_path.iterdir()] == ["data.json"]


def test_read_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "nope.json")


def test_read_json_corrupt_file(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON") as info:
        read_json(target)
    assert "bad.json" in str(info.value)


def test_json_round_trip_preserves_unicode_and_numpy(tmp_path: Path) -> None:
    target = tmp_path / "round.json"
    data = {"name": "café", "vals": [np.int64(1), np.float64(2.5)]}
    write_json_atomic(target, data)
    assert read_json(target) == {"name": "café", "vals": [1, 2.5]}


# --- Paths --------------------------------------------------------------------


def test_content_hash_matches_hashlib_and_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 8192  # 2 MiB: larger than one streaming chunk
    target.write_bytes(payload)
    first = content_hash(target)
    assert first == content_hash(target)
    assert first == hashlib.sha256(payload).hexdigest()


def test_content_hash_matches_for_identical_copies_changes_with_content(tmp_path: Path) -> None:
    original = tmp_path / "original.bin"
    copy = tmp_path / "copy.bin"
    changed = tmp_path / "changed.bin"
    original.write_bytes(b"same bytes")
    copy.write_bytes(b"same bytes")
    changed.write_bytes(b"other bytes")
    assert content_hash(original) == content_hash(copy)
    assert content_hash(original) != content_hash(changed)


def test_content_hash_short_is_truncated_full_hash(tmp_path: Path) -> None:
    target = tmp_path / "asset.bin"
    target.write_bytes(b"asset bytes")
    full = content_hash(target)
    assert len(content_hash_short(target)) == 16
    assert content_hash_short(target) == full[:16]
    assert content_hash_short(target, length=8) == full[:8]


def test_safe_name_is_deterministic_and_replaces_unsafe_characters() -> None:
    raw = 'a/b\\c:d*e?f"g<h>i|j.png'
    assert safe_name(raw) == safe_name(raw)
    result = safe_name(raw)
    for character in '/\\:*?"<>|':
        assert character not in result
    assert result == "a_b_c_d_e_f_g_h_i_j.png"


def test_safe_name_collapses_whitespace_and_keeps_ascii() -> None:
    assert safe_name("my  photo \t final") == "my_photo_final"
    assert safe_name("café") == "cafe"
    assert safe_name("plain-name_2.png") == "plain-name_2.png"


def test_safe_name_falls_back_when_empty() -> None:
    for raw in ("", "   ", "..", "日本語"):
        assert safe_name(raw) == "untitled"


def test_safe_name_bounds_length() -> None:
    assert len(safe_name("x" * 500)) == 80
    assert safe_name("x" * 500, max_length=10) == "x" * 10
    assert safe_name("a b c d e f g h i j k", max_length=10) == "a_b_c_d_e"


def test_ensure_dir_creates_parents_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert ensure_dir(target) == target
    assert target.is_dir()
    assert ensure_dir(target) == target
    assert ensure_dir(str(target)) == target


def test_ensure_dir_rejects_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "occupied.txt"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_dir(target)


# --- Logging setup (architecture §16, §18.3) ----------------------------------


@pytest.fixture
def root_logger_restored() -> logging.Logger:
    """Snapshot the root logger; remove/close handlers that tests installed."""
    root = logging.getLogger()
    original_level = root.level
    original_handlers = list(root.handlers)
    yield root
    for handler in list(root.handlers):
        if handler not in original_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(original_level)


def test_setup_logging_writes_rotating_file(
    tmp_path: Path, root_logger_restored: logging.Logger
) -> None:
    setup_logging(logging.DEBUG, tmp_path)
    logging.getLogger("test.app").debug("hello app log")
    log_file = tmp_path / "app.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello app log" in content
    assert "test.app" in content

    rotating = [
        handler
        for handler in root_logger_restored.handlers
        if isinstance(handler, logging.handlers.RotatingFileHandler)
    ]
    assert len(rotating) == 1
    assert rotating[0].maxBytes == 5 * 1024 * 1024
    assert rotating[0].backupCount == 3


def test_setup_logging_installs_console_handler_at_same_level(
    tmp_path: Path, root_logger_restored: logging.Logger
) -> None:
    setup_logging(logging.INFO, tmp_path)
    consoles = [
        handler
        for handler in root_logger_restored.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        and handler.level == logging.INFO
    ]
    assert len(consoles) == 1


def test_setup_logging_respects_level(tmp_path: Path, root_logger_restored: logging.Logger) -> None:
    setup_logging(logging.WARNING, tmp_path)
    logging.getLogger("test.level").debug("debug record")
    logging.getLogger("test.level").warning("warning record")
    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "warning record" in content
    assert "debug record" not in content


def test_setup_logging_is_idempotent(tmp_path: Path, root_logger_restored: logging.Logger) -> None:
    setup_logging(logging.INFO, tmp_path)
    setup_logging(logging.INFO, tmp_path)
    rotating = [
        handler
        for handler in root_logger_restored.handlers
        if isinstance(handler, logging.handlers.RotatingFileHandler)
    ]
    assert len(rotating) == 1
    logging.getLogger("test.dup").info("once only")
    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert content.count("once only") == 1


def test_setup_logging_without_log_dir_installs_no_file_handler(
    root_logger_restored: logging.Logger,
) -> None:
    setup_logging(logging.INFO, None)
    rotating = [
        handler
        for handler in root_logger_restored.handlers
        if isinstance(handler, logging.handlers.RotatingFileHandler)
    ]
    assert rotating == []


def test_run_logger_writes_run_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"
    run_log = run_logger(run_dir)
    try:
        assert run_dir.is_dir()
        run_log.info("experiment 1 done")
        run_log.debug("debug detail")
        content = (run_dir / "run.log").read_text(encoding="utf-8")
        assert "experiment 1 done" in content
        assert "debug detail" in content
    finally:
        for handler in list(run_log.handlers):
            run_log.removeHandler(handler)
            handler.close()


def test_run_logger_reuses_the_same_logger(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-002"
    first = run_logger(run_dir)
    try:
        second = run_logger(run_dir)
        assert first is second
        first.info("single line")
        content = (run_dir / "run.log").read_text(encoding="utf-8")
        assert content.count("single line") == 1
    finally:
        for handler in list(first.handlers):
            first.removeHandler(handler)
            handler.close()
