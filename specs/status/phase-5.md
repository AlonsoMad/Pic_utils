---
phase: 5
name: Runner & Results Backend
status: passed
tag: phase-5
date: 2026-11-11
---

## Tasks merged

- P4-01 (run store, manifest, curation, session trash)
- P4-02 (worker protocol, shared-memory image store)
- P4-03 (spawn-safe worker, snapshot verification, result writing)
- P4-04 (supervised pool, memory guard)
- P4-05 (run service with memory-guarded dispatch)
- P4-06 (runner integration tests)

Commits: 92e107b, 926bf41, 6d1de6f, 9f9882d, 51b5dfc, 5f73a5b, c43ecca
(plus status-file commits). All committed directly on `main` (workspace
policy blocks `git switch`), matching Phases 1-4.

## Gate commands and results

```
$ uv sync --extra dev
Resolved 61 packages in ...
$ uv run ruff check .
All checks passed!
$ uv run ruff format --check .
67 files already formatted
$ uv run mypy src
Success: no issues found in 39 source files
$ uv run pytest
483 passed in 99.96s (0:01:39)
```

CI: run 42998336536 — success on ubuntu-latest and windows-latest
(https://github.com/AlonsoMad/Pic_utils/actions/runs/42998336536).

## Acceptance checklist

- [x] Run folder layout matches architecture §9.1 (`manifest.json`,
      `results.jsonl`, `curation.json`, `results/`, `thumbs/`, `blocks/`,
      `run.log`).
- [x] Manifest and curation writes are atomic (`core.jsonio.write_json_atomic`);
      `results.jsonl` is append-only and flushed per record; a crash leaves
      no partial JSON (torn-final-line tolerance tested).
- [x] Shared memory round-trips RGB and alpha; names are unique; handles
      are released; no leaks (POSIX `/dev/shm` enumeration + spawn attach).
- [x] Worker writes one result per layer, a 256 px PNG thumbnail, and
      returns a small outcome; no image data crosses the process boundary
      (pickled-outcome size asserted).
- [x] Pool: spawn context; crash detection (futures-based attribution);
      in-flight experiments marked `worker.crashed`; rebuild (max 3, then
      `pool.unstable`); timeout watchdog (`block.timeout`); pool rebuild on
      RSS growth; graceful cancel; force kill.
- [x] RunService: one run at a time; `run_started` after preparation;
      progress/result/finish/cancel signals; source load failures skipped;
      memory guard defers/fails.
- [x] Integration: cancel keeps completed results; a raising block does
      not abort the batch; invalid output fails that experiment; worker
      crash recovers; timeout kills; manifest/curation/trash are consistent.
- [x] `pytest tests/runner tests/results` green on Linux (81 + 22 tests);
      Windows CI green.
- [x] `specs/status/phase-5.md` written; tag `phase-5`.

## Known issues / follow-ups

All follow-ups below were resolved after the phase tag with the owner's
explicit approval (CC), committed on `main` after `phase-5`:

- **CC (resolved):** `BlockSnapshot` and the snapshot-hash payload moved
  from `blocks/registry.py` (Qt) to `blocks/snapshot.py` (Qt-free); the
  worker uses the real class and the shared `snapshot_hash()` — the
  structural mirror and the duplicated hash algorithm are gone.
- **BUG-1 / CC-ERR (resolved):** the exception dataclasses in
  `core/errors.py` are no longer frozen (a frozen dataclass masked typed
  exceptions crossing `@contextmanager` boundaries via
  `FrozenInstanceError`). Regression-tested.
- **block.too_many_outputs (resolved):** the executor raises
  `TooManyOutputsError` for >16 layers and the worker maps it to
  `block.too_many_outputs` (§18.2) instead of the generic
  `block.invalid_output`.
- **psutil stubs (kept):** `psutil` ships no `py.typed` and `types-psutil`
  remains an unapproved dependency; the two documented
  `# type: ignore[import-untyped]` lines stay.
- **`RunnerSettings` (unchanged):** the Qt settings wrapper arrives with
  Phase 6 (P5-01); `RunnerSettings` remains the runner-facing slice.
