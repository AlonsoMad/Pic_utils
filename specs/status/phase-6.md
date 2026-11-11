---
phase: 6
name: Lab UI & Preview
status: passed
tag: phase-6
date: 2026-11-11
---

## Tasks merged

- P5-01 (app shell: settings, i18n, theme, main window, composition roots)
- P5-02 (block library: search, categories, reload, auto-reload)
- P5-03 (pipeline editor: undo commands, inline validation, drag reorder)
- P5-04 (parameter panel: per-kind widgets, mask binding, vary controls)
- P5-05 (preview process + pane: debounce, latest-wins, restart on reload)
- P5-06 (results grid: lazy thumbnails, curation, viewer, A/B compare)
- P5-07 (run controls: sources, config, launch/cancel, summary, memory)
- P5-08 (project save/load, relink dialog, four shipped presets)
- P5-10 (lab page composition + signal wiring)
- P5-09 (end-to-end Lab journey tests)

Commits: f68e0d2, a9f9891, 6f5544c, 79283ae, ea17c48, 07270de, aff2435,
3252537, a3f0b15, d749a16, a3f0b15 (plus this status file). All committed
directly on `main` (workspace policy blocks `git switch`), matching
Phases 1-5.

## Gate commands and results

```
$ uv sync --extra dev
Resolved 61 packages in ...

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
108 files already formatted

$ uv run mypy src
Success: no issues found in 69 source files

$ QT_QPA_PLATFORM=offscreen uv run pytest
649 passed in 130.95s (0:02:10)
```

## Acceptance checklist

- [x] The app launches with `uv run pic-utils`; Lab/Canvas mode switch works
      (offscreen smoke: window shows, clean exit).
- [x] Language switch EN↔ES updates all visible strings without restart;
      no hardcoded user-facing strings remain (review); catalogs committed.
- [x] Settings persist across restarts; paths are created on demand.
- [x] Block library: search, categories, examples, manual reload,
      auto-reload, non-modal errors.
- [x] Pipeline editor: add/remove/reorder/bypass; validation issues shown
      inline; undo/redo works.
- [x] Parameter panel: every param kind; mask binding combo; seed auto;
      fixed/sweep/sample controls; undo merges a drag into one step.
- [x] Preview: debounced, latest-wins, proxy resolution, readable errors,
      restarts after reload/crash.
- [x] Results grid: lazy thumbnails; accept/discard/restore; keyboard
      shortcuts; A/B compare; full-res viewer; seed/params details.
- [x] Run controls: sources, second image, mode, batch size, seed, plan
      count, launch/cancel, summary, memory display.
- [x] Project save/load with relink; presets; four shipped examples load.
- [x] End-to-end Lab journey on a synthetic image: load → pipeline → preview
      → run → curate.
- [x] `QT_QPA_PLATFORM=offscreen uv run pytest tests/ui
      tests/runner/test_preview.py -q` green (165 passed).
- [x] `specs/status/phase-6.md` written.

## Known issues / follow-ups

- `test_worker_count_is_clamped_by_the_budget` timed out once when two full
  suite runs raced; it passes in isolation (~6 s). Resource contention, not
  a regression.
- `QImage.fromData(data, "PNG")` is rejected by PySide6 6.12's overload
  check; the pane relies on format auto-detection.
- Drag-and-drop of project/canvas files onto the window is wired
  (`sources_dropped`/`project_open_requested`) but canvas opening is a
  Phase 7 concern; the signal is in place.

## Fixes during the phase

- BUG-1: `QImage.fromData(data, "PNG")` is rejected by PySide6 6.12's
  overload check even for valid PNG bytes; the pane relies on format
  auto-detection instead.
- The preview pane tests now shut the client down at teardown — leaked
  processes and reader threads interfered with later tests.

