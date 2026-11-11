---
phase: 6
name: Lab UI & Preview
status: in-progress
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

Commits: f68e0d2…(phase-6 series), all committed directly on `main`
(workspace policy blocks `git switch`), matching Phases 1-5.

## Gate results (local)

```
$ uv run ruff check .
All checks passed!
$ uv run ruff format --check .
108 files already formatted
$ uv run mypy src
Success: no issues found in 69 source files
$ QT_QPA_PLATFORM=offscreen uv run pytest
(see final gate run below)
```

## Acceptance checklist

- [x] The app launches with `uv run pic-utils` (offscreen smoke: window
      shows, clean exit; Lab/Canvas stack switches).
- [x] Language switch EN↔ES updates all visible strings without restart;
      catalogs regenerated with `pyside6-lupdate` and compiled with
      `pyside6-lrelease` (72/72 strings finished); `retranslate()` protocol
      on every widget class.
- [x] Settings persist across restarts (INI at `AppConfigLocation`);
      paths created on demand; the blocks folder ships an example block.
- [x] Block library: search, categories, examples, manual reload,
      auto-reload (debounced 500 ms), non-modal deduplicated errors.
- [x] Pipeline editor: add/remove/reorder (drag + buttons)/bypass;
      validation issues shown inline and per-row color; undo/redo merges
      parameter drags into one step; loading a project clears the stack.
- [x] Parameter panel: every param kind (int/float/choice/bool/color/seed);
      mask binding combo of upstream mask steps; seed auto + randomize;
      vary controls (fixed/sweep/sample) edit the RunConfig.
- [x] Preview: debounced, latest-wins, proxy resolution, readable errors,
      restarts after block reload and on crash; the GUI thread never waits.
- [x] Results grid: lazy thumbnails (QThreadPool); accept/discard/restore
      with trash-first ordering; A/D/R shortcuts; A/B compare; full-res
      viewer; run selector; "Show discarded" toggle.
- [x] Run controls: sources (add/remove/checkboxes), second image, mode,
      batch size, seed, plan count, launch/cancel, summary, memory display.
- [x] Project save/load with relink; presets; four shipped examples load
      and validate against the built-in block snapshot.
- [x] End-to-end Lab journey on a synthetic image: load → pipeline →
      preview → run → curate (accept, discard, restore).

## Known issues / follow-ups

- `test_worker_count_is_clamped_by_the_budget` timed out once when two full
  suite runs raced; it passes in isolation (~6 s). Resource contention, not
  a regression.
- `Settings` writes `settings.ini` at `AppConfigLocation`; the smoke test
  passes an explicit path — production behavior unchanged.
