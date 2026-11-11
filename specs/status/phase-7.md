---
phase: 7
name: Canvas & Export
status: passed
tag: phase-7
date: 2026-11-11
---

## Tasks merged

- P6-01 (canvas model, content-addressed assets, save/load)
- P6-02 (scene, transforms, six blend modes, masks, groups, undo commands)
- P6-03 (full-resolution export renderer, bounded layer cache)
- P6-04 (canvas UI: viewport, layer list, properties, asset tray, mask dialog)
- P6-05 (save/load/export actions, export dialog, threaded export)
- P6-06 (canvas integration tests)
- P6-07 (final canvas page composition and shell wiring)

Commits: `abd0629` (P6 implementation; this status file is the phase gate).

## Gate commands and results

```
$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
131 files already formatted

$ uv run mypy src
Success: no issues found in 83 source files

$ QT_QPA_PLATFORM=offscreen uv run pytest
765 passed in 145.03s (0:02:25)
```

## Acceptance checklist

- [x] `CanvasDoc`/`CanvasLayer`/`CanvasGroup` model and JSON schema match
      architecture §10.1/§10.8.
- [x] Assets are content-addressed; save adds only new files; the JSON write
      is the commit point; missing assets refuse to open with a list.
- [x] Scene: move/scale/rotate/flip; z-order; opacity; six blend modes;
      masks; groups; undo/redo.
- [x] Full-res renderer matches the screen at 1:1 (pixel test); group opacity
      is per child; memory stays bounded (one full-res layer at a time, LRU
      size 2).
- [x] Canvas UI: add from asset tray; transform handles; layer list;
      properties; mask dialog; crop/resize; groups; shortcuts.
- [x] Canvas save/reopen restores layers, transforms, masks, groups.
- [x] Export: canvas / canvas+accepted / everything; PNG/JPEG/TIFF; JPEG
      quality; sRGB ICC; collision-safe folder; atomic per file.
- [x] `pytest tests/canvas tests/ui/test_canvas*` green; mypy clean.
- [x] `specs/status/phase-7.md` written; tag `phase-7`.

## Known issues / follow-ups

- Save is synchronous: the content-addressed copy is bounded and the JSON
  commit is atomic, but the copy does not yet run on a worker thread with a
  progress indicator. Export already runs on a `QThread`.
- Groups are applied per child in the scene (visibility and opacity) rather
  than as `QGraphicsItemGroup` parents. This matches the measured Qt
  group-opacity behavior and the export renderer.
- Screen proxies are drawn at the layer's natural size; zooming beyond 100%
  upscales the proxy (§10.7 documented limitation).
- The canvas file suffix is `.pucanvas` (the shell constant from P5-08); the
  assets folder is `<file>.assets`.
- The i18n catalogs were not regenerated for the new Canvas strings; English
  fallback until `pyside6-lupdate`/`pyside6-lrelease` run.
- Canvas test modules use `test_canvas_*` basenames because pytest's prepend
  import mode rejects duplicate basenames across test directories
  (`tests/pipeline/test_model.py` and `tests/runner/test_integration.py`
  already exist).

## Fixes during the phase

- Scene items draw the display proxy scaled to the layer's natural size;
  using the proxy dimensions for the transform made large layers render
  smaller than the canvas.
- The undo stack's `indexChanged` connection uses a direct connection; a
  queued connection fired after the page's C++ object was deleted at
  teardown.
- The selected layer is preserved across model resets so property edits do
  not clear the selection.
