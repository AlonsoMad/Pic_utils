---
phase: 3
name: Block System & Catalog
status: passed
tag: phase-3
date: 2026-11-11
---

## Tasks merged

- P2-01
- P2-02
- P2-03
- P2-04
- P2-05
- P2-06
- P2-07

## Gate commands and results

Run on `main` after the seven task commits:

```
$ uv sync --extra dev
Resolved 44 packages in 1ms
Checked 43 packages in 0.77ms

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
39 files already formatted

$ uv run mypy src
Success: no issues found in 23 source files

$ uv run pytest
260 passed in 7.89s

$ uv run pytest tests/blocks
125 passed in 7.07s
```

Registered built-ins (9):

```
brightness_mask, channel_mask, channel_permute, color_range_mask,
conditional_overlay, double_exposure, overlay, permutation_grid, segment
```

## Acceptance checklist

- [x] The decorator introspects `int`, `float`, `Choice`, `bool`, `Color`,
      `Seed`, `Image`, and `Mask` (including `Mask | None`).
- [x] Invalid signatures are rejected with `BlockLoadError`: missing
      default, `*args`/`**kwargs`, positional-only, missing annotation,
      bare `int`/`str`, image default, optional mask with a non-None
      default, `int | None` without `Seed`, mismatched metadata, choice
      type mismatch, out-of-range default, invalid color/seed default, and
      output/return-annotation mismatch.
- [x] The registry loads built-ins and user blocks; reload diff is correct;
      a syntax error keeps that file's last-good blocks; helper modules
      reload; the snapshot hash is stable for unchanged sources and changes
      when a file changes.
- [x] All nine blocks exist with the architecture §5.7 signatures.
- [x] Every block passes conformance: metadata, defaults in range, output
      type, and deterministic output with a fixed seed.
- [x] `segment` emits at most 16 layers, guards `modulus == 0`, and computes
      superpixel means with `np.bincount` (no per-superpixel Python loop).
- [x] No block imports `io/`, `ui/`, `runner/`, or `canvas/`; no `print`,
      no file I/O, no Pillow/matplotlib, no global RNG.
- [x] `pytest tests/blocks` green (125 passed); mypy clean.
- [x] `specs/status/phase-3.md` written; tag `phase-3` created.

## Known issues / follow-ups

- The `QFileSystemWatcher` auto-reload described in architecture §5.5 is a
  UI concern and is deferred to the Lab shell phase. `BlockRegistry.reload()`
  is the manual entry point and emits `changed`.
- `BlockLoadError` lives in `blocks/api.py` because §18.1 does not list it;
  it subclasses `PicUtilsError`.
- `permutation_grid` caps at 8 layers (the §5.7 table says 6-9, the
  implementation note says at most 8); `channel_order` is 6 or 7 with
  `include_original`, `invert` is 8.
- `segment` clusters superpixel **means** when superpixels are enabled
  (at most `superpixel_count` samples), not every pixel; this preserves the
  legacy mean-image behavior and the `np.bincount` rule.
- Soft-mask semantics are defined in P2-04: the mask carries the underlying
  channel/luminance/mean intensity inside the selection and 0 outside.
- `overlay` defaults to `white_to_transparent=True` and
  `resize_to_match=True`; `conditional_overlay` uses centered pad/crop for a
  smaller second image.
- Task branches were not used: workspace policy blocks `git switch`. Each
  task was committed on `main` after its own gate run, matching the P0-03
  and Phase 2 precedent.
