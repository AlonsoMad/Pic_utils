---
phase: 2
name: Core Image Model & I/O
status: passed
tag: phase-2
date: 2026-11-11
---

## Tasks merged

- P1-01
- P1-02
- P1-03
- P1-04

## Gate commands and results

Run on `main` after the four task commits:

```
$ uv sync --extra dev
Resolved 44 packages in 2ms
Checked 43 packages in 0.95ms

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
23 files already formatted

$ uv run mypy src
Success: no issues found in 14 source files

$ uv run pytest
135 passed in 0.52s
```

Boundary checks:

```
$ grep -R -E "COLOR_BGR|BGR2RGB|RGB2BGR" src/pic_utils/core src/pic_utils/io --include='*.py' | grep -v 'core/color.py'
(no matches)

$ grep -R "import cv2" src/pic_utils/io/
(no matches)

$ grep -R -E "^(from PIL|import PIL)" src/pic_utils/core/
(no matches)
```

## Acceptance checklist

- [x] `Image` rejects wrong dtype/shape and normalizes non-contiguous
      input to C-contiguous uint8; both planes are marked read-only.
- [x] `Image` alpha shape matches `(H, W)`; `to_rgba()` fills alpha 255
      when absent.
- [x] Alpha propagation primitives: `with_alpha` reattaches an alpha
      plane, `resized`/`to_proxy` resize it with the image, and a
      geometry-changing resize returns its own alpha. The executor-level
      reattachment rule is enforced by P3-03 (pipeline executor).
- [x] `to_proxy(max_long_side)` preserves aspect ratio and resizes alpha.
- [x] Load PNG/JPEG/TIFF; grayscale, RGB, RGBA, palette, and CMYK.
- [x] EXIF orientation applied on load; the orientation tag is not
      re-emitted on save.
- [x] 16-bit input raises `UnsupportedImageError` with `image.16bit` in
      the reason; > 48 MP raises `ImageTooLargeError` with dimensions;
      corrupt files raise `ImageLoadError`.
- [x] Save PNG/JPEG/TIFF; JPEG flattens alpha onto white; TIFF LZW;
      sRGB ICC embedded (asserted via `img.info["icc_profile"]`).
- [x] A failed save leaves no partial file.
- [x] No BGR outside `core/color.py` in the new `core/` + `io/` code; no
      Pillow import in `core/`.
- [x] `pytest tests/core tests/io` green; mypy clean.
- [x] `specs/status/phase-2.md` written; tag `phase-2` created.

## Known issues / follow-ups

- The quarantined `src/pic_utils/legacy.py` still contains OpenCV BGR
  calls. It is the P0-02 behavioral reference, not part of the new
  package surface, and disappears with the Phase 3 block
  reimplementation.
- `core/__init__.py` does not re-export the new image/color/stats types
  because P1-01/P1-02 do not own that file. Imports use
  `pic_utils.core.image`, `pic_utils.core.color`, and
  `pic_utils.core.stats` (the form used by the architecture's block
  examples).
- 16-bit rejection uses `UnsupportedImageError` with the `image.16bit`
  code in its reason: `image.16bit` is not one of the §18.2 runtime codes,
  so no new code constant was added.
- PNG saves use `compress_level=3` (the results path). The export path
  (P6-03) can revisit the level if export size needs it.
- Task branches were not used: workspace policy blocks `git switch`.
  Each task was committed on `main` after its own gate run, matching the
  P0-03 precedent.
