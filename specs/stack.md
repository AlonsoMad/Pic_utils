# Pic Utils — App Stack Specification

> **Status:** DECIDED — source of truth. Later agents MUST NOT relitigate these choices;
> propose changes only via a new decision record that supersedes this file.
> Next documents in the chain: Product Requirements (PRD) → Architecture.

| Decision | Value |
|---|---|
| Language / runtime | Python 3.12 (`>=3.12,<3.13`) |
| UI framework | Qt 6 via **PySide6** |
| Platforms | Linux (primary, dev machine) and Windows. macOS out of scope. |
| Environment / deps | **uv** with `pyproject.toml` + lockfile |
| Distribution model | Open-source; users install from source (uv/pip). No installers for v1. |

---

## 1. Runtime: Python 3.12

- Pin `requires-python = ">=3.12,<3.13"` in `pyproject.toml`.
- Rationale: mature compiled-wheel coverage for the whole dependency set
  (numpy, OpenCV, PySide6, scikit-learn) on both Linux and Windows; avoids
  the thinner ecosystem of 3.13/3.14 for compiled packages.
- Consequence for agents: code MUST be 3.12-compatible; no 3.13+-only syntax.

## 2. UI: PySide6 (Qt 6)

- Rationale:
  - `QGraphicsScene`/`QGraphicsItem` natively provide the core interaction
    model of the app: draggable, rotatable, z-ordered, selectable canvas items
    with handles — this maps 1:1 to the manual composition workflow.
  - Direct interop with numpy (`QImage` from array buffers) without a
    serialization boundary.
  - LGPLv3 licensing is compatible with an open-source art tool.
  - Stays in-process with the Python image core: no HTTP boundary, no
    second language.
- Consequences:
  - The app is a **desktop** application; there is no web frontend.
  - Qt widgets for panels (layer stack, parameters, variants); the image
    canvas is a `QGraphicsView` scene.
  - Preview rendering targets a proxy resolution (e.g. long side ≈ 1024 px);
    full-resolution processing happens only on export. Exact policy is an
    architecture concern, but the stack must support it (it does: all core
    image ops are resolution-independent numpy/cv2).
- **UI construction: 100% programmatic Python.** Widgets are instantiated in
  code; no Qt Designer `.ui` files and no QML. Rationale: keeps the whole UI
  reviewable/diffable in git, and agents can build and modify it without
  external design tools. Qt Widgets can express the entire planned surface:
  dockable panels, layer lists (model/view), parameter sliders, dialogs,
  custom-painted canvas items (`QPainter` inside `QGraphicsScene`).

## 3. Core image/ML libraries

| Package | Role |
|---|---|
| `numpy` | Canonical in-memory image representation; all pixel math |
| `opencv-python` | Color-space conversions, filtering, fast primitives |
| `Pillow` | Format I/O, some compositing primitives |
| `scikit-learn` | Clustering (e.g. MiniBatchKMeans segmentation) |
| `scikit-image` | Superpixels (SLIC) and region algorithms |

- `matplotlib` is a **development/extra dependency only**. Library and app
  code MUST NOT call `plt.show()` or any plotting at runtime; visualization
  inside the app is Qt's job.
- All five core packages are resolution- and platform-portable (Linux +
  Windows wheels exist); no compiled first-party extensions in v1.
- GPU acceleration (CUDA / OpenCV-CUDA / torch) is **not** part of the v1
  stack. It may be introduced later behind an interface if profiling demands
  it; nothing in the block/pipeline design may assume CPU-only or GPU-only.

## 4. Tooling and project layout

- **uv** for environments, dependency resolution, and the lockfile
  (`uv.lock` committed).
- `pyproject.toml` is the single metadata/dependency source. No `setup.py`,
  no `requirements.txt` (lockfile + `pyproject.toml` only).
- `src/` layout (`src/pic_utils/…`), importable package name `pic_utils`.
- Quality gates (configured in `pyproject.toml`, run via uv):
  - `ruff` — lint + format.
  - `pytest` — tests.
  - `mypy` — type checking (progressive; new code typed, legacy code
    migrated, never blocked wholesale).
- Python-level loops over pixels are banned in hot paths; per-pixel work
  belongs to numpy/cv2 vectorized ops. (Enforced by review, not tooling.)

**Dependency extension policy.** Adding libraries is a normal, expected
operation — the technique catalog is meant to grow:

- Any well-wheeled library can be added with `uv add`; it lands in
  `[project.dependencies]` and the lockfile.
- Heavy or niche ML libraries (e.g. `torch`, diffusion/upscale models) MUST
  go into `[project.optional-dependencies]` extras (e.g. `pic_utils[ml]`)
  so the base install stays light and platform/wheel issues stay isolated
  to opt-in users.
- Each new runtime dependency needs a one-line justification in the PR that
  adds it (which feature/block needs it); no other approval process.
- The block interface (pending, architecture doc) is the extension point for
  techniques: new libraries should enter the app as new blocks, not by
  threading new libs through the GUI.

## 5. Platforms

- **Linux x86-64** is the primary development and validation platform.
- **Windows x86-64** is a supported target: dependencies must remain
  pure-wheel-installable on Windows; CI (when it exists) must run the test
  suite on both.
- macOS: out of scope. Do not add platform-specific code that would make
  macOS support expensive, but do not test or guarantee it.

## 6. Distribution

- Open-source repository; users install with uv/pip from source
  (`uv tool install` or `uv run` from a clone).
- No installers, no PyInstaller bundles, no app-store packaging for v1.
- Committed assets policy: large binary examples do NOT belong in git;
  they live outside the repo or in Git LFS if needed for docs.

## 7. Rejected alternatives (do not reopen without a superseding decision)

| Rejected | Why |
|---|---|
| Web frontend (React/Svelte + FastAPI) | Two languages, image-data-over-HTTP cost, hand-rolled canvas interaction; QGraphicsScene gives it natively |
| Rust/C++ image core | Premature; the bottleneck is Python-level loops and full-res processing, both fixable in Python; ML ecosystem stays Python-only |
| conda/mamba | Heavier than needed; uv covers the compiled-wheel story on both target platforms |
| Tkinter / DearPyGui | Insufficient canvas/graphics interaction model for layer-based composition |
| Python 3.13/3.14 | Compiled-wheel ecosystem not yet uniformly mature across the dep set |

## 8. Open questions (owned by later documents)

- PRD: exact feature set, layer/variant UX details, block catalog.
- Architecture: canonical image representation (dtype, channel order),
  block/pipeline interface, spec JSON schema, undo model, caching,
  threading model, GPU policy.
