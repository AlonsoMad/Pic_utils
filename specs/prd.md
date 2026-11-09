# Pic Utils — Product Requirements Document

**Version:** 1.0 (consolidated)
**Status:** FINAL for v1 planning
**Date:** 2026-11-09
**Companion documents:**
- `specs/stack.md` — decided technology stack (binding)
- `specs/legacy-modernization.md` — core library modernization plan

**Audience:** implementers; later agents producing the architecture and
design documents; the product owner.

**Conventions:**
- `MUST` / `SHOULD` / `MAY` follow RFC 2119.
- Requirement IDs are stable. Never renumber; deprecate instead.
- `[DECIDED]` — confirmed by the product owner in the questionnaire.
- `[PROPOSED]` — recommendation awaiting confirmation; do not treat as final.
- `[OPEN]` — unresolved; owned by the architecture document or a later
  decision record.
- `[ASSUMPTION]` — inferred; must be validated before implementation relies
  on it.

---

## 0. Document control

This PRD consolidates six questionnaire rounds (R1–R6). Every requirement
traces back to a decision or an explicitly flagged gap; see §16.

Change policy:
- New requirements are appended with a new ID.
- Changed requirements keep their ID and gain a changelog note.
- Anything marked `[OPEN]` MUST NOT be implemented on assumption; resolve it
  first in the architecture document.

---

## 1. Executive summary

Pic Utils is a desktop **laboratory for experimental image art**: glitch,
collage, double exposure, channel surgery, and other unorthodox edits. The
artist builds parametric pipelines out of reusable **blocks**, runs batches of
experiments with parameter sweeps or random sampling, curates the results in a
grid, and composes the winners on a layered **canvas** before exporting.

The product is designed tinkerer-first: writing a new image-processing
function and hot-reloading it into the running app is a first-class workflow,
not a plugin afterthought.

The application is a single-window desktop app (Lab and Canvas modes), local
and offline, built on the stack fixed in `specs/stack.md` (Python 3.12,
PySide6, numpy/OpenCV/Pillow/scikit-learn/scikit-image).

---

## 2. Glossary

| Term | Definition |
|---|---|
| **Block** | A user- or app-provided image-processing function with declared parameters and a declared output type. The atomic unit of technique. |
| **Pipeline** | An ordered chain of blocks with parameter values. Linear, with optional side mask inputs. |
| **Experiment** | One execution of a pipeline with one concrete parameter set (one concrete seed, if stochastic) on one source image. |
| **Batch** | A set of experiments launched together. Default size 32, user-adjustable. |
| **Run** | One batch execution, identified by a timestamped subfolder. |
| **Result** | The output image of one experiment, stored on disk under the run subfolder. |
| **Variant** | A result considered as a candidate for curation. |
| **Curation** | The accept/discard pass over a run's results in the in-app grid. |
| **Canvas** | The freeform layered composition workspace. One open at a time. |
| **Layer** | An image placed on the canvas with transform, opacity, blend mode, optional mask, optional group membership. |
| **Mask** | A per-pixel selector, binary (0/255) or soft (grayscale 0–255), produced or consumed by blocks and applicable to canvas layers. |
| **Preset** | Saved parameters for a block, or a saved pipeline, or a shipped example experiment. |
| **Project** | The saved experiment recipe: pipeline, parameters, seeds, references. Does **not** contain canvas state. |
| **Canvas file** | Separate save containing layers, transforms, and masks of a canvas. |
| **Source image** | An input image selected for experimentation. |
| **Second image** | A fixed auxiliary image paired with every source image in a batch that uses multi-image blocks. |
| **Seed** | The random state that makes a stochastic experiment reproducible. |

---

## 3. Product vision and positioning

### 3.1 Vision statement `[DECIDED R1]`

> "A lab for experimental art and pictures. Imagine a Photoshop but for doing
> glitch art, image experiments, and unorthodox image editing."

### 3.2 What the product is

- A **generator of many variants** under user-defined parametric recipes.
- A **curation surface** for picking winners from those variants.
- A **composition surface** for assembling accepted results into collages.
- A **tinkering platform** where new techniques are small Python functions
  that plug in without modifying the app.

### 3.3 What the product is not `[DECIDED R1]`

| Not this | Why |
|---|---|
| A Photoshop replacement | No retouching, no precision editing workflow |
| A RAW developer | RAW explicitly out of scope |
| A video/animation tool | Static images only |
| A vector/text design tool | Raster only |
| A cloud/collaboration product | Local, offline, single-user |
| A plugin marketplace | Third-party blocks exist, but no registry or monetization |

### 3.4 Design principles

1. **Experimentation over polish.** Optimize for trying 32 things, not for
   perfecting one.
2. **Reproducibility.** Every experiment is parameterized and seeded; any
   accepted result can be regenerated.
3. **Tinkerer-first.** The block API is a product surface; adding a technique
   must be trivial and hot-reloadable.
4. **Non-destructive where recorded.** Recipes and canvas files preserve the
   inputs to every operation.
5. **Memory discipline.** The app must operate within an 8 GB ceiling on a
   12 GB machine, with 48 MP images.
6. **Local-first.** No network dependency for any core workflow.

---

## 4. Users

### 4.1 Primary persona: the tinkerer-artist `[DECIDED R1]`

- Writes Python; comfortable with numpy arrays.
- Treats the app as a lab: builds pipelines, sweeps parameters, writes new
  blocks when a technique is missing.
- Values surprise and volume over determinism of aesthetics.
- Runs the app on a personal Linux or Windows machine.

### 4.2 Secondary persona: the preset artist `[ASSUMPTION]`

- Uses shipped example experiments and saved presets.
- Does not write code, or writes only rarely.
- Must be able to complete the full journey (load → run → curate → canvas →
  export) without opening a Python file.

### 4.3 Anti-personas

- Retouchers needing heal/clone/dodge tools.
- RAW photographers needing a development pipeline.
- Video editors.
- Teams needing shared/cloud projects.

---

## 5. Goals and success metrics

| ID | Goal | Measure |
|---|---|---|
| G1 | End-to-end workflow without touching Python | A preset-artist can load, run, curate, compose, export in one session `[DECIDED R1]` |
| G2 | Block authoring is trivial | A new block is written as one decorated function, hot-reloaded, and usable without restart `[DECIDED R2/R3]` |
| G3 | Batch throughput | 32 experiments per batch completes without exhausting the 8 GB ceiling `[DECIDED R3/R6]` |
| G4 | Reproducibility | Any accepted result can be regenerated from its project + seed `[DECIDED R2]` |
| G5 | Stability | A crashing block does not abort the batch; the app does not crash `[DECIDED R5]` |
| G6 | Portability | Same workflows on Linux and Windows `[DECIDED stack]` |

`[OPEN]` Quantitative time targets (per-experiment latency, preview latency)
are not yet set; see NFR-PERF-05.

---

## 6. User journeys

### 6.1 Journey A — preset experiment to exported image `[DECIDED R2]`

1. Launch the app; the Lab mode opens.
2. Load one or more source images (file picker, folder, or drag-and-drop).
3. Browse the block library; pick a shipped example experiment.
4. Tune parameters; choose sweep or random sampling; launch the batch.
5. Watch the progress bar; optionally cancel.
6. When the run finishes, the results grid opens for that run.
7. Accept the best results; discard the rest (they go to the session trash).
8. Switch to Canvas mode; accepted images appear in the canvas asset tray.
9. Compose: place layers, transform, set z-order/opacity/blend, apply masks,
   crop, group.
10. Export the canvas; save the canvas file and/or the project recipe.

### 6.2 Journey B — authoring a new block `[DECIDED R2/R3]`

1. In Lab mode, open the block library.
2. Create a new `.py` file (or edit an existing one) in the user blocks
   folder, using the documented decorator.
3. Trigger reload (button or automatic file watch `[OPEN]`).
4. The new block appears in the library with auto-generated controls.
5. Use it in a pipeline immediately.

### 6.3 Journey C — sweep, curate, collage `[DECIDED R2/R5]`

1. Build a pipeline of 3–5 blocks.
2. Select parameters to sweep (e.g. 5 masks × 4 blends = 20 experiments).
3. Add random sampling for the remaining parameters (e.g. 12 samples).
4. Launch; results land in a timestamped run subfolder.
5. Curate in the grid with keyboard shortcuts.
6. Compose accepted results into a collage on the canvas.
7. Save the canvas file for later reopening; export the final image.

### 6.4 Journey D — reopen and reproduce `[DECIDED R2/R3]`

1. Open a saved project (recipe only).
2. Source images and the second image are re-linked (paths or re-pick).
3. Re-run the exact recipe with the stored seeds.
4. Results match the original run `[ASSUMPTION: deterministic blocks]`.

---

## 7. Functional requirements

### 7.1 Image input and formats

| ID | Priority | Requirement |
|---|---|---|
| FR-IO-01 | MUST | Accept PNG, JPEG, and TIFF source images. `[DECIDED R4]` |
| FR-IO-02 | MUST | Process 8-bit-per-channel images only; 16-bit inputs are out of scope. `[DECIDED R4]` |
| FR-IO-03 | MUST | Support images up to 48 MP. `[DECIDED R4]` |
| FR-IO-04 | SHOULD | Warn and refuse (rather than crash) on images above the supported limit. |
| FR-IO-05 | MUST | Handle grayscale, RGB, and RGBA inputs, normalizing internally to a single representation. `[ASSUMPTION: RGB + optional alpha]` |
| FR-IO-06 | MUST | Allow loading multiple source images for a batch. `[DECIDED R2]` |
| FR-IO-07 | SHOULD | Support file picker, folder selection, and drag-and-drop. `[ASSUMPTION]` |
| FR-IO-08 | SHOULD | Apply EXIF orientation on load. `[OPEN]` |
| FR-IO-09 | MUST | Allow selecting a fixed **second image** for batches containing multi-image blocks. `[DECIDED R5]` |
| FR-IO-10 | MUST | Report unreadable/corrupt files without aborting the batch. `[DECIDED R5]` |

### 7.2 Block system

| ID | Priority | Requirement |
|---|---|---|
| FR-BLK-01 | MUST | A block is authored as a decorated Python function. `[DECIDED R3]` |
| FR-BLK-02 | MUST | Block parameters are introspected from the function signature/annotations; no separate JSON descriptor. `[DECIDED R3]` |
| FR-BLK-03 | MUST | Supported parameter types: integer, float, choice/enum, boolean, color, seed. Free text is not required. `[DECIDED R6]` |
| FR-BLK-04 | SHOULD | Parameters declare defaults, ranges, and steps for UI generation. `[ASSUMPTION]` |
| FR-BLK-05 | MUST | Blocks declare one of four output types: image→image, image→mask, image→multiple layers, image+mask(s)→image. `[DECIDED R4]` |
| FR-BLK-06 | MUST | Mask outputs may be binary or soft; the block declares which. `[DECIDED R6]` |
| FR-BLK-07 | MUST | Blocks can declare N image inputs; in batch, extra inputs are bound to the fixed second image. `[DECIDED R5]` |
| FR-BLK-08 | MUST | Blocks hot-reload without restarting the app. `[DECIDED R2]` |
| FR-BLK-09 | MUST | Block identity (id/name) survives reload; pipelines referencing a reloaded block keep working. `[DECIDED R2]` |
| FR-BLK-10 | MUST | A crashing block is reported to the runner and the experiment is skipped; the batch continues. `[DECIDED R5]` |
| FR-BLK-11 | MUST | Stochastic blocks accept a seed parameter and are deterministic given the same seed. `[DECIDED R2/R6]` |
| FR-BLK-12 | SHOULD | Blocks expose name, category, and description for library browsing. `[OPEN: metadata shape]` |
| FR-BLK-13 | MUST | The block API is public and documented, but provisional until 1.0. `[DECIDED R3]` |
| FR-BLK-14 | PROPOSED | Blocks execute as arbitrary Python in-process (local trust). `[PROPOSED R2]` |

### 7.3 Pipeline builder

| ID | Priority | Requirement |
|---|---|---|
| FR-PIPE-01 | MUST | A pipeline is a linear chain: each block receives the previous block's image. `[DECIDED R4]` |
| FR-PIPE-02 | MUST | Blocks may consume masks produced earlier in the chain (side inputs). `[DECIDED R4]` |
| FR-PIPE-03 | MUST | Users can add, remove, reorder, and bypass blocks. `[DECIDED R4]` |
| FR-PIPE-04 | MUST | Users can edit block parameters before running. `[DECIDED R2]` |
| FR-PIPE-05 | MUST | Pipelines can be saved as presets and loaded later. `[DECIDED R5]` |
| FR-PIPE-06 | MUST | Shipped example experiments are included. `[DECIDED R5]` |
| FR-PIPE-07 | SHOULD | The pipeline is validated before running (missing inputs, invalid ranges, incompatible output types). `[ASSUMPTION]` |
| FR-PIPE-08 | SHOULD | A preview of the pipeline on one image is available before launching a batch. `[OPEN]` |
| FR-PIPE-09 | MUST | A pipeline can mix sweep-selected and sampled parameters in one run. `[DECIDED R2]` |

### 7.4 Experiment runner

| ID | Priority | Requirement |
|---|---|---|
| FR-RUN-01 | MUST | Support cartesian sweeps over selected parameter values. `[DECIDED R2]` |
| FR-RUN-02 | MUST | Support random sampling of N parameter sets, seeded. `[DECIDED R2]` |
| FR-RUN-03 | MUST | Default batch size is 32; the user can change it per run. `[DECIDED R6]` |
| FR-RUN-04 | MUST | Show a progress bar with a cancel button; the app remains usable during a run. `[DECIDED R6]` |
| FR-RUN-05 | MUST | Cancelling stops scheduling new experiments and keeps completed results. `[ASSUMPTION]` |
| FR-RUN-06 | MUST | Failed experiments are skipped and logged; the batch continues. `[DECIDED R5]` |
| FR-RUN-07 | MUST | Each run writes results to its own timestamped subfolder. `[DECIDED R5]` |
| FR-RUN-08 | MUST | Each result records its parameter values and seed. `[DECIDED R2]` |
| FR-RUN-09 | SHOULD | The runner reports a summary at the end (completed, failed, skipped). `[ASSUMPTION]` |
| FR-RUN-10 | MUST | The runner respects the 8 GB memory ceiling. `[DECIDED R3]` |

### 7.5 Results and curation

| ID | Priority | Requirement |
|---|---|---|
| FR-RES-01 | MUST | Results are displayed in an in-app grid per run. `[DECIDED R6]` |
| FR-RES-02 | MUST | Each result can be accepted or discarded. `[DECIDED R2]` |
| FR-RES-03 | MUST | Keyboard shortcuts exist for accept/discard and navigation. `[DECIDED R6]` |
| FR-RES-04 | MUST | Discarded results move to a session trash folder, recoverable until the session ends. `[DECIDED R4]` |
| FR-RES-05 | MUST | Accepted results are available to the canvas. `[DECIDED R2]` |
| FR-RES-06 | MUST | Variants can be compared side-by-side and/or toggled (A/B). `[DECIDED R1]` |
| FR-RES-07 | SHOULD | The grid shows each result's parameters and seed. `[ASSUMPTION]` |
| FR-RES-08 | SHOULD | Results can be zoomed/inspected at full resolution. `[ASSUMPTION]` |
| FR-RES-09 | MAY | A result can be re-run or used as the starting point of a new experiment. `[OPEN]` |

### 7.6 Canvas

| ID | Priority | Requirement |
|---|---|---|
| FR-CAN-01 | MUST | Place accepted images as layers. `[DECIDED R3]` |
| FR-CAN-02 | MUST | Move, scale, rotate, and flip layers. `[DECIDED R3]` |
| FR-CAN-03 | MUST | Reorder layers (z-order) via the layer list. `[DECIDED R3]` |
| FR-CAN-04 | MUST | Per-layer opacity and blend modes: normal, multiply, screen, overlay, add, difference. `[DECIDED R6]` |
| FR-CAN-05 | MUST | Apply masks (channel/color/segmentation) to layers. `[DECIDED R3]` |
| FR-CAN-06 | MUST | Crop/resize the canvas. `[DECIDED R3]` |
| FR-CAN-07 | MUST | Layer groups. `[DECIDED R3]` |
| FR-CAN-08 | MUST | Canvas size equals the largest placed image. `[DECIDED R5]` |
| FR-CAN-09 | MUST | One canvas open at a time. `[DECIDED R5]` |
| FR-CAN-10 | MUST | Manual canvas edits are not recorded in the project recipe. `[DECIDED R2]` |
| FR-CAN-11 | MUST | Canvas state can be saved to a separate layered canvas file and reopened. `[DECIDED R4]` |
| FR-CAN-12 | MUST NOT | Hand-painted mask brush. `[DECIDED R3]` |
| FR-CAN-13 | MUST NOT | Text layers. `[DECIDED R3]` |

### 7.7 Persistence

| ID | Priority | Requirement |
|---|---|---|
| FR-PERS-01 | MUST | Save/load the experiment recipe (pipeline, parameters, seeds) as JSON. `[DECIDED R1/R3]` |
| FR-PERS-02 | MUST | The project does not store canvas state. `[DECIDED R3]` |
| FR-PERS-03 | MUST | A separate canvas file stores layers, transforms, and masks. `[DECIDED R4]` |
| FR-PERS-04 | MUST | Presets: block parameter presets, pipeline presets, and shipped examples. `[DECIDED R5]` |
| FR-PERS-05 | SHOULD | Project files are portable across machines; paths are stored relatively or re-linked on load. `[ASSUMPTION]` |
| FR-PERS-06 | SHOULD | File formats carry a version field for future migration. `[ASSUMPTION]` |
| FR-PERS-07 | MAY | Autosave/recovery snapshot. `[OPEN]` |

### 7.8 Export

| ID | Priority | Requirement |
|---|---|---|
| FR-EXP-01 | MUST | Export PNG, JPEG, and TIFF. `[DECIDED R5]` |
| FR-EXP-02 | MUST | The user chooses the export scope per export: canvas only, canvas + accepted results, or everything. `[DECIDED R5]` |
| FR-EXP-03 | MUST | Export at full resolution. `[DECIDED R1]` |
| FR-EXP-04 | SHOULD | JPEG quality is configurable. `[OPEN]` |
| FR-EXP-05 | SHOULD | Export naming is predictable and collision-safe. `[OPEN]` |
| FR-EXP-06 | SHOULD | Export preserves or embeds the sRGB profile. `[ASSUMPTION]` |

### 7.9 Undo/redo

| ID | Priority | Requirement |
|---|---|---|
| FR-UNDO-01 | MUST | Undo/redo for parameter changes. `[DECIDED R4]` |
| FR-UNDO-02 | MUST | Undo/redo for block add/remove/reorder. `[DECIDED R4]` |
| FR-UNDO-03 | MUST | Undo/redo for canvas transforms. `[DECIDED R4]` |
| FR-UNDO-04 | MAY NOT | Undo for result deletion or project load/save is not required. `[DECIDED R4]` |
| FR-UNDO-05 | SHOULD | Undo history depth is bounded and documented. `[OPEN]` |

### 7.10 UI shell

| ID | Priority | Requirement |
|---|---|---|
| FR-UI-01 | MUST | Single window with a Lab mode and a Canvas mode switch. `[DECIDED R5]` |
| FR-UI-02 | MUST | Lab mode exposes: block library, pipeline editor, parameter panel, run controls, results grid. `[DECIDED R2/R6]` |
| FR-UI-03 | MUST | Canvas mode exposes: canvas, layer list, layer properties (opacity/blend/mask), export. `[DECIDED R3]` |
| FR-UI-04 | MUST | Block reload action is reachable from the UI. `[DECIDED R2]` |
| FR-UI-05 | MUST | Progress bar with cancel during runs. `[DECIDED R6]` |
| FR-UI-06 | SHOULD | Keyboard shortcuts for curation and common actions. `[DECIDED R6]` |
| FR-UI-07 | MUST | UI is internationalized for English and Spanish. `[DECIDED R6]` |
| FR-UI-08 | MAY | Dark theme. `[OPEN]` |

#### 7.10.1 Lab mode layout (guidance, not binding)

- **Top bar:** mode switch (Lab | Canvas), run controls (launch/cancel),
  progress bar.
- **Left panel:** block library with search, categories, and the reload
  action.
- **Center:** pipeline editor (ordered block cards) plus a preview pane.
- **Right panel:** auto-generated parameter editor for the selected block.
- **Bottom:** results grid for the selected run, with accept/discard state.
- **Status bar:** memory usage and run status.

#### 7.10.2 Canvas mode layout (guidance, not binding)

- **Top bar:** mode switch, export, save canvas, canvas size.
- **Left:** asset tray of accepted results.
- **Center:** canvas view with selection and transform handles.
- **Right:** layer list (z-order, visibility) and layer properties (opacity,
  blend mode, mask, group).
- **Status bar:** canvas dimensions and memory usage.

### 7.11 Settings

| ID | Priority | Requirement |
|---|---|---|
| FR-SET-01 | SHOULD | Configurable user blocks folder. `[ASSUMPTION]` |
| FR-SET-02 | SHOULD | Configurable results root folder and trash location. `[ASSUMPTION]` |
| FR-SET-03 | MUST | Language selection (English/Spanish). `[DECIDED R6]` |
| FR-SET-04 | SHOULD | Memory ceiling is configurable up to 8 GB; default 8 GB. `[DECIDED R3]` |
| FR-SET-05 | MAY | Worker/thread count. `[OPEN]` |

### 7.12 Initial block catalog (v1)

The v1 block set is derived from the legacy `pic_utils.py` functions, but every
block MUST be reimplemented against the new block contract
(`specs/legacy-modernization.md`); nothing is ported as-is.

| Block | Legacy origin | Inputs | Output | Key parameters |
|---|---|---|---|---|
| `channel_permute` | `visualize_permutations` | 1 image | image | per-channel transform choice (identity, invert, …) |
| `permutation_grid` | `visualize_permutations` | 1 image | multiple layers (all permutations) | transform set |
| `overlay` | `overlay_images` | image + second image | image | x/y shift, z-order, white→transparent, resize |
| `double_exposure` | `double_exposure` | image + second image | image | exposure factor |
| `conditional_overlay` | `conditional_overlay` | image + second image + mask | image | brightness threshold, per-channel ranges, strict/loose |
| `segment` | `segment_and_color` | 1 image | multiple layers (one per cluster) | cluster count, superpixels on/off, superpixel count, selected clusters, modular recolor, seed |
| `channel_mask` | mask logic in `conditional_overlay` | 1 image | mask | channel, low/high range, soft |
| `brightness_mask` | mask logic in `conditional_overlay` | 1 image | mask | threshold, soft |
| `color_range_mask` | mask logic in `conditional_overlay` | 1 image | mask | R/G/B ranges, strict/loose, soft |

Notes:
- `find_channel_percentiles` is a **utility**, not a block: it feeds
  auto-threshold parameters for mask blocks. `[PROPOSED]`
- `[PROPOSED]` Additional blocks (blend, levels, curve) may ship if they are
  needed by the shipped presets; they are not required by this PRD.
- Every stochastic block (`segment` modular recolor, random sampling) MUST
  expose and honor a seed (FR-BLK-11).

### 7.13 Initial shipped presets

`[DECIDED R5]` Shipped example experiments exist. `[PROPOSED]` v1 ships at
least:

1. **Channel surgery** — `channel_mask` → `channel_permute`.
2. **Double exposure** — `double_exposure` → `conditional_overlay`.
3. **Cluster collage** — `segment` (explode) with modular recolor.
4. **Masked overlay** — `brightness_mask` → `overlay`.

Presets are ordinary pipeline presets, editable after loading.

---

## 8. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-PERF-01 | The app MUST handle 48 MP, 8-bit images. `[DECIDED R4]` |
| NFR-PERF-02 | Total memory (app + experiment execution) MUST stay under 8 GB. `[DECIDED R3]` |
| NFR-PERF-03 | A default batch of 32 experiments MUST complete without OOM on a 12 GB machine. `[DECIDED R3/R6]` |
| NFR-PERF-04 | The UI MUST remain responsive during a batch (progress and cancel work). `[DECIDED R6]` |
| NFR-PERF-05 | Preview latency target is `[OPEN]`; recommend < 200 ms for proxy previews. |
| NFR-REL-01 | A crashing or invalid block MUST NOT crash the app or abort the batch. `[DECIDED R5]` |
| NFR-REL-02 | Discarded results MUST be recoverable for the session. `[DECIDED R4]` |
| NFR-REL-03 | Saving a project or canvas MUST be atomic (no partial files). `[ASSUMPTION]` |
| NFR-COMPAT-01 | Linux x86-64 and Windows x86-64. `[DECIDED stack]` |
| NFR-USE-01 | English and Spanish UI. `[DECIDED R6]` |
| NFR-SEC-01 | Blocks are trusted local Python; no sandboxing. `[PROPOSED R2]` |
| NFR-OFFLINE-01 | No network access required for any core workflow. `[DECIDED R1]` |

---

## 9. File and folder formats (product level)

### 9.1 Project file (JSON)

Contains:
- format version
- source image references
- second image reference (if any)
- pipeline: ordered blocks with ids, parameter values, seeds
- run configuration: batch mode, batch size, sweep selections
- preset references `[OPEN: embed vs reference]`

Does not contain: canvas state, results, history.

### 9.2 Canvas file

Contains:
- format version
- canvas dimensions
- layer list: image references, transform, opacity, blend mode, mask
  reference, group membership
- layer images and masks stored alongside or embedded `[OPEN]`

### 9.3 Results layout

```
<results_root>/
  <run_id>/                  # timestamped, one per run
    result_0001.png
    result_0002.png
    manifest.json            # params, seeds, status per result
  trash/                     # session trash (recoverable)
```

`[OPEN]` Exact naming scheme for result files and run ids.

---

## 10. Error handling and edge cases

| Situation | Required behavior |
|---|---|
| Block raises an exception | Log, mark experiment failed, continue batch `[DECIDED R5]` |
| Block returns wrong shape/dtype | Treat as failure; log; continue `[ASSUMPTION]` |
| Source image unreadable | Skip image; report; continue `[DECIDED R5]` |
| Image exceeds 48 MP | Warn and refuse, or downscale `[OPEN]` |
| Memory ceiling approached | Abort current experiment, keep batch alive `[ASSUMPTION]` |
| Batch cancelled | Stop scheduling; keep completed results `[ASSUMPTION]` |
| Discard by mistake | Recover from session trash `[DECIDED R4]` |
| Missing second image | Block validation fails before run `[ASSUMPTION]` |
| Reloaded block changed signature | Pipeline validation reports the incompatibility `[ASSUMPTION]` |
| Project references missing files | Prompt to re-link `[ASSUMPTION]` |
| Export target unwritable | Report error; do not lose canvas state `[ASSUMPTION]` |

---

## 11. Constraints and assumptions

- Stack is fixed by `specs/stack.md`.
- 8-bit sRGB working space `[ASSUMPTION]`; no color management beyond sRGB.
- Local trust for blocks `[PROPOSED]`.
- Single user, single machine, no network.
- One canvas at a time; one run at a time `[ASSUMPTION: no concurrent runs]`.
- Canvas manual edits are not part of the reproducible recipe `[DECIDED]`.

---

## 12. Out of scope for v1

- Retouching tools (heal, clone, dodge/burn)
- RAW development
- Animation/video
- Text/vector layers
- Hand-painted mask brush
- Cloud sync/collaboration
- Mobile/web version
- Plugin marketplace/registry
- 16-bit pipeline
- CLI/batch-from-terminal `[candidate for v1.x]`
- macOS `[per stack doc]`
- Installers/packaged binaries `[per stack doc]`

---

## 13. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Provisional block API churn breaks user blocks | Tinkerers lose work | Version the API; document migration; keep a compatibility shim for one release |
| 8 GB ceiling vs 48 MP × 32 experiments | OOM, crashes | Stream results to disk; free intermediates; one experiment at a time; memory guard |
| Python-level loops in user blocks | Slow batches | Document vectorization guidance; profile and warn |
| Canvas work lost (recipe-only project) | Frustration | Separate canvas file (FR-CAN-11) + explicit save prompts |
| Curation at scale (hundreds of results) | UI sluggishness | Thumbnail cache; lazy loading; grid virtualization `[OPEN]` |
| i18n overhead | Slower UI work | Externalize strings from day one |
| Scope creep toward Photoshop | Dilutes product | Non-goals list is binding |

---

## 14. Release criteria (v1 definition of done)

The release is complete when all of the following hold:

- [ ] A user can launch the app on Linux and Windows and load PNG/JPEG/TIFF
      images up to 48 MP.
- [ ] A user can build a pipeline from blocks, or load a shipped example.
- [ ] A user can run a sweep and/or random-sampled batch of ~32 experiments
      with progress and cancel.
- [ ] Results land in a per-run subfolder and appear in the curation grid.
- [ ] Accept/discard works with keyboard shortcuts; discards are recoverable
      for the session.
- [ ] Accepted results can be composed on the canvas with transforms,
      z-order, opacity, the six blend modes, masks, crop, and groups.
- [ ] The canvas can be saved to a layered file and reopened.
- [ ] The project recipe can be saved and reloaded; re-running reproduces
      the results.
- [ ] A user can write a decorated Python function, hot-reload it, and use it
      as a block without restarting.
- [ ] A crashing block does not abort the batch or crash the app.
- [ ] Memory stays under 8 GB during a default batch.
- [ ] UI is available in English and Spanish.
- [ ] Export works to PNG/JPEG/TIFF with user-chosen scope.

### 14.1 Acceptance criteria (given/when/then)

| ID | Given | When | Then |
|---|---|---|---|
| AC-01 | A 48 MP PNG/JPEG/TIFF | The user loads it | It appears as a source and the app stays within the memory ceiling |
| AC-02 | A new `.py` block file using the decorator | The user triggers reload | The block appears in the library with auto-generated controls, no restart |
| AC-03 | A pipeline with a sweep of 20 values | The user launches | 20 results land in one run subfolder |
| AC-04 | A running batch | The user cancels | Scheduling stops; completed results remain |
| AC-05 | A block that raises | The batch runs | That experiment is skipped and logged; the batch completes |
| AC-06 | A run's results grid | The user discards a result | It moves to session trash and can be restored |
| AC-07 | Accepted images | The user composes on canvas | Transforms, z-order, opacity, blends, masks, crop, and groups all work |
| AC-08 | A composed canvas | The user saves and reopens the canvas file | Layers, transforms, and masks are restored |
| AC-09 | A saved project | The user re-runs it | Results match the original run (same seeds) |
| AC-10 | A finished canvas | The user exports | PNG/JPEG/TIFF at full resolution with the chosen scope |
| AC-11 | A default batch of 32 at 48 MP | The batch runs | Peak memory stays below 8 GB |
| AC-12 | The app running in English | The user switches to Spanish | All UI strings change without restart |

---

## 15. Open questions for the architecture document

1. Canonical image representation (dtype, channel order, alpha policy).
2. Block function signature and decorator API; how parameters declare ranges
   and steps; how mask inputs are referenced.
3. How multi-output blocks (segmentation) surface as layers.
4. Block registry identity and hot-reload semantics.
5. Pipeline validation rules and error taxonomy.
6. Project/canvas file schemas and versioning.
7. Result naming, run ids, manifest schema.
8. Preview strategy (proxy resolution, caching, invalidation).
9. Threading/process model for batches; cancellation mechanics.
10. Memory guard implementation and enforcement of the 8 GB ceiling.
11. Trash lifecycle and cleanup policy.
12. i18n framework and string catalog layout.
13. Whether block reload is manual or file-watch driven.
14. Export naming and metadata policy.
15. Undo history depth and storage.

---

## 16. Traceability

| Round | Topics | Sections |
|---|---|---|
| R1 | Vision, users, must-haves, non-goals, DoD | §1, §3, §4, §7, §12, §14 |
| R2 | Session workflow, batch modes, authoring, sandbox canvas, perf constraint | §3, §6, §7.4, §7.6, §8 |
| R3 | Canvas ops, project scope, results deletion, decorator API, API stability, 8 GB | §7.2, §7.5, §7.6, §7.7, §8 |
| R4 | Canvas save, trash, formats, pipeline semantics, block outputs, presets, undo | §7.2, §7.3, §7.7, §7.9 |
| R5 | Multi-image, presets, results folders, window shape, canvas size, export, errors | §6, §7.3, §7.4, §7.5, §7.6, §7.8 |
| R6 | Curation grid, batch size, param types, masks, blends, progress, i18n | §7.2, §7.4, §7.5, §7.6, §7.10 |

---

## Appendix A — Block authoring contract (product-level sketch)

> Illustrative only. The normative signature is an architecture decision.

```python
from pic_utils.blocks import block, Param

@block(
    name="channel_mask",
    category="masks",
    description="Keep pixels whose channel values fall in given ranges",
    output="mask",              # image | mask | layers | image_with_mask
)
def channel_mask(
    image: Image,
    channel: Param.choice(["R", "G", "B", "L"], default="B"),
    low: Param.int(0, 255, default=0),
    high: Param.int(0, 255, default=255),
    soft: Param.bool(default=False),
) -> Mask:
    ...
```

Requirements reflected here:
- Decorator registration (FR-BLK-01).
- Introspected parameter types (FR-BLK-02/03).
- Declared output type (FR-BLK-05).
- Seed parameter for stochastic blocks (FR-BLK-11).

---

## Appendix B — Illustrative project file

```json
{
  "format": "pic_utils.project",
  "version": 1,
  "sources": ["/images/la_imagen.png"],
  "second_image": null,
  "pipeline": [
    {"block": "channel_mask", "params": {"channel": "B", "low": 0, "high": 120}},
    {"block": "double_exposure", "params": {"exposure_factor": 0.5}}
  ],
  "run": {
    "mode": "sweep",
    "batch_size": 32,
    "sweep": {"exposure_factor": [0.3, 0.5, 0.7]},
    "seed": 8347
  }
}
```

---

## Appendix C — Illustrative canvas file

```json
{
  "format": "pic_utils.canvas",
  "version": 1,
  "size": [6000, 4000],
  "layers": [
    {
      "image": "layers/0001.png",
      "transform": {"x": 120, "y": -40, "scale": 1.2, "rotation": -8},
      "opacity": 0.85,
      "blend": "screen",
      "mask": "masks/0001.png",
      "group": null
    }
  ]
}
```
