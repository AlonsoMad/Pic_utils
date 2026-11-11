# Pic Utils user guide

Pic Utils is a local desktop laboratory for experimental image art: build a
pipeline from small Python **blocks**, run batches of seeded experiments,
curate the results, compose the winners on a layered canvas, and export.

The application has two modes in one window:

- **Lab** — sources, pipeline, parameters, preview, batch runs, curation.
- **Canvas** — layered composition of accepted results and exported images.

Everything is local and offline. There is no account, server, or network
dependency.

## Installation and launch

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pic-utils
```

`uv run python -m pic_utils` is equivalent. On first launch Pic Utils creates
its settings file, a user-blocks folder with an example block, and a results
folder.

## The Lab

### 1. Add source images

Use **Add images…** in the run controls, or drag image files onto the window.
Supported inputs are PNG, JPEG, TIFF, BMP, and WebP up to 48 MP. Each source
has a checkbox; unchecked sources stay in the list but are not run.

A **second image** can be selected for blocks that take two images (for
example `overlay` or `double_exposure`). The same second image is paired with
every source in the batch.

### 2. Build a pipeline

The block library on the left lists built-in and user blocks. Double-click a
block (or press Enter) to append it to the pipeline. The pipeline is a linear
chain: the first block receives the source image, each later block receives
the previous output.

Select a step to edit its parameters on the right. Parameters are generated
from the block's declared ranges and choices. The **Vary** control marks a
parameter as fixed, swept, or randomly sampled:

- **Fixed** — one value for the whole batch.
- **Sweep** — a comma-separated list of values; every value becomes an
  experiment.
- **Sample** — drawn from the parameter's declared range for each experiment.

A step can be bypassed, moved, or removed. Mask-producing blocks can be bound
to a later block's mask parameter.

### 3. Preview

The preview pane renders the pipeline on the first selected source at proxy
resolution (1024 px long side by default). Rendering happens in a separate
process; edits are debounced and the latest request wins. A broken block shows
an error without blocking the UI.

### 4. Run a batch

Choose the run mode, batch size, and master seed, then press **Launch**:

- **grid** — one experiment per combination of swept values, per source.
- **random** — `batch size` random parameter sets, per source.
- **hybrid** — every grid combination plus `random samples` random sets.

The plan count updates live. A run writes to a timestamped subfolder under the
results root:

```
<results_root>/<run_id>/
  manifest.json      # run header, experiments, seeds, summary
  results.jsonl      # one record per written result
  curation.json      # accept/discard state
  run.log            # parent-side log, including block failures
  blocks/            # frozen copy of the user blocks used by the run
  result_0000_00.png # result files (one per experiment/layer)
  ...
```

A failing block fails only its own experiment; the batch continues and the
failure is logged. Cancelling stops scheduling; experiments already running
finish and their results remain. Force-cancel stops them immediately.

### 5. Curate results

The results grid shows the current run. Select a result and use:

- **A** — accept
- **D** — discard
- **R** — restore a discarded result

Discarded files move to the session trash and can be restored until the app
closes. Accepted results are available to the Canvas asset tray.

### 6. Projects and presets

A project (`.puproject`) stores the sources, second image, pipeline, and run
configuration. Re-running a loaded project reproduces the same experiments
because the master seed and parameter values are stored. Missing source files
open a relink dialog.

Presets (`.json`) store a pipeline or one block's parameters. Four example
presets ship with the app.

## The Canvas

Switch to **Canvas** with the mode buttons in the top bar.

### Layers

The asset tray lists accepted results and image files. Use **Add layer** or
**Add file…** to place an image on the canvas. The layer list shows the stack
in reverse z-order; use **Up**/**Down** to reorder.

Each layer has:

- position, scale, rotation, and horizontal/vertical flips;
- opacity and one of six blend modes: normal, multiply, screen, overlay, add,
  difference;
- optional mask;
- optional group membership, visibility, and lock.

Drag a layer in the viewport to move it; drag the handles to scale and rotate.
Arrow keys nudge selected layers; Delete removes them. Locked layers reject
edits, deletion, reordering, masking, and grouping.

### Masks

Select a layer and use **Generate mask…**. The mask dialog runs a mask block
on the layer image and previews the result. The mask is stored as a
content-addressed PNG beside the canvas file.

### Crop, resize, groups

- **Crop** starts an interactive crop of the canvas.
- **Resize…** changes the canvas dimensions.
- **Group**/**Ungroup** combine selected layers. Group opacity is applied per
  child, matching the full-resolution renderer.

### Save and reopen

**Ctrl+S** saves the canvas to a `.pucanvas` file; **Ctrl+Shift+S** saves as.
The file is JSON and its assets live in a sibling `<file>.assets` folder.
Saving writes only new assets, then commits the JSON; reopening restores
layers, transforms, masks, groups, and blend settings. Missing assets refuse to
open and are listed.

### Export

**Ctrl+E** opens the export dialog:

- **Format** — PNG, JPEG, or TIFF (JPEG quality is configurable; TIFF uses
  LZW).
- **Scope** — canvas only, canvas + accepted results, or canvas + the current
  run.
- **Export folder** — the parent folder; each export creates a timestamped
  subfolder and never overwrites an earlier one.

Every file is written atomically and carries an sRGB ICC profile.

## Settings

**Ctrl+,** opens Settings. Language changes apply immediately; theme changes
apply immediately; runner, preview, export, undo, and logging settings apply
to subsequent operations.

| Setting | Default | Notes |
|---|---|---|
| Language | system | English or Spanish, no restart |
| Theme | system | system, light, dark |
| Blocks folder | app config `blocks/` | user block files |
| Results root | `Pictures/PicUtils` | runs, trash, exports |
| Workers | 2 | 1–8 |
| Memory ceiling | 8192 MB | 512–8192 MB |
| Experiment timeout | 300 s | watchdog per experiment |
| Auto-reload blocks | on | watches the blocks folder |
| Proxy long side | 1024 px | preview resolution |
| Debounce | 150 ms | preview coalescing |
| Undo limit | 100 | per mode |
| JPEG quality | 92 | export |
| Log level | INFO | app log |

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+O | Open project |
| Ctrl+S | Save canvas |
| Ctrl+Shift+S | Save canvas as |
| Ctrl+E | Export canvas |
| Ctrl+, | Settings |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo (Canvas) |
| A / D / R | Accept / discard / restore result (Lab grid) |
| Arrow keys | Nudge selected canvas layers |
| Delete / Backspace | Delete selected canvas layers |

## Files and locations

| Data | Location |
|---|---|
| Settings | app config `settings.ini` |
| User blocks | app config `blocks/` |
| Logs | app data `logs/app.log` |
| Results | results root (default `Pictures/PicUtils`) |
| Projects | wherever you save `.puproject` |
| Canvases | wherever you save `.pucanvas` |

On Linux the app config folder is typically `~/.config/PicUtils/PicUtils` and
app data is `~/.local/share/PicUtils/PicUtils`. On Windows they live under
`%APPDATA%` and `%LOCALAPPDATA%`.

## Troubleshooting

- **A block does not appear** — check the blocks folder in Settings, then use
  **Tools → Reload blocks**. A file with an import error keeps its last good
  definitions and reports the error in the status bar.
- **A run fails immediately** — check `run.log` in the run folder and the app
  log. Common causes are a missing source, an invalid sweep value, or a
  pipeline that references an unknown block.
- **The canvas will not open** — the assets folder is missing or incomplete.
  The error lists the missing files; restore them beside the canvas file.
- **Memory pressure** — lower the worker count or the memory ceiling, or run
  fewer experiments per batch. The memory guard defers or fails experiments
  rather than exceeding the ceiling.
