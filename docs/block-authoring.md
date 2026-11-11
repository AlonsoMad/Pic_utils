# Writing Pic Utils blocks

A **block** is a pure Python function that transforms an image, a mask, or a
stack of layers. Blocks are discovered from the built-in package and from the
user blocks folder; saving a file reloads it without restarting the app.

This guide matches the v0.1 block API (`BLOCK_API_VERSION`). See
`specs/architecture.md` Appendix C for the normative contract.

## Where blocks live

- Built-ins: `src/pic_utils/blocks/builtin/`.
- User blocks: the folder shown in **Settings → Blocks folder** (by default
  the app config `blocks/` folder). Only top-level `*.py` files are loaded;
  files starting with `_` and `__init__.py` are ignored. Helper modules in the
  same folder are importable.

A fresh install writes `example_blocks.py` with a working `posterize` block.

## The decorator

```python
from typing import Annotated

import numpy as np

from pic_utils.blocks import Choice, IntRange, OutputKind, Seed, block
from pic_utils.core.image import Image


@block(
    name="posterize",
    category="channels",
    description="Reduce each channel to N evenly spaced levels.",
    output=OutputKind.IMAGE,
)
def posterize(
    image: Image,
    levels: Annotated[int, IntRange(2, 32, step=1)] = 6,
    channel: Annotated[str, Choice(("all", "R", "G", "B"))] = "all",
    dither: bool = False,
    seed: Annotated[int | None, Seed()] = None,
) -> Image:
    rgb = image.rgb.astype(np.float32) / 255.0
    quantized = np.round(rgb * (levels - 1)) / (levels - 1)
    if channel != "all":
        index = {"R": 0, "G": 1, "B": 2}[channel]
        quantized[..., index] = rgb[..., index]
    out = np.clip(quantized * 255.0, 0, 255).astype(np.uint8)
    if dither:
        noise = np.random.default_rng(seed).uniform(-0.5, 0.5, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.from_rgb(out)
```

`@block` validates the signature at import time. A bad block raises
`BlockLoadError`; the registry keeps the file's last good definitions and
shows the error non-modally.

### Metadata

| Argument | Required | Meaning |
|---|---|---|
| `name` | yes | Unique block name across built-ins and user blocks. |
| `category` | yes | Library grouping; unknown categories are shown verbatim. |
| `description` | yes | One-line description shown in the library. |
| `output` | yes | `OutputKind.IMAGE`, `MASK`, or `LAYERS`. |

## Parameters

Parameters are ordinary function parameters. The UI generates controls from
their annotations and defaults:

| Annotation | Control | Notes |
|---|---|---|
| `int` with `IntRange(low, high, step=1)` | slider + spin box | clamped to the range |
| `float` with `FloatRange(low, high, step=0.01)` | slider + spin box | clamped to the range |
| `str` with `Choice(values)` | combo box | values must match the default's type |
| `bool` | check box | |
| `Color` | color button | hex color string |
| `int \| None` with `Seed()` | seed field | `None` means "inject the experiment seed" |
| `Mask` | mask binding | bound to an upstream mask block |

Rules:

1. Every scalar parameter has a default. Defaults are real values, not
   sentinels.
2. `Seed` parameters are `int | None`; the executor replaces `None` with the
   experiment's concrete seed.
3. `Mask` parameters are not edited numerically; the pipeline editor binds
   them to an upstream step that outputs a mask.
4. `Choice` values must all share the default's type.

## Images, masks, and layers

The canonical model is 8-bit RGB plus an optional alpha plane:

```python
from pic_utils.core.image import Image, Layer, Mask

image.rgb     # (H, W, 3) uint8
image.alpha   # (H, W) uint8 or None
image.size    # (width, height)
image.width, image.height
Image.from_rgb(rgb)
Image.from_rgba(rgba)
Image(rgb, alpha)
```

- The **first** `Image` parameter is the chain input.
- Additional `Image` parameters receive the run's **second image**.
- A `Mask` is an 8-bit array (`mask.data`, `mask.soft`).
- `OutputKind.LAYERS` returns a sequence of `Layer(image, name=...)`; each
  layer becomes a result file.

Return exactly the declared output type. Returning anything else fails the
experiment with `block.invalid_output`; the batch continues.

## Rules

1. One decorated function per block; unique `name`.
2. No I/O, no plotting, no global RNG, no mutation of the input image.
3. Use `np.random.default_rng(seed)` for stochastic behavior so the same seed
   produces the same output.
4. Keep the function deterministic and side-effect free.
5. Do not import heavy optional dependencies at module import time; import
   them inside the function if they are optional.
6. Test with synthetic arrays and assert the same seed gives the same output.

## Hot reload

Save the file. The blocks folder is watched; the registry reloads within about
0.5 s. You can also press **Tools → Reload blocks**. The reload is atomic:
every snapshot swap is immutable, and a file that fails to import keeps its
last good definitions. A block whose signature changed is reported against any
pipeline that uses it.

Reload errors are deduplicated in the status bar. Fix the file and save again
to clear them.

## Running a block

A block is executed in a spawned worker process with a frozen copy of the user
blocks folder. The worker receives the source image through shared memory,
executes the pipeline, and writes result files atomically. Failures, timeouts,
and worker crashes are contained per experiment.

For a quick check without the UI:

```python
from pathlib import Path

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.image import Image
from pic_utils.pipeline.execute import execute_pipeline
from pic_utils.pipeline.model import ExperimentSpec, Pipeline, PipelineStep

registry = BlockRegistry(Path("~/.config/PicUtils/PicUtils/blocks").expanduser())
pipeline = Pipeline((PipelineStep(id="s1", block="posterize", params={"levels": 4}),))
experiment = ExperimentSpec(
    index=0,
    source_index=0,
    params={},
    seed=7,
    label="check",
)
output = execute_pipeline(pipeline, registry.snapshot(), Image.from_rgb(rgb), None, experiment)
result = output.layers[0].image
```

## Debugging

- Import errors appear in the status bar and in the app log.
- Run failures are logged in `<run>/run.log` with stable codes such as
  `block.raised`, `block.invalid_output`, `block.timeout`, and
  `worker.crashed`.
- Use **Help → Copy diagnostics** to copy the app version, platform, Qt
  version, and log path into a bug report.
