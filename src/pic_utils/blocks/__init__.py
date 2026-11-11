"""Public block authoring API.

Import the decorator and metadata types from here::

    from pic_utils.blocks import Choice, IntRange, OutputKind, block
    from pic_utils.core.image import Image

The registry lives in :mod:`pic_utils.blocks.registry` so this package stays
free of import-time side effects.
"""

from pic_utils.blocks.api import (
    BLOCK_API_VERSION,
    BlockDef,
    BlockLoadError,
    Choice,
    Color,
    FloatRange,
    IntRange,
    OutputKind,
    ParamDef,
    ParamKind,
    Seed,
    block,
)

__all__ = [
    "BLOCK_API_VERSION",
    "BlockDef",
    "BlockLoadError",
    "Choice",
    "Color",
    "FloatRange",
    "IntRange",
    "OutputKind",
    "ParamDef",
    "ParamKind",
    "Seed",
    "block",
]
