"""Public block authoring API (architecture §5.1-§5.3).

A block is one decorated, pure function. The decorator introspects the
signature and annotations, validates the contract, and attaches an immutable
``BlockDef`` to the function. The registry collects those definitions; the
UI derives controls from ``ParamDef`` and the executor derives call arguments
from ``image_inputs`` / ``mask_inputs``.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from pic_utils.core.color import parse_hex_color
from pic_utils.core.errors import PicUtilsError
from pic_utils.core.image import Image, Layer, Mask

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

BLOCK_API_VERSION = 1


class OutputKind(StrEnum):
    """What a block returns."""

    IMAGE = "image"
    MASK = "mask"
    LAYERS = "layers"


class ParamKind(StrEnum):
    """The parameter kinds the UI and executor know how to handle."""

    INT = "int"
    FLOAT = "float"
    CHOICE = "choice"
    BOOL = "bool"
    COLOR = "color"
    SEED = "seed"
    IMAGE = "image"
    MASK = "mask"


@dataclass(frozen=True, slots=True)
class IntRange:
    """Integer slider metadata."""

    min: int
    max: int
    step: int = 1
    unit: str = ""

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"IntRange min {self.min} is greater than max {self.max}")
        if self.step < 1:
            raise ValueError(f"IntRange step must be >= 1, got {self.step}")


@dataclass(frozen=True, slots=True)
class FloatRange:
    """Float slider metadata."""

    min: float
    max: float
    step: float = 0.01
    log: bool = False
    unit: str = ""

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"FloatRange min {self.min} is greater than max {self.max}")
        if self.step <= 0:
            raise ValueError(f"FloatRange step must be > 0, got {self.step}")


@dataclass(frozen=True, slots=True)
class Choice:
    """A finite set of options; ``labels`` are optional display strings."""

    options: tuple[str | int | float, ...]
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if not options:
            raise ValueError("Choice needs at least one option")
        if len(set(options)) != len(options):
            raise ValueError("Choice options must be unique")
        if self.labels is not None:
            labels = tuple(self.labels)
            object.__setattr__(self, "labels", labels)
            if len(labels) != len(options):
                raise ValueError("Choice labels must match the number of options")


@dataclass(frozen=True, slots=True)
class Color:
    """Marker for a ``#rrggbb`` string parameter."""


@dataclass(frozen=True, slots=True)
class Seed:
    """Marker for an ``int | None`` seed parameter."""


@dataclass(frozen=True, slots=True)
class ParamDef:
    """One introspected function parameter."""

    name: str
    kind: ParamKind
    annotation: object
    default: object | None
    required: bool
    spec: IntRange | FloatRange | Choice | Color | Seed | None


@dataclass(frozen=True, slots=True)
class BlockDef:
    """An immutable block definition produced by :func:`block`."""

    name: str
    category: str
    description: str
    output: OutputKind
    fn: Callable[..., object]
    params: tuple[ParamDef, ...]
    image_inputs: tuple[str, ...]
    mask_inputs: tuple[str, ...]
    api_version: int
    module: str
    source: Path | None
    content_hash: str


@dataclass(frozen=True, eq=False)
class BlockLoadError(PicUtilsError):
    """A block source failed to import or did not satisfy the API contract."""

    reason: str
    path: Path | None = None
    block: str | None = None

    def __post_init__(self) -> None:
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))
        super().__init__(self._message())

    def _message(self) -> str:
        location = f" in {self.path}" if self.path is not None else ""
        block = f" block {self.block!r}" if self.block is not None else ""
        return f"block load failed{location}{block}: {self.reason}"


def block(
    *,
    name: str,
    category: str,
    description: str,
    output: OutputKind,
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorate a function as a block and attach its ``BlockDef``."""
    if not isinstance(output, OutputKind):
        raise TypeError(f"output must be an OutputKind, got {type(output).__name__}")
    if not name.strip():
        raise ValueError("block name must not be empty")
    if not category.strip():
        raise ValueError("block category must not be empty")
    if not description.strip():
        raise ValueError("block description must not be empty")

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        if not inspect.isfunction(fn):
            raise TypeError(f"block {name!r} must decorate a function")
        definition = _build_block_def(
            fn,
            name=name,
            category=category,
            description=description,
            output=output,
        )
        fn.__block_def__ = definition  # type: ignore[attr-defined]
        return fn

    return decorator


def _build_block_def(
    fn: Callable[..., object],
    *,
    name: str,
    category: str,
    description: str,
    output: OutputKind,
) -> BlockDef:
    try:
        signature = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)
    except (NameError, TypeError) as error:
        raise BlockLoadError(reason=f"cannot resolve annotations: {error}", block=name) from error

    params: list[ParamDef] = []
    image_inputs: list[str] = []
    mask_inputs: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise BlockLoadError(
                reason=f"parameter {parameter.name!r} must not use *args/**kwargs",
                block=name,
            )
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise BlockLoadError(
                reason=f"parameter {parameter.name!r} must not be positional-only",
                block=name,
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise BlockLoadError(
                reason=f"parameter {parameter.name!r} needs a type annotation",
                block=name,
            )
        param = _build_param_def(parameter, annotation, block_name=name)
        params.append(param)
        if param.kind is ParamKind.IMAGE:
            image_inputs.append(param.name)
        elif param.kind is ParamKind.MASK:
            mask_inputs.append(param.name)

    return_annotation = hints.get("return", signature.return_annotation)
    _validate_return(return_annotation, output, name)

    return BlockDef(
        name=name,
        category=category,
        description=description,
        output=output,
        fn=fn,
        params=tuple(params),
        image_inputs=tuple(image_inputs),
        mask_inputs=tuple(mask_inputs),
        api_version=BLOCK_API_VERSION,
        module=fn.__module__,
        source=None,
        content_hash="",
    )


def _build_param_def(
    parameter: inspect.Parameter,
    annotation: object,
    *,
    block_name: str,
) -> ParamDef:
    base, metadata = _split_annotated(annotation)
    kind, spec = _classify(base, metadata, block_name=block_name, param_name=parameter.name)
    required = parameter.default is inspect.Parameter.empty
    default = None if required else parameter.default

    if kind is ParamKind.IMAGE:
        if not required:
            raise BlockLoadError(
                reason=f"image input {parameter.name!r} must not have a default",
                block=block_name,
            )
    elif kind is ParamKind.MASK:
        if not required and default is not None:
            raise BlockLoadError(
                reason=f"optional mask {parameter.name!r} must default to None",
                block=block_name,
            )
    elif kind is ParamKind.SEED:
        if required:
            raise BlockLoadError(
                reason=f"seed {parameter.name!r} must have a default",
                block=block_name,
            )
        if default is not None and (isinstance(default, bool) or not isinstance(default, int)):
            raise BlockLoadError(
                reason=f"seed {parameter.name!r} must default to None or an int",
                block=block_name,
            )
    else:
        if required:
            raise BlockLoadError(
                reason=f"parameter {parameter.name!r} must have a default",
                block=block_name,
            )
        _validate_default(
            kind,
            spec,
            default,
            block_name=block_name,
            param_name=parameter.name,
        )

    return ParamDef(
        name=parameter.name,
        kind=kind,
        annotation=base,
        default=default,
        required=required,
        spec=spec,
    )


def _split_annotated(annotation: object) -> tuple[object, tuple[object, ...]]:
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return args[0], tuple(args[1:])
    return annotation, ()


def _is_optional(annotation: object, target: object) -> bool:
    if get_origin(annotation) not in (Union, types.UnionType):
        return False
    args = get_args(annotation)
    return len(args) == 2 and target in args and type(None) in args


def _classify(
    base: object,
    metadata: tuple[object, ...],
    *,
    block_name: str,
    param_name: str,
) -> tuple[ParamKind, IntRange | FloatRange | Choice | Color | Seed | None]:
    if len(metadata) > 1:
        raise BlockLoadError(
            reason=f"parameter {param_name!r} has more than one metadata spec",
            block=block_name,
        )
    spec = metadata[0] if metadata else None

    if base is Image:
        if spec is not None:
            raise BlockLoadError(
                reason=f"image input {param_name!r} takes no metadata",
                block=block_name,
            )
        return ParamKind.IMAGE, None
    if base is Mask:
        if spec is not None:
            raise BlockLoadError(
                reason=f"mask input {param_name!r} takes no metadata",
                block=block_name,
            )
        return ParamKind.MASK, None
    if _is_optional(base, Mask):
        if spec is not None:
            raise BlockLoadError(
                reason=f"mask input {param_name!r} takes no metadata",
                block=block_name,
            )
        return ParamKind.MASK, None
    if base is bool:
        if spec is not None:
            raise BlockLoadError(
                reason=f"bool parameter {param_name!r} takes no metadata",
                block=block_name,
            )
        return ParamKind.BOOL, None
    if _is_optional(base, int):
        if not isinstance(spec, Seed):
            raise BlockLoadError(
                reason=f"optional int parameter {param_name!r} needs Seed() metadata",
                block=block_name,
            )
        return ParamKind.SEED, spec
    if base is int:
        if isinstance(spec, IntRange):
            return ParamKind.INT, spec
        if isinstance(spec, Choice):
            _validate_choice_types(spec, int, block_name=block_name, param_name=param_name)
            return ParamKind.CHOICE, spec
        raise BlockLoadError(
            reason=f"int parameter {param_name!r} needs IntRange or Choice metadata",
            block=block_name,
        )
    if base is float:
        if isinstance(spec, FloatRange):
            return ParamKind.FLOAT, spec
        if isinstance(spec, Choice):
            _validate_choice_types(
                spec,
                (int, float),
                block_name=block_name,
                param_name=param_name,
            )
            return ParamKind.CHOICE, spec
        raise BlockLoadError(
            reason=f"float parameter {param_name!r} needs FloatRange or Choice metadata",
            block=block_name,
        )
    if base is str:
        if isinstance(spec, Choice):
            _validate_choice_types(spec, str, block_name=block_name, param_name=param_name)
            return ParamKind.CHOICE, spec
        if isinstance(spec, Color):
            return ParamKind.COLOR, spec
        raise BlockLoadError(
            reason=f"str parameter {param_name!r} needs Choice or Color metadata",
            block=block_name,
        )
    raise BlockLoadError(
        reason=f"parameter {param_name!r} has unsupported annotation {base!r}",
        block=block_name,
    )


def _validate_choice_types(
    spec: Choice,
    expected: type | tuple[type, ...],
    *,
    block_name: str,
    param_name: str,
) -> None:
    for option in spec.options:
        if isinstance(option, bool) or not isinstance(option, expected):
            raise BlockLoadError(
                reason=(
                    f"choice parameter {param_name!r} has option {option!r} "
                    f"that does not match its annotation"
                ),
                block=block_name,
            )


def _validate_default(
    kind: ParamKind,
    spec: IntRange | FloatRange | Choice | Color | Seed | None,
    default: Any,
    *,
    block_name: str,
    param_name: str,
) -> None:
    if kind is ParamKind.INT:
        assert isinstance(spec, IntRange)
        if isinstance(default, bool) or not isinstance(default, int):
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default must be an int",
                block=block_name,
            )
        if not spec.min <= default <= spec.max:
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default {default} is outside its range",
                block=block_name,
            )
    elif kind is ParamKind.FLOAT:
        assert isinstance(spec, FloatRange)
        if isinstance(default, bool) or not isinstance(default, int | float):
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default must be a number",
                block=block_name,
            )
        if not spec.min <= default <= spec.max:
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default {default} is outside its range",
                block=block_name,
            )
    elif kind is ParamKind.CHOICE:
        assert isinstance(spec, Choice)
        if not any(type(default) is type(option) and default == option for option in spec.options):
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default {default!r} is not one of its options",
                block=block_name,
            )
    elif kind is ParamKind.BOOL:
        if not isinstance(default, bool):
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default must be a bool",
                block=block_name,
            )
    elif kind is ParamKind.COLOR:
        if not isinstance(default, str):
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default must be a hex string",
                block=block_name,
            )
        try:
            parse_hex_color(default)
        except ValueError as error:
            raise BlockLoadError(
                reason=f"parameter {param_name!r} default {default!r} is not a hex color",
                block=block_name,
            ) from error


def _validate_return(annotation: object, output: OutputKind, block_name: str) -> None:
    if annotation is inspect.Signature.empty:
        raise BlockLoadError(reason="missing return annotation", block=block_name)
    if output is OutputKind.IMAGE:
        if annotation is not Image:
            raise BlockLoadError(
                reason=f"output is IMAGE but the return annotation is {annotation!r}",
                block=block_name,
            )
        return
    if output is OutputKind.MASK:
        if annotation is not Mask:
            raise BlockLoadError(
                reason=f"output is MASK but the return annotation is {annotation!r}",
                block=block_name,
            )
        return

    origin = get_origin(annotation)
    args = get_args(annotation)
    list_ok = origin is list and args == (Layer,)
    tuple_ok = origin is tuple and len(args) == 2 and args[0] is Layer and args[1] is Ellipsis
    if not (list_ok or tuple_ok):
        raise BlockLoadError(
            reason=f"output is LAYERS but the return annotation is {annotation!r}",
            block=block_name,
        )
