"""Pipeline value types (architecture §6.1, §7.1).

A pipeline is a linear chain of :class:`PipelineStep` values. Steps are
serialized as a plain dict (``{"steps": [...]}``) so projects and presets can
store them verbatim; round-tripping is lossless and step ids survive
unchanged. Step ids are generated with ``uuid.uuid4().hex`` by the caller
(the UI) — this module only stores and transports them.

``derive_seed`` is the single seed-derivation primitive shared by the
executor and the planner: stable across versions and platforms because it
hashes explicit bytes rather than relying on a randomized hash seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ExperimentSpec",
    "Pipeline",
    "PipelineStep",
    "RunConfig",
    "SweepAxis",
    "derive_seed",
]


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """One block instance in a pipeline chain.

    ``params`` holds scalar parameter values only (``None`` means "auto" for
    a seed parameter). ``mask_bindings`` maps a parameter name of this step
    to the step id whose mask output it consumes.
    """

    id: str
    block: str
    params: dict[str, Any] = field(default_factory=dict)
    bypass: bool = False
    mask_bindings: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Pipeline:
    """An ordered chain of steps; executed first to last."""

    steps: tuple[PipelineStep, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-safe dict."""
        return {
            "steps": [
                {
                    "id": step.id,
                    "block": step.block,
                    "params": dict(step.params),
                    "bypass": step.bypass,
                    "mask_bindings": dict(step.mask_bindings),
                }
                for step in self.steps
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """Rebuild a pipeline from :meth:`to_dict` output.

        Raises:
            ValueError: if the shape is not a serialized pipeline.
        """
        if not isinstance(data, dict):
            raise ValueError("pipeline must be a dict")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            raise ValueError("pipeline must contain a 'steps' list")
        steps = [_step_from_dict(raw, index) for index, raw in enumerate(raw_steps)]
        return cls(steps=tuple(steps))


def _step_from_dict(raw: object, index: int) -> PipelineStep:
    if not isinstance(raw, dict):
        raise ValueError(f"step {index} must be a dict")
    where = f"step {index}"
    step_id = raw.get("id")
    block = raw.get("block")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError(f"{where}: 'id' must be a non-empty string")
    if not isinstance(block, str) or not block:
        raise ValueError(f"{where}: 'block' must be a non-empty string")
    params = raw.get("params", {})
    mask_bindings = raw.get("mask_bindings", {})
    bypass = raw.get("bypass", False)
    if not isinstance(params, dict):
        raise ValueError(f"{where}: 'params' must be a dict")
    if not isinstance(mask_bindings, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in mask_bindings.items()
    ):
        raise ValueError(f"{where}: 'mask_bindings' must be a dict of string keys and values")
    if not isinstance(bypass, bool):
        raise ValueError(f"{where}: 'bypass' must be a bool")
    return PipelineStep(
        id=step_id,
        block=block,
        params=dict(params),
        bypass=bypass,
        mask_bindings=dict(mask_bindings),
    )


@dataclass(frozen=True)
class SweepAxis:
    """One swept parameter: every listed value is used in its own run."""

    step_id: str
    param: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class RunConfig:
    """The sweep/sampling configuration for one batch run."""

    mode: str  # "grid" | "random" | "hybrid"
    batch_size: int = 32  # random sample count
    random_samples: int = 0  # hybrid: samples per grid combination
    master_seed: int = 0
    axes: tuple[SweepAxis, ...] = ()
    sampled: tuple[tuple[str, str], ...] = ()  # (step_id, param)


@dataclass(frozen=True)
class ExperimentSpec:
    """One concrete run of the pipeline: resolved params plus a seed."""

    index: int
    source_index: int
    params: dict[str, dict[str, Any]]  # step_id -> param -> value
    seed: int
    label: str


def derive_seed(master: int, *parts: str | int) -> int:
    """Derive a stable 31-bit seed from ``master`` and ordered ``parts``."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(master.to_bytes(8, "little"))
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") & 0x7FFFFFFF
