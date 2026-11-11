"""Cross-module contract tests for the pipeline package (P3-05).

These tests pin the invariants that span the model, validation, executor,
and planner, plus the architectural boundary: ``pipeline/`` imports neither
Qt nor :mod:`pic_utils.io`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import fixtures
import numpy as np
import pytest

from pic_utils.blocks.api import OutputKind, Seed, block
from pic_utils.blocks.registry import BlockRegistry, BlockSnapshot
from pic_utils.core.errors import ValidationError
from pic_utils.core.image import Image
from pic_utils.pipeline.execute import execute_pipeline
from pic_utils.pipeline.model import Pipeline, PipelineStep, RunConfig, SweepAxis
from pic_utils.pipeline.plan import build_plan
from pic_utils.pipeline.validate import is_blocking, validate_pipeline, validate_run_config

PIPELINE_DIR = Path(__file__).resolve().parents[2] / "src" / "pic_utils" / "pipeline"
SNAPSHOT = BlockRegistry().snapshot()


@block(
    name="contract_seeded",
    category="test",
    description="Encode the resolved seed into every pixel.",
    output=OutputKind.IMAGE,
)
def contract_seeded(image: Image, seed: Annotated[int | None, Seed()] = None) -> Image:
    value = 0 if seed is None else seed % 256
    return Image(np.full_like(image.rgb, value))


def snapshot_with_seeded() -> BlockSnapshot:
    from types import MappingProxyType

    definitions = dict(SNAPSHOT.defs)
    definitions["contract_seeded"] = contract_seeded.__block_def__  # type: ignore[attr-defined]
    return BlockSnapshot(defs=MappingProxyType(definitions), hash="test")


def full_pipeline() -> Pipeline:
    return Pipeline(
        steps=(
            PipelineStep(id="perm", block="channel_permute", params={"order": "rgb"}),
            PipelineStep(
                id="mask",
                block="channel_mask",
                params={"channel": "R", "low": 0, "high": 128},
            ),
            PipelineStep(
                id="overlay",
                block="conditional_overlay",
                params={"threshold": 200},
                mask_bindings={"mask": "mask"},
            ),
        )
    )


def test_pipeline_package_has_no_qt_or_io_imports() -> None:
    script = """
import sys
import pic_utils.pipeline
import pic_utils.pipeline.model
import pic_utils.pipeline.validate
import pic_utils.pipeline.execute
import pic_utils.pipeline.plan
forbidden = [
    name
    for name in sys.modules
    if name == "PySide6"
    or name.startswith("PySide6.")
    or name == "pic_utils.io"
    or name.startswith("pic_utils.io.")
]
assert not forbidden, forbidden
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _runtime_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return names


def test_pipeline_sources_do_not_import_forbidden_modules() -> None:
    forbidden_prefixes = ("PySide6", "PyQt5", "PyQt6", "shiboken6", "pic_utils.io")
    modules = sorted(PIPELINE_DIR.glob("*.py"))
    assert modules
    for module in modules:
        for imported in _runtime_imports(module):
            assert imported != "io", f"{module.name} imports stdlib io"
            assert not imported.startswith(forbidden_prefixes), f"{module.name} imports {imported}"


def test_serialization_round_trip_preserves_execution() -> None:
    pipeline = full_pipeline()
    restored = Pipeline.from_dict(pipeline.to_dict())
    assert restored == pipeline
    source = Image(fixtures.rgb_gradient())
    second = Image(fixtures.checkerboard()[..., None].repeat(3, axis=-1))
    spec = build_plan(RunConfig(mode="grid"), pipeline, SNAPSHOT, source_count=1)[0]
    original = execute_pipeline(pipeline, SNAPSHOT, source, second, spec)
    round_tripped = execute_pipeline(restored, SNAPSHOT, source, second, spec)
    assert np.array_equal(original.layers[0].image.rgb, round_tripped.layers[0].image.rgb)


def test_valid_pipeline_validates_and_executes() -> None:
    pipeline = full_pipeline()
    issues = validate_pipeline(
        pipeline, SNAPSHOT, second_image_available=True, sources_available=True
    )
    assert not is_blocking(issues)
    config = RunConfig(
        mode="grid",
        axes=(SweepAxis(step_id="perm", param="order", values=("rgb", "bgr")),),
        master_seed=1,
    )
    plan = build_plan(config, pipeline, SNAPSHOT, source_count=2)
    assert len(plan) == 4
    sources = [
        Image(fixtures.rgb_gradient()),
        Image(fixtures.checkerboard()[..., None].repeat(3, axis=-1)),
    ]
    second = Image(fixtures.rgb_gradient())
    for spec in plan:
        output = execute_pipeline(pipeline, SNAPSHOT, sources[spec.source_index], second, spec)
        assert len(output.layers) == 1
        assert output.layers[0].image.size == sources[spec.source_index].size


def test_validator_and_planner_agree_on_the_experiment_limit() -> None:
    pipeline = full_pipeline()
    config = RunConfig(
        mode="grid",
        axes=(SweepAxis(step_id="mask", param="low", values=tuple(range(101))),),
    )
    issues = validate_run_config(config, pipeline, SNAPSHOT, source_count=100)
    assert "run.too_many_experiments" in [issue.code for issue in issues]
    with pytest.raises(ValidationError) as caught:
        build_plan(config, pipeline, SNAPSHOT, source_count=100)
    assert caught.value.issues[0].code == "run.too_many_experiments"


def test_plan_seeds_are_reproducible_end_to_end() -> None:
    snapshot = snapshot_with_seeded()
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="perm", block="channel_permute"),
            PipelineStep(id="seeded", block="contract_seeded"),
        )
    )
    config = RunConfig(
        mode="grid",
        axes=(SweepAxis(step_id="perm", param="order", values=("rgb", "bgr", "grb", "brg")),),
        master_seed=11,
    )
    plan = build_plan(config, pipeline, snapshot, source_count=1)
    source = Image(fixtures.rgb_gradient())
    first = [
        execute_pipeline(pipeline, snapshot, source, None, spec).layers[0].image.rgb
        for spec in plan
    ]
    second = [
        execute_pipeline(pipeline, snapshot, source, None, spec).layers[0].image.rgb
        for spec in plan
    ]
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert len({int(image[0, 0, 0]) for image in first}) > 1


def test_random_plan_executes_with_sampled_values() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="mask", block="channel_mask", params={"channel": "R"}),
            PipelineStep(id="perm", block="channel_permute"),
        )
    )
    config = RunConfig(mode="random", batch_size=4, sampled=(("mask", "low"),), master_seed=5)
    plan = build_plan(config, pipeline, SNAPSHOT, source_count=2)
    source = Image(fixtures.rgb_gradient())
    for spec in plan:
        output = execute_pipeline(pipeline, SNAPSHOT, source, None, spec)
        assert len(output.layers) == 1
        assert output.layers[0].image.size == source.size
