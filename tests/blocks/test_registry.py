"""Behavior tests for the block registry (P2-02, architecture §5.4-§5.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pic_utils.blocks.registry import BlockRegistry, ReloadReport
from pic_utils.core.image import Image

BUILTIN_NAMES = {
    "brightness_mask",
    "channel_mask",
    "channel_permute",
    "color_range_mask",
    "conditional_overlay",
    "double_exposure",
    "overlay",
    "permutation_grid",
    "segment",
}


def write_block(
    folder: Path,
    filename: str = "user.py",
    *,
    amount: int = 10,
    block_name: str = "user_brighten",
) -> Path:
    path = folder / filename
    path.write_text(
        f'''
from __future__ import annotations

from typing import Annotated

import numpy as np

from pic_utils.blocks import IntRange, OutputKind, block
from pic_utils.core.image import Image


@block(
    name="{block_name}",
    category="user",
    description="Brighten by a constant.",
    output=OutputKind.IMAGE,
)
def {block_name}(
    image: Image,
    amount: Annotated[int, IntRange(0, 255)] = {amount},
) -> Image:
    rgb = np.clip(image.rgb.astype(np.int16) + amount, 0, 255).astype(np.uint8)
    return Image(rgb, image.alpha)
''',
        encoding="utf-8",
    )
    return path


def test_builtins_are_discovered() -> None:
    snapshot = BlockRegistry().snapshot()
    assert set(snapshot) >= BUILTIN_NAMES
    definition = snapshot.get("channel_permute")
    assert definition.source is None
    assert definition.content_hash == ""


def test_user_block_is_loaded(tmp_path: Path) -> None:
    path = write_block(tmp_path)
    registry = BlockRegistry(tmp_path)
    snapshot = registry.snapshot()
    assert "user_brighten" in snapshot
    definition = snapshot.get("user_brighten")
    assert definition.source == path
    assert definition.content_hash
    assert definition.module == "pic_utils_userblocks.user"


def test_user_block_executes(tmp_path: Path) -> None:
    write_block(tmp_path, amount=10)
    definition = BlockRegistry(tmp_path).snapshot().get("user_brighten")
    image = Image.from_rgb(np.full((2, 2, 3), 100, np.uint8))
    result = definition.fn(image)
    assert isinstance(result, Image)
    assert int(result.rgb[0, 0, 0]) == 110


def test_reload_reports_loaded_updated_removed(tmp_path: Path) -> None:
    path = write_block(tmp_path)
    registry = BlockRegistry(tmp_path)
    report = registry.reload()
    assert report.loaded == ()
    assert report.updated == ()
    assert report.removed == ()

    write_block(tmp_path, amount=20)
    report = registry.reload()
    assert report.updated == ("user_brighten",)

    path.unlink()
    report = registry.reload()
    assert report.removed == ("user_brighten",)


def test_syntax_error_keeps_the_last_good_block(tmp_path: Path) -> None:
    path = write_block(tmp_path)
    registry = BlockRegistry(tmp_path)
    before = registry.snapshot().get("user_brighten")
    path.write_text("def broken(:\n", encoding="utf-8")
    report = registry.reload()
    assert report.errors
    after = registry.snapshot().get("user_brighten")
    assert after.content_hash == before.content_hash
    assert after.fn is before.fn


def test_duplicate_user_names_are_rejected(tmp_path: Path) -> None:
    write_block(tmp_path, "a.py", block_name="duplicate")
    write_block(tmp_path, "b.py", block_name="duplicate")
    registry = BlockRegistry(tmp_path)
    snapshot = registry.snapshot()
    assert snapshot.get("duplicate").source == tmp_path / "a.py"
    report = registry.reload()
    assert any("duplicate" in str(error) for error in report.errors)


def test_shadowing_a_builtin_is_rejected(tmp_path: Path) -> None:
    write_block(tmp_path, block_name="channel_permute")
    registry = BlockRegistry(tmp_path)
    snapshot = registry.snapshot()
    assert snapshot.get("channel_permute").source is None
    report = registry.reload()
    assert any("duplicate" in str(error) for error in report.errors)


def test_helper_modules_reload(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("AMOUNT = 10\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        """
from __future__ import annotations

from typing import Annotated

import numpy as np

import helper

from pic_utils.blocks import IntRange, OutputKind, block
from pic_utils.core.image import Image


@block(
    name="helper_block",
    category="user",
    description="Uses a helper module.",
    output=OutputKind.IMAGE,
)
def helper_block(
    image: Image,
    amount: Annotated[int, IntRange(0, 255)] = 10,
) -> Image:
    rgb = np.clip(image.rgb.astype(np.int16) + helper.AMOUNT, 0, 255).astype(np.uint8)
    return Image(rgb, image.alpha)
""",
        encoding="utf-8",
    )
    registry = BlockRegistry(tmp_path)
    image = Image.from_rgb(np.zeros((2, 2, 3), np.uint8))
    definition = registry.snapshot().get("helper_block")
    assert int(definition.fn(image).rgb[0, 0, 0]) == 10

    helper.write_text("AMOUNT = 20\n", encoding="utf-8")
    registry.reload()
    definition = registry.snapshot().get("helper_block")
    assert int(definition.fn(image).rgb[0, 0, 0]) == 20


def test_snapshot_hash_is_stable_and_changes(tmp_path: Path) -> None:
    write_block(tmp_path)
    registry = BlockRegistry(tmp_path)
    first = registry.snapshot().hash
    registry.reload()
    assert registry.snapshot().hash == first

    write_block(tmp_path, amount=20)
    registry.reload()
    assert registry.snapshot().hash != first


def test_snapshot_is_immutable() -> None:
    snapshot = BlockRegistry().snapshot()
    with pytest.raises(TypeError):
        snapshot.defs["x"] = snapshot.get("channel_permute")  # type: ignore[index]


def test_changed_signal_emits_a_reload_report() -> None:
    registry = BlockRegistry()
    reports: list[ReloadReport] = []
    registry.changed.connect(reports.append)
    registry.reload()
    assert len(reports) == 1
    assert isinstance(reports[0], ReloadReport)


def test_get_unknown_block_raises_key_error() -> None:
    with pytest.raises(KeyError):
        BlockRegistry().snapshot().get("missing")


def test_set_user_blocks_dir(tmp_path: Path) -> None:
    registry = BlockRegistry()
    assert "user_brighten" not in registry.snapshot()
    write_block(tmp_path)
    registry.set_user_blocks_dir(tmp_path)
    assert "user_brighten" in registry.snapshot()
    registry.set_user_blocks_dir(None)
    assert "user_brighten" not in registry.snapshot()


def test_bad_signature_is_reported(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(
        """
from typing import Annotated

from pic_utils.blocks import IntRange, OutputKind, block
from pic_utils.core.image import Image


@block(name="bad", category="user", description="Missing default.", output=OutputKind.IMAGE)
def bad(image: Image, amount: Annotated[int, IntRange(0, 255)]) -> Image:
    return image
""",
        encoding="utf-8",
    )
    registry = BlockRegistry(tmp_path)
    assert "bad" not in registry.snapshot()
    report = registry.reload()
    assert any("must have a default" in str(error) for error in report.errors)


def test_include_builtins_false() -> None:
    assert len(BlockRegistry(include_builtins=False).snapshot()) == 0
