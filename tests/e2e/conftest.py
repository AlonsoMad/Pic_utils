"""Offscreen Qt and shared fixtures for the end-to-end acceptance suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pic_utils.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: a temp blocks folder, results root, and undo limit."""
    settings = Settings(path=tmp_path / "settings.ini")
    settings.set_blocks_folder(tmp_path / "blocks")
    settings.set_results_root(tmp_path / "results")
    return settings


@pytest.fixture
def blocks_dir(settings: Settings) -> Path:
    """The blocks folder with the e2e test blocks."""
    directory = settings.blocks_folder
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "e2e_blocks.py").write_text(_E2E_BLOCKS, encoding="utf-8")
    return directory


_E2E_BLOCKS = '''"""End-to-end acceptance blocks."""

import time
from typing import Annotated

import numpy as np

from pic_utils.blocks import FloatRange, IntRange, OutputKind, Seed, block
from pic_utils.core.image import Image


@block(name="e2e_sweep", category="e2e", description="adds a fixed amount",
       output=OutputKind.IMAGE)
def e2e_sweep(image: Image, amount: Annotated[int, IntRange(0, 100, step=1)] = 0) -> Image:
    rgb = np.clip(image.rgb.astype(np.int16) + amount, 0, 255).astype(np.uint8)
    return Image(rgb, image.alpha)


@block(name="e2e_noise", category="e2e", description="seeded noise",
       output=OutputKind.IMAGE)
def e2e_noise(image: Image, seed: Annotated[int | None, Seed()] = None) -> Image:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=image.rgb.shape, dtype=np.uint8)
    return Image(noise, image.alpha)


@block(name="e2e_raise", category="e2e", description="raises when fail is set",
       output=OutputKind.IMAGE)
def e2e_raise(image: Image, fail: bool = True) -> Image:
    if fail:
        raise RuntimeError("e2e boom")
    return image


@block(name="e2e_sleep", category="e2e", description="sleeps",
       output=OutputKind.IMAGE)
def e2e_sleep(image: Image, seconds: Annotated[float, FloatRange(0.0, 10.0)] = 1.0) -> Image:
    time.sleep(seconds)
    return image
'''
