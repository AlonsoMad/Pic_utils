"""Layered canvas backend: document model, assets, scene, and export.

The canvas backend is Qt-light: :mod:`pic_utils.canvas.model` and
:mod:`pic_utils.canvas.assets` are Qt-free; :mod:`pic_utils.canvas.render`,
:mod:`pic_utils.canvas.scene`, and :mod:`pic_utils.canvas.export` use Qt's
painting classes only (no widgets).
"""

from __future__ import annotations

__all__: list[str] = []
