"""Theme application (architecture §16: ``ui/theme`` = system | light | dark).

``system`` leaves the platform palette untouched. ``light``/``dark`` install
an explicit Fusion palette so the app looks the same on every platform.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = ["apply_theme"]

_DARK_WINDOW = QColor(53, 53, 53)
_DARK_BASE = QColor(35, 35, 35)
_DARK_TEXT = QColor(230, 230, 230)
_DARK_BUTTON = QColor(66, 66, 66)
_DARK_DISABLED_TEXT = QColor(110, 110, 110)
_DARK_HIGHLIGHT = QColor(76, 163, 224)
_LIGHT_WINDOW = QColor(240, 240, 240)
_LIGHT_TEXT = QColor(30, 30, 30)


def _palette(window: QColor, base: QColor, text: QColor, button: QColor) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, window)
    palette.setColor(QPalette.ColorRole.ToolTipBase, base)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, button)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 90, 90))
    palette.setColor(QPalette.ColorRole.Highlight, _DARK_HIGHLIGHT)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    placeholder = text.lighter(140) if window.lightness() < 128 else text.darker(140)
    palette.setColor(QPalette.ColorRole.PlaceholderText, placeholder)
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, _DARK_DISABLED_TEXT)
    return palette


def apply_theme(app: QApplication, theme: str) -> None:
    """Apply ``system``, ``light``, or ``dark`` to ``app``."""
    app.setStyle("Fusion")
    if theme == "dark":
        app.setPalette(_palette(_DARK_WINDOW, _DARK_BASE, _DARK_TEXT, _DARK_BUTTON))
    elif theme == "light":
        app.setPalette(_palette(_LIGHT_WINDOW, QColor(255, 255, 255), _LIGHT_TEXT, _LIGHT_WINDOW))
    else:
        app.setPalette(app.style().standardPalette())
