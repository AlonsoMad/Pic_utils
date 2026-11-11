"""Language switching without restart (architecture §15, AC-12).

``LanguageManager`` owns the ``QTranslator`` for the app catalog
``resources/i18n/pic_utils_<lang>.qm``. Every widget class implements
``retranslate()`` and connects to :attr:`language_changed`, so switching the
language re-resolves every user-visible string immediately.
"""

from __future__ import annotations

import importlib.resources

from PySide6.QtCore import QCoreApplication, QLocale, QObject, QTranslator, Signal
from PySide6.QtWidgets import QWidget

from pic_utils.settings import Settings

__all__ = ["LanguageManager", "retranslate_tree"]

_CATALOG_PACKAGE = "pic_utils.resources.i18n"
_CATALOG_TEMPLATE = "pic_utils_{}.qm"


def _resolve(language: str) -> str:
    """Map a settings value to a catalog code; ``system`` follows the locale."""
    if language == "system":
        code = QLocale.system().name()[:2]
        return code if code in ("en", "es") else "en"
    return language


def retranslate_tree(widget: QWidget) -> None:
    """Call ``retranslate()`` on every widget child, depth-first.

    The passed widget's own ``retranslate()`` is NOT called — the caller does
    that (calling it here would recurse when a page delegates to this helper).
    """
    for child in widget.findChildren(QWidget):
        retranslate = getattr(child, "retranslate", None)
        if callable(retranslate):
            retranslate()


class LanguageManager(QObject):
    """Installs and swaps the app translator; emits :attr:`language_changed`."""

    language_changed = Signal(str)  # resolved code: "en" | "es"

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._translator: QTranslator | None = None
        self._code = "en"
        self.set_language(settings.language)

    @property
    def language(self) -> str:
        """The resolved catalog code currently installed."""
        return self._code

    @property
    def setting(self) -> str:
        """The raw settings value (``system``, ``en``, or ``es``)."""
        return self._settings.language

    def set_language(self, language: str) -> None:
        """Install the catalog for ``language`` and notify the UI."""
        code = _resolve(language)
        app = QCoreApplication.instance()
        if self._translator is not None and app is not None:
            app.removeTranslator(self._translator)
            self._translator = None
        if code != "en":
            self._translator = self._install(code)
        changed = code != self._code
        self._code = code
        if changed:
            self.language_changed.emit(code)

    @staticmethod
    def _install(code: str) -> QTranslator | None:
        """Load and install ``pic_utils_<code>.qm``; return the translator."""
        app = QCoreApplication.instance()
        if app is None:
            return None
        source = importlib.resources.files(_CATALOG_PACKAGE) / _CATALOG_TEMPLATE.format(code)
        with importlib.resources.as_file(source) as path:
            translator = QTranslator(app)
            if not translator.load(str(path)):
                return None
        app.installTranslator(translator)
        return translator
