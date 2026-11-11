"""Preview pane: debounced, latest-wins preview rendering (§13.5, §14).

Owns a :class:`PreviewClient`. Any pipeline/param/source change calls
:meth:`request`; a debounce (default 150 ms) coalesces edits, each request
bumps the generation, and stale responses are dropped (latest wins). The
client is restarted after a block reload or a crash. The GUI thread never
waits: responses arrive via a queued signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

from pic_utils.runner.preview import PreviewClient, PreviewRequest, PreviewResponse

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["PreviewPane"]


__all__ = ["PreviewPane"]


class _ResponseBridge(QObject):
    """Delivers reader-thread responses to the GUI thread (queued)."""

    response = Signal(object)


class PreviewPane(QWidget):
    """Shows the pipeline applied to one source at proxy resolution."""

    #: Emitted after every applied response (tests and the status bar).
    preview_updated = Signal(int, int)  # generation, elapsed_ms
    #: Emitted when a render fails (non-modal error display, §13.6).
    preview_failed = Signal(str)

    def __init__(
        self,
        snapshot_dir_provider: Callable[[], str | None],
        debounce_ms: int = 150,
        proxy_long_side: int = 1024,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._snapshot_dir_provider = snapshot_dir_provider
        self._debounce_ms = debounce_ms
        self._proxy_long_side = proxy_long_side

        self._image = QLabel(self)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spinner = QLabel(self)
        self._spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error = QLabel(self)
        self._error.setWordWrap(True)
        self._error.hide()

        self._stack = QStackedLayout()
        self._stack.addWidget(self._image)
        self._stack.addWidget(self._spinner)

        layout = QVBoxLayout(self)
        layout.addLayout(self._stack, 1)
        layout.addWidget(self._error)

        self._generation = 0
        self._pipeline: dict | None = None
        self._source: str | None = None
        self._second: str | None = None
        self._seed: int = 0

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._send)

        self._bridge = _ResponseBridge(self)
        self._bridge.response.connect(self._on_response)
        self._client = PreviewClient(self._bridge.response.emit)
        self._client.start()
        self.retranslate()

    # --- public API ---------------------------------------------------------

    def request(
        self,
        pipeline: dict,
        source_path: str | None,
        second_path: str | None = None,
        seed: int = 0,
    ) -> None:
        """Queue a preview; debounced, latest-wins (§14)."""
        self._pipeline = pipeline
        self._source = source_path
        self._second = second_path
        self._seed = seed
        if source_path is None or not pipeline.get("steps"):
            self._show_placeholder()
            return
        self._debounce.start()

    def restart(self) -> None:
        """Restart the preview process (block reload or crash, §14)."""
        self._client.restart()

    def shutdown(self) -> None:
        self._client.shutdown()

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._spinner.setText(self.tr("Rendering preview…"))
        self._image.setText(self.tr("No preview"))

    # --- internals ------------------------------------------------------------

    def _show_placeholder(self) -> None:
        self._debounce.stop()
        self._generation += 1  # invalidate any in-flight render
        self._stack.setCurrentWidget(self._image)
        self._error.hide()

    def _send(self) -> None:
        if self._pipeline is None or self._source is None:
            return
        self._generation += 1
        request = PreviewRequest(
            generation=self._generation,
            pipeline=self._pipeline,
            snapshot_dir=self._snapshot_dir_provider(),
            source_path=self._source,
            second_path=self._second,
            proxy_long_side=self._proxy_long_side,
            seed=self._seed,
        )
        self._stack.setCurrentWidget(self._spinner)
        self._client.submit(request)

    def _on_response(self, response: PreviewResponse) -> None:
        if response.generation != self._generation:
            return  # stale generation: latest wins (§14)
        if response.error is not None:
            self._stack.setCurrentWidget(self._image)
            self._error.setText(response.error)
            self._error.show()
            self.preview_failed.emit(response.error)
            return
        assert response.png_bytes is not None
        image = QImage.fromData(response.png_bytes, b"PNG")
        if image.isNull():
            self._error.setText(self.tr("The preview returned an unreadable image"))
            self._error.show()
            return
        self._error.hide()
        self._stack.setCurrentWidget(self._image)
        self._image.setPixmap(QPixmap.fromImage(image))
        self.preview_updated.emit(response.generation, response.elapsed_ms)
