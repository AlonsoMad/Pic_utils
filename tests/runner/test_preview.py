"""Preview service and pane tests (P5-05)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pic_utils.runner.preview import PreviewRequest, PreviewResponse, serve_request
from pic_utils.ui.lab.preview_pane import PreviewPane


@pytest.fixture
def source_png(tmp_path: Path) -> Path:
    """A small gradient PNG on disk."""
    import cv2

    gradient = np.zeros((64, 96, 3), dtype=np.uint8)
    gradient[..., 0] = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (64, 1))
    gradient[..., 1] = np.linspace(0, 255, 64, dtype=np.uint8)[:, None]
    path = tmp_path / "source.png"
    cv2.imwrite(str(path), gradient)
    return path


def _request(source: Path, pipeline: dict, generation: int = 1) -> PreviewRequest:
    return PreviewRequest(
        generation=generation,
        pipeline=pipeline,
        snapshot_dir=None,
        source_path=str(source),
        second_path=None,
        proxy_long_side=32,
        seed=1,
    )


class TestServeRequest:
    def test_renders_png_at_proxy_size(self, source_png: Path) -> None:
        response = serve_request(_request(source_png, {"steps": []}))
        assert response.error is None
        assert response.png_bytes is not None
        assert response.png_bytes.startswith(b"\x89PNG")
        assert response.elapsed_ms >= 0

    def test_proxy_long_side_is_respected(self, source_png: Path) -> None:
        import cv2
        import numpy as np

        response = serve_request(_request(source_png, {"steps": []}))
        assert response.png_bytes is not None
        array = cv2.imdecode(np.frombuffer(response.png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert max(array.shape[0], array.shape[1]) == 32

    def test_pipeline_applied(self, source_png: Path) -> None:
        import cv2
        import numpy as np

        pipeline = {"steps": [{"id": "a", "block": "channel_permute", "params": {"order": "bgr"}}]}
        response = serve_request(_request(source_png, pipeline))
        assert response.error is None
        array = cv2.imdecode(np.frombuffer(response.png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        # The source's blue channel ramps horizontally; after a bgr permute it
        # lands on the red channel, which cv2 reads back in slot 0.
        assert array[10, -1, 0] > array[10, -1, 2]

    def test_error_response_carries_message(self, tmp_path: Path) -> None:
        response = serve_request(_request(tmp_path / "missing.png", {"steps": []}))
        assert response.png_bytes is None
        assert response.error is not None
        assert "missing.png" in response.error

    def test_block_error_reported(self, source_png: Path) -> None:
        pipeline = {"steps": [{"id": "a", "block": "nonexistent_block"}]}
        response = serve_request(_request(source_png, pipeline))
        assert response.error is not None

    def test_layers_pipeline_reports_count(self, source_png: Path) -> None:
        pipeline = {"steps": [{"id": "a", "block": "segment"}]}
        response = serve_request(_request(source_png, pipeline))
        assert response.error is None
        assert response.layers >= 1


@pytest.fixture
def pane(qtbot: object):
    """A preview pane whose client is shut down at teardown (no leaks)."""
    widget = PreviewPane(snapshot_dir_provider=lambda: None, debounce_ms=10)
    qtbot.addWidget(widget)  # type: ignore[attr-defined]
    yield widget
    widget.shutdown()


class TestPreviewPane:
    def test_request_renders_after_debounce(
        self, pane, qtbot, source_png: Path
    ) -> None:
        updates: list[tuple[int, int]] = []
        pane.preview_updated.connect(lambda g, ms: updates.append((g, ms)))
        pane.request({"steps": [{"id": "a", "block": "channel_permute"}]}, str(source_png))
        qtbot.waitUntil(lambda: bool(updates), timeout=10_000)
        assert updates[0][0] == 1

    def test_no_source_shows_placeholder(self, pane) -> None:
        pane.request({"steps": []}, None)
        assert pane._stack.currentWidget() is pane._image

    def test_stale_generation_dropped(self, pane) -> None:
        updates: list[int] = []
        pane.preview_updated.connect(lambda g, _ms: updates.append(g))
        stale = PreviewResponse(generation=1, png_bytes=b"x", error=None, elapsed_ms=1, layers=1)
        pane._generation = 2
        pane._on_response(stale)
        assert updates == []

    def test_error_displayed_non_modally(self, pane) -> None:
        failures: list[str] = []
        pane.preview_failed.connect(failures.append)
        error = PreviewResponse(generation=1, png_bytes=None, error="boom", elapsed_ms=1, layers=0)
        pane._generation = 1
        pane._on_response(error)
        assert pane._error.text() == "boom"
        assert not pane._error.isHidden()
        assert failures == ["boom"]

    def test_restart_replaces_client(self, pane) -> None:
        pane.restart()
        assert pane._client is not None
        pane.shutdown()
