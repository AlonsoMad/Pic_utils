"""Performance validation (P7-03, AC-11).

Run explicitly:

    uv run pytest tests/perf -q -m slow -s

The numbers printed by these tests are the evidence recorded in
``docs/perf.md``. The 48 MP batch test uses the real spawned worker pool, the
real shared-memory source, and the real PNG writer.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from perf_support import (
    RssSampler,
    build_request,
    make_48mp_source,
    run_to_completion,
)
from pytestqt.qtbot import QtBot

from pic_utils.runner.preview import PreviewRequest, serve_request
from pic_utils.runner.service import RunService

pytestmark = pytest.mark.slow

_PREVIEW_PIPELINE = {
    "steps": [{"id": "s1", "block": "channel_permute", "params": {"order": "bgr"}}]
}


def test_preview_latency_at_1024px(tmp_path: Path) -> None:
    source = make_48mp_source(tmp_path / "source.png")
    request = PreviewRequest(
        generation=1,
        pipeline=_PREVIEW_PIPELINE,
        snapshot_dir=None,
        source_path=str(source),
        second_path=None,
        proxy_long_side=1024,
        seed=7,
    )

    cold = serve_request(request)
    assert cold.error is None
    warm = serve_request(replace(request, generation=2))
    assert warm.error is None

    print(
        f"\npreview: cold={cold.elapsed_ms} ms warm={warm.elapsed_ms} ms proxy={1024}px source=48MP"
    )
    assert warm.elapsed_ms < 200


def test_default_batch_peak_rss_under_8gb(qtbot: QtBot, tmp_path: Path) -> None:
    source = make_48mp_source(tmp_path / "source.png")
    request = build_request(tmp_path / "results", source, experiments=32)
    service = RunService()

    with RssSampler() as sampler:
        summary = run_to_completion(service, request, qtbot.waitUntil, timeout_ms=900_000)

    assert summary.status == "completed"
    assert summary.completed == 32
    assert summary.failed == 0
    print(
        f"\nbatch: experiments={summary.completed} workers=2 "
        f"peak_rss={sampler.peak_mb} MiB ceiling=8192 MiB "
        f"elapsed={summary.elapsed_ms} ms"
    )
    assert sampler.peak_mb < 8192
