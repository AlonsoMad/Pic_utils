"""Preview service: a persistent preview process (architecture §14, AD-22).

Qt-free. The server process loads the source, downscales it to the proxy
size, executes the pipeline, and returns PNG bytes. The client spawns the
process, forwards requests, and delivers responses from a reader thread —
the GUI thread never waits on the process.

Lifecycle: one persistent process; ``restart()`` replaces it (block reload,
crash); a crashed preview never affects the UI or a batch. Requests carry a
monotonic ``generation``; the caller drops stale responses (latest wins).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pic_utils.core.image import Image
from pic_utils.io.loader import load_image
from pic_utils.pipeline.execute import execute_pipeline
from pic_utils.pipeline.model import ExperimentSpec

__all__ = ["PreviewClient", "PreviewRequest", "PreviewResponse", "serve_request"]

_STOP = None  # sentinel: shut the server down


@dataclass(frozen=True)
class PreviewRequest:
    """One preview render request (§14)."""

    generation: int
    pipeline: dict  # serialized Pipeline
    snapshot_dir: str | None  # live user blocks folder
    source_path: str
    second_path: str | None
    proxy_long_side: int
    seed: int


@dataclass(frozen=True)
class PreviewResponse:
    """One preview render result (§14)."""

    generation: int
    png_bytes: bytes | None
    error: str | None
    elapsed_ms: int
    layers: int


def serve_request(request: PreviewRequest) -> PreviewResponse:
    """Execute one preview request; never raises (errors go in the response)."""
    started = time.perf_counter()
    try:
        png, layers = _render(request)
    except Exception as error:
        return PreviewResponse(
            generation=request.generation,
            png_bytes=None,
            error=f"{type(error).__name__}: {error}",
            elapsed_ms=int((time.perf_counter() - started) * 1000.0),
            layers=0,
        )
    return PreviewResponse(
        generation=request.generation,
        png_bytes=png,
        error=None,
        elapsed_ms=int((time.perf_counter() - started) * 1000.0),
        layers=layers,
    )


# --- server process ---------------------------------------------------------

_source_cache: dict[str, tuple[float, Image]] = {}
_proxy_cache: dict[str, tuple[float, Image]] = {}
_snapshot_cache: dict[str, tuple[float, Any]] = {}


def _load_cached(path: str, cache: dict[str, tuple[float, Image]], scale_to: int | None) -> Image:
    """Load (and optionally downscale) an image, cached by path+mtime."""
    from pathlib import Path

    mtime = Path(path).stat().st_mtime
    cached = cache.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    image = load_image(path)
    if scale_to is not None:
        image = _proxy_image(image, scale_to)
    cache[path] = (mtime, image)
    return image


def _proxy_image(image: Image, long_side: int) -> Image:
    """Downscale so the long side is ``long_side`` (never upscale)."""
    long_side_current = max(image.width, image.height)
    if long_side_current <= long_side:
        return image
    scale = long_side / long_side_current
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resized(width, height)


def _render(request: PreviewRequest) -> tuple[bytes, int]:

    from pic_utils.pipeline.model import Pipeline

    snapshot = _snapshot_for(request.snapshot_dir)
    source = _load_cached(request.source_path, _proxy_cache, request.proxy_long_side)
    second = (
        None
        if request.second_path is None
        else _load_cached(request.second_path, _proxy_cache, request.proxy_long_side)
    )
    pipeline = Pipeline.from_dict(request.pipeline)
    experiment = ExperimentSpec(
        index=0, source_index=0, params={}, seed=request.seed, label="preview"
    )
    output = execute_pipeline(pipeline, snapshot, source, second, experiment)
    first = output.layers[0].image
    return _encode_png(first), len(output.layers)


def _encode_png(image: Image) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".png", image.rgb)
    if not ok:  # pragma: no cover - imencode only fails on bad arrays
        raise ValueError("could not encode the preview to PNG")
    return buffer.tobytes()


def _snapshot_for(snapshot_dir: str | None) -> Any:
    """Build (and cache) the block snapshot for the live blocks folder."""
    from pathlib import Path

    if snapshot_dir is None:
        return _builtin_snapshot()
    folder = Path(snapshot_dir)
    mtime = folder.stat().st_mtime
    cached = _snapshot_cache.get(snapshot_dir)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    snapshot = _build_snapshot(folder)
    _snapshot_cache[snapshot_dir] = (mtime, snapshot)
    return snapshot


def _builtin_snapshot() -> Any:
    from types import MappingProxyType

    from pic_utils.blocks.snapshot import BlockSnapshot

    definitions, _errors = _collect_definitions(None)
    return BlockSnapshot(defs=MappingProxyType(definitions), hash="")


def _build_snapshot(folder) -> Any:
    from pathlib import Path
    from types import MappingProxyType

    from pic_utils.blocks.snapshot import BlockSnapshot, snapshot_hash
    from pic_utils.core.paths import content_hash

    definitions, _errors = _collect_definitions(folder)
    file_hashes: dict[Path, str] = {}
    for path in folder.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        file_hashes[path] = content_hash(path)
    digest = snapshot_hash(definitions, file_hashes, user_blocks_dir=folder)
    return BlockSnapshot(defs=MappingProxyType(definitions), hash=digest)


def _collect_definitions(folder) -> tuple[dict, list]:
    """Built-ins plus user blocks; Qt-free (the worker's algorithm, §8.2)."""
    import importlib
    import pkgutil

    from pic_utils.blocks import builtin as builtin_package
    from pic_utils.blocks.api import BlockLoadError
    from pic_utils.blocks.userblocks import collect_block_defs, load_user_module

    definitions: dict[str, object] = {}
    errors: list[BlockLoadError] = []
    for module_info in pkgutil.iter_modules(builtin_package.__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{builtin_package.__name__}.{module_info.name}")
        except Exception:
            continue
        for definition in collect_block_defs(module, source=None, content_hash=""):
            definitions.setdefault(definition.name, definition)

    if folder is not None:
        for path in sorted(folder.rglob("*.py")):
            if path.name.startswith("_") or "__pycache__" in path.parts:
                continue
            try:
                module = load_user_module(path, folder)
            except Exception:
                continue
            for definition in collect_block_defs(module, source=path, content_hash=""):
                definitions.setdefault(definition.name, definition)
    return definitions, errors


def _serve(request_queue, response_queue, ready: threading.Event) -> None:
    """Server loop: one request at a time, results on the response queue."""

    ready.set()
    while True:
        request = request_queue.get()
        if request is _STOP:
            return
        response = serve_request(request)
        response_queue.put(response)


# --- client -------------------------------------------------------------------


class PreviewClient:
    """Owns the preview process; delivers responses on a reader thread."""

    def __init__(self, on_response: Callable[[PreviewResponse], None]) -> None:
        self._on_response = on_response
        self._request_queue: Any = None
        self._response_queue: Any = None
        self._process: Any = None
        self._reader: threading.Thread | None = None
        self._ready: Any = None
        self._lock = threading.Lock()
        self._closed = False

    def start(self) -> None:
        """Spawn the preview process and the reader thread."""
        import multiprocessing as mp

        context = mp.get_context("spawn")
        self._request_queue = context.Queue()
        self._response_queue = context.Queue()
        self._ready = context.Event()
        self._process = context.Process(
            target=_serve,
            args=(self._request_queue, self._response_queue, self._ready),
            daemon=True,
        )
        self._process.start()
        self._reader = threading.Thread(
            target=self._read_loop, name="pic-utils-preview", daemon=True
        )
        self._reader.start()

    def submit(self, request: PreviewRequest) -> None:
        """Queue one request; restarts the process first if it died."""
        with self._lock:
            if self._closed:
                return
            if self._process is None or not self._process.is_alive():
                self._shutdown_process()
                self.start()
            assert self._request_queue is not None
            self._request_queue.put(request)

    def restart(self) -> None:
        """Drop pending work and start a fresh process (block reload)."""
        with self._lock:
            if self._closed:
                return
            self._shutdown_process()
            self.start()

    def shutdown(self) -> None:
        """Stop the reader and the process."""
        with self._lock:
            self._closed = True
            self._shutdown_process()

    def _shutdown_process(self) -> None:
        if self._request_queue is not None:
            with contextlib.suppress(Exception):
                self._request_queue.put_nowait(_STOP)
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        self._process = None
        self._request_queue = None

    def _read_loop(self) -> None:
        while not self._closed:
            try:
                assert self._response_queue is not None
                response = self._response_queue.get(timeout=0.5)
            except Exception:
                if self._process is None or not self._process.is_alive():
                    return
                continue
            self._on_response(response)
