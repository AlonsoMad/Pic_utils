# Performance validation

Evidence recorded on 2026-11-11.

## Methodology

- **Preview latency** — `serve_request` on a 48 MP source with a 1024 px proxy. The first call loads and proxies the source; the second call uses cached data. The measured `elapsed_ms` is the time spent in the preview process, excluding IPC and process startup.
- **Memory ceiling** — a 32-experiment batch at 48 MP with 2 workers. The peak RSS of the app process tree is measured by sampling every 50 ms. The ceiling is 8192 MB.

## Commands

```bash
uv run pytest tests/perf -q -m slow -s
```

## Evidence

```
preview: cold=2799 ms warm=26 ms proxy=1024px source=48MP
batch: experiments=32 workers=2 peak_rss=2163 MiB ceiling=8192 MiB elapsed=37236 ms
```

## Notes

- The preview latency includes the time to load and proxy the 48 MP source on the first call. Subsequent calls use cached data.
- The batch memory measurement includes the peak RSS of the app process tree, including the spawned workers and the shared memory sources.
- The batch elapsed time includes the time to load the sources, execute the pipeline, and write the results.
