---
phase: 0
name: Replace with the phase name
status: pending   # pending | in-progress | passed | failed
tag: phase-0
date: 2026-11-09
---

## Tasks merged

- P0-00
- P0-01

## Gate commands and results

```
$ uv sync --extra dev
...
$ uv run ruff check .
...
$ uv run ruff format --check .
...
$ uv run mypy src
...
$ uv run pytest -q
...
```

## Acceptance checklist

- [ ] Every phase task merged.
- [ ] CI green on Linux and Windows.
- [ ] Phase-specific criteria from `specs/implementation-plan.md` §8.
- [ ] No open blocking review findings.

## Known issues / follow-ups

None.
