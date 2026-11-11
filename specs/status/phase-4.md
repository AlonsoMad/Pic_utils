---
phase: 4
name: Pipeline & Planning
status: passed
tag: phase-4
date: 2026-11-11
---

## Tasks merged

- P3-01
- P3-02
- P3-03
- P3-04
- P3-05

## Gate commands and results

Run on `main` after the five task commits:

```
$ uv sync --extra dev
Resolved 44 packages in 3ms
Checked 43 packages in 0.75ms

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
49 files already formatted

$ uv run mypy src
Success: no issues found in 28 source files

$ uv run pytest
380 passed in 7.61s

$ uv run pytest tests/pipeline
120 passed in 1.94s
```

CI: GitHub Actions run on `main` (`ubuntu-latest`, `windows-latest`;
Python 3.12) — recorded below after the run completes.

## Acceptance checklist

- [x] `Pipeline`/`PipelineStep` serialize and deserialize losslessly; step
      IDs are stable (`tests/pipeline/test_model.py`).
- [x] Every validation code in architecture §6.3 has a test: unknown block,
      duplicate ID, LAYERS not last, trailing mask, no output, missing/unknown
      param, type/range/choice, unbound/forward/bypassed mask, type mismatch,
      missing second image, API version, run config errors, source errors
      (`tests/pipeline/test_validate.py`).
- [x] Executor: chain order; mask side inputs; bypass; LAYERS terminal; alpha
      propagation; per-step seed derivation; block exception → experiment
      failure; invalid output → `block.invalid_output`
      (`tests/pipeline/test_execute.py`).
- [x] Planner: grid (empty product = 1 per source), random, hybrid; per-source
      distribution; deterministic seeds; 10,000 guard; disk preflight
      (`tests/pipeline/test_plan.py`).
- [x] `pipeline/` imports no Qt and no `io/` (`tests/pipeline/test_contract.py`).
- [x] `pytest tests/pipeline` green; mypy clean.
- [x] `specs/status/phase-4.md` written; tag `phase-4`.

## Known issues / follow-ups

None.
