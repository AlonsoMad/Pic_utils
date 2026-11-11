---
phase: 1
name: Foundation & Guardrails
status: passed
tag: phase-1
date: 2026-11-11
---

## Tasks merged

- P0-01
- P0-02
- P0-03

## Gate commands and results

Run on `main` at the `phase-1` tag commit (`c02536e`):

```
$ uv sync --extra dev
Resolved 44 packages in 2ms
Checked 43 packages in 0.95ms

$ uv run ruff check .
All checks passed!

$ uv run ruff format --check .
11 files already formatted

$ uv run mypy src
Success: no issues found in 7 source files

$ uv run pytest
54 passed in 0.02s
```

Acceptance checks:

```
$ uv run python -c "import sys, pic_utils.legacy; assert 'matplotlib' not in sys.modules"
legacy import ok, no matplotlib

$ grep -R "plt.show" src/
(no matches)

$ grep -R "raise Exception" src/
(no matches)
```

CI: GitHub Actions run `42644763415` on `main` completed successfully
(`ubuntu-latest` + `windows-latest`, Python 3.12).

## Acceptance checklist

- [x] `uv sync --extra dev` succeeds from a clean clone.
- [x] `uv run ruff check .` and `uv run ruff format --check .` exit 0.
- [x] `uv run mypy src` exits 0.
- [x] `uv run pytest -q` exits 0 (54 passed).
- [x] CI runs on push/PR with `ubuntu-latest` and `windows-latest`, Python
      3.12, and is green.
- [x] `import pic_utils.legacy` succeeds without importing matplotlib.
- [x] `grep -R "plt.show" src/` returns nothing.
- [x] `core/jsonio.write_json_atomic` leaves the original file intact when
      the write fails (test proves it).
- [x] `core/errors.py` defines the full §18.1 hierarchy and §18.2 code
      table; no bare `raise Exception`.
- [x] `pyproject.toml` pins `requires-python = ">=3.12,<3.13"` and the
      §21.2 dependency set.
- [x] `specs/status/phase-1.md` written; tag `phase-1` created.
- [x] No open blocking review findings.

## Known issues / follow-ups

- The `phase-1` tag was created on `c02536e` before this status file
  existed. This record is a follow-up commit, so the tag does not include
  it; the tag target itself is the fully gated Phase 1 tree.
- `specs/status/P0-01.md` was missing its closing frontmatter delimiter;
  fixed in the same housekeeping commit as this file.
