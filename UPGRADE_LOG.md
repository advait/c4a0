# Dependency Upgrade Log

**Date:** 2026-06-29
**Project:** c4a0
**Language:** Python + Rust
**Manifest:** pyproject.toml, uv.lock, rust/Cargo.toml, rust/Cargo.lock

---

## Summary

| Metric | Count |
|--------|-------|
| **Total dependencies considered** | 45 |
| **Updated** | 7 |
| **Skipped** | 0 |
| **Failed (rolled back)** | 0 |
| **Requires attention** | 0 |

---

## Strategy

Follow `library-updater`: update one dependency at a time, research before each update, test after each update, rollback after three failed fix attempts, and ask before refactoring that touches more than 10 files.

Slices:
1. Dev/tooling dependencies
2. Small Python libraries
3. Small Rust crates
4. Medium-risk Rust crates
5. PyO3/Rust NumPy/Python NumPy native-boundary milestone
6. Torch/PyTorch Lightning/TorchMetrics ML milestone
7. Notebook/experiment tooling
8. Toolchains

---

## Successfully Updated

### pyright: 1.1.396 → 1.1.411

**Changelog:** GitHub releases / PyPI metadata reviewed via web search.

**Breaking changes:** None found in release-note search for 1.1.396 → 1.1.411.

**Notable changes:** Type checker bug fixes and inference/narrowing improvements across patch releases.

**Migration applied:** Raised dev dependency lower bound to `pyright>=1.1.411`; `uv.lock` resolved `pyright==1.1.411`. First `uv add` also normalized `uv.lock` to the newer uv lock revision with upload-time metadata.

**Tests:** ✓ `mise run typecheck` passed with 0 errors, 0 warnings.

### ruff: 0.9.9 → 0.15.20

**Changelog:** Ruff release notes/versioning docs reviewed.

**Breaking changes:** Ruff is pre-1.0; its versioning docs state minor bumps may include lint/config behavior changes. No project config migration was required.

**Notable changes:** Newer linter release with potential rule behavior additions/changes; existing project still passes.

**Migration applied:** Raised dev dependency lower bound to `ruff>=0.15.20`; `uv.lock` resolved `ruff==0.15.20`.

**Tests:** ✓ `mise run lint` passed.

### maturin: 1.8.2 → 1.14.1

**Changelog:** Official maturin changelog / release pages reviewed.

**Breaking changes:** None explicitly identified for the upgrade range.

**Notable changes:** Newer maturin build behavior uses `rust/target/maturin` for the extension build artifact in this project; build/install still works. Existing patchelf/rpath warning remains non-fatal.

**Migration applied:** Raised project dependency to `maturin>=1.14.1` and build backend requirement to `maturin>=1.14.1,<2`; `uv.lock` resolved `maturin==1.14.1`.

**Tests:** ✓ `mise run build` passed; ✓ `mise run test:python` passed.

### pytest: 8.3.5 → 9.1.1

**Changelog:** Official pytest 9 changelog reviewed.

**Breaking changes:** Pytest 9 drops Python 3.9 support; `PytestRemovedIn9Warning` deprecations become errors/default removals; overlapping duplicate path CLI arguments are normalized differently. This project runs Python 3.11 and uses simple `pytest -q` invocation.

**Notable changes:** Native TOML configuration support and strict-mode additions. Existing `[tool.pytest.ini_options]` remains supported.

**Migration applied:** Raised dev dependency lower bound to `pytest>=9.1.1`; `uv.lock` resolved `pytest==9.1.1`.

**Tests:** ✓ `mise run test:python` passed.

### pytest-asyncio: 0.25.3 → 1.4.0

**Changelog:** Official pytest-asyncio changelog / migration docs reviewed.

**Breaking changes:** Notable async fixture/event-loop behavior changes in the 0.25+ and 1.x line; pytest compatibility requirements increased. No project test changes were required.

**Notable changes:** The previous `asyncio_default_fixture_loop_scope` deprecation warning is gone after the update.

**Migration applied:** Raised dev dependency lower bound to `pytest-asyncio>=1.4.0`; `uv.lock` resolved `pytest-asyncio==1.4.0`.

**Tests:** ✓ `mise run test:python` passed.

### einops: 0.8.1 → 0.8.2

**Changelog:** PyPI/changelog search reviewed.

**Breaking changes:** None found for core `rearrange` usage; Python minimum remains compatible with project Python 3.11.

**Migration applied:** Raised dependency lower bound to `einops>=0.8.2`; `uv.lock` resolved `einops==0.8.2`.

**Tests:** ✓ `mise run test:python` passed.

### tqdm: 4.67.1 → 4.68.3

**Changelog:** Release-history/PyPI search reviewed.

**Breaking changes:** None found in available release-history search for this project’s progress-bar usage.

**Migration applied:** Raised dependency lower bound to `tqdm>=4.68.3`; `uv.lock` resolved `tqdm==4.68.3`.

**Tests:** ✓ `mise run test:python` passed.

---

## Skipped

_To be filled._

---

## Failed Updates (Rolled Back)

_To be filled._

---

## Requires Attention

_To be filled._

---

## Deprecation Warnings Fixed

| Package | Warning | Fix Applied |
|---------|---------|-------------|
| uv/pyproject | `tool.uv.dev-dependencies` is deprecated | Moved dev dependencies to `[dependency-groups].dev`; `mise run ci` passed. |
| pytest-asyncio | `asyncio_default_fixture_loop_scope` unset warning | Updating `pytest-asyncio` from 0.25.3 to 1.4.0 removed the warning under current tests. |

---

## Security Notes

_To be filled after audits._

---

## Commands Used

```bash
mise exec -- uv lock --check
mise run ci
mise exec -- uv add --dev pyright --upgrade-package pyright
mise exec -- uv add --dev 'pyright>=1.1.411'
mise run typecheck
mise exec -- uv add --dev ruff --upgrade-package ruff
mise exec -- uv add --dev 'ruff>=0.15.20'
mise run lint
mise exec -- uv add maturin --upgrade-package maturin
mise exec -- uv add 'maturin>=1.14.1'
mise exec -- uv lock
mise run build
mise run test:python
mise exec -- uv add --dev pytest --upgrade-package pytest
mise exec -- uv add --dev 'pytest>=9.1.1'
mise exec -- uv add --dev 'pytest-asyncio>=1.4.0'
mise run test:python
mise exec -- uv add einops --upgrade-package einops
mise exec -- uv add 'einops>=0.8.2'
mise run test:python
mise exec -- uv add tqdm --upgrade-package tqdm
mise exec -- uv add 'tqdm>=4.68.3'
mise run test:python
```

---

## Notes

- Baseline CI passed locally and remotely on `chore/mise-ci-packaging` before dependency work began.
- Dependency upgrades are being performed on `chore/dependency-upgrades`.
- Slice 1 (dev/tooling) completed: `pyright`, `ruff`, `maturin`, `pytest`, and `pytest-asyncio` updated; final `mise run ci` passed.
