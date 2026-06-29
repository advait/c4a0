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
| **Updated** | 2 |
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
```

---

## Notes

- Baseline CI passed locally and remotely on `chore/mise-ci-packaging` before dependency work began.
- Dependency upgrades are being performed on `chore/dependency-upgrades`.
