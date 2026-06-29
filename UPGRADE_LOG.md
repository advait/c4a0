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
| **Updated** | 0 |
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

_To be filled as dependencies are updated._

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
```

---

## Notes

- Baseline CI passed locally and remotely on `chore/mise-ci-packaging` before dependency work began.
- Dependency upgrades are being performed on `chore/dependency-upgrades`.
