---
name: workspace
description: "Read before giving a parallel writer or experiment its own c4a0 checkout or run directory."
---

# c4a0 workspaces

Status: implemented

- Make a checkout with `git worktree add /tmp/c4a0-<name> <ref>` and remove it with
  `git worktree remove --force /tmp/c4a0-<name>`.
- In a new worktree, run `mise trust`, `mise run bootstrap` and `mise run build` first. A
  cold `uv sync` builds the Rust extension, which takes about 2 minutes. Each worktree gets
  its own `.venv` and `rust/target`.
- Give every concurrent training or convergence run its own `--base-dir` and
  `--solutions-path`. RocksDB locks `solutions.db` per process, and runs in one base dir
  resume each other's generations.
- Concurrent runs share 32 cores. Self-play uses all of them, so parallel arms slow each
  other. For timings you will compare, run arms one at a time, or pin each arm to its own
  cores with `taskset` and record that.
