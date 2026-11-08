---
name: workspace
description: "Read before giving a parallel writer or experiment its own c4a0 checkout or run directory."
---

# c4a0 workspaces

Status: implemented

- Make a checkout with `git worktree add /tmp/c4a0-<name> <ref>` and remove it with
  `git worktree remove --force /tmp/c4a0-<name>`.
- In a new worktree, run `mise trust` and `mise run build` first. A
  cold `uv sync` builds the Rust extension, which takes about 2 minutes. Each worktree gets
  its own `.venv` and `rust/target`.
- Give every concurrent training or convergence run its own `--base-dir`. Runs in one base
  dir resume each other's generations. A worktree needs the solver only to rebuild suites or
  label positions; symlink the main checkout's `solver/` into it (read-only use).
- Concurrent runs share 32 cores. Self-play uses all of them, so parallel arms slow each
  other. For timings you will compare, run arms one at a time, or pin each arm to its own
  cores with `taskset` and record that.

## The GPU machine (alapin)

- `ssh alapin` works through the `alapin` entry in `~/.ssh/config` (tailnet IP, because
  MagicDNS does not resolve in the dev container). Tailscale SSH may ask the maintainer to
  approve a login URL. Stop and ask. Do not retry in a loop.
- It is the maintainer's desktop. Write only under `~/agent-work/` there, never use `sudo`,
  and never kill processes you did not start. The desktop session holds about 7.5 GB of GPU
  memory, so keep total GPU use under about 17 GB.
- `~/agent-work/c4a0` is the checkout. `~/agent-work/env.sh` keeps caches inside
  `~/agent-work`. Drive it with `scripts/alapin`:
  - `scripts/alapin sync <worktree> <ref>` bundles an unpushed ref over and rebuilds only when
    sources or the lock changed. It refuses while a run is active.
  - `ALAPIN_TIMEOUT=6h scripts/alapin run <name> -- <cmd>` runs detached in the checkout with
    `$RUN_DIR` set to `~/agent-work/runs/<name>`. Put outputs there.
  - `scripts/alapin status [name]`, `scripts/alapin stop <name>` (kills the run's whole
    session), and `scripts/alapin fetch <name>` (to `/tmp/alapin-runs/<name>`).
- Labels for the ceiling live at `~/agent-work/data/train-s7.tsv`.
- GPU memory is budgeted per training process. `scripts/alapin run` caps each process at
  `ALAPIN_GPU_FRACTION` of the card (default 0.12, about 2.8 GB) and refuses to start when the
  live processes plus `ALAPIN_GPU_PROCS` new ones would not fit beside a 6 GB desktop reserve.
  Set `ALAPIN_GPU_PROCS` to the number of `train` processes the command starts. At the default
  cap, at most 5 run at once.
