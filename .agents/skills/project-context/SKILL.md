---
name: project-context
description: "Read before the first decision on c4a0: who uses it, what is one-way, what failure costs most, the machine it runs on, the mise task names, and how to run Codex here."
---

# c4a0 project context

Status: implemented

**What it is.** An AlphaZero-style Connect Four engine. Rust (`rust/src`) holds the rules,
MCTS, multi-threaded self-play, position sampling and splits, the Pascal Pons solver interface
and the TUI. Python
(`src/c4a0`) holds the PyTorch Lightning network, the generation loop and the CLI
(`src/c4a0/main.py`). PyO3 joins them (`rust/src/pybridge.rs`, stubs in
`src/c4a0_rust/__init__.pyi`).

**Who uses it.** One maintainer, Advait Shinde (every commit in `git log`). The repo
`advait/c4a0` is public, and outside contributors occasionally open PRs. Nothing is deployed.

**The current goal.** Self-play training that converges to 100% solver-optimal moves. Treat
every change as a step toward that, measured per the **verify** skill.

**What failure costs most.** A wrong number. A convergence metric that measures the wrong
model, the wrong positions, or the wrong notion of "optimal" wastes hours of CPU and sends
the work the wrong way. Make metric code at least as correct as training code, and prove it
with tests.

**One-way state.**
- Training runs under `--base-dir` (default `training/`, gitignored). Each generation
  directory holds `metadata.json` (parsed by `TrainingGen` in `src/c4a0/training.py`),
  `model.pkl` and `games.pkl`. Renaming or reinterpreting a `TrainingGen` field breaks
  loading or silently changes meaning for existing runs. Pickles break on class changes.
- `metadata.json` `evals` are keyed by suite name (`v1/dev`) and carry `suite_sha256`. The
  suite files under `src/c4a0/suite/v1/` and the split salt in `rust/src/labels.rs` are
  one-way: changing either makes old `evals` incomparable, so add a `v2` suite instead.
- The PyO3 API is internal to this repo, so it may change freely as long as the `.pyi`
  stubs and every Python caller change in the same commit.

**Machine.** A Docker container with no GPU, 32 cores, 124 GB RAM and no `sudo`. The
container is the isolation boundary.

**Codex here.** `~/.codex/config.toml` sets `sandbox_mode = "danger-full-access"`. Never pass
`-s` or `--sandbox` to `codex exec`. Its `bwrap` sandbox cannot create namespaces in this
container, and the run will read nothing. Read-only work is a brief instruction, not a
sandbox flag.

**Python tools.** uv owns the interpreter and environment. `project.requires-python`
is the single version pin. ty is the only type checker. Ruff checks annotations and style.
All validation tasks enforce `uv.lock` with `--locked`.

**Task runner.** mise (`mise.toml`). Canonical tasks: `install`, `build`, `lint`,
`typecheck`, `test:rust`, `test:python`, `check`, `train:smoke`, `ci`, `train`, `play`, `eval`, `report`, and `suite:build`
(needs the solver). See the **verify** skill for which to run when.

**Follow-ups.** GitHub issues on `advait/c4a0`, or list them in the reply. Convergence
hypotheses, their measured results and the open next steps live in `docs/experiments.md`.
Add a row there for every experiment you run, kept or not. The old `docs/improvements.md`
backlog was deliberately removed. Do not recreate it.
