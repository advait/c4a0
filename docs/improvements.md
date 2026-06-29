# Ranked Improvement Backlog

Ranking is ordered for **highest impact with lowest difficulty first**. Impact and difficulty use a 1-5 scale where 5 is highest. Difficulty 1 means quick/easy; difficulty 5 means large/risky.

| Rank | Improvement | Impact | Difficulty | Why this order / expected payoff |
|---:|---|---:|---:|---|
| 1 | Add a fast end-to-end smoke training check to CI | 5 | 2 | Done: `mise run ci` includes `train:smoke`, catching broken Rust/Python/PyTorch integration before merge. |
| 2 | Run Python tests and Pyright in CI | 5 | 2 | Done: CI runs `mise run ci`, which includes Ruff, Pyright, Rust tests, Python tests, and smoke training. |
| 3 | Add mise tasks for common commands | 4 | 1 | Done: `mise.toml` bootstraps uv/Rust/clang and exposes install/build/check/test/smoke tasks. |
| 4 | Fix Python packaging so `import c4a0` works after install | 5 | 3 | Done: maturin mixed package layout installs `c4a0` and `c4a0_rust` without `PYTHONPATH` hacks. |
| 5 | Add a first-class self-play stats CLI command | 4 | 2 | Today stats require ad-hoc pickle loading. A command like `stats --base-dir training` should report games, samples, unique positions, lengths, win rates, losses, and solver scores. |
| 6 | Make training duration configurable | 4 | 2 | `max_epochs=100`, early stopping patience, and logging cadence are hard-coded. CLI/config knobs would make smoke runs, CPU runs, and serious GPU runs less awkward. |
| 7 | Move Lightning logs under each training run directory | 4 | 2 | `lightning_logs/` currently lands at repo root. Keeping logs with generation artifacts improves reproducibility and cleanup. |
| 8 | Add robust Rust thread/Python callback error propagation | 5 | 4 | A Python callback error in the NN thread can panic and leave other threads waiting. This is critical reliability work for long training runs. |
| 9 | Replace or wrap pickle artifacts with versioned formats | 4 | 3 | `model.pkl` and `games.pkl` are fragile across code changes. Prefer `state_dict` + JSON config and a versioned CBOR/NPZ/parquet format for self-play samples. |
| 10 | Save full experiment config and environment metadata | 4 | 2 | Store CLI args, model config, git SHA, package versions, device, seed, and command in each run for reproducibility. |
| 11 | Add deterministic seed controls | 4 | 3 | Training currently mixes Rust/Python/PyTorch randomness. Explicit seeds enable reproducible smoke tests and more reliable comparisons. |
| 12 | Modernize `pyproject.toml` dependency groups | 3 | 1 | `tool.uv.dev-dependencies` is deprecated. Move to `dependency-groups.dev` and pin/organize dev tools. |
| 13 | Add Rust formatting/linting checks | 3 | 1 | Add `cargo fmt --check` and `cargo clippy -- -D warnings` to CI to catch maintainability issues early. |
| 14 | Generate or maintain PyO3 type stubs | 3 | 2 | A checked-in `c4a0_rust.pyi` helps Pyright, but it can drift. Generate or test stubs against the extension API. |
| 15 | Add solver setup automation | 3 | 2 | Provide a script or documented task to fetch/build Pascal Pons solver and opening book, with cache paths managed consistently. |
| 16 | Add MCTS/self-play performance benchmarks | 4 | 3 | Throughput matters. Benchmarks would guide threading, batching, and MCTS parameter changes. |
| 17 | Improve training metric logging | 3 | 2 | Log policy KL, value MSEs, unique positions, game lengths, win/draw rates, and solver score in a consistent dashboard-friendly format. |
| 18 | Split CLI into smaller modules and test CLI parsing | 3 | 3 | `main.py` contains several commands and defaults. Smaller command modules plus CLI tests reduce regressions. |
| 19 | Add artifact cleanup and run management commands | 2 | 2 | Commands like `list-runs`, `delete-run`, `export-run`, and `best-model` would make experiments easier to manage. |
| 20 | Containerize the dev/training environment | 3 | 4 | A Docker/devcontainer with Rust, clang/libclang, uv, and optional solver would eliminate setup drift, but is more involved. |
