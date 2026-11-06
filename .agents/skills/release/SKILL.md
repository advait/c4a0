---
name: release
description: "Read before landing a c4a0 change: branches, review, who pushes, and what CI checks."
---

# Releasing c4a0

Status: implemented

Nothing is deployed. Landing means a commit on `master` of `advait/c4a0`.

- Work on a branch. Commit there. `master` has no branch protection, but agents never push
  to it.
- Push a branch only when the maintainer says to push, merge or open a PR for that work.
  An approval for one change does not cover the next.
- CI (`.github/workflows/ci.yaml`) runs `mise run ci` on every push and on manual dispatch,
  but not on pull requests. A fork PR therefore has no checks, so run `mise run ci` on its
  head yourself before judging it.
- After a push, watch the CI run for your SHA until it is green.
- Diffs touching the metric, MCTS, training targets or persisted run formats get a review
  from a model that did not write them, per the contract default.
- Comments on outside contributors' PRs are public. Write them only when asked, and keep
  them specific and kind.
