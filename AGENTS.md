## FIRST: Refresh From Origin, Preserve Local Work

For every repository task, make refreshing from `origin` the first repository action. This keeps active local code current without losing local work.

1. Run `git fetch origin --prune --tags` before other repository actions; if no usable `origin` exists, record that blocker before continuing.
2. Inspect `git status`, `git fsck --full`, and divergence from the tracked branch.
3. If `origin` has commits we do not have, preserve every local modification first in a recoverable commit or patch, refresh the clean base, and merge the preserved local changes back.
4. Never discard, overwrite, or silently reset local changes. Resolve conflicts deliberately while retaining local behavior, then run affected tests and review the resulting diff.
5. Record resulting commit IDs and verification evidence before continuing with normal work.
