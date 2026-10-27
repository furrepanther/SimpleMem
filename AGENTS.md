## REQUIRED: Personal Fork First; Preserve Local Work

All repository code, tests, configuration, build files, and behavior-changing documentation work happens in the personal GitHub fork at `/mnt/linuxdata/Github/SimpleMem`. If that personal fork is absent, create or obtain it and clone it only at this canonical path before editing; do not edit an upstream checkout, production install, or upgrade candidate in its place. The personal fork is both the backup and the historical record of local changes.

1. Verify this path and the exact remotes before the first edit: `origin` must be the personal fork used for backup/push, and `upstream` must be the canonical source when one exists. If fork creation or remote ownership is ambiguous, stop source changes and report the blocker.
2. Fetch the configured remotes with `--prune --tags`, then capture remotes, branch/tracking state, divergence, `git status --short`, tracked/untracked inventory, dirty diff evidence, and `git fsck --full`.
3. If the fork is dirty or behind source, preserve every local change in a recoverable form before refresh/reconcile, including untracked files. Never reset, clean, overwrite, silently rebase, force-push, or rewrite fork history without explicit authorization.
4. Compare fork history and dirty state with the source/upgrade tree using normal and end-of-line/whitespace-insensitive diffs. Classify deltas as `PORT`, `ALREADY_UPSTREAM`, `OBSOLETE`, `POLICY_ONLY`, `WHITESPACE_ONLY`, or `UNRESOLVED`; port only applicable changes as focused commits.
5. Before promotion, commit substantive work in this personal fork and record source/target commit IDs, changed files, classifications, tests, adversarial review, and unresolved items. Do not copy a dirty tree wholesale.
