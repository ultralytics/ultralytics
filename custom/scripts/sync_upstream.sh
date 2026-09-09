#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

set -euo pipefail

upstream_remote="${UPSTREAM_REMOTE:-upstream}"
upstream_branch="${UPSTREAM_BRANCH:-main}"

if ! git remote get-url "${upstream_remote}" > /dev/null 2>&1; then
  echo "Missing Git remote '${upstream_remote}'. Add it with:"
  echo "  git remote add ${upstream_remote} https://github.com/ultralytics/ultralytics.git"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash local changes before syncing."
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ -z "${current_branch}" || "${current_branch}" == "main" ]]; then
  echo "Run this script from a feature or sync branch, not directly from main."
  exit 1
fi

git fetch "${upstream_remote}" --prune --tags
git rebase "${upstream_remote}/${upstream_branch}"

cat << EOF

Rebased ${current_branch} onto ${upstream_remote}/${upstream_branch}.
Run the smoke checks from custom/README.md, then update custom/UPSTREAM.md.
EOF
