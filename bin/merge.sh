#!/usr/bin/env bash
set -euo pipefail

remote="${REMOTE:-upstream}"
branch="${1:-$(git branch --show-current)}"
commit_message="${COMMIT_MESSAGE:-chore: checkpoint before rebasing from the latest $remote release}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: must be run inside a git repository" >&2
  exit 1
fi

if ! git remote get-url "$remote" >/dev/null 2>&1; then
  echo "error: git remote '$remote' is not configured" >&2
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "error: local branch '$branch' does not exist" >&2
  exit 1
fi

current_branch="$(git branch --show-current)"
has_tracked_changes=0
has_untracked_files=0

if ! git diff --quiet || ! git diff --cached --quiet; then
  has_tracked_changes=1
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  has_untracked_files=1
fi

if (( has_tracked_changes || has_untracked_files )); then
  echo "Committing current changes on $current_branch..."
  git add -A
  git commit -m "$commit_message"
fi

echo "Fetching $remote..."
git fetch --prune --tags "$remote"

latest_release_tag="$(git tag --list 'v*' --sort=-version:refname | head -n 1)"

if [[ -z "$latest_release_tag" ]]; then
  echo "error: no release tags matching 'v*' were found after fetching $remote" >&2
  exit 1
fi

if [[ "$current_branch" != "$branch" ]]; then
  echo "Switching to $branch..."
  git switch "$branch"
fi

echo "Rebasing $branch onto latest $remote release $latest_release_tag..."
git rebase "$latest_release_tag"

if ! git merge-base --is-ancestor "$latest_release_tag" HEAD; then
  echo "error: rebase completed, but HEAD is not based on $latest_release_tag" >&2
  exit 1
fi

repo_version="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' ultralytics/__init__.py | head -n 1)"
git_version="$(git describe --tags --always --dirty)"

echo "Rebase successful."
echo "Base release: $latest_release_tag"
if [[ -n "$repo_version" ]]; then
  echo "Package version: $repo_version"
fi
echo "Git version: $git_version"
