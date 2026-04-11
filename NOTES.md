# Upstream Updates

This fork already has an `upstream` remote configured:

```bash
git remote -v
```

Expected upstream:

```bash
upstream  git@github.com:ultralytics/ultralytics.git
```

Fetch the latest refs from upstream without changing your working branch:

```bash
git fetch upstream
```

Fetch all branches and tags, and prune deleted remote refs:

```bash
git fetch --prune --tags upstream
```

Inspect what changed upstream compared with your current branch:

```bash
git log --oneline HEAD..upstream/main
```

If you want to update your local `main` branch after fetching:

```bash
git switch main
git merge --ff-only upstream/main
```

If your default branch is not `main`, replace `main` with the correct branch name.
