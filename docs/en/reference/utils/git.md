---
description: Utilities for retrieving Git repository metadata such as branch, commit, and origin URL, used in Ultralytics projects.
keywords: Ultralytics, Git, GitRepo, branch, commit, origin, repository, utils, metadata, version control
---

# Reference for `ultralytics/utils/git.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`GitRepo`](#ultralytics.utils.git.GitRepo)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`GitRepo.head`](#ultralytics.utils.git.GitRepo.head)
        - [`GitRepo.is_repo`](#ultralytics.utils.git.GitRepo.is_repo)
        - [`GitRepo.branch`](#ultralytics.utils.git.GitRepo.branch)
        - [`GitRepo.commit`](#ultralytics.utils.git.GitRepo.commit)
        - [`GitRepo.origin`](#ultralytics.utils.git.GitRepo.origin)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`GitRepo._find_root`](#ultralytics.utils.git.GitRepo._find_root)
        - [`GitRepo._gitdir`](#ultralytics.utils.git.GitRepo._gitdir)
        - [`GitRepo._read`](#ultralytics.utils.git.GitRepo._read)
        - [`GitRepo._ref_commit`](#ultralytics.utils.git.GitRepo._ref_commit)


## Class `ultralytics.utils.git.GitRepo` {#ultralytics.utils.git.GitRepo}

```python
GitRepo(self, path: Path = Path(__file__).resolve())
```

Represent a local Git repository and expose branch, commit, and remote metadata.

This class discovers the repository root by searching for a .git entry from the given path upward, resolves the actual .git directory (including worktrees), and reads Git metadata directly from on-disk files. It does not invoke the git binary and therefore works in restricted environments. All metadata properties are resolved lazily and cached; construct a new instance to refresh state.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `Path, optional` | File or directory path used as the starting point to locate the repository root. | `Path(__file__).resolve()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `root` | `Path | None` | Repository root directory containing the .git entry; None if not in a repository. |
| `gitdir` | `Path | None` | Resolved .git directory path; handles worktrees; None if unresolved. |
| `head` | `str | None` | Raw contents of HEAD; a SHA for detached HEAD or "ref: <refname>" for branch heads. |
| `is_repo` | `bool` | Whether the provided path resides inside a Git repository. |
| `branch` | `str | None` | Current branch name when HEAD points to a branch; None for detached HEAD or non-repo. |
| `commit` | `str | None` | Current commit SHA for HEAD; None if not determinable. |
| `origin` | `str | None` | URL of the "origin" remote as read from gitdir/config; None if unset or unavailable. |

**Methods**

| Name | Description |
| --- | --- |
| [`head`](#ultralytics.utils.git.GitRepo.head) | HEAD file contents. |
| [`is_repo`](#ultralytics.utils.git.GitRepo.is_repo) | True if inside a git repo. |
| [`branch`](#ultralytics.utils.git.GitRepo.branch) | Current branch or None. |
| [`commit`](#ultralytics.utils.git.GitRepo.commit) | Current commit SHA or None. |
| [`origin`](#ultralytics.utils.git.GitRepo.origin) | Origin URL or None. |
| [`_find_root`](#ultralytics.utils.git.GitRepo._find_root) | Return repo root or None. |
| [`_gitdir`](#ultralytics.utils.git.GitRepo._gitdir) | Resolve actual .git directory (handles worktrees). |
| [`_read`](#ultralytics.utils.git.GitRepo._read) | Read and strip file if exists. |
| [`_ref_commit`](#ultralytics.utils.git.GitRepo._ref_commit) | Commit for ref (handles packed-refs). |

**Examples**

```python
Initialize from the current working directory and read metadata
>>> from pathlib import Path
>>> repo = GitRepo(Path.cwd())
>>> repo.is_repo
True
>>> repo.branch, repo.commit[:7], repo.origin
('main', '1a2b3c4', 'https://example.com/owner/repo.git')
```

!!! note "Notes"

    - Resolves metadata by reading files: HEAD, packed-refs, and config; no subprocess calls are used.
    - Caches properties on first access using cached_property; recreate the object to reflect repository changes.

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L9-L126"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class GitRepo:
    """Represent a local Git repository and expose branch, commit, and remote metadata.

    This class discovers the repository root by searching for a .git entry from the given path upward, resolves the
    actual .git directory (including worktrees), and reads Git metadata directly from on-disk files. It does not invoke
    the git binary and therefore works in restricted environments. All metadata properties are resolved lazily and
    cached; construct a new instance to refresh state.

    Attributes:
        root (Path | None): Repository root directory containing the .git entry; None if not in a repository.
        gitdir (Path | None): Resolved .git directory path; handles worktrees; None if unresolved.
        head (str | None): Raw contents of HEAD; a SHA for detached HEAD or "ref: <refname>" for branch heads.
        is_repo (bool): Whether the provided path resides inside a Git repository.
        branch (str | None): Current branch name when HEAD points to a branch; None for detached HEAD or non-repo.
        commit (str | None): Current commit SHA for HEAD; None if not determinable.
        origin (str | None): URL of the "origin" remote as read from gitdir/config; None if unset or unavailable.

    Examples:
        Initialize from the current working directory and read metadata
        >>> from pathlib import Path
        >>> repo = GitRepo(Path.cwd())
        >>> repo.is_repo
        True
        >>> repo.branch, repo.commit[:7], repo.origin
        ('main', '1a2b3c4', 'https://example.com/owner/repo.git')

    Notes:
        - Resolves metadata by reading files: HEAD, packed-refs, and config; no subprocess calls are used.
        - Caches properties on first access using cached_property; recreate the object to reflect repository changes.
    """

    def __init__(self, path: Path = Path(__file__).resolve()):
        """Initialize a Git repository context by discovering the repository root from a starting path.

        Args:
            path (Path, optional): File or directory path used as the starting point to locate the repository root.
        """
        self.root = self._find_root(path)
        self.gitdir = self._gitdir(self.root) if self.root else None
```
</details>

<br>

### Property `ultralytics.utils.git.GitRepo.head` {#ultralytics.utils.git.GitRepo.head}

```python
def head(self) -> str | None
```

HEAD file contents.

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L72-L74"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@cached_property
def head(self) -> str | None:
    """HEAD file contents."""
    return self._read(self.gitdir / "HEAD" if self.gitdir else None)
```
</details>

<br>

### Property `ultralytics.utils.git.GitRepo.is_repo` {#ultralytics.utils.git.GitRepo.is\_repo}

```python
def is_repo(self) -> bool
```

True if inside a git repo.

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L93-L95"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def is_repo(self) -> bool:
    """True if inside a git repo."""
    return self.gitdir is not None
```
</details>

<br>

### Property `ultralytics.utils.git.GitRepo.branch` {#ultralytics.utils.git.GitRepo.branch}

```python
def branch(self) -> str | None
```

Current branch or None.

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L98-L103"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@cached_property
def branch(self) -> str | None:
    """Current branch or None."""
    if not self.is_repo or not self.head or not self.head.startswith("ref: "):
        return None
    ref = self.head[5:].strip()
    return ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref
```
</details>

<br>

### Property `ultralytics.utils.git.GitRepo.commit` {#ultralytics.utils.git.GitRepo.commit}

```python
def commit(self) -> str | None
```

Current commit SHA or None.

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L106-L110"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@cached_property
def commit(self) -> str | None:
    """Current commit SHA or None."""
    if not self.is_repo or not self.head:
        return None
    return self._ref_commit(self.head[5:].strip()) if self.head.startswith("ref: ") else self.head
```
</details>

<br>

### Property `ultralytics.utils.git.GitRepo.origin` {#ultralytics.utils.git.GitRepo.origin}

```python
def origin(self) -> str | None
```

Origin URL or None.

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L113-L126"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@cached_property
def origin(self) -> str | None:
    """Origin URL or None."""
    if not self.is_repo:
        return None
    cfg = self.gitdir / "config"
    remote, url = None, None
    for s in (self._read(cfg) or "").splitlines():
        t = s.strip()
        if t.startswith("[") and t.endswith("]"):
            remote = t.lower()
        elif t.lower().startswith("url =") and remote == '[remote "origin"]':
            url = t.split("=", 1)[1].strip()
            break
    return url
```
</details>

<br>

### Method `ultralytics.utils.git.GitRepo._find_root` {#ultralytics.utils.git.GitRepo.\_find\_root}

```python
def _find_root(p: Path) -> Path | None
```

Return repo root or None.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `p` | `Path` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L50-L52"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _find_root(p: Path) -> Path | None:
    """Return repo root or None."""
    return next((d for d in [p, *list(p.parents)] if (d / ".git").exists()), None)
```
</details>

<br>

### Method `ultralytics.utils.git.GitRepo._gitdir` {#ultralytics.utils.git.GitRepo.\_gitdir}

```python
def _gitdir(root: Path) -> Path | None
```

Resolve actual .git directory (handles worktrees).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `root` | `Path` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L55-L64"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _gitdir(root: Path) -> Path | None:
    """Resolve actual .git directory (handles worktrees)."""
    g = root / ".git"
    if g.is_dir():
        return g
    if g.is_file():
        t = g.read_text(errors="ignore").strip()
        if t.startswith("gitdir:"):
            return (root / t.split(":", 1)[1].strip()).resolve()
    return None
```
</details>

<br>

### Method `ultralytics.utils.git.GitRepo._read` {#ultralytics.utils.git.GitRepo.\_read}

```python
def _read(p: Path | None) -> str | None
```

Read and strip file if exists.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `p` | `Path | None` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L67-L69"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _read(p: Path | None) -> str | None:
    """Read and strip file if exists."""
    return p.read_text(errors="ignore").strip() if p and p.exists() else None
```
</details>

<br>

### Method `ultralytics.utils.git.GitRepo._ref_commit` {#ultralytics.utils.git.GitRepo.\_ref\_commit}

```python
def _ref_commit(self, ref: str) -> str | None
```

Commit for ref (handles packed-refs).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ref` | `str` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/git.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/git.py#L76-L90"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _ref_commit(self, ref: str) -> str | None:
    """Commit for ref (handles packed-refs)."""
    rf = self.gitdir / ref
    if s := self._read(rf):
        return s
    pf = self.gitdir / "packed-refs"
    b = pf.read_bytes().splitlines() if pf.exists() else []
    tgt = ref.encode()
    for line in b:
        if line[:1] in (b"#", b"^") or b" " not in line:
            continue
        sha, name = line.split(b" ", 1)
        if name.strip() == tgt:
            return sha.decode()
    return None
```
</details>

<br><br>
