# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare and validate the complete documentation tree with Zensical."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from build_reference import build_reference_docs
from minijinja import Environment, load_from_path

from ultralytics.utils import LOGGER
from ultralytics.utils.tqdm import TQDM

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"  # fix DeprecationWarning: Jupyter is migrating to use standard platformdirs
DOCS = Path(__file__).parent.resolve()
SITE = DOCS.parent / "site"


def prepare_docs_markdown():
    """Prepare documentation Markdown for validation."""
    LOGGER.info("Removing existing build artifacts")
    shutil.rmtree(SITE, ignore_errors=True)
    shutil.rmtree(DOCS / "repos", ignore_errors=True)

    repo = "https://github.com/ultralytics/docs"
    local_dir = DOCS / "repos/docs"
    subprocess.run(
        ["git", "clone", "-q", "--depth=1", "--single-branch", "-b", "main", repo, str(local_dir)], check=True
    )
    shutil.rmtree(DOCS / "en/compare", ignore_errors=True)
    shutil.copytree(local_dir / "docs/en/compare", DOCS / "en/compare")
    LOGGER.info(f"Loaded {repo} from {local_dir}")

    # Add frontmatter
    for file in TQDM((DOCS / "en").rglob("*.md"), desc="Adding frontmatter"):
        update_markdown_files(file)


def update_markdown_files(md_filepath: Path):
    """Create or update a Markdown file, ensuring frontmatter is present."""
    if md_filepath.exists():
        content = md_filepath.read_text().strip()

        # Replace apostrophes
        content = content.replace("‘", "'").replace("’", "'")

        # Add frontmatter if missing
        if not content.strip().startswith("---\n"):
            header = (
                "---\ncomments: true\n"
                "description: Ultralytics documentation for YOLO model training, validation, prediction, export, and deployment.\n"
                "keywords: Ultralytics, YOLO, computer vision, model training, model export, deployment\n---\n\n"
            )
            content = header + content

        # Ensure content-tab "=== " lines are preceded and followed by empty newlines
        lines = content.split("\n")
        new_lines = []
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith("=== "):
                if i > 0 and new_lines[-1] != "":
                    new_lines.append("")
                new_lines.append(line)
                if i < len(lines) - 1 and lines[i + 1].strip() != "":
                    new_lines.append("")
            else:
                new_lines.append(line)
        content = "\n".join(new_lines)

        # Add EOF newline if missing
        if not content.endswith("\n"):
            content += "\n"

        # Save page
        md_filepath.write_text(content)


def render_jinja_macros() -> None:
    """Render MiniJinja macros in Markdown files before validating with Zensical."""
    mkdocs_yml = DOCS.parent / "mkdocs.yml"
    default_yaml = DOCS.parent / "ultralytics" / "cfg" / "default.yaml"

    class SafeFallbackLoader(yaml.SafeLoader):
        """SafeLoader that gracefully skips unknown configuration tags."""

    def _ignore_unknown(loader, tag_suffix, node):
        """Gracefully handle YAML tags that aren't registered."""
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return None

    SafeFallbackLoader.add_multi_constructor("", _ignore_unknown)

    def load_yaml(path: Path, *, safe_loader: yaml.Loader = yaml.SafeLoader) -> dict:
        """Load YAML safely, returning an empty dict when the file is absent."""
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.load(f, Loader=safe_loader) or {}

    mkdocs_cfg = load_yaml(mkdocs_yml, safe_loader=SafeFallbackLoader)
    extra_vars = mkdocs_cfg.get("extra", {}) or {}
    site_name = mkdocs_cfg.get("site_name", "Ultralytics Docs")
    extra_vars.update(load_yaml(default_yaml))

    env = Environment(
        loader=load_from_path([DOCS / "en", DOCS]),
        auto_escape_callback=lambda _: False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        comment_start_string="{##",
        comment_end_string="##}",
    )

    def indent_filter(value: str, width: int = 4, first: bool = False, blank: bool = False) -> str:
        """Mimic Jinja's indent filter to preserve macros compatibility."""
        prefix = " " * int(width)
        result = []
        for i, line in enumerate(str(value).splitlines(keepends=True)):
            if not line.strip() and not blank:
                result.append(line)
                continue
            if i == 0 and not first:
                result.append(line)
            else:
                result.append(prefix + line)
        return "".join(result)

    env.add_filter("indent", indent_filter)
    reserved_keys = {"name"}
    base_context = {**extra_vars, "page": {"meta": {}}, "config": {"site_name": site_name}}

    files_with_macros = 0
    macros_total = 0

    pbar = TQDM((DOCS / "en").rglob("*.md"), desc="MiniJinja: 0 macros, 0 pages")
    for md_file in pbar:
        if "macros" in md_file.parts or "reference" in md_file.parts:
            continue
        content = md_file.read_text(encoding="utf-8")
        if "{{" not in content and "{%" not in content:
            continue

        parts = content.split("---\n")
        frontmatter = ""
        frontmatter_data = {}
        markdown_content = content
        if content.startswith("---\n") and len(parts) >= 3:
            frontmatter = f"---\n{parts[1]}---\n"
            markdown_content = "---\n".join(parts[2:])
            frontmatter_data = yaml.safe_load(parts[1]) or {}

        macro_hits = markdown_content.count("{{") + markdown_content.count("{%")
        if not macro_hits:
            continue

        context = {k: v for k, v in base_context.items() if k not in reserved_keys}
        context.update({k: v for k, v in frontmatter_data.items() if k not in reserved_keys})
        context["page"] = context.get("page", {})
        context["page"]["meta"] = frontmatter_data

        rendered = env.render_str(markdown_content, name=str(md_file.relative_to(DOCS)), **context)

        md_file.write_text(frontmatter + rendered, encoding="utf-8")
        files_with_macros += 1
        macros_total += macro_hits
        pbar.set_description(f"MiniJinja: {macros_total} macros, {files_with_macros} pages")


def backup_docs_sources() -> tuple[Path, list[tuple[Path, Path]]]:
    """Create a temporary backup of docs sources so we can fully restore after building."""
    backup_root = Path(tempfile.mkdtemp(prefix="docs_backup_", dir=str(DOCS.parent)))
    sources = [DOCS / "en", DOCS / "macros"]
    copied: list[tuple[Path, Path]] = []
    for src in sources:
        if not src.exists():
            continue
        dst = backup_root / src.name
        shutil.copytree(src, dst)
        copied.append((src, dst))
    return backup_root, copied


def restore_docs_sources(backup_root: Path, backups: list[tuple[Path, Path]]):
    """Restore docs sources from the temporary backup."""
    for src, dst in backups:
        shutil.rmtree(src, ignore_errors=True)
        if dst.exists():
            shutil.copytree(dst, src)
    shutil.rmtree(backup_root, ignore_errors=True)


def main():
    """Prepare and validate the complete documentation tree."""
    if not shutil.which("zensical"):
        raise SystemExit('zensical is not installed. Install it with: uv pip install -e ".[dev]"')

    backup_root: Path | None = None
    docs_backups: list[tuple[Path, Path]] = []
    restored = False

    def restore_all():
        """Restore docs sources from backup once build steps complete."""
        nonlocal restored
        if backup_root:
            LOGGER.info("Restoring docs directory from backup")
            restore_docs_sources(backup_root, docs_backups)
        restored = True

    try:
        backup_root, docs_backups = backup_docs_sources()
        prepare_docs_markdown()
        build_reference_docs(update_nav=False)
        render_jinja_macros()

        # Remove cloned repos before validation to keep the tree lean
        shutil.rmtree(DOCS / "repos", ignore_errors=True)

        # Build the main documentation
        LOGGER.info(f"Building docs from {DOCS}")
        subprocess.run(["zensical", "build", "-f", str(DOCS.parent / "mkdocs.yml"), "--strict"], check=True)
        LOGGER.info(f"Site built at {SITE}")
        LOGGER.info("Docs built correctly ✅")
        restore_all()
    finally:
        if not restored:
            restore_all()
        shutil.rmtree(DOCS / "repos", ignore_errors=True)


if __name__ == "__main__":
    main()
