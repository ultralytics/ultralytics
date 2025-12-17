# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Automates building and post-processing of MkDocs documentation, especially for multilingual projects.

This script streamlines generating localized documentation and updating HTML links for correct formatting.

Key Features:
    - Automated building of MkDocs documentation: Compiles main documentation and localized versions from separate
      MkDocs configuration files.
    - Post-processing of generated HTML files: Updates HTML files to remove '.md' from internal links, ensuring
      correct navigation in web-based documentation.

Usage:
    - Run from the root directory of your MkDocs project.
    - Ensure MkDocs is installed and configuration files (main and localized) are present.
    - The script builds documentation using MkDocs, then scans HTML files in 'site' to update links.
    - Ideal for projects with Markdown documentation served as a static website.

Note:
    - Requires Python and MkDocs to be installed and configured.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from minijinja import Environment, load_from_path

try:
    from plugin import postprocess_site  # mkdocs-ultralytics-plugin
except ImportError:
    postprocess_site = None

from build_reference import build_reference_docs, build_reference_for

from ultralytics.utils import LINUX, LOGGER, MACOS
from ultralytics.utils.tqdm import TQDM

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"  # fix DeprecationWarning: Jupyter is migrating to use standard platformdirs
DOCS = Path(__file__).parent.resolve()
SITE = DOCS.parent / "site"
LINK_PATTERN = re.compile(r"(https?://[^\s()<>]*[^\s()<>.,:;!?\'\"])")
TITLE_PATTERN = re.compile(r"<title>(.*?)</title>", flags=re.IGNORECASE | re.DOTALL)
MD_LINK_PATTERN = re.compile(r'(["\']?)([^"\'>\s]+?)\.md(["\']?)')
DOC_KIND_LABELS = {"Class", "Function", "Method", "Property"}
DOC_KIND_COLORS = {
    "Class": "#039dfc",  # blue
    "Method": "#ef5eff",  # magenta
    "Function": "#fc9803",  # orange
    "Property": "#02e835",  # green
}


def prepare_docs_markdown(clone_repos: bool = True):
    """Build docs using mkdocs."""
    LOGGER.info("Removing existing build artifacts")
    shutil.rmtree(SITE, ignore_errors=True)
    shutil.rmtree(DOCS / "repos", ignore_errors=True)

    if clone_repos:
        # Get hub-sdk repo
        repo = "https://github.com/ultralytics/hub-sdk"
        local_dir = DOCS / "repos" / Path(repo).name
        subprocess.run(
            ["git", "clone", "-q", "--depth=1", "--single-branch", "-b", "main", repo, str(local_dir)], check=True
        )
        shutil.rmtree(DOCS / "en/hub/sdk", ignore_errors=True)  # delete if exists
        shutil.copytree(local_dir / "docs", DOCS / "en/hub/sdk")  # for docs
        LOGGER.info(f"Cloned/Updated {repo} in {local_dir}")

        # Get docs repo
        repo = "https://github.com/ultralytics/docs"
        local_dir = DOCS / "repos" / Path(repo).name
        subprocess.run(
            ["git", "clone", "-q", "--depth=1", "--single-branch", "-b", "main", repo, str(local_dir)], check=True
        )
        shutil.rmtree(DOCS / "en/compare", ignore_errors=True)  # delete if exists
        shutil.copytree(local_dir / "docs/en/compare", DOCS / "en/compare")  # for docs
        LOGGER.info(f"Cloned/Updated {repo} in {local_dir}")

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
            header = "---\ncomments: true\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"
            content = header + content

        # Ensure MkDocs admonitions "=== " lines are preceded and followed by empty newlines
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
    return


def update_docs_html():
    """Update titles, edit links, and convert plaintext links in HTML documentation in one pass."""
    from concurrent.futures import ProcessPoolExecutor

    html_files = list(SITE.rglob("*.html"))
    if not html_files:
        LOGGER.info("Updated HTML files: 0")
        return
    desc = f"Updating HTML at {SITE}"
    max_workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pbar = TQDM(executor.map(_process_html_file, html_files), total=len(html_files), desc=desc)
        updated = 0
        for res in pbar:
            updated += bool(res)
            pbar.set_description(f"{desc} ({updated}/{len(html_files)} updated)")


def _process_html_file(html_file: Path) -> bool:
    """Process a single HTML file; returns True if modified."""
    try:
        content = html_file.read_text(encoding="utf-8")
    except Exception as e:
        LOGGER.warning(f"Could not read {html_file}: {e}")
        return False

    changed = False
    try:
        rel_path = html_file.relative_to(SITE).as_posix()
    except ValueError:
        rel_path = html_file.name

    # For pages sourced from external repos (hub-sdk, compare), drop edit/copy buttons to avoid wrong links
    if rel_path.startswith(("hub/sdk/", "compare/")):
        before = content
        content = re.sub(
            r'<a[^>]*class="[^"]*md-content__button[^"]*"[^>]*>.*?</a>',
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if content != before:
            changed = True

    if rel_path == "404.html":
        new_content = re.sub(r"<title>.*?</title>", "<title>Ultralytics Docs - Not Found</title>", content)
        if new_content != content:
            content, changed = new_content, True

    new_content = update_docs_soup(content, html_file=html_file)
    if new_content != content:
        content, changed = new_content, True

    new_content = _rewrite_md_links(content)
    if new_content != content:
        content, changed = new_content, True

    if changed:
        try:
            html_file.write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            LOGGER.warning(f"Could not write {html_file}: {e}")
    return False


def update_docs_soup(content: str, html_file: Path | None = None, max_title_length: int = 70) -> str:
    """Convert plaintext links to HTML hyperlinks, truncate long meta titles, and remove code line hrefs."""
    title_match = TITLE_PATTERN.search(content)
    needs_title_trim = bool(
        title_match and len(title_match.group(1)) > max_title_length and "-" in title_match.group(1)
    )
    needs_link_conversion = ("<p" in content or "<li" in content) and bool(LINK_PATTERN.search(content))
    needs_codelineno_cleanup = "__codelineno-" in content
    rel_path = ""
    if html_file:
        try:
            rel_path = html_file.relative_to(SITE).as_posix()
        except Exception:
            rel_path = html_file.as_posix()
    needs_kind_highlight = "reference" in rel_path or "reference" in content

    if not (needs_title_trim or needs_link_conversion or needs_codelineno_cleanup or needs_kind_highlight):
        return content

    try:
        soup = BeautifulSoup(content, "lxml")
    except Exception:
        soup = BeautifulSoup(content, "html.parser")
    modified = False

    # Truncate long meta title if needed
    title_tag = soup.find("title") if needs_title_trim else None
    if title_tag and len(title_tag.text) > max_title_length and "-" in title_tag.text:
        title_tag.string = title_tag.text.rsplit("-", 1)[0].strip()
        modified = True

    # Find the main content area
    main_content = soup.find("main") or soup.find("div", class_="md-content")
    if not main_content:
        return str(soup) if modified else content

    # Convert plaintext links to HTML hyperlinks
    if needs_link_conversion:
        for paragraph in main_content.select("p, li"):
            for text_node in paragraph.find_all(string=True, recursive=False):
                if text_node.parent.name not in {"a", "code"}:
                    new_text = LINK_PATTERN.sub(r'<a href="\1">\1</a>', str(text_node))
                    if "<a href=" in new_text:
                        text_node.replace_with(BeautifulSoup(new_text, "html.parser"))
                        modified = True

    # Remove href attributes from code line numbers in code blocks
    if needs_codelineno_cleanup:
        for a in soup.select('a[href^="#__codelineno-"], a[id^="__codelineno-"]'):
            if a.string:  # If the a tag has text (the line number)
                # Check if parent is a span with class="normal"
                if a.parent and a.parent.name == "span" and "normal" in a.parent.get("class", []):
                    del a.parent["class"]
                a.replace_with(a.string)  # Replace with just the text
            else:  # If it has no text
                a.replace_with(soup.new_tag("span"))  # Replace with an empty span
            modified = True

    def highlight_labels(nodes):
        """Inject doc-kind badges into headings and nav entries."""
        nonlocal modified

        for node in nodes:
            if not node.contents:
                continue
            first = node.contents[0]
            if hasattr(first, "get") and "doc-kind" in (first.get("class") or []):
                continue
            text = first if isinstance(first, str) else getattr(first, "string", "")
            if not text:
                continue
            stripped = str(text).strip()
            if not stripped:
                continue
            kind = stripped.split()[0].rstrip(":")
            if kind not in DOC_KIND_LABELS:
                continue
            span = soup.new_tag("span", attrs={"class": f"doc-kind doc-kind-{kind.lower()}"})
            span.string = kind.lower()
            first.replace_with(span)
            tail = str(text)[len(kind) :]
            tail_stripped = tail.lstrip()
            if tail_stripped.startswith(kind):
                tail = tail_stripped[len(kind) :]
            if not tail and len(node.contents) > 0:
                tail = " "
            if tail:
                span.insert_after(tail)
            modified = True

    highlight_labels(soup.select("main h1, main h2, main h3, main h4, main h5"))
    highlight_labels(soup.select("nav.md-nav--secondary .md-ellipsis, nav.md-nav__list .md-ellipsis"))

    if "reference" in rel_path:
        for ellipsis in soup.select("nav.md-nav--secondary .md-ellipsis"):
            kind = ellipsis.find(class_=lambda c: c and "doc-kind" in c.split())
            text = str(kind.next_sibling).strip() if kind and kind.next_sibling else ellipsis.get_text(strip=True)
            if "." not in text:
                continue
            ellipsis.clear()
            short = text.rsplit(".", 1)[-1]
            if kind:
                ellipsis.append(kind)
                ellipsis.append(f" {short}")
            else:
                ellipsis.append(short)
            modified = True

    if needs_kind_highlight and not modified and soup.select(".doc-kind"):
        # Ensure style injection when pre-existing badges are present
        modified = True

    if modified:
        head = soup.find("head")
        if head and not soup.select("style[data-doc-kind]"):
            style = soup.new_tag("style", attrs={"data-doc-kind": "true"})
            style.string = (
                ".doc-kind{display:inline-flex;align-items:center;gap:0.25em;padding:0.21em 0.59em;border-radius:999px;"
                "font-weight:700;font-size:0.81em;letter-spacing:0.06em;text-transform:uppercase;"
                "line-height:1;color:var(--doc-kind-color,#f8fafc);"
                "background:var(--doc-kind-bg,rgba(255,255,255,0.12));}"
                f".doc-kind-class{{--doc-kind-color:{DOC_KIND_COLORS['Class']};--doc-kind-bg:rgba(3,157,252,0.22);}}"
                f".doc-kind-function{{--doc-kind-color:{DOC_KIND_COLORS['Function']};--doc-kind-bg:rgba(252,152,3,0.22);}}"
                f".doc-kind-method{{--doc-kind-color:{DOC_KIND_COLORS['Method']};--doc-kind-bg:rgba(239,94,255,0.22);}}"
                f".doc-kind-property{{--doc-kind-color:{DOC_KIND_COLORS['Property']};--doc-kind-bg:rgba(2,232,53,0.22);}}"
            )
            head.append(style)

    return str(soup) if modified else content


def _rewrite_md_links(content: str) -> str:
    """Replace .md references with trailing slashes in HTML content, skipping GitHub links."""
    if ".md" not in content:
        return content

    lines = []
    for line in content.split("\n"):
        if "github.com" not in line:
            line = line.replace("index.md", "")
            line = MD_LINK_PATTERN.sub(r"\1\2/\3", line)
        lines.append(line)
    return "\n".join(lines)


# Precompiled regex patterns for minification
HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
HTML_PRESERVE = re.compile(r"<(pre|code|textarea|script)[^>]*>[\s\S]*?</\1>", re.IGNORECASE)
HTML_TAG_SPACE = re.compile(r">\s+<")
HTML_MULTI_SPACE = re.compile(r"\s{2,}")
HTML_EMPTY_LINE = re.compile(r"^\s*$\n", re.MULTILINE)
CSS_COMMENT = re.compile(r"/\*[\s\S]*?\*/")


def remove_comments_and_empty_lines(content: str, file_type: str) -> str:
    """Remove comments and empty lines from a string of code, preserving newlines and URLs.

    Args:
        content (str): Code content to process.
        file_type (str): Type of file ('html', 'css', or 'js').

    Returns:
        (str): Cleaned content with comments and empty lines removed.

    Notes:
        Typical reductions for Ultralytics Docs are:
        - Total HTML reduction: 2.83% (1301.56 KB saved)
        - Total CSS reduction: 1.75% (2.61 KB saved)
        - Total JS reduction: 13.51% (99.31 KB saved)
    """
    if file_type == "html":
        content = HTML_COMMENT.sub("", content)  # Remove HTML comments
        # Preserve whitespace in <pre>, <code>, <textarea> tags
        preserved = []

        def preserve(match):
            """Mark HTML blocks that should not be minified."""
            preserved.append(match.group(0))
            return f"___PRESERVE_{len(preserved) - 1}___"

        content = HTML_PRESERVE.sub(preserve, content)
        content = HTML_TAG_SPACE.sub("><", content)  # Remove whitespace between tags
        content = HTML_MULTI_SPACE.sub(" ", content)  # Collapse multiple spaces
        content = HTML_EMPTY_LINE.sub("", content)  # Remove empty lines
        # Restore preserved content
        for i, text in enumerate(preserved):
            content = content.replace(f"___PRESERVE_{i}___", text)
    elif file_type == "css":
        content = CSS_COMMENT.sub("", content)  # Remove CSS comments
        # Remove whitespace around specific characters
        content = re.sub(r"\s*([{}:;,])\s*", r"\1", content)
        # Remove empty lines
        content = re.sub(r"^\s*\n", "", content, flags=re.MULTILINE)
        # Collapse multiple spaces to single space
        content = re.sub(r"\s{2,}", " ", content)
        # Remove all newlines
        content = re.sub(r"\n", "", content)
    elif file_type == "js":
        # Handle JS single-line comments (preserving http:// and https://)
        lines = content.split("\n")
        processed_lines = []
        for line in lines:
            # Only remove comments if they're not part of a URL
            if "//" in line and "http://" not in line and "https://" not in line:
                processed_lines.append(line.partition("//")[0])
            else:
                processed_lines.append(line)
        content = "\n".join(processed_lines)

        # Remove JS multi-line comments and clean whitespace
        content = re.sub(r"/\*[\s\S]*?\*/", "", content)
        # Remove empty lines
        content = re.sub(r"^\s*\n", "", content, flags=re.MULTILINE)
        # Collapse multiple spaces to single space
        content = re.sub(r"\s{2,}", " ", content)

        # Safe space removal around punctuation and operators (never include colons - breaks JS)
        content = re.sub(r"\s*([;{}])\s*", r"\1", content)
        content = re.sub(r"(\w)\s*\(|\)\s*{|\s*([+\-*/=])\s*", lambda m: m.group(0).replace(" ", ""), content)

    return content


def minify_files(html: bool = True, css: bool = True, js: bool = True):
    """Minify HTML, CSS, and JS files and print total reduction stats."""
    minify, compress, jsmin = None, None, None
    try:
        if html:
            from minify_html import minify
        if css:
            from csscompressor import compress
        if js:
            import jsmin
    except ImportError as e:
        LOGGER.info(f"Missing required package: {e}")
        return

    stats = {}
    for ext, minifier in {
        "html": (lambda x: minify(x, keep_closing_tags=True, minify_css=True, minify_js=True)) if html else None,
        "css": compress if css else None,
        "js": jsmin.jsmin if js else None,
    }.items():
        orig = minified = 0
        files = list(SITE.rglob(f"*.{ext}"))
        if not files:
            continue
        pbar = TQDM(files, desc=f"Minifying {ext.upper()} - reduced 0.00% (0.00 KB saved)")
        for f in pbar:
            content = f.read_text(encoding="utf-8")
            out = minifier(content) if minifier else remove_comments_and_empty_lines(content, ext)
            orig += len(content)
            minified += len(out)
            f.write_text(out, encoding="utf-8")
            saved = orig - minified
            pct = (saved / orig) * 100 if orig else 0.0
            pbar.set_description(f"Minifying {ext.upper()} - reduced {pct:.2f}% ({saved / 1024:.2f} KB saved)")
        stats[ext] = {"original": orig, "minified": minified}


def render_jinja_macros() -> None:
    """Render MiniJinja macros in markdown files before building with MkDocs."""
    mkdocs_yml = DOCS.parent / "mkdocs.yml"
    default_yaml = DOCS.parent / "ultralytics" / "cfg" / "default.yaml"

    class SafeFallbackLoader(yaml.SafeLoader):
        """SafeLoader that gracefully skips unknown tags (required for mkdocs.yml)."""

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
        """Load YAML safely, returning an empty dict on errors."""
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.load(f, Loader=safe_loader) or {}
        except Exception as e:
            LOGGER.warning(f"Could not load {path}: {e}")
            return {}

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

    files_processed = 0
    files_with_macros = 0
    macros_total = 0

    pbar = TQDM((DOCS / "en").rglob("*.md"), desc="MiniJinja: 0 macros, 0 pages")
    for md_file in pbar:
        if "macros" in md_file.parts or "reference" in md_file.parts:
            continue
        files_processed += 1

        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as e:
            LOGGER.warning(f"Could not read {md_file}: {e}")
            continue
        if "{{" not in content and "{%" not in content:
            continue

        parts = content.split("---\n")
        frontmatter = ""
        frontmatter_data = {}
        markdown_content = content
        if content.startswith("---\n") and len(parts) >= 3:
            frontmatter = f"---\n{parts[1]}---\n"
            markdown_content = "---\n".join(parts[2:])
            try:
                frontmatter_data = yaml.safe_load(parts[1]) or {}
            except Exception as e:
                LOGGER.warning(f"Could not parse frontmatter in {md_file}: {e}")

        macro_hits = markdown_content.count("{{") + markdown_content.count("{%")
        if not macro_hits:
            continue

        context = {k: v for k, v in base_context.items() if k not in reserved_keys}
        context.update({k: v for k, v in frontmatter_data.items() if k not in reserved_keys})
        context["page"] = context.get("page", {})
        context["page"]["meta"] = frontmatter_data

        try:
            rendered = env.render_str(markdown_content, name=str(md_file.relative_to(DOCS)), **context)
        except Exception as e:
            LOGGER.warning(f"Error rendering macros in {md_file}: {e}")
            continue

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
    """Build docs, update titles and edit links, minify HTML, and print local server command."""
    start_time = time.perf_counter()
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
        # Render reference docs for any extra packages present (e.g., hub-sdk)
        extra_refs = [
            {
                "package": DOCS / "repos" / "hub-sdk" / "hub_sdk",
                "reference_dir": DOCS / "en" / "hub" / "sdk" / "reference",
                "repo": "ultralytics/hub-sdk",
            },
        ]
        for ref in extra_refs:
            if ref["package"].exists():
                build_reference_for(ref["package"], ref["reference_dir"], ref["repo"], update_nav=False)
        render_jinja_macros()

        # Remove cloned repos before serving/building to keep the tree lean during mkdocs processing
        shutil.rmtree(DOCS / "repos", ignore_errors=True)

        # Build the main documentation
        LOGGER.info(f"Building docs from {DOCS}")
        subprocess.run(["zensical", "build", "-f", str(DOCS.parent / "mkdocs.yml")], check=True)
        LOGGER.info(f"Site built at {SITE}")

        # Remove search index JSON files to disable search
        Path(SITE / "search.json").unlink(missing_ok=True)

        # Update docs HTML pages
        update_docs_html()

        # Post-process site for meta tags, authors, social cards, and mkdocstrings polish
        if postprocess_site:
            postprocess_site(
                site_dir=SITE,
                docs_dir=DOCS / "en",
                site_url="https://docs.ultralytics.com",
                default_image="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png",
                default_author="glenn.jocher@ultralytics.com",
                add_desc=False,
                add_image=True,
                add_authors=True,
                add_json_ld=True,
                add_share_buttons=True,
                add_css=False,
                verbose=True,
            )
        else:
            LOGGER.warning("postprocess_site not available; skipping mkdocstrings postprocessing")

        # Minify files
        minify_files(html=False, css=False, js=False)

        # Print results and auto-serve on macOS
        size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) >> 20
        duration = time.perf_counter() - start_time
        LOGGER.info(f"Docs built correctly ✅ ({size:.1f}MB, {duration:.1f}s)")

        # Restore sources before optionally serving
        restore_all()

        if (MACOS or LINUX) and not os.getenv("GITHUB_ACTIONS"):
            import webbrowser

            url = "http://localhost:8000"
            LOGGER.info(f"Opening browser at {url}")
            webbrowser.open(url)
            try:
                subprocess.run(["python", "-m", "http.server", "--directory", str(SITE), "8000"], check=True)
            except KeyboardInterrupt:
                LOGGER.info(f"\n✅ Server stopped. Restart at {url}")
            except Exception as e:
                if "Address already in use" in str(e):
                    LOGGER.info("Port 8000 in use; skipping auto-serve. Serve manually if needed.")
                else:
                    LOGGER.info(f"\n❌ Server failed: {e}")
        else:
            LOGGER.info('Serve site at http://localhost:8000 with "python -m http.server --directory site"')
    finally:
        if not restored:
            restore_all()
        shutil.rmtree(DOCS.parent / "hub_sdk", ignore_errors=True)
        shutil.rmtree(DOCS / "repos", ignore_errors=True)


if __name__ == "__main__":
    main()
