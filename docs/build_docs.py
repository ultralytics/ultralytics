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

import os
import re
import shutil
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup

from ultralytics.utils import LINUX, LOGGER, MACOS
from ultralytics.utils.tqdm import TQDM

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"  # fix DeprecationWarning: Jupyter is migrating to use standard platformdirs
DOCS = Path(__file__).parent.resolve()
SITE = DOCS.parent / "site"
LINK_PATTERN = re.compile(r"(https?://[^\s()<>]*[^\s()<>.,:;!?\'\"])")


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
            ["git", "clone", repo, str(local_dir), "--depth", "1", "--single-branch", "--branch", "main"], check=True
        )
        shutil.rmtree(DOCS / "en/hub/sdk", ignore_errors=True)  # delete if exists
        shutil.copytree(local_dir / "docs", DOCS / "en/hub/sdk")  # for docs
        shutil.rmtree(DOCS.parent / "hub_sdk", ignore_errors=True)  # delete if exists
        shutil.copytree(local_dir / "hub_sdk", DOCS.parent / "hub_sdk")  # for mkdocstrings
        LOGGER.info(f"Cloned/Updated {repo} in {local_dir}")

        # Get docs repo
        repo = "https://github.com/ultralytics/docs"
        local_dir = DOCS / "repos" / Path(repo).name
        subprocess.run(
            ["git", "clone", repo, str(local_dir), "--depth", "1", "--single-branch", "--branch", "main"], check=True
        )
        shutil.rmtree(DOCS / "en/compare", ignore_errors=True)  # delete if exists
        shutil.copytree(local_dir / "docs/en/compare", DOCS / "en/compare")  # for docs
        LOGGER.info(f"Cloned/Updated {repo} in {local_dir}")

    # Add frontmatter
    for file in TQDM((DOCS / "en").rglob("*.md"), desc="Adding frontmatter"):
        update_markdown_files(file)


def update_page_title(file_path: Path, new_title: str):
    """Update the title of an HTML file."""
    with open(file_path, encoding="utf-8") as file:
        content = file.read()

    # Replace the existing title with the new title
    updated_content = re.sub(r"<title>.*?</title>", f"<title>{new_title}</title>", content)

    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)


def update_html_head(script: str = ""):
    """Update the HTML head section of each file."""
    html_files = Path(SITE).rglob("*.html")
    for html_file in TQDM(html_files, desc="Processing HTML files"):
        with html_file.open("r", encoding="utf-8") as file:
            html_content = file.read()

        if script in html_content:  # script already in HTML file
            return

        head_end_index = html_content.lower().rfind("</head>")
        if head_end_index != -1:
            # Add the specified JavaScript to the HTML file just before the end of the head tag.
            new_html_content = html_content[:head_end_index] + script + html_content[head_end_index:]
            with html_file.open("w", encoding="utf-8") as file:
                file.write(new_html_content)


def update_subdir_edit_links(subdir: str = "", docs_url: str = ""):
    """Update the HTML head section of each file."""
    if str(subdir[0]) == "/":
        subdir = str(subdir[0])[1:]
    html_files = (SITE / subdir).rglob("*.html")
    for html_file in TQDM(html_files, desc="Processing subdir files", mininterval=1.0):
        with html_file.open("r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        # Find the anchor tag and update its href attribute
        a_tag = soup.find("a", {"class": "md-content__button md-icon"})
        if a_tag and a_tag["title"] == "Edit this page":
            a_tag["href"] = f"{docs_url}{a_tag['href'].rpartition(subdir)[-1]}"

        # Write the updated HTML back to the file
        with open(html_file, "w", encoding="utf-8") as file:
            file.write(str(soup))


def update_markdown_files(md_filepath: Path):
    """Create or update a Markdown file, ensuring frontmatter is present."""
    if md_filepath.exists():
        content = md_filepath.read_text().strip()

        # Replace apostrophes
        content = content.replace("‘", "'").replace("’", "'")

        # Add frontmatter if missing
        if not content.strip().startswith("---\n") and "macros" not in md_filepath.parts:  # skip macros directory
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
    """Update titles, edit links, head sections, and convert plaintext links in HTML documentation."""
    # Update 404 titles
    update_page_title(SITE / "404.html", new_title="Ultralytics Docs - Not Found")

    # Update edit button links
    for subdir, docs_url in (
        ("hub/sdk/", "https://github.com/ultralytics/hub-sdk/tree/main/docs/"),  # do not use leading slash
        ("compare/", "https://github.com/ultralytics/docs/tree/main/docs/en/compare/"),
    ):
        update_subdir_edit_links(subdir=subdir, docs_url=docs_url)

    # Convert plaintext links to HTML hyperlinks
    files_modified = 0
    for html_file in TQDM(SITE.rglob("*.html"), desc="Updating bs4 soup", mininterval=1.0):
        with open(html_file, encoding="utf-8") as file:
            content = file.read()
        updated_content = update_docs_soup(content, html_file=html_file)
        if updated_content != content:
            with open(html_file, "w", encoding="utf-8") as file:
                file.write(updated_content)
            files_modified += 1
    LOGGER.info(f"Modified bs4 soup in {files_modified} files.")

    # Update HTML file head section
    script = ""
    if any(script):
        update_html_head(script)

    # Delete the /macros directory from the built site
    macros_dir = SITE / "macros"
    if macros_dir.exists():
        LOGGER.info(f"Removing /macros directory from site: {macros_dir}")
        shutil.rmtree(macros_dir)


def update_docs_soup(content: str, html_file: Path = None, max_title_length: int = 70) -> str:
    """Convert plaintext links to HTML hyperlinks, truncate long meta titles, and remove code line hrefs."""
    soup = BeautifulSoup(content, "html.parser")
    modified = False

    # Truncate long meta title if needed
    title_tag = soup.find("title")
    if title_tag and len(title_tag.text) > max_title_length and "-" in title_tag.text:
        title_tag.string = title_tag.text.rsplit("-", 1)[0].strip()
        modified = True

    # Find the main content area
    main_content = soup.find("main") or soup.find("div", class_="md-content")
    if not main_content:
        return str(soup) if modified else content

    # Convert plaintext links to HTML hyperlinks
    for paragraph in main_content.select("p, li"):
        for text_node in paragraph.find_all(string=True, recursive=False):
            if text_node.parent.name not in {"a", "code"}:
                new_text = LINK_PATTERN.sub(r'<a href="\1">\1</a>', str(text_node))
                if "<a href=" in new_text:
                    text_node.replace_with(BeautifulSoup(new_text, "html.parser"))
                    modified = True

    # Remove href attributes from code line numbers in code blocks
    for a in soup.select('a[href^="#__codelineno-"], a[id^="__codelineno-"]'):
        if a.string:  # If the a tag has text (the line number)
            # Check if parent is a span with class="normal"
            if a.parent and a.parent.name == "span" and "normal" in a.parent.get("class", []):
                del a.parent["class"]
            a.replace_with(a.string)  # Replace with just the text
        else:  # If it has no text
            a.replace_with(soup.new_tag("span"))  # Replace with an empty span
        modified = True

    return str(soup) if modified else content


def remove_macros():
    """Remove the /macros directory and related entries in sitemap.xml from the built site."""
    shutil.rmtree(SITE / "macros", ignore_errors=True)
    (SITE / "sitemap.xml.gz").unlink(missing_ok=True)

    # Process sitemap.xml
    sitemap = SITE / "sitemap.xml"
    lines = sitemap.read_text(encoding="utf-8").splitlines(keepends=True)

    # Find indices of '/macros/' lines
    macros_indices = [i for i, line in enumerate(lines) if "/macros/" in line]

    # Create a set of indices to remove (including lines before and after)
    indices_to_remove = set()
    for i in macros_indices:
        indices_to_remove.update(range(i - 1, i + 3))  # i-1, i, i+1, i+2, i+3

    # Create new list of lines, excluding the ones to remove
    new_lines = [line for i, line in enumerate(lines) if i not in indices_to_remove]

    # Write the cleaned content back to the file
    sitemap.write_text("".join(new_lines), encoding="utf-8")

    LOGGER.info(f"Removed {len(macros_indices)} URLs containing '/macros/' from {sitemap}")


def remove_comments_and_empty_lines(content: str, file_type: str) -> str:
    """
    Remove comments and empty lines from a string of code, preserving newlines and URLs.

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
        # Remove HTML comments
        content = re.sub(r"<!--[\s\S]*?-->", "", content)
        # Only remove empty lines for HTML, preserve indentation
        content = re.sub(r"^\s*$\n", "", content, flags=re.MULTILINE)
    elif file_type == "css":
        # Remove CSS comments
        content = re.sub(r"/\*[\s\S]*?\*/", "", content)
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

        # Safe space removal around punctuation and operators (NEVER include colons - breaks JS)
        content = re.sub(r"\s*([,;{}])\s*", r"\1", content)
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
        LOGGER.info(f"Missing required package: {str(e)}")
        return

    stats = {}
    for ext, minifier in {
        "html": (lambda x: minify(x, keep_closing_tags=True, minify_css=True, minify_js=True)) if html else None,
        "css": compress if css else None,
        "js": jsmin.jsmin if js else None,
    }.items():
        stats[ext] = {"original": 0, "minified": 0}
        directory = ""  # "stylesheets" if ext == css else "javascript" if ext == "js" else ""
        for f in TQDM((SITE / directory).rglob(f"*.{ext}"), desc=f"Minifying {ext.upper()}", mininterval=1.0):
            content = f.read_text(encoding="utf-8")
            minified = minifier(content) if minifier else remove_comments_and_empty_lines(content, ext)
            stats[ext]["original"] += len(content)
            stats[ext]["minified"] += len(minified)
            f.write_text(minified, encoding="utf-8")

    for ext, data in stats.items():
        if data["original"]:
            r = data["original"] - data["minified"]  # reduction
            LOGGER.info(f"Total {ext.upper()} reduction: {(r / data['original']) * 100:.2f}% ({r / 1024:.2f} KB saved)")


def main():
    """Build docs, update titles and edit links, minify HTML, and print local server command."""
    prepare_docs_markdown()

    # Build the main documentation
    LOGGER.info(f"Building docs from {DOCS}")
    subprocess.run(["mkdocs", "build", "-f", str(DOCS.parent / "mkdocs.yml"), "--strict"], check=True)
    remove_macros()
    LOGGER.info(f"Site built at {SITE}")

    # Update docs HTML pages
    update_docs_html()

    # Minify files
    minify_files(html=False, css=False, js=False)

    # Cleanup
    shutil.rmtree(DOCS.parent / "hub_sdk", ignore_errors=True)
    shutil.rmtree(DOCS / "repos", ignore_errors=True)

    # Print results and auto-serve on macOS
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) >> 20
    LOGGER.info(f"Docs built correctly ✅ ({size:.1f} MB)")

    if (MACOS or LINUX) and not os.getenv("GITHUB_ACTIONS"):
        import webbrowser

        webbrowser.open("http://localhost:8000")
        try:
            subprocess.run(["python", "-m", "http.server", "--directory", str(SITE), "8000"], check=True)
        except KeyboardInterrupt:
            LOGGER.info("\nServer stopped.")
    else:
        LOGGER.info('Serve site at http://localhost:8000 with "python -m http.server --directory site"')


if __name__ == "__main__":
    main()
