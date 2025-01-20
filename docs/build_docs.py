# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Automates the building and post-processing of MkDocs documentation, particularly for projects with multilingual content.
It streamlines the workflow for generating localized versions of the documentation and updating HTML links to ensure
they are correctly formatted.

Key Features:
    - Automated building of MkDocs documentation: The script compiles both the main documentation and
      any localized versions specified in separate MkDocs configuration files.
    - Post-processing of generated HTML files: After the documentation is built, the script updates all
      HTML files to remove the '.md' extension from internal links. This ensures that links in the built
      HTML documentation correctly point to other HTML pages rather than Markdown files, which is crucial
      for proper navigation within the web-based documentation.

Usage:
    - Run the script from the root directory of your MkDocs project.
    - Ensure that MkDocs is installed and that all MkDocs configuration files (main and localized versions)
      are present in the project directory.
    - The script first builds the documentation using MkDocs, then scans the generated HTML files in the 'site'
      directory to update the internal links.
    - It's ideal for projects where the documentation is written in Markdown and needs to be served as a static website.

Note:
    - This script is built to be run in an environment where Python and MkDocs are installed and properly configured.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"  # fix DeprecationWarning: Jupyter is migrating to use standard platformdirs
DOCS = Path(__file__).parent.resolve()
SITE = DOCS.parent / "site"


def create_vercel_config():
    """Create vercel.json in the site directory with customized configuration settings."""
    config = {"trailingSlash": True}
    with open(SITE / "vercel.json", "w") as f:
        json.dump(config, f, indent=2)


def prepare_docs_markdown(clone_repos=True):
    """Build docs using mkdocs."""
    if SITE.exists():
        print(f"Removing existing {SITE}")
        shutil.rmtree(SITE)

    # Get hub-sdk repo
    if clone_repos:
        repo = "https://github.com/ultralytics/hub-sdk"
        local_dir = DOCS.parent / Path(repo).name
        if not local_dir.exists():
            os.system(f"git clone {repo} {local_dir}")
        os.system(f"git -C {local_dir} pull")  # update repo
        shutil.rmtree(DOCS / "en/hub/sdk", ignore_errors=True)  # delete if exists
        shutil.copytree(local_dir / "docs", DOCS / "en/hub/sdk")  # for docs
        shutil.rmtree(DOCS.parent / "hub_sdk", ignore_errors=True)  # delete if exists
        shutil.copytree(local_dir / "hub_sdk", DOCS.parent / "hub_sdk")  # for mkdocstrings
        print(f"Cloned/Updated {repo} in {local_dir}")

    # Add frontmatter
    for file in tqdm((DOCS / "en").rglob("*.md"), desc="Adding frontmatter"):
        update_markdown_files(file)


def update_page_title(file_path: Path, new_title: str):
    """Update the title of an HTML file."""
    # Read the content of the file
    with open(file_path, encoding="utf-8") as file:
        content = file.read()

    # Replace the existing title with the new title
    updated_content = re.sub(r"<title>.*?</title>", f"<title>{new_title}</title>", content)

    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)


def update_html_head(script=""):
    """Update the HTML head section of each file."""
    html_files = Path(SITE).rglob("*.html")
    for html_file in tqdm(html_files, desc="Processing HTML files"):
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


def update_subdir_edit_links(subdir="", docs_url=""):
    """Update the HTML head section of each file."""
    if str(subdir[0]) == "/":
        subdir = str(subdir[0])[1:]
    html_files = (SITE / subdir).rglob("*.html")
    for html_file in tqdm(html_files, desc="Processing subdir files"):
        with html_file.open("r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        # Find the anchor tag and update its href attribute
        a_tag = soup.find("a", {"class": "md-content__button md-icon"})
        if a_tag and a_tag["title"] == "Edit this page":
            a_tag["href"] = f"{docs_url}{a_tag['href'].split(subdir)[-1]}"

        # Write the updated HTML back to the file
        with open(html_file, "w", encoding="utf-8") as file:
            file.write(str(soup))


def update_markdown_files(md_filepath: Path):
    """Creates or updates a Markdown file, ensuring frontmatter is present."""
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
    """Updates titles, edit links, head sections, and converts plaintext links in HTML documentation."""
    # Update 404 titles
    update_page_title(SITE / "404.html", new_title="Ultralytics Docs - Not Found")

    # Update edit links
    update_subdir_edit_links(
        subdir="hub/sdk/",  # do not use leading slash
        docs_url="https://github.com/ultralytics/hub-sdk/tree/main/docs/",
    )

    # Convert plaintext links to HTML hyperlinks
    files_modified = 0
    for html_file in tqdm(SITE.rglob("*.html"), desc="Converting plaintext links"):
        with open(html_file, encoding="utf-8") as file:
            content = file.read()
        updated_content = convert_plaintext_links_to_html(content)
        if updated_content != content:
            with open(html_file, "w", encoding="utf-8") as file:
                file.write(updated_content)
            files_modified += 1
    print(f"Modified plaintext links in {files_modified} files.")

    # Update HTML file head section
    script = ""
    if any(script):
        update_html_head(script)

    # Delete the /macros directory from the built site
    macros_dir = SITE / "macros"
    if macros_dir.exists():
        print(f"Removing /macros directory from site: {macros_dir}")
        shutil.rmtree(macros_dir)


def convert_plaintext_links_to_html(content):
    """Convert plaintext links to HTML hyperlinks in the main content area only."""
    soup = BeautifulSoup(content, "html.parser")

    # Find the main content area (adjust this selector based on your HTML structure)
    main_content = soup.find("main") or soup.find("div", class_="md-content")
    if not main_content:
        return content  # Return original content if main content area not found

    modified = False
    for paragraph in main_content.find_all(["p", "li"]):  # Focus on paragraphs and list items
        for text_node in paragraph.find_all(string=True, recursive=False):
            if text_node.parent.name not in {"a", "code"}:  # Ignore links and code blocks
                new_text = re.sub(
                    r"(https?://[^\s()<>]*[^\s()<>.,:;!?\'\"])",
                    r'<a href="\1">\1</a>',
                    str(text_node),
                )
                if "<a href=" in new_text:
                    # Parse the new text with BeautifulSoup to handle HTML properly
                    new_soup = BeautifulSoup(new_text, "html.parser")
                    text_node.replace_with(new_soup)
                    modified = True

    return str(soup) if modified else content


def remove_macros():
    """Removes the /macros directory and related entries in sitemap.xml from the built site."""
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

    print(f"Removed {len(macros_indices)} URLs containing '/macros/' from {sitemap}")


def minify_files(html=True, css=True, js=True):
    """Minifies HTML, CSS, and JS files and prints total reduction stats."""
    minify, compress, jsmin = None, None, None
    try:
        if html:
            from minify_html import minify
        if css:
            from csscompressor import compress
        if js:
            import jsmin
    except ImportError as e:
        print(f"Missing required package: {str(e)}")
        return

    stats = {}
    for ext, minifier in {
        "html": (lambda x: minify(x, keep_closing_tags=True, minify_css=True, minify_js=True)) if html else None,
        "css": compress if css else None,
        "js": jsmin.jsmin if js else None,
    }.items():
        if not minifier:
            continue

        stats[ext] = {"original": 0, "minified": 0}
        directory = ""  # "stylesheets" if ext == css else "javascript" if ext == "js" else ""
        for f in tqdm((SITE / directory).rglob(f"*.{ext}"), desc=f"Minifying {ext.upper()}"):
            content = f.read_text(encoding="utf-8")
            minified = minifier(content)
            stats[ext]["original"] += len(content)
            stats[ext]["minified"] += len(minified)
            f.write_text(minified, encoding="utf-8")

    for ext, data in stats.items():
        if data["original"]:
            r = data["original"] - data["minified"]  # reduction
            print(f"Total {ext.upper()} reduction: {(r / data['original']) * 100:.2f}% ({r / 1024:.2f} KB saved)")


def main():
    """Builds docs, updates titles and edit links, minifies HTML, and prints local server command."""
    prepare_docs_markdown()

    # Build the main documentation
    print(f"Building docs from {DOCS}")
    subprocess.run(f"mkdocs build -f {DOCS.parent}/mkdocs.yml --strict", check=True, shell=True)
    remove_macros()
    create_vercel_config()
    print(f"Site built at {SITE}")

    # Update docs HTML pages
    update_docs_html()

    # Minify files
    minify_files(html=False, css=False, js=False)

    # Show command to serve built website
    print('Docs built correctly ✅\nServe site at http://localhost:8000 with "python -m http.server --directory site"')


if __name__ == "__main__":
    main()
