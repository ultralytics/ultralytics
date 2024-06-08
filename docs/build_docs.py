# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
This Python script is designed to automate the building and post-processing of MkDocs documentation, particularly for
projects with multilingual content. It streamlines the workflow for generating localized versions of the documentation
and updating HTML links to ensure they are correctly formatted.

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

import os
import re
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

os.environ["JUPYTER_PLATFORM_DIRS"] = "1"  # fix DeprecationWarning: Jupyter is migrating to use standard platformdirs
DOCS = Path(__file__).parent.resolve()
SITE = DOCS.parent / "site"


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
    from bs4 import BeautifulSoup

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
        if not content.strip().startswith("---\n"):
            header = "---\ncomments: true\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"
            content = header + content

        # Add EOF newline if missing
        if not content.endswith("\n"):
            content += "\n"

        # Save page
        md_filepath.write_text(content)
    return


def update_docs_html():
    """Updates titles, edit links and head sections of HTML documentation for improved accessibility and relevance."""
    update_page_title(SITE / "404.html", new_title="Ultralytics Docs - Not Found")

    # Update edit links
    update_subdir_edit_links(
        subdir="hub/sdk/",  # do not use leading slash
        docs_url="https://github.com/ultralytics/hub-sdk/tree/main/docs/",
    )

    # Update HTML file head section
    script = ""
    if any(script):
        update_html_head(script)


def main():
    """Builds docs, updates titles and edit links, and prints local server command."""
    prepare_docs_markdown()

    # Build the main documentation
    print(f"Building docs from {DOCS}")
    subprocess.run(f"mkdocs build -f {DOCS.parent}/mkdocs.yml --strict", check=True, shell=True)
    print(f"Site built at {SITE}")

    # Update docs HTML pages
    update_docs_html()

    # Show command to serve built website
    print('Docs built correctly ✅\nServe site at http://localhost:8000 with "python -m http.server --directory site"')


if __name__ == "__main__":
    main()
