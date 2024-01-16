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

DOCS = Path(__file__).parent.resolve()
SITE = DOCS.parent / "site"
LANGUAGES = True


def build_docs():
    """Build docs using mkdocs."""
    if SITE.exists():
        print(f"Removing existing {SITE}")
        shutil.rmtree(SITE)

    # Build the main documentation
    print(f"Building docs from {DOCS}")
    subprocess.run(f"mkdocs build -f {DOCS}/mkdocs.yml", check=True, shell=True)

    # Build other localized documentations
    if LANGUAGES:
        for file in DOCS.glob("mkdocs_*.yml"):
            print(f"Building MkDocs site with configuration file: {file}")
            subprocess.run(f"mkdocs build -f {file}", check=True, shell=True)
    print(f"Site built at {SITE}")


def update_html_links():
    """Update href links in HTML files to remove '.md' and '/index.md', excluding links starting with 'https://'."""
    html_files = Path(SITE).rglob("*.html")
    total_updated_links = 0

    for html_file in html_files:
        with open(html_file, "r+", encoding="utf-8") as file:
            content = file.read()
            # Find all links to be updated, excluding those starting with 'https://'
            links_to_update = re.findall(r'href="(?!https://)([^"]+?)(/index)?\.md"', content)

            # Update the content and count the number of links updated
            updated_content, number_of_links_updated = re.subn(
                r'href="(?!https://)([^"]+?)(/index)?\.md"', r'href="\1"', content
            )
            total_updated_links += number_of_links_updated

            # Special handling for '/index' links
            updated_content, number_of_index_links_updated = re.subn(
                r'href="([^"]+)/index"', r'href="\1/"', updated_content
            )
            total_updated_links += number_of_index_links_updated

            # Write the updated content back to the file
            file.seek(0)
            file.write(updated_content)
            file.truncate()

            # Print updated links for this file
            for link in links_to_update:
                print(f"Updated link in {html_file}: {link[0]}")

    print(f"Total number of links updated: {total_updated_links}")


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


def update_html_head(key=""):
    """Update the HTML head section of each file."""
    html_files = Path(SITE).rglob("*.html")
    for html_file in tqdm(html_files, desc="Processing HTML files"):
        with html_file.open("r", encoding="utf-8") as file:
            html_content = file.read()

        script = f"""
<script type="text/javascript" src="https://cdn.weglot.com/weglot.min.js"></script>
<script>
    Weglot.initialize({{
        api_key: '{key}'
    }});
</script>
"""
        if script in html_content:  # script already in HTML file
            return

        head_end_index = html_content.lower().rfind("</head>")
        if head_end_index != -1:
            # Add the specified JavaScript to the HTML file just before the end of the head tag.
            new_html_content = html_content[:head_end_index] + script + html_content[head_end_index:]
            with html_file.open("w", encoding="utf-8") as file:
                file.write(new_html_content)


def main():
    # Build the docs
    build_docs()

    # Update titles
    update_page_title(SITE / "404.html", new_title="Ultralytics Docs - Not Found")

    # Update .md in href links
    if LANGUAGES:
        update_html_links()

    # Update HTML file head section
    key = os.environ.get("WEGLOT_KEY")
    if not LANGUAGES and key:
        update_html_head(key)

    # Show command to serve built website
    print('Serve site at http://localhost:8000 with "python -m http.server --directory site"')


if __name__ == "__main__":
    main()
