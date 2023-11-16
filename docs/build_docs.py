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
from pathlib import Path

DOCS = Path(__file__).parent.resolve()
SITE = DOCS.parent / 'site'


def build_docs():
    """Build docs using mkdocs."""
    if SITE.exists():
        print(f'Removing existing {SITE}')
        shutil.rmtree(SITE)

    # Build the main documentation
    print(f'Building docs from {DOCS}')
    os.system(f'mkdocs build -f {DOCS}/mkdocs.yml')

    # Build other localized documentations
    for file in DOCS.glob('mkdocs_*.yml'):
        print(f'Building MkDocs site with configuration file: {file}')
        os.system(f'mkdocs build -f {file}')
    print(f'Site built at {SITE}')


def update_html_links():
    """Update href links in HTML files to remove '.md'."""
    html_files = SITE.rglob('*.html')
    for html_file in html_files:
        with open(html_file, 'r+', encoding='utf-8') as file:
            content = file.read()
            updated_content = re.sub(r'href="([^"]+)\.md"', r'href="\1"', content)
            file.seek(0)
            file.write(updated_content)
            file.truncate()


def main():
    # Build the docs
    build_docs()

    # Update .md in href links
    update_html_links()

    # Show command to serve built website
    print('Serve site at http://localhost:8000 with "python -m http.server --directory site"')


if __name__ == '__main__':
    main()
