# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
This script is designed to facilitate the building of MkDocs documentation for a multilingual site.
It specifically addresses the scenario where the English documentation (located in an 'en' subdirectory)
needs to be temporarily copied to the root level of the documentation directory for MkDocs to properly
build the site. This is a workaround to handle MkDocs' limitations with multilingual documentation.

The script performs the following steps:
1. Copies all files from the 'en' directory to the root of the documentation directory.
2. Builds the MkDocs site using the main mkdocs.yml file and any other mkdocs_*.yml files in the 'docs' directory,
   allowing for building localized versions of the documentation.
3. Cleans up by removing the files that were copied to the root after the build is complete.

Usage:
    Run this script from the root directory of your MkDocs project.
    Ensure mkdocs and other dependencies are installed and accessible in your environment.

Note:
    The script uses Python's standard library modules like shutil and os for file operations
    and subprocess calls, respectively. It's built to be run in an environment where MkDocs
    and Python are already installed and configured.
"""

import os
import shutil
from pathlib import Path

DOCS = Path(__name__).parent


def copy_files(src_dir, dest_dir):
    """Copy files from src_dir to dest_dir and return a list of copied files."""
    copied_files = []
    for item in src_dir.iterdir():
        dest = dest_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
        copied_files.append(dest)
    return copied_files


def remove_files(files):
    """Remove files from the file system."""
    for file in files:
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)


def build_docs():
    """Build docs using mkdocs."""
    site_dir = Path('site')
    if site_dir.exists():
        shutil.rmtree(site_dir)

    # Build the main documentation
    os.system(f'mkdocs build -f {DOCS}/mkdocs.yml')

    # Build other localized documentations
    for file in DOCS.glob('mkdocs_*.yml'):
        print(f"Building MkDocs site with configuration file: {file}")
        os.system(f'mkdocs build -f {file}')


def main():
    # Copy files and remember them
    copied_files = copy_files(DOCS / 'en', DOCS)

    # Build the docs
    build_docs()

    # Remove the copied files
    remove_files(copied_files)


if __name__ == '__main__':
    main()
