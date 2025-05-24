# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Helper file to build Ultralytics Docs reference section.

This script recursively walks through the ultralytics directory and builds an MkDocs reference section of *.md files
composed of classes and functions, and also creates a navigation menu for use in mkdocs.yaml.

Note: Must be run from repository root directory. Do not run from docs directory.
"""

import re
import subprocess
from collections import defaultdict
from pathlib import Path

# Constants
hub_sdk = False
if hub_sdk:
    PACKAGE_DIR = Path("/Users/glennjocher/PycharmProjects/hub-sdk/hub_sdk")
    REFERENCE_DIR = PACKAGE_DIR.parent / "docs/reference"
    GITHUB_REPO = "ultralytics/hub-sdk"
else:
    FILE = Path(__file__).resolve()
    PACKAGE_DIR = FILE.parents[1] / "ultralytics"
    REFERENCE_DIR = PACKAGE_DIR.parent / "docs/en/reference"
    GITHUB_REPO = "ultralytics/ultralytics"

MKDOCS_YAML = PACKAGE_DIR.parent / "mkdocs.yml"


def extract_classes_and_functions(filepath: Path) -> tuple[list[str], list[str]]:
    """Extract class and function names from a given Python file."""
    content = filepath.read_text()
    return (re.findall(r"(?:^|\n)class\s(\w+)(?:\(|:)", content), re.findall(r"(?:^|\n)def\s(\w+)\(", content))


def create_markdown(py_filepath: Path, module_path: str, classes: list[str], functions: list[str]) -> Path:
    """Create a Markdown file containing the API reference for the given Python module."""
    md_filepath = py_filepath.with_suffix(".md")
    exists = md_filepath.exists()

    # Read existing content and retain header metadata if available
    header_content = ""
    if exists:
        existing_content = md_filepath.read_text()
        header_parts = existing_content.split("---")
        for part in header_parts:
            if "description:" in part or "comments:" in part:
                header_content += f"---{part}---\n\n"
    if not any(header_content):
        header_content = "---\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"

    module_name = module_path.replace(".__init__", "")
    module_path = module_path.replace(".", "/")
    url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path}.py"
    edit = f"https://github.com/{GITHUB_REPO}/edit/main/{module_path}.py"
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")  # Properly display __init__.py filenames

    # Build markdown content
    title_content = (
        f"# Reference for `{module_path}.py`\n\n"
        f"!!! note\n\n"
        f"    This file is available at [{pretty}]({url}). If you spot a problem please help fix it by [contributing]"
        f"(https://docs.ultralytics.com/help/contributing/) a [Pull Request]({edit}) 🛠️. Thank you 🙏!\n\n"
    )
    md_content = ["<br>\n\n"]
    md_content.extend(f"## ::: {module_name}.{cls}\n\n<br><br><hr><br>\n\n" for cls in classes)
    md_content.extend(f"## ::: {module_name}.{func}\n\n<br><br><hr><br>\n\n" for func in functions)
    if md_content[-1:]:  # Remove last horizontal rule if content exists
        md_content[-1] = md_content[-1].replace("<hr><br>\n\n", "")

    # Write to file
    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    md_filepath.write_text(header_content + title_content + "".join(md_content) + "\n")

    if not exists:
        print(f"Created new file '{md_filepath}'")
        subprocess.run(["git", "add", "-f", str(md_filepath)], check=True, cwd=PACKAGE_DIR)

    return md_filepath.relative_to(PACKAGE_DIR.parent)


def nested_dict():
    """Create and return a nested defaultdict."""
    return defaultdict(nested_dict)


def sort_nested_dict(d: dict) -> dict:
    """Sort a nested dictionary recursively."""
    return {k: sort_nested_dict(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}


def create_nav_menu_yaml(nav_items: list[str]) -> str:
    """Create and return a YAML string for the navigation menu."""
    nav_tree = nested_dict()

    for item_str in nav_items:
        item = Path(item_str)
        parts = item.parts
        current_level = nav_tree["reference"]
        for part in parts[2:-1]:  # Skip docs/reference and filename
            current_level = current_level[part]
        current_level[parts[-1].replace(".md", "")] = item

    def _dict_to_yaml(d, level=0):
        """Convert a nested dictionary to a YAML-formatted string with indentation."""
        yaml_str = ""
        indent = "  " * level
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                yaml_str += f"{indent}- {k}:\n{_dict_to_yaml(v, level + 1)}"
            else:
                yaml_str += f"{indent}- {k}: {str(v).replace('docs/en/', '')}\n"
        return yaml_str

    reference_yaml = _dict_to_yaml(sort_nested_dict(nav_tree))
    print(f"Scan complete, generated reference section with {len(reference_yaml.splitlines())} lines")
    return reference_yaml


def extract_document_paths(yaml_section: str) -> list[str]:
    """Extract document paths from a YAML section, ignoring formatting and structure."""
    paths = []
    # Match all paths that appear after a colon in the YAML
    path_matches = re.findall(r":\s*([^\s][^:\n]*?)(?:\n|$)", yaml_section)
    for path in path_matches:
        # Clean up the path
        path = path.strip()
        if path and not path.startswith("-") and not path.endswith(":"):
            paths.append(path)
    return sorted(paths)


def update_mkdocs_file(reference_yaml: str) -> None:
    """Update the mkdocs.yaml file with the new reference section only if changes in document paths are detected."""
    mkdocs_content = MKDOCS_YAML.read_text()

    # Find the top-level Reference section
    ref_pattern = r"(\n  - Reference:[\s\S]*?)(?=\n  - \w|$)"
    ref_match = re.search(ref_pattern, mkdocs_content)

    # Build new section with proper indentation
    new_section_lines = ["\n  - Reference:"]
    for line in reference_yaml.splitlines():
        if line.strip() == "- reference:":  # Skip redundant header
            continue
        new_section_lines.append(f"    {line}")
    new_ref_section = "\n".join(new_section_lines) + "\n"

    if ref_match:
        # We found an existing Reference section
        ref_section = ref_match.group(1)
        print(f"Found existing top-level Reference section ({len(ref_section)} chars)")

        # Compare only document paths
        existing_paths = extract_document_paths(ref_section)
        new_paths = extract_document_paths(new_ref_section)

        # Check if the document paths are the same (ignoring structure or formatting differences)
        if len(existing_paths) == len(new_paths) and set(existing_paths) == set(new_paths):
            print(f"No changes detected in document paths ({len(existing_paths)} items). Skipping update.")
            return

        print(f"Changes detected: {len(new_paths)} document paths vs {len(existing_paths)} existing")

        # Update content
        new_content = mkdocs_content.replace(ref_section, new_ref_section)
        MKDOCS_YAML.write_text(new_content)
        subprocess.run(["npx", "prettier", "--write", str(MKDOCS_YAML)], check=False, cwd=PACKAGE_DIR.parent)
        print(f"Updated Reference section in {MKDOCS_YAML}")
    else:
        # No existing Reference section, we need to add it
        help_match = re.search(r"(\n  - Help:)", mkdocs_content)
        if help_match:
            help_section = help_match.group(1)
            # Insert before Help section
            new_content = mkdocs_content.replace(help_section, f"{new_ref_section}{help_section}")
            MKDOCS_YAML.write_text(new_content)
            print(f"Added new Reference section before Help in {MKDOCS_YAML}")
        else:
            print("Could not find a suitable location to add Reference section")


def main():
    """Extract class/function names, create Markdown files, and update mkdocs.yaml."""
    nav_items = []

    for py_filepath in PACKAGE_DIR.rglob("*.py"):
        classes, functions = extract_classes_and_functions(py_filepath)
        if classes or functions:
            py_filepath_rel = py_filepath.relative_to(PACKAGE_DIR)
            md_filepath = REFERENCE_DIR / py_filepath_rel
            module_path = f"{PACKAGE_DIR.name}.{py_filepath_rel.with_suffix('').as_posix().replace('/', '.')}"
            md_rel_filepath = create_markdown(md_filepath, module_path, classes, functions)
            nav_items.append(str(md_rel_filepath))

    # Update mkdocs.yaml with generated YAML
    update_mkdocs_file(create_nav_menu_yaml(nav_items))


if __name__ == "__main__":
    main()
