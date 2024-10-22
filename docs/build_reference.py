# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Helper file to build Ultralytics Docs reference section. Recursively walks through ultralytics dir and builds an MkDocs
reference section of *.md files composed of classes and functions, and also creates a nav menu for use in mkdocs.yaml.

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
    PACKAGE_DIR = FILE.parents[1] / "ultralytics"  # i.e. /Users/glennjocher/PycharmProjects/ultralytics/ultralytics
    REFERENCE_DIR = PACKAGE_DIR.parent / "docs/en/reference"
    GITHUB_REPO = "ultralytics/ultralytics"


def extract_classes_and_functions(filepath: Path) -> tuple:
    """Extracts class and function names from a given Python file."""
    content = filepath.read_text()
    class_pattern = r"(?:^|\n)class\s(\w+)(?:\(|:)"
    func_pattern = r"(?:^|\n)def\s(\w+)\("

    classes = re.findall(class_pattern, content)
    functions = re.findall(func_pattern, content)

    return classes, functions


def create_markdown(py_filepath: Path, module_path: str, classes: list, functions: list):
    """Creates a Markdown file containing the API reference for the given Python module."""
    md_filepath = py_filepath.with_suffix(".md")
    exists = md_filepath.exists()

    # Read existing content and keep header content between first two ---
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
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")  # properly display __init__.py filenames
    title_content = (
        f"# Reference for `{module_path}.py`\n\n"
        f"!!! note\n\n"
        f"    This file is available at [{pretty}]({url}). If you spot a problem please help fix it by [contributing]"
        f"(https://docs.ultralytics.com/help/contributing/) a [Pull Request]({edit}) 🛠️. Thank you 🙏!\n\n"
    )
    md_content = ["<br>\n"] + [f"## ::: {module_name}.{class_name}\n\n<br><br><hr><br>\n" for class_name in classes]
    md_content.extend(f"## ::: {module_name}.{func_name}\n\n<br><br><hr><br>\n" for func_name in functions)
    md_content[-1] = md_content[-1].replace("<hr><br>", "")  # remove last horizontal line
    md_content = header_content + title_content + "\n".join(md_content)
    if not md_content.endswith("\n"):
        md_content += "\n"

    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    md_filepath.write_text(md_content)

    if not exists:
        # Add new markdown file to the git staging area
        print(f"Created new file '{md_filepath}'")
        subprocess.run(["git", "add", "-f", str(md_filepath)], check=True, cwd=PACKAGE_DIR)

    return md_filepath.relative_to(PACKAGE_DIR.parent)


def nested_dict() -> defaultdict:
    """Creates and returns a nested defaultdict."""
    return defaultdict(nested_dict)


def sort_nested_dict(d: dict) -> dict:
    """Sorts a nested dictionary recursively."""
    return {key: sort_nested_dict(value) if isinstance(value, dict) else value for key, value in sorted(d.items())}


def create_nav_menu_yaml(nav_items: list, save: bool = False):
    """Creates a YAML file for the navigation menu based on the provided list of items."""
    nav_tree = nested_dict()

    for item_str in nav_items:
        item = Path(item_str)
        parts = item.parts
        current_level = nav_tree["reference"]
        for part in parts[2:-1]:  # skip the first two parts (docs and reference) and the last part (filename)
            current_level = current_level[part]

        md_file_name = parts[-1].replace(".md", "")
        current_level[md_file_name] = item

    nav_tree_sorted = sort_nested_dict(nav_tree)

    def _dict_to_yaml(d, level=0):
        """Converts a nested dictionary to a YAML-formatted string with indentation."""
        yaml_str = ""
        indent = "  " * level
        for k, v in d.items():
            if isinstance(v, dict):
                yaml_str += f"{indent}- {k}:\n{_dict_to_yaml(v, level + 1)}"
            else:
                yaml_str += f"{indent}- {k}: {str(v).replace('docs/en/', '')}\n"
        return yaml_str

    # Print updated YAML reference section
    print("Scan complete, new mkdocs.yaml reference section is:\n\n", _dict_to_yaml(nav_tree_sorted))

    # Save new YAML reference section
    if save:
        (PACKAGE_DIR.parent / "nav_menu_updated.yml").write_text(_dict_to_yaml(nav_tree_sorted))


def main():
    """Main function to extract class and function names, create Markdown files, and generate a YAML navigation menu."""
    nav_items = []

    for py_filepath in PACKAGE_DIR.rglob("*.py"):
        classes, functions = extract_classes_and_functions(py_filepath)

        if classes or functions:
            py_filepath_rel = py_filepath.relative_to(PACKAGE_DIR)
            md_filepath = REFERENCE_DIR / py_filepath_rel
            module_path = f"{PACKAGE_DIR.name}.{py_filepath_rel.with_suffix('').as_posix().replace('/', '.')}"
            md_rel_filepath = create_markdown(md_filepath, module_path, classes, functions)
            nav_items.append(str(md_rel_filepath))

    create_nav_menu_yaml(nav_items)


if __name__ == "__main__":
    main()
