# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Helper file to build Ultralytics Docs reference section. Recursively walks through the 'ultralytics' directory and
builds an MkDocs reference section of *.md files. These files are composed of classes and functions, and also creates a
navigation menu for use in the mkdocs.yaml file.

Note: This script must be run from the repository root directory, not the 'docs' directory.
"""

import re  # Regular expression library to search for patterns in text
import subprocess  # Used to run Git commands
from collections import defaultdict  # Provides a default dictionary structure
from pathlib import Path  # For handling file system paths in a platform-independent way

# Constants to determine which directory to process (hub_sdk or ultralytics)
hub_sdk = False  # Switch to determine which package is being worked on

# Conditional paths depending on the value of `hub_sdk`
if hub_sdk:
    PACKAGE_DIR = Path("/Users/glennjocher/PycharmProjects/hub-sdk/hub_sdk")
    REFERENCE_DIR = PACKAGE_DIR.parent / "docs/reference"
    GITHUB_REPO = "ultralytics/hub-sdk"
else:
    FILE = Path(__file__).resolve()  # Get the path of the current file
    PACKAGE_DIR = FILE.parents[1] / "ultralytics"  # Define the root directory for 'ultralytics'
    REFERENCE_DIR = PACKAGE_DIR.parent / "docs/en/reference"  # Directory where the reference files will be stored
    GITHUB_REPO = "ultralytics/ultralytics"  # GitHub repository link


def extract_classes_and_functions(filepath: Path) -> tuple:
    """Extracts class and function names from a given Python file."""
    content = filepath.read_text()  # Read the file contents
    class_pattern = r"(?:^|\n)class\s(\w+)(?:\(|:)"  # Regular expression to find class names
    func_pattern = r"(?:^|\n)def\s(\w+)\("  # Regular expression to find function names

    # Find all class and function names in the file
    classes = re.findall(class_pattern, content)
    functions = re.findall(func_pattern, content)

    return classes, functions  # Return the lists of classes and functions


def create_markdown(py_filepath: Path, module_path: str, classes: list, functions: list):
    """Creates a Markdown file containing the API reference for the given Python module."""
    md_filepath = py_filepath.with_suffix(".md")  # Change the file extension to .md
    exists = md_filepath.exists()  # Check if the Markdown file already exists

    # Read existing content and keep header content (YAML front matter) between the first two ---
    header_content = ""
    if exists:
        existing_content = md_filepath.read_text()  # Read the existing file
        header_parts = existing_content.split("---")  # Split the file into parts by ---
        for part in header_parts:  # Keep only the header with metadata like description or comments
            if "description:" in part or "comments:" in part:
                header_content += f"---{part}---\n\n"
    if not any(header_content):  # If there's no header, add a default one
        header_content = "---\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"

    # Format the module name and URL paths for the GitHub repo
    module_name = module_path.replace(".__init__", "")  # Handle __init__.py special case
    module_path = module_path.replace(".", "/")  # Convert module path to file system style
    url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path}.py"  # GitHub link to the file
    edit = f"https://github.com/{GITHUB_REPO}/edit/main/{module_path}.py"  # GitHub link to edit the file
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")  # Properly display __init__.py filenames in Markdown

    # Create the title and reference links content for the Markdown file
    title_content = (
        f"# Reference for `{module_path}.py`\n\n"
        f"!!! note\n\n"
        f"    This file is available at [{pretty}]({url}). If you spot a problem please help fix it by [contributing]"
        f"(https://docs.ultralytics.com/help/contributing/) a [Pull Request]({edit}) 🛠️. Thank you 🙏!\n\n"
    )

    # Create the markdown content for the classes and functions found in the Python file
    md_content = ["<br>\n"] + [f"## ::: {module_name}.{class_name}\n\n<br><br><hr><br>\n" for class_name in classes]
    md_content.extend(f"## ::: {module_name}.{func_name}\n\n<br><br><hr><br>\n" for func_name in functions)

    # Remove the last horizontal line for neatness
    md_content[-1] = md_content[-1].replace("<hr><br>", "")

    # Combine header, title, and content into one complete Markdown file
    md_content = header_content + title_content + "\n".join(md_content)
    if not md_content.endswith("\n"):
        md_content += "\n"

    # Ensure the directory exists and write the new Markdown file
    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    md_filepath.write_text(md_content)

    # If the file didn't exist before, add it to Git
    if not exists:
        print(f"Created new file '{md_filepath}'")
        subprocess.run(["git", "add", "-f", str(md_filepath)], check=True, cwd=PACKAGE_DIR)

    return md_filepath.relative_to(PACKAGE_DIR.parent)  # Return the relative path of the markdown file


def nested_dict() -> defaultdict:
    """Creates and returns a nested defaultdict."""
    return defaultdict(nested_dict)  # A recursive defaultdict structure


def sort_nested_dict(d: dict) -> dict:
    """Sorts a nested dictionary recursively."""
    return {key: sort_nested_dict(value) if isinstance(value, dict) else value for key, value in sorted(d.items())}


def create_nav_menu_yaml(nav_items: list, save: bool = False):
    """Creates a YAML file for the navigation menu based on the provided list of items."""
    nav_tree = nested_dict()  # Create a nested dictionary to represent the navigation structure

    # Loop through each Markdown file's relative path and populate the navigation tree
    for item_str in nav_items:
        item = Path(item_str)  # Convert string path to Path object
        parts = item.parts  # Split the path into parts
        current_level = nav_tree["reference"]  # Start at the reference section
        for part in parts[2:-1]:  # Skip 'docs' and 'reference', and the last part (filename)
            current_level = current_level[part]  # Traverse into the nested dictionary

        # Add the final Markdown file as a navigation item
        md_file_name = parts[-1].replace(".md", "")
        current_level[md_file_name] = item

    # Sort the navigation tree for consistency
    nav_tree_sorted = sort_nested_dict(nav_tree)

    # Convert the nested dictionary to a YAML-formatted string
    def _dict_to_yaml(d, level=0):
        """Converts a nested dictionary to a YAML-formatted string with indentation."""
        yaml_str = ""
        indent = "  " * level  # Indentation for YAML
        for k, v in d.items():
            if isinstance(v, dict):
                yaml_str += f"{indent}- {k}:\n{_dict_to_yaml(v, level + 1)}"
            else:
                yaml_str += f"{indent}- {k}: {str(v).replace('docs/en/', '')}\n"
        return yaml_str

    # Print the new navigation structure
    print("Scan complete, new mkdocs.yaml reference section is:\n\n", _dict_to_yaml(nav_tree_sorted))

    # Optionally save the new navigation structure to a YAML file
    if save:
        (PACKAGE_DIR.parent / "nav_menu_updated.yml").write_text(_dict_to_yaml(nav_tree_sorted))


def main():
    """Main function to extract class and function names, create Markdown files, and generate a YAML navigation menu."""
    nav_items = []  # List to hold the paths of all generated Markdown files

    # Recursively search for all Python files in the package directory
    for py_filepath in PACKAGE_DIR.rglob("*.py"):
        classes, functions = extract_classes_and_functions(py_filepath)  # Extract classes and functions

        if classes or functions:  # If there are any classes or functions, generate a Markdown file
            py_filepath_rel = py_filepath.relative_to(PACKAGE_DIR)  # Get the relative path of the Python file
            md_filepath = REFERENCE_DIR / py_filepath_rel  # Define the path where the Markdown file will be saved
            module_path = (
                f"{PACKAGE_DIR.name}.{py_filepath_rel.with_suffix('').as_posix().replace('/', '.')}"  # Get module path
            )
            md_rel_filepath = create_markdown(md_filepath, module_path, classes, functions)  # Create the Markdown
            nav_items.append(str(md_rel_filepath))  # Add the Markdown file to the nav items list

    # Create the navigation menu in YAML format
    create_nav_menu_yaml(nav_items)


if __name__ == "__main__":
    main()  # Run the main function if this script is executed directly
