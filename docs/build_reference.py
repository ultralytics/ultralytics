# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Helper file to build Ultralytics Docs reference section.

This script recursively walks through the ultralytics directory and builds a MkDocs reference section of *.md files
composed of classes and functions, and also creates a navigation menu for use in mkdocs.yaml.

Note: Must be run from repository root directory. Do not run from docs directory.
"""

from __future__ import annotations

import ast
import html
import re
import subprocess
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from ultralytics.utils.tqdm import TQDM

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
INCLUDE_SPECIAL_METHODS = {
    "__call__",
    "__dir__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__next__",
    "__getattr__",
}
PROPERTY_DECORATORS = {"property", "cached_property"}


@dataclass
class ParameterDoc:
    """Structured documentation for parameters, attributes, and exceptions."""

    name: str
    type: str | None
    description: str
    default: str | None = None


@dataclass
class ReturnDoc:
    """Structured documentation for return and yield values."""

    type: str | None
    description: str


@dataclass
class ParsedDocstring:
    """Normalized representation of a Google-style docstring."""

    summary: str = ""
    description: str = ""
    params: list[ParameterDoc] = field(default_factory=list)
    attributes: list[ParameterDoc] = field(default_factory=list)
    returns: list[ReturnDoc] = field(default_factory=list)
    yields: list[ReturnDoc] = field(default_factory=list)
    raises: list[ParameterDoc] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class DocItem:
    """Represents a documented symbol (class, function, method, or property)."""

    name: str
    qualname: str
    kind: Literal["class", "function", "method", "property"]
    signature: str
    doc: ParsedDocstring
    signature_params: list[ParameterDoc]
    lineno: int
    end_lineno: int
    bases: list[str] = field(default_factory=list)
    children: list["DocItem"] = field(default_factory=list)
    module_path: str = ""
    source: str = ""


@dataclass
class DocumentedModule:
    """Container for all documented items within a Python module."""

    path: Path
    module_path: str
    classes: list[DocItem]
    functions: list[DocItem]


# --------------------------------------------------------------------------------------------- #
# Placeholder (legacy) generation for mkdocstrings-style stubs
# --------------------------------------------------------------------------------------------- #


def extract_classes_and_functions(filepath: Path) -> tuple[list[str], list[str]]:
    """Extract top-level class and (a)sync function names from a Python file."""
    content = filepath.read_text()
    classes = re.findall(r"(?:^|\n)class\s(\w+)(?:\(|:)", content)
    functions = re.findall(r"(?:^|\n)(?:async\s+)?def\s(\w+)\(", content)
    return classes, functions


def create_placeholder_markdown(py_filepath: Path, module_path: str, classes: list[str], functions: list[str]) -> Path:
    """Create a minimal Markdown stub used by mkdocstrings."""
    md_filepath = REFERENCE_DIR / py_filepath.relative_to(PACKAGE_DIR).with_suffix(".md")
    exists = md_filepath.exists()

    if exists:
        return md_filepath.relative_to(PACKAGE_DIR.parent)

    header_content = "---\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"
    module_path_dots = module_path
    module_path = module_path.replace(".", "/")
    url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path}.py"
    edit = f"https://github.com/{GITHUB_REPO}/edit/main/{module_path}.py"
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")

    title_content = (
        f"# Reference for `{module_path}.py`\n\n"
        f"!!! note\n\n"
        f"    This file is available at [{pretty}]({url}). If you spot a problem please help fix it by [contributing]"
        f"(https://docs.ultralytics.com/help/contributing/) a [Pull Request]({edit}) 🛠️. Thank you 🙏!\n\n"
    )
    md_content = ["<br>\n\n"]
    md_content.extend(f"## ::: {module_path_dots}.{cls}\n\n<br><br><hr><br>\n\n" for cls in classes)
    md_content.extend(f"## ::: {module_path_dots}.{func}\n\n<br><br><hr><br>\n\n" for func in functions)
    if md_content[-1:]:
        md_content[-1] = md_content[-1].replace("<hr><br>\n\n", "")

    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    md_filepath.write_text(header_content + title_content + "".join(md_content) + "\n")
    return md_filepath.relative_to(PACKAGE_DIR.parent)


def slugify(value: str) -> str:
    """Create a simple anchor slug similar to MkDocs."""
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _get_source(src: str, node: ast.AST) -> str:
    """Return the source segment for an AST node with safe fallbacks."""
    segment = ast.get_source_segment(src, node)
    if segment:
        return segment
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _format_annotation(annotation: ast.AST | None, src: str) -> str | None:
    """Format a type annotation into a compact string."""
    if annotation is None:
        return None
    text = _get_source(src, annotation).strip()
    return " ".join(text.split()) if text else None


def _format_default(default: ast.AST | None, src: str) -> str | None:
    """Format a default value expression for display."""
    if default is None:
        return None
    text = _get_source(src, default).strip()
    return " ".join(text.split()) if text else None


def _format_parameter(arg: ast.arg, default: ast.AST | None, src: str) -> str:
    """Render a single parameter with annotation and default value."""
    annotation = _format_annotation(arg.annotation, src)
    rendered = arg.arg
    if annotation:
        rendered += f": {annotation}"
    default_value = _format_default(default, src)
    if default_value is not None:
        rendered += f" = {default_value}"
    return rendered


def collect_signature_parameters(args: ast.arguments, src: str, *, skip_self: bool = True) -> list[ParameterDoc]:
    """Collect parameters from an ast.arguments object with types and defaults."""
    params: list[ParameterDoc] = []

    def add_param(arg: ast.arg, default_value: ast.AST | None = None):
        name = arg.arg
        if skip_self and name in {"self", "cls"}:
            return
        params.append(
            ParameterDoc(
                name=name,
                type=_format_annotation(arg.annotation, src),
                description="",
                default=_format_default(default_value, src),
            )
        )

    posonly = list(getattr(args, "posonlyargs", []))
    regular = list(getattr(args, "args", []))
    defaults = list(getattr(args, "defaults", []))
    total_regular = len(posonly) + len(regular)
    default_offset = total_regular - len(defaults)

    combined = posonly + regular
    for idx, arg in enumerate(combined):
        default = defaults[idx - default_offset] if idx >= default_offset else None
        add_param(arg, default)

    vararg = getattr(args, "vararg", None)
    if vararg:
        add_param(vararg)
        params[-1].name = f"*{params[-1].name}"

    kwonly = list(getattr(args, "kwonlyargs", []))
    kw_defaults = list(getattr(args, "kw_defaults", []))
    for kwarg, default in zip(kwonly, kw_defaults):
        add_param(kwarg, default)

    kwarg = getattr(args, "kwarg", None)
    if kwarg:
        add_param(kwarg)
        params[-1].name = f"**{params[-1].name}"

    return params


def format_signature(
    node: ast.AST, src: str, *, is_class: bool = False, is_async: bool = False, display_name: str | None = None
) -> str:
    """Build a readable signature string for classes, functions, and methods."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ""

    if isinstance(node, ast.ClassDef):
        init_method = next(
            (n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"),
            None,
        )
        args = (
            init_method.args
            if init_method
            else ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        )
    else:
        args = node.args
    name = display_name or getattr(node, "name", "")
    params: list[str] = []

    posonly = list(getattr(args, "posonlyargs", []))
    regular = list(getattr(args, "args", []))
    defaults = list(getattr(args, "defaults", []))
    total_regular = len(posonly) + len(regular)
    default_offset = total_regular - len(defaults)

    combined = posonly + regular
    for idx, arg in enumerate(combined):
        default = defaults[idx - default_offset] if idx >= default_offset else None
        params.append(_format_parameter(arg, default, src))
        if posonly and idx == len(posonly) - 1:
            params.append("/")

    vararg = getattr(args, "vararg", None)
    if vararg:
        rendered = _format_parameter(vararg, None, src)
        params.append(f"*{rendered}")

    kwonly = list(getattr(args, "kwonlyargs", []))
    kw_defaults = list(getattr(args, "kw_defaults", []))
    if kwonly:
        if not vararg:
            params.append("*")
        for kwarg, default in zip(kwonly, kw_defaults):
            params.append(_format_parameter(kwarg, default, src))

    kwarg = getattr(args, "kwarg", None)
    if kwarg:
        rendered = _format_parameter(kwarg, None, src)
        params.append(f"**{rendered}")

    signature = f"{name}({', '.join(params)})"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
        annotation = _format_annotation(node.returns, src)
        if annotation:
            signature += f" -> {annotation}"

    if is_class:
        return signature
    prefix = "async def " if is_async else "def "
    return prefix + signature


def _split_section_entries(lines: list[str]) -> list[list[str]]:
    """Split a docstring section into entries based on indentation."""
    entries: list[list[str]] = []
    current: list[str] = []
    base_indent: int | None = None

    for raw_line in lines:
        if not raw_line.strip():
            if current:
                current.append("")
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if base_indent is None:
            base_indent = indent
        if indent <= base_indent and current:
            entries.append(current)
            current = [raw_line]
        else:
            current.append(raw_line)
    if current:
        entries.append(current)
    return entries


def _parse_named_entries(lines: list[str]) -> list[ParameterDoc]:
    """Parse Args/Attributes/Raises style sections."""
    entries = []
    for block in _split_section_entries(lines):
        text = textwrap.dedent("\n".join(block)).strip()
        if not text:
            continue
        first_line, *rest = text.splitlines()
        match = re.match(r"([\w*]+)\s*(?:\(([^)]+)\))?:\s*(.*)", first_line)
        if match:
            name, type_hint, desc = match.groups()
            description = " ".join(desc.split())
            if rest:
                description = f"{description}\n" + "\n".join(rest)
            entries.append(ParameterDoc(name=name, type=type_hint, description=_normalize_text(description)))
        else:
            entries.append(ParameterDoc(name=text, type=None, description=""))
    return entries


def _parse_returns(lines: list[str]) -> list[ReturnDoc]:
    """Parse Returns/Yields sections."""
    entries = []
    for block in _split_section_entries(lines):
        text = textwrap.dedent("\n".join(block)).strip()
        if not text:
            continue
        match = re.match(r"([^:]+):\s*(.*)", text)
        if match:
            type_hint, desc = match.groups()
            cleaned_type = type_hint.strip()
            if cleaned_type.startswith("(") and cleaned_type.endswith(")"):
                cleaned_type = cleaned_type[1:-1].strip()
            entries.append(ReturnDoc(type=cleaned_type, description=_normalize_text(desc.strip())))
        else:
            entries.append(ReturnDoc(type=None, description=_normalize_text(text)))
    return entries


SECTION_ALIASES = {
    "args": "params",
    "arguments": "params",
    "parameters": "params",
    "params": "params",
    "returns": "returns",
    "return": "returns",
    "yields": "yields",
    "yield": "yields",
    "raises": "raises",
    "exceptions": "raises",
    "exception": "raises",
    "attributes": "attributes",
    "attr": "attributes",
    "examples": "examples",
    "example": "examples",
    "notes": "notes",
    "note": "notes",
    "methods": "methods",
}


def _normalize_text(text: str) -> str:
    """Collapse single newlines within paragraphs while preserving paragraph breaks."""
    if not text:
        return ""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def parse_google_docstring(docstring: str | None) -> ParsedDocstring:
    """Parse a Google-style docstring into structured data."""
    if not docstring:
        return ParsedDocstring()

    lines = textwrap.dedent(docstring).splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ParsedDocstring()

    summary = _normalize_text(lines[0].strip())
    body = lines[1:]

    sections: defaultdict[str, list[str]] = defaultdict(list)
    current = "description"
    for line in body:
        stripped = line.strip()
        key = SECTION_ALIASES.get(stripped.rstrip(":").lower())
        if key and stripped.endswith(":"):
            current = key
            continue
        if current != "methods":  # ignore "Methods:" sections; methods are rendered from AST
            sections[current].append(line)

    description = "\n".join(sections.pop("description", [])).strip("\n")
    description = _normalize_text(description)

    return ParsedDocstring(
        summary=summary,
        description=description,
        params=_parse_named_entries(sections.get("params", [])),
        attributes=_parse_named_entries(sections.get("attributes", [])),
        returns=_parse_returns(sections.get("returns", [])),
        yields=_parse_returns(sections.get("yields", [])),
        raises=_parse_named_entries(sections.get("raises", [])),
        notes=[textwrap.dedent("\n".join(sections.get("notes", []))).strip()] if sections.get("notes") else [],
        examples=[textwrap.dedent("\n".join(sections.get("examples", []))).strip()] if sections.get("examples") else [],
    )


def merge_docstrings(base: ParsedDocstring, extra: ParsedDocstring, ignore_summary: bool = True) -> ParsedDocstring:
    """Merge init docstring content into a class docstring."""
    if not base.summary and extra.summary and not ignore_summary:
        base.summary = extra.summary
    if extra.description:
        base.description = "\n\n".join(filter(None, [base.description, extra.description]))
    base.params.extend(extra.params)
    base.attributes.extend(extra.attributes)
    base.returns.extend(extra.returns)
    base.yields.extend(extra.yields)
    base.raises.extend(extra.raises)
    base.notes.extend(extra.notes)
    base.examples.extend(extra.examples)
    return base


def _should_document(name: str, *, allow_private: bool = False) -> bool:
    """Decide whether to include a symbol based on its name."""
    if name in INCLUDE_SPECIAL_METHODS:
        return True
    if name.startswith("_"):
        return allow_private
    return True


def _collect_source_block(src: str, node: ast.AST, end_line: int | None = None) -> str:
    """Return a dedented source snippet for the given node up to an optional end line."""
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    lines = src.splitlines()
    start = max(node.lineno - 1, 0)
    end = end_line or getattr(node, "end_lineno", node.lineno)
    snippet = "\n".join(lines[start:end])
    return textwrap.dedent(snippet).rstrip()


def parse_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module_path: str,
    src: str,
    *,
    parent: str | None = None,
    allow_private: bool = False,
) -> DocItem | None:
    """Parse a function or method node into a DocItem."""
    raw_docstring = ast.get_docstring(node)
    if not _should_document(node.name, allow_private=allow_private) and not raw_docstring:
        return None

    is_async = isinstance(node, ast.AsyncFunctionDef)
    doc = parse_google_docstring(raw_docstring)
    qualname = f"{module_path}.{node.name}" if not parent else f"{parent}.{node.name}"
    decorators = {_get_source(src, d).split(".")[-1] for d in node.decorator_list}
    kind: Literal["function", "method", "property"] = "method" if parent else "function"
    if decorators & PROPERTY_DECORATORS:
        kind = "property"

    signature_params = collect_signature_parameters(node.args, src, skip_self=bool(parent))

    return DocItem(
        name=node.name,
        qualname=qualname,
        kind=kind,
        signature=format_signature(node, src, is_async=is_async),
        doc=doc,
        signature_params=signature_params,
        lineno=node.lineno,
        end_lineno=node.end_lineno or node.lineno,
        bases=[],
        children=[],
        module_path=module_path,
        source=_collect_source_block(src, node),
    )


def parse_class(node: ast.ClassDef, module_path: str, src: str) -> DocItem:
    """Parse a class node, merging __init__ docs and collecting methods."""
    class_doc = parse_google_docstring(ast.get_docstring(node))

    init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = next(
        (n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"),
        None,
    )
    signature_params: list[ParameterDoc] = []
    if init_node:
        init_doc = parse_google_docstring(ast.get_docstring(init_node))
        class_doc = merge_docstrings(class_doc, init_doc, ignore_summary=True)
        signature_params = collect_signature_parameters(init_node.args, src, skip_self=True)

    bases = [_get_source(src, b) for b in node.bases] if node.bases else []
    signature_node = init_node or node
    class_signature = format_signature(signature_node, src, is_class=True, display_name=node.name)

    methods: list[DocItem] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not init_node:
            method_doc = parse_function(child, module_path, src, parent=f"{module_path}.{node.name}")
            if method_doc:
                methods.append(method_doc)

    return DocItem(
        name=node.name,
        qualname=f"{module_path}.{node.name}",
        kind="class",
        signature=class_signature,
        doc=class_doc,
        signature_params=signature_params,
        lineno=node.lineno,
        end_lineno=node.end_lineno or node.lineno,
        bases=bases,
        children=methods,
        module_path=module_path,
        source=_collect_source_block(src, node, end_line=init_node.end_lineno if init_node else node.lineno),
    )


def parse_module(py_filepath: Path) -> DocumentedModule | None:
    """Parse a Python module into structured documentation objects."""
    try:
        src = py_filepath.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    module_path = (
        f"{PACKAGE_DIR.name}.{py_filepath.relative_to(PACKAGE_DIR).with_suffix('').as_posix().replace('/', '.')}"
    )
    classes: list[DocItem] = []
    functions: list[DocItem] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(parse_class(node, module_path, src))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = parse_function(node, module_path, src, parent=None)
            if func:
                functions.append(func)

    return DocumentedModule(path=py_filepath, module_path=module_path, classes=classes, functions=functions)


def _render_section(title: str, entries: Iterable[str], level: int) -> str:
    """Render a section with a given heading level."""
    entries = list(entries)
    if not entries:
        return ""
    heading = f"{'#' * level} {title}\n"
    body = "\n".join(entries).rstrip()
    return f"{heading}{body}\n\n"


def _render_table(headers: list[str], rows: list[list[str]], level: int, title: str | None = None) -> str:
    """Render a Markdown table with an optional heading."""
    if not rows:
        return ""
    def _clean_cell(value: str | None) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", "<br>").strip()

    rows = [[_clean_cell(c) for c in row] for row in rows]
    table_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    heading = f"{'#' * level} {title}\n" if title else ""
    return f"{heading}" + "\n".join(table_lines) + "\n\n"


def _code_fence(source: str, lang: str = "python") -> str:
    """Return a fenced code block with optional language for highlighting."""
    return f"```{lang}\n{source}\n```"


def _merge_params(doc_params: list[ParameterDoc], signature_params: list[ParameterDoc]) -> list[ParameterDoc]:
    """Merge docstring params with signature params to include defaults/types."""
    sig_map = {p.name.lstrip("*"): p for p in signature_params}
    merged: list[ParameterDoc] = []

    seen = set()
    for dp in doc_params:
        sig = sig_map.get(dp.name.lstrip("*"))
        merged.append(
            ParameterDoc(
                name=dp.name,
                type=dp.type or (sig.type if sig else None),
                description=dp.description,
                default=sig.default if sig else None,
            )
        )
        seen.add(dp.name.lstrip("*"))

    for name, sig in sig_map.items():
        if name in seen:
            continue
        merged.append(sig)

    return merged


def render_docstring(doc: ParsedDocstring, level: int, signature_params: list[ParameterDoc] | None = None) -> str:
    """Convert a ParsedDocstring into Markdown with tables similar to mkdocstrings."""
    parts: list[str] = []
    if doc.summary:
        parts.append(doc.summary)
    if doc.description:
        parts.append(doc.description)

    sig_params = signature_params or []
    merged_params = _merge_params(doc.params, sig_params)

    sections: list[str] = []
    ordered_sections: list[str] = []

    if merged_params:
        rows = []
        for p in merged_params:
            default_val = f"`{p.default}`" if p.default not in (None, "") else "*required*"
            rows.append(
                [
                    f"`{p.name}`",
                    f"`{p.type}`" if p.type else "",
                    p.description.strip() if p.description else "",
                    default_val,
                ]
            )
        ordered_sections.append(_render_table(["Name", "Type", "Description", "Default"], rows, level, "Parameters"))

    if doc.attributes:
        rows = []
        for a in doc.attributes:
            rows.append(
                [f"`{a.name}`", f"`{a.type}`" if a.type else "", a.description.strip() if a.description else ""]
            )
        ordered_sections.append(_render_table(["Name", "Type", "Description"], rows, level, "Attributes"))

    if doc.returns:
        rows = []
        for r in doc.returns:
            rows.append([f"`{r.type}`" if r.type else "", r.description])
        ordered_sections.append(_render_table(["Type", "Description"], rows, level, "Returns"))

    if doc.yields:
        rows = []
        for r in doc.yields:
            rows.append([f"`{r.type}`" if r.type else "", r.description])
        ordered_sections.append(_render_table(["Type", "Description"], rows, level, "Yields"))

    if doc.raises:
        rows = []
        for e in doc.raises:
            type_cell = e.type or e.name
            rows.append([f"`{type_cell}`" if type_cell else "", e.description or ""])
        ordered_sections.append(_render_table(["Type", "Description"], rows, level, "Raises"))

    if doc.notes:
        note_text = "\n\n".join(doc.notes).strip()
        indented = textwrap.indent(note_text, "    ")
        ordered_sections.append(f'!!! note "Notes"\n\n{indented}\n\n')

    if doc.examples:
        code_block = "\n\n".join(f"```python\n{example.strip()}\n```" for example in doc.examples if example.strip())
        if code_block:
            ordered_sections.append(f"{'#' * level} Examples\n{code_block}\n\n")

    sections.extend(ordered_sections)

    parts.extend(filter(None, sections))
    return "\n\n".join([p.rstrip() for p in parts if p]).strip() + ("\n\n" if parts else "")


def item_anchor(item: DocItem) -> str:
    """Create a stable anchor for a documented item."""
    return slugify(item.qualname)


def render_item(item: DocItem, module_url: str, module_path: str, level: int = 2) -> str:
    """Render a class, function, or method to Markdown."""
    anchor = item_anchor(item)
    title_prefix = item.kind.capitalize()
    heading = f"{'#' * level} {title_prefix} `{item.name}` {{#{anchor}}}"
    signature_block = f"```python\n{item.signature}\n```\n"

    parts = [heading, signature_block]

    if item.bases:
        bases = ", ".join(f"`{b}`" for b in item.bases)
        parts.append(f"**Bases:** {bases}\n")

    parts.append(render_docstring(item.doc, level + 1, signature_params=item.signature_params))

    if item.kind == "class" and item.source:
        source_url = f"{module_url}#L{item.lineno}-L{item.end_lineno}"
        summary = f"&lt;&gt; Source code in <code>{html.escape(module_path)}.py</code>"
        parts.append(
            "<details>\n"
            f"<summary>{summary}</summary>\n\n"
            f'<p><a href="{source_url}">View source</a></p>\n\n'
            f"{_code_fence(item.source)}\n"
            "</details>\n"
        )

    if item.children:
        method_rows = []
        for child in item.children:
            summary = child.doc.summary or (
                _normalize_text(child.doc.description).split("\n\n")[0] if child.doc.description else ""
            )
            summary = summary.strip()
            method_rows.append([f"[`{child.name}`](#{item_anchor(child)})", summary])
        parts.append(_render_table(["Name", "Description"], method_rows, level + 1, "Methods"))
        for child in item.children:
            parts.append(render_item(child, module_url, module_path, level + 2))

    if item.source and item.kind != "class":
        source_url = f"{module_url}#L{item.lineno}-L{item.end_lineno}"
        summary = f"&lt;&gt; Source code in <code>{html.escape(module_path)}.py</code>"
        parts.append(
            "<details>\n"
            f"<summary>{summary}</summary>\n\n"
            f'<p><a href="{source_url}">View source</a></p>\n\n'
            f"{_code_fence(item.source)}\n"
            "</details>\n"
        )

    return "\n".join(p for p in parts if p).rstrip() + "\n"


def render_module_markdown(module: DocumentedModule) -> str:
    """Render the full module reference content."""
    module_path = module.module_path.replace(".", "/")
    module_url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path}.py"
    content: list[str] = ["<br>\n"]

    if module.classes:
        content.append("## Classes\n\n")
        content.extend(f"- [`{cls.name}`](#{item_anchor(cls)})\n" for cls in module.classes)
        content.append("\n")
    if module.functions:
        content.append("## Functions\n\n")
        content.extend(f"- [`{func.name}`](#{item_anchor(func)})\n" for func in module.functions)
        content.append("\n")

    sections: list[str] = []
    for idx, cls in enumerate(module.classes):
        sections.append(render_item(cls, module_url, module_path, level=2))
        if idx != len(module.classes) - 1 or module.functions:
            sections.append("<br><br><hr><br>\n")
    for idx, func in enumerate(module.functions):
        sections.append(render_item(func, module_url, module_path, level=2))
        if idx != len(module.functions) - 1:
            sections.append("<br><br><hr><br>\n")

    content.extend(sections)
    return "\n".join(content).rstrip() + "\n\n<br><br>\n"


def create_markdown(module: DocumentedModule) -> Path:
    """Create a Markdown file containing the API reference for the given Python module."""
    md_filepath = REFERENCE_DIR / module.path.relative_to(PACKAGE_DIR).with_suffix(".md")
    exists = md_filepath.exists()

    header_content = ""
    if exists:
        existing_content = md_filepath.read_text()
        header_parts = existing_content.split("---")
        for part in header_parts:
            if "description:" in part or "comments:" in part:
                header_content += f"---{part}---\n\n"
    if not any(header_content):
        header_content = "---\ndescription: TODO ADD DESCRIPTION\nkeywords: TODO ADD KEYWORDS\n---\n\n"

    module_path = module.module_path.replace(".", "/")
    url = f"https://github.com/{GITHUB_REPO}/blob/main/{module_path}.py"
    edit = f"https://github.com/{GITHUB_REPO}/edit/main/{module_path}.py"
    pretty = url.replace("__init__.py", "\\_\\_init\\_\\_.py")  # Properly display __init__.py filenames

    title_content = (
        f"# Reference for `{module_path}.py`\n\n"
        f"!!! note\n\n"
        f"    This file is available at [{pretty}]({url}). If you spot a problem please help fix it by [contributing]"
        f"(https://docs.ultralytics.com/help/contributing/) a [Pull Request]({edit}) 🛠️. Thank you 🙏!\n\n"
    )

    md_filepath.parent.mkdir(parents=True, exist_ok=True)
    md_filepath.write_text(header_content + title_content + render_module_markdown(module))

    if not exists:
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
    new_section_lines.extend(
        f"    {line}"
        for line in reference_yaml.splitlines()
        if line.strip() != "- reference:"  # Skip redundant header
    )
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
    elif help_match := re.search(r"(\n  - Help:)", mkdocs_content):
        # No existing Reference section, we need to add it
        help_section = help_match.group(1)
        # Insert before Help section
        new_content = mkdocs_content.replace(help_section, f"{new_ref_section}{help_section}")
        MKDOCS_YAML.write_text(new_content)
        print(f"Added new Reference section before Help in {MKDOCS_YAML}")
    else:
        print("Could not find a suitable location to add Reference section")


def build_reference(update_nav: bool = True) -> list[str]:
    """Create placeholder reference files (legacy mkdocstrings flow)."""
    return build_reference_placeholders(update_nav=update_nav)


def build_reference_placeholders(update_nav: bool = True) -> list[str]:
    """Create minimal placeholder reference files (mkdocstrings-style) and optionally update nav."""
    nav_items: list[str] = []
    created = 0

    for py_filepath in TQDM(list(PACKAGE_DIR.rglob("*.py")), desc="Building reference stubs", unit="file"):
        classes, functions = extract_classes_and_functions(py_filepath)
        if not classes and not functions:
            continue
        module_path = f"{PACKAGE_DIR.name}.{py_filepath.relative_to(PACKAGE_DIR).with_suffix('').as_posix().replace('/', '.')}"
        md_rel = create_placeholder_markdown(py_filepath, module_path, classes, functions)
        nav_items.append(str(md_rel))
        md_abs = PACKAGE_DIR.parent / md_rel
        if not md_abs.exists():  # safety, though create_placeholder_markdown writes when missing
            created += 1
    if update_nav:
        update_mkdocs_file(create_nav_menu_yaml(nav_items))
    if created:
        print(f"Created {created} new reference stub files")
    return nav_items


def build_reference_docs(update_nav: bool = False) -> list[str]:
    """Render full docstring-based reference content."""
    nav_items: list[str] = []
    created = 0

    for py_filepath in TQDM(list(PACKAGE_DIR.rglob("*.py")), desc="Rendering docstrings", unit="file"):
        md_target = REFERENCE_DIR / py_filepath.relative_to(PACKAGE_DIR).with_suffix(".md")
        exists_before = md_target.exists()
        module = parse_module(py_filepath)
        if not module or (not module.classes and not module.functions):
            continue
        md_rel_filepath = create_markdown(module)
        if not exists_before:
            created += 1
        nav_items.append(str(md_rel_filepath))

    if update_nav:
        update_mkdocs_file(create_nav_menu_yaml(nav_items))
    if created:
        print(f"Created {created} new reference files")
    return nav_items


def main():
    """CLI entrypoint."""
    build_reference(update_nav=True)


if __name__ == "__main__":
    main()
