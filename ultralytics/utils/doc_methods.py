# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
from pathlib import Path

import yaml

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_PATH, MODES


def type2str_list(obj: object) -> list[str]:
    """
    Converts a nested type object into a list of strings representing the types.

    Args:
        obj (object): The object to convert.

    Returns:
        A list of strings representing the types.
    """
    out = []
    if isinstance(obj, (list, tuple)) and len(obj) > 1:
        return [type2str_list(o) for o in obj]
    else:
        args = getattr(obj, "__args__", "")
        if args:
            [out.extend(a if isinstance(a, list) else [a]) for a in type2str_list(args)]
        a = ".".join([getattr(obj, "__module__", ""), getattr(obj, "__name__", "")]).replace("builtins.", "")
        out.append(a.strip(".")) if a not in {"", "typing.Union"} else None
    return out


def union_args(arg_list: list[list[str], str]) -> str:
    """
    Concatenates the elements of the given list of lists and strings into a single string.

    Args:
        arg_list (list[list[str], str]): A list containing lists of strings and/or strings.

    Returns:
        str: The concatenated string of all the elements in the arg_list.
    """
    txt = ""
    if any([isinstance(a, list) for a in arg_list]) and len(arg_list) > 1:
        for a in sorted(arg_list, key=len):
            if isinstance(a, list):
                txt += "[" + " | ".join(a) + "]"
            else:
                txt += a
    else:
        txt = " | ".join([a for a in arg_list if a])
    return txt


# Get all the methods of the YOLO class and their arguments
method_args = {}
for name, method in inspect.getmembers(YOLO, predicate=inspect.isfunction):
    if name.startswith("__") or name not in MODES:
        continue
    argspec = inspect.getfullargspec(method)
    args_with_defaults = dict(
        zip(argspec.args[-len(argspec.defaults or []) :], argspec.defaults or [None] * len(argspec.args))
    )
    method_args[name] = {**args_with_defaults, **(argspec.kwonlydefaults or {}).copy()}
    for a, v in method_args[name].items():
        if a in {"self", "args", "kwargs"}:
            continue
        # Convert the type annotations to strings
        anno = argspec.annotations.get(a, type(v))
        anno = sorted({t for t in type2str_list(anno) if t}, key=len)
        if len(anno) > 1:
            anno = [union_args(a) if isinstance(a, list) else a for a in anno]
            anno = " | ".join(anno) if len(anno) > 1 else anno[0]
        elif len(anno) == 1:
            anno = anno[0]
        method_args[name][a] = {"default": v, "type": anno}

    # Add the return type of the method
    returns = argspec.annotations.get("return", None)
    if returns:
        method_args[name]["returns"] = {"type": union_args(type2str_list(returns))}

    # Remove 'self' from the method arguments
    _ = [v.pop("self") for v in method_args.values() if "self" in v]

    # Parse Google-style docstring of the method
    for name, args in method_args.items():
        docstring = inspect.getdoc(getattr(YOLO, name))
        if docstring:
            lines = docstring.split("\n")
            arg_lines = lines[lines.index("Args:") : lines.index("Returns:")]
            for a, v in args.items():
                line = [l.strip() for l in arg_lines if l.strip().startswith(f"{a} ")]
                if line:
                    v["description"] = line[0].split(": ", 1)[-1]

_ = Path(DEFAULT_CFG_PATH).write_text(yaml.safe_dump(method_args), "utf-8")
