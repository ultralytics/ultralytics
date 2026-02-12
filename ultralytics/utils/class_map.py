# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import torch

DEFAULT_CLASS_ALIASES: dict[str, list[str]] = {
    # COCO -> Obj365 naming differences
    "bird": ["wild bird"],
    "sports ball": ["basketball", "soccer", "baseball", "tennis ball", "golf ball"],
}

DEFAULT_CLASS_KEY_HINTS: tuple[str, ...] = ("score_head", "class_embed")


def names_to_list(names) -> list[str]:
    """Convert class names to a list sorted by class index."""
    if names is None:
        return []
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    if isinstance(names, Mapping):
        return [str(v) for _, v in sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])]
    return []


def is_default_numeric_names(names) -> bool:
    """Return True if names look like default numeric placeholders: {0:'0', 1:'1', ...}."""
    name_list = names_to_list(names)
    return bool(name_list) and all(str(i) == str(v) for i, v in enumerate(name_list))


def resolve_names(names, data_yaml):
    """Resolve class names, falling back to data YAML when names are numeric placeholders."""
    if names and not is_default_numeric_names(names):
        return names
    if isinstance(data_yaml, Path):
        data_yaml = str(data_yaml)
    if not isinstance(data_yaml, str) or not data_yaml.endswith((".yaml", ".yml")):
        return names
    try:
        from ultralytics.utils import YAML
        from ultralytics.utils.checks import check_yaml

        yaml_names = YAML.load(check_yaml(data_yaml)).get("names")
        if yaml_names and not is_default_numeric_names(yaml_names):
            return yaml_names
    except Exception:
        pass
    return names


def normalize_class_name(name: str) -> str:
    """Normalize class names for robust matching across datasets."""
    name = str(name).lower().strip()
    name = name.replace("&", "and")
    name = re.sub(r"[/_-]+", " ", name)
    return re.sub(r"\s+", " ", name)


def build_class_row_map(
    src_names: list[str], dst_names: list[str], aliases: Mapping[str, list[str]] | None = None
) -> tuple[dict[int, int], list[tuple[int, str]]]:
    """Build destination->source class row mapping from class-name lists."""
    aliases = aliases or DEFAULT_CLASS_ALIASES
    src_norm_to_idx = {normalize_class_name(name): i for i, name in enumerate(src_names)}

    row_map: dict[int, int] = {}
    missing: list[tuple[int, str]] = []
    for dst_idx, dst_name in enumerate(dst_names):
        key = normalize_class_name(dst_name)
        candidates = [key] + [normalize_class_name(x) for x in aliases.get(key, [])]
        src_idx = next((src_norm_to_idx[c] for c in candidates if c in src_norm_to_idx), None)
        if src_idx is None:
            missing.append((dst_idx, dst_name))
        else:
            row_map[dst_idx] = src_idx
    return row_map, missing


def remap_class_row_state_dict(
    src_state: Mapping[str, torch.Tensor],
    dst_state: Mapping[str, torch.Tensor],
    src_names,
    dst_names,
    aliases: Mapping[str, list[str]] | None = None,
    key_hints: tuple[str, ...] = DEFAULT_CLASS_KEY_HINTS,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]], list[tuple[int, str]]]:
    """Remap class-row tensors from src_state to dst_state using class-name matching."""
    src_name_list = names_to_list(src_names)
    dst_name_list = names_to_list(dst_names)
    if not src_name_list or not dst_name_list:
        return dict(src_state), [], []

    row_map, missing = build_class_row_map(src_name_list, dst_name_list, aliases=aliases)
    remapped = dict(src_state)
    remapped_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key, src_tensor in src_state.items():
        if key not in dst_state:
            continue
        dst_tensor = dst_state[key]
        if src_tensor.shape == dst_tensor.shape:
            continue
        if not any(h in key for h in key_hints):
            continue
        if src_tensor.ndim != dst_tensor.ndim or src_tensor.shape[1:] != dst_tensor.shape[1:]:
            continue

        out = dst_tensor.clone()
        for dst_idx, src_idx in row_map.items():
            if dst_idx < out.shape[0] and src_idx < src_tensor.shape[0]:
                out[dst_idx] = src_tensor[src_idx]
        remapped[key] = out
        remapped_keys.append((key, tuple(src_tensor.shape), tuple(dst_tensor.shape)))

    return remapped, remapped_keys, missing
