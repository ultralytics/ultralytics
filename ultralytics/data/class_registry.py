# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ultralytics.utils import LOGGER, colorstr


class ClassRegistry:
    """Resolve semantic classes across multiple datasets into a unified global class mapping.

    When training on multiple datasets simultaneously, each dataset may define the same semantic classes with different
    numeric IDs. This class reads the ``names`` dict from each dataset's parsed data dictionary, builds a deterministic
    global name-to-ID mapping (union of all names, ordered deterministically by first appearance across the dataset
    list, and sorted local class IDs within each dataset), and provides efficient per-dataset local-to-global ID
    translation arrays.

    Attributes:
        global_names (dict[int, str]): Global class names mapping ``{global_id: class_name}``.
        global_nc (int): Total number of unique classes across all datasets.
        remaps (list[np.ndarray]): Per-dataset local-to-global ID translation arrays.

    Methods:
        get_remap: Return the local-to-global ID mapping array for a specific dataset.
        remap_labels: Apply the local-to-global ID mapping to a list of label dicts in-place.

    Examples:
        >>> dataset_dicts = [
        ...     {"names": {0: "blood", 1: "stool"}, "nc": 2},
        ...     {"names": {0: "stool", 1: "blood"}, "nc": 2},
        ... ]
        >>> registry = ClassRegistry(dataset_dicts)
        >>> registry.global_names
        {0: 'blood', 1: 'stool'}
        >>> registry.get_remap(1)  # dataset B: local 0 (stool) -> global 1, local 1 (blood) -> global 0
        array([1, 0])
    """

    def __init__(self, dataset_dicts: list[dict[str, Any]], global_names: dict[int, str] | None = None):
        """Build global class registry from parsed dataset dicts.

        Args:
            dataset_dicts (list[dict[str, Any]]): List of data dicts (from ``check_det_dataset``), each containing a
                ``names`` key mapping ``{local_id: class_name}`` and an ``nc`` key with the class count.
            global_names (dict[int, str], optional): User-defined global names mapping. If provided, acts as the
                canonical global class list.
        """
        if not dataset_dicts:
            raise ValueError("ClassRegistry requires at least one dataset dict.")

        if global_names is not None:
            # Validate duplicate global names
            seen_global = set()
            for gid, name in global_names.items():
                if name in seen_global:
                    raise ValueError(f"Duplicate class name '{name}' found in global 'names' definition.")
                seen_global.add(name)

            self.global_names = global_names
            self.global_nc = len(global_names)
            name_to_global = {name: gid for gid, name in global_names.items()}
        else:
            # Automatically resolve global names (union of all sub-datasets) using deterministic first-seen order
            # The order is defined by the dataset sequence in the list, then sorted by local class IDs.
            name_to_global = {}
            for i, data in enumerate(dataset_dicts):
                names = data.get("names", {})
                seen_names = set()
                for local_id in sorted(names.keys()):
                    name = names[local_id]
                    if name in seen_names:
                        raise ValueError(
                            f"Duplicate class name '{name}' found in local dataset {i} ({data.get('yaml_file')})."
                        )
                    seen_names.add(name)
                    if name not in name_to_global:
                        name_to_global[name] = len(name_to_global)
            self.global_names = {gid: name for name, gid in name_to_global.items()}
            self.global_nc = len(self.global_names)

        # Build per-dataset local→global remap arrays
        self.remaps: list[np.ndarray] = []
        for i, data in enumerate(dataset_dicts):
            names = data.get("names", {})
            remap = np.arange(len(names), dtype=np.intp)  # identity by default
            is_identity = True
            for local_id in sorted(names.keys()):
                name = names[local_id]
                if name not in name_to_global:
                    raise ValueError(
                        f"Class '{name}' in dataset {i} ({data.get('yaml_file')}) is not present in the user-defined global 'names' list."
                    )
                global_id = name_to_global[name]
                remap[local_id] = global_id
                if local_id != global_id:
                    is_identity = False
            self.remaps.append(remap)
            if not is_identity:
                LOGGER.info(
                    f"{colorstr('ClassRegistry:')} dataset {i} class remap: "
                    + ", ".join(f"{names[k]}({k}→{remap[k]})" for k in sorted(names.keys()) if k != remap[k])
                )

        LOGGER.info(
            f"{colorstr('ClassRegistry:')} {len(dataset_dicts)} datasets, "
            f"{self.global_nc} global classes: {self.global_names}"
        )

    def get_remap(self, dataset_index: int) -> np.ndarray:
        """Return local-to-global ID mapping array for the dataset at the given index.

        Args:
            dataset_index (int): Index of the dataset in the original list.

        Returns:
            (np.ndarray): Array where ``remap[local_id] = global_id``.
        """
        return self.remaps[dataset_index]

    def is_identity(self, dataset_index: int) -> bool:
        """Check whether the remap for a dataset is the identity (no remapping needed).

        Args:
            dataset_index (int): Index of the dataset in the original list.

        Returns:
            (bool): True if local IDs match global IDs for every class.
        """
        remap = self.remaps[dataset_index]
        return np.array_equal(remap, np.arange(len(remap), dtype=remap.dtype))

    @staticmethod
    def remap_labels(labels: list[dict], remap: np.ndarray) -> list[dict]:
        """Apply a local-to-global class ID mapping to a list of label dicts in-place.

        Args:
            labels (list[dict]): Label dicts, each with a ``cls`` key containing a numpy array of shape ``(N, 1)``.
            remap (np.ndarray): Local-to-global ID mapping array where ``remap[local_id] = global_id``.

        Returns:
            (list[dict]): The same label list, modified in-place.
        """
        for lb in labels:
            cls = lb["cls"]
            if cls.size > 0:
                lb["cls"] = remap[cls.astype(np.intp)].astype(cls.dtype)
        return labels
