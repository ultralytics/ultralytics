# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import IterableSimpleNamespace, yaml_load


def extract_cfg_data(FILE):
    ROOT = FILE.parents[0]  # solutions
    DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
    DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
    for k, v in DEFAULT_CFG_DICT.items():
        if isinstance(v, str) and v.lower() == "none":
            DEFAULT_CFG_DICT[k] = None
    DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    return DEFAULT_CFG
