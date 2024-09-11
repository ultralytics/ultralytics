# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
import tlc

from ultralytics.utils import colorstr

# Column names
TRAINING_PHASE = "Training Phase"
IMAGE_COLUMN_NAME = tlc.IMAGE
CLASSIFY_LABEL_COLUMN_NAME = tlc.LABEL
DETECTION_LABEL_COLUMN_NAME = "bbs.bb_list.label"

# Other
DEFAULT_TRAIN_RUN_DESCRIPTION = ""
DEFAULT_COLLECT_RUN_DESCRIPTION = "Created with model.collect()"

TLC_PREFIX = "3LC://"
TLC_COLORSTR = colorstr("3lc: ")
TLC_REQUIRED_VERSION = "2.7.1"
