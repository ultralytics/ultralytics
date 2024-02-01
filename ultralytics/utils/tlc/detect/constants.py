from ultralytics.utils import colorstr

# Column names
TRAINING_PHASE = "Training Phase"

TLC_PREFIX = '3LC://'
TLC_COLORSTR = colorstr('3LC: ')

# Environment Variables
TLC_SUPPORTED_ENV_VARS = [
    {
        "name": "TLC_CONF_THRES",
        "internal_name": "CONF_THRES",
        "default": "0.1",
        "type": float,
        "description": "Confidence threshold for detections"},
    {
        "name": "TLC_MAX_DET",
        "internal_name": "MAX_DET",
        "default": "300",
        "type": int,
        "description": "Maximum number of detections collected per image"},
    {
        "name": "TLC_IMAGE_EMBEDDINGS_DIM",
        "internal_name": "IMAGE_EMBEDDINGS_DIM",
        "default": "0",
        "type": int,
        "description":
        "Dimension of image embeddings. 0 means no embeddings, 2 means 2D embeddings, 3 means 3D embeddings"},
    {
        "name": "TLC_SAMPLING_WEIGHTS",
        "internal_name": "SAMPLING_WEIGHTS",
        "default": "false",
        "type": bool,
        "description": "Flag to enable/disable sampling weights"},
    {
        "name": "TLC_COLLECT_LOSS",
        "internal_name": "COLLECT_LOSS",
        "default": "false",
        "type": bool,
        "description": "Flag to enable/disable loss collection",
        "not_supported": True},
    {
        "name": "TLC_COLLECTION_VAL_ONLY",
        "internal_name": "COLLECTION_VAL_ONLY",
        "default": "false",
        "type": bool,
        "description": "Flag to collect detections and metrics only on validation data",
        "not_supported": True},
    {
        "name": "TLC_COLLECTION_DISABLE",
        "internal_name": "COLLECTION_DISABLE",
        "default": "false",
        "type": bool,
        "description":
        "Flag to disable metrics collection entirely. This overrides all other metrics collection flags"},
    {
        "name": "TLC_COLLECTION_EPOCH_START",
        "internal_name": "COLLECTION_EPOCH_START",
        "default": "-1",
        "type": int,
        "description": "Epoch to start metrics collection"},
    {
        "name": "TLC_COLLECTION_EPOCH_INTERVAL",
        "internal_name": "COLLECTION_EPOCH_INTERVAL",
        "default": "1",
        "type": int,
        "description":
        "Interval between metrics collection epochs. 1 means every epoch, 2 means every other epoch, etc"}, ]
