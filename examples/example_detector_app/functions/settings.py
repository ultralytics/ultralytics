import sys
from pathlib import Path

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
# IMAGE = 'Image'
# VIDEO = 'Video'
# WEBCAM = 'Webcam'
# RTSP = 'RTSP'
# YOUTUBE = 'YouTube'

# SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
# IMAGES_DIR = ROOT / 'images'
# DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
# DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
# VIDEO_DIR = ROOT / 'videos'
# VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
# VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
# VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
# VIDEOS_DICT = {
#     'video_1': VIDEO_1_PATH,
#     'video_2': VIDEO_2_PATH,
#     'video_3': VIDEO_3_PATH,
# }

# ML Model config
# MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = "C:\\Users\\BAB AL SAFA\\Desktop\\MINE\\Detector-App\\weights\\yolov8n.pt"
# In case of your custom model comment out the line above and
# Place your custom model pt file name at the line below
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

# SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
