import os

# Global variables
REPO_URL = "https://github.com/ultralytics/yolov5.git"
REPO_BRANCH = "ultralytics/HUB"  # "master"

ENVIRONMENT = os.environ.get("ULTRALYTICS_ENV", "production")
if ENVIRONMENT == 'production':
    HUB_API_ROOT = "https://api.ultralytics.com"
else:
    HUB_API_ROOT = "http://127.0.0.1:8000"
    print(f'Connected to development server on {HUB_API_ROOT}')
