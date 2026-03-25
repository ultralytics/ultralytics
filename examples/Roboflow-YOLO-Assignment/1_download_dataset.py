"""
1_download_dataset.py
---------------------
ดาวน์โหลด dataset จาก Roboflow แล้วบันทึก path ไว้ใน dataset_path.txt
เพื่อให้ script อื่นนำไปใช้ต่อได้

Usage:
    python 1_download_dataset.py
"""

from roboflow import Roboflow
from config import RF_API_KEY, RF_WORKSPACE, RF_PROJECT, RF_VERSION

rf      = Roboflow(api_key=RF_API_KEY)
project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
version = project.version(RF_VERSION)
dataset = version.download("yolov8")

data_yaml = f"{dataset.location}/data.yaml"

# บันทึก path ไว้ให้ script อื่นอ่าน
with open("dataset_path.txt", "w") as f:
    f.write(data_yaml)

print(f"[OK] Dataset saved to : {dataset.location}")
print(f"[OK] data.yaml path   : {data_yaml}")
print("[OK] Path saved to dataset_path.txt")
