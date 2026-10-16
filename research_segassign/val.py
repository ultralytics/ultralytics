"""Val one run best.pt on COCO val2017 with save_json, copy predictions.json next to the run: python val.py <name> <device>."""
import shutil, sys
from ultralytics import YOLO

name, device = sys.argv[1], int(sys.argv[2])
r = YOLO(f"/data/rick/segassign/runs/{name}/weights/best.pt").val(
    data="/data/rick/segassign/coco-seg.yaml", batch=32, imgsz=640, device=device, save_json=True,
    project="/data/rick/segassign/val", name=name, exist_ok=True, plots=False,
)
shutil.copy(r.save_dir / "predictions.json", f"/data/rick/segassign/runs/{name}/predictions.json")
