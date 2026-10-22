"""Score H10 arms (plus F_clip) on 3DPW, COCO val (teacher), framed/truncated splits and COCO GT 2D."""
import sys
sys.path.insert(0, "/home/rick/ultralytics_pose3d")
from ultralytics import YOLO

device, arms = sys.argv[1], sys.argv[2:]
DATA = ["3dpw-pose3d-r12.yaml", "coco-pose3d-refocal.yaml", "/home/rick/h10/coco-val-framed.yaml",
        "/home/rick/h10/coco-val-trunc.yaml", "/home/rick/h10/coco-pose-gt-18x4.yaml"]
keys = ("MPJPE", "PA-MPJPE", "AbsRel(Z)", "delta1(Z)", "mAP50-95(P)", "mAP50-95(B)")
for arm in arms:
    for data in DATA:
        m = YOLO(f"/home/rick/runs/pose3d/{arm}/weights/best.pt")
        name = data.split("/")[-1][:-5]
        r = m.val(data=data, imgsz=640, batch=32, device=device, plots=False, verbose=False, rect="pose-gt" not in name,
                  project="/home/rick/runs/pose3d_h10", name=f"{arm}-{name}", exist_ok=True)
        d = {k.split("/")[-1]: v for k, v in r.results_dict.items()}
        print(f"RESULT {arm} {name} " + " ".join(f"{k}={d[k]:.5g}" for k in keys if k in d), flush=True)
