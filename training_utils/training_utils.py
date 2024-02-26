import os
import YOLO


dataset_yaml_path = "/training/ultralytics/ultralytics/cfg/datasets/verdant.yaml"
coco_classes_file = "/dataset/classes.txt"
runs_directory = "/training/runs"
training_task = os.environ.get("TASK", "detect")   # 'detect' for bbox | 'pose' for hybrid model 
experiment_name = os.environ.get("EXP_NAME", None)  # Results will be saved with this name  under runs/<task>/<exp_name>


def PrepareDataset(coco_classes_file, dataset_yaml, training_task):
    classes = []
    with open(coco_classes_file, "r") as f:
        for l in f.readlines():
            l = l.strip("\n")
            l = l.strip(" ")
            classes.append(l)

    with open(dataset_yaml, "w") as f:
        f.write("path: /dataset\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for i in range(len(classes)):
            f.write(f"  {i}: {classes[i]}\n")
        if training_task == "pose":
            f.write("\nkpt_shape: [1, 3]\n")
    return


def GetModelYaml(task):
    if task == "detect":
        return "yolov8n.yaml"
    elif task == "pose":
        return "yolov8n-pose.yaml"
    print(f"Unknown task {task}")
    return None


def GiveModel(ckpt_path):
    if os.path.exists(ckpt_path):
        return YOLO(ckpt_path)
    print(f"[ERROR] : Model {ckpt_path} does not exists")
    return None


def GetLatestRunDir():
    base_dir = f"{runs_directory}/{training_task}"
    # Get latest experiment name from the directory
    exp_name = sorted(os.listdir(base_dir))[-1]
    return f"{base_dir}/{exp_name}"


def LoadBestModel():
    best_ckpt = f"{GetLatestRunDir()}/weights/best.pt"
    return GiveModel(best_ckpt)


