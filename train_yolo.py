from ultralytics import YOLO


def train_yolo():
    # Load model from YAML (training from scratch)
    model = YOLO("ultralytics/cfg/models/v8/yolov8m.yaml")

    # Train
    model.train(
        data="mydata.yaml",  # dataset config
        epochs=300,
        patience=50,
        imgsz=640,
        batch=16,
    )

    print("✅ Training completed. Check runs/detect/train/ for results.")


if __name__ == "__main__":
    train_yolo()
