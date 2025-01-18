from ultralytics import YOLO


def test_dataset(model="yolo11n.pt", data="coco8.yaml", imgsz=640, workers=32, batch=32):
    """
    Function created to test the Ultralytics YOLO latest model on a specified dataset.

    Args:
        model (str): The path to the YOLO model file. Default is "yolo11n.pt".
        data (str): The path to the dataset configuration file (e.g., .yaml). Default is "coco8.yaml".
        imgsz (int): The size of the input image used for model processing.
        workers (int): Number of CPU workers used for loading the image dataset.
        batch (int): Size of each batch used during YOLO model training.

    Functionality
        Loads the specified YOLO model using the YOLO class.
        Trains the model on the given dataset for 3 epochs with an image size of 640 or any other pixels.
    """
    model = YOLO(model)  # load a model
    _ = model.train(data=data, epochs=1, imgsz=imgsz, workers=workers, batch=batch)  # train the model


# Object detection datasets
test_dataset(data="coco8.yaml")  # https://docs.ultralytics.com/datasets/detect/coco8/
test_dataset(data="brain-tumor.yaml")  # https://docs.ultralytics.com/datasets/detect/brain-tumor/
test_dataset(data="african-wildlife.yaml")  # https://docs.ultralytics.com/datasets/detect/african-wildlife/
test_dataset(data="signature.yaml")  # https://docs.ultralytics.com/datasets/detect/signature/
test_dataset(data="medical-pills.yaml")  # https://docs.ultralytics.com/datasets/detect/medical-pills/

# Image segmentation datasets
test_dataset(model="yolo11n-seg.pt", data="coco8-seg.yaml")  # https://docs.ultralytics.com/datasets/segment/coco8-seg/
test_dataset(model="yolo11n-seg.pt", data="crack-seg.yaml")  # https://docs.ultralytics.com/datasets/segment/crack-seg/
test_dataset(
    model="yolo11n-seg.pt", data="carparts-seg.yaml"
)  # https://docs.ultralytics.com/datasets/segment/carparts-seg/
test_dataset(
    model="yolo11n-seg.pt", data="package-seg.yaml"
)  # https://docs.ultralytics.com/datasets/segment/package-seg/

# Pose estimation datasets
test_dataset(model="yolo11n-pose.pt", data="coco8-pose.yaml")  # https://docs.ultralytics.com/datasets/pose/coco8-pose/
test_dataset(model="yolo11n-pose.pt", data="tiger-pose.yaml")  # https://docs.ultralytics.com/datasets/pose/tiger-pose/
test_dataset(
    model="yolo11n-pose.pt", data="hand-keypoints.yaml"
)  # https://docs.ultralytics.com/datasets/pose/hand-keypoints/
test_dataset(model="yolo11n-pose.pt", data="dog-pose.yaml")  # https://docs.ultralytics.com/datasets/pose/dog-pose/

# Oriented bounding boxes datasets
test_dataset(model="yolo11n-obb.pt", data="dota8.yaml")  # https://docs.ultralytics.com/datasets/obb/dota8/

# Image classification datasets
test_dataset(
    model="yolo11n-cls.pt", data="cifar10", imgsz=32
)  # https://docs.ultralytics.com/datasets/classify/cifar10/
test_dataset(
    model="yolo11n-cls.pt", data="cifar100", imgsz=32
)  # https://docs.ultralytics.com/datasets/classify/cifar100/
test_dataset(
    model="yolo11n-cls.pt", data="caltech101", imgsz=416
)  # https://docs.ultralytics.com/datasets/classify/caltech101/
test_dataset(
    model="yolo11n-cls.pt", data="fashion-mnist", imgsz=28
)  # https://docs.ultralytics.com/datasets/classify/fashion-mnist/
test_dataset(
    model="yolo11n-cls.pt", data="imagenette160", imgsz=160
)  # https://docs.ultralytics.com/datasets/classify/imagenette/
test_dataset(model="yolo11n-cls.pt", data="mnist", imgsz=32)  # https://docs.ultralytics.com/datasets/classify/mnist/
