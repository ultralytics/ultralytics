import torch

from ultralytics import YOLO


def test_model_init():
    model = YOLO.new("yolov8n.yaml")
    model.info()
    try:
        YOLO()
    except Exception:
        print("Successfully caught constructor assert!")
    raise Exception("constructor error didn't occur")


def test_model_forward():
    model = YOLO.new("yolov8n.yaml")
    img = torch.rand(512 * 512 * 3).view(1, 3, 512, 512)
    model.forward(img)
    model(img)


def test_model_info():
    model = YOLO.new("yolov8n.yaml")
    model.info()
    model = model.load("best.pt")
    model.info(verbose=True)


def test_model_fuse():
    model = YOLO.new("yolov8n.yaml")
    model.fuse()
    model.load("best.pt")
    model.fuse()


def test_visualize_preds():
    model = YOLO.load("best.pt")
    model.predict(source="ultralytics/assets")


def test_val():
    model = YOLO.load("best.pt")
    model.val(data="coco128.yaml", imgsz=32)


def test_model_resume():
    model = YOLO.new("yolov8n.yaml")
    model.train(epochs=1, imgsz=32, data="coco128.yaml")
    try:
        model.resume(task="detect")
    except AssertionError:
        print("Successfully caught resume assert!")


def test_model_train_pretrained():
    model = YOLO.load("best.pt")
    model.train(data="coco128.yaml", epochs=1, imgsz=32)
    model = model.new("yolov8n.yaml")
    model.train(data="coco128.yaml", epochs=1, imgsz=32)
    img = torch.rand(512 * 512 * 3).view(1, 3, 512, 512)
    model(img)


def test():
    test_model_forward()
    test_model_info()
    test_model_fuse()
    test_visualize_preds()
    test_val()
    test_model_resume()
    test_model_train_pretrained()


if __name__ == "__main__":
    test()
