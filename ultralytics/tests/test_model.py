import torch

from ultralytics.yolo import YOLO


def test_model_forward():
    model = YOLO()
    model.new("yolov5n-seg.yaml")
    img = torch.rand(512 * 512 * 3).view(1, 3, 512, 512)
    model.forward(img)
    model(img)


def test_model_info():
    model = YOLO()
    model.new("yolov5n.yaml")
    model.info()
    model.load("balloon-detect.pt")
    model.info(verbose=True)


def test_model_fuse():
    model = YOLO()
    model.new("yolov5n.yaml")
    model.fuse()
    model.load("balloon-detect.pt")
    model.fuse()


def test_visualize_preds():
    model = YOLO()
    model.load("balloon-segment.pt")
    model.predict(source="ultralytics/assets")


def test_val():
    model = YOLO()
    model.load("balloon-segment.pt")
    model.val(data="coco128-seg.yaml", img_size=32)


def test_model_resume():
    model = YOLO()
    model.new("yolov5n-seg.yaml")
    model.train(epochs=1, img_size=32, data="coco128-seg.yaml")
    try:
        model.resume(task="segment")
    except AssertionError:
        print("Successfully caught resume assert!")


def test_model_train_pretrained():
    model = YOLO()
    model.load("balloon-detect.pt")
    model.train(data="coco128.yaml", epochs=1, img_size=32)
    model.new("yolov5n.yaml")
    model.train(data="coco128.yaml", epochs=1, img_size=32)
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
