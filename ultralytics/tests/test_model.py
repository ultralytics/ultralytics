from ultralytics.yolo import YOLO


def test_model():
    model = YOLO()
    model.new("assets/dummy_model.yaml")
    model.model = "squeezenet1_0"  # temp solution before get_model is implemented
    # model.load("yolov5n.pt")
    model.train(data="imagenette160", epochs=1, lr0=0.01)


if __name__ == "__main__":
    test_model()
