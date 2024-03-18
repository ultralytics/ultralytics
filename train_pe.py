from ultralytics import YOLO


def main():

    model = YOLO('yolov8s.yaml')

    model = YOLO('yolov8s.pt')

    results = model.train(data='pe_module.yaml', epochs=100)

    results = model.val()


if __name__ == '__main__':
    main()