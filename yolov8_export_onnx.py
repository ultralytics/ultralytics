from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    print("1--------------------------------------------------------------------")
    model = YOLO('./runs/detect/train4/weights/best.pt')
    # Export the model
    result = model.export(format='onnx', imgsz=(640, 640), opset=12, simplify=True)
    print("2--------------------------------------------------------------------")
    print(result)
