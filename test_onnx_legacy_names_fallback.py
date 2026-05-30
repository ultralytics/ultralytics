from ultralytics import YOLO

# model = YOLO("/home/mayanksingh/ultralytics/yolov8n.pt")
# print("--------------------------------------------------")
# print("         COMPLETELY FINE                          ")
# print("--------------------------------------------------")
# print(model.names)
# model.export(format="onnx", dynamic=True, imgsz=32)
# print("--------------------------------------------------")


# print("--------------------------------------------------")
# print("        COMPLETELY EMPTY OR NONE                  ")
# print("--------------------------------------------------")
# Below logic saves a model with no 'names' property
# model = YOLO("/home/mayanksingh/ultralytics/yolov8n.pt")
# delattr(model.model, "names")  -> # Delete 'names' to replicate a model with no 'names' property
# print(hasattr(model, "names")) -> # prints False
# model.save("/home/mayanksingh/ultralytics/model_no_names_prop.pt")
# print("SAVED")

# model = YOLO("/home/mayanksingh/ultralytics/model_no_names_prop.pt")
# model.export(format='onnx') # Try exporting
# print(model.names)

# print("--------------------------------------------------")
# print("      'names' has invalid type (Not dict)         ")
# print("--------------------------------------------------")
# model = YOLO("/home/mayanksingh/ultralytics/yolov8n.pt")
# model.model.names = ['human', 'cat', 'dog']
# print(model.names)
# model.export(format="onnx")
# print("--------------------------------------------------")

print("--------------------------------------------------")
print("     Malformed 'names' values                     ")
print("--------------------------------------------------")
model_1 = YOLO("/home/mayanksingh/ultralytics/yolov8n.pt")
model_2 = YOLO("/home/mayanksingh/ultralytics/yolov8n.pt")
model_3 = YOLO("/home/mayanksingh/ultralytics/yolov8n.pt")


print("      class ID is not int                          ")
model_1.model.names = {"0": "cat", "1": "dog"}
print(model_1.names)
model_1.export(format="onnx")
print(model_1.names)
print("--------------------------------------------------")

print("    class names are not str                      ")
model_2.model.names = {0: 0, 1: 1}
print(model_2.names)
model_2.export(format="onnx")
print(model_2.names)
print("--------------------------------------------------")

print("     Mixed types                                 ")
model_3.model.names = {0: "cat", "1": 4.5}
print(model_3.names)
model_3.export(format="onnx")
print(model_3.names)
print("--------------------------------------------------")
