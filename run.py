from ultralytics import YOLO
model = YOLO("yolov8.yaml").load("yolov8n.pt")

model.export(format='onnx', 
             simplify=True, 
            #  export_hw_optimized=True, 
            #  separate_6_outputs=True,
             dynamic=True
            #  int8=True,
            #  data="coco128-seg.yaml"
            )

from ultralytics import YOLO
model = YOLO('yolov8.onnx', task='detect')
model.val(data='coco128.yaml', imgsz=640, device="cpu")

# results = model("img2.jpg", imgsz=640)

# from PIL import Image

# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image