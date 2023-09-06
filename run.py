from ultralytics import YOLO

model = YOLO('yolov8-seg.yaml').load('yolov8n-seg.pt')

model.export(
    format='onnx',
    simplify=True,
    export_hw_optimized=True,
    separate_outputs=True,
    imgsz=160
    #  dynamic=True
    #  int8=True,
    #  data="coco128.yaml"
)
from ultralytics import YOLO

model = YOLO('yolov8-seg.onnx', task='segment')
model.val(data='coco128-seg.yaml', imgsz=160, device='cpu', separate_outputs=True)

from ultralytics import YOLO

model = YOLO('yolov8-seg.onnx', task='segment')
model.predict('./ultralytics/assets/bus.jpg', imgsz=160, device='cpu', separate_outputs=True)

from ultralytics import YOLO

model = YOLO('yolov8-pose_saved_model/yolov8-pose_float32.tflite', task='pose')
model.val(data='coco8-pose.yaml', imgsz=640, device='cpu', separate_outputs=True)

from ultralytics import YOLO

model = YOLO('yolov8-pose_saved_model/yolov8-pose_float32.tflite', task='pose')
model.predict('./ultralytics/assets/bus.jpg', imgsz=640, device='cpu', separate_outputs=False)

from ultralytics import YOLO

model = YOLO('yolov8-pose.yaml').load('yolov8n-seg.pt')

model.export(
    format='onnx',
    simplify=True,
    export_hw_optimized=True,
    separate_outputs=True,
    #  dynamic=True
    #  int8=True,
    #  data="coco128.yaml"
)

from ultralytics import YOLO

model = YOLO('yolov8-pose_saved_model/yolov8-pose_int8.tflite', task='pose')
model.val(data='coco-pose.yaml', imgsz=640, device='cpu', separate_outputs=False)

# from ultralytics import YOLO
# model = YOLO('yolov5-p6.onnx', task='detect')
# model.val(data='coco128.yaml', imgsz=640, device="cpu")

# from ultralytics import YOLO
# model = YOLO('yolov8-seg.onnx', task='segment')
# model.val(data='coco128-seg.yaml', imgsz=640, device="cpu")

from ultralytics import YOLO

model = YOLO('yolov8-pose_saved_model/yolov8-pose_float32.tflite', task='pose')
model.predict('./ultralytics/assets/bus.jpg', imgsz=640, device='cpu', separate_outputs=True)

# from ultralytics import YOLO
# model = YOLO('yolov8n-pose.pt', task='pose')
# model.val(data='coco-pose.yaml', device="cpu")

# detect, segment - yolov8n

# yolov5n6u

# results = model("img2.jpg", imgsz=640)

# from PIL import Image

# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image
import csv

from ultralytics import YOLO


def dg_export(model, config, data, quant, format):
    model = YOLO(config).load(model)
    model.export(format=format, simplify=True, export_hw_optimized=True, separate_outputs=True, int8=quant, data=data)


def off_export(model, config, data, quant, format):
    model = YOLO(config).load(model)
    model.export(format=format, simplify=True, int8=quant, data=data)


def model_val(m_name, task, data):
    model = YOLO(m_name, task=task)
    return model.val(data=data, imgsz=640, device='cpu')


format = {'onnx', 'tflite'}
quant = [False, True]
dgexp = [False, True]
model = ['yolov8n.pt', 'yolov8n-seg.pt', 'yolov5n6u.pt']
onnx_name = ['yolov8.onnx', 'yolov8-seg.onnx', 'yolov5-p6.onnx']
tflite_name = [
    'yolov8_saved_model/yolov8_float32.tflite', 'yolov8-seg_saved_model/yolov8-seg_float32.tflite',
    'yolov5-p6_saved_model/yolov5-p6_float32.tflite']
int8_tflite_name = [
    'yolov8_saved_model/yolov8_full_integer_quant.tflite',
    'yolov8-seg_saved_model/yolov8-seg_full_integer_quant.tflite',
    'yolov5-p6_saved_model/yolov5-p6_full_integer_quant.tflite']
config = ['yolov8.yaml', 'yolov8-seg.yaml', 'yolov5-p6.yaml']
data = ['coco128.yaml', 'coco128-seg.yaml', 'coco128.yaml']
task = ['detect', 'segment', 'detect']

for index, value in enumerate(model):
    if value == 'yolov8n-seg.pt':
        for f in format:
            for q in quant:
                for dg in dgexp:
                    if dg == True:
                        dg_export(value, config[index], data[index], q, f)
                    else:
                        off_export(value, config[index], data[index], q, f)

                    if f == 'tflite':
                        m_name = int8_tflite_name[index] if q else tflite_name[index]
                    else:
                        m_name = onnx_name[index]

                    metric = model_val(m_name, task[index], data[index])

                    csv_data = [value.replace('.pt', ''), f, str(q), str(dg), metric.box.map, metric.box.map50]
                    with open('metrics.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows([csv_data])
