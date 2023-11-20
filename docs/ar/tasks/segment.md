---
comments: true
description: تعلم كيفية استخدام نماذج فصل الأشكال الفردية مع Ultralytics YOLO. تعليمات حول التدريب والتحقق من الصحة وتوقع الصورة وتصدير النموذج.
keywords: yolov8 ، فصل الأشكال الفردية ، Ultralytics ، مجموعة بيانات COCO ، تجزئة الصورة ، كشف الكائنات ، تدريب النموذج ، التحقق من صحة النموذج ، توقع الصورة ، تصدير النموذج
---

# فصل الأشكال الفردية

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="أمثلة على فصل الأشكال الفردية">

يذهب فصل الأشكال الفردية خطوة أبعد من كشف الكائنات وينطوي على تحديد الكائنات الفردية في صورة وتجزيئها عن بقية الصورة.

ناتج نموذج فصل الأشكال الفردية هو مجموعة من الأقنعة أو الحدود التي تحدد كل كائن في الصورة ، جنبًا إلى جنب مع تصنيف الصنف ونقاط الثقة لكل كائن. يكون فصل الأشكال الفردية مفيدًا عندما تحتاج إلى معرفة ليس فقط أين توجد الكائنات في الصورة ، ولكن أيضًا ما هو شكلها الدقيق.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="مشغل فيديو YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>المشاهدة:</strong> تشغيل فصل الأشكال مع نموذج Ultralytics YOLOv8 مدرب مسبقًا باستخدام Python.
</p>

!!! Tip "نصيحة"

    تستخدم نماذج YOLOv8 Seg اللاحقة `-seg`، أي `yolov8n-seg.pt` وتكون مدربة مسبقًا على [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [النماذج](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

تُعرض هنا النماذج الجاهزة المدربة مسبقًا لـ YOLOv8 Segment. يتم تدريب نماذج الكشف والتجزيء والمواقف على مجموعة البيانات [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) ، بينما تدرب نماذج التصنيف على مجموعة البيانات [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

تتم تنزيل [النماذج](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) تلقائيًا من [الإصدار](https://github.com/ultralytics/assets/releases) الأخير لـ Ultralytics عند أول استخدام.

| النموذج                                                                                      | الحجم<br><sup>بكسل | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | السرعة<br><sup>CPU ONNX<br>(مللي ثانية) | السرعة<br><sup>A100 TensorRT<br>(مللي ثانية) | المعلمات<br><sup>(مليون) | FLOPs<br><sup>(مليار) |
|----------------------------------------------------------------------------------------------|--------------------|----------------------|-----------------------|-----------------------------------------|----------------------------------------------|--------------------------|-----------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                | 36.7                 | 30.5                  | 96.1                                    | 1.21                                         | 3.4                      | 12.6                  |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                | 44.6                 | 36.8                  | 155.7                                   | 1.47                                         | 11.8                     | 42.6                  |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                | 49.9                 | 40.8                  | 317.0                                   | 2.18                                         | 27.3                     | 110.2                 |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                | 52.3                 | 42.6                  | 572.4                                   | 2.79                                         | 46.0                     | 220.5                 |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                | 53.4                 | 43.4                  | 712.1                                   | 4.02                                         | 71.8                     | 344.1                 |

- تُستخدم قيم **mAP<sup>val</sup>** لنموذج واحد وحجم واحد على مجموعة بيانات [COCO val2017](http://cocodataset.org).
  <br>يمكن إعادة إنتاجها باستخدام `yolo val segment data=coco.yaml device=0`
- **تُحسب السرعة** كمتوسط على صور COCO val باستخدام [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>يمكن إعادة إنتاجها باستخدام `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`

## التدريب

قم بتدريب YOLOv8n-seg على مجموعة بيانات COCO128-seg لمدة 100 دورة عند حجم صورة 640. للحصول على قائمة كاملة بالوسائط المتاحة ، راجع صفحة [التكوين](/../usage/cfg.md).

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل النموذج
        model = YOLO('yolov8n-seg.yaml')  # قم ببناء نموذج جديد من ملف YAML
        model = YOLO('yolov8n-seg.pt')  # قم بتحميل نموذج مدرب مسبقًا (موصى به للتدريب)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # قم ببنائه من YAML ونقل الوزن

        # قم بتدريب النموذج
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # قم ببناء نموذج جديد من ملف YAML وبدء التدريب من البداية
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # قم ببدء التدريب من نموذج *.pt مدرب مسبقًا
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # قم ببناء نموذج جديد من YAML ونقل الأوزان المدربة مسبَقًا إليه وابدأ التدريب
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### تنسيق مجموعة البيانات

يمكن العثور على تنسيق مجموعة بيانات تجزيء YOLO بالتفصيل في [دليل مجموعة البيانات](../../../datasets/segment/index.md). لتحويل مجموعة البيانات الحالية التي تتبع تنسيقات أخرى (مثل COCO إلخ) إلى تنسيق YOLO ، يُرجى استخدام أداة [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) من Ultralytics.

## التحقق من الصحة

قم بالتحقق من دقة نموذج YOLOv8n-seg المدرب على مجموعة بيانات COCO128-seg. لا حاجة لتمرير أي وسيطة كما يحتفظ النموذج ببيانات "تدريبه" والوسيطات كسمات النموذج.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل النموذج
        model = YOLO('yolov8n-seg.pt')  # قم بتحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # قم بتحميل نموذج مخصص

        # قم بالتحقق من النموذج
        metrics = model.val()  # لا حاجة إلى أي وسيطة ، يتذكر النموذج بيانات التدريب والوسيطات كسمات النموذج
        metrics.box.map    # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps   # قائمة تحتوي على map50-95(B) لكل فئة
        metrics.seg.map    # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps   # قائمة تحتوي على map50-95(M) لكل فئة
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # التحقق من النموذج الرسمي
        yolo segment val model=path/to/best.pt  # التحقق من النموذج المخصص
        ```

## التنبؤ

استخدم نموذج YOLOv8n-seg المدرب للقيام بالتنبؤات على الصور.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل النموذج
        model = YOLO('yolov8n-seg.pt')  # قم بتحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # قم بتحميل نموذج مخصص

        # التنبؤ باستخدام النموذج
        results = model('https://ultralytics.com/images/bus.jpg')  # التنبؤ على صورة
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # التنبؤ باستخدام النموذج الرسمي
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # التنبؤ باستخدام النموذج المخصص
        ```

انظر تفاصيل "التنبؤ" الكاملة في [الصفحة](https://docs.ultralytics.com/modes/predict/).

## التصدير

قم بتصدير نموذج YOLOv8n-seg إلى تنسيق مختلف مثل ONNX و CoreML وما إلى ذلك.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل النموذج
        model = YOLO('yolov8n-seg.pt')  # قم بتحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # قم بتحميل نموذج مدرب مخصص

        # قم بتصدير النموذج
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # تصدير نموذج رسمي
        yolo export model=path/to/best.pt format=onnx  # تصدير نموذج مدرب مخصص
        ```

صيغ تصدير YOLOv8-seg المتاحة في الجدول أدناه. يمكنك التنبؤ أو التحقق من صحة الموديل المصدر بشكل مباشر ، أي `yolo predict model=yolov8n-seg.onnx`. يتم عرض أمثلة عن الاستخدام لنموذجك بعد اكتمال التصدير.

| الصيغة                                                             | `format` Argument | النموذج                       | التعليمات | الخيارات                                        |
|--------------------------------------------------------------------|-------------------|-------------------------------|-----------|-------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-seg.pt`              | ✅         | -                                               |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-seg.torchscript`     | ✅         | `الحجم ، الأمان`                                |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-seg.onnx`            | ✅         | `الحجم ، half ، dynamic ، simplify ، opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-seg_openvino_model/` | ✅         | `الحجم ، half`                                  |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-seg.engine`          | ✅         | `الحجم ، half ، dynamic ، simplify ، workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-seg.mlpackage`       | ✅         | `الحجم ، half ، int8 ، nms`                     |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-seg_saved_model/`    | ✅         | `الحجم ، keras`                                 |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-seg.pb`              | ❌         | `الحجم`                                         |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-seg.tflite`          | ✅         | `الحجم ، half ، int8`                           |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-seg_edgetpu.tflite`  | ✅         | `الحجم`                                         |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-seg_web_model/`      | ✅         | `الحجم`                                         |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-seg_paddle_model/`   | ✅         | `الحجم`                                         |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-seg_ncnn_model/`     | ✅         | `الحجم ، half`                                  |

انظر تفاصيل "التصدير" الكاملة في [الصفحة](https://docs.ultralytics.com/modes/export/).
