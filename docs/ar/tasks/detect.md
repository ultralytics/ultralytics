---
comments: true
description: وثائق رسمية لـ YOLOv8 بواسطة Ultralytics. تعلم كيفية تدريب و التحقق من صحة و التنبؤ و تصدير النماذج بتنسيقات مختلفة. تتضمن إحصائيات الأداء التفصيلية.
keywords: YOLOv8, Ultralytics, التعرف على الكائنات, النماذج المدربة من قبل, التدريب, التحقق من الصحة, التنبؤ, تصدير النماذج, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# التعرف على الكائنات

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="Beispiele für die Erkennung von Objekten">

Task التعرف على الكائنات هو عبارة عن تعرف على موقع و فئة الكائنات في صورة أو فيديو.

مخرجات جهاز الاستشعار هي مجموعة من مربعات تحيط بالكائنات في الصورة، مع تصنيف الفئة ودرجات وثقة لكل مربع. التعرف على الكائنات هو اختيار جيد عندما تحتاج إلى تحديد كائنات مهمة في مشهد، ولكنك لا تحتاج إلى معرفة بالضبط أين يكمن الكائن أو شكله الدقيق.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="مشغل فيديو YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>شاهد:</strong> التعرف على الكائنات باستخدام نموذج Ultralytics YOLOv8 مع تدريب مسبق.
</p>

!!! Tip "تلميح"

    نماذج YOLOv8 Detect هي النماذج الافتراضية YOLOv8، أي `yolov8n.pt` و هي مدربة مسبقًا على [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [النماذج](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

تُعرض هنا النماذج المدربة مسبقًا لـ YOLOv8 Detect. النماذج Detect و Segment و Pose معتمدة على مجموعة البيانات [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)، بينما النماذج Classify معتمدة على مجموعة البيانات [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

تُقوم النماذج بالتنزيل تلقائيًا من أحدث [إصدار Ultralytics](https://github.com/ultralytics/assets/releases) عند الاستخدام لأول مرة.

| النموذج                                                                              | الحجم<br><sup>(بكسل) | mAP<sup>val<br>50-95 | السرعة<br><sup>CPU ONNX<br>(مللي ثانية) | السرعة<br><sup>A100 TensorRT<br>(مللي ثانية) | الوزن<br><sup>(ميغا) | FLOPs<br><sup>(مليار) |
|--------------------------------------------------------------------------------------|----------------------|----------------------|-----------------------------------------|----------------------------------------------|----------------------|-----------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                  | 37.3                 | 80.4                                    | 0.99                                         | 3.2                  | 8.7                   |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                  | 44.9                 | 128.4                                   | 1.20                                         | 11.2                 | 28.6                  |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                  | 50.2                 | 234.7                                   | 1.83                                         | 25.9                 | 78.9                  |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                  | 52.9                 | 375.2                                   | 2.39                                         | 43.7                 | 165.2                 |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                  | 53.9                 | 479.1                                   | 3.53                                         | 68.2                 | 257.8                 |

- قيم mAP<sup>val</sup> تنطبق على مقياس نموذج واحد-مقياس واحد على مجموعة بيانات [COCO val2017](https://cocodataset.org).
  <br>اعيد حسابها بواسطة `yolo val detect data=coco.yaml device=0`
- السرعةتمت متوسطة على صور COCO val باستخدام [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>اعيد حسابها بواسطة `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

## تدريب

قم بتدريب YOLOv8n على مجموعة البيانات COCO128 لمدة 100 دورة على حجم صورة 640. للحصول على قائمة كاملة بالوسائط المتاحة انظر الصفحة [التكوين](/../usage/cfg.md).

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل نموذج
        model = YOLO('yolov8n.yaml')  # بناء نموذج جديد من YAML
        model = YOLO('yolov8n.pt')  # قم بتحميل نموذج مدرب مسبقًا (موصى به للتدريب)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # بناء من YAML و نقل الأوزان

        # قم بتدريب النموذج
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # قم ببناء نموذج جديد من YAML وابدأ التدريب من الصفر
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # ابدأ التدريب من نموذج *.pt مدرب مسبقًا
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # بناء نموذج جديد من YAML، ونقل الأوزان المدربة مسبقاً إلى النموذج وابدأ التدريب
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### تنسيق مجموعة بيانات

يمكن العثور على تنسيق مجموعة بيانات التعرف على الكائنات بالتفصيل في [دليل مجموعة البيانات](../../../datasets/detect/index.md). لتحويل مجموعة البيانات الحالية من تنسيقات أخرى (مثل COCO الخ) إلى تنسيق YOLO، يرجى استخدام أداة [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) المقدمة من Ultralytics.

## التحقق من الصحة

قم بتحقق من دقة النموذج المدرب مسبقًا YOLOv8n على مجموعة البيانات COCO128. ليس هناك حاجة إلى تمرير أي وسيطات حيث يحتفظ النموذج ببياناته التدريبية والوسيطات كسمات النموذج.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل نموذج
        model = YOLO('yolov8n.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مخصص

        # قم بالتحقق من النموذج
        metrics = model.val()  # لا حاجة لأي بيانات، يتذكر النموذج بيانات التدريب و الوسيطات
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # قائمة تحتوي map50-95 لكل فئة
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # التحقق من النموذج الرسمي
        yolo detect val model=path/to/best.pt  # التحقق من النموذج المخصص
        ```

## التنبؤ

استخدم نموذج YOLOv8n المدرب مسبقًا لتشغيل التنبؤات على الصور.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل نموذج
        model = YOLO('yolov8n.pt')  # قم بتحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # قم بتحميل نموذج مخصص

        # أجرِ التنبؤ باستخدام النموذج
        results = model('https://ultralytics.com/images/bus.jpg')  # التنبؤ على صورة
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # التنبؤ باستخدام النموذج الرسمي
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # التنبؤ بالنموذج المخصص
        ```

انظر تفاصيل وضع الـ `predict` الكامل في صفحة [Predict](https://docs.ultralytics.com/modes/predict/).

## تصدير

قم بتصدير نموذج YOLOv8n إلى تنسيق مختلف مثل ONNX، CoreML وغيرها.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # قم بتحميل نموذج
        model = YOLO('yolov8n.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مدرب مخصص

        # قم بتصدير النموذج
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # تصدير النموذج الرسمي
        yolo export model=path/to/best.pt format=onnx  # تصدير النموذج المدرب مخصص
        ```

التنسيقات المدعومة لتصدير YOLOv8 مدرجة في الجدول أدناه. يمكنك التنبؤ أو التحقق من صحة النماذج المصدرة مباشرة، على سبيل المثال `yolo predict model=yolov8n.onnx`. سيتم عرض أمثلة استخدام لنموذجك بعد اكتمال التصدير.

| الشكل                                                              | مسافة `format` | النموذج                   | بيانات الوصف | وسيطات                                              |
|--------------------------------------------------------------------|----------------|---------------------------|--------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | - أو           | `yolov8n.pt`              | ✅            | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`  | `yolov8n.torchscript`     | ✅            | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`         | `yolov8n.onnx`            | ✅            | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`     | `yolov8n_openvino_model/` | ✅            | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`       | `yolov8n.engine`          | ✅            | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`       | `yolov8n.mlpackage`       | ✅            | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`  | `yolov8n_saved_model/`    | ✅            | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`           | `yolov8n.pb`              | ❌            | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`       | `yolov8n.tflite`          | ✅            | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`      | `yolov8n_edgetpu.tflite`  | ✅            | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`         | `yolov8n_web_model/`      | ✅            | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`       | `yolov8n_paddle_model/`   | ✅            | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`         | `yolov8n_ncnn_model/`     | ✅            | `imgsz`, `half`                                     |

انظر تفاصيل كاملة للـ `export` في صفحة [Export](https://docs.ultralytics.com/modes/export/).
