---
comments: true
description: تعرّف على نماذج YOLOv8 Classify لتصنيف الصور. احصل على معلومات مفصلة حول قائمة النماذج المدرّبة مسبقًا وكيفية التدريب والتحقق والتنبؤ وتصدير النماذج.
keywords: Ultralytics، YOLOv8، تصنيف الصور، النماذج المدربة مسبقًا، YOLOv8n-cls، التدريب، التحقق، التنبؤ، تصدير النماذج
---

# تصنيف الصور

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="أمثلة على تصنيف الصور">

تعتبر عملية تصنيف الصور أبسط المهام الثلاثة وتنطوي على تصنيف صورة كاملة في إحدى الفئات المحددة سابقًا.

ناتج نموذج تصنيف الصور هو تسمية فئة واحدة ودرجة ثقة. يكون تصنيف الصور مفيدًا عندما تحتاج فقط إلى معرفة فئة الصورة ولا تحتاج إلى معرفة موقع الكائنات التابعة لتلك الفئة أو شكلها الدقيق.

!!! Tip "نصيحة"

    تستخدم نماذج YOLOv8 Classify اللاحقة "-cls"، مثالًا "yolov8n-cls.pt" وتم تدريبها على [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [النماذج](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

تظهر هنا النماذج المدرّبة مسبقًا لـ YOLOv8 للتصنيف. تم تدريب نماذج الكشف والشعبة والموضع على مجموعة البيانات [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)، بينما تم تدريب نماذج التصنيف مسبقًا على مجموعة البيانات [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

يتم تنزيل [النماذج](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) تلقائيًا من أحدث إصدار لـ Ultralytics [releases](https://github.com/ultralytics/assets/releases) عند الاستخدام الأول.

| النموذج                                                                                      | الحجم<br><sup>(بكسل) | دقة (أعلى 1)<br><sup>acc | دقة (أعلى 5)<br><sup>acc | سرعة التنفيذ<br><sup>ONNX للوحدة المركزية<br>(مللي ثانية) | سرعة التنفيذ<br><sup>A100 TensorRT<br>(مللي ثانية) | المعلمات<br><sup>(مليون) | FLOPs<br><sup>(مليار) لحجم 640 |
|----------------------------------------------------------------------------------------------|----------------------|--------------------------|--------------------------|-----------------------------------------------------------|----------------------------------------------------|--------------------------|--------------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt) | 224                  | 66.6                     | 87.0                     | 12.9                                                      | 0.31                                               | 2.7                      | 4.3                            |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224                  | 72.3                     | 91.1                     | 23.4                                                      | 0.35                                               | 6.4                      | 13.5                           |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224                  | 76.4                     | 93.2                     | 85.4                                                      | 0.62                                               | 17.0                     | 42.7                           |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224                  | 78.0                     | 94.1                     | 163.0                                                     | 0.87                                               | 37.5                     | 99.7                           |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224                  | 78.4                     | 94.3                     | 232.0                                                     | 1.01                                               | 57.4                     | 154.8                          |

- قيمة **acc** هي دقة النماذج على مجموعة بيانات التحقق [ImageNet](https://www.image-net.org/).
  <br>لإعادة إنتاج ذلك، استخدم `yolo val classify data=path/to/ImageNet device=0`
- يتم حساب سرعة **Speed** بناءً على متوسط صور التحقق من ImageNet باستخدام [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>لإعادة إنتاج ذلك، استخدم `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## التدريب

قم بتدريب YOLOv8n-cls على مجموعة بيانات MNIST160 لمدة 100 دورة عند حجم الصورة 64 بكسل. للحصول على قائمة كاملة بالوسائط المتاحة، اطلع على صفحة [تكوين](/../usage/cfg.md).

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل نموذج
        model = YOLO('yolov8n-cls.yaml')  # إنشاء نموذج جديد من نموذج YAML
        model = YOLO('yolov8n-cls.pt')  # تحميل نموذج مدرّب مسبقًا (موصى به للتدريب)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # إنشاء من YAML ونقل الأوزان

        # تدريب النموذج
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # إنشاء نموذج جديد من YAML وبدء التدريب من البداية
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # بدء التدريب من نموذج مدرّب بصيغة pt
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # إنشاء نموذج جديد من YAML ونقل الأوزان المدرّبة مسبقًا وبدء التدريب
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### تنسيق مجموعة البيانات

يمكن العثور على تنسيق مجموعة بيانات تصنيف YOLO بالتفصيل في [مرشد المجموعة](../../../datasets/classify/index.md).

## التحقق

قم بتحديد دقة النموذج YOLOv8n-cls المدرّب على مجموعة بيانات MNIST160. لا يلزم تمرير أي وسيطة حيث يحتفظ `model` ببيانات التدريب والوسائط كسمات النموذج.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل نموذج
        model = YOLO('yolov8n-cls.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مخصص

        # التحقق من النموذج
        metrics = model.val()  # لا تحتاج إلى وسائط، يتم تذكر مجموعة البيانات والإعدادات النموذج
        metrics.top1   # دقة أعلى 1
        metrics.top5   # دقة أعلى 5
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # تحقق من النموذج الرسمي
        yolo classify val model=path/to/best.pt  # تحقق من النموذج المخصص
        ```

## التنبؤ

استخدم نموذج YOLOv8n-cls المدرّب لتنفيذ تنبؤات على الصور.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل نموذج
        model = YOLO('yolov8n-cls.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مخصص

        # تنبؤ باستخدام النموذج
        results = model('https://ultralytics.com/images/bus.jpg')  # تنبؤ على صورة
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # تنبؤ باستخدام النموذج الرسمي
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # تنبؤ باستخدام النموذج المخصص
        ```

راجع تفاصيل كاملة حول وضع `predict` في الصفحة [Predict](https://docs.ultralytics.com/modes/predict/).

## تصدير

قم بتصدير نموذج YOLOv8n-cls إلى تنسيق مختلف مثل ONNX، CoreML، وما إلى ذلك.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل نموذج
        model = YOLO('yolov8n-cls.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مدرّب مخصص

        # تصدير النموذج
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # تصدير النموذج الرسمي
        yolo export model=path/to/best.pt format=onnx  # تصدير نموذج مدرّب مخصص
        ```

تتوفر صيغ تصدير YOLOv8-cls في الجدول أدناه. يمكنك تنبؤ أو التحقق من الصحة مباشرةً على النماذج المصدر، أي "yolo predict model=yolov8n-cls.onnx". يتم عرض أمثلة لاستخدام النموذج الخاص بك بعد الانتهاء من التصدير.

| الصيغة                                                             | وسيطة الصيغة  | النموذج                       | البيانات الوصفية | الوسيطات                                            |
|--------------------------------------------------------------------|---------------|-------------------------------|------------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n-cls.pt`              | ✅                | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n-cls.torchscript`     | ✅                | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n-cls.onnx`            | ✅                | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n-cls_openvino_model/` | ✅                | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n-cls.engine`          | ✅                | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n-cls.mlpackage`       | ✅                | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-cls_saved_model/`    | ✅                | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-cls.pb`              | ❌                | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-cls.tflite`          | ✅                | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-cls_edgetpu.tflite`  | ✅                | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-cls_web_model/`      | ✅                | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n-cls_paddle_model/`   | ✅                | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n-cls_ncnn_model/`     | ✅                | `imgsz`, `half`                                     |

راجع التفاصيل الكاملة حول `export` في الصفحة [Export](https://docs.ultralytics.com/modes/export/).
