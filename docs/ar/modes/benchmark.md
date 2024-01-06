---
comments: true
description: تعرف على كيفية قياس سرعة ودقة YOLOv8 عبر تنسيقات التصدير المختلفة. احصل على رؤى حول مقاييس mAP50-95 وaccuracy_top5 والمزيد.
keywords: Ultralytics، YOLOv8، اختبار الأداء، قياس السرعة، قياس الدقة، مقاييس mAP50-95 وaccuracy_top5، ONNX، OpenVINO، TensorRT، تنسيقات تصدير YOLO
---

# اختبار النموذج باستخدام Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## المقدمة

بمجرد أن يتم تدريب نموذجك وتحقق صحته ، فإن الخطوة التالية بشكل منطقي هي تقييم أدائه في سيناريوهات العالم الحقيقي المختلفة. يوفر وضع الاختبار في Ultralytics YOLOv8 هذا الهدف من خلال توفير إطار قوي لتقييم سرعة ودقة النموذج عبر مجموعة من صيغ التصدير.

## لماذا هو اختبار الأداء مهم؟

- **قرارات مستنيرة:** اكتساب رؤى حول التنازلات بين السرعة والدقة.
- **تخصيص الموارد:** فهم كيفية أداء تنسيقات التصدير المختلفة على أجهزة مختلفة.
- **تحسين:** تعلم أي تنسيق تصدير يقدم أفضل أداء لحالتك الاستخدامية المحددة.
- **كفاءة التكلفة:** استخدام الموارد الأجهزة بشكل أكثر كفاءة بناءً على نتائج الاختبار.

### المقاييس الرئيسية في وضع الاختبار

- **mAP50-95:** لكشف الكائنات وتقسيمها وتحديد الوضع.
- **accuracy_top5:** لتصنيف الصور.
- **وقت التتبع:** الوقت المستغرق لكل صورة بالميلي ثانية.

### تنسيقات التصدير المدعومة

- **ONNX:** لأفضل أداء على وحدة المعالجة المركزية.
- **TensorRT:** لأقصى استفادة من وحدة المعالجة الرسومية.
- **OpenVINO:** لتحسين الأجهزة من إنتل.
- **CoreML و TensorFlow SavedModel وما إلى ذلك:** لتلبية احتياجات النشر المتنوعة.

!!! Tip "نصيحة"

    * قم بتصدير إلى نموذج ONNX أو OpenVINO لزيادة سرعة وحدة المعالجة المركزية بمقدار 3 مرات.
    * قم بتصدير إلى نموذج TensorRT لزيادة سرعة وحدة المعالجة الرسومية بمقدار 5 مرات.

## أمثلة على الاستخدام

قم بتشغيل اختبارات YOLOv8n على جميع تنسيقات التصدير المدعومة بما في ذلك ONNX و TensorRT وما إلى ذلك. انظر القسم الموجود أدناه للحصول على قائمة كاملة من وسيطات التصدير.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # اختبار على وحدة المعالجة الرسومية
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## وسيطات

توفر الوسائط مثل `model` و `data` و `imgsz` و `half` و `device` و `verbose` مرونة للمستخدمين لضبط الاختبارات حسب احتياجاتهم المحددة ومقارنة أداء تنسيقات التصدير المختلفة بسهولة.

| المفتاح   | القيمة  | الوصف                                                                                             |
|-----------|---------|---------------------------------------------------------------------------------------------------|
| `model`   | `None`  | مسار إلى ملف النموذج ، على سبيل المثال yolov8n.pt ، yolov8n.yaml                                  |
| `data`    | `None`  | مسار إلى YAML يشير إلى مجموعة بيانات اختبار الأداء (بتحتوى على بيانات `val`)                      |
| `imgsz`   | `640`   | حجم الصورة كرقم ، أو قائمة (h ، w) ، على سبيل المثال (640، 480)                                   |
| `half`    | `False` | تقليل دقة العدد العشرى للأبعاد (FP16 quantization)                                                |
| `int8`    | `False` | تقليل دقة العدد الصحيح 8 بت (INT8 quantization)                                                   |
| `device`  | `None`  | الجهاز الذى ستعمل عليه العملية ، على سبيل المثال cuda device=0 أو device=0,1,2,3 أو device=cpu    |
| `verbose` | `False` | عدم المتابعة عند حدوث خطأ (مقدار منطقى)، أو مستوى الكشف عند تجاوز حد القيمة المطلوبة (قيمة عائمة) |

## صيغ التصدير

سيحاول التطبيق تشغيل الاختبارات تلقائيًا على جميع صيغ التصدير الممكنة الموجودة أدناه.

| Format                                                             | `format` Argument | Model                     | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

انظر تفاصيل التصدير الكاملة في الصفحة [Export](https://docs.ultralytics.com/modes/export/)
