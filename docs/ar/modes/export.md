---
comments: true
description: دليل خطوة بخطوة حول تصدير نماذج YOLOv8 الخاصة بك إلى تنسيقات مختلفة مثل ONNX و TensorRT و CoreML وغيرها للنشر. استكشف الآن!.
keywords: YOLO، YOLOv8، Ultralytics، تصدير النموذج، ONNX، TensorRT، CoreML، TensorFlow SavedModel، OpenVINO، PyTorch، تصدير النموذج
---

# تصدير النموذج باستخدام يولو من Ultralytics

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="إكوسيستم يولو من Ultralytics والتكاملات">

## مقدمة

الهدف النهائي لتدريب نموذج هو نشره لتطبيقات العالم الحقيقي. يوفر وضع التصدير في يولو من Ultralytics مجموعة متنوعة من الخيارات لتصدير النموذج المدرب إلى تنسيقات مختلفة، مما يجعله يمكن استخدامه في مختلف الأنظمة والأجهزة. يهدف هذا الدليل الشامل إلى مساعدتك في فهم تفاصيل تصدير النموذج، ويعرض كيفية تحقيق أقصى توافق وأداء.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="مشغل فيديو YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>شاهد:</strong> كيفية تصدير نموذج Ultralytics YOLOv8 التدريب المخصص وتشغيل الاستدلال المباشر على كاميرا الويب.
</p>

## لماذا اختيار وضع تصدير YOLOv8؟

- **التنوع:** تصدير إلى تنسيقات متعددة بما في ذلك ONNX و TensorRT و CoreML ، وغيرها.
- **الأداء:** الحصول على سرعة تسريع تصل إلى 5 أضعاف باستخدام TensorRT وسرعة تسريع معالج الكمبيوتر المركزي بنسبة 3 أضعاف باستخدام ONNX أو OpenVINO.
- **التوافقية:** جعل النموذج قابلاً للنشر على الأجهزة والبرامج المختلفة.
- **سهولة الاستخدام:** واجهة سطر الأوامر البسيطة وواجهة برمجة Python لتصدير النموذج بسرعة وسهولة.

### الميزات الرئيسية لوضع التصدير

إليك بعض من الميزات المميزة:

- **تصدير بنقرة واحدة:** أوامر بسيطة لتصدير إلى تنسيقات مختلفة.
- **تصدير الدُفعات:** تصدير نماذج قادرة على العمل مع الدُفعات.
- **تنفيذ محسَّن:** يتم تحسين النماذج المصدرة لتوفير وقت تنفيذ أسرع.
- **فيديوهات تعليمية:** مرشدين وفيديوهات تعليمية لتجربة تصدير سلسة.

!!! Tip "نصيحة"

    * صدّر إلى ONNX أو OpenVINO للحصول على تسريع معالج الكمبيوتر المركزي بنسبة 3 أضعاف.
    * صدّر إلى TensorRT للحصول على تسريع وحدة المعالجة الرسومية بنسبة 5 أضعاف.

## أمثلة للاستخدام

قم بتصدير نموذج YOLOv8n إلى تنسيق مختلف مثل ONNX أو TensorRT. انظر الجدول أدناه للحصول على قائمة كاملة من وسائط التصدير.

!!! Example "مثال"

    === "بايثون"

        ```python
        from ultralytics import YOLO

        # قم بتحميل نموذج
        model = YOLO('yolov8n.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مدرب مخصص

        # قم بتصدير النموذج
        model.export(format='onnx')
        ```
    === "واجهة سطر الأوامر"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # تصدير نموذج رسمي
        yolo export model=path/to/best.pt format=onnx  # تصدير نموذج مدرب مخصص
        ```

## الوسائط

تشير إعدادات تصدير YOLO إلى التكوينات والخيارات المختلفة المستخدمة لحفظ أو تصدير النموذج للاستخدام في بيئات أو منصات أخرى. يمكن أن تؤثر هذه الإعدادات على أداء النموذج وحجمه وتوافقه مع الأنظمة المختلفة. تشمل بعض إعدادات تصدير YOLO الشائعة تنسيق ملف النموذج المصدر (مثل ONNX وتنسيق TensorFlow SavedModel) والجهاز الذي سيتم تشغيل النموذج عليه (مثل المعالج المركزي أو وحدة المعالجة الرسومية) ووجود ميزات إضافية مثل الأقنعة أو التسميات المتعددة لكل مربع. قد تؤثر عوامل أخرى قد تؤثر عملية التصدير تشمل المهمة النموذجة المحددة التي يتم استخدام النموذج لها ومتطلبات أو قيود البيئة أو المنصة المستهدفة. من المهم أن ننظر بعناية ونقوم بتكوين هذه الإعدادات لضمان أن النموذج المصدر هو محسَّن للحالة الاستخدام المقصودة ويمكن استخدامه بشكل فعال في البيئة المستهدفة.

| المفتاح     | القيمة          | الوصف                                                                 |
|-------------|-----------------|-----------------------------------------------------------------------|
| `format`    | `'torchscript'` | التنسيق المراد تصديره                                                 |
| `imgsz`     | `640`           | حجم الصورة كمقدار علمي أو قائمة (h ، w) ، على سبيل المثال (640 ، 480) |
| `keras`     | `False`         | استخدام Keras لتصدير TF SavedModel                                    |
| `optimize`  | `False`         | TorchScript: الأمثل للجوال                                            |
| `half`      | `False`         | تكميم FP16                                                            |
| `int8`      | `False`         | تكميم INT8                                                            |
| `dynamic`   | `False`         | ONNX/TensorRT: المحاور الديناميكية                                    |
| `simplify`  | `False`         | ONNX/TensorRT: تبسيط النموذج                                          |
| `opset`     | `None`          | ONNX: إصدار opset (اختياري ، الافتراضي هو الأحدث)                     |
| `workspace` | `4`             | TensorRT: حجم مساحة العمل (GB)                                        |
| `nms`       | `False`         | CoreML: إضافة NMS                                                     |

## تنسيقات التصدير

صيغ تصدير YOLOv8 المتاحة في الجدول أدناه. يمكنك التصدير إلى أي تنسيق باستخدام الوسيطة `format` ، مثل `format='onnx'` أو `format='engine'`.

| التنسيق                                                            | وسيطة format  | النموذج                   | البيانات الوصفية | الوسائط                                             |
|--------------------------------------------------------------------|---------------|---------------------------|------------------|-----------------------------------------------------|
| [بايثورش](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅                | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `تورتشسيريبت` | `yolov8n.torchscript`     | ✅                | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅                | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅                | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅                | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅                | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅                | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌                | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅                | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅                | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅                | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅                | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅                | `imgsz`, `half`                                     |
