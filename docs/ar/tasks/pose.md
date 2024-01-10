---
comments: true
description: تعرّف على كيفية استخدام Ultralytics YOLOv8 لمهام تقدير الوضعية. اعثر على نماذج مدرّبة مسبقًا، وتعلم كيفية التدريب والتحقق والتنبؤ وتصدير نموذجك الخاص.
keywords: Ultralytics، YOLO، YOLOv8، تقدير الوضعية ، كشف نقاط المفاتيح ، كشف الكائنات ، نماذج مدرّبة مسبقًا ، تعلم الآلة ، الذكاء الاصطناعي
---

# تقدير الوضعية

تقدير الوضعية هو مهمة تنطوي على تحديد موقع نقاط محددة في الصورة ، وعادةً ما يشار إليها بنقاط الوضوح. يمكن أن تمثل نقاط الوضوح أجزاءً مختلفةً من الكائن مثل المفاصل أو العلامات المميزة أو الميزات البارزة الأخرى. عادةً ما يتم تمثيل مواقع نقاط الوضوح كمجموعة من الإحداثيات 2D `[x ، y]` أو 3D `[x ، y ، visible]`.

يكون ناتج نموذج تقدير الوضعية مجموعة من النقاط التي تمثل نقاط الوضوح على كائن في الصورة ، عادةً مع نقاط الثقة لكل نقطة. تقدير الوضعية هو خيار جيد عندما تحتاج إلى تحديد أجزاء محددة من كائن في مشهد، وموقعها بالنسبة لبعضها البعض.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="مشغل فيديو YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>شاهد:</strong> تقدير الوضعية مع Ultralytics YOLOv8.
</p>

!!! Tip "نصيحة"

    النماذج التي تحتوي على البادئة "-pose" تستخدم لنماذج YOLOv8 pose ، على سبيل المثال `yolov8n-pose.pt`. هذه النماذج مدربة على [مجموعة بيانات نقاط الوضوح COCO]("https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml") وهي مناسبة لمجموعة متنوعة من مهام تقدير الوضعية.

## [النماذج]("https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8")

تعرض نماذج مدرّبة مسبقًا لـ YOLOv8 التي تستخدم لتقدير الوضعية هنا. النماذج للكشف والشريحة والوضعية يتم تدريبها على [مجموعة بيانات COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)، بينما تتم تدريب نماذج التصنيف على مجموعة بيانات ImageNet.

يتم تنزيل النماذج من [آخر إصدار Ultralytics]("https://github.com/ultralytics/assets/releases") تلقائيًا عند استخدامها لأول مرة.

| النموذج                                                                                              | الحجم (بالبكسل) | mAP<sup>الوضعية 50-95 | mAP<sup>الوضعية 50 | سرعة<sup>الوحدة المركزية ONNX<sup>(ms) | سرعة<sup>A100 TensorRT<sup>(ms) | المعلمات (مليون) | FLOPs (بالمليار) |
|------------------------------------------------------------------------------------------------------|-----------------|-----------------------|--------------------|----------------------------------------|---------------------------------|------------------|------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640             | 50.4                  | 80.1               | 131.8                                  | 1.18                            | 3.3              | 9.2              |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640             | 60.0                  | 86.2               | 233.2                                  | 1.42                            | 11.6             | 30.2             |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640             | 65.0                  | 88.8               | 456.3                                  | 2.00                            | 26.4             | 81.0             |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640             | 67.6                  | 90.0               | 784.5                                  | 2.59                            | 44.4             | 168.6            |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640             | 69.2                  | 90.2               | 1607.1                                 | 3.73                            | 69.4             | 263.2            |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280            | 71.6                  | 91.2               | 4088.7                                 | 10.04                           | 99.1             | 1066.4           |

- تعتبر القيم **mAP<sup>val</sup>** لنموذج واحد ومقياس واحد فقط على [COCO Keypoints val2017](http://cocodataset.org)
  مجموعة البيانات.
  <br>يمكن إعادة إنتاجه بواسطة `يولو val pose data=coco-pose.yaml device=0`
- يتم حساب **السرعة** من خلال متوسط صور COCO val باستخدام [المروحة الحرارية Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  مثيل.
  <br>يمكن إعادة إنتاجه بواسطة `يولو val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## التدريب

يتم تدريب نموذج YOLOv8-pose على مجموعة بيانات COCO128-pose.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل النموذج
        model = YOLO('yolov8n-pose.yaml')  # بناء نموذج جديد من ملف YAML
        model = YOLO('yolov8n-pose.pt')  # تحميل نموذج مدرّب مسبقًا (موصى به للتدريب)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # بناء نموذج من YAML ونقل الوزن

        # تدريب النموذج
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # بناء نموذج جديد من YAML وبدء التدريب من البداية.
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # البدء في التدريب من نموذج مدرب مسبقًا *.pt
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # بناء نموذج جديد من YAML ، ونقل الأوزان المدرّبة مسبقًا إليه ، والبدء في التدريب.
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### تنسيق مجموعة البيانات

يمكن العثور على تنسيق مجموعات بيانات نقاط الوضوح YOLO في [دليل المجموعة البيانات](../../../datasets/pose/index.md). لتحويل مجموعة البيانات الحالية التي لديك من تنسيقات أخرى (مثل COCO إلخ) إلى تنسيق YOLO ، يرجى استخدام أداة [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) من Ultralytics.

## التحقق من الصحة

تحقق من دقة نموذج YOLOv8n-pose المدرّب على مجموعة بيانات COCO128-pose. لا يلزم تمرير سبب ما كوسيط إلى `model`
عند استدعاء.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل النموذج
        model = YOLO('yolov8n-pose.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مخصص

        # التحقق من النموذج
        metrics = model.val()  # لا يوجد حاجة لأي سبب، يتذكر النموذج البيانات والوسائط كمجالات للنموذج
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # قائمة تحتوي على map50-95 لكل فئة
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # التحقق من النموذج الرسمي
        yolo pose val model=path/to/best.pt  # التحقق من النموذج المخصص
        ```

## التنبؤ

استخدم نموذج YOLOv8n-pose المدرّب لتشغيل توقعات على الصور.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل النموذج
        model = YOLO('yolov8n-pose.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مخصص

        # التنبؤ باستخدام النموذج
        results = model('https://ultralytics.com/images/bus.jpg')  # التنبؤ بصورة
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # التنبؤ باستخدام النموذج الرسمي
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # التنبؤ باستخدام النموذج المخصص
        ```

انظر تفاصيل `predict` كاملة في [صفحة التنبؤ](https://docs.ultralytics.com/modes/predict/).

## التصدير

قم بتصدير نموذج YOLOv8n-pose إلى تنسيق مختلف مثل ONNX، CoreML، الخ.

!!! Example "مثال"

    === "Python"

        ```python
        from ultralytics import YOLO

        # تحميل النموذج
        model = YOLO('yolov8n-pose.pt')  # تحميل نموذج رسمي
        model = YOLO('path/to/best.pt')  # تحميل نموذج مدرب مخصص

        # تصدير النموذج
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # تصدير نموذج رسمي
        yolo export model=path/to/best.pt format=onnx  # تصدير نموذج مخصص
        ```

تتوفر تنسيقات تصدير YOLOv8-pose في الجدول أدناه. يمكنك التنبؤ أو التحقق مباشرةً على النماذج المصدرة ، على سبيل المثال `yolo predict model=yolov8n-pose.onnx`. توجد أمثلة استخدام متاحة لنموذجك بعد اكتمال عملية التصدير.

| تنسيق                                                              | إجراء `format` | النموذج                        | البيانات الوصفية | الوسائط                                             |
|--------------------------------------------------------------------|----------------|--------------------------------|------------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -              | `yolov8n-pose.pt`              | ✅                | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`  | `yolov8n-pose.torchscript`     | ✅                | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`         | `yolov8n-pose.onnx`            | ✅                | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`     | `yolov8n-pose_openvino_model/` | ✅                | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`       | `yolov8n-pose.engine`          | ✅                | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`       | `yolov8n-pose.mlpackage`       | ✅                | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`  | `yolov8n-pose_saved_model/`    | ✅                | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`           | `yolov8n-pose.pb`              | ❌                | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`       | `yolov8n-pose.tflite`          | ✅                | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`      | `yolov8n-pose_edgetpu.tflite`  | ✅                | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`         | `yolov8n-pose_web_model/`      | ✅                | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`       | `yolov8n-pose_paddle_model/`   | ✅                | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`         | `yolov8n-pose_ncnn_model/`     | ✅                | `imgsz`, `half`                                     |

انظر تفاصيل `export` كاملة في [صفحة التصدير](https://docs.ultralytics.com/modes/export/).
