---
comments: true
description: استكشف ميزات YOLOv8 المثيرة، أحدث إصدار من أداة الكشف عن الكائنات الفعلية لدينا! تعرف على التطورات المتقدمة، والنماذج المدربة مسبقًا، والتوازن المثالي بين الدقة والسرعة التي تجعل YOLOv8 الخيار المثالي لمهام الكشف عن الكائنات الخاصة بك.
keywords: YOLOv8، Ultralytics، أداة الكشف عن الكائنات الفعلية في الوقت الحقيقي، النماذج المدربة مسبقًا، الوثائق، الكشف عن الكائنات، سلسلة YOLO، التطبيقات المتقدمة، الدقة، السرعة
---

# YOLOv8

## نظرة عامة

YOLOv8 هو أحدث إصدار في سلسلة YOLO لأدوات الكشف في الوقت الحقيقي، ويوفر أداءً قطعيًا من حيث الدقة والسرعة. يستند YOLOv8 إلى تطورات الإصدارات السابقة من YOLO ويقدم ميزات وتحسينات جديدة تجعله خيارًا مثاليًا لمختلف مهام الكشف عن الكائنات في مجموعة واسعة من التطبيقات.

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## الميزات الرئيسية

- **معماريات ظهرية وبينية متقدمة:** يستخدم YOLOv8 معماريات ظهرية وبينية حديثة، مما يؤدي إلى تحسين استخراج الميزات وأداء الكشف عن الكائنات.
- **رأس معماري Anchor-free Split Ultralytics:** يعتمد YOLOv8 رأس Ultralytics بدون محور للأنكور، مما يسهم في زيادة الدقة وعملية الكشف الأكثر كفاءة مقارنةً بالأساليب التي تعتمد على الأنكور.
- **توازن الدقة والسرعة المحسن:** بتركيز على الحفاظ على توازن مثالي بين الدقة والسرعة، يعد YOLOv8 مناسبًا لمهام الكشف عن الكائنات في الوقت الفعلي في مجالات التطبيق المتنوعة.
- **مجموعة متنوعة من النماذج المدربة مسبقًا:** يقدم YOLOv8 مجموعة من النماذج المدربة مسبقًا لتلبية مختلف المهام ومتطلبات الأداء، مما يجعل من الأسهل إيجاد النموذج المناسب لحالة الاستخدام الخاصة بك.

## المهام المدعومة

| نوع النموذج | وزن مدرب مسبقًا                                                                                                     | المهمة            |
|-------------|---------------------------------------------------------------------------------------------------------------------|-------------------|
| YOLOv8      | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | كشف               |
| YOLOv8-seg  | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | تجزئة النمط       |
| YOLOv8-pose | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | موضع/نقاط مفتاحية |
| YOLOv8-cls  | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | تصنيف             |

## الوضعيات المدعومة

| الوضعية | مدعومة |
|---------|--------|
| التصدير | ✅      |
| التحقق  | ✅      |
| التدريب | ✅      |

!!! الأداء

    === "الكشف (COCO)"

        | النموذج                                                                                                  | الحجم<br><sup>(بكسل) | mAP<sup>val<br>50-95 | السرعة<br><sup>CPU ONNX<br>(ملي ثانية) | السرعة<br><sup>أي ١٠٠ TensorRT<br>(ملي ثانية) | الآليات<br><sup>(مليون) | FLOPs<br><sup>(بليون) |
        | ------------------------------------------------------------------------------------------------------- | ---------------------- | -------------------- | ---------------------------------------- | ---------------------------------------- | ----------------------- | -------------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)                      | 640                    | 37.3                 | 80.4                                   | 0.99                                     | 3.2                     | 8.7                  |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)                      | 640                    | 44.9                 | 128.4                                  | 1.20                                     | 11.2                    | 28.6                 |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)                      | 640                    | 50.2                 | 234.7                                  | 1.83                                     | 25.9                    | 78.9                 |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)                      | 640                    | 52.9                 | 375.2                                  | 2.39                                     | 43.7                    | 165.2                |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)                      | 640                    | 53.9                 | 479.1                                  | 3.53                                     | 68.2                    | 257.8                |

    === "الكشف (صور مفتوحة V7)"

        انظر إلى [وثائق الكشف](https://docs.ultralytics.com/tasks/detect/) لمعرفة أمثلة استخدام هذه النماذج المدربة على [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/). التي تشمل 600 فئة مدربة مسبقًا.

        | النموذج                                                                                                    | الحجم<br><sup>(بكسل) | mAP<sup>val<br>50-95 | السرعة<br><sup>CPU ONNX<br>(ملي ثانية) | السرعة<br><sup>أي ١٠٠ TensorRT<br>(ملي ثانية) | الآليات<br><sup>(مليون) | FLOPs<br><sup>(بليون) |
        | -------------------------------------------------------------------------------------------------------- | ---------------------- | -------------------- | ---------------------------------------- | ---------------------------------------- | ----------------------- | -------------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt)                   | 640                    | 18.4                 | 142.4                                  | 1.21                                     | 3.5                     | 10.5                 |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt)                   | 640                    | 27.7                 | 183.1                                  | 1.40                                     | 11.4                    | 29.7                 |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt)                   | 640                    | 33.6                 | 408.5                                  | 2.26                                     | 26.2                    | 80.6                 |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt)                   | 640                    | 34.9                 | 596.9                                  | 2.43                                     | 44.1                    | 167.4                |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt)                   | 640                    | 36.3                 | 860.6                                  | 3.56                                     | 68.7                    | 260.6                |

    === "التجزئة (COCO)"

        | النموذج                                                                                                    | الحجم<br><sup>(بكسل) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | السرعة<br><sup>CPU ONNX<br>(ملي ثانية) | السرعة<br><sup>أي ١٠٠ TensorRT<br>(ملي ثانية) | الآليات<br><sup>(مليون) | FLOPs<br><sup>(بليون) |
        | -------------------------------------------------------------------------------------------------------- | ---------------------- | -------------------- | --------------------- | ---------------------------------------- | ---------------------------------------- | ----------------------- | -------------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt)               | 640                    | 36.7                 | 30.5                  | 96.1                                   | 1.21                                     | 3.4                     | 12.6                 |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt)               | 640                    | 44.6                 | 36.8                  | 155.7                                  | 1.47                                     | 11.8                    | 42.6                 |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt)               | 640                    | 49.9                 | 40.8                  | 317.0                                  | 2.18                                     | 27.3                    | 110.2                |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt)               | 640                    | 52.3                 | 42.6                  | 572.4                                  | 2.79                                     | 46.0                    | 220.5                |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt)               | 640                    | 53.4                 | 43.4                  | 712.1                                  | 4.02                                     | 71.8                    | 344.1                |

    === "التصنيف (ImageNet)"

        | النموذج                                                                                                    | الحجم<br><sup>(بكسل) | دققة TP-الأعلى | دققة TP-الأعلى | السرعة<br><sup>CPU ONNX<br>(ملي ثانية) | السرعة<br><sup>أي ١٠٠ TensorRT<br>(ملي ثانية) | الآليات<br><sup>(مليون) | FLOPs<br><sup>(بليون) عند 640 |
        | -------------------------------------------------------------------------------------------------------- | ---------------------- | ----------------- | ----------------- | ---------------------------------------- | ---------------------------------------- | ----------------------- | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt)               | 224                    | 66.6              | 87.0              | 12.9                                   | 0.31                                     | 2.7                     | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt)               | 224                    | 72.3              | 91.1              | 23.4                                   | 0.35                                     | 6.4                     | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt)               | 224                    | 76.4              | 93.2              | 85.4                                   | 0.62                                     | 17.0                    | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt)               | 224                    | 78.0              | 94.1              | 163.0                                  | 0.87                                     | 37.5                    | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt)               | 224                    | 78.4              | 94.3              | 232.0                                  | 1.01                                     | 57.4                    | 154.8                    |

    === "موضع/نقاط مفتاحية (COCO)"

        | النموذج                                                                                                        | الحجم<br><sup>(بكسل) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | السرعة<br><sup>CPU ONNX<br>(ملي ثانية) | السرعة<br><sup>أي ١٠٠ TensorRT<br>(ملي ثانية) | الآليات<br><sup>(مليون) | FLOPs<br><sup>(بليون) |
        | ------------------------------------------------------------------------------------------------------------ | ---------------------- | -------------------- | ------------------ | ---------------------------------------- | ---------------------------------------- | ----------------------- | -------------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)                 | 640                    | 50.4                 | 80.1               | 131.8                                   | 1.18                                     | 3.3                     | 9.2                  |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)                 | 640                    | 60.0                 | 86.2               | 233.2                                   | 1.42                                     | 11.6                    | 30.2                 |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)                 | 640                    | 65.0                 | 88.8               | 456.3                                   | 2.00                                     | 26.4                    | 81.0                 |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)                 | 640                    | 67.6                 | 90.0               | 784.5                                   | 2.59                                     | 44.4                    | 168.6                |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)                 | 640                    | 69.2                 | 90.2               | 1607.1                                  | 3.73                                     | 69.4                    | 263.2                |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt)           | 1280                   | 71.6                 | 91.2               | 4088.7                                  | 10.04                                    | 99.1                    | 1066.4               |

## الاستخدام

يمكنك استخدام YOLOv8 لمهام الكشف عن الكائنات باستخدام حزمة Ultralytics pip. يعرض المقطع التالي رمزًا نموذجيًا يوضح كيفية استخدام نماذج YOLOv8 للكشف:

!!! Exemple ""

    يوفر هذا المثال رمزًا بسيطًا للتكشف باستخدام YOLOv8. لمزيد من الخيارات بما في ذلك التعامل مع نتائج التكشف، انظر إلى [Predict](../modes/predict.md) "الوضع". لاستخدام YOLOv8 مع وضعيات إضافية أخرى، انظر إلى [Train](../modes/train.md)، [Val](../modes/val.md) و [Export](../modes/export.md) "الوضع".

    === "Python"

        يمكن تمرير نماذج PyTorch الجاهزة `*.pt` بالإضافة إلى ملفات التكوين `*.yaml` إلى فئة `YOLO()` في بيثون لإنشاء نموذج:

        ```python
        from ultralytics import YOLO

        # تحميل نموذج YOLOv8n المعتمد مسبقًا على COCO
        model = YOLO('yolov8n.pt')

        # عرض معلومات النموذج (اختياري)
        model.info()

        # قم بتدريب النموذج على مجموعة بيانات مثال COCO8 لمدة 100 دورة تدريب
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # قم بتشغيل التكهنات باستخدام نموذج YOLOv8n على صورة 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        تتوفر أوامر CLI لتشغيل النماذج مباشرة:

        ```bash
        # تحميل نموذج YOLOv8n المعتمد مسبقًا على COCO وتدريبه على مجموعة بيانات مثال COCO8 لمدة 100 دورة تدريب
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # تحميل نموذج YOLOv8n المعتمد مسبقًا على COCO وتشغيل التكهنات على صورة 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## الاقتباسات والاعترافات

إذا قمت باستخدام نموذج YOLOv8 أو أي برنامج آخر من هذا المستودع في عملك، يرجى استشهاده باستخدام التنسيق التالي:

!!! Note ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

يرجى ملاحظة أن هوية الموضع (DOI) غير متوفرة حاليًا وسيتم إضافتها إلى الاقتباس بمجرد توفرها. استخدام البرنامج يتم وفقًا لترخيص AGPL-3.0.
