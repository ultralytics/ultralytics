---
comments: true
description: استكشف الميزات المثيرة لـ YOLOv8 ، أحدث إصدار من مكتشف الكائنات الحية الخاص بنا في الوقت الحقيقي! تعرّف على العمارات المتقدمة والنماذج المدرّبة مسبقًا والتوازن المثلى بين الدقة والسرعة التي تجعل YOLOv8 الخيار المثالي لمهام الكشف عن الكائنات الخاصة بك.
keywords: YOLOv8, Ultralytics, مكتشف الكائنات الحية الخاص بنا في الوقت الحقيقي, النماذج المدرّبة مسبقًا, وثائق, الكشف عن الكائنات, سلسلة YOLO, العمارات المتقدمة, الدقة, السرعة
---

# YOLOv8

## نظرة عامة

YOLOv8 هو التطور الأخير في سلسلة YOLO لمكتشفات الكائنات الحية الخاصة بنا في الوقت الحقيقي ، والذي يقدم أداءً متقدمًا في مجال الدقة والسرعة. بناءً على التقدمات التي تم إحرازها في إصدارات YOLO السابقة ، يقدم YOLOv8 ميزات وتحسينات جديدة تجعله الخيار المثالي لمهام الكشف عن الكائنات في مجموعة واسعة من التطبيقات.

![YOLOv8 المقدمة من Ultralytics](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## الميزات الرئيسية

- **العمارات المتقدمة للظهر والعنق:** يعتمد YOLOv8 على عمارات الظهر والعنق على أحدث طراز ، مما يؤدي إلى تحسين استخراج الميزات وأداء الكشف عن الكائنات.
- **Ultralytics Head بدون إثبات خطافي:** يعتمد YOLOv8 على Ultralytics Head بدون إثبات خطافي ، مما يسهم في زيادة الدقة وتوفير عملية كشف أكثر كفاءة مقارنةً بالطرق التي تعتمد على الإثبات.
- **توازن مثالي بين الدقة والسرعة محسَّن:** بتركيزه على الحفاظ على توازن مثالي بين الدقة والسرعة ، فإن YOLOv8 مناسب لمهام الكشف عن الكائنات في الوقت الحقيقي في مجموعة متنوعة من المجالات التطبيقية.
- **تشكيلة من النماذج المدرّبة مسبقًا:** يقدم YOLOv8 مجموعة من النماذج المدرّبة مسبقًا لتلبية متطلبات المهام المختلفة ومتطلبات الأداء ، مما يجعل من السهل إيجاد النموذج المناسب لحالتك الاستخدامية الخاصة.

## المهام والأوضاع المدعومة

تقدم سلسلة YOLOv8 مجموعة متنوعة من النماذج ، يتم تخصيص كل نموذج للمهام المحددة في رؤية الكمبيوتر. تم تصميم هذه النماذج لتلبية متطلبات مختلفة ، بدءًا من الكشف عن الكائنات إلى المهام الأكثر تعقيدًا مثل تجزئة الشخصيات ، وكشف المواقف / النقاط الإرشادية ، والتصنيف.

تم تحسين كل نسخة من سلسلة YOLOv8 للمهمة المعنية ، مما يضمن الأداء العالي والدقة. بالإضافة إلى ذلك ، تتوافق هذه النماذج مع وسائط تشغيل مختلفة بما في ذلك [العمليات](../modes/predict.md), [التحقق](../modes/val.md), [التدريب](../modes/train.md), و [التصدير](../modes/export.md) ، مما يسهل استخدامها في مراحل مختلفة من التنفيذ والتطوير.

| النموذج     | أسماء الملفات                                                                                                  | المهمة                                         | التنبؤ | التحقق | التدريب | التصدير |
|-------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------|--------|--------|---------|---------|
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [الكشف](../tasks/detect.md)                    | ✅      | ✅      | ✅       | ✅       |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [تجزئة الشخصيات](../tasks/segment.md)          | ✅      | ✅      | ✅       | ✅       |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [التوصيل / النقاط الإرشادية](../tasks/pose.md) | ✅      | ✅      | ✅       | ✅       |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [التصنيف](../tasks/classify.md)                | ✅      | ✅      | ✅       | ✅       |

توفر هذه الجدولة نظرة عامة على النماذج المتخصصة في سلسلة YOLOv8 ، مُسلطًا الضوء على استخدامها في المهام المحددة وتوافقها مع أوضاع التشغيل المختلفة مثل التنبؤ والتحقق والتدريب والتصدير. إنه يظهر مرونة وقوة سلسلة YOLOv8 ، مما يجعلها مناسبة لمجموعة متنوعة من تطبيقات رؤية الكمبيوتر.

## قياسات الأداء

!!! الأداء

    === "الكشف (COCO)"

        انظر إلى [وثائق الكشف](https://docs.ultralytics.com/tasks/detect/) للحصول على أمثلة عن الاستخدام مع تلك النماذج المدربة مسبقًا على [COCO](https://docs.ultralytics.com/datasets/detect/coco/) ، وهي تتضمن 80 فئة مدربة مسبقًا.

        | النموذج                                                                                | الحجم<br><sup>(بكسل) | mAP<sup>val<br>50-95 | السرعة<br><sup>CPU ONNX<br>(مللي ثانية) | السرعة<br><sup>A100 TensorRT<br>(مللي ثانية) | البارامترات<br><sup>(ملايين) | FLOPs<br><sup>(بلايين) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "الكشف (Open Images V7)"

        انظر إلى [وثائق الكشف](https://docs.ultralytics.com/tasks/detect/) للحصول على أمثلة عن الاستخدام مع تلك النماذج المدربة مسبقًا على [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) ، وهي تتضمن 600 فئة مدربة مسبقًا.

        | النموذج                                                                                     | الحجم<br><sup>(بكسل) | mAP<sup>val<br>50-95 | السرعة<br><sup>CPU ONNX<br>(مللي ثانية) | السرعة<br><sup>A100 TensorRT<br>(مللي ثانية) | البارامترات<br><sup>(ملايين) | FLOPs<br><sup>(بلايين) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "تجزئة الشخصيات (COCO)"

        انظر إلى [وثائق تجريب](https://docs.ultralytics.com/tasks/segment/) للحصول على أمثلة عن الاستخدام مع تلك النماذج المدربة مسبقًا على [COCO](https://docs.ultralytics.com/datasets/segment/coco/) ، والتي تتضمن 80 فئة مدربة مسبقًا.

        | النموذج                                                                                        | الحجم<br><sup>(بكسل) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | السرعة<br><sup>CPU ONNX<br>(مللي ثانية) | السرعة<br><sup>A100 TensorRT<br>(مللي ثانية) | البارامترات<br><sup>(ملايين) | FLOPs<br><sup>(بلايين) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "تصنيف (ImageNet)"

        انظر إلى [وثائق التصنيف](https://docs.ultralytics.com/tasks/classify/) للحصول على أمثلة عن الاستخدام مع تلك النماذج المدربة مسبقًا على [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) ، والتي تتضمن 1000 فئة مدربة مسبقًا.

        | النموذج                                                                                        | الحجم<br><sup>(بكسل) | الدقة العلوية<br>في المرتبة الأولى | الدقة العلوية<br>تصنيف 5 | السرعة<br><sup>CPU ONNX<br>(مللي ثانية) | السرعة<br><sup>A100 TensorRT<br>(مللي ثانية) | البارامترات<br><sup>(ملايين) | FLOPs<br><sup>(بلايين) عند 640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "المواقف (COCO)"

        انظر إلى [وثائق تقدير المواقف](https://docs.ultralytics.com/tasks/segment/) للحصول على أمثلة عن الاستخدام مع تلك النماذج المدربة مسبقًا على [COCO](https://docs.ultralytics.com/datasets/pose/coco/) ، والتي تتضمن فئة واحدة مدربة مسبقًا ، 'شخص'.

        | النموذج                                                                                                | الحجم<br><sup>(بكسل) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | السرعة<br><sup>CPU ONNX<br>(مللي ثانية) | السرعة<br><sup>A100 TensorRT<br>(مللي ثانية) | البارامترات<br><sup>(ملايين) | FLOPs<br><sup>(بلايين) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## أمثلة الاستخدام

يوفر هذا المثال أمثلة بسيطة للتدريب والتنبؤ باستخدام YOLOv8. بالنسبة للتوثيق الكامل حول هذه وأنماط أخرى [الوضع](../modes/index.md) انظر إلى صفحات [التنبؤ](../modes/predict.md),  [التدريب](../modes/train.md), [التحقق](../modes/val.md) و [التصدير](../modes/export.md).

يُرجى ملاحظة أن المثال أدناه هو لنماذج YOLOv8 [الكشف](../tasks/detect.md) للكشف عن الكائنات. لرؤية المهام المدعومة الإضافية ، راجع [التجزئة](../tasks/segment.md) ، [التصنيف](../tasks/classify.md) ، و[المواقف](../tasks/pose.md) docs.

!!! Examlpe

    === "بالبايثون"

        يمكن تمرير نماذج PyTorch المعتمدة مسبقًا `*.pt` بالإضافة إلى ملفات تكوين `*.yaml` إلى فئة `YOLO ()` لإنشاء نموذج في بيثون:

        ```python
        from ultralytics import YOLO

        # تحميل نموذج YOLOv8n المعتمد مسبقًا على COCO
        model = YOLO('yolov8n.pt')

        # عرض معلومات النموذج (اختياري)
        model.info()

        # تدريب النموذج على مجموعة بيانات مثال COCO8 لمدة 100 حقبة
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # تشغيل التنبؤ بنموذج YOLOv8n على صورة 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "السطر الطرفي"

        يتوفر أوامر CLI لتشغيل النماذج مباشرةً:

        ```bash
        # تحميل نموذج YOLOv8n المعتمد مسبقًا على COCO ثم تدريبه على مجموعة بيانات مثال COCO8 لمدة 100 حقبة
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # تحميل نموذج YOLOv8n المعتمد مسبقًا على COCO ثم تشغيل التنبؤ على صورة 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/image.jpg
        ```

## اقتباسات وتحيات

إذا قمت باستخدام نموذج YOLOv8 أو أي برنامج آخر من هذا المستودع في عملك ، يرجى استشهاده باستخدام النموذج التالي:

!!! اقتباس ""

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

    يُرجى ملاحظة أن رمز التعريف الرقمي الذي تم استخدامه لم يتم منحه بعد وسيتم إضافته إلى الاقتباس بمجرد توفره. تتوفر نماذج YOLOv8 بموجب [تراخيص](https://ultralytics.com/license) [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) و [Enterprise](https://ultralytics.com/license)
