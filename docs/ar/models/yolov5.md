---
comments: true
description: اكتشف YOLOv5u، وهو إصدار معزز لنموذج YOLOv5 يوفر توازنًا محسنًا بين الدقة والسرعة والعديد من النماذج المدربة مسبقًا لمهام كشف الكائنات المختلفة.
keywords: YOLOv5u، كشف الكائنات، النماذج المدربة مسبقًا، Ultralytics، التشخيص، التحقق، YOLOv5، YOLOv8، بدون قاعدة تثبيت العقدة الرئيسية، بدون قيمة الكائن، التطبيقات الفعلية، تعلم الآلة
---

# YOLOv5

## نظرة عامة

يمثل YOLOv5u تقدمًا في منهجيات كشف الكائنات. يندرج YOLOv5u تحت البنية المعمارية الأساسية لنموذج [YOLOv5](https://github.com/ultralytics/yolov5) الذي طورته شركة Ultralytics، و يدمج نموذج YOLOv5u ميزة القسمة على جزئين للكائنات المستقلة عن القاعدة التي تم تقديمها في نماذج [YOLOv8](yolov8.md). تحسين هذا النمط يحسن نمط النموذج، مما يؤدي إلى تحسين التوازن بين الدقة والسرعة في مهام كشف الكائنات. بناءً على النتائج التجريبية والمزايا المشتقة منها، يقدم YOLOv5u بديلاً فعالًا لأولئك الذين يسعون لإيجاد حلول قوية في الأبحاث والتطبيقات العملية.

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## المزايا الرئيسية

- **رأس Ultralytics للقسمة بدون قاعدة تثبيت العقدة:** يعتمد نماذج كشف الكائنات التقليدية على صناديق قاعدة محددة مسبقًا لتوقع مواقع الكائنات. ومع ذلك، يحدث تحديث في نهج YOLOv5u هذا. من خلال اعتماد رأس Ultralytics المُقسم بدون قاعدة تثبيت العقدة، يضمن هذا النمط آلية كشف أكثر مرونة واندفاعًا، مما يعزز الأداء في سيناريوهات متنوعة.

- **توازن محسن بين الدقة والسرعة:** تتصارع السرعة والدقة في العديد من الأحيان. ولكن YOLOv5u يتحدى هذا التوازن. يقدم توازنًا معايرًا، ويضمن كشفًا في الوقت الفعلي دون المساومة على الدقة. تعد هذه الميزة ذات قيمة خاصة للتطبيقات التي تتطلب استجابة سريعة، مثل المركبات المستقلة والروبوتات وتحليل الفيديو في الوقت الفعلي.

- **مجموعة متنوعة من النماذج المدربة مسبقًا:** على فهم الأمور التي تحتاج إلى مجموعات أدوات مختلفة YOLOv5u يوفر العديد من النماذج المدربة مسبقًا. سواء كنت تركز على التشخيص أو التحقق أو التدريب، هناك نموذج مصمم خصيصًا ينتظرك. يضمن هذا التنوع أنك لا تستخدم حلاً من نوع واحد يناسب الجميع، ولكن نموذج موازن حسب حاجتك الفريدة.

## المهام والأوضاع المدعومة

تتفوق نماذج YOLOv5u، مع مجموعة متنوعة من الأوزان المدربة مسبقًا، في مهام [كشف الكائنات](../tasks/detect.md). تدعم هذه النماذج مجموعة شاملة من الأوضاع، مما يجعلها مناسبة لتطبيقات متنوعة، من التطوير إلى التنفيذ.

| نوع النموذج | الأوزان المدربة مسبقًا                                                                                                      | المهمة                             | التشخيص | التحقق | التدريب | التصدير |
|-------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------|---------|--------|---------|---------|
| YOLOv5u     | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [كشف الكائنات](../tasks/detect.md) | ✅       | ✅      | ✅       | ✅       |

يوفر هذا الجدول نظرة عامة مفصلة عن البدائل من نماذج نموذج YOLOv5u، ويسلط الضوء على تطبيقاتها في مهام كشف الكائنات ودعمها لأوضاع تشغيل متنوعة مثل [التشخيص](../modes/predict.md)، [التحقق](../modes/val.md)، [التدريب](../modes/train.md)، و[التصدير](../modes/export.md). يضمن هذا الدعم الشامل أن يمكن للمستخدمين استغلال قدرات نماذج YOLOv5u بشكل كامل في مجموعة واسعة من سيناريوهات كشف الكائنات.

## الأداء

!!! الأداء

    === "كشف"

    راجع [وثائق الكشف](https://docs.ultralytics.com/tasks/detect/) للحصول على أمثلة استخدام مع هذه النماذج المدربة على [COCO](https://docs.ultralytics.com/datasets/detect/coco/)، التي تشمل 80 فئة مدربة مسبقًا.

    | النموذج                                                                                      | يامل                                                                                                           | حجم<br><sup>(بكسل) | mAP<sup>val<br>50-95 | سرعة<br><sup>معالج الجهاز ONNX<br>(مللي ثانية) | سرعة<br><sup>حويصلة A100 TensorRT<br>(مللي ثانية) | المعلمات<br><sup>(مليون) | FLOPs<br><sup>(بليون) |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
    |                                                                                             |                                                                                                                |                       |                      |                                |                                     |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                 | 211.0                          | 1.83                                | 4.3                | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                 | 422.6                          | 2.34                                | 15.3               | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                 | 810.9                          | 4.36                                | 41.2               | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                 | 1470.9                         | 5.47                                | 86.1               | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                 | 2436.5                         | 8.98                                | 155.4              | 250.7             |

## أمثلة للاستخدام

يقدم هذا المثال أمثلة بسيطة للغاية للتدريب والتشخيص باستخدام YOLOv5. يُمكن إنشاء نموذج مثيل في البرمجة باستخدام نماذج PyTorch المدربة مسبقًا في صيغة `*.pt` وملفات التكوين `*.yaml`:

```python
from ultralytics import YOLO

#قم بتحميل نموذج YOLOv5n المدرب مسبقًا على مجموعة بيانات COCO
model = YOLO('yolov5n.pt')

# قم بعرض معلومات النموذج (اختياري)
model.info()

# قم بتدريب النموذج على مجموعة البيانات COCO8 لمدة 100 دورة
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

# قم بتشغيل التشخيص بنموذج YOLOv5n على صورة 'bus.jpg'
results = model('path/to/bus.jpg')
```

=== "سطر الأوامر"

    يتاح سطر الأوامر لتشغيل النماذج مباشرة:

    ```bash
    # قم بتحميل نموذج YOLOv5n المدرب مسبقًا على مجموعة بيانات COCO8 وقم بتدريبه لمدة 100 دورة
    yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

    # قم بتحميل نموذج YOLOv5n المدرب مسبقًا على مجموعة بيانات COCO8 وتشغيل حالة التشخيص على صورة 'bus.jpg'
    yolo predict model=yolov5n.pt source=path/to/bus.jpg
    ```

## الاستشهادات والتقدير

إذا قمت باستخدام YOLOv5 أو YOLOv5u في بحثك، يرجى استشهاد نموذج Ultralytics YOLOv5 بطريقة الاقتباس التالية:

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

يرجى ملاحظة أن نماذج YOLOv5 متاحة بترخيص [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) و[Enterprise](https://ultralytics.com/license).
