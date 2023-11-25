---
comments: true
description: استكشف FastSAM ، وهو حلاً مبنيًا على الشبكات العصبية السريعة لتجزئة الكائنات في الوقت الحقيقي في الصور. تفاعل المستخدم المحسّن ، والكفاءة الحسابية ، والقابلية للتكيف في مهام الرؤية المختلفة.
keywords: FastSAM ، التعلم الآلي ، حلاً مبنيًا على الشبكات العصبية السريعة ، قسيمة الكائنات ، حلاً في الوقت الحقيقي ، Ultralytics ، مهام الرؤية ، معالجة الصور ، تطبيقات صناعية ، تفاعل المستخدم
---

# نموذج تجزئة أي شيء بسرعة عالية (FastSAM)

نموذج تجزئة أي شيء بسرعة عالية (FastSAM) هو حلاً مبتكرًا للعصب الشبكي يعمل بالزمن الحقيقي لمهمة تجزئة أي كائن داخل صورة ما. تم تصميم هذه المهمة لتجزئة أي كائن داخل صورة بناءً على إشارات تفاعل المستخدم المختلفة الممكنة. يقلل الـ FastSAM من الاحتياجات الحسابية بشكل كبير مع الحفاظ على أداء تنافسي ، مما يجعله خيارًا عمليًا لمجموعة متنوعة من مهام الرؤية.

![نظرة عامة على تصميم نموذج تجزئة أي شيء بسرعة عالية (FastSAM)](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## نظرة عامة

تم تصميم FastSAM للتغلب على القيود الموجودة في [نموذج تجزئة ما شيء (SAM)](sam.md) ، وهو نموذج تحويل ثقيل يتطلب موارد حسابية كبيرة. يفصل FastSAM عملية تجزئة أي شيء إلى مرحلتين متسلسلتين: تجزئة جميع الأمثلة واختيار موجه بناءً على التعليمات. تستخدم المرحلة الأولى [YOLOv8-seg](../tasks/segment.md) لإنتاج قناع التجزئة لجميع الأمثلة في الصورة. في المرحلة الثانية ، يتم إخراج منطقة الاهتمام المتعلقة بالتعليمة.

## المميزات الرئيسية

1. **حلاً في الوقت الحقيقي**: من خلال استغلال كفاءة الشبكات العصبية الحاسوبية ، يوفر FastSAM حلاً في الوقت الحقيقي لمهمة تجزئة أي شيء ، مما يجعله قيمًا للتطبيقات الصناعية التي تتطلب نتائج سريعة.

2. **كفاءة وأداء**: يقدم FastSAM تقليل كبير في الاحتياجات الحسابية واستخدام الموارد دون التنازل عن جودة الأداء. يحقق أداءً قابلاً للمقارنة مع SAM ولكن بموارد حسابية مخفضة بشكل كبير ، مما يمكن من تطبيقه في الوقت الحقيقي.

3. **تجزئة يستند إلى الموجه**: يمكن لـ FastSAM تجزئة أي كائن داخل صورة ترشده مختلف إشارات تفاعل المستخدم الممكنة ، مما يوفر مرونة وقابلية للتكيف في سيناريوهات مختلفة.

4. **يستند إلى YOLOv8-seg**: يستند FastSAM إلى [YOLOv8-seg](../tasks/segment.md) ، وهو كاشف كائنات مجهز بفرع تجزئة المثيلات. يمكنه بشكل فعال إنتاج قناع التجزئة لجميع الأمثلة في صورة.

5. **نتائج تنافسية في الاختبارات التحضيرية**: في مهمة اقتراح الكائن على MS COCO ، يحقق FastSAM درجات عالية بسرعة أسرع بكثير من [SAM](sam.md) على بطاقة NVIDIA RTX 3090 واحدة ، مما يدل على كفاءته وقدرته.

6. **تطبيقات عملية**: توفر الطريقة المقترحة حلاً جديدًا وعمليًا لعدد كبير من مهام الرؤية بسرعة عالية حقًا ، بمعدلات سرعة عشرات أو مئات المرات أسرع من الطرق الحالية.

7. **جدوى ضغط النموذج**: يظهر FastSAM إمكانية تقليل الجهد الحسابي بشكل كبير من خلال إدخال سابق اصطناعي للهيكل ، مما يفتح إمكانيات جديدة لهندسة هيكل النموذج الكبير لمهام الرؤية العامة.

## النماذج المتاحة ، المهام المدعومة ، وأوضاع التشغيل

يعرض هذا الجدول النماذج المتاحة مع أوزانها المحددة ، والمهام التي تدعمها ، ومدى توافقها مع أوضاع التشغيل المختلفة مثل [الاستنتاج](../modes/predict.md) ، [التحقق](../modes/val.md) ، [التدريب](../modes/train.md) ، و[التصدير](../modes/export.md) ، مشار إليها برموز الـ✅ للأوضاع المدعومة والرموز ❌ للأوضاع غير المدعومة.

| نوع النموذج | أوزان تم تدريبها مسبقًا | المهام المدعومة                       | الاستنتاج | التحقق | التدريب | التصدير |
|-------------|-------------------------|---------------------------------------|-----------|--------|---------|---------|
| FastSAM-s   | `FastSAM-s.pt`          | [تجزئة المثيلات](../tasks/segment.md) | ✅         | ❌      | ❌       | ✅       |
| FastSAM-x   | `FastSAM-x.pt`          | [تجزئة المثيلات](../tasks/segment.md) | ✅         | ❌      | ❌       | ✅       |

## أمثلة الاستخدام

يسهل دمج نماذج FastSAM في تطبيقات Python الخاصة بك. يوفر Ultralytics واجهة برمجة تطبيقات Python سهلة الاستخدام وأوامر CLI لتسهيل التطوير.

### استخدام التوقعات

للقيام بكشف الكائنات في صورة ، استخدم طريقة `predict` كما هو موضح أدناه:

!!! Example "مثال"

    === "بايثون"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # حدد مصدر التوقع
        source = 'path/to/bus.jpg'

        # قم بإنشاء نموذج FastSAM
        model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

        # تنفيذ توقعات على صورة
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # قم بتجهيز كائن معالج مع قواعد التوقع
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # التوقع باستخدام كل شيء
        ann = prompt_process.everything_prompt()

        # bbox الشكل الافتراضي [0،0،0،0] -> [x1،y1،x2،y2]
        ann = prompt_process.box_prompt(bbox=[200، 200، 300، 300])

        # التوقع النصي
        ann = prompt_process.text_prompt(text='صورة لكلب')

        # التوقع النقطي
        ann = prompt_process.point_prompt(points=[[200، 200]]، pointlabel=[1])
        prompt_process.plot(annotations=ann، output='./')
        ```

    === "CLI"
        ```bash
        # قم بتحميل نموذج FastSAM وتجزئة كل شيء به
        yolo segment predict model=FastSAM-s.pt source=path/to/bus.jpg imgsz=640
        ```

توضح هذه المقاطع البساطة في تحميل نموذج مدرب مسبقًا وتنفيذ توقع على صورة.

### استخدام مهام التحقق

يمكن تنفيذ التحقق من النموذج على مجموعة بيانات على النحو التالي:

!!! Example "مثال"

    === "بايثون"
        ```python
        from ultralytics import FastSAM

        # قم بإنشاء نموذج FastSAM
        model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

        # قم بتنفيذ التحقق من النموذج
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # قم بتحميل نموذج FastSAM وأجرِ التحقق منه بخصوص مجموعة البيانات مثال كوكو 8 بحجم صورة 640
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

يرجى ملاحظة أن الـ FastSAM يدعم فقط الكشف والتجزئة لفئة واحدة من الكائن. هذا يعني أنه سيتعرف ويجزء جميع الكائنات على أنها نفس الفئة. لذلك ، عند إعداد مجموعة البيانات ، يجب تحويل جميع معرفات فئة الكائن إلى 0.

## استخدام FastSAM الرسمي

يتوفر نموذج FastSAM مباشرةً من مستودع [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM). فيما يلي نظرة عامة موجزة على الخطوات التقليدية التي قد تتخذها لاستخدام FastSAM:

### التثبيت

1. استنسخ مستودع FastSAM:
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. أنشئ بيئة Conda وفعّلها بـ Python 3.9:
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. انتقل إلى المستودع المنسخ وقم بتثبيت الحزم المطلوبة:
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. قم بتثبيت نموذج CLIP:
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### مثال الاستخدام

1. قم بتنزيل [تفويض نموذج](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing).

2. استخدم FastSAM للتوقع. أمثلة الأوامر:

    - تجزئة كل شيء في صورة:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - تجزئة كائنات محددة باستخدام تعليمات النص:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "الكلب الأصفر"
      ```

    - تجزئة كائنات داخل مربع محدد (تقديم إحداثيات الصندوق في تنسيق xywh):
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - تجزئة كائنات قرب النقاط المحددة:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

بالإضافة إلى ذلك ، يمكنك تجربة FastSAM من خلال [Colab demo](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing) أو على [HuggingFace web demo](https://huggingface.co/spaces/An-619/FastSAM) لتجربة بصرية.

## الاقتباسات والشكر

نود أن نشكر أباء FastSAM على مساهماتهم الهامة في مجال تجزئة المثيلات في الوقت الحقيقي:

!!! Quote ""

    === "بيب تيكس"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

يمكن العثور على ورقة FastSAM الأصلية على [arXiv](https://arxiv.org/abs/2306.12156). قام الأباء بجعل أعمالهم متاحة للجمهور ، ويمكن الوصول إلى قاعدة الكود على [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM). نقدر جهودهم في تطوير المجال وجعل أعمالهم متاحة للمجتمع الأوسع.
