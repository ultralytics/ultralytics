---
comments: true
description: تعرّف على MobileSAM وتطبيقه، وقارنه مع SAM الأصلي، وكيفية تنزيله واختباره في إطار Ultralytics. قم بتحسين تطبيقاتك المحمولة اليوم.
keywords: MobileSAM، Ultralytics، SAM، التطبيقات المحمولة، Arxiv، GPU، API، مُشفّر الصورة، فك تشفير القناع، تنزيل النموذج، طريقة الاختبار
---

![شعار MobileSAM](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# التمييز المحمول لأي شيء (MobileSAM)

الآن يمكنك الاطّلاع على ورقة MobileSAM في [arXiv](https://arxiv.org/pdf/2306.14289.pdf).

يمكن الوصول إلى عرض مباشر لـ MobileSAM يعمل على وحدة المعالجة المركزية CPU من [هنا](https://huggingface.co/spaces/dhkim2810/MobileSAM). يستغرق الأداء على وحدة المعالجة المركزية Mac i5 تقريبًا 3 ثوانٍ. في عرض الواجهة التفاعلية الخاص بهنغ فيس، تؤدي واجهة المستخدم ووحدات المعالجة المركزية ذات الأداء المنخفض إلى استجابة أبطأ، لكنها تواصل العمل بفعالية.

تم تنفيذ MobileSAM في عدة مشاريع بما في ذلك [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) و [AnyLabeling](https://github.com/vietanhdev/anylabeling) و [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D).

تم تدريب MobileSAM على وحدة المعالجة الرسومية (GPU) الواحدة باستخدام مجموعة بيانات تحتوي على 100000 صورة (1% من الصور الأصلية) في أقل من يوم واحد. سيتم توفير الشفرة المصدرية لعملية التدريب هذه في المستقبل.

## النماذج المتاحة، المهام المدعومة، وأوضاع التشغيل

يُعرض في هذا الجدول النماذج المتاحة مع وزنها المدرب مسبقًا، والمهام التي تدعمها، وتوافقها مع أوضاع التشغيل المختلفة مثل [الاستدلال](../modes/predict.md)، [التحقق](../modes/val.md)، [التدريب](../modes/train.md)، و [التصدير](../modes/export.md)، حيث يُشير إيموجي ✅ للأوضاع المدعومة وإيموجي ❌ للأوضاع غير المدعومة.

| نوع النموذج | الأوزان المدربة مسبقًا | المهام المدعومة                      | الاستدلال | التحقق | التدريب | التصدير |
|-------------|------------------------|--------------------------------------|-----------|--------|---------|---------|
| MobileSAM   | `mobile_sam.pt`        | [تجزئة العناصر](../tasks/segment.md) | ✅         | ❌      | ❌       | ✅       |

## التحويل من SAM إلى MobileSAM

نظرًا لأن MobileSAM يحتفظ بنفس سير العمل لـ SAM الأصلي، قمنا بدمج التجهيزات المسبقة والتجهيزات اللاحقة للنموذج الأصلي وجميع الواجهات الأخرى. نتيجة لذلك، يمكن لأولئك الذين يستخدمون حاليًا SAM الأصلي الانتقال إلى MobileSAM بقدر أدنى من الجهد.

يؤدي MobileSAM بشكل مقارب لـ SAM الأصلي ويحتفظ بنفس سير العمل باستثناء تغيير في مُشفر الصورة. على وحدة المعالجة الرسومية (GPU) الواحدة، يعمل MobileSAM بمعدل 12 مللي ثانية لكل صورة: 8 مللي ثانية لمُشفر الصورة و4 مللي ثانية لفك تشفير القناع.

يوفر الجدول التالي مقارنة بين مُشفرات الصور القائمة على ViT:

| مُشفّر الصورة | SAM الأصلي     | MobileSAM    |
|---------------|----------------|--------------|
| العوامل       | 611 مليون      | 5 مليون      |
| السرعة        | 452 مللي ثانية | 8 مللي ثانية |

يستخدم SَM الأصلي و MobileSAM نفس فك تشفير القناع الذي يعتمد على التوجيه بواسطة الرموز:

| فك تشفير القناع | SAM الأصلي   | MobileSAM    |
|-----------------|--------------|--------------|
| العوامل         | 3.876 مليون  | 3.876 مليون  |
| السرعة          | 4 مللي ثانية | 4 مللي ثانية |

فيما يلي مقارنة لكامل سير العمل:

| السير الكامل (التشفير+الفك) | SAM الأصلي     | MobileSAM     |
|-----------------------------|----------------|---------------|
| العوامل                     | 615 مليون      | 9.66 مليون    |
| السرعة                      | 456 مللي ثانية | 12 مللي ثانية |

يتم عرض أداء MobileSAM و SAM الأصلي باستخدام كل من النقطة ومربع كلمة المحفز.

![صورة بالنقطة ككلمة محفز](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![صورة بالمربع ككلمة محفز](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

بفضل أدائه المتفوق، يكون MobileSAM أصغر بحوالي 5 أضعاف وأسرع بحوالي 7 أضعاف من FastSAM الحالي. يتوفر مزيد من التفاصيل على [صفحة مشروع MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

## اختبار MobileSAM في Ultralytics

مثل SAM الأصلي، نقدم طريقة اختبار مبسّطة في Ultralytics، بما في ذلك وضعي النقطة والصندوق.

### تنزيل النموذج

يمكنك تنزيل النموذج [هنا](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt).

### النقطة ككلمة محفز

!!! Example "مثال"

    === "Python"
        ```python
        from ultralytics import SAM

        # تحميل النموذج
        model = SAM('mobile_sam.pt')

        # توقع جزء بناءً على نقطة محفز
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### الصندوق ككلمة محفز

!!! Example "مثال"

    === "Python"
        ```python
        from ultralytics import SAM

        # تحميل النموذج
        model = SAM('mobile_sam.pt')

        # توقع جزء بناءً على صندوق محفز
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

لقد قمنا بتنفيذ "MobileSAM" و "SAM" باستخدام نفس API. لمزيد من معلومات الاستخدام، يُرجى الاطّلاع على [صفحة SAM](sam.md).

## الاقتباس والشكر

إذا وجدت MobileSAM مفيدًا في أبحاثك أو عملك التطويري، يُرجى النظر في استشهاد ورقتنا:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
