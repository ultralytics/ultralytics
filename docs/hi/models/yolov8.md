---
comments: true
description: YOLOv8 की रोमांचक विशेषताओं का अन्वेषण करें, हमारे वास्तविक समय वस्तु निर्धारक के नवीनतम संस्करण। देखें कैसे प्रगतिशील शृंखलाओं, पूर्व-प्रशिक्षित मॉडलों और सटीकता और गति के बीच सही संतुलन को YOLOv8 के विकल्प में सटे करते हैं संज्ञानघन वस्तुनिर्धारण कार्यों के लिए YOLOv8 को आपके वस्तु आरोप के लिए सही चुनाव बनाता है।
keywords: YOLOv8, Ultralytics, वास्तविक समय वस्तुनिर्धारक, पूर्व-प्रशिक्षित मॉडल, दस्तावेज़ीकरण, वस्तुवाहीनिर्धारण, YOLO श्रृंखला, प्रगतिशील शृंखलाएं, सटीकता, गति
---

# YOLOv8

## अवलोकन

YOLOv8 योलो श्रृंखला का नवीनतम संस्करण है, जो सटीकता और गति के मामले में कटिंग-एज प्रदान करता है। पिछले YOLO संस्करणों की प्रगति को अवधारणा करते हुए, YOLOv8 उन्नत सुविधाओं और अनुकूलन को प्रस्तुत करता है, जो इसे विभिन्न वस्तुनिर्धारण कार्यों के लिए एक आदर्श चुनाव बनाता है विभिन्न अनुप्रयोगों में।

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## मुख्य विशेषताएं

- **उन्नत पीठ और गर्दन शृंखलाएं:** YOLOv8 उन्नत पीठ और गर्दन शृंखलाएं प्रयोग करता है, जिससे विशेषता निष्कर्षण और वस्तु निर्धारण क्षमता की सुधार होती है।
- **एंकर-मुक्त स्प्लिट Ultralytics हैड:** YOLOv8 एंकर-आधारित दृष्टिकोणों की तुलना में अधिक सटीकता और एक अधिक संचालनयोग्य निर्धारण प्रक्रिया के लिए एक एंकर-मुक्त स्प्लिट Ultralytics हेड अपनाता है।
- **सुधारित सटीकता-गति का संतुलन:** सटीकता और गति के मध्य में उचित संतुलन बनाए रखने के ध्यान के साथ, YOLOv8 वास्तविक समय वस्तुनिर्धारण कार्यों के लिए उपयुक्त है जो विभिन्न अनुप्रयोग क्षेत्रों में हो सकते हैं।
- **विभिन्न पूर्व-प्रशिक्षित मॉडल:** YOLOv8 विभिन्न कार्यों और प्रदर्शन आवश्यकताओं के लिए एक विस्तृत पूर्व-प्रशिक्षित मॉडल रेंज प्रदान करता है, इससे अपने विशेषता उपयोग के लिए सही मॉडल खोजना आसान हो जाता है।

## समर्थित कार्य और मोड

YOLOv8 श्रृंखला वास्तविक समय वस्तुनिर्धारण के लिए विशेषकृत कई मॉडल प्रदान करती है। ये मॉडल विभिन्न आवश्यकताओं को पूरा करने के लिए डिजाइन किए गए हैं, वैश्विक स्तर पहुंचने से लेकर इंस्टेंस सेगमेंटेशन, पोज/किंतुमांक निर्धारण और श्रेणीकरण जैसे जटिल कार्यों तक।

Yएक मॉडल के हर मानक, विशिष्ट कार्यों में अपनी विशेषताओं को ध्यान में रखते हुए, उच्च प्रदर्शन और सटीकता सुनिश्चित किए जाते हैं। इसके अलावा, ये मॉडल विभिन्न संचालन मोड के साथ अनुकूलित हैं जैसे [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), और [Export](../modes/export.md), जो उनका उपयोग वितरण और विकास के विभिन्न स्तरों में सरल बनाने में मदद करता है।

| मॉडल        | फ़ाइलनेम                                                                                                       | कार्य                                      | Inference | Validation | Training | Export |
|-------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------|-----------|------------|----------|--------|
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [वस्तुनिर्धारण](../tasks/detect.md)        | ✅         | ✅          | ✅        | ✅      |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [इंस्टेंस सेगमेंटेशन](../tasks/segment.md) | ✅         | ✅          | ✅        | ✅      |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [पोज/किंतुमांक](../tasks/pose.md)          | ✅         | ✅          | ✅        | ✅      |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [श्रेणीबद्दीकरण](../tasks/classify.md)     | ✅         | ✅          | ✅        | ✅      |

इस सारणी में YOLOv8 मॉडल विभिन्न कार्यों के लिए उपयुक्तता और विभिन्न संचालन मोड के साथ मॉडल के विभिन्न रूपों का अवलोकन प्रदान करती है। यह YOLOv8 श्रृंखला की व्याप्ति और मजबूती का प्रदर्शन करती है, जो कंप्यूटर दृष्टि में विभिन्न अनुप्रयोगों के लिए उपयुक्त बनाती है।

## प्रदर्शन की मापदंड

!!! Note "प्रदर्शन"

    === "वस्तुनिर्धारण (COCO)"

        [वस्तुनिर्धारण दस्तावेज़ीकरण](https://docs.ultralytics.com/tasks/detect/) पर उपयोग उदाहरण देखें जहां COCO ट्रेन किए गए [80 पूर्व-प्रशिक्षित वर्गों](https://docs.ultralytics.com/datasets/detect/coco/) के साथ ये मॉडल दिए गए हैं।

        | मॉडल                                                                                | आकार<br><sup>(पिक्स) | mAP<sup>वैल<br>50-95 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(एम) | FLOPs<br><sup>(बी) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "वस्तुनिर्धारण (Open Images V7)"

        [वस्तुनिर्धारण दस्तावेज़ीकरण](https://docs.ultralytics.com/tasks/detect/) पर उपयोग उदाहरण देखें जहां इन मॉडलों को [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) पर ट्रेन किया गया है, जिसमें 600 पूर्व-प्रशिक्षित वर्ग हैं।

        | मॉडल                                                                                     | आकार<br><sup>(पिक्स) | mAP<sup>वैल<br>50-95 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(एम) | FLOPs<br><sup>(बी) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "सेगमेंटेशन (COCO)"

        [सेगमेंटेशन दस्तावेज़ीकरण](https://docs.ultralytics.com/tasks/segment/) पर उपयोग उदाहरण देखें जहां इन मॉडलों को [COCO](https://docs.ultralytics.com/datasets/segment/coco/) पर ट्रेन किया गया है, जिसमें 80 पूर्व-प्रशिक्षित वर्ग हैं।

        | मॉडल                                                                                        | आकार<br><sup>(पिक्स) | mAP<sup>बॉक्स<br>50-95 | mAP<sup>मास्क<br>50-95 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(एम) | FLOPs<br><sup>(बी) |
        | -------------------------------------------------------------------------------------------- | --------------------- | --------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                  | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                  | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                  | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                  | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                  | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "श्रेणीकरण (ImageNet)"

        [श्रेणीकरण दस्तावेज़ीकरण](https://docs.ultralytics.com/tasks/classify/) पर उपयोग उदाहरण देखें जहां इन मॉडलों को [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) पर ट्रेन किया गया है, जिसमें 1000 पूर्व-प्रशिक्षित वर्ग हैं।

        | मॉडल                                                                                        | आकार<br><sup>(पिक्स) | शीर्ष1 विजयी<br>योग्यता | शीर्ष5 विजयी<br>योग्यता | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(एम) | FLOPs<br><sup>(बी) at 640 |
        | ------------------------------------------------------------------------------------------ | --------------------- | ------------------------ | ------------------------ | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6                     | 87.0                     | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3                     | 91.1                     | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4                     | 93.2                     | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0                     | 94.1                     | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4                     | 94.3                     | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "पोज (COCO)"

        [पोज निर्धारण दस्तावेज़ीकरण](https://docs.ultralytics.com/tasks/pose/) पर उपयोग उदाहरण देखें जहां इन मॉडलों को [COCO](https://docs.ultralytics.com/datasets/pose/coco/) पर ट्रेन किया गया है, जिसमें 1 पूर्व-प्रशिक्षित वर्ग, 'person' शामिल है।

        | मॉडल                                                                                                 | आकार<br><sup>(पिक्स) | mAP<sup>शामिती<br>50-95 | mAP<sup>शामिती<br>50 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(एम) | FLOPs<br><sup>(बी) |
        | ----------------------------------------------------------------------------------------------------- | --------------------- | ------------------------ | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                     | 80.1                 | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                     | 86.2                 | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                     | 88.8                 | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                     | 90.0                 | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                     | 90.2                 | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                     | 91.2                 | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## उपयोग की उदाहरण

यह उदाहरण सरल YOLOv8 प्रशिक्षण और निर्धारण उदाहरण प्रदान करता है। इन और अन्य [मोड](../modes/index.md) की पूरी दस्तावेज़ीकरण के लिए दस्तावेज़ पृष्ठों [Predict](../modes/predict.md),  [Train](../modes/train.md), [Val](../modes/val.md) और [Export](../modes/export.md) का उपयोग करें।

इसे ध्यान दें कि नीचे दिए गए उदाहरण योलोवी [वस्तुनिर्धारण](../tasks/detect.md) मॉडल के लिए हैं। अतिरिक्त समर्थित कार्यों के लिए [Segment](../tasks/segment.md), [Classify](../tasks/classify.md) और [Pose](../tasks/pose.md) दस्तावेज़ीकरण देखें।

!!! Example "उदाहरण"

    === "पायथन"

        पायटोर्च का पूर्व-प्रशिक्षित `*.pt` मॉडल और विन्यास `*.yaml` फ़ाइल पायटन में एक मॉडल नमूना बनाने के लिए `YOLO()` कक्षा को पारित किया जा सकता है:

        ```python
        from ultralytics import YOLO

        # कोहली के COCO-pretrained YOLOv8n मॉडल को लोड करें
        model = YOLO('yolov8n.pt')

        # मॉडल जानकारी दिखाएँ (वैकल्पिक)
        model.info()

        # COCO8 उदाहरण डेटासेट पर 100 एपोक के लिए मॉडल को प्रशिक्षित करें
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' छवि पर YOLOv8n मॉडल के साथ निर्धारण चलाएँ
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI कमांड को सीधे चलाने के लिए उपलब्ध हैं:

        ```bash
        # COCO-pretrained YOLOv8n मॉडल को लोड करें और उसे COCO8 उदाहरण डेटासेट पर 100 एपोक के लिए प्रशिक्षित करें
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO-pretrained YOLOv8n मॉडल को लोड करें और 'bus.jpg' छवि पर निर्धारण चलाएँ
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## सन्दर्भ और पुरस्कार

यदि आप अपने काम में YOLOv8 मॉडल या इस रिपॉजिटरी के किसी अन्य सॉफ़्टवेयर का उपयोग करते हैं, तो कृपया इसकी उद्धरण इस प्रकार करें:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {ग्लेन जोचर and आयुष चौरसिया and जिंग क्यू},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

कृपया ध्यान दें कि DOI लंबित है और जब यह उपलब्ध हो जाएगा तो उद्धरण में इसे शामिल किया जाएगा। YOLOv8 मॉडल [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) और [एंटरप्राइज](https://ultralytics.com/license) लाइसेंस के तहत उपलब्ध हैं।
