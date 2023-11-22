---
comments: true
description: योलोवी8 की रोमांचक विशेषताओं का पता लगाएं, हमारे तत्वविवर्धन में प्रवीणता और गति का संतुष्ट संतुलन बनाने वाले उन्नत वास्तुकला,पूर्व-प्रशिक्षित मॉडेल और योग्यता और गति के बीच समाप्ति कोण योलोवी8 को आपके वस्तुसंग्रह कार्यों के लिए सही विचार बनाते हैं।
keywords: योलोवी8, Ultralytics, वास्तविक समय वस्तु ट्रेसर,पूर्व-प्रशिक्षित मॉडल,दस्तावेज़ीकरण,वस्तु ट्रेसर,योलो सीरीज, उन्नत वास्तुकला, योग्यता, गति
---

# योलोवी8

## सिंपर झलक

योलोवी8 संख्यात्मक विज्ञान वाली वास्तविक समय वस्तु ट्रेसरों के योलो धराओं का नवीनतम संस्करण है, जो योलो धराओं के पिछले संस्करणों के प्रगति पर आधारित विशेषताओं और अनुकरणों को लेकर नए सुविधाओं और समुदायों को परिचालित करते हुए, विभिन्न वस्तु ट्रेसर कार्यों के लिए एक आदर्श विकल्प बनाने में मदद करती है। इसके साथ-साथ, यह विभिन्न अनुप्रयोग क्षेत्रों में वस्तु ट्रेसर कार्यों के लिए उच्च प्रदर्शन करने वाली प्रगत वास्तु वास्तुकला और गति के बीच स्थानांतरण प्रदान करती है।

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## मुख्य विशेषताएं

- **उन्नत मूलभूत और गर्दन वास्तुकला:** योलोवी8 उन्नत मूलभूत और गर्दन वास्तुकला का उपयोग करता है, जिससे उन्नत सुविधा प्राप्ति और वस्तु ट्रेसर प्रदर्शन होता है।
- **एंकर-मुक्त विभाजन युल्‍त्रालिटिक्स सिर:** योलोवी8 एंकर-मुक्त विभाजन युल्‍ट्रालिटिक्स सिर का उपयोग करता है, जो एंकर-आधारित दृष्टिकोणों की तुलना में बेहतर सुविधाओं और अधिक दक्ष ट्रेसर प्रक्रिया के लिए योगदान देता है।
- **योग्यता-गति का समर्थन सुधार:** योलोवी8 में योग्यता और गति के बीच एक समाप्ति कोण बनाए रखने पर ध्यान केंद्रित करके, वह वास्तविक समय वस्तु ट्रेसर कार्यों के लिए उपयुक्त है।
- **पूर्व-प्रशिक्षित मॉडल का विकल्प:** योलोवी8 विभिन्न कार्यों और प्रदर्शन आवश्यकताओं के लिए कई पूर्व-प्रशिक्षित मॉडल प्रदान करता है, जिससे आपके विशिष्ट उपयोग मामले के लिए एक सही मॉडल ढूंढना आसान हो जाता है।

## समर्थित कार्य

| मॉडल प्रकार | प्रशिक्षित वज़न                                                                                                     | कार्य            |
|-------------|---------------------------------------------------------------------------------------------------------------------|------------------|
| योलोवी8     | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | पहचान            |
| योलोवी8-अंश | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | उदाहरण अवियोजकता |
| योलोवी8-पोज | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | स्थिति/कुंजिका   |
| योलोवी8-cls | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | वर्गीकरण         |

## समर्थित मोड

| मोड       | समर्थित |
|-----------|---------|
| हाइफरेंस  | ✅       |
| मान्यता   | ✅       |
| प्रशिक्षण | ✅       |

!!! प्रदर्शन

    === "पहचान (COCO)"

        | मॉडल                                                                                | आकार<br><sup>(पिक्सल) | mAP<sup>val<br>50-95 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [योलोवी8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [योलोवी8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [योलोवी8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [योलोवी8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [योलोवी8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "पहचान (Open Images V7)"

        उपयोग मिसालों के लिए [पहचान दस्तावेज़](https://docs.ultralytics.com/tasks/detect/) देखें।

        | मॉडल                                                                                     | आकार<br><sup>(पिक्सल) | mAP<sup>val<br>50-95 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [योलोवी8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [योलोवी8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [योलोवी8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [योलोवी8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [योलोवी8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "औचित्यवादी (COCO)"

        | मॉडल                                                                                        | आकार<br><sup>(पिक्सल) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [योलोवी8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [योलोवी8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [योलोवी8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [योलोवी8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [योलोवी8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "वर्गीकरण (ImageNet)"

        | मॉडल                                                                                        | आकार<br><sup>(पिक्सल) | acc<br><sup>top1 | acc<br><sup>top5 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [योलोवी8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [योलोवी8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [योलोवी8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [योलोवी8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [योलोवी8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "स्थिति (COCO)"

        | मॉडल                                                                                                | आकार<br><sup>(पिक्सल) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | गति<br><sup>CPU ONNX<br>(ms) | गति<br><sup>A100 TensorRT<br>(ms) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [योलोवी8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [योलोवी8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [योलोवी8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [योलोवी8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [योलोवी8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [योलोवी8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## उपयोग

आप Ultralytics पिप पैकेज का उपयोग करके वस्तु ट्रेसर कार्यों के लिए YOLOv8 का उपयोग कर सकते हैं। निम्नलिखित एक उदाहरण कोड स्निपट दिखाता है जिसमें दिखाया गया है कि YOLOv8 मॉडल को ईंधन के रूप में कैसे उपयोग किया जाए:

!!! उदाहरण ""

    इस उदाहरण में YOLOv8 के लिए सरल उपयोग कोड दिया गया है। अधिक विकल्पों, सहायता प्राप्त करने और अन्य मोड्स को संघर्ष करने के लिए [पहचान](../modes/predict.md) मोड देखें। अतिरिक्‍त मोड्स के साथ YOLOv8 का उपयोग करने के लिए [प्रशिक्षण](../modes/train.md), [सत्यापन](../modes/val.md) और [निर्यात](../modes/export.md) देखें।

    === "पायथन"

        पायटोर्च पूर्व-प्रशिक्षित `*.pt` मॉडल के साथ कॉन्‍फ़िगरेशन `*.yaml` फ़ाइल आप पास कर सकते हैं जिससे वैद्य मिशनाधिकारी की पायथन में संयोजना करने के लिए एक मॉडल उदाहरण बनाया जा सकता है:

        ```python
        from ultralytics import YOLO

        # COCO-pretrained YOLOv8n मॉडल लोड करें
        model = YOLO('yolov8n.pt')

        # मॉडल सूचना प्रदर्शित करें (वैकल्पिक)
        model.info()

        # COCO8 उदाहरण डेटासेट पर 100 एपोक में मॉडल को प्रशिक्षित करें
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'bus.jpg' छवि पर YOLOv8n मॉडल के साथ अनुमान पर चलाएं
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI कमांड उपलब्ध हैं जो सीधे मॉडलों को चलाने के लिए संघर्ष कर सकते हैं:

        ```bash
        # COCO-pretrained YOLOv8n मॉडल लोड करें और इसे COCO8 मिसाल फ़ाइल डेटासेट पर 100 एपोक ट्रेन करें
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO-pretrained YOLOv8n मॉडल लोड करें और 'bus.jpg' छवि पर अनुमान पर चलाएं
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## उद्धरण और सराहना

यदि आप अपने काम में योलोवी8 मॉडल या इस रिपोज़िटरी के किसी अन्य सॉफ़्टवेयर का उपयोग करते हैं, तो कृपया इसे निम्न रूप में उद्धरण देकर सिट करें:

!!! नोट ""

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

कृपया ध्यान दें कि DOI अपूर्ण है और जब वह उपलब्ध हो जाएगा तो हस्तांतरण को उद्धरण में जोड़ा जाएगा। सॉफ़्टवेयर का उपयोग AGPL-3.0 लाइसेंस के अनुसार होता है।
