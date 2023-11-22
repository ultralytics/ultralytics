---
comments: true
description: YOLOv8 की रोमांचक विशेषताओं का अन्वेषण करें, हमारे वास्तविक समय वस्तु निर्धारक के नवीनतम संस्करण! एडवांस्ड आर्किटेक्चर, पूर्व-प्रशिक्षित मॉडल और सटीकता और गति के बीच आपूर्ति का सही संतुलन स्थापित के साथ, YOLOv8 आपके वस्तु पहचान कार्यों के लिए सही उपाय है।
keywords: YOLOv8, Ultralytics, वास्तविक समय वस्तु निर्धारक, पूर्व-प्रशिक्षित मॉडल, प्रलेखन, वस्तुविज्ञान निर्देशिका, YOLO श्रृंगार, एड्वांस्ड आर्किटेक्चर, सटीकता, गति
---

# YOLOv8

## सारांश

YOLOv8, वास्तविक समय में वस्तु निर्धारकों के YOLO श्रृंगार के नवीनतम प्रतिरूप है, जो सटीकता और गति के मामले में कटिपयि प्रदर्शन प्रदान करता है। पिछले YOLO संस्करणों के उन्नयन पर निर्मित YOLOv8 प्रस्तुत करता है नई विशेषताएं और अनुकूलनों की यह नई संस्करण, कई विभिन्न वस्तु निर्धारण कार्यों के लिए एक आदर्श विकल्प है, इन अनुप्रयोगों में चयन करें।

![यूल्ट्रालिक्स YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## मुख्य विशेषताएं

- **एडवांस्ड बैकबोन और गर्दन आर्किटेक्चर:** YOLOv8 परिष्कृत एडवांस्ड बैकबोन और गर्दन आर्किटेक्चर का उपयोग करता है, जिससे विशेषता निष्कर्षण और वस्तु निर्धारण प्रदर्शन में सुधार होता है।
- **एंकर-मुक्त विभाजन Ultralytics हेड:** YOLOv8 एंकर-आधारित दृष्टिकोण की तुलना में बेहतर सटीकता और अधिक दक्ष डिटेक्शन प्रक्रिया में योगदान देने वाली एंकर-मुक्त विभाजन Ultralytics हेड का अपनान करता है।
- **सटीकता-गति के मध्यस्थता का सुधार:** सटीकता और गति के बीच सुधारित खेल के लिए ध्यान केंद्रित करते हुए, YOLOv8 वास्तविक समय में वस्तु निर्धारक कार्यों के लिए उपयुक्त हैं विविध आवेदन क्षेत्रों में।
- **पूर्व-प्रशिक्षित मॉडल की विविधता:** YOLOv8 विभिन्न कार्य और प्रदर्शन आवश्यकताओं को पूरा करने के लिए विभिन्न पूर्व-प्रशिक्षित मॉडलों की श्रृंखला प्रदान करता है, जिससे आपके विशिष्ट उपयोग मामले के लिए सही मॉडल ढूंढ़ना आसान होता है।

## समर्थित कार्य और मोड

YOLOv8 श्रृंगार की श्रृंखला में विभिन्न मॉडल प्रदान करता है, प्रत्येक कंप्यूटर विज्ञान के विशिष्ट कार्यों के लिए विशेषज्ञीकृत हैं। ये मॉडल विभिन्न आवश्यकताओं को पूरा करने के लिए डिजाइन किये गए हैं, वस्तु निर्धारण से लेकर ज्यादा जटिल कार्यों जैसे आइंस्टेंस सेगमेंटेशन, पोज/कुण्डलिनी का पता लगाना और वर्गीकरण जैसे मुश्किल कार्यों तक।

YOLOv8 श्रृंगार की प्रत्येक प्रकारलय, अपनी संबंधित कार्य के लिए अनुकूल बनाने के लिए अनुकूलित होती है, इनका उपयोग विभिन्न विकास और लागू करने के अवसरों में किया जाता है।

| मॉडल        | फ़ाइलनेम                                                                                                       | कार्य                                       | वस्तुनिश्चय | मान्यता | प्रशिक्षण | निर्यात |
|-------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------|-------------|---------|-----------|---------|
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [पहचान](../tasks/detect.md)                 | ✅           | ✅       | ✅         | ✅       |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [आइंस्टेंस सेगमेंटेशन](../tasks/segment.md) | ✅           | ✅       | ✅         | ✅       |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [पोज/कुण्डलिनी](../tasks/pose.md)           | ✅           | ✅       | ✅         | ✅       |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [वर्गीकरण](../tasks/classify.md)            | ✅           | ✅       | ✅         | ✅       |

यह टेबल मॉडलों के खंडन, उनके विशेष कार्य में लागू होने की अवधारणा प्रदान करता है और प्रशिक्षण, वैधानिकता, प्रशिक्षण, और निर्यात की समर्थन में उनके संगतता हाइलाइट करता है। यह YOLOv8 श्रृंगार की विविधता और दमदारता को प्रदर्शित करती है, जिससे कंप्यूटर विज्ञान में विभिन्न अनुप्रयोगों के लिए उपयुक्त हैं।

## प्रदर्शन माप

!!! प्रदर्शन

    === "पहचान (COCO)"

        [COCO](https://docs.ultralytics.com/datasets/detect/coco/) पर प्रशिक्षित इन मॉडलों के उपयोग के उदाहरण देखने के लिए, [Detection डॉक्स](https://docs.ultralytics.com/tasks/detect/) देखें, जिनमें 80 पूर्व-प्रशिक्षित वर्ग शामिल हैं।

        | मॉडल                                                                                | size<br><sup>(पिक्सेल) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(मिलीसेकंड) | Speed<br><sup>A100 TensorRT<br>(मिलीसेकंड) | params<br><sup>(M) | FLOPs<br><sup>(बिलियन) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "पहचान (Open Images V7)"

        [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/) पर प्रशिक्षित इन मॉडलों के उपयोग के उदाहरण देखने के लिए, [Detection डॉक्स](https://docs.ultralytics.com/tasks/detect/) देखें, जिनमें 600 पूर्व-प्रशिक्षित वर्ग शामिल हैं।

        | मॉडल                                                                                     | size<br><sup>(पिक्सेल) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(मिलीसेकंड) | Speed<br><sup>A100 TensorRT<br>(मिलीसेकंड) | params<br><sup>(M) | FLOPs<br><sup>(बिलियन) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "सेगमेंटेशन (COCO)"

        [COCO](https://docs.ultralytics.com/datasets/segment/coco/) पर प्रशिक्षित इन मॉडलों के उपयोग के उदाहरण देखने के लिए, [Segmentation डॉक्स](https://docs.ultralytics.com/tasks/segment/) देखें, जिनमें 80 पूर्व-प्रशिक्षित वर्ग शामिल हैं।

        | मॉडल                                                                                        | size<br><sup>(पिक्सेल) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(मिलीसेकंड) | Speed<br><sup>A100 TensorRT<br>(मिलीसेकंड) | params<br><sup>(M) | FLOPs<br><sup>(बिलियन) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "वर्गीकरण (ImageNet)"

        [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) पर प्रशिक्षित इन मॉडलों के उपयोग के उदाहरण देखने के लिए, [Classification डॉक्स](https://docs.ultralytics.com/tasks/classify/) देखें, जिनमें 1000 पूर्व-प्रशिक्षित वर्ग शामिल हैं।

        | मॉडल                                                                                        | size<br><sup>(पिक्सेल) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(मिलीसेकंड) | Speed<br><sup>A100 TensorRT<br>(मिलीसेकंड) | params<br><sup>(M) | FLOPs<br><sup>(बिलियन)<br>640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "पोज (COCO)"

        'Person' एक पूर्व-प्रशिक्षित वर्ग के साथ कोई मॉडल्स के साथ इन मॉडलों का उपयोग करने के लिए, [Pose Estimation डॉक्स](https://docs.ultralytics.com/tasks/segment/) देखें।

        | मॉडल                                                                                                | size<br><sup>(पिक्सेल) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(मिलीसेकंड) | Speed<br><sup>A100 TensorRT<br>(मिलीसेकंड) | params<br><sup>(M) | FLOPs<br><sup>(बिलियन) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## उपयोग के उदाहरण

यह उदाहरण साधारित YOLOv8 प्रशिक्षण और भविष्यवाणी उदाहरण प्रदान करता है। इन और अन्य [मोड](../modes/index.md) पर पूर्ण दस्तावेज़ीकरण के लिए, [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) और [Export](../modes/export.md) दस्तावेज़ पृष्ठों को देखें।

ध्यान दें कि नीचे दिए गए उदाहरण योलोवी8 [डिटेक्ट](../tasks/detect.md) मॉडलों के लिए हैं, अतिरिक्त समर्थित कार्यों के लिए [विभाजन](../tasks/segment.md), [वर्गीकरण](../tasks/classify.md) और [पोज](../tasks/pose.md) डॉक्स देखें।

!!! उदाहरण

    === "Python"

        पायटोर्च प्री-प्रशिक्षित `*.pt` मॉडल्स और कॉन्फ़िगरेशन `*.yaml` फ़ाइलें योग्यताएँ देने के लिए `YOLO()` वर्ग को भेजा जा सकता है जिससे पाइथन में एक मॉडल उदाह्रण बन सकता है:

        ```python
        from ultralytics import YOLO

        # एक कोको-प्रीट्रेन किया हुआ YOLOv8n मॉडल लोड करें
        model = YOLO('yolov8n.pt')

        # मॉडल जानकारी प्रदर्शित करें (वैकल्पिक)
        model.info()

        # 100 epochs के लिए COCO8 उदाहरण डेटासेट पर मॉडल को प्रशिक्षित करें
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 'बस.jpg' छवि पर YOLOv8n मॉडल के साथ भविष्यवाणी चलाएं
        results = model('रास्था/से/बस.jpg')
        ```

    === "CLI"

        सीएलआई कमांड प्रत्यक्षरूप से मॉडलों को सीधे चलाने के लिए उपलब्ध हैं:

        ```bash
        # एक कोको-प्रीट्रेन किया हुआ YOLOv8n मॉडल लोड करें और इसे 100 epochs के लिए COCO8 उदाहरण डेटासेट पर प्रशिक्षित करें
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # एक कोको-प्रीट्रेन किया हुआ YOLOv8n मॉडल लोड करें और 'बस.jpg' छवि पर भविष्यवाणी चलाएं
        yolo predict model=yolov8n.pt source=रास्था/से/बस.jpg
        ```

## संदर्भ और पुरस्कार

यदि आप अपने काम में YOLOv8 मॉडल या इस थक्क के अन्य सॉफ़्टवेयर का उपयोग करते हैं, तो कृपया इसे निम्न प्रारूप का उद्धरण देकर उद्धरण करें:

!!! पुरस्कार ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {ग्लेन जोचर and आयुष चौरसिया and जिंग क्यू},
          title = {यूल्ट्रालिक्स YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

ख्याल रखें कि DOI की प्रतीक्षा हो रही है और जब यह उपलब्ध हो जाएगा, तो उद्धरण में उसे जोड़ा जाएगा। YOLOv8 मॉडलों की प्रदान की जाती है [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) और [एंटरप्राइज](https://ultralytics.com/license) लाइसेंस के तहत।
