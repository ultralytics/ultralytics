---
comments: true
description: योलोवी5यू की खोज करें, योलोवी5 मॉडल का एक बढ़ाया हुआ संस्करण जिसमें एक निश्चित रफ़्तार के बदलाव और विभिन्न वस्तु ज्ञापन कार्यों के लिए कई पूर्व प्रशिक्षित मॉडल शामिल हैं।
keywords: YOLOv5u, वस्तु ज्ञापन, पूर्व प्रशिक्षित मॉडल, Ultralytics, Inference, Validation, YOLOv5, YOLOv8, एंचर-मुक्त, वस्तुनिपाति रहित, वास्तविक समय अनुप्रयोग, मशीन लर्निंग
---

# योलोवी5

## समीक्षा

YOLOv5u वस्तु ज्ञापन के तरीकों में एक पटल बढ़ोतरी को प्रतिष्ठानित करता है। योग्यता ग्रहण और समय की मूल्य-माप बदलती शैली के आधार पर आधारित योलोवी5 मॉडल की स्थापना से परिचय में सुधार लाती है। तात्कालिक परिणामों और इसकी प्राप्त विशेषताओं के मद्देनजर, YOLOv5u एक ऐसा कुशल स्थानांतरण प्रदान करता है जो नवीन रंगेंगर में शोध और व्यावसायिक अनुप्रयोगों में सठिक समाधानों की तलाश कर रहे हैं।

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## मुख्य विशेषताएं

- **एंचर-मुक्त हिस्सा उल्ट्रालिटिक्स हेड:** पारंपरिक वस्तु ज्ञापन मॉडल निश्चित प्रमुख बॉक्सों पर आधारित होते हैं। हालांकि, YOLOv5u इस दृष्टिकोण को आधुनिक बनाता है। एक एंचर-मुक्त हिस्सा उल्ट्रालिटिक्स हेड की अपनाने से यह सुनिश्चित करता है कि एक और उचित और अनुरूप ज्ञापन मेकेनिज़म निर्धारित करें, जिससे विभिन्न परिदृश्यों में प्रदर्शन में सुधार होता है।

- **में सुधार गया गुणांक गति वस्तु:** गति और सुधार का anomaly रहता हैं। लेकिन YOLOv5u इस विरोधाभासी को चुनौती देता है। इस रंगेंगर व पुष्टि दृढ़ कर सुनिश्चित करता है वास्तविक समयगत ज्ञापन में स्थैतिकता नुकसान के बिना। यह विशेषता वाहन स्वतंत्र, रोबोटिक्स, और वास्तविक समयगत वीडियो विश्लेषण जैसे तत्वों को चाहती अनुप्रयोगों के लिए विशिष्ट सबक की अनमोलता होती है।

- **प्रशिक्षित मॉडल के विभिन्न वस्तुधापर्यावथाएं:** यह समझने कि लिए कि विभिन्न कार्यों के लिए विभिन्न उपकरण की जरूरत होती है, YOLOv5u एक कई पूर्व प्रशिक्षित मॉडल प्रदान करता है। चाहे आप ज्ञापन, मान्यता, या प्रशिक्षण पर ध्यान केंद्रित कर रहे हैं, आपकी अद्वितीय चुनौती के लिए एक टेलरमेड मॉडल है। यह विविधता यह सुनिश्चित करती है कि आप एक वन-साइज-फिट ऑल समाधान ही नहीं उपयोग कर रहे हैं, बल्कि अपनी अद्यापित अद्वितीय चुनौती के लिए एक मॉडल का उपयोग कर रहे हैं।

## समर्थित कार्य तथा मोड

योलोवी5u मॉडल, विभिन्न पूर्व प्रशिक्षित वेट वाली, [वस्तु ज्ञापन](../tasks/detect.md) कार्यों में उत्कृष्ट हैं। इन्हें विभिन्न ऑपरेशन मोड्स का समर्थन है, इसलिए इन्हें विकास से लेकर अंतर्गत उन्नतिशील अनुप्रयोगों के लिए उपयुक्त ठहराया जा सकता है।

| मॉडल प्रकार | पूर्व प्रशिक्षित वेट                                                                                                        | कार्य                              | ज्ञापन | मान्यता | प्रशिक्षण | निर्यात |
|-------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------|--------|---------|-----------|---------|
| YOLOv5u     | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [वस्तु ज्ञापन](../tasks/detect.md) | ✅      | ✅       | ✅         | ✅       |

यह तालिका योलोवी5u मॉडल के विभिन्न जैविक वेशभूषा प्रस्तुत करती है, इनके वस्तु ज्ञापन कार्यों में लागूहोने और [ज्ञापन](../modes/predict.md), [मान्यता](../modes/val.md), [प्रशिक्षण](../modes/train.md), और [निर्यात](../modes/export.md) की समर्थनता को उज्ज्वल बनाती है। इस समर्थन की पूर्णता सुनिश्चित करती है कि उपयोगकर्ता योलोवी5u मॉडल्स की संपूर्ण क्षमताओं का खास लाभ उठा सकते हैं विभिन्न ऑब्जेक्ट ज्ञापन स्थितियों में।

## प्रदर्शन पैमाने

!!! Performance

    === "ज्ञापन"

    [देखें ज्ञापन डॉकस](https://docs.ultralytics.com/tasks/detect/) को [COCO](https://docs.ultralytics.com/datasets/detect/coco/) पर प्रशिक्षित इन मॉडल्स के उपयोग के साथ उपयोग उदाहरण जैसे विविध पूर्व-प्रशिक्षित वर्गों को शामिल करता है।

    | मॉडल                                                                                       | YAML                                                                                                           | साइज़<br><sup>(पिक्सेल) | mAP<sup>वैल<br>50-95 | गति<br><sup>CPU ONNX<br>(मि.से.) | गति<br><sup>A100 TensorRT<br>(मि.से.) | params<sup><br>(M) | FLOPs<sup><br>(B) |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|---------------------------|------------------------|--------------------------------|-----------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
    |                                                                                             |                                                                                                                |                       |                      |                                |                                     |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                 | 211.0                          | 1.83                                | 4.3                | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                 | 422.6                          | 2.34                                | 15.3               | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                 | 810.9                          | 4.36                                | 41.2               | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                 | 1470.9                         | 5.47                                | 86.1               | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                 | 2436.5                         | 8.98                                | 155.4              | 250.7             |

## उपयोग उदाहरण

इस उदाहरण में सरल YOLOv5 चालन और ज्ञापन उदाहरण प्रदान किए गए हैं। इन और अन्य [modes](../modes/index.md) के लिए पूर्ण संदर्भ सामग्री के लिए दस्तावेज़ीकरण पृष्ठों में जाएं।

!!! Example "उदाहरण"

    === "पायथन"

        पायथन में एक मॉडल उदाहरण के लिए योलोवी5 आईएमजेड हालत में `*.pt` मॉडल्स के साथ मॉडल निर्माण के लिए `YOLO()` श्रेणी को पारित किया जा सकता है:

        ```python
        from ultralytics import YOLO

        # COCO-pretrained YOLOv5n मॉडल लोड करें
        model = YOLO('yolov5n.pt')

        # मॉडल जानकारी प्रदर्शित करें (वैकल्पिक)
        model.info()

        # COCO8 प्रायोगिक उदाहरण डेटासेट पर 100 एपॉक के लिए मॉडल
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # YOLOv5n मॉडल के साथ 'bus.jpg' छविमें ज्ञापन चलाएं
        results = model('path/to/bus.jpg')
        ```

    === "सी.एल.आई."

        मालिशी आदेशों का उपयोग सीधे मॉडलों को चलाने के लिए उपलब्ध हैं:

        ```bash
        # COCO-प्रशिक्षित YOLOv5n मॉडल खोलें और 100 एपॉक के लिए इसे COCO8 प्रायोगिक उदाहरण डेटासेट पर प्रशिक्षित करें
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # COCO-प्रशिक्षित YOLOv5n मॉडल खोलें और 'bus.jpg' छवि में ज्ञापन चलाएं
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## उद्धरण और मान्यता

यदि आप अपने शोध में YOLOv5 या YOLOv5u का उपयोग करते हैं, तो कृपया Ultralytics YOLOv5 दस्तावेज़ीकरण में मुख्य रूप से उल्लेख करें:

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

कृपया ध्यान दें कि YOLOv5 मॉडलें [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) और [एंटरप्राइज](https://ultralytics.com/license) लाइसेंस में उपलब्ध हैं।
