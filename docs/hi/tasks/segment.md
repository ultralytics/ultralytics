---
comments: true
description: Ultralytics YOLO के साथ उदाहरण देखें कि कैसे instance segmentation मॉडल का उपयोग करें। प्रशिक्षण, मान्यता, छवि की भविष्यवाणी और मॉडल निर्यात पर निर्देश।
keywords: yolov8, instance segmentation, Ultralytics, COCO dataset, image segmentation, object detection, model training, model validation, image prediction, model export
---

# Instance Segmentation

इंस्टेंस सेगमेंटेशन ऑब्जेक्ट डिटेक्शन से एक कदम आगे जाता है और छवि में व्यक्ति ऑब्जेक्ट की पहचान करता है और उन्हें छवि के बाकी हिस्से से विभाजित करता है।

इंस्टेंस सेगमेंटेशन मॉडल का आउटपुट एक सेट मास्क या कंटोर होता है जो छवि में प्रत्येक ऑब्जेक्ट का संकेत देता है, साथ ही प्रत्येक ऑब्जेक्ट के लिए वर्ग लेबल और आत्मविश्वास स्कोर होता है। इंस्टेंस सेगमेंटेशन उपयोगी होता है जब आपको न केवल पता चलेगा कि छवि में ऑब्जेक्ट कहाँ हैं, बल्कि वास्तव में उनका वास्तविक आकार क्या है।

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>देखें:</strong> पायथन में पूर्व-प्रशिक्षित Ultralytics YOLOv8 मॉडल के साथ Segmentation चलाएं।
</p>

!!! Tip "टिप"

    YOLOv8 Segment मॉडल `yolov8n-seg.pt` का उपयोग करते हैं, और इसे [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) पर पूरी प्रशिक्षित किया जाता है।

## [मॉडल](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 पूर्व प्रशिक्षित Segment मॉडल यहां दिखाए गए हैं। Detect, Segment और Pose मॉडल [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) डेटासेट पर पूर्व प्रशिक्षित हैं, जबकि Classify मॉडल [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) डेटासेट पर पूर्व प्रशिक्षित हैं।

[मॉडल](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) को उपयोग करके Ultralytics [रिलीज़](https://github.com/ultralytics/assets/releases) से पूर्ण डाउनलोड होते हैंं।

| मॉडल                                                                                         | आकार<br><sup>(पिक्सेल) | mAP<sup>बॉक्स<br>50-95 | mAP<sup>मास्क<br>50-95 | स्पीड<br><sup>CPU ONNX<br>(मि.सेकंड) | स्पीड<br><sup>A100 TensorRT<br>(मि.सेकंड) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|------------------------|------------------------|------------------------|--------------------------------------|-------------------------------------------|---------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                    | 36.7                   | 30.5                   | 96.1                                 | 1.21                                      | 3.4                 | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                    | 44.6                   | 36.8                   | 155.7                                | 1.47                                      | 11.8                | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                    | 49.9                   | 40.8                   | 317.0                                | 2.18                                      | 27.3                | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                    | 52.3                   | 42.6                   | 572.4                                | 2.79                                      | 46.0                | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                    | 53.4                   | 43.4                   | 712.1                                | 4.02                                      | 71.8                | 344.1             |

- **mAP<sup>val</sup>** मान एकल मॉडल एकल स्केल के लिए [COCO val2017](http://cocodataset.org) डेटासेट पर होते हैं।
  <br>`yolo val segment data=coco.yaml device=0` के द्वारा पुनर्जीवित किए जाएं।
- **स्पीड** एक [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) इंस्टेंस का उपयोग करते हुए COCO val छवियों के बीच औसतन।
  <br>`yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu` के द्वारा पुनर्जीवित किए जा सकते हैं।

## प्रशिक्षण

100 एपॉक्स पर 640 छवि के आकार के COCO128-seg डेटासेट पर YOLOv8n-seg को प्रशिक्षित करें। उपलब्ध तार्किक तर्क की पूरी सूची के लिए [Configuration](/../usage/cfg.md) पृष्ठ देखें।

!!! Example "उदाहरण"

    === "पायथन"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n-seg.yaml')  # YAML से नया मॉडल बनाएं
        model = YOLO('yolov8n-seg.pt')  # पूर्व-प्रशिक्षित मॉडल लोड करें (प्रशिक्षण के लिए सिफारिश की जाती है)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # YAML से नए मॉडल बनाएं और धारित करें

        # मॉडल प्रशिक्षित करें
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAML से नया मॉडल बनाएं और शून्य से प्रशिक्षण शुरू करें
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # पूर्व-प्रशिक्षित *.pt मॉडल से प्रशिक्षण शुरू करें
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # YAML से नया मॉडल बनाएं, पूर्व-प्रशिक्षित वजनों को इसे ट्रांसफर करें और प्रशिक्षण शुरू करें
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### डेटासेट प्रारूप

YOLO सेगमेंटेशन डेटासेट प्रारूप [डेटासेट गाइड](../../../datasets/segment/index.md) में विस्तार से देखा जा सकता है। कृपया अपने मौजूदा डेटासेट को अन्य प्रारूपों (जैसे कि COCO आदि) से YOLO प्रारूप में परिवर्तित करने के लिए [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) उपकरण का उपयोग करें।

## मान्यता

COCO128-seg डेटासेट पर प्रशिक्षित YOLOv8n-seg मॉडल की सत्यापन करें। `मॉडल` पास करने के लिए कोई तर्क आवश्यक नहीं होता है क्योंकि `मॉडल`
प्रशिक्षण के `डेटा` और तर्कों का ध्यान रखता है।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n-seg.pt')  # आधिकारिक मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # कस्टम मॉडल लोड करें

        # मॉडल की सत्यापना करें
        metrics = model.val()  # कोई तर्क आवश्यक नहीं है, डेटा और सेटिंग्स याद रखे जाते हैं
        metrics.box.map    # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps   # एक सूची है जिसमें प्रत्येक श्रेणी का map50-95(B) होता है
        metrics.seg.map    # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps   # एक सूची है जिसमें प्रत्येक श्रेणी का map50-95(M) होता है
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # आधिकारिक मॉडल की मान्यता
        yolo segment val model=path/to/best.pt  # कस्टम मॉडल की मान्यता
        ```

## भविष्यवाणी

प्रशिक्षित YOLOv8n-seg मॉडल का उपयोग छवियों पर भविष्यवाणी करने के लिए करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n-seg.pt')  # आधिकारिक मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # कस्टम मॉडल लोड करें

        # मॉडल के साथ भविष्यवाणी करें
        results = model('https://ultralytics.com/images/bus.jpg')  # एक छवि पर भविष्यवाणी करें
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # आधिकारिक मॉडल के साथ भविष्यवाणी करें
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # कस्टम मॉडल के साथ भविष्यवाणी करें
        ```

भविष्यवाणी मोड के पूर्ण विवरण को [Predict](https://docs.ultralytics.com/modes/predict/) पृष्ठ में देखें।

## निर्यात

YOLOv8n-seg मॉडल को ONNX, CoreML आदि जैसे अन्य प्रारूप में निर्यात करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n-seg.pt')  # आधिकारिक मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # कस्टम प्रशिक्षित मॉडल लोड करें

        # मॉडल निर्यात करें
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # आधिकारिक मॉडल को निर्यात करें
        yolo export model=path/to/best.pt format=onnx  # कस्टम प्रशिक्षित मॉडल को निर्यात करें
        ```

YOLOv8-seg निर्यात प्रारूप निम्नलिखित तालिका में बताए गए हैं। आप निर्यात किए गए मॉडल पर सीधे भविष्यवाणी या मान्यता कर सकते हैं, अर्थात `yolo predict model=yolov8n-seg.onnx`। निर्यात होने के बाद अपने मॉडल के लिए उपयोग के उदाहरण देखें।

| प्रारूप                                                            | `format` Argument | मॉडल                          | मेटाडेटा | तर्क                                                |
|--------------------------------------------------------------------|-------------------|-------------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-seg.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-seg.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-seg.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-seg_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-seg.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-seg.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-seg_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-seg.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-seg.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-seg_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-seg_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-seg_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-seg_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

[Export](https://docs.ultralytics.com/modes/export/) पृष्ठ में पूर्ण `निर्यात` विवरण देखें।
