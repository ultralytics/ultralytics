---
comments: true
description: Ultralytics YOLOv8 का उपयोग पोज निर्धारण कार्यों के लिए कैसे किया जाता है इसकी जानें। प्री-शिक्षित मॉडल ढूंढें, प्रशिक्षण, मान्यता प्राप्त करें, पूर्वानुमान लगाएं, और अपना खुद का निर्यात करें।
keywords: Ultralytics, YOLO, YOLOv8, pose estimation, keypoints detection, object detection, pre-trained models, machine learning, artificial intelligence
---

# पोज निर्धारण

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="पोज निर्धारण उदाहरण">

पोज निर्धारण एक कार्य है जिसमें एक छवि में विशेष बिंदुओं के स्थान की पहचान करना शामिल होता है, जिसे आमतौर पर कीपॉइंट्स के रूप में कहा जाता है। कीपॉइंट्स विभिन्न अंगों, भूमिकाओं या अन्य विशिष्ट सुविधाओं आदि के रूप में वस्तु के विभिन्न हिस्सों को प्रतिष्ठित कर सकते हैं। कीपॉइंट्स के स्थान आमतौर पर 2D `[x, y]` या 3D `[x, y, दिखाई देने वाला]` कोआर्डिनेट के सेट के रूप में प्रदर्शित होते हैं।

पोज निर्धारण मॉडल की उत्पादन एक छवि में वस्तु के कीपॉइंट्स को प्रतिष्ठित करने वाले कुछ बिंदुओं का सेट होती है, आमतौर पर हर बिंदु के लिए विश्वसनीयता स्कोर के साथ। पोज निर्धारण उचित विकल्प है जब आपको स्टीन में एक वस्तु के विशेष हिस्सों की पहचान करनी होती है और विभिन्न हिस्सों के लिए उनके स्थान की पहचान करनी होती है।

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="YouTube वीडियो प्लेयर" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>देखें:</strong> Ultralytics YOLOv8 के साथ पोज निर्धारण।
</p>

!!! Tip "युक्ति"

    YOLOv8 _pose_ मॉडल में `-pose` सफिक्स का उपयोग किया जाता है, जैसे `yolov8n-pose.pt`। ये मॉडल [COCO कीपॉइंट](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) डेटासेट पर प्रशिक्षित होते हैं और विभिन्न पोज निर्धारण कार्यों के लिए उपयुक्त होते हैं।

## [मॉडल्स](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 पूर्वानुमानित पोज मॉडलस यहाँ दिखाए जाते हैं। पहचानें, अंश और पोज मॉडल मुख्यतः [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) डेटासेट पर प्रशिक्षित हैं, जबकि क्लासिफाई मॉडल्स को [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) डेटासेट पर प्रशिक्षित किया जाता है।

पूर्वानुमानित मॉडल `Models` को Ultralytics के नवीनतम [रिलीज़](https://github.com/ultralytics/assets/releases) से स्वचालित रूप से डाउनलोड करेंगे।

| मॉडल                                                                                                 | आकार<br><sup>(तत्व) | mAP<sup>पोज<br>50-95 | mAP<sup>पोज<br>50 | ह्वेग<br><sup>CPU ONNX<br>(ms) | ह्वेग<br><sup>A100 TensorRT<br>(ms) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|---------------------|----------------------|-------------------|--------------------------------|-------------------------------------|---------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640                 | 50.4                 | 80.1              | 131.8                          | 1.18                                | 3.3                 | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640                 | 60.0                 | 86.2              | 233.2                          | 1.42                                | 11.6                | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640                 | 65.0                 | 88.8              | 456.3                          | 2.00                                | 26.4                | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640                 | 67.6                 | 90.0              | 784.5                          | 2.59                                | 44.4                | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640                 | 69.2                 | 90.2              | 1607.1                         | 3.73                                | 69.4                | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280                | 71.6                 | 91.2              | 4088.7                         | 10.04                               | 99.1                | 1066.4            |

- **mAP<sup>val</sup>** मान एकल मॉडल एकल स्केल पर [COCO कीपॉइंट val2017](https://cocodataset.org) डेटासेट पर है।
  <br>`yolo val pose data=coco-pose.yaml device=0` के द्वारा पुनरोत्पादित करें
- **Speed** [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) इन्स्टेंस का उपयोग करते हुए COCO val छवियों पर औसतित गणना।
  <br>`yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu` के द्वारा पुनरार्चन करें

## ट्रेन

COCO128-pose डेटासेट पर YOLOv8-pose मॉडल को प्रशिक्षित करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n-pose.yaml')  # YAML से एक नया मॉडल बनाएँ
        model = YOLO('yolov8n-pose.pt')  # पूर्वानुमानित मॉडल लोड करें (प्रशिक्षण के लिए सिफारिश किया जाता है)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # YAML से बनाएँ और वजन स्थानांतरित करें

        # मॉडल को प्रशिक्षित करें
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAML से नया मॉडल बनाएँ और पूर्वानुमानित वजन स्थानांतरित करना शुरू करें
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # पूर्वानुमानित *.pt मॉडल से प्रशिक्षण शुरू करें
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # YAML से नया मॉडल बनाएँ, पूर्वानुमानित वजनों को स्थानांतरित करें और प्रशिक्षण शुरू करें
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### डेटासेट प्रारूप

YOLO पोज डेटासेट प्रारूप को विस्तार से [डेटासेट गाइड](../../../datasets/pose/index.md) में दिया गया है। अपनी मौजूदा डेटासेट को अन्य प्रारूपों (जैसे कि COCO आदि) से YOLO प्रारूप में रूपांतरित करने के लिए कृपया [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) उपकरण का उपयोग करें।

## मान्यता प्राप्त करें

COCO128-pose डेटासेट पर प्रशिक्षित YOLOv8n-pose मॉडल की सटीकता को मान्यता प्राप्त करें। `model` के रूप में कोई आर्ग्युमेंट पारित करने की आवश्यकता नहीं है प्रशिक्षण `data` और सेटिंग्स को मॉडल खिताबों के रूप में रखता है।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n-pose.pt')  # रिपोर्टेड मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # एक कस्टम मॉडल लोड करें

        # मॉडल की सटीकता मान्यता प्राप्त करें
        metrics = model.val()  # कोई आर्ग्युमेंट आवश्यक नहीं है, डेटासेट और सेटिंग्स याद रखा जाता है
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # प्रत्येक श्रेणी के map50-95 सूची में है
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # आधिकारिक मॉडल मान्यांकन करें
        yolo pose val model=path/to/best.pt  # कस्टम मॉडल को मान्यता प्राप्त करें
        ```

## पूर्वानुमान लगाएं

प्रशिक्षित YOLOv8n-pose मॉडल के साथ छवियों पर पूर्वानुमान चलाएं।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n-pose.pt')  # रिपोर्टेड मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # एक कस्टम मॉडल लोड करें

        # मॉडल के साथ पूर्वानुमान करें
        results = model('https://ultralytics.com/images/bus.jpg')  # एक छवि पर पूर्वानुमान करें
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # आधिकारिक मॉडल के साथ पूर्वानुमान लगाएं
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # कस्टम मॉडल के साथ पूर्वानुमान लगाएं
        ```

एक्सपोर्ट

YOLOv8n पोज मॉडल को ONNX, CoreML जैसे अन्य प्रारूप में निर्यात करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n-pose.pt')  # रिपोर्टेड मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # एक कस्टम प्रशिक्षित मॉडल लोड करें

        # मॉडल को निर्यात करें
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # आधिकारिक मॉडल को निर्यात करें
        yolo export model=path/to/best.pt format=onnx  # कस्टम प्रशिक्षित मॉडल को निर्यात करें
        ```

निर्यात के लिए उपलब्ध YOLOv8-pose निर्यात प्रारूप नीचे करें दिए गए हैं। आप निर्यात किए गए मॉडल पर सीधा पूर्वानुमान या मान्यता कर सकते हैं, उदाहरण के लिए `yolo predict model=yolov8n-pose.onnx`। निर्यात पूरा होने के बाद अपने मॉडल के उपयोग के उदाहरण दिखाए गए हैं।

| प्रारूप                                                            | `format` आर्ग्युमेंट | मॉडल                           | मेटाडेटा | आर्ग्युमेंट।                                        |
|--------------------------------------------------------------------|----------------------|--------------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                    | `yolov8n-pose.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`        | `yolov8n-pose.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`               | `yolov8n-pose.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`           | `yolov8n-pose_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`             | `yolov8n-pose.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`             | `yolov8n-pose.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`        | `yolov8n-pose_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`                 | `yolov8n-pose.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`             | `yolov8n-pose.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`            | `yolov8n-pose_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`               | `yolov8n-pose_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`             | `yolov8n-pose_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`               | `yolov8n-pose_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

निर्यात विवरण के लिए [निर्यात](https://docs.ultralytics.com/modes/export/) पृष्ठ देखें।
