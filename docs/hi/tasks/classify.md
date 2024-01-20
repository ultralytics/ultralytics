---
comments: true
description: YOLOv8 Classify मॉडल्स के बारे में जानें इमेज क्लासिफिकेशन के लिए। प्रीट्रेन्ड माॅडेल्स की सूची और ट्रेन, वेलिडेट, प्रेडिक्ट और एक्सपोर्ट माॅडेल्स के बारे में विस्तृत जानकारी प्राप्त करें।
keywords: Ultralytics, YOLOv8, इमेज क्लासिफिकेशन, प्रीट्रेन्ड माॅडेल्स, YOLOv8n-cls, ट्रेन, वेलिडेट, प्रेडिक्ट, माॅडेल एक्सपोर्ट
---

# इमेज क्लासिफिकेशन

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="इमेज क्लासिफिकेशन उदाहरण">

इमेज क्लासिफिकेशन तीन कार्यों में से सबसे सरल है और पूरी तस्वीर को एक पूर्वनिर्धारित कक्षा में वर्गीकृत करना शामिल होता है।

इमेज क्लासिफायर का आउटपुट एक एकल क्लास लेबल और एक विश्वास प्रामाणिकता स्कोर होता है। इमेज क्लासिफिकेशन उपयोगी होता है जब आपको केवल इसे जानने की जरूरत होती है कि एक इमेज किस कक्षा में सम्मिलित है और आपको नहीं पता होना चाहिए कि उस कक्षा के वस्त्राणु किस स्थान पर स्थित हैं या उनकी सटीक आकृति क्या है।

!!! Tip "टिप"

    YOLOv8 Classify मॉडेल्स में `-cls` संकेतक प्रयोग किया जाता है, जैसे `yolov8n-cls.pt` और इन्हें पूर्व प्रशिक्षित किया जाता है [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) पर।

## [मॉडेल](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

यहां YOLOv8 पूर्व प्रशिक्षित Classify मॉडेल दिखाए गए हैं। Detect, Segment, और Pose मॉडेल्स [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) डेटासेट पर पूर्व प्रशिक्षित होते हैं, जबकि Classify मॉडेल [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) डेटासेट पर पूर्व प्रशिक्षित होते हैं।

[मॉडेल](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) डाउनलोड पहली बार उपयोग पर ताजगी Ultralytics [प्रकाशन](https://github.com/ultralytics/assets/releases) से स्वतः होता है।

| मॉडेल                                                                                        | आकार<br><sup>(पिक्सेल) | तालिका<br><sup>शीर्ष 1 | तालिका<br><sup>शीर्ष 5 | स्पीड<br><sup>सीपीयू ONNX<br>(मि. सेकंड) | स्पीड<br><sup>A100 TensorRT<br>(मि. सेकंड) | पैरामीटर<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
|----------------------------------------------------------------------------------------------|------------------------|------------------------|------------------------|------------------------------------------|--------------------------------------------|----------------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt) | 224                    | 66.6                   | 87.0                   | 12.9                                     | 0.31                                       | 2.7                  | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224                    | 72.3                   | 91.1                   | 23.4                                     | 0.35                                       | 6.4                  | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224                    | 76.4                   | 93.2                   | 85.4                                     | 0.62                                       | 17.0                 | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224                    | 78.0                   | 94.1                   | 163.0                                    | 0.87                                       | 37.5                 | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224                    | 78.4                   | 94.3                   | 232.0                                    | 1.01                                       | 57.4                 | 154.8                    |

- **तालिका** मॉडेलों की ImageNet डेटासेट मान्यीकरण सेट पर सटीकता है।
  <br>`yolo val classify data=path/to/ImageNet device=0` द्वारा पुनः उत्पन्न करें
- **स्पीड** एक [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) इंस्टेंस का उपयोग करके ImageNet के वैल छवियों पर औसत जोड़ी गई है।
  <br>`yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` द्वारा पुनः उत्पन्न करें

## ट्रेन

100 एपॉक्स के लिए MNIST160 डेटासेट पर YOLOv8n-cls को 64 इमेज आकार पर रिक्तियों के साथ ट्रेन करें। उपलब्ध विकल्पों की पूरी सूची के लिए [Configuration](/../usage/cfg.md) पेज देखें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडेल लोड करें
        model = YOLO('yolov8n-cls.yaml')  # YAML से एक नया मॉडेल बनाएं
        model = YOLO('yolov8n-cls.pt')  # पूर्व प्रशिक्षित मॉडेल लोड करें (ट्रेनिंग के लिए सिफारिश की जाती है)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # YAML से बनाएँ और भार ट्रांसफर करें

        # मॉडेल ट्रेन करें
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # YAML से नया मॉडेल बनाएं और अच्छे से प्रशिक्षण शुरू करें
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # पूर्व प्रशिक्षित *.pt मॉडेल से प्रशिक्षण शुरू करें
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # YAML से नया मॉडेल बनाएँ, उसमें पूर्व प्रशिक्षित भार भी स्थानांतरित करें और प्रशिक्षण शुरू करें
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### डेटासेट प्रारूप

YOLO क्लासिफिकेशन डेटासेट प्रारूप [Dataset Guide](../../../datasets/classify/index.md) में विस्तृत रूप में दिया गया है।

## वेलिडेट

MNIST160 डेटासेट पर प्रशिक्षित YOLOv8n-cls मॉडेल की सटीकता का मूल्यांकन करें। कोई आर्गुमेंट चक्रवात नहीं करना चाहिए क्योंकि `मॉडेल` अपने प्रशिक्षण यथार्थ डेटा और आर्गुमेंट्स को स्मरण रखता है।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडेल लोड करें
        model = YOLO('yolov8n-cls.pt')  # एक आधिकारिक मॉडेल लोड करें
        model = YOLO('path/to/best.pt')  # एक स्वचालित मॉडेल लोड करें

        # मॉडेल का मूल्यांकन करें
        metrics = model.val()  # कोई आर्गुमेंट आवश्यक नहीं हैं, डेटासेट और सेटिंग्स याद रखे जाते हैं
        metrics.top1   # शीर्ष1 सटीकता
        metrics.top5   # शीर्ष5 सटीकता
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # आधिकारिक मॉडेल का मूल्यांकन करें
        yolo classify val model=path/to/best.pt  # कस्टम मॉडेल का मूल्यांकन करें
        ```

## प्रेडिक्ट

प्रशिक्षित YOLOv8n-cls मॉडेल का उपयोग तस्वीरों पर पूर्वानुमान चलाने के लिए करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडेल लोड करें
        model = YOLO('yolov8n-cls.pt')  # एक आधिकारिक मॉडेल लोड करें
        model = YOLO('path/to/best.pt')  # एक स्वचालित मॉडेल लोड करें

        # मॉडेल के साथ पूर्वानुमान करें
        results = model('https://ultralytics.com/images/bus.jpg')  # एक इमेज पर पूर्वानुमान करें
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # आधिकारिक मॉडेल के साथ पूर्वानुमान करें
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # कस्टम मॉडेल के साथ पूर्वानुमान करें
        ```

पूर्वानुमान पूरा होने के बाद निर्यात को सीधे पूर्वानुमानित मॉडेल पर लागू कर सकते हैं, जैसे `yolo predict model=yolov8n-cls.onnx`। एक्सपोर्ट पूर्ण होने के बाद, अपने मॉडेल के उपयोग के लिए आपको उपयोग उदाहरण दिखाए गए हैं।

## एक्सपोर्ट

YOLOv8n-cls मॉडल को ONNX, CoreML आदि जैसे विभिन्न प्रारूपों में निर्यात करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडेल लोड करें
        model = YOLO('yolov8n-cls.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained model

        # मॉडेल को निर्यात करें
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

टेबल में उपलब्ध YOLOv8-cls निर्यात प्रारूप निम्नानुसार हैं। निर्यात पूरा होने के बाद आप सीधे निर्यात किए गए मॉडेल पर पूर्व-आश्रिताओं की तरह पूर्वानुमान या मूल्यांकन कर सकते हैं, जैसे `yolo predict model=yolov8n-cls.onnx`। उपयोग की उदाहरण आपके मॉडेल के लिए निर्यात पूरा होने के बाद दिखाए गए हैं।

| प्रारूप                                                            | `format` आर्गुमेंट | मॉडेल                         | मेटाडेटा | आर्गुमेंट्स                                         |
|--------------------------------------------------------------------|--------------------|-------------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n-cls.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n-cls.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n-cls.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n-cls_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n-cls.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n-cls.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n-cls_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n-cls.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n-cls.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n-cls_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n-cls_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n-cls_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n-cls_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

[Export](https://docs.ultralytics.com/modes/export/) पेज में `export` के पूरी विवरण देखें।
