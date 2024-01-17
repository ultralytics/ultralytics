---
comments: true
description: Ultralytics द्वारा YOLOv8 के आधिकारिक दस्तावेज़ीकरण। Various प्रारूपों में मॉडल को प्रशिक्षित, मान्य करें, निरुपित और निर्यात करने का कैसे करें सीखें। विस्तृत प्रदर्शन आँकड़े समेत।
keywords: YOLOv8, Ultralytics, वस्तु पहचान, पूर्वप्रशिक्षित मॉडल, प्रशिक्षण, मान्यता, भविष्यवाणी, मॉडल निर्यात, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# वस्तु पहचान

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="वस्तु पहचान उदाहरण">

वस्तु पहचान एक कार्य है जिसमें चित्र या वीडियो स्ट्रीम में वस्तुओं की स्थान और वर्ग की पहचान करने का समय शामिल होता है।

वस्तु पहचान एक सेट होती है जिसमें वस्तुओं को घेरने वाले बाउंडिंग बॉक्स का पता लगाया जाता है, साथ ही प्रत्येक बॉक्स के लिए वर्ग लेबल और विश्वसनीयता स्कोर शामिल होते हैं। चित्र में हरी उड़ी रेस सामग्री डिटेक्ट करी, बांदर को डिटेक्ट करें. प्रतिस्थान से यह पता चलता है कि वस्तु कहाँ है या उसकी सटीक आकृति क्या है, परंतु कुछ तो हैं है।

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube वीडियो प्लेयर" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>देखें:</strong> पूर्व प्रशिक्षित Ultralytics YOLOv8 मॉडल के साथ वस्तु पहचान।
</p>


!!! Tip "टिप"

YOLOv8 Detect मॉडल डिफ़ॉल्ट YOLOv8 मॉडल हैं, यानी `yolov8n.pt` और [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) पर प्रशिक्षित हैं।

## [मॉडल](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 पूर्व प्रशिक्षित Detect मॉडल यहाँ दिखाए गए हैं। Detect, Segment और Pose मॉडल [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) डेटासेट पर पूर्वप्रशिक्षित होते हैं, जबकि Classify मॉडल [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) डेटासेट पर पूर्वप्रशिक्षित होते हैं।

[मॉडल](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) पहली बार इस्तेमाल पर Ultralytics के नवीनतम [प्रकाशन](https://github.com/ultralytics/assets/releases) से स्वचालित रूप से डाउनलोड होते हैं।

| मॉडल                                                                                 | साइज़<br><sup>(pixels) | mAP<sup>val<br>50-95 | स्पीड<sup>CPU ONNX<br>(ms) | स्पीड<sup>A100 TensorRT<br>(ms) | पैराम्स<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|------------------------|----------------------|----------------------------|---------------------------------|---------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                    | 37.3                 | 80.4                       | 0.99                            | 3.2                 | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                    | 44.9                 | 128.4                      | 1.20                            | 11.2                | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                    | 50.2                 | 234.7                      | 1.83                            | 25.9                | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                    | 52.9                 | 375.2                      | 2.39                            | 43.7                | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                    | 53.9                 | 479.1                      | 3.53                            | 68.2                | 257.8             |

- **mAP<sup>val</sup>** मान को [COCO val2017](https://cocodataset.org) डेटासेट पर सिंगल-मॉडेल सिंगल-स्केल के लिए है।
  <br>`yolo` द्वारा पुनः उत्पन्न करें `के द्वारा विन्यास करें yolo val data=coco.yaml device=0`
- **Speed** [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  इंस्टेंस का उपयोग करके COCO val छवियों पर औसत लिया जाता है।
  <br>`yolo` के द्वारा पुनः उत्पन्न करें `के द्वारा विन्यास करें yolo val data=coco128.yaml batch=1 device=0|cpu`

## प्रशिक्षण

100 युगों में 640 आकृति वाले प्रशिक्षित योलोवी8 एन को COCO128 डेटासेट पर प्रशिक्षित करें। उपलब्ध तार्किक तर्कों की पूरी सूची के लिए [कॉन्फ़िगरेशन](/../usage/cfg.md) पृष्ठ देखें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n.yaml')  # YAML से नया मॉडल बनाएँ
        model = YOLO('yolov8n.pt')  # प्रशिक्षण के लिए सिफारिश किए गए पूर्वप्रशिक्षित मॉडल लोड करें
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # YAML से बनाएं और भार ट्रांसफर करें और प्रशिक्षित करें

        # मॉडल को प्रशिक्षित करें
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # YAML से एक नया मॉडल बनाकर खाली से शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # पूर्व प्रशिक्षित *.pt मॉडल से प्रशिक्षण शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # यैतायत्मिक रूप से भार ट्रांसफर करके नया मॉडल बनाएँ और प्रशिक्षण शुरू करें
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### डेटासेट प्रारूप

YOLO डिटेक्शन डेटासेट प्रारूप को [डेटासेट गाइड](../../../datasets/detect/index.md) में विस्तार से देखा जा सकता है। कृपया अपने मौजूदा डेटासेट को अन्य प्रारूपों (जैसे COCO आदि) से YOLO प्रारूप में बदलने के लिए [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) उपकरण का उपयोग करें।

## मान्यता

COCO128 डेटासेट पर प्रशिक्षित YOLOv8n मॉडल की सटीकता को मान्यता दें। मॉडल प्रदर्शन से जुड़ी कोई विधि नहीं होनी चाहिए।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n.pt')  # आधिकारिक मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # कस्टम मॉडल लोड करें

        # मॉडल की मान्यता जांचें
        metrics = model.val()  # तुलना करने के लिए कोई विधि की आवश्यकता नहीं है, डेटासेट और सेटिंग्स याद रखे जाते हैं
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # हर श्रेणी के map50-95 से संबंधित सूची
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # आधिकारिक मॉडल की मान्यता
        yolo detect val model=path/to/best.pt  # कस्टम मॉडल की मान्यता
        ```

## भविष्यवाणी

प्रशिक्षित YOLOv8n मॉडल का उपयोग चित्रों पर भविष्यवाणी करने के लिए करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n.pt')  # आधिकारिक मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # कस्टम मॉडल लोड करें

        # मॉडल के साथ भविष्यवाणी करें
        results = model('https://ultralytics.com/images/bus.jpg')  # एक छवि पर भविष्यवाणी करें
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # आधिकारिक मॉडल के साथ भविष्यवाणी
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # कस्टम मॉडल के साथ भविष्यवाणी
        ```

पूर्ण `predict` मोड़ विवरण को [भविष्यवाणी](https://docs.ultralytics.com/modes/predict/) पृष्ठ में देखें।

## निर्यात

YOLOv8n मॉडल को अन्य प्रारूप (जैसे ONNX, CoreML आदि) में निर्यात करें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # मॉडल लोड करें
        model = YOLO('yolov8n.pt')  # आधिकारिक मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # कस्टम प्रशिक्षित मॉडल लोड करें

        # मॉडल को निर्यात करें
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # आधिकारिक मॉडल को निर्यात करें
        yolo export model=path/to/best.pt format=onnx  # कस्टम प्रशिक्षित मॉडल को निर्यात करें
        ```

उपलब्ध YOLOv8 निर्यात प्रारूप नीचे की सारणी में हैं। आप निर्यातित मॉडल पर सीधे भविष्यवाणी या मान्यता कर सकते हैं, जैसे 'yolo predict model=yolov8n.onnx' आदि। निर्यात पूर्ण होने के बाद आपके मॉडल के उपयोग के उदाहरण दिखाए जाते हैं।

| प्रारूप                                                            | `format` तर्क | मॉडल                      | मेटाडाटा | तर्क                                                |
|--------------------------------------------------------------------|---------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

पूर्ण `export` विवरण को [निर्यात](https://docs.ultralytics.com/modes/export/) पृष्ठ में देखें।
