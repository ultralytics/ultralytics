---
comments: true
description: सभी प्रकार के निर्यात स्तर पर YOLOv8 मॉडल्स को निर्यात करने के लिए आपके लिए चरण-दर-चरण मार्गदर्शिका। अब निर्यात की जांच करें!
keywords: YOLO, YOLOv8, Ultralytics, मॉडल निर्यात, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, निर्यात मॉडल
---

# Ultralytics YOLO के साथ मॉडल निर्यात

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="यूल्ट्रालिक्स YOLO ecosystem and integrations">

## परिचय

एक मॉडल की प्रशिक्षण की अंतिम लक्ष्य उसे वास्तविक दुनिया के आवेदनों के लिए तैनात करना होता है। उल्ट्रालिटीक्स YOLOv8 में निर्यात मोड में आपको अभिनवता रेंज के ऑप्शन प्रदान करता है, वायरले किए गए मॉडल को विभिन्न स्वरूपों में निर्यात करने के लिए, जिससे वे विभिन्न प्लेटफॉर्मों और उपकरणों पर प्रदर्शित किए जा सकें। यह व्यापक मार्गदर्शिका अधिकतम संगतता और प्रदर्शन प्राप्त करने के तरीकों को दिखाने का लक्ष्य रखती है।

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube वीडियो प्लेयर" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>देखें:</strong> अपने उत्पादन को निर्यात करने के लिए कस्टम प्रशिक्षित Ultralytics YOLOv8 मॉडल निर्यात करने और वेबकैम पर लाइव अनुमान चलाने।
</p>

## YOLOv8 के निर्यात मोड को क्यों चुनें?

- **विविधता:** ONNX, TensorRT, CoreML और अन्य सहित कई फॉर्मेट में निर्यात करें।
- **प्रदर्शन:** TensorRT में 5x जीपीयू स्पीडअप और ONNX या OpenVINO में 3x सीपीयू स्पीडअप प्राप्त करें।
- **संगतता:** अपने मॉडल को कई हार्डवेयर और सॉफ़्टवेयर पर संगठित करें।
- **उपयोग की सुविधा:** त्वरित और सीधी मॉडल निर्यात के लिए सरल CLI और Python API।

### निर्यात मोड की प्रमुख विशेषताएं

यहाँ कुछ मुख्य विशेषताएँ हैं:

- **एक-क्लिक निर्यात:** अलग-अलग फॉर्मेट में निर्यात करने के लिए सरल कमांड।
- **बैच निर्यात:** बैच-इन्फरेंस क्षमता वाले मॉडलों को निर्यात करें।
- **सुधारित अनुमान:** निर्यात किए गए मॉडल अनुमान समय के लिए अनुकूलन किए जाते हैं।
- **ट्यूटोरियल वीडियो:** सुविधाएं और ट्यूटोरियल सुनिश्चित करने के लिए गहन मार्गदर्शिकाओं का उपयोग करें।

!!! Tip "सुझाव"

    * 3x सीपीयू स्पीडअप के लिए ONNX या OpenVINO में निर्यात करें।
    * 5x जीपीयू स्पीडअप के लिए TensorRT में निर्यात करें।

## उपयोग उदाहरण

YOLOv8n मॉडल को ONNX या TensorRT जैसे अलग फॉर्मेट में निर्यात करें। पूरी सूची निर्यात तर्कों के लिए नीचे दिए गए Arguments खंड को देखें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics import YOLO

        # एक मॉडल लोड करें
        model = YOLO('yolov8n.pt')  # एक आधिकारिक मॉडल लोड करें
        model = YOLO('path/to/best.pt')  # एक कस्टम प्रशिक्षित मॉडल लोड करें

        # मॉडल निर्यात करें
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # आधिकारिक मॉडल का निर्यात करें
        yolo export model=path/to/best.pt format=onnx  # कस्टम प्रशिक्षित मॉडल का निर्यात करें
        ```

## Arguments

YOLO मॉडलों के निर्यात सेटिंग्स निर्यात के विभिन्न विन्यास और विकल्पों के बारे में होते हैं, जिन्हें यूज़ करके मॉडल को अन्य पर्यावरण या प्लेटफ़ॉर्म में सहेजने या निर्यात करने के लिए उपयोग किया जा सकता है। इन सेटिंग्स से मॉडल के प्रदर्शन, आकार और विभिन्न सिस्टम के साथ संगतता प्रभावित हो सकती हैं। कुछ सामान्य YOLO निर्यात सेटिंग्स में निर्यात की गई मॉडल फ़ाइल का स्वरूप (जैसे ONNX, TensorFlow SavedModel), मॉडल कोरी सहवास में चलाने वाली उपकरण (जैसे CPU, GPU) और मास्क या प्रत्येक बॉक्स पर कई लेबलों की उपस्थिति जैसे अतिरिक्त विशेषताएँ शामिल हो सकते हैं। निर्यात प्रक्रिया प्रभावित करने वाले अन्य कारकों में मॉडल द्वारा उपयोग के लिए एक विशेष कार्य और लक्षित पर्यावरण या प्लेटफ़ॉर्म की आवश्यकताओं या सीमाओं का ध्यान देना महत्वपूर्ण है। लक्ष्य प्रयोजन और लक्ष्यित वातावरण में प्रभावी ढंग से उपयोग होने के लिए इन सेटिंग्स को ध्यान से विचार करना महत्वपूर्ण है।

| कुंजी       | मान             | विवरण                                                                  |
|-------------|-----------------|------------------------------------------------------------------------|
| `format`    | `'torchscript'` | योग्यता के लिए निर्यात करने के लिए स्वरूप                              |
| `imgsz`     | `640`           | एकल रूप में छवि का आकार या (h, w) सूची, जैसे (640, 480)                |
| `keras`     | `False`         | TF SavedModel निर्यात के लिए केरस का प्रयोग करें                       |
| `optimize`  | `False`         | TorchScript: मोबाइल के लिए ऑप्टिमाइज़ करें                             |
| `half`      | `False`         | FP16 संगणना                                                            |
| `int8`      | `False`         | INT8 संगणना                                                            |
| `dynamic`   | `False`         | ONNX/TensorRT: गतिशील ध्यान दिलाने वाले ध्यान                          |
| `simplify`  | `False`         | ONNX/TensorRT: मॉडल को सरल बनाएं                                       |
| `opset`     | `None`          | ONNX: ऑपसेट संस्करण (वैकल्पिक, डिफ़ॉल्ट्स को नवीनतम के रूप में छोड़ें) |
| `workspace` | `4`             | TensorRT: कार्यक्षेत्र आकार (GB)                                       |
| `nms`       | `False`         | CoreML: NMS जोड़ें                                                     |

## निर्यात स्वरूप

नीचे दिए गए तालिका में YOLOv8 निर्यात स्वरूप दिए गए हैं। आप किसी भी स्वरूप में निर्यात कर सकते हैं, जैसे `format='onnx'` या `format='engine'`।

| स्वरूप                                                             | `format` तर्क | मॉडल                      | मेटाडाटा | तर्क                                                |
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
