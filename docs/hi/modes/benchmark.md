---
comments: true
description: Ultralytics YOLO के विभिन्न निर्यात प्रारूपों के जरिए YOLOv8 की गति और सटीकता का जांच करें; mAP50-95, accuracy_top5 माप, और अन्य मापों पर अनुभव प्राप्त करें।
keywords: Ultralytics, YOLOv8, बंचमार्किंग, गति प्रोफाइलिंग, सटीकता प्रोफाइलिंग, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLO निर्यात प्रारूप
---

# उल्ट्राल्याटिक्स YOLO के साथ मॉडल बंचमार्किंग

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="उल्ट्राल्याटिक्स YOLO पारिस्थितिकी और समावेश">

## परिचय

जब आपका मॉडल प्रशिक्षित और सत्यापित हो जाता है, तो आगामी तार्किक चरण होता है कि तत्कालिक वास्तविक-दुनिया की स्थितियों में इसके प्रदर्शन का मूल्यांकन करें। Ultralytics YOLOv8 में बेंचमार्क मोड इस उद्देश्य की सेवा करता है, जहां उपयोगकर्ताओं को अपने मॉडल की गति और सटीकता का मूल्यांकन करने के लिए एक मजबूत ढांचा प्रदान करता है।

## बंचमार्किंग क्यों महत्वपूर्ण है?

- **जागरूक निर्णय:** गति और सटीकता के बीच ट्रेड-ऑफ के बारे में जानकारी प्राप्त करें।
- **संसाधन आवंटन:** अलग-अलग निर्यात प्रारूपों का विभिन्न हार्डवेयर पर कैसा काम करता है इसकी समझ पाएं।
- **अनुकूलन:** अपने विशिष्ट उपयोग मामले में सर्वोत्तम प्रदर्शन प्रदान करने वाला निर्यात प्रारूप कौन सा है, इसकी जानकारी प्राप्त करें।
- **लागत संचय:** बंचमार्क परिणामों के आधार पर हार्डवेयर संसाधनों का अधिक अभिकल्प सेवन करें।

### बंचमार्क मोड में मुख्य माप

- **mAP50-95:** वस्तु का पता लगाने, विभाजन करने और स्थिति मान के लिए।
- **accuracy_top5:** छवि वर्गीकरण के लिए।
- **परिन्दता समय:** प्रति छवि के लिए लिया गया समय मिलीसेकंड में।

### समर्थित निर्यात प्रारूप

- **ONNX:** CPU प्रदर्शन के लिए आदर्श
- **TensorRT:** अधिकतम GPU क्षमता के लिए
- **OpenVINO:** Intel हार्डवेयर संशोधन के लिए
- **CoreML, TensorFlow SavedModel, और अधिक:** विविध डिप्लॉयमेंट आवश्यकताओं के लिए।

!!! Tip "युक्ति"

    * तकनीकी कारणों से कंप्यूटिंग संसाधनों का उपयोग करते समय ONNX या OpenVINO में निर्यात करें, ताकि आप CPU स्पीड तक upto 3x तक स्पीडअप कर सकें।
    * GPU स्पीड तक अपने कंप्यूटिंग संसाधनों का उपयोग करते समय TensorRT में निर्यात करें ताकि आप तक 5x तक स्पीडअप कर सकें।

## उपयोग उदाहरण

समर्थित सभी निर्यात प्रारूपों पर ONNX, TensorRT आदि के साथ YOLOv8n बंचमार्क चलाएं। पूरी निर्यात विवरण के लिए नीचे Arguments अनुभाग देखें।

!!! Example "उदाहरण"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # GPU पर बंचमार्क
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo बंचमार्क model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Arguments

`model`, `data`, `imgsz`, `half`, `device`, और `verbose` जैसे तर्क उपयोगकर्ताओं को मानदंडों को अपनी विशेष आवश्यकताओं के लिए सुगमता के साथ बंचमार्क को संशोधित करने की सुविधा प्रदान करते हैं, और विभिन्न निर्यात प्रारूपों के प्रदर्शन की तुलना करने की सुविधा प्रदान करते हैं।

| कुंजी     | मान        | विवरण                                                                           |
|-----------|------------|---------------------------------------------------------------------------------|
| `model`   | `कोई नहीं` | मॉडल फ़ाइल का पथ, यानी yolov8n.pt, yolov8n.yaml                                 |
| `data`    | `कोई नहीं` | बेंचमार्किंग डेटासेट को संदर्भित करने वाले YAML फ़ाइल का पथ (val लेबल के तहत)   |
| `imgsz`   | `640`      | छवि का आकार स्कैलर या (h, w) सूची, अर्थात (640, 480)                            |
| `half`    | `असत्य`    | FP16 माप्यांकन                                                                  |
| `int8`    | `असत्य`    | INT8 माप्यांकन                                                                  |
| `device`  | `कोई नहीं` | चलाने के लिए युक्ति उपकरण, अर्थात cuda device=0 या device=0,1,2,3 या device=cpu |
| `verbose` | `असत्य`    | त्रुटि में न जारी रखे (बूल), या वाल (फ्लोट)                                     |

## निर्यात प्रारूप

बंचमार्क प्रयास होगा निम्नलिखित सभी संभावित निर्यात प्रारूपों पर स्वचालित रूप से चलाने की कोशिश करेगा।

| प्रारूप                                                            | `प्रारूप` तर्क | मॉडल                      | मेटाडेटा | तर्क                                                |
|--------------------------------------------------------------------|----------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -              | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`  | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`         | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`     | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`       | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`       | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`  | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`           | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`       | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`      | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`         | `yolov8n_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`       | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`         | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

पूर्ण निर्यात विवरण देखें निर्यात पृष्ठ में [Export](https://docs.ultralytics.com/modes/export/)।
