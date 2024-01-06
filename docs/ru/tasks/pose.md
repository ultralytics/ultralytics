---
comments: true
description: Узнайте, как использовать Ultralytics YOLOv8 для задач оценки позы. Найдите предварительно обученные модели, узнайте, как обучать, проверять, предсказывать и экспортировать свои собственные.
---

# Оценка позы

![Примеры оценки позы](https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png)

Оценка позы — это задача, заключающаяся в определении местоположения определённых точек на изображении, обычно называемых контрольными точками. Контрольные точки могут представлять различные части объекта, такие как суставы, ориентиры или другие характерные особенности. Расположение контрольных точек обычно представлено в виде набора 2D `[x, y]` или 3D `[x, y, visible]` координат.

Результат работы модели оценки позы — это набор точек, представляющих контрольные точки на объекте в изображении, обычно вместе с оценками уверенности для каждой точки. Оценка позы является хорошим выбором, когда вам нужно идентифицировать конкретные части объекта в сцене и их расположение относительно друг друга.

[Смотрите: Оценка позы с Ultralytics YOLOv8.](https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ)

!!! Tip "Совет"

    Модели _pose_ YOLOv8 используют суффикс `-pose`, т.е. `yolov8n-pose.pt`. Эти модели обучены на наборе данных [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) и подходят для различных задач оценки позы.

## [Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Здесь представлены предварительно обученные модели YOLOv8 Pose. Модели Detect, Segment и Pose предварительно обучены на наборе данных [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), а модели Classify — на наборе данных [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) скачиваются автоматически из последнего [релиза](https://github.com/ultralytics/assets/releases) Ultralytics при первом использовании.

| Модель                                                                                               | размер<br><sup>(пиксели) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(М) | FLOPs<br><sup>(Б) |
|------------------------------------------------------------------------------------------------------|--------------------------|-----------------------|--------------------|-----------------------------------|----------------------------------------|-----------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                      | 50.4                  | 80.1               | 131.8                             | 1.18                                   | 3.3                   | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                      | 60.0                  | 86.2               | 233.2                             | 1.42                                   | 11.6                  | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                      | 65.0                  | 88.8               | 456.3                             | 2.00                                   | 26.4                  | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                      | 67.6                  | 90.0               | 784.5                             | 2.59                                   | 44.4                  | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                      | 69.2                  | 90.2               | 1607.1                            | 3.73                                   | 69.4                  | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                     | 71.6                  | 91.2               | 4088.7                            | 10.04                                  | 99.1                  | 1066.4            |

- **mAP<sup>val</sup>** значения для одной модели одиночного масштаба на наборе данных [COCO Keypoints val2017](http://cocodataset.org).
  <br>Воспроизводится с помощью: `yolo val pose data=coco-pose.yaml device=0`
- **Скорость** усреднена по изображениям COCO val на [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) инстансе.
  <br>Воспроизводится с помощью: `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## Обучение

Обучите модель YOLOv8-pose на наборе данных COCO128-pose.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-pose.yaml')  # создать новую модель из YAML
        model = YOLO('yolov8n-pose.pt')  # загрузить предварительно обученную модель (рекомендуется для обучения)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # создать из YAML и перенести веса

        # Обучить модель
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Создать новую модель из YAML и начать обучение с нуля
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Начать обучение с предварительно обученной модели *.pt
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Создать новую модель из YAML, перенести предварительно обученные веса и начать обучение
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### Формат набора данных

Формат набора данных YOLO pose можно найти в подробностях в [Руководстве по наборам данных](../../../datasets/pose/index.md). Для преобразования существующего набора данных из других форматов (например, COCO и т.д.) в формат YOLO, пожалуйста, используйте инструмент [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) от Ultralytics.

## Проверка

Проверьте точность обученной модели YOLOv8n-pose на наборе данных COCO128-pose. Аргументы не нужны, так как `model`
запоминает свои `data` и аргументы как атрибуты модели.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-pose.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить свою модель

        # Проверить модель
        metrics = model.val()  # аргументы не нужны, набор данных и настройки запомнены
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # список содержит map50-95 для каждой категории
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # проверить официальную модель
        yolo pose val model=path/to/best.pt  # проверить свою модель
        ```

## Предсказание

Используйте обученную модель YOLOv8n-pose для выполнения предсказаний на изображениях.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-pose.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить свою модель

        # Сделать предсказание моделью
        results = model('https://ultralytics.com/images/bus.jpg')  # предсказать по изображению
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # предсказать официальной моделью
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # предсказать своей моделью
        ```

Полные детали работы в режиме `predict` смотрите на странице [Predict](https://docs.ultralytics.com/modes/predict/).

## Экспорт

Экспортируйте модель YOLOv8n Pose в другой формат, такой как ONNX, CoreML и т.д.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-pose.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить свою обученную модель

        # Экспортировать модель
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # экспортировать официальную модель
        yolo export model=path/to/best.pt format=onnx  # экспортировать свою обученную модель
        ```

Доступные форматы экспорта модели YOLOv8-pose приведены в таблице ниже. Вы можете делать предсказания или проверки непосредственно с экспортированных моделей, например, `yolo predict model=yolov8n-pose.onnx`. Примеры использования показаны для вашей модели после завершения экспорта.

| Формат                                                             | Аргумент `format` | Модель                         | Метаданные | Аргументы                                           |
|--------------------------------------------------------------------|-------------------|--------------------------------|------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-pose.pt`              | ✅          | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-pose.torchscript`     | ✅          | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-pose.onnx`            | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-pose_openvino_model/` | ✅          | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-pose.engine`          | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-pose.mlpackage`       | ✅          | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-pose_saved_model/`    | ✅          | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-pose.pb`              | ❌          | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-pose.tflite`          | ✅          | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-pose_edgetpu.tflite`  | ✅          | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-pose_web_model/`      | ✅          | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-pose_paddle_model/`   | ✅          | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-pose_ncnn_model/`     | ✅          | `imgsz`, `half`                                     |

Полные детали экспорта смотрите на странице [Export](https://docs.ultralytics.com/modes/export/).
