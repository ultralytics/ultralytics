---
comments: true
description: Официальная документация YOLOv8 от Ultralytics. Узнайте, как проводить обучение, проверку, предсказание и экспорт моделей в различных форматах. Включая подробные статистические данные о производительности.
keywords: YOLOv8, Ultralytics, обнаружение объектов, предобученные модели, обучение, валидация, предсказание, экспорт моделей, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# Обнаружение объектов

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="Примеры обнаружения объектов">

Обнаружение объектов – это задача, которая включает идентификацию местоположения и класса объектов на изображении или видео.

Результат работы детектора объектов – это набор ограничивающих рамок, которые заключают в себе объекты на изображении, вместе с метками классов и уровнями достоверности для каждой рамки. Обнаружение объектов является хорошим выбором, когда необходимо определить объекты интереса в сцене, но не нужно точно знать, где находится объект или его точную форму.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Смотрите:</strong> Обнаружение объектов с предобученной моделью Ultralytics YOLOv8.
</p>

!!! Tip "Совет"

    YOLOv8 Detect модели являются стандартными моделями YOLOv8, то есть `yolov8n.pt`, и предобучены на [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Здесь показаны предобученные модели YOLOv8 Detect. Модели Detect, Segment и Pose предобучены на датасете [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), в то время как модели Classify предобучены на датасете [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) автоматически загружаются с последнего релиза Ultralytics [release](https://github.com/ultralytics/assets/releases) при первом использовании.

| Модель                                                                               | размер<br><sup>(пиксели) | mAP<sup>val<br>50-95 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(М) | FLOPs<br><sup>(Б) |
|--------------------------------------------------------------------------------------|--------------------------|----------------------|-----------------------------------|----------------------------------------|-----------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                      | 37.3                 | 80.4                              | 0.99                                   | 3.2                   | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                      | 44.9                 | 128.4                             | 1.20                                   | 11.2                  | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                      | 50.2                 | 234.7                             | 1.83                                   | 25.9                  | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                      | 52.9                 | 375.2                             | 2.39                                   | 43.7                  | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                      | 53.9                 | 479.1                             | 3.53                                   | 68.2                  | 257.8             |

- **mAP<sup>val</sup>** значения для одиночной модели одиночного масштаба на датасете [COCO val2017](https://cocodataset.org).
  <br>Для воспроизведения используйте `yolo val detect data=coco.yaml device=0`
- **Скорость** усреднена по изображениям COCO val на экземпляре [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Для воспроизведения используйте `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

## Обучение

Обучите модель YOLOv8n на датасете COCO128 в течение 100 эпох с размером изображения 640. Полный список доступных аргументов см. на странице [Конфигурация](/../usage/cfg.md).

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n.yaml')  # создать новую модель из YAML
        model = YOLO('yolov8n.pt')  # загрузить предобученную модель (рекомендуется для обучения)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # создать из YAML и перенести веса

        # Обучите модель
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Создать новую модель из YAML и начать обучение с нуля
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Начать обучение с предобученной модели *.pt
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Создать новую модель из YAML, перенести в нее предобученные веса и начать обучение
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Формат датасета

Формат датасета для обнаружения YOLO можно найти более подробно в [Руководстве по датасетам](../../../datasets/detect/index.md). Чтобы конвертировать ваш существующий датасет из других форматов (например, COCO и т.д.) в формат YOLO, пожалуйста, используйте инструмент [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) от Ultralytics.

## Валидация

Проверьте точность обученной модели YOLOv8n на датасете COCO128. Необходимо передать аргументы, поскольку `model` сохраняет свои `data` и аргументы обучения как атрибуты модели.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить собственную модель

        # Проверьте модель
        metrics = model.val()  # аргументы не нужны, набор данных и настройки запоминаются
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # список содержит map50-95 для каждой категории
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # val официальная модель
        yolo detect val model=path/to/best.pt  # val собственная модель
        ```

## Предсказание

Используйте обученную модель YOLOv8n для выполнения предсказаний на изображениях.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить собственную модель

        # Сделайте предсказание с помощью модели
        results = model('https://ultralytics.com/images/bus.jpg')  # сделать предсказание на изображении
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # предсказание с официальной моделью
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # предсказание с собственной моделью
        ```

Полные детали режима `predict` смотрите на странице [Предсказание](https://docs.ultralytics.com/modes/predict/).

## Экспорт

Экспортируйте модель YOLOv8n в другой формат, такой как ONNX, CoreML и др.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить собственную модель после обучения

        # Экспортируйте модель
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # экспорт официальной модели
        yolo export model=path/to/best.pt format=onnx  # экспорт собственной модели после обучения
        ```

Доступные форматы экспорта YOLOv8 приведены в таблице ниже. Вы можете выполнять предсказания или проверку непосредственно на экспортированных моделях, например `yolo predict model=yolov8n.onnx`. Примеры использования для вашей модели показаны после завершения экспорта.

| Формат                                                             | Аргумент `format` | Модель                    | Метаданные | Аргументы                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅          | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅          | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅          | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅          | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅          | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌          | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅          | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅          | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅          | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅          | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅          | `imgsz`, `half`                                     |

Полные детали режима `export` смотрите на странице [Экспорт](https://docs.ultralytics.com/modes/export/).
