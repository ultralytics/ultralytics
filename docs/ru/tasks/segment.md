---
comments: true
description: Научитесь использовать модели сегментации объектов с помощью Ultralytics YOLO. Инструкции по обучению, валидации, предсказанию изображений и экспорту моделей.
keywords: yolov8, сегментация объектов, Ultralytics, набор данных COCO, сегментация изображений, обнаружение объектов, обучение моделей, валидация моделей, предсказания изображений, экспорт моделей
---

# Сегментация экземпляров

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="Примеры сегментации экземпляров">

Сегментация экземпляров идёт на шаг дальше по сравнению с обнаружением объектов и включает идентификацию отдельных объектов на изображении и их сегментацию от остальной части изображения.

Результатом модели сегментации экземпляров является набор масок или контуров, очерчивающих каждый объект на изображении, вместе с классовыми метками и коэффициентами уверенности для каждого объекта. Сегментация экземпляров полезна, когда вам нужно знать не только, где находятся объекты на изображении, но и их точную форму.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Смотрите:</strong> Запуск сегментации с предварительно обученной моделью Ultralytics YOLOv8 на Python.
</p>

!!! Tip "Совет"

    Модели YOLOv8 Segment используют суффикс `-seg`, например `yolov8n-seg.pt` и предварительно обучены на [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Здесь показаны предварительно обученные модели Segment YOLOv8. Модели Detect, Segment и Pose предварительно обучены на наборе данных [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), в то время как модели Classify предварительно обучены на наборе данных [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) автоматически загружаются из последнего [релиза](https://github.com/ultralytics/assets/releases) Ultralytics при первом использовании.

| Модель                                                                                       | размер<br><sup>(пиксели) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(М) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|--------------------------|----------------------|-----------------------|-----------------------------------|----------------------------------------|-----------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                      | 36.7                 | 30.5                  | 96.1                              | 1.21                                   | 3.4                   | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                      | 44.6                 | 36.8                  | 155.7                             | 1.47                                   | 11.8                  | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                      | 49.9                 | 40.8                  | 317.0                             | 2.18                                   | 27.3                  | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                      | 52.3                 | 42.6                  | 572.4                             | 2.79                                   | 46.0                  | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                      | 53.4                 | 43.4                  | 712.1                             | 4.02                                   | 71.8                  | 344.1             |

- Значения **mAP<sup>val</sup>** для одиночной модели одиночного масштаба на наборе данных [COCO val2017](http://cocodataset.org).
  <br>Воспроизведите с помощью `yolo val segment data=coco.yaml device=0`
- **Скорость** усреднена для изображений COCO val на [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  инстансе.
  <br>Воспроизведите с помощью `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`

## Обучение

Обучите модель YOLOv8n-seg на наборе данных COCO128-seg в течение 100 эпох при размере изображения 640. Полный список доступных аргументов см. на странице [Конфигурация](/../usage/cfg.md).

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-seg.yaml')  # создать новую модель из YAML
        model = YOLO('yolov8n-seg.pt')  # загрузить предварительно обученную модель (рекомендуется для обучения)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # создать из YAML и перенести веса

        # Обучить модель
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Создать новую модель из YAML и начать обучение с нуля
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # Начать обучение с предварительно обученной модели *.pt
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # Создать новую модель из YAML, перенести предварительно обученные веса и начать обучение
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### Формат набора данных

Формат набора данных для сегментации YOLO можно найти детально в [Руководстве по наборам данных](../../../datasets/segment/index.md). Чтобы конвертировать свой существующий набор данных из других форматов (например, COCO и т.д.) в формат YOLO, пожалуйста, используйте инструмент [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) от Ultralytics.

## Валидация

Проверьте точность обученной модели YOLOv8n-seg на наборе данных COCO128-seg. Аргументы передавать не нужно, так как `model` сохраняет `data` и аргументы обучения в качестве атрибутов модели.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-seg.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить пользовательскую модель

        # Провалидировать модель
        metrics = model.val()  # аргументы не нужны, набор данных и настройки запомнены
        metrics.box.map    # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps   # список содержит map50-95(B) каждой категории
        metrics.seg.map    # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps   # список содержит map50-95(M) каждой категории
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # валидация официальной модели
        yolo segment val model=path/to/best.pt  # валидация пользовательской модели
        ```

## Предсказание

Используйте обученную модель YOLOv8n-seg для выполнения предсказаний на изображениях.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-seg.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить пользовательскую модель

        # Сделать предсказание с помощью модели
        results = model('https://ultralytics.com/images/bus.jpg')  # предсказать по изображению
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # предсказать с официальной моделью
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # предсказать с пользовательской моделью
        ```

Полная информация о режиме `predict` на странице [Predict](https://docs.ultralytics.com/modes/predict/).

## Экспорт

Экспортируйте модель YOLOv8n-seg в другой формат, например ONNX, CoreML и т.д.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузить модель
        model = YOLO('yolov8n-seg.pt')  # загрузить официальную модель
        model = YOLO('path/to/best.pt')  # загрузить пользовательскую обученную модель

        # Экспортировать модель
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # экспортировать официальную модель
        yolo export model=path/to/best.pt format=onnx  # экспортировать пользовательскую обученную модель
        ```

Доступные форматы экспорта YOLOv8-seg приведены в таблице ниже. После завершения экспорта для вашей модели показаны примеры использования, включая прямое предсказание или валидацию на экспортированных моделях, например `yolo predict model=yolov8n-seg.onnx`.

| Формат                                                             | Аргумент `format` | Модель                        | Метаданные | Аргументы                                           |
|--------------------------------------------------------------------|-------------------|-------------------------------|------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-seg.pt`              | ✅          | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-seg.torchscript`     | ✅          | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-seg.onnx`            | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-seg_openvino_model/` | ✅          | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-seg.engine`          | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-seg.mlpackage`       | ✅          | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-seg_saved_model/`    | ✅          | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-seg.pb`              | ❌          | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-seg.tflite`          | ✅          | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-seg_edgetpu.tflite`  | ✅          | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-seg_web_model/`      | ✅          | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-seg_paddle_model/`   | ✅          | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-seg_ncnn_model/`     | ✅          | `imgsz`, `half`                                     |

Подробности о режиме `export` смотрите на странице [Export](https://docs.ultralytics.com/modes/export/).
