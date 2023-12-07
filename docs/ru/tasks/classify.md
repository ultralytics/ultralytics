---
comments: true
description: Узнайте о моделях классификации изображений YOLOv8 Classify. Получите подробную информацию о списке предварительно обученных моделей и как провести Обучение, Валидацию, Предсказание и Экспорт моделей.
keywords: Ultralytics, YOLOv8, классификация изображений, предварительно обученные модели, YOLOv8n-cls, обучение, валидация, предсказание, экспорт модели
---

# Классификация изображений

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="Примеры классификации изображений">

Классификация изображений - это самая простая из трех задач и заключается в классификации всего изображения по одному из предварительно определенных классов.

Выход классификатора изображений - это один классовый ярлык и уровень доверия. Классификация изображений полезна, когда вам нужно знать только к какому классу относится изображение, и не нужно знать, где находятся объекты данного класса или какова их точная форма.

!!! Tip "Совет"

    Модели YOLOv8 Classify используют суффикс `-cls`, например `yolov8n-cls.pt`, и предварительно обучены на [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Здесь показаны предварительно обученные модели классификации YOLOv8. Модели для обнаружения, сегментации и позы обучаются на наборе данных [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), в то время как модели классификации обучаются на наборе данных [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Модели](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) автоматически загружаются из последнего релиза Ultralytics [release](https://github.com/ultralytics/assets/releases) при первом использовании.

| Модель                                                                                       | Размер<br><sup>(пиксели) | Точность<br><sup>top1 | Точность<br><sup>top5 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | Параметры<br><sup>(М) | FLOPs<br><sup>(Б) на 640 |
|----------------------------------------------------------------------------------------------|--------------------------|-----------------------|-----------------------|-----------------------------------|----------------------------------------|-----------------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                      | 66.6                  | 87.0                  | 12.9                              | 0.31                                   | 2.7                   | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                      | 72.3                  | 91.1                  | 23.4                              | 0.35                                   | 6.4                   | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                      | 76.4                  | 93.2                  | 85.4                              | 0.62                                   | 17.0                  | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                      | 78.0                  | 94.1                  | 163.0                             | 0.87                                   | 37.5                  | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                      | 78.4                  | 94.3                  | 232.0                             | 1.01                                   | 57.4                  | 154.8                    |

- Значения **точность** указывают на точность модели на валидационном наборе данных [ImageNet](https://www.image-net.org/).
  <br>Повторить результаты можно с помощью `yolo val classify data=path/to/ImageNet device=0`.
- **Скорость** усреднена по изображениям для валидации ImageNet, используя инстанс [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Повторить результаты можно с помощью `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`.

## Обучение

Обучите модель YOLOv8n-cls на наборе данных MNIST160 на протяжении 100 эпох с размером изображения 64. Полный список доступных аргументов приведен на странице [Конфигурация](/../usage/cfg.md).

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n-cls.yaml')  # создайте новую модель из YAML
        model = YOLO('yolov8n-cls.pt')    # загрузите предварительно обученную модель (рекомендуется для обучения)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # создайте из YAML и перенесите веса

        # Обучите модель
        результаты = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Создайте новую модель из YAML и начните обучение с нуля
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # Начните обучение с предварительно обученной *.pt модели
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # Создайте новую модель из YAML, перенесите предварительно обученные веса и начните обучение
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### Формат набора данных

Формат набора данных для классификации YOLO можно подробно изучить в [Руководстве по наборам данных](../../../datasets/classify/index.md).

## Валидация

Проверьте точность обученной модели YOLOv8n-cls на наборе данных MNIST160. Не нужно передавать какие-либо аргументы, так как `model` сохраняет свои `data` и аргументы в качестве атрибутов модели.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n-cls.pt')  # загрузите официальную модель
        model = YOLO('path/to/best.pt')  # загрузите собственную модель

        # Проведите валидацию модели
        метрики = model.val()  # аргументы не нужны, набор данных и настройки запомнены
        метрики.top1           # точность top1
        метрики.top5           # точность top5
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # валидация официальной модели
        yolo classify val model=path/to/best.pt  # валидация собственной модели
        ```

## Предсказание

Используйте обученную модель YOLOv8n-cls для выполнения предсказаний на изображениях.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n-cls.pt')  # загрузите официальную модель
        model = YOLO('path/to/best.pt')  # загрузите собственную модель

        # Сделайте предсказание с помощью модели
        результаты = model('https://ultralytics.com/images/bus.jpg')  # сделайте предсказание на изображении
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # предсказание с официальной моделью
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # предсказание с собственной моделью
        ```

Подробная информация о режиме `predict` приведена на странице [Предсказание](https://docs.ultralytics.com/modes/predict/).

## Экспорт

Экспортируйте модель YOLOv8n-cls в другой формат, например, ONNX, CoreML и т. д.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n-cls.pt')  # загрузите официальную модель
        model = YOLO('path/to/best.pt')  # загрузите собственную обученную модель

        # Экспортируйте модель
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # экспорт официальной модели
        yolo export model=path/to/best.pt format=onnx  # экспорт собственной обученной модели
        ```

Доступные форматы экспорта YOLOv8-cls представлены в таблице ниже. Вы можете выполнять предсказания или валидацию прямо на экспортированных моделях, например, `yolo predict model=yolov8n-cls.onnx`. Примеры использования показаны для вашей модели после завершения экспорта.

| Формат                                                             | Аргумент `format` | Модель                        | Метаданные | Аргументы                                           |
|--------------------------------------------------------------------|-------------------|-------------------------------|------------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-cls.pt`              | ✅          | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-cls.torchscript`     | ✅          | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-cls.onnx`            | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-cls_openvino_model/` | ✅          | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-cls.engine`          | ✅          | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-cls.mlpackage`       | ✅          | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-cls_saved_model/`    | ✅          | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-cls.pb`              | ❌          | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-cls.tflite`          | ✅          | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-cls_edgetpu.tflite`  | ✅          | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-cls_web_model/`      | ✅          | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-cls_paddle_model/`   | ✅          | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-cls_ncnn_model/`     | ✅          | `imgsz`, `half`                                     |

Подробная информация об экспорте приведена на странице [Экспорт](https://docs.ultralytics.com/modes/export/).
