---
comments: true
description: Пошаговое руководство по экспорту ваших моделей YOLOv8 в различные форматы, такие как ONNX, TensorRT, CoreML и другие, для развертывания. Изучите сейчас!.
keywords: YOLO, YOLOv8, Ultralytics, Экспорт модели, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, экспорт модели
---

# Экспорт модели с Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Экосистема и интеграции Ultralytics YOLO">

## Введение

Основная цель тренировки модели — её развертывание для реальных приложений. Режим экспорта в Ultralytics YOLOv8 предлагает множество вариантов для экспорта обученной модели в различные форматы, обеспечивая возможность развертывания на разных платформах и устройствах. Это исчерпывающее руководство направлено на то, чтобы провести вас через тонкости экспорта моделей, демонстрируя, как достичь максимальной совместимости и производительности.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Смотрите:</strong> Как экспортировать обученную пользовательскую модель Ultralytics YOLOv8 и запустить живое воспроизведение на веб-камере.
</p>

## Почему стоит выбрать режим экспорта YOLOv8?

- **Универсальность:** Экспорт в несколько форматов, включая ONNX, TensorRT, CoreML и другие.
- **Производительность:** Увеличение скорости на GPU до 5 раз с TensorRT и ускорение на CPU до 3 раз с ONNX или OpenVINO.
- **Совместимость:** Сделайте вашу модель универсально развертываемой в различных аппаратных и программных средах.
- **Простота использования:** Простой интерфейс командной строки и Python API для быстрого и простого экспорта моделей.

### Ключевые особенности режима экспорта

Вот некоторые из ключевых функций:

- **Экспорт одним кликом:** Простые команды для экспорта в разные форматы.
- **Пакетный экспорт:** Экспорт моделей, способных к пакетной обработке.
- **Оптимизированное предсказание:** Экспортированные модели оптимизированы для более быстрого предсказания.
- **Учебные видео:** Глубокие руководства и обучающие видео для гладкого опыта экспорта.

!!! Tip "Совет"

    * Экспортируйте в ONNX или OpenVINO для ускорения CPU до 3 раз.
    * Экспортируйте в TensorRT для увеличения скорости на GPU до 5 раз.

## Примеры использования

Экспорт модели YOLOv8n в другой формат, например ONNX или TensorRT. Смотрите раздел Аргументы ниже для полного списка аргументов экспорта.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Загрузите модель
        model = YOLO('yolov8n.pt')  # загрузка официальной модели
        model = YOLO('path/to/best.pt')  # загрузка обученной пользовательской модели

        # Экспорт модели
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # экспорт официальной модели
        yolo export model=path/to/best.pt format=onnx  # экспорт обученной пользовательской модели
        ```

## Аргументы

Настройки экспорта моделей YOLO относятся к различным конфигурациям и опциям, используемым для сохранения или экспорта модели для использования в других средах или платформах. Эти настройки могут влиять на производительность модели, размер и совместимость с разными системами. Некоторые общие настройки экспорта YOLO включают формат экспортируемого файла модели (например, ONNX, TensorFlow SavedModel), устройство, на котором будет запущена модель (например, CPU, GPU), а также наличие дополнительных функций, таких как маски или несколько меток на коробку. Другие факторы, которые могут повлиять на процесс экспорта, включают конкретное задание, для которого используется модель, и требования или ограничения целевой среды или платформы. Важно тщательно рассмотреть и настроить эти параметры, чтобы убедиться, что экспортированная модель оптимизирована для предполагаемого использования и может быть эффективно использована в целевой среде.

| Ключ        | Значение        | Описание                                                                  |
|-------------|-----------------|---------------------------------------------------------------------------|
| `format`    | `'torchscript'` | формат для экспорта                                                       |
| `imgsz`     | `640`           | размер изображения в виде скаляра или списка (h, w), например, (640, 480) |
| `keras`     | `False`         | использовать Keras для экспорта TF SavedModel                             |
| `optimize`  | `False`         | TorchScript: оптимизация для мобильных устройств                          |
| `half`      | `False`         | квантование FP16                                                          |
| `int8`      | `False`         | квантование INT8                                                          |
| `dynamic`   | `False`         | ONNX/TensorRT: динамические оси                                           |
| `simplify`  | `False`         | ONNX/TensorRT: упрощение модели                                           |
| `opset`     | `None`          | ONNX: версия набора операций (необязательный, по умолчанию последний)     |
| `workspace` | `4`             | TensorRT: размер рабочей области (ГБ)                                     |
| `nms`       | `False`         | CoreML: добавление NMS                                                    |

## Форматы экспорта

Доступные форматы экспорта YOLOv8 указаны в таблице ниже. Вы можете экспортировать в любой формат, используя аргумент `format`, например, `format='onnx'` или `format='engine'`.

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
