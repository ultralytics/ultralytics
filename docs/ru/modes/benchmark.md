---
comments: true
description: Узнайте, как профилировать скорость и точность YOLOv8 в различных форматах экспорта; получите информацию о метриках mAP50-95, accuracy_top5 и др.
keywords: Ultralytics, YOLOv8, бенчмаркинг, профилирование скорости, профилирование точности, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, форматы экспорта YOLO
---

# Бенчмаркинг моделей с Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Экосистема и интеграции Ultralytics YOLO">

## Введение

После того, как ваша модель обучена и валидирована, следующим логическим шагом является оценка ее производительности в различных реальных сценариях. Режим бенчмаркинга в Ultralytics YOLOv8 служит этой цели, предоставляя надежный инструментарий для оценки скорости и точности вашей модели в ряде форматов экспорта.

## Почему бенчмаркинг критичен?

- **Обоснованные решения:** Получение представления о компромиссе между скоростью и точностью.
- **Распределение ресурсов:** Понимание производительности различных форматов экспорта на разном оборудовании.
- **Оптимизация:** Выяснение, какой формат экспорта предлагает лучшую производительность для вашего конкретного случая.
- **Эффективность затрат:** Сделайте использование аппаратных ресурсов более эффективным на основе результатов бенчмаркинга.

### Ключевые метрики в режиме бенчмаркинга

- **mAP50-95:** Для детектирования объектов, сегментации и оценки поз.
- **accuracy_top5:** Для классификации изображений.
- **Время инференса:** Время, затрачиваемое на каждое изображение в миллисекундах.

### Поддерживаемые форматы экспорта

- **ONNX:** Для оптимальной производительности ЦП
- **TensorRT:** Для максимальной эффективности GPU
- **OpenVINO:** Для оптимизации под аппаратное обеспечение Intel
- **CoreML, TensorFlow SavedModel и другие:** Для разнообразных потребностей развертывания.

!!! Tip "Совет"

    * Экспортируйте в ONNX или OpenVINO для ускорения процессора до 3 раз.
    * Экспортируйте в TensorRT для ускорения GPU до 5 раз.

## Примеры использования

Запустите бенчмарк YOLOv8n на всех поддерживаемых форматах экспорта, включая ONNX, TensorRT и т. д. Смотрите раздел Аргументы ниже для полного списка параметров экспорта.

!!! Example "Пример"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Бенчмарк на GPU
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Аргументы

Аргументы, такие как `model`, `data`, `imgsz`, `half`, `device` и `verbose`, предоставляют пользователям гибкость для тонкой настройки бенчмарков под их конкретные потребности и сравнения производительности различных форматов экспорта с легкостью.

| Ключ      | Значение | Описание                                                                         |
|-----------|----------|----------------------------------------------------------------------------------|
| `model`   | `None`   | путь к файлу модели, например yolov8n.pt, yolov8n.yaml                           |
| `data`    | `None`   | путь к YAML, ссылающемуся на набор данных для бенчмаркинга (под меткой `val`)    |
| `imgsz`   | `640`    | размер изображения как скаляр или список (h, w), например (640, 480)             |
| `half`    | `False`  | квантование FP16                                                                 |
| `int8`    | `False`  | квантование INT8                                                                 |
| `device`  | `None`   | устройство для запуска, например cuda device=0 или device=0,1,2,3 или device=cpu |
| `verbose` | `False`  | не продолжать при ошибке (bool), или пороговое значение для `val` (float)        |

## Форматы экспорта

Бенчмарки попытаются автоматически запустить для всех возможных форматов экспорта ниже.

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

Смотрите полную информацию о `export` на странице [Экспорт](https://docs.ultralytics.com/modes/export/).
