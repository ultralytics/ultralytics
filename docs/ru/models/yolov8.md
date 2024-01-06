---
comments: true
description: Изучите захватывающие возможности YOLOv8, последней версии нашего детектора объектов в реальном времени! Узнайте, как передовая архитектура, предварительно обученные модели и оптимальное сочетание точности и скорости делают YOLOv8 идеальным выбором для ваших задач по обнаружению объектов.
keywords: YOLOv8, Ultralytics, детектор объектов в реальном времени, предварительно обученные модели, документация, обнаружение объектов, серия YOLO, передовая архитектура, точность, скорость
---

# YOLOv8

## Обзор

YOLOv8 - это последняя версия в серии детекторов объектов в реальном времени YOLO, обеспечивающая передовую производительность в терминах точности и скорости. Основываясь на достижениях предыдущих версий YOLO, YOLOv8 вводит новые возможности и оптимизации, делая его идеальным выбором для различных задач по обнаружению объектов в широком спектре приложений.

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## Основные возможности

- **Передовые архитектуры основы и шеи:** YOLOv8 использует передовые архитектуры основы и шеи, что приводит к улучшенному извлечению признаков и производительности обнаружения объектов.
- **Ключевая голова Ultralytics без якорей:** YOLOv8 применяет ключевую голову Ultralytics без якорей, что способствует более точному обнаружению и более эффективному процессу обнаружения по сравнению с якорными подходами.
- **Оптимальное сочетание точности и скорости:** С основным акцентом на поддержании оптимального баланса между точностью и скоростью, YOLOv8 подходит для задач обнаружения объектов в режиме реального времени в различных областях применения.
- **Разнообразие предварительно обученных моделей:** YOLOv8 предлагает ряд предварительно обученных моделей для различных задач и требований к производительности, что упрощает выбор подходящей модели для конкретного случая использования.

## Поддерживаемые задачи и режимы работы

Серия YOLOv8 предлагает разнообразные модели, каждая из которых специализирована для конкретных задач в компьютерном зрении. Эти модели разработаны для удовлетворения различных требований, от обнаружения объектов до более сложных задач, таких как сегментация экземпляров, определение позы/ключевых точек и классификация.

Каждая вариация серии YOLOv8 оптимизирована для своей соответствующей задачи, обеспечивая высокую производительность и точность. Кроме того, эти модели совместимы со множеством режимов работы, включая [Вывод](../modes/predict.md), [Проверку](../modes/val.md), [Обучение](../modes/train.md) и [Экспорт](../modes/export.md), что облегчает их использование на различных этапах развертывания и разработки.

| Модель      | Названия файлов                                                                                                | Задача                                         | Вывод | Проверка | Обучение | Экспорт |
|-------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------|-------|----------|----------|---------|
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [Обнаружение](../tasks/detect.md)              | ✅     | ✅        | ✅        | ✅       |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [Сегментация экземпляров](../tasks/segment.md) | ✅     | ✅        | ✅        | ✅       |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [Поза/ключевые точки](../tasks/pose.md)        | ✅     | ✅        | ✅        | ✅       |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [Классификация](../tasks/classify.md)          | ✅     | ✅        | ✅        | ✅       |

Данная таблица предоставляет обзор вариантов моделей YOLOv8, подчеркивая их применимость к конкретным задачам и их совместимость с различными режимами работы, такими как Вывод, Проверка, Обучение и Экспорт. Это демонстрирует гибкость и надежность серии YOLOv8, что делает их подходящими для широкого спектра приложений в компьютерном зрении.

## Показатели производительности

!!! Производительность

    === "Обнаружение (COCO)"

        См. [Документацию по обнаружению](https://docs.ultralytics.com/tasks/detect/) для примеров использования этих моделей, обученных на [COCO](https://docs.ultralytics.com/datasets/detect/coco/), включающих 80 предварительно обученных классов.

        | Модель                                                                                | размер<br><sup>(пиксели) | mAP<sup>val<br>50-95 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(М) | FLOPs<br><sup>(Б) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "Обнаружение (Open Images V7)"

        См. [Документацию по обнаружению](https://docs.ultralytics.com/tasks/detect/) для примеров использования этих моделей, обученных на [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), включающих 600 предварительно обученных классов.

        | Модель                                                                                     | размер<br><sup>(пиксели) | mAP<sup>val<br>50-95 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(М) | FLOPs<br><sup>(Б) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "Сегментация (COCO)"

        См. [Документацию по сегментации](https://docs.ultralytics.com/tasks/segment/) для примеров использования этих моделей, обученных на [COCO](https://docs.ultralytics.com/datasets/segment/coco/), включающих 80 предварительно обученных классов.

        | Модель                                                                                        | размер<br><sup>(пиксели) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(M) | FLOPs<br><sup>(Б) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "Классификация (ImageNet)"

        См. [Документацию по классификации](https://docs.ultralytics.com/tasks/classify/) для примеров использования этих моделей, обученных на [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/), включающих 1000 предварительно обученных классов.

        | Модель                                                                                        | размер<br><sup>(пиксели) | acc<br><sup>top1 | acc<br><sup>top5 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(M) | FLOPs<br><sup>(Б) при 640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "Поза (COCO)"

        См. [Документацию по оценке позы](https://docs.ultralytics.com/tasks/segment/) для примеров использования этих моделей, обученных на [COCO](https://docs.ultralytics.com/datasets/pose/coco/), включающих 1 предварительно обученный класс - 'person'.

        | Модель                                                                                                | размер<br><sup>(пиксели) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(M) | FLOPs<br><sup>(Б) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## Примеры использования

В этом примере представлены простые примеры обучения и вывода с использованием YOLOv8. Для полной документации об этих и других [режимах](../modes/index.md) см. страницы документации по [Предсказанию](../modes/predict.md),  [Обучению](../modes/train.md), [Проверке](../modes/val.md) и [Экспорту](../modes/export.md).

Обратите внимание, что приведенный ниже пример относится к моделям YOLOv8 для [Детекции](../tasks/detect.md) объектов. Для дополнительных поддерживаемых задач см. документацию по [Сегментации](../tasks/segment.md), [Классификации](../tasks/classify.md) и [Позе](../tasks/pose.md).

!!! Example "Пример"

    === "Python"

        Предварительно обученные модели PyTorch `*.pt`, а также файлы конфигурации `*.yaml` могут быть переданы классу `YOLO()` для создания экземпляра модели на Python:

        ```python
        from ultralytics import YOLO

        # Загрузите предварительно обученную модель YOLOv8n для COCO
        model = YOLO('yolov8n.pt')

        # Отобразить информацию о модели (по желанию)
        model.info()

        # Обучите модель на примере набора данных COCO8 в течение 100 эпох
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Выполните вывод с использованием модели YOLOv8n на изображении 'bus.jpg'
        results = model('путь/к/изображению/bus.jpg')
        ```

    === "CLI"

        Доступны команды CLI для прямого запуска моделей:

        ```bash
        # Загрузите предварительно обученную модель YOLOv8n для COCO и обучите ее на примере набора данных COCO8 в течение 100 эпох
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Загрузить предварительно обученную модель YOLOv8n для COCO и выполнить вывод на изображении 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Цитирование и благодарности

Если вы используете модель YOLOv8 или любое другое программное обеспечение из этого репозитория в своей работе, пожалуйста, процитируйте его в следующем формате:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

Обратите внимание, что идентификатор цифрового объекта (DOI) находится на стадии получения и будет добавлен в цитирование, как только он станет доступным. Модели YOLOv8 предоставляются под лицензией [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) и лицензией [Enterprise](https://ultralytics.com/license).
