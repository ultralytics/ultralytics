---
comments: true
description: Познакомьтесь с YOLOv5u, улучшенной версией модели YOLOv5 с улучшенным компромиссом между точностью и скоростью и множеством готовых моделей для различных задач обнаружения объектов.
keywords: YOLOv5u, обнаружение объектов, готовые модели, Ultralytics, Вывод, Проверка, YOLOv5, YOLOv8, без якорей, без учета объектности, реальное время работы, машинное обучение
---

# YOLOv5

## Обзор

YOLOv5u представляет собой прогресс в методологиях обнаружения объектов. Исходя из основной архитектуры модели [YOLOv5](https://github.com/ultralytics/yolov5), разработанной компанией Ultralytics, YOLOv5u интегрирует разделение головы без якорей и объектности, функциональность, ранее представленную в моделях [YOLOv8](yolov8.md). Эта адаптация улучшает архитектуру модели, что приводит к улучшенному компромиссу между точностью и скоростью в задачах обнаружения объектов. Учитывая эмпирические результаты и полученные характеристики, YOLOv5u предлагает эффективную альтернативу для тех, кто ищет надежные решения как в научных исследованиях, так и в практических приложениях.

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## Основные возможности

- **Разделение головы без якорей**: Традиционные модели обнаружения объектов полагаются на заранее определенные привязочные рамки для предсказания расположения объектов. Однако YOLOv5u модернизирует этот подход. Принимая безякорную голову, она обеспечивает более гибкий и адаптивный механизм обнаружения, что в итоге повышает производительность в различных сценариях.

- **Оптимизированный компромисс между точностью и скоростью**: Скорость и точность часто движутся в противоположных направлениях. Но YOLOv5u вызывает этот компромисс. Она предлагает настроенный баланс, обеспечивая обнаружение в режиме реального времени без ущерба для точности. Эта функция особенно ценна для приложений, которым требуются быстрые ответы, таких как автономные транспортные средства, робототехника и аналитика видеозаписей в режиме реального времени.

- **Разнообразие готовых моделей**: Понимая, что различные задачи требуют разного инструментария, YOLOv5u предлагает множество готовых моделей. Независимо от того, придерживаетесь ли вы вывода, проверки или обучения, вас ожидает модель, разработанная специально под вашу уникальную задачу. Это разнообразие гарантирует, что вы не используете универсальное решение, а модель, специально настроенную для вашего уникального вызова.

## Поддерживаемые задачи и режимы

Модели YOLOv5u с различными предварительно обученными весами превосходят в задачах [Обнаружение объектов](../tasks/detect.md). Они поддерживают широкий спектр режимов работы, что делает их подходящими для разных приложений, от разработки до развертывания.

| Тип модели | Предварительно обученные веса                                                                                               | Задача                                     | Вывод | Проверка | Обучение | Экспорт |
|------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|-------|----------|----------|---------|
| YOLOv5u    | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [Обнаружение объектов](../tasks/detect.md) | ✅     | ✅        | ✅        | ✅       |

В этой таблице предоставлена подробная информация о вариантах моделей YOLOv5u, основных задачах обнаружения объектов и поддержке различных операционных режимов, таких как [Вывод](../modes/predict.md), [Проверка](../modes/val.md), [Обучение](../modes/train.md) и [Экспорт](../modes/export.md). Эта всесторонняя поддержка позволяет пользователям полностью использовать возможности моделей YOLOv5u в широком спектре задач обнаружения объектов.

## Показатели производительности

!!! Производительность

    === "Обнаружение"

    См. [Документацию по обнаружению](https://docs.ultralytics.com/tasks/detect/) для примеров использования этих моделей, обученных на [COCO](https://docs.ultralytics.com/datasets/detect/coco/), которая включает 80 предварительно обученных классов.

    | Модель                                                                                     | YAML                                                                                                           | размер<br><sup>(пиксели) | mAP<sup>val<br>50-95 | Скорость<br><sup>CPU ONNX<br>(мс) | Скорость<br><sup>A100 TensorRT<br>(мс) | параметры<br><sup>(М) | FLOPs<br><sup>(Б) |
    |-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------------|----------------------|--------------------------------|-------------------------------------|----------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5nu.pt) | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                     | 34.3                 | 73.6                           | 1.06                                | 2.6                  | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5su.pt) | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                     | 43.0                 | 120.7                          | 1.27                                | 9.1                  | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5mu.pt) | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                     | 49.0                 | 233.9                          | 1.86                                | 25.1                 | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5lu.pt) | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                     | 52.2                 | 408.4                          | 2.50                                | 53.2                 | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5xu.pt) | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                     | 53.2                 | 763.2                          | 3.81                                | 97.2                 | 246.4             |
    |                                                                                           |                                                                                                                |                         |                      |                                |                                     |                      |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                    | 42.1                 | 211.0                          | 1.83                                | 4.3                  | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                    | 48.6                 | 422.6                          | 2.34                                | 15.3                 | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                    | 53.6                 | 810.9                          | 4.36                                | 41.2                 | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                    | 55.7                 | 1470.9                         | 5.47                                | 86.1                 | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                    | 56.8                 | 2436.5                         | 8.98                                | 155.4                | 250.7             |

## Примеры использования

В этом примере приведены простые примеры обучения и вывода моделей YOLOv5. Для получения полной документации по этим и другим [режимам работы](../modes/index.md) см. страницы документации по [Predict](../modes/predict.md),  [Train](../modes/train.md), [Val](../modes/val.md) и [Export](../modes/export.md).

!!! Example "Пример"

    === "Python"

        Предварительно обученные модели PyTorch `*.pt` и файлы конфигурации `*.yaml` можно передать классу `YOLO()` для создания экземпляра модели на Python:

        ```python
        from ultralytics import YOLO

        # Загрузите предварительно обученную модель YOLOv5n на COCO
        model = YOLO('yolov5n.pt')

        # Отобразить информацию о модели (опционально)
        model.info()

        # Обучение модели на примере набора данных на основе COCO8 в течение 100 эпох
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Выполнение вывода с моделью YOLOv5n на изображении 'bus.jpg'
        results = model('путь/к/файлу/bus.jpg')
        ```

    === "CLI"

        Доступны команды CLI для непосредственного выполнения моделей:

        ```bash
        # Загрузка предварительно обученной модели YOLOv5n на COCO и обучение на примере набора данных на основе COCO8 в течение 100 эпох
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # Загрузка предварительно обученной модели YOLOv5n на COCO и выполнение вывода на изображении 'bus.jpg'
        yolo predict model=yolov5n.pt source=путь/к/файлу/bus.jpg
        ```

## Цитирование и благодарности

Если вы используете YOLOv5 или YOLOv5u в своих исследованиях, пожалуйста, ссылайтесь на репозиторий Ultralytics YOLOv5 следующим образом:

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

Пожалуйста, обратите внимание, что модели YOLOv5 предоставляются под [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) и [Enterprise](https://ultralytics.com/license) лицензиями.
