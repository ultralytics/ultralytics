---
comments: true
description: Изучение различных методов установки Ultralytics с использованием pip, conda, git и Docker. Освоение работы с Ultralytics через интерфейс командной строки или в рамках ваших проектов на Python.
keywords: установка Ultralytics, установка pip Ultralytics, установка Docker Ultralytics, интерфейс командной строки Ultralytics, Python интерфейс Ultralytics
---

## Установка Ultralytics

Ultralytics предлагает различные методы установки, включая pip, conda и Docker. Установите YOLOv8 через пакет `ultralytics` pip для последнего стабильного выпуска или путем клонирования [репозитория Ultralytics на GitHub](https://github.com/ultralytics/ultralytics) для получения самой актуальной версии. Docker можно использовать для выполнения пакета в изолированном контейнере, избегая локальной установки.

!!! example "Установка"

    === "Установка через Pip (рекомендуется)"
        Установите пакет `ultralytics` с помощью pip или обновите существующую установку, запустив `pip install -U ultralytics`. Посетите индекс пакетов Python (PyPI) для получения дополнительной информации о пакете `ultralytics`: [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/).

        [![Версия PyPI](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Загрузки](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # Установка пакета ultralytics из PyPI
        pip install ultralytics
        ```

        Вы также можете установить пакет `ultralytics` напрямую из [репозитория на GitHub](https://github.com/ultralytics/ultralytics). Это может быть полезно, если вы хотите получить последнюю версию для разработки. Убедитесь, что в вашей системе установлен инструмент командной строки Git. Команда `@main` устанавливает ветку `main`, которую можно изменить на другую, к примеру, `@my-branch`, или удалить полностью, чтобы по умолчанию использовалась ветка `main`.

        ```bash
        # Установка пакета ultralytics из GitHub
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```

    === "Установка через Conda"
        Conda - это альтернативный менеджер пакетов для pip, который также может быть использован для установки. Посетите Anaconda для получения дополнительной информации: [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics). Репозиторий для обновления conda пакета Ultralytics находится здесь: [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/).

        [![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Загрузки](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Версия](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda Платформы](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # Установка пакета ultralytics с помощью conda
        conda install -c conda-forge ultralytics
        ```

        !!! note

            Если вы устанавливаете пакет в среде CUDA, лучшей практикой будет установка `ultralytics`, `pytorch` и `pytorch-cuda` одной командой, чтобы менеджер пакетов conda мог разрешить любые конфликты или установить `pytorch-cuda` последним, чтобы при необходимости он мог заменить пакет `pytorch`, предназначенный для ЦП.

            ```bash
            # Установка всех пакетов вместе с помощью conda
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Образ Conda для Docker

        Образы Conda Ultralytics также доступны на [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics). Эти образы основаны на [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) и являются простым способом начать использовать `ultralytics` в среде Conda.

        ```bash
        # Установка имени образа в переменную
        t=ultralytics/ultralytics:latest-conda

        # Скачивание последнего образа ultralytics с Docker Hub
        sudo docker pull $t

        # Запуск образа ultralytics в контейнере с поддержкой GPU
        sudo docker run -it --ipc=host --gpus all $t  # все GPU
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # выбор GPU
        ```

    === "Клонирование Git"
        Клонируйте репозиторий `ultralytics`, если вы заинтересованы в участии в разработке или хотите экспериментировать с последним исходным кодом. После клонирования перейдите в каталог и установите пакет в режиме редактирования `-e` с помощью pip.

        ```bash
        # Клонирование репозитория ultralytics
        git clone https://github.com/ultralytics/ultralytics

        # Переход в клонированный каталог
        cd ultralytics

        # Установка пакета в режиме редактирования для разработки
        pip install -e .
        ```

Смотрите файл [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) `ultralytics` для списка зависимостей. Обратите внимание, что все приведенные выше примеры устанавливают все необходимые зависимости.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

!!! tip "Совет"

    Требования PyTorch зависят от операционной системы и требований CUDA, поэтому рекомендуется сначала установить PyTorch, следуя инструкциям на [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="Инструкции по установке PyTorch" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## Использование Ultralytics с CLI

Интерфейс командной строки (CLI) Ultralytics позволяет выполнять простые команды одной строкой без необходимости настройки Python среды. CLI не требует настройки или кода на Python. Все задачи можно легко выполнить из терминала с помощью команды `yolo`. Прочтите [Руководство по CLI](/../usage/cli.md), чтобы узнать больше о использовании YOLOv8 из командной строки.

!!! example

    === "Cинтаксис"

        Команды Ultralytics `yolo` используют следующий синтаксис:
        ```bash
        yolo ЗАДАЧА РЕЖИМ АРГУМЕНТЫ

        Где   ЗАДАЧА (необязательно) одна из [detect, segment, classify]
                РЕЖИМ (обязательно) один из [train, val, predict, export, track]
                АРГУМЕНТЫ (необязательно) любое количество пар 'arg=value', которые переопределяют настройки по умолчанию.
        ```
        Смотрите все АРГУМЕНТЫ в полном [Руководстве по конфигурации](/../usage/cfg.md) или с помощью `yolo cfg`

    === "Train"

        Обучение модели для детекции на 10 эпохах с начальной скоростью обучения 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Прогнозирование видео с YouTube с использованием предварительно обученной модели сегментации при размере изображения 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Валидация предварительно обученной модели детекции с размером партии 1 и размером изображения 640:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Export"

        Экспорт модели классификации YOLOv8n в формат ONNX с размером изображения 224 на 128 (TASK не требуется)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Special"

        Выполнение специальных команд для просмотра версии, настроек, запуска проверок и другого:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! warning "Предупреждение"

    Аргументы должны передаваться в виде пар `arg=val`, разделенных знаком равенства `=`, и разделены пробелами ` ` между парами. Не используйте префиксы аргументов `--` или запятые `,` между аргументами.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[Руководство по CLI](/../usage/cli.md){ .md-button .md-button--primary}

## Использование Ultralytics с Python

Python интерфейс YOLOv8 позволяет легко интегрировать его в ваши Python проекты, упрощая загрузку, выполнение и обработку результатов работы модели. Интерфейс Python разработан с акцентом на простоту и удобство использования, позволяя пользователям быстро внедрять функции обнаружения объектов, сегментации и классификации в их проектах. Это делает интерфейс Python YOLOv8 незаменимым инструментом для тех, кто хочет включить эти функции в свои Python проекты.

Например, пользователи могут загрузить модель, обучить ее, оценить ее производительность на валидационном наборе, и даже экспортировать ее в формат ONNX всего за несколько строк кода. Подробнее о том, как использовать YOLOv8 в ваших Python проектах, читайте в [Руководстве по Python](/../usage/python.md).

!!! example

    ```python
    from ultralytics import YOLO

    # Создание новой YOLO модели с нуля
    model = YOLO('yolov8n.yaml')

    # Загрузка предварительно обученной YOLO модели (рекомендуется для обучения)
    model = YOLO('yolov8n.pt')

    # Обучение модели с использованием набора данных 'coco128.yaml' на 3 эпохи
    results = model.train(data='coco128.yaml', epochs=3)

    # Оценка производительности модели на валидационном наборе
    results = model.val()

    # Выполнение обнаружения объектов на изображении с помощью модели
    results = model('https://ultralytics.com/images/bus.jpg')

    # Экспорт модели в формат ONNX
    success = model.export(format='onnx')
    ```

[Руководство по Python](/../usage/python.md){.md-button .md-button--primary}
