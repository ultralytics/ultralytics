---
comments: true
description: Explore diversos métodos para instalar Ultralytics usando pip, conda, git y Docker. Aprende cómo usar Ultralytics con la interfaz de línea de comandos o dentro de tus proyectos de Python.
keywords: instalación de Ultralytics, pip install Ultralytics, instalación de Docker Ultralytics, interfaz de línea de comandos de Ultralytics, interfaz de Python de Ultralytics
---

## Instalar Ultralytics

Ultralytics ofrece varios métodos de instalación incluyendo pip, conda y Docker. Instala YOLOv8 a través del paquete `ultralytics` de pip para la última versión estable o clonando el [repositorio de GitHub de Ultralytics](https://github.com/ultralytics/ultralytics) para obtener la versión más actualizada. Docker se puede utilizar para ejecutar el paquete en un contenedor aislado, evitando la instalación local.

!!! Example "Instalar"

    === "Instalación con Pip (recomendado)"
        Instala el paquete `ultralytics` usando pip o actualiza una instalación existente ejecutando `pip install -U ultralytics`. Visita el Índice de Paquetes de Python (PyPI) para más detalles sobre el paquete `ultralytics`: [https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/).

        [![Versión en PyPI](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Descargas](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # Instalar el paquete ultralytics desde PyPI
        pip install ultralytics
        ```

        También puedes instalar el paquete `ultralytics` directamente del [repositorio](https://github.com/ultralytics/ultralytics) en GitHub. Esto puede ser útil si quieres la última versión de desarrollo. Asegúrate de tener la herramienta de línea de comandos Git instalada en tu sistema. El comando `@main` instala la rama `main` y puede modificarse a otra rama, es decir, `@my-branch`, o eliminarse por completo para volver por defecto a la rama `main`.

        ```bash
        # Instalar el paquete ultralytics desde GitHub
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```


    === "Instalación con Conda"
        Conda es un gestor de paquetes alternativo a pip que también puede utilizarse para la instalación. Visita Anaconda para más detalles en [https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics). El repositorio de paquetes de alimentación de Ultralytics para actualizar el paquete de conda está en [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/).


        [![Receta de Conda](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Descargas de Conda](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Versión de Conda](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Plataformas de Conda](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # Instalar el paquete ultralytics usando conda
        conda install -c conda-forge ultralytics
        ```

        !!! Note "Nota"

            Si estás instalando en un entorno CUDA, la mejor práctica es instalar `ultralytics`, `pytorch` y `pytorch-cuda` en el mismo comando para permitir que el gestor de paquetes de conda resuelva cualquier conflicto, o en su defecto instalar `pytorch-cuda` al final para permitir que sobrescriba el paquete específico de CPU `pytorch` si es necesario.
            ```bash
            # Instalar todos los paquetes juntos usando conda
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Imagen Docker de Conda

        Las imágenes Docker de Conda de Ultralytics también están disponibles en [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics). Estas imágenes están basadas en [Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) y son una manera simple de comenzar a usar `ultralytics` en un entorno Conda.

        ```bash
        # Establecer el nombre de la imagen como una variable
        t=ultralytics/ultralytics:latest-conda

        # Descargar la última imagen de ultralytics de Docker Hub
        sudo docker pull $t

        # Ejecutar la imagen de ultralytics en un contenedor con soporte para GPU
        sudo docker run -it --ipc=host --gpus all $t  # todas las GPUs
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # especificar GPUs
        ```

    === "Clonar con Git"
        Clona el repositorio `ultralytics` si estás interesado en contribuir al desarrollo o deseas experimentar con el código fuente más reciente. Después de clonar, navega al directorio e instala el paquete en modo editable `-e` usando pip.
        ```bash
        # Clonar el repositorio ultralytics
        git clone https://github.com/ultralytics/ultralytics

        # Navegar al directorio clonado
        cd ultralytics

        # Instalar el paquete en modo editable para desarrollo
        pip install -e .
        ```

Consulta el archivo [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) de `ultralytics` para ver una lista de dependencias. Ten en cuenta que todos los ejemplos anteriores instalan todas las dependencias requeridas.

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

!!! Tip "Consejo"

    Los requisitos de PyTorch varían según el sistema operativo y los requisitos de CUDA, por lo que se recomienda instalar primero PyTorch siguiendo las instrucciones en [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="Instrucciones de Instalación de PyTorch" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## Usar Ultralytics con CLI

La interfaz de línea de comandos (CLI) de Ultralytics permite el uso de comandos simples de una sola línea sin la necesidad de un entorno de Python. La CLI no requiere personalización ni código Python. Puedes simplemente ejecutar todas las tareas desde el terminal con el comando `yolo`. Consulta la [Guía de CLI](/../usage/cli.md) para aprender más sobre el uso de YOLOv8 desde la línea de comandos.

!!! Example "Ejemplo"

    === "Sintaxis"

        Los comandos `yolo` de Ultralytics usan la siguiente sintaxis:
        ```bash
        yolo TAREA MODO ARGUMENTOS

        Donde   TAREA (opcional) es uno de [detectar, segmentar, clasificar]
                MODO (requerido) es uno de [train, val, predict, export, track]
                ARGUMENTOS (opcionales) son cualquier número de pares personalizados 'arg=valor' como 'imgsz=320' que sobrescriben los valores por defecto.
        ```
        Ver todos los ARGUMENTOS en la guía completa [Configuration Guide](/../usage/cfg.md) o con `yolo cfg`

    === "Entrenar"

        Entrenar un modelo de detección durante 10 épocas con una tasa de aprendizaje inicial de 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predecir"

        Predecir un video de YouTube usando un modelo de segmentación preentrenado con un tamaño de imagen de 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Validar"

        Validar un modelo de detección preentrenado con un tamaño de lote de 1 y un tamaño de imagen de 640:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Exportar"

        Exportar un modelo de clasificación YOLOv8n a formato ONNX con un tamaño de imagen de 224 por 128 (no se requiere TAREA)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Especial"

        Ejecutar comandos especiales para ver la versión, ver configuraciones, ejecutar chequeos y más:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! Warning "Advertencia"

    Los argumentos deben pasarse como pares `arg=valor`, separados por un signo igual `=` y delimitados por espacios ` ` entre pares. No utilices prefijos de argumentos `--` ni comas `,` entre los argumentos.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[Guía de CLI](/../usage/cli.md){.md-button .md-button--primary}

## Usar Ultralytics con Python

La interfaz de Python de YOLOv8 permite una integración perfecta en tus proyectos de Python, facilitando la carga, ejecución y procesamiento de la salida del modelo. Diseñada con sencillez y facilidad de uso en mente, la interfaz de Python permite a los usuarios implementar rápidamente la detección de objetos, segmentación y clasificación en sus proyectos. Esto hace que la interfaz de Python de YOLOv8 sea una herramienta invaluable para cualquier persona que busque incorporar estas funcionalidades en sus proyectos de Python.

Por ejemplo, los usuarios pueden cargar un modelo, entrenarlo, evaluar su rendimiento en un conjunto de validación e incluso exportarlo al formato ONNX con solo unas pocas líneas de código. Consulta la [Guía de Python](/../usage/python.md) para aprender más sobre el uso de YOLOv8 dentro de tus proyectos de Python.

!!! Example "Ejemplo"

    ```python
    from ultralytics import YOLO

    # Crear un nuevo modelo YOLO desde cero
    model = YOLO('yolov8n.yaml')

    # Cargar un modelo YOLO preentrenado (recomendado para entrenamiento)
    model = YOLO('yolov8n.pt')

    # Entrenar el modelo usando el conjunto de datos 'coco128.yaml' durante 3 épocas
    results = model.train(data='coco128.yaml', epochs=3)

    # Evaluar el rendimiento del modelo en el conjunto de validación
    results = model.val()

    # Realizar detección de objetos en una imagen usando el modelo
    results = model('https://ultralytics.com/images/bus.jpg')

    # Exportar el modelo al formato ONNX
    success = model.export(format='onnx')
    ```

[Guía de Python](/../usage/python.md){.md-button .md-button--primary}
