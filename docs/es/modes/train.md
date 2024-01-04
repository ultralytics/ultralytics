---
comments: true
description: Guía paso a paso para entrenar modelos YOLOv8 con Ultralytics YOLO incluyendo ejemplos de entrenamiento con una sola GPU y múltiples GPUs
keywords: Ultralytics, YOLOv8, YOLO, detección de objetos, modo de entrenamiento, conjunto de datos personalizado, entrenamiento GPU, multi-GPU, hiperparámetros, ejemplos CLI, ejemplos Python
---

# Entrenamiento de Modelos con Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecosistema e integraciones de Ultralytics YOLO">

## Introducción

Entrenar un modelo de aprendizaje profundo implica alimentarlo con datos y ajustar sus parámetros para que pueda hacer predicciones precisas. El modo de entrenamiento en Ultralytics YOLOv8 está diseñado para un entrenamiento efectivo y eficiente de modelos de detección de objetos, aprovechando al máximo las capacidades del hardware moderno. Esta guía tiene como objetivo cubrir todos los detalles que necesita para comenzar a entrenar sus propios modelos utilizando el robusto conjunto de características de YOLOv8.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="Reproductor de video de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ver:</strong> Cómo Entrenar un modelo YOLOv8 en Tu Conjunto de Datos Personalizado en Google Colab.
</p>

## ¿Por Qué Elegir Ultralytics YOLO para Entrenamiento?

Aquí hay algunas razones convincentes para optar por el modo Entrenamiento de YOLOv8:

- **Eficiencia:** Aprovecha al máximo tu hardware, ya sea en una configuración de una sola GPU o escalando entre múltiples GPUs.
- **Versatilidad:** Entrena con conjuntos de datos personalizados además de los ya disponibles como COCO, VOC e ImageNet.
- **Amigable al Usuario:** Interfaces CLI y Python simples pero potentes para una experiencia de entrenamiento sencilla.
- **Flexibilidad de Hiperparámetros:** Una amplia gama de hiperparámetros personalizables para ajustar el rendimiento del modelo.

### Características Clave del Modo Entrenamiento

Las siguientes son algunas características notables del modo Entrenamiento de YOLOv8:

- **Descarga Automática de Conjuntos de Datos:** Conjuntos de datos estándar como COCO, VOC e ImageNet se descargan automáticamente en el primer uso.
- **Soporte Multi-GPU:** Escala tus esfuerzos de entrenamiento sin problemas en múltiples GPUs para acelerar el proceso.
- **Configuración de Hiperparámetros:** La opción de modificar hiperparámetros a través de archivos de configuración YAML o argumentos CLI.
- **Visualización y Monitoreo:** Seguimiento en tiempo real de métricas de entrenamiento y visualización del proceso de aprendizaje para una mejor comprensión.

!!! Tip "Consejo"

    * Los conjuntos de datos de YOLOv8 como COCO, VOC, ImageNet y muchos otros se descargan automáticamente en el primer uso, es decir, `yolo train data=coco.yaml`

## Ejemplos de Uso

Entrena YOLOv8n en el conjunto de datos COCO128 durante 100 épocas con un tamaño de imagen de 640. El dispositivo de entrenamiento se puede especificar usando el argumento `device`. Si no se pasa ningún argumento, se usará la GPU `device=0` si está disponible; de lo contrario, se usará `device=cpu`. Consulta la sección de Argumentos a continuación para una lista completa de argumentos de entrenamiento.

!!! Example "Ejemplo de Entrenamiento con una sola GPU y CPU"

    El dispositivo se determina automáticamente. Si hay una GPU disponible, se usará; de lo contrario, el entrenamiento comenzará en la CPU.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.yaml')  # construir un modelo nuevo desde YAML
        model = YOLO('yolov8n.pt')    # cargar un modelo preentrenado (recomendado para entrenamiento)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # construir desde YAML y transferir pesos

        # Entrenar el modelo
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Construir un modelo nuevo desde YAML y comenzar el entrenamiento desde cero
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Comenzar el entrenamiento desde un modelo preentrenado *.pt
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Construir un modelo nuevo desde YAML, transferir pesos preentrenados a él y comenzar el entrenamiento
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Entrenamiento Multi-GPU

El entrenamiento Multi-GPU permite una utilización más eficiente de los recursos de hardware disponibles, distribuyendo la carga de entrenamiento en varias GPUs. Esta característica está disponible tanto a través de la API de Python como de la interfaz de línea de comandos. Para habilitar el entrenamiento Multi-GPU, especifica los IDs de los dispositivos GPU que deseas usar.

!!! Example "Ejemplo de Entrenamiento Multi-GPU"

    Para entrenar con 2 GPUs, dispositivos CUDA 0 y 1, usa los siguientes comandos. Amplía a GPUs adicionales según sea necesario.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # cargar un modelo preentrenado (recomendado para entrenamiento)

        # Entrenar el modelo con 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # Comenzar el entrenamiento desde un modelo preentrenado *.pt usando las GPUs 0 y 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### Entrenamiento con Apple M1 y M2 MPS

Con el soporte para los chips Apple M1 y M2 integrados en los modelos Ultralytics YOLO, ahora es posible entrenar tus modelos en dispositivos que utilizan el potente marco de Metal Performance Shaders (MPS). El MPS ofrece una forma de alto rendimiento para ejecutar tareas de cálculo y procesamiento de imágenes en el silicio personalizado de Apple.

Para habilitar el entrenamiento en chips Apple M1 y M2, debes especificar 'mps' como tu dispositivo al iniciar el proceso de entrenamiento. A continuación se muestra un ejemplo de cómo podrías hacer esto en Python y a través de la línea de comandos:

!!! Example "Ejemplo de Entrenamiento MPS"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # cargar un modelo preentrenado (recomendado para entrenamiento)

        # Entrenar el modelo con 2 GPUs
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # Comenzar el entrenamiento desde un modelo preentrenado *.pt usando las GPUs 0 y 1
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

Al aprovechar el poder computacional de los chips M1/M2, esto permite un procesamiento más eficiente de las tareas de entrenamiento. Para obtener una guía más detallada y opciones de configuración avanzadas, consulta la [documentación de PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html).

## Registros (Logging)

Al entrenar un modelo YOLOv8, puedes encontrar valioso llevar un registro del rendimiento del modelo con el tiempo. Aquí es donde entra en juego el registro. Ultralytics' YOLO ofrece soporte para tres tipos de registradores: Comet, ClearML y TensorBoard.

Para usar un registrador, selecciónalo en el menú desplegable en el fragmento de código anterior y ejecútalo. El registrador elegido se instalará e inicializará.

### Comet

[Comet](https://www.comet.ml/site/) es una plataforma que permite a los científicos de datos y desarrolladores rastrear, comparar, explicar y optimizar experimentos y modelos. Ofrece funcionalidades como métricas en tiempo real, diferencias de código y seguimiento de hiperparámetros.

Para usar Comet:

!!! Example "Ejemplo"

    === "Python"
        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

Recuerda iniciar sesión en tu cuenta de Comet en su sitio web y obtener tu clave API. Necesitarás agregar esto a tus variables de entorno o tu script para registrar tus experimentos.

### ClearML

[ClearML](https://www.clear.ml/) es una plataforma de código abierto que automatiza el seguimiento de experimentos y ayuda con la compartición eficiente de recursos. Está diseñado para ayudar a los equipos a gestionar, ejecutar y reproducir su trabajo de ML de manera más eficiente.

Para usar ClearML:

!!! Example "Ejemplo"

    === "Python"
        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

Después de ejecutar este script, necesitarás iniciar sesión en tu cuenta de ClearML en el navegador y autenticar tu sesión.

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) es una herramienta de visualización para TensorFlow. Te permite visualizar tu grafo TensorFlow, trazar métricas cuantitativas sobre la ejecución de tu grafo y mostrar datos adicionales como imágenes que lo atraviesan.

Para usar TensorBoard en [Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb):

!!! Example "Ejemplo"

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # reemplazar con el directorio 'runs'
        ```

Para usar TensorBoard localmente, ejecuta el siguiente comando y visualiza los resultados en http://localhost:6006/.

!!! Example "Ejemplo"

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # reemplazar con el directorio 'runs'
        ```

Esto cargará TensorBoard y lo dirigirá al directorio donde se guardan tus registros de entrenamiento.

Después de configurar tu registrador, puedes proceder con tu entrenamiento de modelo. Todas las métricas de entrenamiento se registrarán automáticamente en la plataforma elegida y podrás acceder a estos registros para monitorear el rendimiento de tu modelo con el tiempo, comparar diferentes modelos e identificar áreas de mejora.
