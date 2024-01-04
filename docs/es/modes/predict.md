---
comments: true
description: Descubra cómo utilizar el modo predictivo de YOLOv8 para diversas tareas. Aprenda acerca de diferentes fuentes de inferencia como imágenes, videos y formatos de datos.
keywords: Ultralytics, YOLOv8, modo predictivo, fuentes de inferencia, tareas de predicción, modo de transmisión, procesamiento de imágenes, procesamiento de videos, aprendizaje automático, IA
---

# Predicción del Modelo con YOLO de Ultralytics

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecosistema de YOLO de Ultralytics e integraciones">

## Introducción

En el mundo del aprendizaje automático y la visión por computadora, el proceso de dar sentido a los datos visuales se denomina 'inferencia' o 'predicción'. YOLOv8 de Ultralytics ofrece una característica poderosa conocida como **modo predictivo** que está diseñada para inferencias de alto rendimiento y en tiempo real en una amplia gama de fuentes de datos.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="Reproductor de video de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ver:</strong> Cómo Extraer las Salidas del Modelo YOLOv8 de Ultralytics para Proyectos Personalizados.
</p>

## Aplicaciones en el Mundo Real

|                                                                Manufactura                                                                |                                                                Deportes                                                                |                                                               Seguridad                                                               |
|:-----------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| ![Detección de Repuestos de Vehículos](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![Detección de Jugadores de Fútbol](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![Detección de Caídas de Personas](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
|                                                    Detección de Repuestos de Vehículos                                                    |                                                    Detección de Jugadores de Fútbol                                                    |                                                    Detección de Caídas de Personas                                                    |

## ¿Por Qué Utilizar YOLO de Ultralytics para la Inferencia?

Estas son algunas razones para considerar el modo predictivo de YOLOv8 para sus necesidades de inferencia:

- **Versatilidad:** Capaz de realizar inferencias en imágenes, videos e incluso transmisiones en vivo.
- **Rendimiento:** Diseñado para procesamiento en tiempo real y de alta velocidad sin sacrificar precisión.
- **Facilidad de Uso:** Interfaces de Python y CLI intuitivas para una rápida implementación y pruebas.
- **Alta Personalización:** Diversos ajustes y parámetros para afinar el comportamiento de inferencia del modelo según sus requisitos específicos.

### Características Principales del Modo Predictivo

El modo predictivo de YOLOv8 está diseñado para ser robusto y versátil, y cuenta con:

- **Compatibilidad con Múltiples Fuentes de Datos:** Ya sea que sus datos estén en forma de imágenes individuales, una colección de imágenes, archivos de video o transmisiones de video en tiempo real, el modo predictivo le tiene cubierto.
- **Modo de Transmisión:** Utilice la función de transmisión para generar un generador eficiente de memoria de objetos `Results`. Active esto configurando `stream=True` en el método de llamada del predictor.
- **Procesamiento por Lotes:** La capacidad de procesar múltiples imágenes o fotogramas de video en un solo lote, acelerando aún más el tiempo de inferencia.
- **Amigable para la Integración:** Se integra fácilmente con pipelines de datos existentes y otros componentes de software, gracias a su API flexible.

Los modelos YOLO de Ultralytics devuelven ya sea una lista de objetos `Results` de Python, o un generador de objetos `Results` de Python eficiente en memoria cuando se pasa `stream=True` al modelo durante la inferencia:

!!! Example "Predict"

    === "Devolver una lista con `stream=False`"
        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # modelo YOLOv8n preentrenado

        # Ejecutar inferencia por lotes en una lista de imágenes
        results = model(['im1.jpg', 'im2.jpg'])  # devuelve una lista de objetos Results

        # Procesar lista de resultados
        for result in results:
            boxes = result.boxes  # Objeto Boxes para salidas de bbox
            masks = result.masks  # Objeto Masks para salidas de máscaras de segmentación
            keypoints = result.keypoints  # Objeto Keypoints para salidas de postura
            probs = result.probs  # Objeto Probs para salidas de clasificación
        ```

    === "Devolver un generador con `stream=True`"
        ```python
        from ultralytics import YOLO

        # Cargar un modelo
        model = YOLO('yolov8n.pt')  # modelo YOLOv8n preentrenado

        # Ejecutar inferencia por lotes en una lista de imágenes
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # devuelve un generador de objetos Results

        # Procesar generador de resultados
        for result in results:
            boxes = result.boxes  # Objeto Boxes para salidas de bbox
           .masks = result.masks  # Objeto Masks para salidas de máscaras de segmentación
            keypoints = result.keypoints  # Objeto Keypoints para salidas de postura
            probs = result.probs  # Objeto Probs para salidas de clasificación
        ```

## Fuentes de Inferencia

YOLOv8 puede procesar diferentes tipos de fuentes de entrada para la inferencia, como se muestra en la tabla a continuación. Las fuentes incluyen imágenes estáticas, transmisiones de video y varios formatos de datos. La tabla también indica si cada fuente se puede utilizar en modo de transmisión con el argumento `stream=True` ✅. El modo de transmisión es beneficioso para procesar videos o transmisiones en vivo ya que crea un generador de resultados en lugar de cargar todos los fotogramas en la memoria.

!!! Tip "Consejo"

    Utilice `stream=True` para procesar videos largos o conjuntos de datos grandes para gestionar eficientemente la memoria. Cuando `stream=False`, los resultados de todos los fotogramas o puntos de datos se almacenan en la memoria, lo que puede aumentar rápidamente y causar errores de memoria insuficiente para entradas grandes. En contraste, `stream=True` utiliza un generador, que solo mantiene los resultados del fotograma o punto de datos actual en la memoria, reduciendo significativamente el consumo de memoria y previniendo problemas de falta de memoria.

| Fuente              | Argumento                                  | Tipo           | Notas                                                                                                                           |
|---------------------|--------------------------------------------|----------------|---------------------------------------------------------------------------------------------------------------------------------|
| imagen              | `'image.jpg'`                              | `str` o `Path` | Archivo único de imagen.                                                                                                        |
| URL                 | `'https://ultralytics.com/images/bus.jpg'` | `str`          | URL a una imagen.                                                                                                               |
| captura de pantalla | `'screen'`                                 | `str`          | Captura una captura de pantalla.                                                                                                |
| PIL                 | `Image.open('im.jpg')`                     | `PIL.Image`    | Formato HWC con canales RGB.                                                                                                    |
| OpenCV              | `cv2.imread('im.jpg')`                     | `np.ndarray`   | Formato HWC con canales BGR `uint8 (0-255)`.                                                                                    |
| numpy               | `np.zeros((640,1280,3))`                   | `np.ndarray`   | Formato HWC con canales BGR `uint8 (0-255)`.                                                                                    |
| torch               | `torch.zeros(16,3,320,640)`                | `torch.Tensor` | Formato BCHW con canales RGB `float32 (0.0-1.0)`.                                                                               |
| CSV                 | `'sources.csv'`                            | `str` o `Path` | Archivo CSV que contiene rutas a imágenes, videos o directorios.                                                                |
| video ✅             | `'video.mp4'`                              | `str` o `Path` | Archivo de video en formatos como MP4, AVI, etc.                                                                                |
| directorio ✅        | `'path/'`                                  | `str` o `Path` | Ruta a un directorio que contiene imágenes o videos.                                                                            |
| glob ✅              | `'path/*.jpg'`                             | `str`          | Patrón glob para coincidir con múltiples archivos. Utilice el carácter `*` como comodín.                                        |
| YouTube ✅           | `'https://youtu.be/LNwODJXcvt4'`           | `str`          | URL a un video de YouTube.                                                                                                      |
| transmisión ✅       | `'rtsp://example.com/media.mp4'`           | `str`          | URL para protocolos de transmisión como RTSP, RTMP, TCP o una dirección IP.                                                     |
| multi-transmisión ✅ | `'list.streams'`                           | `str` o `Path` | Archivo de texto `*.streams` con una URL de transmisión por fila, es decir, 8 transmisiones se ejecutarán con tamaño de lote 8. |

A continuación se muestran ejemplos de código para usar cada tipo de fuente:

!!! Example "Fuentes de predicción"

    === "imagen"
        Ejecute inferencia en un archivo de imagen.
        ```python
        from ultralytics import YOLO

        # Cargar el modelo YOLOv8n preentrenado
        model = YOLO('yolov8n.pt')

        # Definir la ruta al archivo de imagen
        source = 'ruta/a/imagen.jpg'

        # Ejecutar inferencia en la fuente
        results = model(source)  # lista de objetos Results
        ```

    === "captura de pantalla"
        Ejecute inferencia en el contenido actual de la pantalla como captura de pantalla.
        ```python
        from ultralytics import YOLO

        # Cargar el modelo YOLOv8n preentrenado
        model = YOLO('yolov8n.pt')

        # Definir captura de pantalla actual como fuente
        source = 'screen'

        # Ejecutar inferencia en la fuente
        results = model(source)  # lista de objetos Results
        ```

    === "URL"
        Ejecute inferencia en una imagen o video alojados remotamente a través de URL.
        ```python
        from ultralytics import YOLO

        # Cargar el modelo YOLOv8n preentrenado
        model = YOLO('yolov8n.pt')

        # Definir URL remota de imagen o video
        source = 'https://ultralytics.com/images/bus.jpg'

        # Ejecutar inferencia en la fuente
        results = model(source)  # lista de objetos Results
        ```

    === "PIL"
        Ejecute inferencia en una imagen abierta con la Biblioteca de Imágenes de Python (PIL).
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # Cargar el modelo YOLOv8n preentrenado
        model = YOLO('yolov8n.pt')

        # Abrir una imagen usando PIL
        source = Image.open('ruta/a/imagen.jpg')

        # Ejecutar inferencia en la fuente
        results = model(source)  # lista de objetos Results
        ```

    === "OpenCV"
        Ejecute inferencia en una imagen leída con OpenCV.
        ```python
        import cv2
        from ultralytics import YOLO

        # Cargar el modelo YOLOv8n preentrenado
        model = YOLO('yolov8n.pt')

        # Leer una imagen usando OpenCV
        source = cv2.imread('ruta/a/imagen.jpg')

        # Ejecutar inferencia en la fuente
        results = model(source)  # lista de objetos Results
        ```

    === "numpy"
        Ejecute inferencia en una imagen representada como un array de numpy.
        ```python
        import numpy as np
        from ultralytics import YOLO

        # Cargar el modelo YOLOv8n preentrenado
        model = YOLO('yolov8n.pt')

        # Crear un array aleatorio de numpy con forma HWC (640, 640, 3) con valores en rango [0, 255] y tipo uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # Ejecutar inferencia en la fuente
        results = model(source)  # lista de objetos Results
        ```

    === "torch"
        Ejecute inferencia en una imagen representada como un tensor de PyTorch.
        ```python
        import torch
        from ultralytics import YOLO

        # Cargar el modelo YOLOv8n preentrenado
        model = YOLO('yolov8n.pt')

        # Crear un tensor aleatorio de torch con forma BCHW (1, 3, 640, 640) con valores en rango [0, 1] y tipo float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # Ejecutar inferencia en la fuente
        results = model(source)  # lista de objetos Results
        ```
