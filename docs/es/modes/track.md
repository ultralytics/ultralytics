---
comments: true
description: Aprende a utilizar Ultralytics YOLO para el seguimiento de objetos en flujos de video. Guías para usar diferentes rastreadores y personalizar la configuración del rastreador.
keywords: Ultralytics, YOLO, seguimiento de objetos, flujos de video, BoT-SORT, ByteTrack, guía de Python, guía de CLI
---

# Seguimiento de Múltiples Objetos con Ultralytics YOLO

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="Ejemplos de seguimiento de múltiples objetos">

El seguimiento de objetos en el ámbito del análisis de video es una tarea crítica que no solo identifica la ubicación y clase de objetos dentro del cuadro, sino que también mantiene una ID única para cada objeto detectado a medida que avanza el video. Las aplicaciones son ilimitadas, desde vigilancia y seguridad hasta análisis deportivos en tiempo real.

## ¿Por Qué Elegir Ultralytics YOLO para el Seguimiento de Objetos?

La salida de los rastreadores de Ultralytics es consistente con la detección de objetos estándar, pero con el valor añadido de las IDs de objetos. Esto facilita el seguimiento de objetos en flujos de video y la realización de análisis posteriores. Aquí tienes algunas razones por las que deberías considerar usar Ultralytics YOLO para tus necesidades de seguimiento de objetos:

- **Eficiencia:** Procesa flujos de video en tiempo real sin comprometer la precisión.
- **Flexibilidad:** Soporta múltiples algoritmos de seguimiento y configuraciones.
- **Facilidad de Uso:** API simple de Python y opciones CLI para una rápida integración y despliegue.
- **Personalización:** Fácil de usar con modelos YOLO entrenados a medida, permitiendo la integración en aplicaciones específicas del dominio.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="Reproductor de video de YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Ver:</strong> Detección de Objetos y Seguimiento con Ultralytics YOLOv8.
</p>

## Aplicaciones en el Mundo Real

|                                                           Transporte                                                           |                                                      Venta al por Menor                                                       |                                                        Acuicultura                                                         |
|:------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
| ![Seguimiento de Vehículos](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab) | ![Seguimiento de Personas](https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527) | ![Seguimiento de Peces](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142) |
|                                                    Seguimiento de Vehículos                                                    |                                                    Seguimiento de Personas                                                    |                                                    Seguimiento de Peces                                                    |

## Características a Simple Vista

Ultralytics YOLO extiende sus características de detección de objetos para proporcionar un seguimiento de objetos robusto y versátil:

- **Seguimiento en Tiempo Real:** Rastrea sin problemas los objetos en videos de alta frecuencia de cuadros.
- **Soporte de Múltiples Rastreadores:** Elige entre una variedad de algoritmos de seguimiento establecidos.
- **Configuraciones de Rastreador Personalizables:** Adapta el algoritmo de seguimiento para satisfacer requisitos específicos ajustando diversos parámetros.

## Rastreadores Disponibles

Ultralytics YOLO soporta los siguientes algoritmos de seguimiento. Pueden ser habilitados pasando el archivo de configuración YAML relevante como `tracker=tracker_type.yaml`:

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - Usa `botsort.yaml` para habilitar este rastreador.
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - Usa `bytetrack.yaml` para habilitar este rastreador.

El rastreador predeterminado es BoT-SORT.

## Seguimiento

Para ejecutar el rastreador en flujos de video, usa un modelo Detect, Segment o Pose entrenado tales como YOLOv8n, YOLOv8n-seg y YOLOv8n-pose.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar un modelo oficial o personalizado
        model = YOLO('yolov8n.pt')  # Cargar un modelo oficial Detect
        model = YOLO('yolov8n-seg.pt')  # Cargar un modelo oficial Segment
        model = YOLO('yolov8n-pose.pt')  # Cargar un modelo oficial Pose
        model = YOLO('path/to/best.pt')  # Cargar un modelo entrenado a medida

        # Realizar el seguimiento con el modelo
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Seguimiento con el rastreador predeterminado
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Seguimiento con el rastreador ByteTrack
        ```

    === "CLI"

        ```bash
        # Realizar seguimiento con varios modelos usando la interfaz de línea de comandos
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo oficial Detect
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo oficial Segment
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo oficial Pose
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4"  # Modelo entrenado a medida

        # Realizar seguimiento usando el rastreador ByteTrack
        yolo track model=path/to/best.pt tracker="bytetrack.yaml"
        ```

Como se puede ver en el uso anterior, el seguimiento está disponible para todos los modelos Detect, Segment y Pose ejecutados en videos o fuentes de transmisión.

## Configuración

### Argumentos de Seguimiento

La configuración de seguimiento comparte propiedades con el modo Predict, como `conf`, `iou` y `show`. Para configuraciones adicionales, consulta la página del modelo [Predict](https://docs.ultralytics.com/modes/predict/).

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Configurar los parámetros de seguimiento y ejecutar el rastreador
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # Configurar parámetros de seguimiento y ejecutar el rastreador usando la interfaz de línea de comandos
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### Selección de Rastreador

Ultralytics también te permite usar un archivo de configuración de rastreador modificado. Para hacerlo, simplemente haz una copia de un archivo de configuración de rastreador (por ejemplo, `custom_tracker.yaml`) de [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) y modifica cualquier configuración (excepto el `tracker_type`) según tus necesidades.

!!! Example "Ejemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Cargar el modelo y ejecutar el rastreador con un archivo de configuración personalizado
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # Cargar el modelo y ejecutar el rastreador con un archivo de configuración personalizado usando la interfaz de línea de comandos
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

Para obtener una lista completa de los argumentos de seguimiento, consulta la página [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers).

## Ejemplos en Python

### Bucle de Seguimiento Persistente

Aquí hay un script en Python que utiliza OpenCV (`cv2`) y YOLOv8 para ejecutar el seguimiento de objetos en fotogramas de video. Este script aún asume que ya has instalado los paquetes necesarios (`opencv-python` y `ultralytics`). El argumento `persist=True` le indica al rastreador que la imagen o fotograma actual es el siguiente en una secuencia y que espera rastros de la imagen anterior en la imagen actual.

!!! Example "Bucle de transmisión en vivo con seguimiento"

    ```python
    import cv2
    from ultralytics import YOLO

    # Cargar el modelo YOLOv8
    model = YOLO('yolov8n.pt')

    # Abrir el archivo de video
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # Bucle a través de los fotogramas del video
    while cap.isOpened():
        # Leer un fotograma del video
        success, frame = cap.read()

        if success:
            # Ejecutar seguimiento YOLOv8 en el fotograma, persistiendo los rastreos entre fotogramas
            results = model.track(frame, persist=True)

            # Visualizar los resultados en el fotograma
            annotated_frame = results[0].plot()

            # Mostrar el fotograma anotado
            cv2.imshow("Seguimiento YOLOv8", annotated_frame)

            # Romper el bucle si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Romper el bucle si se alcanza el final del video
            break

    # Liberar el objeto de captura de video y cerrar la ventana de visualización
    cap.release()
    cv2.destroyAllWindows()
    ```

Toma en cuenta el cambio de `model(frame)` a `model.track(frame)`, que habilita el seguimiento de objetos en lugar de simplemente la detección. Este script modificado ejecutará el rastreador en cada fotograma del video, visualizará los resultados y los mostrará en una ventana. El bucle puede ser terminado presionando 'q'.

## Contribuir con Nuevos Rastreadores

¿Eres experto en seguimiento de múltiples objetos y has implementado o adaptado exitosamente un algoritmo de seguimiento con Ultralytics YOLO? Te invitamos a contribuir en nuestra sección de Rastreadores en [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)! Tus aplicaciones en el mundo real y soluciones podrían ser invaluables para los usuarios que trabajan en tareas de seguimiento.

Al contribuir en esta sección, ayudarás a ampliar el alcance de las soluciones de seguimiento disponibles dentro del marco de trabajo de Ultralytics YOLO, añadiendo otra capa de funcionalidad y utilidad para la comunidad.

Para iniciar tu contribución, por favor consulta nuestra [Guía de Contribución](https://docs.ultralytics.com/help/contributing) para obtener instrucciones completas sobre cómo enviar una Solicitud de Extracción (PR) 🛠️. ¡Estamos emocionados de ver lo que traes a la mesa!

Juntos, vamos a mejorar las capacidades de seguimiento del ecosistema Ultralytics YOLO 🙏!
