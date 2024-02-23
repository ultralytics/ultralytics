---
comments: true
description: Explora el revolucionario Segment Anything Model (SAM) de Ultralytics que permite la segmentación de imágenes en tiempo real. Aprende sobre su segmentación por indicación, rendimiento en la transferencia sin entrenamiento y cómo usarlo.
keywords: Ultralytics, segmentación de imágenes, Segment Anything Model, SAM, SA-1B dataset, rendimiento en tiempo real, transferencia sin entrenamiento, detección de objetos, análisis de imágenes, aprendizaje automático
---

# Segment Anything Model (SAM)

Bienvenido al frontera de la segmentación de imágenes con el Segment Anything Model, o SAM. Este modelo revolucionario ha cambiado el juego al introducir la segmentación de imágenes por indicación con rendimiento en tiempo real, estableciendo nuevos estándares en el campo.

## Introducción a SAM: Segment Anything Model

El Segment Anything Model, o SAM, es un modelo de segmentación de imágenes de vanguardia que permite la segmentación por indicación, ofreciendo una versatilidad sin igual en las tareas de análisis de imágenes. SAM forma el corazón de la iniciativa Segment Anything, un proyecto innovador que presenta un modelo, una tarea y un conjunto de datos nuevos para la segmentación de imágenes.

El diseño avanzado de SAM le permite adaptarse a nuevas distribuciones y tareas de imágenes sin conocimientos previos, una característica conocida como transferencia sin entrenamiento. Entrenado en el extenso [conjunto de datos SA-1B](https://ai.facebook.com/datasets/segment-anything/), que contiene más de mil millones de máscaras distribuidas en once millones de imágenes seleccionadas cuidadosamente, SAM ha demostrado un impresionante rendimiento en la transferencia sin entrenamiento, superando en muchos casos los resultados de supervisión completa anteriores.

![Ejemplo de imagen del conjunto de datos](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
Imágenes de ejemplo con máscaras superpuestas de nuestro nuevo conjunto de datos, SA-1B. SA-1B contiene 11 millones de imágenes diversas de alta resolución, con licencia y protección de la privacidad, y 1.1 mil millones de máscaras de segmentación de alta calidad. Estas máscaras fueron anotadas completamente automáticamente por SAM y, según las calificaciones humanas y numerosos experimentos, tienen una alta calidad y diversidad. Las imágenes se agrupan por número de máscaras por imagen para su visualización (hay aproximadamente 100 máscaras por imagen en promedio).

## Características clave del Segment Anything Model (SAM)

- **Tarea de segmentación por indicación**: SAM fue diseñado teniendo en cuenta una tarea de segmentación por indicación, lo que le permite generar máscaras de segmentación válidas a partir de cualquier indicación dada, como pistas espaciales o de texto que identifican un objeto.
- **Arquitectura avanzada**: El Segment Anything Model utiliza un potente codificador de imágenes, un codificador de indicaciones y un decodificador de máscaras ligero. Esta arquitectura única permite la indicación flexible, el cálculo de máscaras en tiempo real y la conciencia de ambigüedades en las tareas de segmentación.
- **El conjunto de datos SA-1B**: Introducido por el proyecto Segment Anything, el conjunto de datos SA-1B cuenta con más de mil millones de máscaras en once millones de imágenes. Como el conjunto de datos de segmentación más grande hasta la fecha, proporciona a SAM una fuente de datos de entrenamiento diversa y a gran escala.
- **Rendimiento en la transferencia sin entrenamiento**: SAM muestra un destacado rendimiento en la transferencia sin entrenamiento en diversas tareas de segmentación, lo que lo convierte en una herramienta lista para usar en diversas aplicaciones con una necesidad mínima de ingeniería de indicación.

Para obtener una visión más detallada del Segment Anything Model y el conjunto de datos SA-1B, visita el [sitio web de Segment Anything](https://segment-anything.com) y consulta el artículo de investigación [Segment Anything](https://arxiv.org/abs/2304.02643).

## Modelos disponibles, tareas admitidas y modos de funcionamiento

Esta tabla muestra los modelos disponibles con sus pesos pre-entrenados específicos, las tareas que admiten y su compatibilidad con diferentes modos de funcionamiento como [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md) y [Export](../modes/export.md), indicados con emojis ✅ para los modos admitidos y emojis ❌ para los modos no admitidos.

| Tipo de modelo | Pesos pre-entrenados | Tareas admitidas                                  | Inference | Validation | Training | Export |
|----------------|----------------------|---------------------------------------------------|-----------|------------|----------|--------|
| SAM base       | `sam_b.pt`           | [Segmentación de instancias](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |
| SAM large      | `sam_l.pt`           | [Segmentación de instancias](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |

## Cómo usar SAM: Versatilidad y potencia en la segmentación de imágenes

El Segment Anything Model se puede utilizar para una multitud de tareas posteriores que van más allá de sus datos de entrenamiento. Esto incluye detección de bordes, generación de propuestas de objetos, segmentación de instancias y predicción preliminar de texto a máscara. Con la ingeniería de indicación, SAM puede adaptarse rápidamente a nuevas tareas y distribuciones de datos de manera sin entrenamiento, estableciéndolo como una herramienta versátil y potente para todas tus necesidades de segmentación de imágenes.

### Ejemplo de predicción con SAM

!!! Example "Segmentar con indicaciones"

    Segmenta la imagen con las indicaciones proporcionadas.

    === "Python"

        ```python
        from ultralytics import SAM

        # Cargar un modelo
        modelo = SAM('sam_b.pt')

        # Mostrar información del modelo (opcional)
        modelo.info()

        # Ejecutar inferencia con indicaciones de bboxes
        modelo('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

        # Ejecutar inferencia con indicaciones de puntos
        modelo('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

!!! Example "Segmentar todo"

    Segmenta toda la imagen.

    === "Python"

        ```python
        from ultralytics import SAM

        # Cargar un modelo
        modelo = SAM('sam_b.pt')

        # Mostrar información del modelo (opcional)
        modelo.info()

        # Ejecutar inferencia
        modelo('ruta/hacia/imagen.jpg')
        ```

    === "CLI"

        ```bash
        # Ejecutar inferencia con un modelo SAM
        yolo predict model=sam_b.pt source=ruta/hacia/imagen.jpg
        ```

- La lógica aquí es segmentar toda la imagen si no se proporcionan indicaciones (bboxes/puntos/máscaras).

!!! Example "Ejemplo de SAMPredictor"

    De esta manera, puedes configurar una imagen una vez y ejecutar inferencia con indicaciones múltiples sin ejecutar el codificador de imágenes múltiples veces.

    === "Inferencia con indicaciones"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Crear SAMPredictor
        opciones = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(opciones=opciones)

        # Establecer imagen
        predictor.set_image("ultralytics/assets/zidane.jpg")  # establecer con archivo de imagen
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # establecer con np.ndarray
        resultados = predictor(bboxes=[439, 437, 524, 709])
        resultados = predictor(points=[900, 370], labels=[1])

        # Restablecer imagen
        predictor.reset_image()
        ```

    Segmentar todo con argumentos adicionales.

    === "Segmentar todo"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Crear SAMPredictor
        opciones = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(opciones=opciones)

        # Segmentar con argumentos adicionales
        resultados = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- Más argumentos adicionales para `Segmentar todo` en [`Referencia de Predictor/generate`](../../../reference/models/sam/predict.md).

## SAM comparado con YOLOv8

Aquí comparamos el modelo SAM más pequeño de Meta, SAM-b, con el modelo de segmentación más pequeño de Ultralytics, [YOLOv8n-seg](../tasks/segment.md):

| Modelo                                          | Tamaño                              | Parámetros                   | Velocidad (CPU)                     |
|-------------------------------------------------|-------------------------------------|------------------------------|-------------------------------------|
| SAM-b de Meta                                   | 358 MB                              | 94.7 M                       | 51096 ms/im                         |
| [MobileSAM](mobile-sam.md)                      | 40.7 MB                             | 10.1 M                       | 46122 ms/im                         |
| [FastSAM-s](fast-sam.md) con respaldo de YOLOv8 | 23.7 MB                             | 11.8 M                       | 115 ms/im                           |
| YOLOv8n-seg de Ultralytics                      | **6.7 MB** (53.4 veces más pequeño) | **3.4 M** (27.9 veces menos) | **59 ms/im** (866 veces más rápido) |

Esta comparación muestra las diferencias de órdenes de magnitud en los tamaños y velocidades de los modelos. Si bien SAM presenta capacidades únicas para la segmentación automática, no es un competidor directo de los modelos de segmentación YOLOv8, que son más pequeños, más rápidos y más eficientes.

Las pruebas se realizaron en una MacBook Apple M2 de 2023 con 16 GB de RAM. Para reproducir esta prueba:

!!! Example "Ejemplo"

    === "Python"
        ```python
        from ultralytics import FastSAM, SAM, YOLO

        # Perfil del modelo SAM-b
        modelo = SAM('sam_b.pt')
        modelo.info()
        modelo('ultralytics/assets')

        # Perfil de MobileSAM
        modelo = SAM('mobile_sam.pt')
        modelo.info()
        modelo('ultralytics/assets')

        # Perfil de FastSAM-s
        modelo = FastSAM('FastSAM-s.pt')
        modelo.info()
        modelo('ultralytics/assets')

        # Perfil de YOLOv8n-seg
        modelo = YOLO('yolov8n-seg.pt')
        modelo.info()
        modelo('ultralytics/assets')
        ```

## Auto-anotación: un camino rápido hacia conjuntos de datos de segmentación

La auto-anotación es una característica clave de SAM que permite a los usuarios generar un [conjunto de datos de segmentación](https://docs.ultralytics.com/datasets/segment) utilizando un modelo de detección pre-entrenado. Esta función permite una anotación rápida y precisa de un gran número de imágenes, evitando la necesidad de una etiquetación manual que consume mucho tiempo.

### Generar tu conjunto de datos de segmentación utilizando un modelo de detección

Para auto-anotar tu conjunto de datos con el marco de trabajo de Ultralytics, utiliza la función `auto_annotate` como se muestra a continuación:

!!! Example "Ejemplo"

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="ruta/a/las/imagenes", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

| Argumento  | Tipo                | Descripción                                                                                                           | Predeterminado |
|------------|---------------------|-----------------------------------------------------------------------------------------------------------------------|----------------|
| data       | str                 | Ruta a una carpeta que contiene las imágenes a anotar.                                                                |                |
| det_model  | str, opcional       | Modelo de detección YOLO pre-entrenado. Por defecto, 'yolov8x.pt'.                                                    | 'yolov8x.pt'   |
| sam_model  | str, opcional       | Modelo de segmentación SAM pre-entrenado. Por defecto, 'sam_b.pt'.                                                    | 'sam_b.pt'     |
| device     | str, opcional       | Dispositivo en el que ejecutar los modelos. Por defecto, una cadena vacía (CPU o GPU, si está disponible).            |                |
| output_dir | str, None, opcional | Directorio para guardar los resultados anotados. Por defecto, una carpeta 'labels' en el mismo directorio que 'data'. | None           |

La función `auto_annotate` toma la ruta de tus imágenes, con argumentos opcionales para especificar los modelos de detección y segmentación SAM pre-entrenados, el dispositivo en el que ejecutar los modelos, y el directorio de salida para guardar los resultados anotados.

La auto-anotación con modelos pre-entrenados puede reducir drásticamente el tiempo y el esfuerzo requeridos para crear conjuntos de datos de segmentación de alta calidad. Esta característica es especialmente beneficiosa para investigadores y desarrolladores que trabajan con grandes colecciones de imágenes, ya que les permite centrarse en el desarrollo y la evaluación de modelos en lugar de en la anotación manual.

## Citas y agradecimientos

Si encuentras útil SAM en tu trabajo de investigación o desarrollo, considera citar nuestro artículo:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Nos gustaría expresar nuestro agradecimiento a Meta AI por crear y mantener este valioso recurso para la comunidad de visión por computadora.

*keywords: Segment Anything, Segment Anything Model, SAM, Meta SAM, segmentación de imágenes, segmentación por indicación, rendimiento en la transferencia sin entrenamiento, conjunto de datos SA-1B, arquitectura avanzada, auto-anotación, Ultralytics, modelos pre-entrenados, SAM base, SAM large, segmentación de instancias, visión por computadora, IA, inteligencia artificial, aprendizaje automático, anotación de datos, máscaras de segmentación, modelo de detección, modelo de detección YOLO, bibtex, Meta AI.*
