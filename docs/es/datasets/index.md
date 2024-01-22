---
comments: true
description: Explore diferentes conjuntos de datos de visión por computadora soportados por Ultralytics para la detección de objetos, segmentación, estimación de poses, clasificación de imágenes y seguimiento de múltiples objetos.
keywords: visión por computadora, conjuntos de datos, Ultralytics, YOLO, detección de objetos, segmentación de instancias, estimación de poses, clasificación de imágenes, seguimiento de múltiples objetos
---

# Resumen de Conjuntos de Datos

Ultralytics brinda soporte para varios conjuntos de datos para facilitar tareas de visión por computadora como detección, segmentación de instancias, estimación de poses, clasificación y seguimiento de múltiples objetos. A continuación se presenta una lista de los principales conjuntos de datos de Ultralytics, seguido por un resumen de cada tarea de visión por computadora y los respectivos conjuntos de datos.

!!! Note "Nota"

    🚧 Nuestra documentación multilingüe está actualmente en construcción y estamos trabajando arduamente para mejorarla. ¡Gracias por su paciencia! 🙏

## [Conjuntos de Datos de Detección](../../datasets/detect/index.md)

La detección de objetos con cuadros delimitadores es una técnica de visión por computadora que implica detectar y localizar objetos en una imagen dibujando un cuadro alrededor de cada objeto.

- [Argoverse](../../datasets/detect/argoverse.md): Un conjunto de datos que contiene datos de seguimiento en 3D y predicción de movimientos en entornos urbanos con anotaciones detalladas.
- [COCO](../../datasets/detect/coco.md): Un conjunto de datos a gran escala diseñado para detección de objetos, segmentación y subtitulado con más de 200 mil imágenes etiquetadas.
- [COCO8](../../datasets/detect/coco8.md): Contiene las primeras 4 imágenes de COCO train y COCO val, adecuado para pruebas rápidas.
- [Global Wheat 2020](../../datasets/detect/globalwheat2020.md): Un conjunto de datos de imágenes de cabezas de trigo recolectadas alrededor del mundo para tareas de detección y localización de objetos.
- [Objects365](../../datasets/detect/objects365.md): Un conjunto de datos a gran escala y de alta calidad para la detección de objetos con 365 categorías y más de 600 mil imágenes anotadas.
- [OpenImagesV7](../../datasets/detect/open-images-v7.md): Un conjunto de datos completo de Google con 1.7 millones de imágenes de entrenamiento y 42 mil imágenes de validación.
- [SKU-110K](../../datasets/detect/sku-110k.md): Un conjunto de datos que presenta detección de objetos densa en entornos minoristas con más de 11 mil imágenes y 1.7 millones de cuadros delimitadores.
- [VisDrone](../../datasets/detect/visdrone.md): Un conjunto de datos que contiene datos de detección de objetos y seguimiento de múltiples objetos de imágenes capturadas por drones con más de 10 mil imágenes y secuencias de video.
- [VOC](../../datasets/detect/voc.md): El conjunto de datos de Clases de Objetos Visuales de Pascal (VOC) para la detección de objetos y segmentación con 20 clases de objetos y más de 11 mil imágenes.
- [xView](../../datasets/detect/xview.md): Un conjunto de datos para la detección de objetos en imágenes aéreas con 60 categorías de objetos y más de un millón de objetos anotados.

## [Conjuntos de Datos de Segmentación de Instancias](../../datasets/segment/index.md)

La segmentación de instancias es una técnica de visión por computadora que implica identificar y localizar objetos en una imagen a nivel de píxel.

- [COCO](../../datasets/segment/coco.md): Un conjunto de datos a gran escala diseñado para tareas de detección de objetos, segmentación y subtitulado con más de 200 mil imágenes etiquetadas.
- [COCO8-seg](../../datasets/segment/coco8-seg.md): Un conjunto de datos más pequeño para tareas de segmentación de instancias, que contiene un subconjunto de 8 imágenes de COCO con anotaciones de segmentación.

## [Estimación de Poses](../../datasets/pose/index.md)

La estimación de poses es una técnica utilizada para determinar la pose del objeto en relación con la cámara o el sistema de coordenadas del mundo.

- [COCO](../../datasets/pose/coco.md): Un conjunto de datos a gran escala con anotaciones de pose humana diseñado para tareas de estimación de poses.
- [COCO8-pose](../../datasets/pose/coco8-pose.md): Un conjunto de datos más pequeño para tareas de estimación de poses, que contiene un subconjunto de 8 imágenes de COCO con anotaciones de pose humana.
- [Tiger-pose](../../datasets/pose/tiger-pose.md): Un conjunto de datos compacto que consiste en 263 imágenes centradas en tigres, anotadas con 12 puntos clave por tigre para tareas de estimación de poses.

## [Clasificación](../../datasets/classify/index.md)

La clasificación de imágenes es una tarea de visión por computadora que implica categorizar una imagen en una o más clases o categorías predefinidas basadas en su contenido visual.

- [Caltech 101](../../datasets/classify/caltech101.md): Un conjunto de datos que contiene imágenes de 101 categorías de objetos para tareas de clasificación de imágenes.
- [Caltech 256](../../datasets/classify/caltech256.md): Una versión extendida de Caltech 101 con 256 categorías de objetos y imágenes más desafiantes.
- [CIFAR-10](../../datasets/classify/cifar10.md): Un conjunto de datos de 60 mil imágenes a color de 32x32 en 10 clases, con 6 mil imágenes por clase.
- [CIFAR-100](../../datasets/classify/cifar100.md): Una versión extendida de CIFAR-10 con 100 categorías de objetos y 600 imágenes por clase.
- [Fashion-MNIST](../../datasets/classify/fashion-mnist.md): Un conjunto de datos compuesto por 70 mil imágenes en escala de grises de 10 categorías de moda para tareas de clasificación de imágenes.
- [ImageNet](../../datasets/classify/imagenet.md): Un conjunto de datos a gran escala para detección de objetos y clasificación de imágenes con más de 14 millones de imágenes y 20 mil categorías.
- [ImageNet-10](../../datasets/classify/imagenet10.md): Un subconjunto más pequeño de ImageNet con 10 categorías para experimentación y pruebas más rápidas.
- [Imagenette](../../datasets/classify/imagenette.md): Un subconjunto más pequeño de ImageNet que contiene 10 clases fácilmente distinguibles para entrenamientos y pruebas más rápidos.
- [Imagewoof](../../datasets/classify/imagewoof.md): Un subconjunto más desafiante de ImageNet que contiene 10 categorías de razas de perros para tareas de clasificación de imágenes.
- [MNIST](../../datasets/classify/mnist.md): Un conjunto de datos de 70 mil imágenes en escala de grises de dígitos escritos a mano para tareas de clasificación de imágenes.

## [Cuadros Delimitadores Orientados (OBB)](../../datasets/obb/index.md)

Los Cuadros Delimitadores Orientados (OBB) es un método en visión por computadora para detectar objetos angulados en imágenes utilizando cuadros delimitadores rotados, a menudo aplicado en imágenes aéreas y satelitales.

- [DOTAv2](../../datasets/obb/dota-v2.md): Un popular conjunto de datos de imágenes aéreas de OBB con 1.7 millones de instancias y 11,268 imágenes.

## [Seguimiento de Múltiples Objetos](../../datasets/track/index.md)

El seguimiento de múltiples objetos es una técnica de visión por computadora que implica detectar y seguir múltiples objetos a lo largo del tiempo en una secuencia de video.

- [Argoverse](../../datasets/detect/argoverse.md): Un conjunto de datos que contiene datos de seguimiento en 3D y predicción de movimientos en entornos urbanos con anotaciones detalladas para tareas de seguimiento de múltiples objetos.
- [VisDrone](../../datasets/detect/visdrone.md): Un conjunto de datos que contiene datos de detección de objetos y seguimiento de múltiples objetos de imágenes capturadas por drones con más de 10 mil imágenes y secuencias de video.

## Contribuir con Nuevos Conjuntos de Datos

Contribuir con un nuevo conjunto de datos implica varios pasos para garantizar que se alinee bien con la infraestructura existente. A continuación, se presentan los pasos necesarios:

### Pasos para Contribuir con un Nuevo Conjunto de Datos

1. **Recolectar Imágenes**: Reúne las imágenes que pertenecen al conjunto de datos. Estas podrían ser recopiladas de varias fuentes, tales como bases de datos públicas o tu propia colección.

2. **Annotar Imágenes**: Anota estas imágenes con cuadros delimitadores, segmentos o puntos clave, dependiendo de la tarea.

3. **Exportar Anotaciones**: Convierte estas anotaciones en el formato de archivo `*.txt` de YOLO que Ultralytics soporta.

4. **Organizar Conjunto de Datos**: Organiza tu conjunto de datos en la estructura de carpetas correcta. Deberías tener directorios de nivel superior `train/` y `val/`, y dentro de cada uno, un subdirectorio `images/` y `labels/`.

    ```
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

5. **Crear un Archivo `data.yaml`**: En el directorio raíz de tu conjunto de datos, crea un archivo `data.yaml` que describa el conjunto de datos, clases y otra información necesaria.

6. **Optimizar Imágenes (Opcional)**: Si deseas reducir el tamaño del conjunto de datos para un procesamiento más eficiente, puedes optimizar las imágenes usando el código a continuación. Esto no es requerido, pero se recomienda para tamaños de conjuntos de datos más pequeños y velocidades de descarga más rápidas.

7. **Comprimir Conjunto de Datos**: Comprime toda la carpeta del conjunto de datos en un archivo .zip.

8. **Documentar y PR**: Crea una página de documentación describiendo tu conjunto de datos y cómo encaja en el marco existente. Después de eso, envía una Solicitud de Extracción (PR). Consulta las [Pautas de Contribución de Ultralytics](https://docs.ultralytics.com/help/contributing) para obtener más detalles sobre cómo enviar una PR.

### Código de Ejemplo para Optimizar y Comprimir un Conjunto de Datos

!!! Example "Optimizar y Comprimir un Conjunto de Datos"

    === "Python"

    ```python
    from pathlib import Path
    from ultralytics.data.utils import compress_one_image
    from ultralytics.utils.downloads import zip_directory

    # Definir el directorio del conjunto de datos
    path = Path('ruta/al/conjunto-de-datos')

    # Optimizar imágenes en el conjunto de datos (opcional)
    for f in path.rglob('*.jpg'):
        compress_one_image(f)

    # Comprimir el conjunto de datos en 'ruta/al/conjunto-de-datos.zip'
    zip_directory(path)
    ```

Siguiendo estos pasos, puedes contribuir con un nuevo conjunto de datos que se integre bien con la estructura existente de Ultralytics.
