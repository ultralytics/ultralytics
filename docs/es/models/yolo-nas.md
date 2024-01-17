---
comments: true
description: Explora la documentación detallada de YOLO-NAS, un modelo de detección de objetos superior. Aprende sobre sus características, modelos pre-entrenados, uso con la API de Ultralytics Python, y más.
keywords: YOLO-NAS, Deci AI, detección de objetos, aprendizaje profundo, búsqueda de arquitectura neural, API de Ultralytics Python, modelo YOLO, modelos pre-entrenados, cuantización, optimización, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## Visión general

Desarrollado por Deci AI, YOLO-NAS es un modelo revolucionario de detección de objetos. Es el producto de una tecnología avanzada de Búsqueda de Arquitectura Neural, meticulosamente diseñada para abordar las limitaciones de los modelos YOLO anteriores. Con mejoras significativas en el soporte de cuantización y el equilibrio entre precisión y latencia, YOLO-NAS representa un gran avance en la detección de objetos.

![Ejemplo de imagen del modelo](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**Visión general de YOLO-NAS.** YOLO-NAS utiliza bloques conscientes de cuantización y cuantización selectiva para un rendimiento óptimo. El modelo, cuando se convierte en su versión cuantizada INT8, experimenta una caída mínima de precisión, una mejora significativa en comparación con otros modelos. Estos avances culminan en una arquitectura superior con capacidades de detección de objetos sin precedentes y un rendimiento sobresaliente.

### Características clave

- **Bloque básico compatible con cuantización:** YOLO-NAS introduce un nuevo bloque básico que es compatible con la cuantización, abordando una de las limitaciones significativas de los modelos YOLO anteriores.
- **Entrenamiento sofisticado y cuantización:** YOLO-NAS utiliza esquemas avanzados de entrenamiento y cuantización posterior para mejorar el rendimiento.
- **Optimización AutoNAC y pre-entrenamiento:** YOLO-NAS utiliza la optimización AutoNAC y se pre-entrena en conjuntos de datos prominentes como COCO, Objects365 y Roboflow 100. Este pre-entrenamiento lo hace extremadamente adecuado para tareas de detección de objetos en entornos de producción.

## Modelos pre-entrenados

Experimenta el poder de la detección de objetos de próxima generación con los modelos pre-entrenados de YOLO-NAS proporcionados por Ultralytics. Estos modelos están diseñados para ofrecer un rendimiento de primera clase tanto en velocidad como en precisión. Elige entre una variedad de opciones adaptadas a tus necesidades específicas:

| Modelo           | mAP   | Latencia (ms) |
|------------------|-------|---------------|
| YOLO-NAS S       | 47.5  | 3.21          |
| YOLO-NAS M       | 51.55 | 5.85          |
| YOLO-NAS L       | 52.22 | 7.87          |
| YOLO-NAS S INT-8 | 47.03 | 2.36          |
| YOLO-NAS M INT-8 | 51.0  | 3.78          |
| YOLO-NAS L INT-8 | 52.1  | 4.78          |

Cada variante del modelo está diseñada para ofrecer un equilibrio entre la Precisión Promedio de las Areas (mAP, por sus siglas en inglés) y la latencia, ayudándote a optimizar tus tareas de detección de objetos en términos de rendimiento y velocidad.

## Ejemplos de uso

Ultralytics ha facilitado la integración de los modelos YOLO-NAS en tus aplicaciones de Python a través de nuestro paquete `ultralytics`. El paquete proporciona una API de Python fácil de usar para agilizar el proceso.

Los siguientes ejemplos muestran cómo usar los modelos YOLO-NAS con el paquete `ultralytics` para inferencia y validación:

### Ejemplos de inferencia y validación

En este ejemplo validamos YOLO-NAS-s en el conjunto de datos COCO8.

!!! Example "Ejemplo"

    Este ejemplo proporciona un código simple de inferencia y validación para YOLO-NAS. Para manejar los resultados de la inferencia, consulta el modo [Predict](../modes/predict.md). Para usar YOLO-NAS con modos adicionales, consulta [Val](../modes/val.md) y [Export](../modes/export.md). El paquete `ultralytics` para YOLO-NAS no admite entrenamiento.

    === "Python"

        Los archivos de modelos pre-entrenados `*.pt` de PyTorch se pueden pasar a la clase `NAS()` para crear una instancia del modelo en Python:

        ```python
        from ultralytics import NAS

        # Carga un modelo YOLO-NAS-s pre-entrenado en COCO
        modelo = NAS('yolo_nas_s.pt')

        # Muestra información del modelo (opcional)
        modelo.info()

        # Valida el modelo en el conjunto de datos de ejemplo COCO8
        resultados = modelo.val(data='coco8.yaml')

        # Ejecuta inferencia con el modelo YOLO-NAS-s en la imagen 'bus.jpg'
        resultados = modelo('path/to/bus.jpg')
        ```

    === "CLI"

        Los comandos CLI están disponibles para ejecutar directamente los modelos:

        ```bash
        # Carga un modelo YOLO-NAS-s pre-entrenado en COCO y valida su rendimiento en el conjunto de datos de ejemplo COCO8
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # Carga un modelo YOLO-NAS-s pre-entrenado en COCO y ejecuta inferencia en la imagen 'bus.jpg'
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## Tareas y modos compatibles

Ofrecemos tres variantes de los modelos YOLO-NAS: Small (s), Medium (m) y Large (l). Cada variante está diseñada para satisfacer diferentes necesidades computacionales y de rendimiento:

- **YOLO-NAS-s**: Optimizado para entornos donde los recursos computacionales son limitados pero la eficiencia es clave.
- **YOLO-NAS-m**: Ofrece un enfoque equilibrado, adecuado para la detección de objetos de propósito general con mayor precisión.
- **YOLO-NAS-l**: Adaptados para escenarios que requieren la mayor precisión, donde los recursos computacionales son menos restrictivos.

A continuación se muestra una descripción detallada de cada modelo, incluyendo enlaces a sus pesos pre-entrenados, las tareas que admiten y su compatibilidad con diferentes modos de funcionamiento.

| Tipo de modelo | Pesos pre-entrenados                                                                          | Tareas admitidas                           | Inferencia | Validación | Entrenamiento | Exportación |
|----------------|-----------------------------------------------------------------------------------------------|--------------------------------------------|------------|------------|---------------|-------------|
| YOLO-NAS-s     | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_s.pt) | [Detección de objetos](../tasks/detect.md) | ✅          | ✅          | ❌             | ✅           |
| YOLO-NAS-m     | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_m.pt) | [Detección de objetos](../tasks/detect.md) | ✅          | ✅          | ❌             | ✅           |
| YOLO-NAS-l     | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_l.pt) | [Detección de objetos](../tasks/detect.md) | ✅          | ✅          | ❌             | ✅           |

## Citaciones y agradecimientos

Si utilizas YOLO-NAS en tu investigación o trabajo de desarrollo, por favor cita SuperGradients:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

Agradecemos al equipo de [SuperGradients](https://github.com/Deci-AI/super-gradients/) de Deci AI por sus esfuerzos en la creación y mantenimiento de este valioso recurso para la comunidad de visión por computadora. Creemos que YOLO-NAS, con su arquitectura innovadora y sus capacidades de detección de objetos superiores, se convertirá en una herramienta fundamental tanto para desarrolladores como para investigadores.

*keywords: YOLO-NAS, Deci AI, detección de objetos, aprendizaje profundo, búsqueda de arquitectura neural, API de Ultralytics Python, modelo YOLO, SuperGradients, modelos pre-entrenados, bloque básico compatible con cuantización, esquemas avanzados de entrenamiento, cuantización posterior, optimización AutoNAC, COCO, Objects365, Roboflow 100*
