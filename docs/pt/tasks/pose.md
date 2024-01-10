---
comments: true
description: Aprenda a usar o Ultralytics YOLOv8 para tarefas de estimativa de pose. Encontre modelos pré-treinados, aprenda a treinar, validar, prever e exportar seu próprio modelo.
keywords: Ultralytics, YOLO, YOLOv8, estimativa de pose, detecção de pontos-chave, detecção de objetos, modelos pré-treinados, aprendizado de máquina, inteligência artificial
---

# Estimativa de Pose

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="Exemplos de estimativa de pose">

A estimativa de pose é uma tarefa que envolve identificar a localização de pontos específicos em uma imagem, geralmente referidos como pontos-chave. Os pontos-chave podem representar várias partes do objeto como articulações, pontos de referência ou outras características distintas. As localizações dos pontos-chave são geralmente representadas como um conjunto de coordenadas 2D `[x, y]` ou 3D `[x, y, visível]`.

A saída de um modelo de estimativa de pose é um conjunto de pontos que representam os pontos-chave em um objeto na imagem, geralmente junto com os escores de confiança para cada ponto. A estimativa de pose é uma boa escolha quando você precisa identificar partes específicas de um objeto em uma cena, e sua localização relativa entre si.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="Reprodutor de vídeo YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Estimativa de Pose com Ultralytics YOLOv8.
</p>

!!! Tip "Dica"

    Modelos YOLOv8 _pose_ usam o sufixo `-pose`, isto é `yolov8n-pose.pt`. Esses modelos são treinados no conjunto de dados [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) e são adequados para uma variedade de tarefas de estimativa de pose.

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Os modelos YOLOv8 Pose pré-treinados são mostrados aqui. Os modelos Detect, Segment e Pose são pré-treinados no conjunto de dados [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), enquanto os modelos Classify são pré-treinados no conjunto de dados [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) são baixados automaticamente do último lançamento da Ultralytics [release](https://github.com/ultralytics/assets/releases) no primeiro uso.

| Modelo                                                                                               | tamanho<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|--------------------------|-----------------------|--------------------|-------------------------------------|------------------------------------------|------------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640                      | 50.4                  | 80.1               | 131.8                               | 1.18                                     | 3.3                    | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640                      | 60.0                  | 86.2               | 233.2                               | 1.42                                     | 11.6                   | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640                      | 65.0                  | 88.8               | 456.3                               | 2.00                                     | 26.4                   | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640                      | 67.6                  | 90.0               | 784.5                               | 2.59                                     | 44.4                   | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640                      | 69.2                  | 90.2               | 1607.1                              | 3.73                                     | 69.4                   | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280                     | 71.6                  | 91.2               | 4088.7                              | 10.04                                    | 99.1                   | 1066.4            |

- **mAP<sup>val</sup>** valores são para um único modelo em escala única no conjunto de dados [COCO Keypoints val2017](http://cocodataset.org)
  .
  <br>Reproduza `yolo val pose data=coco-pose.yaml device=0`
- **Velocidade** média em imagens COCO val usando uma instância [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  .
  <br>Reproduza `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## Treinar

Treine um modelo YOLOv8-pose no conjunto de dados COCO128-pose.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-pose.yaml')  # construir um novo modelo a partir do YAML
        model = YOLO('yolov8n-pose.pt')  # carregar um modelo pré-treinado (recomendado para treinamento)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # construir a partir do YAML e transferir pesos

        # Treinar o modelo
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construir um novo modelo a partir do YAML e começar o treinamento do zero
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Começar treinamento de um modelo *.pt pré-treinado
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Construir um novo modelo a partir do YAML, transferir pesos pré-treinados para ele e começar o treinamento
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### Formato do conjunto de dados

O formato do conjunto de dados de pose YOLO pode ser encontrado em detalhes no [Guia de Conjuntos de Dados](../../../datasets/pose/index.md). Para converter seu conjunto de dados existente de outros formatos (como COCO etc.) para o formato YOLO, por favor, use a ferramenta [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) da Ultralytics.

## Validar

Valide a acurácia do modelo YOLOv8n-pose treinado no conjunto de dados COCO128-pose. Não é necessário passar nenhum argumento, pois o `model`
retém seus `data` de treinamento e argumentos como atributos do modelo.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-pose.pt')  # carregar um modelo oficial
        model = YOLO('caminho/para/melhor.pt')  # carregar um modelo personalizado

        # Validar o modelo
        metrics = model.val()  # nenhum argumento necessário, conjunto de dados e configurações lembradas
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # uma lista contém map50-95 de cada categoria
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # validar modelo oficial
        yolo pose val model=caminho/para/melhor.pt  # validar modelo personalizado
        ```

## Prever

Use um modelo YOLOv8n-pose treinado para executar previsões em imagens.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-pose.pt')  # carregar um modelo oficial
        model = YOLO('caminho/para/melhor.pt')  # carregar um modelo personalizado

        # Prever com o modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # prever em uma imagem
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # prever com modelo oficial
        yolo pose predict model=caminho/para/melhor.pt source='https://ultralytics.com/images/bus.jpg'  # prever com modelo personalizado
        ```

Veja detalhes completos do modo `predict` na página [Prever](https://docs.ultralytics.com/modes/predict/).

## Exportar

Exporte um modelo YOLOv8n Pose para um formato diferente como ONNX, CoreML, etc.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-pose.pt')  # carregar um modelo oficial
        model = YOLO('caminho/para/melhor.pt')  # carregar um modelo treinado personalizado

        # Exportar o modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # exportar modelo oficial
        yolo export model=caminho/para/melhor.pt format=onnx  # exportar modelo treinado personalizado
        ```

Os formatos de exportação YOLOv8-pose disponíveis estão na tabela abaixo. Você pode prever ou validar diretamente em modelos exportados, ou seja, `yolo predict model=yolov8n-pose.onnx`. Exemplos de uso são mostrados para o seu modelo após a conclusão da exportação.

| Formato                                                            | Argumento `format` | Modelo                         | Metadados | Argumentos                                          |
|--------------------------------------------------------------------|--------------------|--------------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n-pose.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n-pose.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n-pose.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n-pose_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n-pose.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n-pose.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n-pose_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n-pose.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n-pose.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n-pose_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n-pose_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n-pose_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n-pose_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Veja detalhes completos da `exportação` na página [Exportar](https://docs.ultralytics.com/modes/export/).
