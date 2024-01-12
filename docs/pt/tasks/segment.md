---
comments: true
description: Aprenda a usar modelos de segmentação de instâncias com o Ultralytics YOLO. Instruções sobre treinamento, validação, previsão de imagem e exportação de modelo.
keywords: yolov8, segmentação de instâncias, Ultralytics, conjunto de dados COCO, segmentação de imagem, detecção de objeto, treinamento de modelo, validação de modelo, previsão de imagem, exportação de modelo
---

# Segmentação de Instâncias

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418644-7df320b8-098d-47f1-85c5-26604d761286.png" alt="Exemplos de segmentação de instâncias">

A segmentação de instâncias vai além da detecção de objetos e envolve a identificação de objetos individuais em uma imagem e a sua segmentação do resto da imagem.

A saída de um modelo de segmentação de instâncias é um conjunto de máscaras ou contornos que delineiam cada objeto na imagem, juntamente com rótulos de classe e pontuações de confiança para cada objeto. A segmentação de instâncias é útil quando você precisa saber não apenas onde os objetos estão em uma imagem, mas também qual é a forma exata deles.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="Reprodutor de vídeo do YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Executar Segmentação com o Modelo Treinado Ultralytics YOLOv8 em Python.
</p>

!!! Tip "Dica"

    Modelos YOLOv8 Segment usam o sufixo `-seg`, ou seja, `yolov8n-seg.pt` e são pré-treinados no [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Os modelos Segment pré-treinados do YOLOv8 estão mostrados aqui. Os modelos Detect, Segment e Pose são pré-treinados no conjunto de dados [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), enquanto os modelos Classify são pré-treinados no conjunto de dados [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) são baixados automaticamente do último lançamento da Ultralytics [release](https://github.com/ultralytics/assets/releases) na primeira utilização.

| Modelo                                                                                       | Tamanho<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>máscara<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | Parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
|----------------------------------------------------------------------------------------------|--------------------------|----------------------|--------------------------|-------------------------------------|------------------------------------------|------------------------|-------------------|
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                      | 36.7                 | 30.5                     | 96.1                                | 1.21                                     | 3.4                    | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                      | 44.6                 | 36.8                     | 155.7                               | 1.47                                     | 11.8                   | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                      | 49.9                 | 40.8                     | 317.0                               | 2.18                                     | 27.3                   | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                      | 52.3                 | 42.6                     | 572.4                               | 2.79                                     | 46.0                   | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                      | 53.4                 | 43.4                     | 712.1                               | 4.02                                     | 71.8                   | 344.1             |

- Os valores de **mAP<sup>val</sup>** são para um único modelo em uma única escala no conjunto de dados [COCO val2017](https://cocodataset.org).
  <br>Reproduza por meio de `yolo val segment data=coco.yaml device=0`
- **Velocidade** média em imagens COCO val usando uma instância [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Reproduza por meio de `yolo val segment data=coco128-seg.yaml batch=1 device=0|cpu`

## Treinar

Treine o modelo YOLOv8n-seg no conjunto de dados COCO128-seg por 100 épocas com tamanho de imagem 640. Para uma lista completa de argumentos disponíveis, consulte a página [Configuração](/../usage/cfg.md).

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-seg.yaml')  # construir um novo modelo a partir do YAML
        model = YOLO('yolov8n-seg.pt')  # carregar um modelo pré-treinado (recomendado para treinamento)
        model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # construir a partir do YAML e transferir os pesos

        # Treinar o modelo
        results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construir um novo modelo a partir do YAML e começar o treinamento do zero
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640

        # Começar o treinamento a partir de um modelo *.pt pré-treinado
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640

        # Construir um novo modelo a partir do YAML, transferir pesos pré-treinados para ele e começar o treinamento
        yolo segment train data=coco128-seg.yaml model=yolov8n-seg.yaml pretrained=yolov8n-seg.pt epochs=100 imgsz=640
        ```

### Formato do conjunto de dados

O formato do conjunto de dados de segmentação YOLO pode ser encontrado em detalhes no [Guia de Conjuntos de Dados](../../../datasets/segment/index.md). Para converter seu conjunto de dados existente de outros formatos (como COCO etc.) para o formato YOLO, utilize a ferramenta [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) da Ultralytics.

## Val

Valide a acurácia do modelo YOLOv8n-seg treinado no conjunto de dados COCO128-seg. Não é necessário passar nenhum argumento, pois o `modelo` retém seus `dados` de treino e argumentos como atributos do modelo.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-seg.pt')  # carregar um modelo oficial
        model = YOLO('path/to/best.pt')  # carregar um modelo personalizado

        # Validar o modelo
        metrics = model.val()  # sem necessidade de argumentos, conjunto de dados e configurações são lembrados
        metrics.box.map    # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps   # uma lista contendo map50-95(B) de cada categoria
        metrics.seg.map    # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps   # uma lista contendo map50-95(M) de cada categoria
        ```
    === "CLI"

        ```bash
        yolo segment val model=yolov8n-seg.pt  # val modelo oficial
        yolo segment val model=path/to/best.pt  # val modelo personalizado
        ```

## Prever

Use um modelo YOLOv8n-seg treinado para realizar previsões em imagens.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-seg.pt')  # carregar um modelo oficial
        model = YOLO('path/to/best.pt')  # carregar um modelo personalizado

        # Realizar previsão com o modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # prever em uma imagem
        ```
    === "CLI"

        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # previsão com modelo oficial
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # previsão com modelo personalizado
        ```

Veja detalhes completos do modo `predict` na página [Prever](https://docs.ultralytics.com/modes/predict/).

## Exportar

Exporte um modelo YOLOv8n-seg para um formato diferente como ONNX, CoreML, etc.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-seg.pt')  # carregar um modelo oficial
        model = YOLO('path/to/best.pt')  # carregar um modelo treinado personalizado

        # Exportar o modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-seg.pt format=onnx  # exportar modelo oficial
        yolo export model=path/to/best.pt format=onnx  # exportar modelo treinado personalizado
        ```

Os formatos de exportação disponíveis para YOLOv8-seg estão na tabela abaixo. Você pode prever ou validar diretamente em modelos exportados, ou seja, `yolo predict model=yolov8n-seg.onnx`. Exemplos de uso são mostrados para o seu modelo após a conclusão da exportação.

| Formato                                                            | Argumento `format` | Modelo                        | Metadados | Argumentos                                          |
|--------------------------------------------------------------------|--------------------|-------------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n-seg.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n-seg.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n-seg.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n-seg_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n-seg.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n-seg.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n-seg_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n-seg.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n-seg.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n-seg_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n-seg_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n-seg_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n-seg_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Veja detalhes completos da `exportação` na página [Exportar](https://docs.ultralytics.com/modes/export/).
