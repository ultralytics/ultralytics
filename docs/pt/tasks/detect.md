---
comments: true
description: Documentação oficial do YOLOv8 por Ultralytics. Aprenda como treinar, validar, predizer e exportar modelos em vários formatos. Incluindo estatísticas detalhadas de desempenho.
keywords: YOLOv8, Ultralytics, detecção de objetos, modelos pré-treinados, treinamento, validação, predição, exportação de modelos, COCO, ImageNet, PyTorch, ONNX, CoreML
---

# Detecção de Objetos

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418624-5785cb93-74c9-4541-9179-d5c6782d491a.png" alt="Exemplos de detecção de objetos">

Detecção de objetos é uma tarefa que envolve identificar a localização e a classe de objetos em uma imagem ou fluxo de vídeo.

A saída de um detector de objetos é um conjunto de caixas delimitadoras que cercam os objetos na imagem, junto com rótulos de classe e pontuações de confiança para cada caixa. A detecção de objetos é uma boa escolha quando você precisa identificar objetos de interesse em uma cena, mas não precisa saber exatamente onde o objeto está ou seu formato exato.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/5ku7npMrW40?si=6HQO1dDXunV8gekh"
    title="Reprodutor de vídeo do YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Detecção de Objetos com Modelo Pre-treinado Ultralytics YOLOv8.
</p>

!!! Tip "Dica"

    Os modelos YOLOv8 Detect são os modelos padrão do YOLOv8, ou seja, `yolov8n.pt` e são pré-treinados no [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Os modelos pré-treinados YOLOv8 Detect são mostrados aqui. Os modelos Detect, Segment e Pose são pré-treinados no dataset [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), enquanto os modelos Classify são pré-treinados no dataset [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

Os [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) são baixados automaticamente a partir do último lançamento da Ultralytics [release](https://github.com/ultralytics/assets/releases) no primeiro uso.

| Modelo                                                                               | Tamanho<br><sup>(pixels) | mAP<sup>val<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | Parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------|--------------------------|----------------------|-------------------------------------|------------------------------------------|------------------------|-------------------|
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                      | 37.3                 | 80.4                                | 0.99                                     | 3.2                    | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                      | 44.9                 | 128.4                               | 1.20                                     | 11.2                   | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                      | 50.2                 | 234.7                               | 1.83                                     | 25.9                   | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                      | 52.9                 | 375.2                               | 2.39                                     | 43.7                   | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                      | 53.9                 | 479.1                               | 3.53                                     | 68.2                   | 257.8             |

- Os valores de **mAP<sup>val</sup>** são para um único modelo e uma única escala no dataset [COCO val2017](http://cocodataset.org).
  <br>Reproduza usando `yolo val detect data=coco.yaml device=0`
- A **Velocidade** é média tirada sobre as imagens do COCO val num [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instância.
  <br>Reproduza usando `yolo val detect data=coco128.yaml batch=1 device=0|cpu`

## Treinar

Treine o YOLOv8n no dataset COCO128 por 100 épocas com tamanho de imagem 640. Para uma lista completa de argumentos disponíveis, veja a página [Configuração](/../usage/cfg.md).

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.yaml')  # construir um novo modelo pelo YAML
        model = YOLO('yolov8n.pt')  # carregar um modelo pré-treinado (recomendado para treinamento)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # construir pelo YAML e transferir pesos

        # Treinar o modelo
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Construir um novo modelo pelo YAML e começar o treinamento do zero
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Começar o treinamento a partir de um modelo pré-treinado *.pt
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Construir um novo modelo pelo YAML, transferir pesos pré-treinados e começar o treinamento
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### Formato do Dataset

O formato do dataset de detecção do YOLO pode ser encontrado em detalhes no [Guia de Datasets](../../../datasets/detect/index.md). Para converter seu dataset existente de outros formatos (como COCO, etc.) para o formato YOLO, por favor utilize a ferramenta [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) da Ultralytics.

## Validar

Valide a precisão do modelo YOLOv8n treinado no dataset COCO128. Não é necessário passar nenhum argumento, pois o `modelo` mantém seus `dados` de treino e argumentos como atributos do modelo.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.pt')  # carregar um modelo oficial
        model = YOLO('caminho/para/best.pt')  # carregar um modelo personalizado

        # Validar o modelo
        metrics = model.val()  # sem a necessidade de argumentos, dataset e configurações lembradas
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # uma lista contém map50-95 de cada categoria
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # validação do modelo oficial
        yolo detect val model=caminho/para/best.pt  # validação do modelo personalizado
        ```

## Predizer

Use um modelo YOLOv8n treinado para fazer predições em imagens.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.pt')  # carregar um modelo oficial
        model = YOLO('caminho/para/best.pt')  # carregar um modelo personalizado

        # Predizer com o modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # predizer em uma imagem
        ```
    === "CLI"

        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # predizer com modelo oficial
        yolo detect predict model=caminho/para/best.pt source='https://ultralytics.com/images/bus.jpg'  # predizer com modelo personalizado
        ```

Veja os detalhes completos do modo `predict` na página [Predição](https://docs.ultralytics.com/modes/predict/).

## Exportar

Exporte um modelo YOLOv8n para um formato diferente, como ONNX, CoreML, etc.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.pt')  # carregar um modelo oficial
        model = YOLO('caminho/para/best.pt')  # carregar um modelo treinado personalizado

        # Exportar o modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # exportar modelo oficial
        yolo export model=caminho/para/best.pt format=onnx  # exportar modelo treinado personalizado
        ```

Os formatos de exportação YOLOv8 disponíveis estão na tabela abaixo. Você pode fazer predições ou validar diretamente em modelos exportados, ou seja, `yolo predict model=yolov8n.onnx`. Exemplos de uso são mostrados para o seu modelo após a exportação ser concluída.

| Formato                                                            | Argumento `format` | Modelo                    | Metadados | Argumentos                                          |
|--------------------------------------------------------------------|--------------------|---------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Veja os detalhes completos de `exportar` na página [Exportação](https://docs.ultralytics.com/modes/export/).
