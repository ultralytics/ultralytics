---
comments: true
description: Aprenda sobre modelos YOLOv8 Classify para classificação de imagens. Obtenha informações detalhadas sobre Lista de Modelos Pré-treinados e como Treinar, Validar, Prever e Exportar modelos.
keywords: Ultralytics, YOLOv8, Classificação de Imagem, Modelos Pré-treinados, YOLOv8n-cls, Treinamento, Validação, Previsão, Exportação de Modelo
---

# Classificação de Imagens

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="Exemplos de classificação de imagens">

A classificação de imagens é a tarefa mais simples das três e envolve classificar uma imagem inteira em uma de um conjunto de classes pré-definidas.

A saída de um classificador de imagem é um único rótulo de classe e uma pontuação de confiança. A classificação de imagem é útil quando você precisa saber apenas a qual classe uma imagem pertence e não precisa conhecer a localização dos objetos dessa classe ou o formato exato deles.

!!! Tip "Dica"

    Os modelos YOLOv8 Classify usam o sufixo `-cls`, ou seja, `yolov8n-cls.pt` e são pré-treinados na [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

Aqui são mostrados os modelos pré-treinados YOLOv8 Classify. Modelos de Detecção, Segmentação e Pose são pré-treinados no dataset [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), enquanto que os modelos de Classificação são pré-treinados no dataset [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

[Modelos](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) são baixados automaticamente do último lançamento da Ultralytics [release](https://github.com/ultralytics/assets/releases) no primeiro uso.

| Modelo                                                                                       | Tamanho<br><sup>(pixels) | acurácia<br><sup>top1 | acurácia<br><sup>top5 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) a 640 |
|----------------------------------------------------------------------------------------------|--------------------------|-----------------------|-----------------------|-------------------------------------|------------------------------------------|------------------------|-------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                      | 66.6                  | 87.0                  | 12.9                                | 0.31                                     | 2.7                    | 4.3                     |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                      | 72.3                  | 91.1                  | 23.4                                | 0.35                                     | 6.4                    | 13.5                    |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                      | 76.4                  | 93.2                  | 85.4                                | 0.62                                     | 17.0                   | 42.7                    |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                      | 78.0                  | 94.1                  | 163.0                               | 0.87                                     | 37.5                   | 99.7                    |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                      | 78.4                  | 94.3                  | 232.0                               | 1.01                                     | 57.4                   | 154.8                   |

- Os valores de **acc** são as acurácias dos modelos no conjunto de validação do dataset [ImageNet](https://www.image-net.org/).
  <br>Reproduza com `yolo val classify data=path/to/ImageNet device=0`
- **Velocidade** média observada sobre imagens de validação da ImageNet usando uma instância [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/).
  <br>Reproduza com `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Treino

Treine o modelo YOLOv8n-cls no dataset MNIST160 por 100 épocas com tamanho de imagem 64. Para uma lista completa de argumentos disponíveis, veja a página de [Configuração](/../usage/cfg.md).

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-cls.yaml')  # construir um novo modelo a partir do YAML
        model = YOLO('yolov8n-cls.pt')  # carregar um modelo pré-treinado (recomendado para treino)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # construir a partir do YAML e transferir pesos

        # Treinar o modelo
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Construir um novo modelo a partir do YAML e começar treino do zero
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # Começar treino de um modelo pré-treinado *.pt
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # Construir um novo modelo do YAML, transferir pesos pré-treinados e começar treino
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### Formato do dataset

O formato do dataset de classificação YOLO pode ser encontrado em detalhes no [Guia de Datasets](../../../datasets/classify/index.md).

## Val

Valide a acurácia do modelo YOLOv8n-cls treinado no dataset MNIST160. Não é necessário passar argumento, pois o `modelo` retém seus dados de treinamento e argumentos como atributos do modelo.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-cls.pt')  # carregar um modelo oficial
        model = YOLO('path/to/best.pt')  # carregar um modelo personalizado

        # Validar o modelo
        metrics = model.val()  # sem argumentos necessários, dataset e configurações lembrados
        metrics.top1   # acurácia top1
        metrics.top5   # acurácia top5
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # validar modelo oficial
        yolo classify val model=path/to/best.pt  # validar modelo personalizado
        ```

## Previsão

Use um modelo YOLOv8n-cls treinado para realizar previsões em imagens.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-cls.pt')  # carregar um modelo oficial
        model = YOLO('path/to/best.pt')  # carregar um modelo personalizado

        # Prever com o modelo
        results = model('https://ultralytics.com/images/bus.jpg')  # prever em uma imagem
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # prever com modelo oficial
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # prever com modelo personalizado
        ```

Veja detalhes completos do modo de `previsão` na página [Predict](https://docs.ultralytics.com/modes/predict/).

## Exportar

Exporte um modelo YOLOv8n-cls para um formato diferente, como ONNX, CoreML, etc.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n-cls.pt')  # carregar um modelo oficial
        model = YOLO('path/to/best.pt')  # carregar um modelo treinado personalizado

        # Exportar o modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # exportar modelo oficial
        yolo export model=path/to/best.pt format=onnx  # exportar modelo treinado personalizado
        ```

Os formatos de exportação YOLOv8-cls disponíveis estão na tabela abaixo. Você pode prever ou validar diretamente nos modelos exportados, ou seja, `yolo predict model=yolov8n-cls.onnx`. Exemplos de uso são mostrados para seu modelo após a conclusão da exportação.

| Formato                                                            | Argumento `format` | Modelo                        | Metadata | Argumentos                                          |
|--------------------------------------------------------------------|--------------------|-------------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                  | `yolov8n-cls.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`      | `yolov8n-cls.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`             | `yolov8n-cls.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`         | `yolov8n-cls_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`           | `yolov8n-cls.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`           | `yolov8n-cls.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n-cls_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n-cls.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n-cls.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n-cls_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n-cls_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`           | `yolov8n-cls_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`             | `yolov8n-cls_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

Veja detalhes completos da `exportação` na página [Export](https://docs.ultralytics.com/modes/export/).
