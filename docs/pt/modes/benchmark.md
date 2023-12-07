---
comments: true
description: Aprenda a avaliar a velocidade e a precisão do YOLOv8 em diversos formatos de exportação; obtenha informações sobre métricas mAP50-95, accuracy_top5 e mais.
keywords: Ultralytics, YOLOv8, benchmarking, perfilagem de velocidade, perfilagem de precisão, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, formatos de exportação YOLO
---

# Benchmarking de Modelos com o Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecossistema Ultralytics YOLO e integrações">

## Introdução

Uma vez que seu modelo esteja treinado e validado, o próximo passo lógico é avaliar seu desempenho em diversos cenários do mundo real. O modo de benchmark no Ultralytics YOLOv8 serve a esse propósito, oferecendo uma estrutura robusta para avaliar a velocidade e a precisão do seu modelo em uma gama de formatos de exportação.

## Por Que o Benchmarking é Crucial?

- **Decisões Informadas:** Obtenha insights sobre o equilíbrio entre velocidade e precisão.
- **Alocação de Recursos:** Entenda como diferentes formatos de exportação se comportam em diferentes hardwares.
- **Otimização:** Aprenda qual formato de exportação oferece o melhor desempenho para o seu caso específico.
- **Eficiência de Custos:** Faça uso mais eficiente dos recursos de hardware com base nos resultados do benchmark.

### Métricas Chave no Modo de Benchmark

- **mAP50-95:** Para detecção de objetos, segmentação e estimativa de pose.
- **accuracy_top5:** Para classificação de imagens.
- **Tempo de Inferência:** Tempo levado para cada imagem em milissegundos.

### Formatos de Exportação Suportados

- **ONNX:** Para desempenho ótimo em CPU
- **TensorRT:** Para eficiência máxima em GPU
- **OpenVINO:** Para otimização em hardware Intel
- **CoreML, TensorFlow SavedModel e Mais:** Para uma variedade de necessidades de implantação.

!!! Tip "Dica"

    * Exporte para ONNX ou OpenVINO para acelerar até 3x a velocidade em CPU.
    * Exporte para TensorRT para acelerar até 5x em GPU.

## Exemplos de Uso

Execute benchmarks do YOLOv8n em todos os formatos de exportação suportados incluindo ONNX, TensorRT etc. Consulte a seção Argumentos abaixo para ver uma lista completa de argumentos de exportação.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark na GPU
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Argumentos

Argumentos como `model`, `data`, `imgsz`, `half`, `device` e `verbose` proporcionam aos usuários flexibilidade para ajustar os benchmarks às suas necessidades específicas e comparar o desempenho de diferentes formatos de exportação com facilidade.

| Chave     | Valor   | Descrição                                                                              |
|-----------|---------|----------------------------------------------------------------------------------------|
| `model`   | `None`  | caminho para o arquivo do modelo, ou seja, yolov8n.pt, yolov8n.yaml                    |
| `data`    | `None`  | caminho para o YAML com dataset de benchmarking (sob o rótulo `val`)                   |
| `imgsz`   | `640`   | tamanho da imagem como um escalar ou lista (h, w), ou seja, (640, 480)                 |
| `half`    | `False` | quantização FP16                                                                       |
| `int8`    | `False` | quantização INT8                                                                       |
| `device`  | `None`  | dispositivo para execução, ou seja, dispositivo cuda=0 ou device=0,1,2,3 ou device=cpu |
| `verbose` | `False` | não continuar em erro (bool), ou limiar mínimo para val (float)                        |

## Formatos de Exportação

Os benchmarks tentarão executar automaticamente em todos os possíveis formatos de exportação listados abaixo.

| Formato                                                               | Argumento `format` | Modelo                    | Metadados | Argumentos                                          |
|-----------------------------------------------------------------------|--------------------|---------------------------|-----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                       | -                  | `yolov8n.pt`              | ✅         | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)               | `torchscript`      | `yolov8n.torchscript`     | ✅         | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                              | `onnx`             | `yolov8n.onnx`            | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)                | `openvino`         | `yolov8n_openvino_model/` | ✅         | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                     | `engine`           | `yolov8n.engine`          | ✅         | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                        | `coreml`           | `yolov8n.mlpackage`       | ✅         | `imgsz`, `half`, `int8`, `nms`                      |
| [Modelo Salvo do TF](https://www.tensorflow.org/guide/saved_model)    | `saved_model`      | `yolov8n_saved_model/`    | ✅         | `imgsz`, `keras`                                    |
| [GraphDef do TF](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n.pb`              | ❌         | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                            | `tflite`           | `yolov8n.tflite`          | ✅         | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)            | `edgetpu`          | `yolov8n_edgetpu.tflite`  | ✅         | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                                | `tfjs`             | `yolov8n_web_model/`      | ✅         | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                       | `paddle`           | `yolov8n_paddle_model/`   | ✅         | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                               | `ncnn`             | `yolov8n_ncnn_model/`     | ✅         | `imgsz`, `half`                                     |

Veja os detalhes completos de `exportação` na página [Export](https://docs.ultralytics.com/modes/export/).
