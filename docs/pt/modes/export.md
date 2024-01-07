---
comments: true
description: Guia passo a passo sobre como exportar seus modelos YOLOv8 para vários formatos como ONNX, TensorRT, CoreML e mais para implantação. Explore agora!
keywords: YOLO, YOLOv8, Ultralytics, Exportação de modelo, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, exportar modelo
---

# Exportação de Modelo com Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ecossistema Ultralytics YOLO e integrações">

## Introdução

O objetivo final de treinar um modelo é implantá-lo para aplicações no mundo real. O modo de exportação no Ultralytics YOLOv8 oferece uma ampla gama de opções para exportar seu modelo treinado para diferentes formatos, tornando-o implantável em várias plataformas e dispositivos. Este guia abrangente visa orientá-lo através das nuances da exportação de modelos, mostrando como alcançar a máxima compatibilidade e performance.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="Reprodutor de vídeo do YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Como Exportar Modelo Treinado Customizado do Ultralytics YOLOv8 e Executar Inferência ao Vivo na Webcam.
</p>

## Por Que Escolher o Modo de Exportação do YOLOv8?

- **Versatilidade:** Exporte para múltiplos formatos incluindo ONNX, TensorRT, CoreML e mais.
- **Performance:** Ganhe até 5x aceleração em GPU com TensorRT e 3x aceleração em CPU com ONNX ou OpenVINO.
- **Compatibilidade:** Torne seu modelo universalmente implantável em numerosos ambientes de hardware e software.
- **Facilidade de Uso:** Interface de linha de comando simples e API Python para exportação rápida e direta de modelos.

### Principais Recursos do Modo de Exportação

Aqui estão algumas das funcionalidades de destaque:

- **Exportação com Um Clique:** Comandos simples para exportação em diferentes formatos.
- **Exportação em Lote:** Exporte modelos capazes de inferência em lote.
- **Inferência Otimizada:** Modelos exportados são otimizados para tempos de inferência mais rápidos.
- **Vídeos Tutoriais:** Guias e tutoriais detalhados para uma experiência de exportação tranquila.

!!! Tip "Dica"

    * Exporte para ONNX ou OpenVINO para até 3x aceleração em CPU.
    * Exporte para TensorRT para até 5x aceleração em GPU.

## Exemplos de Uso

Exporte um modelo YOLOv8n para um formato diferente como ONNX ou TensorRT. Veja a seção de Argumentos abaixo para uma lista completa dos argumentos de exportação.

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Carregar um modelo
        model = YOLO('yolov8n.pt')  # carrega um modelo oficial
        model = YOLO('caminho/para/best.pt')  # carrega um modelo treinado personalizado

        # Exportar o modelo
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # exporta modelo oficial
        yolo export model=caminho/para/best.pt format=onnx  # exporta modelo treinado personalizado
        ```

## Argumentos

Configurações de exportação para modelos YOLO referem-se às várias configurações e opções usadas para salvar ou exportar o modelo para uso em outros ambientes ou plataformas. Essas configurações podem afetar a performance, tamanho e compatibilidade do modelo com diferentes sistemas. Algumas configurações comuns de exportação de YOLO incluem o formato do arquivo de modelo exportado (por exemplo, ONNX, TensorFlow SavedModel), o dispositivo em que o modelo será executado (por exemplo, CPU, GPU) e a presença de recursos adicionais como máscaras ou múltiplos rótulos por caixa. Outros fatores que podem afetar o processo de exportação incluem a tarefa específica para a qual o modelo está sendo usado e os requisitos ou restrições do ambiente ou plataforma alvo. É importante considerar e configurar cuidadosamente essas configurações para garantir que o modelo exportado seja otimizado para o caso de uso pretendido e possa ser usado eficazmente no ambiente alvo.

| Chave       | Valor           | Descrição                                                           |
|-------------|-----------------|---------------------------------------------------------------------|
| `format`    | `'torchscript'` | formato para exportação                                             |
| `imgsz`     | `640`           | tamanho da imagem como escalar ou lista (h, w), ou seja, (640, 480) |
| `keras`     | `False`         | usar Keras para exportação TF SavedModel                            |
| `optimize`  | `False`         | TorchScript: otimizar para mobile                                   |
| `half`      | `False`         | quantização FP16                                                    |
| `int8`      | `False`         | quantização INT8                                                    |
| `dynamic`   | `False`         | ONNX/TensorRT: eixos dinâmicos                                      |
| `simplify`  | `False`         | ONNX/TensorRT: simplificar modelo                                   |
| `opset`     | `None`          | ONNX: versão do opset (opcional, padrão para a mais recente)        |
| `workspace` | `4`             | TensorRT: tamanho do espaço de trabalho (GB)                        |
| `nms`       | `False`         | CoreML: adicionar NMS                                               |

## Formatos de Exportação

Os formatos de exportação disponíveis para YOLOv8 estão na tabela abaixo. Você pode exportar para qualquer formato usando o argumento `format`, ou seja, `format='onnx'` ou `format='engine'`.

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
