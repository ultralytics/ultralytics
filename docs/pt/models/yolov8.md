---
comments: true
description: Explore as emocionantes características do YOLOv8, a versão mais recente do nosso detector de objetos em tempo real! Saiba como as arquiteturas avançadas, modelos pré-treinados e o equilíbrio ideal entre precisão e velocidade tornam o YOLOv8 a escolha perfeita para as suas tarefas de detecção de objetos.
keywords: YOLOv8, Ultralytics, detector de objetos em tempo real, modelos pré-treinados, documentação, detecção de objetos, série YOLO, arquiteturas avançadas, precisão, velocidade
---

# YOLOv8

## Visão Geral

O YOLOv8 é a versão mais recente da série YOLO de detectores de objetos em tempo real, oferecendo um desempenho de ponta em termos de precisão e velocidade. Construindo sobre as inovações das versões anteriores do YOLO, o YOLOv8 introduz novas características e otimizações que o tornam uma escolha ideal para diversas tarefas de detecção de objetos em uma ampla variedade de aplicações.

![YOLOv8 da Ultralytics](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## Principais Características

- **Arquiteturas Avançadas de Backbone e Neck:** O YOLOv8 utiliza arquiteturas avançadas de backbone e neck, resultando em uma melhor extração de características e desempenho na detecção de objetos.
- **Anchor-free Split Ultralytics Head:** O YOLOv8 adota um head Ultralytics dividido sem ancoragem, o que contribui para uma melhor precisão e um processo de detecção mais eficiente em comparação com abordagens baseadas em âncoras.
- **Equilíbrio Otimizado entre Precisão e Velocidade:** Com foco em manter um equilíbrio ideal entre precisão e velocidade, o YOLOv8 é adequado para tarefas de detecção de objetos em tempo real em diversas áreas de aplicação.
- **Variedade de Modelos Pré-treinados:** O YOLOv8 oferece uma variedade de modelos pré-treinados para atender a diversas tarefas e requisitos de desempenho, tornando mais fácil encontrar o modelo adequado para o seu caso de uso específico.

## Tarefas e Modos Suportados

A série YOLOv8 oferece uma variedade de modelos, cada um especializado em tarefas específicas de visão computacional. Esses modelos são projetados para atender a diversos requisitos, desde a detecção de objetos até tarefas mais complexas, como segmentação de instâncias, detecção de poses/pontos-chave e classificação.

Cada variante da série YOLOv8 é otimizada para a respectiva tarefa, garantindo alto desempenho e precisão. Além disso, esses modelos são compatíveis com diversos modos operacionais, incluindo [Inferência](../modes/predict.md), [Validação](../modes/val.md), [Treinamento](../modes/train.md) e [Exportação](../modes/export.md), facilitando o uso em diferentes estágios de implantação e desenvolvimento.

| Modelo      | Nomes de Arquivo                                                                                               | Tarefa                                           | Inferência | Validação | Treinamento | Exportação |
|-------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------|------------|-----------|-------------|------------|
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [Detecção](../tasks/detect.md)                   | ✅          | ✅         | ✅           | ✅          |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [Segmentação de Instâncias](../tasks/segment.md) | ✅          | ✅         | ✅           | ✅          |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [Pose/Pontos-chave](../tasks/pose.md)            | ✅          | ✅         | ✅           | ✅          |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [Classificação](../tasks/classify.md)            | ✅          | ✅         | ✅           | ✅          |

Esta tabela fornece uma visão geral das variantes de modelos YOLOv8, destacando suas aplicações em tarefas específicas e sua compatibilidade com diversos modos operacionais, como inferência, validação, treinamento e exportação. Ela demonstra a versatilidade e robustez da série YOLOv8, tornando-os adequados para diversas aplicações em visão computacional.

## Métricas de Desempenho

!!! Desempenho

    === "Detecção (COCO)"

        Consulte a [Documentação de Detecção](https://docs.ultralytics.com/tasks/detect/) para exemplos de uso com esses modelos treinados no conjunto de dados [COCO](https://docs.ultralytics.com/datasets/detect/coco/), que inclui 80 classes pré-treinadas.

        | Modelo                                                                                  | tamanho<br><sup>(pixels) | mAP<sup>val<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | --------------------------------------------------------------------------------------- | ----------------------- | -------------------- | ----------------------------------- | --------------------------------------- | ---------------------- | ------------------ |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)   | 640                     | 37,3                 | 80,4                               | 0,99                                    | 3,2                    | 8,7                |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)   | 640                     | 44,9                 | 128,4                              | 1,20                                    | 11,2                   | 28,6               |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)   | 640                     | 50,2                 | 234,7                              | 1,83                                    | 25,9                   | 78,9               |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)   | 640                     | 52,9                 | 375,2                              | 2,39                                    | 43,7                   | 165,2              |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)   | 640                     | 53,9                 | 479,1                              | 3,53                                    | 68,2                   | 257,8              |

    === "Detecção (Open Images V7)"

        Consulte a [Documentação de Detecção](https://docs.ultralytics.com/tasks/detect/) para exemplos de uso com esses modelos treinados no conjunto de dados [Open Images V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), que inclui 600 classes pré-treinadas.

        | Modelo                                                                                     | tamanho<br><sup>(pixels) | mAP<sup>val<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | ----------------------------------------------------------------------------------------- | ----------------------- | -------------------- | ----------------------------------- | --------------------------------------- | ---------------------- | ------------------ |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                     | 18,4                 | 142,4                              | 1,21                                    | 3,5                    | 10,5               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                     | 27,7                 | 183,1                              | 1,40                                    | 11,4                   | 29,7               |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                     | 33,6                 | 408,5                              | 2,26                                    | 26,2                   | 80,6               |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                     | 34,9                 | 596,9                              | 2,43                                    | 44,1                   | 167,4              |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                     | 36,3                 | 860,6                              | 3,56                                    | 68,7                   | 260,6              |

    === "Segmentação (COCO)"

        Consulte a [Documentação de Segmentação](https://docs.ultralytics.com/tasks/segment/) para exemplos de uso com esses modelos treinados no conjunto de dados [COCO](https://docs.ultralytics.com/datasets/segment/coco/), que inclui 80 classes pré-treinadas.

        | Modelo                                                                                          | tamanho<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>máscara<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------- | ----------------------- | -------------------- | ------------------------ | ----------------------------------- | --------------------------------------- | ---------------------- | ------------------ |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt)   | 640                     | 36,7                 | 30,5                     | 96,1                                | 1,21                                    | 3,4                    | 12,6               |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt)   | 640                     | 44,6                 | 36,8                     | 155,7                               | 1,47                                    | 11,8                   | 42,6               |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt)   | 640                     | 49,9                 | 40,8                     | 317,0                               | 2,18                                    | 27,3                   | 110,2              |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt)   | 640                     | 52,3                 | 42,6                     | 572,4                               | 2,79                                    | 46,0                   | 220,5              |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt)   | 640                     | 53,4                 | 43,4                     | 712,1                               | 4,02                                    | 71,8                   | 344,1              |

    === "Classificação (ImageNet)"

        Consulte a [Documentação de Classificação](https://docs.ultralytics.com/tasks/classify/) para exemplos de uso com esses modelos treinados no conjunto de dados [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/), que inclui 1000 classes pré-treinadas.

        | Modelo                                                                                           | tamanho<br><sup>(pixels) | acurácia<br><sup>top1 | acurácia<br><sup>top5 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) a 640 |
        | ------------------------------------------------------------------------------------------------ | ----------------------- | --------------------- | --------------------- | ----------------------------------- | --------------------------------------- | ---------------------- | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt)     | 224                     | 66,6                  | 87,0                  | 12,9                                | 0,31                                    | 2,7                    | 4,3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt)     | 224                     | 72,3                  | 91,1                  | 23,4                                | 0,35                                    | 6,4                    | 13,5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt)     | 224                     | 76,4                  | 93,2                  | 85,4                                | 0,62                                    | 17,0                   | 42,7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt)     | 224                     | 78,0                  | 94,1                  | 163,0                               | 0,87                                    | 37,5                   | 99,7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt)     | 224                     | 78,4                  | 94,3                  | 232,0                               | 1,01                                    | 57,4                   | 154,8                    |

    === "Pose (COCO)"

        Consulte a [Documentação de Estimativa de Pose](https://docs.ultralytics.com/tasks/segment/) para exemplos de uso com esses modelos treinados no conjunto de dados [COCO](https://docs.ultralytics.com/datasets/pose/coco/), que inclui 1 classe pré-treinada, 'person'.

        | Modelo                                                                                         | tamanho<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------- | ----------------------- | --------------------- | ------------------ | ----------------------------------- | --------------------------------------- | ---------------------- | ------------------ |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt) | 640                     | 50,4                  | 80,1               | 131,8                               | 1,18                                    | 3,3                    | 9,2                |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt) | 640                     | 60,0                  | 86,2               | 233,2                               | 1,42                                    | 11,6                   | 30,2               |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt) | 640                     | 65,0                  | 88,8               | 456,3                               | 2,00                                    | 26,4                   | 81,0               |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt) | 640                     | 67,6                  | 90,0               | 784,5                               | 2,59                                    | 44,4                   | 168,6              |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt) | 640                     | 69,2                  | 90,2               | 1607,1                              | 3,73                                    | 69,4                   | 263,2              |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                    | 71,6                  | 91,2               | 4088,7                              | 10,04                                   | 99,1                   | 1066,4             |

## Exemplos de Uso

Este exemplo fornece exemplos simples de treinamento e inferência do YOLOv8. Para a documentação completa desses e outros [modos](../modes/index.md), consulte as páginas de documentação [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) e [Export](../modes/export.md).

Observe que o exemplo abaixo é para modelos YOLOv8 de [Detecção](../tasks/detect.md) para detecção de objetos. Para outras tarefas suportadas, consulte a documentação de [Segmentação](../tasks/segment.md), [Classificação](../tasks/classify.md) e [Pose](../tasks/pose.md).

!!! Example "Exemplo"

    === "Python"

        Modelos pré-treinados `*.pt` PyTorch, bem como arquivos de configuração `*.yaml`, podem ser passados para a classe `YOLO()` para criar uma instância do modelo em Python:

        ```python
        from ultralytics import YOLO

        # Carregar um modelo YOLOv8n pré-treinado para COCO
        model = YOLO('yolov8n.pt')

        # Exibir informações do modelo (opcional)
        model.info()

        # Treinar o modelo no exemplo de conjunto de dados COCO8 por 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Executar inferência com o modelo YOLOv8n na imagem 'bus.jpg'
        results = model('caminho/para/bus.jpg')
        ```

    === "CLI"

        Comandos da CLI estão disponíveis para executar os modelos diretamente:

        ```bash
        # Carregar um modelo YOLOv8n pré-treinado para COCO e treiná-lo no exemplo de conjunto de dados COCO8 por 100 épocas
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Carregar um modelo YOLOv8n pré-treinado para COCO e executar inferência na imagem 'bus.jpg'
        yolo predict model=yolov8n.pt source=caminho/para/bus.jpg
        ```

## Citações e Reconhecimentos

Se você utilizar o modelo YOLOv8 ou qualquer outro software deste repositório em seu trabalho, por favor cite-o utilizando o formato abaixo:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

Observe que o DOI está pendente e será adicionado à citação assim que estiver disponível. Os modelos YOLOv8 são disponibilizados sob as licenças [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) e [Enterprise](https://ultralytics.com/license).
