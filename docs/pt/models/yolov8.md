---
comments: true
description: Explore as emocionantes funcionalidades do YOLOv8, a versão mais recente do nosso detector de objetos em tempo real! Saiba como arquiteturas avançadas, modelos pré-treinados e o equilíbrio ideal entre precisão e velocidade tornam o YOLOv8 a escolha perfeita para suas tarefas de detecção de objetos.
keywords: YOLOv8, Ultralytics, detector de objetos em tempo real, modelos pré-treinados, documentação, detecção de objetos, série YOLO, arquiteturas avançadas, precisão, velocidade
---

# YOLOv8

## Visão Geral

O YOLOv8 é a versão mais recente na série YOLO de detectores de objetos em tempo real, oferecendo desempenho de ponta em termos de precisão e velocidade. Com base nos avanços das versões anteriores do YOLO, o YOLOv8 introduz novas funcionalidades e otimizações que o tornam uma escolha ideal para várias tarefas de detecção de objetos em várias aplicações.

![YOLOv8 Ultralytics](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## Funcionalidades Principais

- **Arquiteturas Avançadas de Backbone e Neck:** O YOLOv8 utiliza arquiteturas de backbone e neck de última geração, resultando em melhor extração de características e desempenho de detecção de objetos.
- **Cabeça Ultralytics Sem Âncora:** O YOLOv8 adota uma cabeça Ultralytics sem âncoras, o que contribui para uma melhor precisão e um processo de detecção mais eficiente em comparação com abordagens baseadas em âncoras.
- **Equilíbrio Otimizado entre Precisão e Velocidade:** Com foco em manter um equilíbrio ideal entre precisão e velocidade, o YOLOv8 é adequado para tarefas de detecção de objetos em tempo real em diversas áreas de aplicação.
- **Variedade de Modelos Pré-Treinados:** O YOLOv8 oferece uma variedade de modelos pré-treinados para atender a várias tarefas e requisitos de desempenho, tornando mais fácil encontrar o modelo certo para o seu caso de uso específico.

## Tarefas Suportadas

| Tipo de Modelo | Pesos Pré-Treinados                                                                                                 | Tarefa                   |
|----------------|---------------------------------------------------------------------------------------------------------------------|--------------------------|
| YOLOv8         | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`                                                | Detecção                 |
| YOLOv8-seg     | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`                            | Segmentação de Instância |
| YOLOv8-pose    | `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, `yolov8l-pose.pt`, `yolov8x-pose.pt`, `yolov8x-pose-p6.pt` | Poses/Pontos Chave       |
| YOLOv8-cls     | `yolov8n-cls.pt`, `yolov8s-cls.pt`, `yolov8m-cls.pt`, `yolov8l-cls.pt`, `yolov8x-cls.pt`                            | Classificação            |

## Modos Suportados

| Modo        | Suportado |
|-------------|-----------|
| Inferência  | ✅         |
| Validação   | ✅         |
| Treinamento | ✅         |

!!! Performance

    === "Detecção (COCO)"

        | Modelo                                                                                | tamanho<br><sup>(pixels) | mAP<sup>val<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37,3                 | 80,4                           | 0,99                                | 3,2                | 8,7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44,9                 | 128,4                          | 1,20                                | 11,2               | 28,6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50,2                 | 234,7                          | 1,83                                | 25,9               | 78,9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52,9                 | 375,2                          | 2,39                                | 43,7               | 165,2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53,9                 | 479,1                          | 3,53                                | 68,2               | 257,8             |

    === "Detecção (Open Images V7)"

        Consulte a [Documentação de Detecção](https://docs.ultralytics.com/tasks/detect/) para exemplos de uso desses modelos treinados no [Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), que incluem 600 classes pré-treinadas.

        | Modelo                                                                                     | tamanho<br><sup>(pixels) | mAP<sup>val<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | ----------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640                   | 18,4                 | 142,4                          | 1,21                                | 3,5                | 10,5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640                   | 27,7                 | 183,1                          | 1,40                                | 11,4               | 29,7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640                   | 33,6                 | 408,5                          | 2,26                                | 26,2               | 80,6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640                   | 34,9                 | 596,9                          | 2,43                                | 44,1               | 167,4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640                   | 36,3                 | 860,6                          | 3,56                                | 68,7               | 260,6             |

    === "Segmentação (COCO)"

        | Modelo                                                                                        | tamanho<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>máscara<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36,7                 | 30,5                    | 96,1                           | 1,21                                | 3,4                | 12,6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44,6                 | 36,8                    | 155,7                          | 1,47                                | 11,8               | 42,6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49,9                 | 40,8                    | 317,0                          | 2,18                                | 27,3               | 110,2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52,3                 | 42,6                    | 572,4                          | 2,79                                | 46,0               | 220,5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53,4                 | 43,4                    | 712,1                          | 4,02                                | 71,8               | 344,1             |

    === "Classificação (ImageNet)"

        | Modelo                                                                                        | tamanho<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) a 640 |
        | -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66,6             | 87,0             | 12,9                           | 0,31                                | 2,7                | 4,3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72,3             | 91,1             | 23,4                           | 0,35                                | 6,4                | 13,5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76,4             | 93,2             | 85,4                           | 0,62                                | 17,0               | 42,7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78,0             | 94,1             | 163,0                          | 0,87                                | 37,5               | 99,7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78,4             | 94,3             | 232,0                          | 1,01                                | 57,4               | 154,8                    |

    === "Pose (COCO)"

        | Modelo                                                                                                | tamanho<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50,4                  | 80,1               | 131,8                          | 1,18                                | 3,3                | 9,2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60,0                  | 86,2               | 233,2                          | 1,42                                | 11,6               | 30,2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65,0                  | 88,8               | 456,3                          | 2,00                                | 26,4               | 81,0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67,6                  | 90,0               | 784,5                          | 2,59                                | 44,4               | 168,6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69,2                  | 90,2               | 1607,1                         | 3,73                                | 69,4               | 263,2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71,6                  | 91,2               | 4088,7                         | 10,04                               | 99,1               | 1066,4            |

## Uso

Você pode usar o YOLOv8 para tarefas de detecção de objetos usando o pacote pip do Ultralytics. O seguinte é um trecho de código de exemplo mostrando como usar os modelos YOLOv8 para inferência:

!!! Exemplo ""

    Este exemplo fornece um código de inferência simples para o YOLOv8. Para obter mais opções, inclusive para lidar com os resultados da inferência, consulte o modo [Prever](../modes/predict.md). Para usar o YOLOv8 com modos adicionais, consulte [Treinar](../modes/train.md), [Val](../modes/val.md) e [Exportar](../modes/export.md).

    === "Python"

        Modelos pré-treinados `*.pt` do PyTorch, assim como arquivos de configuração `*.yaml`, podem ser passados à classe `YOLO()` para criar uma instância do modelo em Python:

        ```python
        from ultralytics import YOLO

        # Carregar um modelo YOLOv8n pré-treinado no COCO
        modelo = YOLO('yolov8n.pt')

        # Exibir informações sobre o modelo (opcional)
        modelo.info()

        # Treinar o modelo no conjunto de dados de exemplo COCO8 por 100 épocas
        resultados = modelo.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Fazer inferência com o modelo YOLOv8n na imagem 'bus.jpg'
        resultados = modelo('path/to/bus.jpg')
        ```

    === "CLI"

        Comandos CLI estão disponíveis para executar os modelos diretamente:

        ```bash
        # Carregar um modelo YOLOv8n pré-treinado no COCO e treiná-lo no conjunto de dados de exemplo COCO8 por 100 épocas
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Carregar um modelo YOLOv8n pré-treinado no COCO e fazer inferência na imagem 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Citações e Agradecimentos

Se você usar o modelo YOLOv8 ou qualquer outro software deste repositório em seu trabalho, cite-o usando o seguinte formato:

!!! Nota ""

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

Observe que o DOI está pendente e será adicionado à citação assim que estiver disponível. O uso deste software está em conformidade com a licença AGPL-3.0.
