---
comments: true
description: Descubra o YOLOv5u, uma versão aprimorada do modelo YOLOv5 com uma relação aprimorada entre precisão e velocidade e vários modelos pré-treinados para várias tarefas de detecção de objetos.
keywords: YOLOv5u, detecção de objetos, modelos pré-treinados, Ultralytics, Inferência, Validação, YOLOv5, YOLOv8, sem âncora, sem certeza de objectness, aplicativos em tempo real, machine learning
---

# YOLOv5

## Visão Geral

O YOLOv5u representa um avanço nas metodologias de detecção de objetos. Originário da arquitetura fundamental do modelo [YOLOv5](https://github.com/ultralytics/yolov5) desenvolvido pela Ultralytics, o YOLOv5u integra a divisão da cabeça do Ultralytics sem âncora e sem certeza de objectness, uma formação introduzida anteriormente nos modelos [YOLOv8](yolov8.md). Essa adaptação aprimora a arquitetura do modelo, resultando em uma relação aprimorada entre precisão e velocidade em tarefas de detecção de objetos. Com base nos resultados empíricos e em suas características derivadas, o YOLOv5u oferece uma alternativa eficiente para aqueles que procuram soluções robustas tanto na pesquisa quanto em aplicações práticas.

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## Principais Recursos

- **Cabeça do Ultralytics sem Âncora:** Modelos tradicionais de detecção de objetos dependem de caixas âncora predefinidas para prever as localizações dos objetos. No entanto, o YOLOv5u moderniza essa abordagem. Ao adotar uma cabeça do Ultralytics sem âncora, ele garante um mecanismo de detecção mais flexível e adaptável, melhorando consequentemente o desempenho em cenários diversos.

- **Equilíbrio otimizado entre precisão e velocidade:** Velocidade e precisão muitas vezes puxam em direções opostas. Mas o YOLOv5u desafia esse equilíbrio. Ele oferece um equilíbrio calibrado, garantindo detecções em tempo real sem comprometer a precisão. Esse recurso é particularmente valioso para aplicativos que exigem respostas rápidas, como veículos autônomos, robótica e análise de vídeo em tempo real.

- **Variedade de Modelos Pré-Treinados:** Entendendo que diferentes tarefas exigem conjuntos de ferramentas diferentes, o YOLOv5u oferece uma variedade de modelos pré-treinados. Se você está focado em Inferência, Validação ou Treinamento, há um modelo personalizado esperando por você. Essa variedade garante que você não esteja apenas usando uma solução genérica, mas sim um modelo ajustado especificamente para o seu desafio único.

## Tarefas e Modos Suportados

Os modelos YOLOv5u, com vários pesos pré-treinados, se destacam nas tarefas de [Detecção de Objetos](../tasks/detect.md). Eles suportam uma ampla gama de modos, tornando-os adequados para aplicações diversas, desde o desenvolvimento até a implantação.

| Tipo de Modelo | Pesos Pré-Treinados                                                                                                         | Tarefa                                    | Inferência | Validação | Treinamento | Exportação |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|------------|-----------|-------------|------------|
| YOLOv5u        | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |

Essa tabela oferece uma visão detalhada das variantes do modelo YOLOv5u, destacando sua aplicabilidade em tarefas de detecção de objetos e suporte a diversos modos operacionais, como [Inferência](../modes/predict.md), [Validação](../modes/val.md), [Treinamento](../modes/train.md) e [Exportação](../modes/export.md). Esse suporte abrangente garante que os usuários possam aproveitar totalmente as capacidades dos modelos YOLOv5u em uma ampla gama de cenários de detecção de objetos.

## Métricas de Desempenho

!!! Desempenho

    === "Detecção"

    Consulte a [Documentação de Detecção](https://docs.ultralytics.com/tasks/detect/) para exemplos de uso com esses modelos treinados no conjunto de dados [COCO](https://docs.ultralytics.com/datasets/detect/coco/), que incluem 80 classes pré-treinadas.

    | Modelo                                                                                       | YAML                                                                                                           | tamanho<br><sup>(pixels) | mAP<sup>val<br>50-95 | Velocidade<br><sup>CPU ONNX<br>(ms) | Velocidade<br><sup>A100 TensorRT<br>(ms) | parâmetros<br><sup>(M) | FLOPs<br><sup>(B) |
    | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------| -------------------------| ----------------------| -------------------------------------| -------------------------------------- | ---------------------- | ----------------- |
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt)    | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                  | 73.6                               | 1.06                                   | 2.6                     | 7.7              |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt)    | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                  | 120.7                              | 1.27                                   | 9.1                     | 24.0             |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt)    | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                  | 233.9                              | 1.86                                   | 25.1                    | 64.2             |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt)    | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                  | 408.4                              | 2.50                                   | 53.2                    | 135.0            |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5xu.pt)    | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                  | 763.2                              | 3.81                                   | 97.2                    | 246.4            |
    |                                                                                              |                                                                                                                 |                         |                       |                                     |                                        |                        |                  |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt)  | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                  | 211.0                              | 1.83                                   | 4.3                     | 7.8              |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt)  | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                  | 422.6                              | 2.34                                   | 15.3                    | 24.6             |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt)  | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                  | 810.9                              | 4.36                                   | 41.2                    | 65.7             |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt)  | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                  | 1470.9                             | 5.47                                   | 86.1                    | 137.4            |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5x6u.pt)  | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                  | 2436.5                             | 8.98                                   | 155.4                   | 250.7            |

## Exemplos de Uso

Este exemplo fornece exemplos simples de treinamento e inferência do YOLOv5. Para documentação completa sobre esses e outros [modos](../modes/index.md), consulte as páginas de documentação [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) e [Export](../modes/export.md).

!!! Example "Exemplo"

    === "Python"

        Modelos pré-treinados `*.pt` do PyTorch, assim como os arquivos de configuração `*.yaml`, podem ser passados para a classe `YOLO()` para criar uma instância do modelo em Python:

        ```python
        from ultralytics import YOLO

        # Carrega um modelo YOLOv5n pré-treinado no COCO
        modelo = YOLO('yolov5n.pt')

        # Mostra informações do modelo (opcional)
        modelo.info()

        # Treina o modelo no conjunto de dados de exemplo COCO8 por 100 épocas
        resultados = modelo.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Executa a inferência com o modelo YOLOv5n na imagem 'bus.jpg'
        resultados = modelo('path/to/bus.jpg')
        ```

    === "CLI"

        Comandos CLI estão disponíveis para executar diretamente os modelos:

        ```bash
        # Carrega um modelo YOLOv5n pré-treinado no COCO e o treina no conjunto de dados de exemplo COCO8 por 100 épocas
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # Carrega um modelo YOLOv5n pré-treinado no COCO e executa a inferência na imagem 'bus.jpg'
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## Citações e Agradecimentos

Se você usar o YOLOv5 ou YOLOv5u em sua pesquisa, por favor, cite o repositório YOLOv5 da Ultralytics da seguinte forma:

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

Observe que os modelos YOLOv5 são fornecidos sob licenças [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) e [Enterprise](https://ultralytics.com/license).
