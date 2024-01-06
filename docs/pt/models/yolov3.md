---
comments: true
description: Obtenha uma visão geral do YOLOv3, YOLOv3-Ultralytics e YOLOv3u. Saiba mais sobre suas principais características, uso e tarefas suportadas para detecção de objetos.
keywords: YOLOv3, YOLOv3-Ultralytics, YOLOv3u, Detecção de Objetos, Inferência, Treinamento, Ultralytics
---

# YOLOv3, YOLOv3-Ultralytics, e YOLOv3u

## Visão Geral

Este documento apresenta uma visão geral de três modelos de detecção de objetos intimamente relacionados, nomeadamente o [YOLOv3](https://pjreddie.com/darknet/yolo/), [YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3) e [YOLOv3u](https://github.com/ultralytics/ultralytics).

1. **YOLOv3:** Esta é a terceira versão do algoritmo de detecção de objetos You Only Look Once (YOLO). Originalmente desenvolvido por Joseph Redmon, o YOLOv3 melhorou seus predecessores ao introduzir recursos como previsões em várias escalas e três tamanhos diferentes de kernels de detecção.

2. **YOLOv3-Ultralytics:** Esta é a implementação do YOLOv3 pela Ultralytics. Ela reproduz a arquitetura original do YOLOv3 e oferece funcionalidades adicionais, como suporte para mais modelos pré-treinados e opções de personalização mais fáceis.

3. **YOLOv3u:** Esta é uma versão atualizada do YOLOv3-Ultralytics que incorpora o cabeçalho dividido livre de âncoras e sem "objectness" usado nos modelos YOLOv8. O YOLOv3u mantém a mesma arquitetura de "backbone" e "neck" do YOLOv3, mas com o cabeçalho de detecção atualizado do YOLOv8.

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## Principais Características

- **YOLOv3:** Introduziu o uso de três escalas diferentes para detecção, aproveitando três tamanhos diferentes de kernels de detecção: 13x13, 26x26 e 52x52. Isso melhorou significativamente a precisão da detecção para objetos de diferentes tamanhos. Além disso, o YOLOv3 adicionou recursos como previsões multi-rótulos para cada caixa delimitadora e uma rede de extração de características melhor.

- **YOLOv3-Ultralytics:** A implementação do YOLOv3 pela Ultralytics oferece o mesmo desempenho do modelo original, porém possui suporte adicional para mais modelos pré-treinados, métodos de treinamento adicionais e opções de personalização mais fáceis. Isso torna o modelo mais versátil e fácil de usar para aplicações práticas.

- **YOLOv3u:** Este modelo atualizado incorpora o cabeçalho dividido livre de âncoras e "objectness" do YOLOv8. Ao eliminar a necessidade de caixas de âncoras pré-definidas e pontuações de "objectness", esse design de cabeçalho de detecção pode melhorar a capacidade do modelo de detectar objetos de tamanhos e formatos variados. Isso torna o YOLOv3u mais robusto e preciso para tarefas de detecção de objetos.

## Tarefas e Modos Suportados

A série YOLOv3, incluindo YOLOv3, YOLOv3-Ultralytics e YOLOv3u, foi projetada especificamente para tarefas de detecção de objetos. Esses modelos são conhecidos por sua eficácia em vários cenários do mundo real, equilibrando precisão e velocidade. Cada variante oferece recursos e otimizações únicos, tornando-os adequados para uma variedade de aplicações.

Os três modelos suportam um conjunto abrangente de modos, garantindo versatilidade em várias etapas do desenvolvimento e implantação de modelos. Esses modos incluem [Inferência](../modes/predict.md), [Validação](../modes/val.md), [Treinamento](../modes/train.md) e [Exportação](../modes/export.md), fornecendo aos usuários um conjunto completo de ferramentas para detecção eficaz de objetos.

| Tipo de Modelo     | Tarefas Suportadas                        | Inferência | Validação | Treinamento | Exportação |
|--------------------|-------------------------------------------|------------|-----------|-------------|------------|
| YOLOv3             | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |
| YOLOv3-Ultralytics | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |
| YOLOv3u            | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |

Esta tabela fornece uma visão rápida das capacidades de cada variante do YOLOv3, destacando sua versatilidade e adequação para várias tarefas e modos operacionais em fluxos de trabalho de detecção de objetos.

## Exemplos de Uso

Este exemplo apresenta exemplos simples de treinamento e inferência do YOLOv3. Para obter documentação completa sobre esses e outros [modos](../modes/index.md), consulte as páginas de documentação do [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) e [Export](../modes/export.md).

!!! Example "Exemplo"

    === "Python"

        Modelos pré-treinados do PyTorch `*.pt`, bem como arquivos de configuração `*.yaml`, podem ser passados para a classe `YOLO()` para criar uma instância do modelo em Python:

        ```python
        from ultralytics import YOLO

        # Carregue um modelo YOLOv3n pré-treinado na COCO
        model = YOLO('yolov3n.pt')

        # Exiba informações sobre o modelo (opcional)
        model.info()

        # Treine o modelo no conjunto de dados de exemplo COCO8 por 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Execute inferência com o modelo YOLOv3n na imagem 'bus.jpg'
        results = model('caminho/para/bus.jpg')
        ```

    === "CLI"

        Comandos CLI estão disponíveis para executar diretamente os modelos:

        ```bash
        # Carregue um modelo YOLOv3n pré-treinado na COCO e treine-o no conjunto de dados de exemplo COCO8 por 100 épocas
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # Carregue um modelo YOLOv3n pré-treinado na COCO e execute inferência na imagem 'bus.jpg'
        yolo predict model=yolov3n.pt source=caminho/para/bus.jpg
        ```

## Citações e Reconhecimentos

Se você utilizar o YOLOv3 em sua pesquisa, por favor, cite os artigos originais do YOLO e o repositório Ultralytics YOLOv3:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

Agradecemos a Joseph Redmon e Ali Farhadi por desenvolverem o YOLOv3 original.
