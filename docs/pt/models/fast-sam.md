---
comments: true
description: Explore o FastSAM, uma solução baseada em CNN para segmentação em tempo real de objetos em imagens. Melhor interação do usuário, eficiência computacional e adaptabilidade em tarefas de visão.
keywords: FastSAM, aprendizado de máquina, solução baseada em CNN, segmentação de objetos, solução em tempo real, Ultralytics, tarefas de visão, processamento de imagens, aplicações industriais, interação do usuário
---

# Fast Segment Anything Model (FastSAM)

O Fast Segment Anything Model (FastSAM) é uma solução inovadora baseada em CNN em tempo real para a tarefa de Segmentar Qualquer Coisa. Essa tarefa foi projetada para segmentar qualquer objeto dentro de uma imagem com base em várias possíveis instruções de interação do usuário. O FastSAM reduz significativamente as demandas computacionais, mantendo um desempenho competitivo, tornando-o uma escolha prática para uma variedade de tarefas de visão.

![Visão geral da arquitetura do Fast Segment Anything Model (FastSAM)](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## Visão Geral

O FastSAM é projetado para abordar as limitações do [Segment Anything Model (SAM)](sam.md), um modelo Transformer pesado com requisitos substanciais de recursos computacionais. O FastSAM divide a tarefa de segmentar qualquer coisa em duas etapas sequenciais: segmentação de todas as instâncias e seleção guiada por instruções. A primeira etapa usa o [YOLOv8-seg](../tasks/segment.md) para produzir as máscaras de segmentação de todas as instâncias na imagem. Na segunda etapa, ele gera a região de interesse correspondente à instrução.

## Recursos Principais

1. **Solução em Tempo Real:** Aproveitando a eficiência computacional das CNNs, o FastSAM fornece uma solução em tempo real para a tarefa de segmentar qualquer coisa, tornando-o valioso para aplicações industriais que exigem resultados rápidos.

2. **Eficiência e Desempenho:** O FastSAM oferece uma redução significativa nas demandas computacionais e de recursos sem comprometer a qualidade do desempenho. Ele alcança um desempenho comparável ao SAM, mas com recursos computacionais drasticamente reduzidos, permitindo aplicações em tempo real.

3. **Segmentação Guiada por Instruções:** O FastSAM pode segmentar qualquer objeto dentro de uma imagem com base em várias possíveis instruções de interação do usuário, proporcionando flexibilidade e adaptabilidade em diferentes cenários.

4. **Baseado em YOLOv8-seg:** O FastSAM é baseado no [YOLOv8-seg](../tasks/segment.md), um detector de objetos equipado com um ramo de segmentação de instâncias. Isso permite que ele produza efetivamente as máscaras de segmentação de todas as instâncias em uma imagem.

5. **Resultados Competitivos em Bancos de Dados de Referência:** Na tarefa de proposta de objetos no MS COCO, o FastSAM alcança pontuações altas em uma velocidade significativamente mais rápida do que o [SAM](sam.md) em uma única NVIDIA RTX 3090, demonstrando sua eficiência e capacidade.

6. **Aplicações Práticas:** A abordagem proposta fornece uma nova solução prática para um grande número de tarefas de visão em alta velocidade, dezenas ou centenas de vezes mais rápido do que os métodos atuais.

7. **Viabilidade de Compressão do Modelo:** O FastSAM demonstra a viabilidade de um caminho que pode reduzir significativamente o esforço computacional, introduzindo uma prioridade artificial à estrutura, abrindo assim novas possibilidades para arquiteturas de modelos grandes para tarefas gerais de visão.

## Modelos Disponíveis, Tarefas Suportadas e Modos de Operação

Esta tabela apresenta os modelos disponíveis com seus pesos pré-treinados específicos, as tarefas que eles suportam e sua compatibilidade com diferentes modos de operação, como [Inferência](../modes/predict.md), [Validação](../modes/val.md), [Treinamento](../modes/train.md) e [Exportação](../modes/export.md), indicados por emojis ✅ para modos suportados e emojis ❌ para modos não suportados.

| Tipo de Modelo | Pesos Pré-treinados | Tarefas Suportadas                               | Inferência | Validação | Treinamento | Exportação |
|----------------|---------------------|--------------------------------------------------|------------|-----------|-------------|------------|
| FastSAM-s      | `FastSAM-s.pt`      | [Segmentação de Instâncias](../tasks/segment.md) | ✅          | ❌         | ❌           | ✅          |
| FastSAM-x      | `FastSAM-x.pt`      | [Segmentação de Instâncias](../tasks/segment.md) | ✅          | ❌         | ❌           | ✅          |

## Exemplos de Uso

Os modelos FastSAM são fáceis de integrar em suas aplicações Python. A Ultralytics fornece uma API Python amigável ao usuário e comandos de linha de comando (CLI) para facilitar o desenvolvimento.

### Uso de Predição

Para realizar detecção de objetos em uma imagem, use o método `predict` conforme mostrado abaixo:

!!! Example "Exemplo"

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # Definir uma fonte de inferência
        source = 'caminho/para/onibus.jpg'

        # Criar um modelo FastSAM
        model = FastSAM('FastSAM-s.pt')  # ou FastSAM-x.pt

        # Executar inferência em uma imagem
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # Preparar um objeto de Processo de Instruções
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Instrução: tudo
        ann = prompt_process.everything_prompt()

        # Forma padrão (bbox) [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # Instrução: texto
        ann = prompt_process.text_prompt(text='uma foto de um cachorro')

        # Instrução: ponto
        # pontos padrão [[0,0]] [[x1,y1],[x2,y2]]
        # ponto_label padrão [0] [1,0] 0:fundo, 1:frente
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```

    === "CLI"
        ```bash
        # Carregar um modelo FastSAM e segmentar tudo com ele
        yolo segment predict model=FastSAM-s.pt source=caminho/para/onibus.jpg imgsz=640
        ```

Este trecho de código demonstra a simplicidade de carregar um modelo pré-treinado e executar uma predição em uma imagem.

### Uso de Validação

A validação do modelo em um conjunto de dados pode ser feita da seguinte forma:

!!! Example "Exemplo"

    === "Python"
        ```python
        from ultralytics import FastSAM

        # Criar um modelo FastSAM
        model = FastSAM('FastSAM-s.pt')  # ou FastSAM-x.pt

        # Validar o modelo
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # Carregar um modelo FastSAM e validá-lo no conjunto de dados de exemplo COCO8 com tamanho de imagem 640
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

Observe que o FastSAM suporta apenas detecção e segmentação de uma única classe de objeto. Isso significa que ele reconhecerá e segmentará todos os objetos como a mesma classe. Portanto, ao preparar o conjunto de dados, você precisará converter todos os IDs de categoria de objeto para 0.

## Uso Oficial do FastSAM

O FastSAM também está disponível diretamente no repositório [https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM). Aqui está uma visão geral breve das etapas típicas que você pode seguir para usar o FastSAM:

### Instalação

1. Clone o repositório do FastSAM:
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. Crie e ative um ambiente Conda com Python 3.9:
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. Navegue até o repositório clonado e instale os pacotes necessários:
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. Instale o modelo CLIP:
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### Exemplo de Uso

1. Baixe um [checkpoint do modelo](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing).

2. Use o FastSAM para inferência. Exemplos de comandos:

    - Segmentar tudo em uma imagem:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - Segmentar objetos específicos usando uma instrução de texto:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "o cachorro amarelo"
      ```

    - Segmentar objetos dentro de uma caixa delimitadora (fornecer coordenadas da caixa no formato xywh):
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - Segmentar objetos próximos a pontos específicos:
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

Além disso, você pode experimentar o FastSAM através de um [demo no Colab](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing) ou no [demo web do HuggingFace](https://huggingface.co/spaces/An-619/FastSAM) para ter uma experiência visual.

## Citações e Reconhecimentos

Gostaríamos de reconhecer os autores do FastSAM por suas contribuições significativas no campo da segmentação de instâncias em tempo real:

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

O artigo original do FastSAM pode ser encontrado no [arXiv](https://arxiv.org/abs/2306.12156). Os autores disponibilizaram seu trabalho publicamente, e o código pode ser acessado no [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM). Agradecemos seus esforços em avançar o campo e tornar seu trabalho acessível à comunidade em geral.
