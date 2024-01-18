---
comments: true
description: Descubra as características e benefícios do RT-DETR da Baidu, um detector de objetos em tempo real eficiente e adaptável baseado em Vision Transformers, incluindo modelos pré-treinados.
keywords: RT-DETR, Baidu, Vision Transformers, detecção de objetos, desempenho em tempo real, CUDA, TensorRT, seleção de consulta IoU, Ultralytics, API Python, PaddlePaddle
---

# RT-DETR da Baidu: Um Detector de Objetos em Tempo Real Baseado em Vision Transformers

## Visão Geral

O Real-Time Detection Transformer (RT-DETR), desenvolvido pela Baidu, é um detector de objetos de última geração que proporciona desempenho em tempo real mantendo alta precisão. Ele utiliza a potência dos Vision Transformers (ViT) para processar eficientemente recursos multiescala, separando a interação intra-escala e a fusão entre escalas. O RT-DETR é altamente adaptável, com suporte para ajuste flexível da velocidade de inferência usando diferentes camadas de decodificador sem a necessidade de retratamento. O modelo se destaca em backends acelerados como o CUDA com o TensorRT, superando muitos outros detectores de objetos em tempo real.

![Exemplo de imagem do modelo](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**Visão geral do RT-DETR da Baidu.** O diagrama da arquitetura do modelo RT-DETR mostra as últimas três etapas da espinha dorsal {S3, S4, S5} como entrada para o codificador. O codificador híbrido eficiente transforma recursos multiescala em uma sequência de recursos de imagem por meio da interação de recursos intra-escala (AIFI) e do módulo de fusão de recursos entre escalas (CCFM). A seleção de consulta, consciente da IoU, é utilizada para selecionar um número fixo de recursos de imagem para servir como consultas de objeto iniciais para o decodificador. Por fim, o decodificador com cabeçotes de previsão auxiliares otimiza iterativamente as consultas de objeto para gerar caixas e pontuações de confiança ([fonte](https://arxiv.org/pdf/2304.08069.pdf)).

### Características Principais

- **Codificador Híbrido Eficiente:** O RT-DETR da Baidu utiliza um codificador híbrido eficiente para processar recursos multiescala por meio da separação da interação intra-escala e da fusão entre escalas. Esse design exclusivo baseado em Vision Transformers reduz os custos computacionais e permite a detecção de objetos em tempo real.
- **Seleção de Consulta Consciente de IoU:** O RT-DETR da Baidu melhora a inicialização das consultas de objeto ao utilizar seleção de consulta consciente de IoU. Isso permite que o modelo foque nos objetos mais relevantes na cena, aprimorando a precisão da detecção.
- **Velocidade de Inferência Adaptável:** O RT-DETR da Baidu suporta ajustes flexíveis da velocidade de inferência ao utilizar diferentes camadas de decodificador sem a necessidade de retratamento. Essa adaptabilidade facilita a aplicação prática em diversos cenários de detecção de objetos em tempo real.

## Modelos Pré-Treinados

A API Python do Ultralytics fornece modelos pré-treinados do RT-DETR do PaddlePaddle com diferentes escalas:

- RT-DETR-L: 53,0% de AP em COCO val2017, 114 FPS em GPU T4
- RT-DETR-X: 54,8% de AP em COCO val2017, 74 FPS em GPU T4

## Exemplos de Uso

Este exemplo fornece exemplos simples de treinamento e inferência com o RT-DETRR. Para obter documentação completa sobre esses e outros [modos](../modes/index.md), consulte as páginas de documentação [Predict](../modes/predict.md), [Train](../modes/train.md), [Val](../modes/val.md) e [Export](../modes/export.md).

!!! Example "Exemplo"

    === "Python"

        ```python
        from ultralytics import RTDETR

        # Carregue um modelo RT-DETR-l pré-treinado no COCO
        model = RTDETR('rtdetr-l.pt')

        # Exiba informações do modelo (opcional)
        model.info()

        # Treine o modelo com o conjunto de dados de exemplo COCO8 por 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Execute a inferência com o modelo RT-DETR-l na imagem 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        ```bash
        # Carregue um modelo RT-DETR-l pré-treinado no COCO e treine-o com o conjunto de dados de exemplo COCO8 por 100 épocas
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # Carregue um modelo RT-DETR-l pré-treinado no COCO e execute a inferência na imagem 'bus.jpg'
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## Tarefas e Modos Suportados

Esta tabela apresenta os tipos de modelo, os pesos pré-treinados específicos, as tarefas suportadas por cada modelo e os vários modos ([Train](../modes/train.md), [Val](../modes/val.md), [Predict](../modes/predict.md), [Export](../modes/export.md)) que são suportados, indicados por emojis ✅.

| Tipo de Modelo       | Pesos Pré-treinados | Tarefas Suportadas                        | Inferência | Validação | Treinamento | Exportação |
|----------------------|---------------------|-------------------------------------------|------------|-----------|-------------|------------|
| RT-DETR Grande       | `rtdetr-l.pt`       | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |
| RT-DETR Extra-Grande | `rtdetr-x.pt`       | [Detecção de Objetos](../tasks/detect.md) | ✅          | ✅         | ✅           | ✅          |

## Citações e Reconhecimentos

Se você utilizar o RT-DETR da Baidu em seu trabalho de pesquisa ou desenvolvimento, por favor cite o [artigo original](https://arxiv.org/abs/2304.08069):

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2023detrs,
              title={DETRs Beat YOLOs on Real-time Object Detection},
              author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
              year={2023},
              eprint={2304.08069},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Gostaríamos de agradecer à Baidu e à equipe do [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) por criar e manter esse recurso valioso para a comunidade de visão computacional. Sua contribuição para o campo com o desenvolvimento do detector de objetos em tempo real baseado em Vision Transformers, RT-DETR, é muito apreciada.

*keywords: RT-DETR, Transformer, ViT, Vision Transformers, RT-DETR da Baidu, PaddlePaddle, modelos pré-treinados PaddlePaddle RT-DETR, uso do RT-DETR da Baidu, API Python do Ultralytics*
