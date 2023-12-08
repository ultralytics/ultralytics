---
comments: true
description: Explore o Modelo de Segmentação de Qualquer Coisa (SAM) de última geração da Ultralytics que permite a segmentação de imagens em tempo real. Aprenda sobre a segmentação baseada em prompts, o desempenho de transferência zero e como utilizá-lo.
keywords: Ultralytics, segmentação de imagem, Modelo de Segmentação de Qualquer Coisa, SAM, conjunto de dados SA-1B, desempenho em tempo real, transferência zero, detecção de objetos, análise de imagens, aprendizado de máquina
---

# Modelo de Segmentação de Qualquer Coisa (SAM)

Bem-vindo à fronteira da segmentação de imagem com o Modelo de Segmentação de Qualquer Coisa, ou SAM. Este modelo revolucionário mudou o jogo ao introduzir a segmentação de imagem baseada em prompts com desempenho em tempo real, estabelecendo novos padrões no campo.

## Introdução ao SAM: O Modelo de Segmentação de Qualquer Coisa

O Modelo de Segmentação de Qualquer Coisa, ou SAM, é um modelo de segmentação de imagem de ponta que permite a segmentação baseada em prompts, proporcionando uma versatilidade incomparável em tarefas de análise de imagem. O SAM é o cerne da iniciativa Segment Anything, um projeto inovador que introduz um modelo, tarefa e conjunto de dados novos para a segmentação de imagem.

O design avançado do SAM permite que ele se adapte a novas distribuições de imagem e tarefas sem conhecimento prévio, um recurso conhecido como transferência zero. Treinado no abrangente [conjunto de dados SA-1B](https://ai.facebook.com/datasets/segment-anything/), que contém mais de 1 bilhão de máscaras espalhadas por 11 milhões de imagens cuidadosamente selecionadas, o SAM tem demonstrado um impressionante desempenho de transferência zero, superando os resultados totalmente supervisionados anteriores em muitos casos.

![Exemplo de imagem do conjunto de dados](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
Imagens de exemplo com máscaras sobrepostas do nosso conjunto de dados recém-introduzido, SA-1B. O SA-1B contém 11 milhões de imagens diversas, de alta resolução, licenciadas e com proteção de privacidade, e 1,1 bilhão de máscaras de segmentação de alta qualidade. Essas máscaras foram anotadas totalmente automaticamente pelo SAM, e, como verificado por classificações humanas e inúmeros experimentos, são de alta qualidade e diversidade. As imagens são agrupadas pelo número de máscaras por imagem para visualização (em média, há ∼100 máscaras por imagem).

## Recursos Principais do Modelo de Segmentação de Qualquer Coisa (SAM)

- **Tarefa de Segmentação Baseada em Prompts:** O SAM foi projetado com uma tarefa de segmentação baseada em prompts em mente, permitindo que ele gere máscaras de segmentação válidas a partir de qualquer prompt fornecido, como dicas espaciais ou textuais que identifiquem um objeto.
- **Arquitetura Avançada:** O Modelo de Segmentação de Qualquer Coisa utiliza um poderoso codificador de imagens, um codificador de prompts e um decodificador de máscaras leve. Essa arquitetura única possibilita o uso flexível de prompts, cálculo de máscaras em tempo real e consciência de ambiguidade em tarefas de segmentação.
- **O Conjunto de Dados SA-1B:** Introduzido pelo projeto Segment Anything, o conjunto de dados SA-1B apresenta mais de 1 bilhão de máscaras em 11 milhões de imagens. Como o maior conjunto de dados de segmentação até o momento, ele fornece ao SAM uma fonte diversificada e em grande escala de dados de treinamento.
- **Desempenho de Transferência Zero:** O SAM apresenta um desempenho de transferência zero excepcional em diversas tarefas de segmentação, tornando-se uma ferramenta pronta para uso em aplicações diversas com necessidade mínima de engenharia de prompts.

Para obter uma visão mais aprofundada do Modelo de Segmentação de Qualquer Coisa e do conjunto de dados SA-1B, visite o [site do Segment Anything](https://segment-anything.com) e consulte o artigo de pesquisa [Segment Anything](https://arxiv.org/abs/2304.02643).

## Modelos Disponíveis, Tarefas Suportadas e Modos de Operação

Esta tabela apresenta os modelos disponíveis com seus pesos pré-treinados específicos, as tarefas suportadas por eles e sua compatibilidade com diferentes modos de operação, como [Inferência](../modes/predict.md), [Validação](../modes/val.md), [Treinamento](../modes/train.md) e [Exportação](../modes/export.md), indicados pelos emojis ✅ para modos suportados e ❌ para modos não suportados.

| Tipo de Modelo | Pesos Pré-Treinados | Tarefas Suportadas                               | Inferência | Validação | Treinamento | Exportação |
|----------------|---------------------|--------------------------------------------------|------------|-----------|-------------|------------|
| SAM base       | `sam_b.pt`          | [Segmentação de Instâncias](../tasks/segment.md) | ✅          | ❌         | ❌           | ✅          |
| SAM large      | `sam_l.pt`          | [Segmentação de Instâncias](../tasks/segment.md) | ✅          | ❌         | ❌           | ✅          |

## Como Usar o SAM: Versatilidade e Poder na Segmentação de Imagens

O Modelo de Segmentação de Qualquer Coisa pode ser utilizado para uma variedade de tarefas secundárias que vão além dos dados de treinamento. Isso inclui detecção de bordas, geração de propostas de objeto, segmentação de instâncias e predição preliminar de texto para máscara. Com a engenharia de prompts, o SAM pode se adaptar rapidamente a novas tarefas e distribuições de dados de maneira inovadora, estabelecendo-se como uma ferramenta versátil e poderosa para todas as suas necessidades de segmentação de imagem.

### Exemplo de predição do SAM

!!! Example "Segmentar com prompts"

    Segmenta a imagem com prompts fornecidos.

    === "Python"

        ```python
        from ultralytics import SAM

        # Carregar o modelo
        modelo = SAM('sam_b.pt')

        # Exibir informações do modelo (opcional)
        modelo.info()

        # Executar inferência com prompt de bboxes
        modelo('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

        # Executar inferência com prompt de pontos
        modelo('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

!!! Example "Segmentar tudo"

    Segmenta toda a imagem.

    === "Python"

        ```python
        from ultralytics import SAM

        # Carregar o modelo
        modelo = SAM('sam_b.pt')

        # Exibir informações do modelo (opcional)
        modelo.info()

        # Executar inferência
        modelo('caminho/para/imagem.jpg')
        ```

    === "CLI"

        ```bash
        # Executar inferência com um modelo SAM
        yolo predict model=sam_b.pt source=caminho/para/imagem.jpg
        ```

- A lógica aqui é segmentar toda a imagem se nenhum prompt (bboxes/pontos/máscaras) for especificado.

!!! Example "Exemplo do SAMPredictor"

    Desta forma, você pode definir a imagem uma vez e executar inferência de prompts várias vezes sem executar o codificador de imagem várias vezes.

    === "Inferência com prompt"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Criar o SAMPredictor
        substituições = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(substituições=substituições)

        # Definir imagem
        predictor.set_image("ultralytics/assets/zidane.jpg")  # definir com arquivo de imagem
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # definir com np.ndarray
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # Redefinir imagem
        predictor.reset_image()
        ```

    Segmentar tudo com argumentos adicionais.

    === "Segmentar tudo"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Criar o SAMPredictor
        substituições = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(substituições=substituições)

        # Segmentar com argumentos adicionais
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- Mais argumentos adicionais para `Segmentar tudo` consulte a [Referência do `Predictor/generate`](../../../reference/models/sam/predict.md).

## Comparação SAM vs. YOLOv8

Aqui, comparamos o menor modelo SAM-b da Meta com o menor modelo de segmentação da Ultralytics, [YOLOv8n-seg](../tasks/segment.md):

| Modelo                                        | Tamanho                       | Parâmetros                     | Velocidade (CPU)                     |
|-----------------------------------------------|-------------------------------|--------------------------------|--------------------------------------|
| SAM-b da Meta                                 | 358 MB                        | 94,7 M                         | 51096 ms/im                          |
| [MobileSAM](mobile-sam.md)                    | 40,7 MB                       | 10,1 M                         | 46122 ms/im                          |
| [FastSAM-s](fast-sam.md) com YOLOv8 como base | 23,7 MB                       | 11,8 M                         | 115 ms/im                            |
| YOLOv8n-seg da Ultralytics                    | **6,7 MB** (53,4 vezes menor) | **3,4 M** (27,9 vezes a menos) | **59 ms/im** (866 vezes mais rápido) |

Essa comparação mostra as diferenças de ordem de magnitude nos tamanhos e velocidades dos modelos. Enquanto o SAM apresenta capacidades exclusivas para segmentação automática, ele não é um concorrente direto dos modelos de segmentação YOLOv8, que são menores, mais rápidos e mais eficientes.

Os testes foram executados em um MacBook Apple M2 de 2023 com 16GB de RAM. Para reproduzir este teste:

!!! Example "Exemplo"

    === "Python"
        ```python
        from ultralytics import FastSAM, SAM, YOLO

        # Perfil do SAM-b
        modelo = SAM('sam_b.pt')
        modelo.info()
        modelo('ultralytics/assets')

        # Perfil do MobileSAM
        modelo = SAM('mobile_sam.pt')
        modelo.info()
        modelo('ultralytics/assets')

        # Perfil do FastSAM-s
        modelo = FastSAM('FastSAM-s.pt')
        modelo.info()
        modelo('ultralytics/assets')

        # Perfil do YOLOv8n-seg
        modelo = YOLO('yolov8n-seg.pt')
        modelo.info()
        modelo('ultralytics/assets')
        ```

## Autoanotação: Um Caminho Rápido para Conjuntos de Dados de Segmentação

A autoanotação é um recurso-chave do SAM que permite aos usuários gerar um [conjunto de dados de segmentação](https://docs.ultralytics.com/datasets/segment) usando um modelo de detecção pré-treinado. Esse recurso permite a anotação rápida e precisa de um grande número de imagens, contornando a necessidade de anotação manual demorada.

### Gere seu Conjunto de Dados de Segmentação Usando um Modelo de Detecção

Para fazer a autoanotação do seu conjunto de dados com o framework Ultralytics, use a função `auto_annotate` conforme mostrado abaixo:

!!! Example "Exemplo"

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="caminho/para/imagens", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

| Argumento  | Tipo                | Descrição                                                                                                 | Padrão       |
|------------|---------------------|-----------------------------------------------------------------------------------------------------------|--------------|
| data       | str                 | Caminho para uma pasta que contém as imagens a serem anotadas.                                            |              |
| det_model  | str, opcional       | Modelo de detecção YOLO pré-treinado. O padrão é 'yolov8x.pt'.                                            | 'yolov8x.pt' |
| sam_model  | str, opcional       | Modelo de segmentação SAM pré-treinado. O padrão é 'sam_b.pt'.                                            | 'sam_b.pt'   |
| device     | str, opcional       | Dispositivo no qual executar os modelos. O padrão é uma string vazia (CPU ou GPU, se disponível).         |              |
| output_dir | str, None, opcional | Diretório para salvar os resultados anotados. O padrão é uma pasta 'labels' no mesmo diretório de 'data'. | None         |

A função `auto_annotate` recebe o caminho para suas imagens, com argumentos opcionais para especificar os modelos de detecção pré-treinados e de segmentação SAM, o dispositivo onde executar os modelos e o diretório de saída para salvar os resultados anotados.

A autoanotação com modelos pré-treinados pode reduzir drasticamente o tempo e o esforço necessários para criar conjuntos de dados de segmentação de alta qualidade. Esse recurso é especialmente benéfico para pesquisadores e desenvolvedores que lidam com grandes coleções de imagens, pois permite que eles se concentrem no desenvolvimento e na avaliação do modelo, em vez de na anotação manual.

## Citações e Reconhecimentos

Se você encontrar o SAM útil em seu trabalho de pesquisa ou desenvolvimento, considere citar nosso artigo:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

Gostaríamos de expressar nossa gratidão à Meta AI por criar e manter esse recurso valioso para a comunidade de visão computacional.

*keywords: Segment Anything, Modelo de Segmentação de Qualquer Coisa, SAM, SAM da Meta, segmentação de imagem, segmentação baseada em prompts, desempenho de transferência zero, conjunto de dados SA-1B, arquitetura avançada, autoanotação, Ultralytics, modelos pré-treinados, SAM base, SAM large, segmentação de instâncias, visão computacional, IA, inteligência artificial, aprendizado de máquina, anotação de dados, máscaras de segmentação, modelo de detecção, modelo de detecção YOLO, bibtex, Meta AI.*
