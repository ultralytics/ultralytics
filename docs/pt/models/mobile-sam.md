---
comments: true
description: Saiba mais sobre o MobileSAM, sua implementação, comparação com o SAM original e como baixá-lo e testá-lo no framework Ultralytics. Melhore suas aplicações móveis hoje.
keywords: MobileSAM, Ultralytics, SAM, aplicações móveis, Arxiv, GPU, API, codificador de imagens, decodificador de máscaras, download do modelo, método de teste
---

![Logotipo do MobileSAM](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Segmentação Móvel de Qualquer Coisa (MobileSAM)

O artigo do MobileSAM agora está disponível no [arXiv](https://arxiv.org/pdf/2306.14289.pdf).

Uma demonstração do MobileSAM executando em uma CPU pode ser acessada neste [link de demonstração](https://huggingface.co/spaces/dhkim2810/MobileSAM). O desempenho em um Mac i5 CPU leva aproximadamente 3 segundos. Na demonstração do Hugging Face, a interface e CPUs de menor desempenho contribuem para uma resposta mais lenta, mas ela continua funcionando efetivamente.

O MobileSAM é implementado em vários projetos, incluindo [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling) e [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D).

O MobileSAM é treinado em uma única GPU com um conjunto de dados de 100 mil imagens (1% das imagens originais) em menos de um dia. O código para esse treinamento será disponibilizado no futuro.

## Modelos Disponíveis, Tarefas Suportadas e Modos de Operação

Esta tabela apresenta os modelos disponíveis com seus pesos pré-treinados específicos, as tarefas que eles suportam e sua compatibilidade com diferentes modos de operação, como [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md) e [Export](../modes/export.md), indicados pelos emojis ✅ para os modos suportados e ❌ para os modos não suportados.

| Tipo de Modelo | Pesos Pré-treinados | Tarefas Suportadas                               | Inference | Validation | Training | Export |
|----------------|---------------------|--------------------------------------------------|-----------|------------|----------|--------|
| MobileSAM      | `mobile_sam.pt`     | [Segmentação de Instâncias](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |

## Adaptação de SAM para MobileSAM

Como o MobileSAM mantém o mesmo pipeline do SAM original, incorporamos o pré-processamento original, pós-processamento e todas as outras interfaces. Consequentemente, aqueles que estão atualmente usando o SAM original podem fazer a transição para o MobileSAM com um esforço mínimo.

O MobileSAM tem um desempenho comparável ao SAM original e mantém o mesmo pipeline, exceto por uma mudança no codificador de imagens. Especificamente, substituímos o codificador de imagens ViT-H original (632M) por um ViT menor (5M). Em uma única GPU, o MobileSAM opera em cerca de 12 ms por imagem: 8 ms no codificador de imagens e 4 ms no decodificador de máscaras.

A tabela a seguir fornece uma comparação dos codificadores de imagens baseados em ViT:

| Codificador de Imagens | SAM Original | MobileSAM |
|------------------------|--------------|-----------|
| Parâmetros             | 611M         | 5M        |
| Velocidade             | 452ms        | 8ms       |

Tanto o SAM original quanto o MobileSAM utilizam o mesmo decodificador de máscaras baseado em prompt:

| Decodificador de Máscaras | SAM Original | MobileSAM |
|---------------------------|--------------|-----------|
| Parâmetros                | 3,876M       | 3,876M    |
| Velocidade                | 4ms          | 4ms       |

Aqui está a comparação de todo o pipeline:

| Pipeline Completo (Enc+Dec) | SAM Original | MobileSAM |
|-----------------------------|--------------|-----------|
| Parâmetros                  | 615M         | 9,66M     |
| Velocidade                  | 456ms        | 12ms      |

O desempenho do MobileSAM e do SAM original é demonstrado usando tanto um ponto quanto uma caixa como prompts.

![Imagem com Ponto como Prompt](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![Imagem com Caixa como Prompt](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

Com seu desempenho superior, o MobileSAM é aproximadamente 5 vezes menor e 7 vezes mais rápido que o FastSAM atual. Mais detalhes estão disponíveis na [página do projeto MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

## Testando o MobileSAM no Ultralytics

Assim como o SAM original, oferecemos um método de teste simples no Ultralytics, incluindo modos para prompts de Ponto e Caixa.

### Download do Modelo

Você pode baixar o modelo [aqui](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt).

### Prompt de Ponto

!!! Example "Exemplo"

    === "Python"
        ```python
        from ultralytics import SAM

        # Carregar o modelo
        model = SAM('mobile_sam.pt')

        # Prever um segmento com base em um prompt de ponto
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### Prompt de Caixa

!!! Example "Exemplo"

    === "Python"
        ```python
        from ultralytics import SAM

        # Carregar o modelo
        model = SAM('mobile_sam.pt')

        # Prever um segmento com base em um prompt de caixa
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
        ```

Implementamos `MobileSAM` e `SAM` usando a mesma API. Para obter mais informações sobre o uso, consulte a [página do SAM](sam.md).

## Citações e Agradecimentos

Se você achar o MobileSAM útil em sua pesquisa ou trabalho de desenvolvimento, considere citar nosso artigo:

!!! Citar ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
