---
comments: true
description: Explore a ampla gama de modelos da família YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS e RT-DETR suportados pela Ultralytics. Comece com exemplos para uso tanto em CLI quanto em Python.
keywords: Ultralytics, documentação, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, modelos, arquiteturas, Python, CLI
---

# Modelos Suportados pela Ultralytics

Bem-vindo à documentação de modelos da Ultralytics! Oferecemos suporte para uma ampla variedade de modelos, cada um adaptado para tarefas específicas como [detecção de objetos](../tasks/detect.md), [segmentação de instâncias](../tasks/segment.md), [classificação de imagens](../tasks/classify.md), [estimativa de pose](../tasks/pose.md), e [rastreamento de múltiplos objetos](../modes/track.md). Se você tem interesse em contribuir com sua arquitetura de modelo para a Ultralytics, confira nosso [Guia de Contribuição](../../help/contributing.md).

!!! Note "Nota"

    🚧 Nossa documentação em vários idiomas está atualmente em construção, e estamos trabalhando arduamente para melhorá-la. Agradecemos sua paciência! 🙏

## Modelos em Destaque

Aqui estão alguns dos principais modelos suportados:

1. **[YOLOv3](yolov3.md)**: A terceira iteração da família de modelos YOLO, originalmente por Joseph Redmon, conhecida por suas capacidades eficientes de detecção de objetos em tempo real.
2. **[YOLOv4](yolov4.md)**: Uma atualização nativa para o darknet do YOLOv3, lançada por Alexey Bochkovskiy em 2020.
3. **[YOLOv5](yolov5.md)**: Uma versão aprimorada da arquitetura YOLO pela Ultralytics, oferecendo melhor desempenho e compensações de velocidade em comparação com as versões anteriores.
4. **[YOLOv6](yolov6.md)**: Lançado pela [Meituan](https://about.meituan.com/) em 2022, e em uso em muitos dos robôs autônomos de entregas da empresa.
5. **[YOLOv7](yolov7.md)**: Modelos YOLO atualizados lançados em 2022 pelos autores do YOLOv4.
6. **[YOLOv8](yolov8.md) NOVO 🚀**: A versão mais recente da família YOLO, apresentando capacidades aprimoradas, como segmentação de instâncias, estimativa de pose/pontos-chave e classificação.
7. **[Segment Anything Model (SAM)](sam.md)**: Modelo Segment Anything (SAM) da Meta.
8. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**: MobileSAM para aplicações móveis, pela Universidade Kyung Hee.
9. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**: FastSAM pelo Grupo de Análise de Imagem e Vídeo, Instituto de Automação, Academia Chinesa de Ciências.
10. **[YOLO-NAS](yolo-nas.md)**: Modelos de Pesquisa de Arquitetura Neural YOLO (NAS).
11. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)**: Modelos de Transformador de Detecção em Tempo Real (RT-DETR) do PaddlePaddle da Baidu.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Execute modelos YOLO da Ultralytics em apenas algumas linhas de código.
</p>

## Introdução: Exemplos de Uso

Este exemplo oferece exemplos simples de treinamento e inferência com YOLO. Para uma documentação completa sobre estes e outros [modos](../modes/index.md), veja as páginas de documentação de [Previsão](../modes/predict.md), [Treinamento](../modes/train.md), [Validação](../modes/val.md) e [Exportação](../modes/export.md).

Note que o exemplo abaixo é para modelos YOLOv8 [Detect](../tasks/detect.md) para detecção de objetos. Para tarefas suportadas adicionais, veja as documentações de [Segmentação](../tasks/segment.md), [Classificação](../tasks/classify.md) e [Pose](../tasks/pose.md).

!!! Example "Exemplo"

    === "Python"

        Modelos `*.pt` pré-treinados em PyTorch, bem como arquivos de configuração `*.yaml`, podem ser passados para as classes `YOLO()`, `SAM()`, `NAS()` e `RTDETR()` para criar uma instância de modelo em Python:

        ```python
        from ultralytics import YOLO

        # Carregar um modelo YOLOv8n pré-treinado no COCO
        modelo = YOLO('yolov8n.pt')

        # Exibir informações do modelo (opcional)
        modelo.info()

        # Treinar o modelo no conjunto de dados de exemplo COCO8 por 100 épocas
        resultados = modelo.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Executar inferência com o modelo YOLOv8n na imagem 'bus.jpg'
        resultados = modelo('path/to/bus.jpg')
        ```

    === "CLI"

        Comandos CLI estão disponíveis para executar diretamente os modelos:

        ```bash
        # Carregar um modelo YOLOv8n pré-treinado no COCO e treiná-lo no conjunto de dados de exemplo COCO8 por 100 épocas
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # Carregar um modelo YOLOv8n pré-treinado no COCO e executar inferência na imagem 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## Contribuindo com Novos Modelos

Interessado em contribuir com seu modelo para a Ultralytics? Ótimo! Estamos sempre abertos a expandir nosso portfólio de modelos.

1. **Fork do Repositório**: Comece fazendo um fork do [repositório no GitHub da Ultralytics](https://github.com/ultralytics/ultralytics).

2. **Clone Seu Fork**: Clone seu fork para a sua máquina local e crie uma nova branch para trabalhar.

3. **Implemente Seu Modelo**: Adicione seu modelo seguindo as normas e diretrizes de codificação fornecidas no nosso [Guia de Contribuição](../../help/contributing.md).

4. **Teste Cuidadosamente**: Assegure-se de testar seu modelo rigorosamente, tanto isoladamente quanto como parte do pipeline.

5. **Crie um Pull Request**: Uma vez que estiver satisfeito com seu modelo, crie um pull request para o repositório principal para revisão.

6. **Revisão de Código & Mesclagem**: Após a revisão, se seu modelo atender aos nossos critérios, ele será integrado ao repositório principal.

Para etapas detalhadas, consulte nosso [Guia de Contribuição](../../help/contributing.md).
