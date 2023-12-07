---
comments: true
description: Explore a diversificada gama de modelos da família YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS e RT-DETR suportados pela Ultralytics. Comece com exemplos de uso tanto para CLI quanto para Python.
keywords: Ultralytics, documentação, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, modelos, arquiteturas, Python, CLI
---

# Modelos Suportados pela Ultralytics

Bem-vindo à documentação de modelos da Ultralytics! Oferecemos suporte para uma ampla gama de modelos, cada um adaptado para tarefas específicas como [detecção de objetos](../tasks/detect.md), [segmentação de instâncias](../tasks/segment.md), [classificação de imagens](../tasks/classify.md), [estimativa de pose](../tasks/pose.md) e [rastreamento de múltiplos objetos](../modes/track.md). Se você está interessado em contribuir com sua arquitetura de modelo para a Ultralytics, confira nosso [Guia de Contribuição](../../help/contributing.md).

!!! Note "Nota"

    🚧 Nossa documentação multilíngue está atualmente em construção e estamos trabalhando duro para melhorá-la. Obrigado pela sua paciência! 🙏

## Modelos em Destaque

Aqui estão alguns dos principais modelos suportados:

1. **[YOLOv3](../../models/yolov3.md)**: A terceira iteração da família de modelos YOLO, originalmente por Joseph Redmon, conhecida por suas capacidades eficientes de detecção de objetos em tempo real.
2. **[YOLOv4](../../models/yolov4.md)**: Uma atualização nativa do darknet para o YOLOv3, lançada por Alexey Bochkovskiy em 2020.
3. **[YOLOv5](../../models/yolov5.md)**: Uma versão aprimorada da arquitetura YOLO pela Ultralytics, oferecendo melhores trade-offs de desempenho e velocidade comparado às versões anteriores.
4. **[YOLOv6](../../models/yolov6.md)**: Lançado pela [Meituan](https://about.meituan.com/) em 2022, e em uso em muitos dos robôs autônomos de entrega da empresa.
5. **[YOLOv7](../../models/yolov7.md)**: Modelos YOLO atualizados lançados em 2022 pelos autores do YOLOv4.
6. **[YOLOv8](../../models/yolov8.md)**: A versão mais recente da família YOLO, com capacidades aprimoradas como segmentação de instâncias, estimativa de pose/pontos-chave e classificação.
7. **[Segment Anything Model (SAM)](../../models/sam.md)**: Modelo de Segment Everything (SAM) do Meta.
8. **[Mobile Segment Anything Model (MobileSAM)](../../models/mobile-sam.md)**: MobileSAM para aplicações móveis, pela Universidade Kyung Hee.
9. **[Fast Segment Anything Model (FastSAM)](../../models/fast-sam.md)**: FastSAM pelo Grupo de Análise de Imagem e Vídeo, Instituto de Automação, Academia Chinesa de Ciências.
10. **[YOLO-NAS](../../models/yolo-nas.md)**: Modelos YOLO de Pesquisa de Arquitetura Neural (NAS).
11. **[Realtime Detection Transformers (RT-DETR)](../../models/rtdetr.md)**: Modelos do Transformer de Detecção em Tempo Real (RT-DETR) da PaddlePaddle da Baidu.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="Reprodutor de vídeos do YouTube" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Assista:</strong> Execute modelos YOLO da Ultralytics em apenas algumas linhas de código.
</p>

## Começando: Exemplos de Uso

!!! Example "Exemplo"

    === "Python"

        Modelos `*.pt` pré-treinados com PyTorch, bem como arquivos de configuração `*.yaml`, podem ser passados para as classes `YOLO()`, `SAM()`, `NAS()` e `RTDETR()` para criar uma instância de modelo em Python:

        ```python
        from ultralytics import YOLO

        # Carregar um modelo YOLOv8n pré-treinado no COCO
        model = YOLO('yolov8n.pt')

        # Exibir informações do modelo (opcional)
        model.info()

        # Treinar o modelo no conjunto de dados de exemplo COCO8 por 100 épocas
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # Executar inferência com o modelo YOLOv8n na imagem 'bus.jpg'
        results = model('path/to/bus.jpg')
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

Interessado em contribuir com o seu modelo para a Ultralytics? Ótimo! Estamos sempre abertos à expansão de nosso portfólio de modelos.

1. **Fork no Repositório**: Comece fazendo um fork do [repositório GitHub da Ultralytics](https://github.com/ultralytics/ultralytics).

2. **Clone o Seu Fork**: Clone o seu fork para a sua máquina local e crie uma nova branch para trabalhar.

3. **Implemente Seu Modelo**: Adicione o seu modelo seguindo os padrões de codificação e diretrizes fornecidos em nosso [Guia de Contribuição](../../help/contributing.md).

4. **Teste Completamente**: Certifique-se de testar seu modelo rigorosamente, isoladamente e como parte do pipeline.

5. **Crie um Pull Request**: Uma vez que esteja satisfeito com seu modelo, crie um pull request para o repositório principal para revisão.

6. **Revisão de Código & Merge**: Após a revisão, se o seu modelo atender os nossos critérios, ele será combinado com o repositório principal.

Para etapas detalhadas, consulte nosso [Guia de Contribuição](../../help/contributing.md).
