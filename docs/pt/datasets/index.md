---
comments: true
description: Explore diversos conjuntos de dados de visão computacional suportados pela Ultralytics para detecção de objetos, segmentação, estimativa de pose, classificação de imagens e rastreamento de múltiplos objetos.
keywords: visão computacional, conjuntos de dados, Ultralytics, YOLO, detecção de objetos, segmentação de instância, estimativa de pose, classificação de imagens, rastreamento de múltiplos objetos
---

# Visão Geral de Conjuntos de Dados

A Ultralytics oferece suporte para diversos conjuntos de dados para facilitar tarefas de visão computacional, como detecção, segmentação de instância, estimativa de pose, classificação e rastreamento de múltiplos objetos. Abaixo está uma lista dos principais conjuntos de dados da Ultralytics, seguidos por um resumo de cada tarefa de visão computacional e os respectivos conjuntos de dados.

!!! Note "Nota"

    🚧 Nossa documentação multilíngue está atualmente em construção e estamos trabalhando arduamente para melhorá-la. Obrigado pela sua paciência! 🙏

## [Conjuntos de Dados de Detecção](../../datasets/detect/index.md)

A técnica de detecção de objetos com caixas delimitadoras envolve detectar e localizar objetos em uma imagem desenhando uma caixa delimitadora ao redor de cada objeto.

- [Argoverse](../../datasets/detect/argoverse.md): Um conjunto de dados contendo dados de rastreamento 3D e previsão de movimento de ambientes urbanos com anotações detalhadas.
- [COCO](../../datasets/detect/coco.md): Um conjunto de dados em grande escala projetado para detecção de objetos, segmentação e legendagem com mais de 200 mil imagens etiquetadas.
- [COCO8](../../datasets/detect/coco8.md): Contém as primeiras 4 imagens do COCO train e COCO val, adequado para testes rápidos.
- [Global Wheat 2020](../../datasets/detect/globalwheat2020.md): Um conjunto de dados de imagens de espiga de trigo coletadas ao redor do mundo para tarefas de detecção e localização de objetos.
- [Objects365](../../datasets/detect/objects365.md): Um conjunto de dados de alta qualidade de grande escala para detecção de objetos com 365 categorias e mais de 600 mil imagens anotadas.
- [OpenImagesV7](../../datasets/detect/open-images-v7.md): Um conjunto de dados abrangente do Google com 1,7 milhão de imagens de treino e 42 mil imagens de validação.
- [SKU-110K](../../datasets/detect/sku-110k.md): Um conjunto de dados apresentando detecção de objetos densos em ambientes de varejo com mais de 11 mil imagens e 1,7 milhão de caixas delimitadoras.
- [VisDrone](../../datasets/detect/visdrone.md): Um conjunto de dados que contém informação de detecção de objetos e rastreamento de múltiplos objetos a partir de imagens capturadas por drones com mais de 10 mil imagens e sequências de vídeo.
- [VOC](../../datasets/detect/voc.md): O conjunto de dados Visual Object Classes (VOC) Pascal para detecção de objetos e segmentação com 20 classes de objetos e mais de 11 mil imagens.
- [xView](../../datasets/detect/xview.md): Um conjunto de dados para detecção de objetos em imagens aéreas com 60 categorias de objetos e mais de 1 milhão de objetos anotados.

## [Conjuntos de Dados de Segmentação de Instância](../../datasets/segment/index.md)

A segmentação de instância é uma técnica de visão computacional que identifica e localiza objetos em uma imagem ao nível de pixel.

- [COCO](../../datasets/segment/coco.md): Um conjunto de dados em grande escala projetado para detecção de objetos, tarefas de segmentação e legendagem com mais de 200 mil imagens etiquetadas.
- [COCO8-seg](../../datasets/segment/coco8-seg.md): Um conjunto de dados menor para tarefas de segmentação de instâncias, contendo um subconjunto de 8 imagens COCO com anotações de segmentação.

## [Estimativa de Pose](../../datasets/pose/index.md)

A estimativa de pose é uma técnica usada para determinar a pose do objeto em relação à câmera ou ao sistema de coordenadas do mundo.

- [COCO](../../datasets/pose/coco.md): Um conjunto de dados em grande escala com anotações de pose humana projetado para tarefas de estimativa de pose.
- [COCO8-pose](../../datasets/pose/coco8-pose.md): Um conjunto de dados menor para tarefas de estimativa de pose, contendo um subconjunto de 8 imagens COCO com anotações de pose humana.
- [Tiger-pose](../../datasets/pose/tiger-pose.md): Um conjunto de dados compacto consistindo de 263 imagens focadas em tigres, anotadas com 12 pontos-chave por tigre para tarefas de estimativa de pose.

## [Classificação](../../datasets/classify/index.md)

Classificação de imagens é uma tarefa de visão computacional que envolve categorizar uma imagem em uma ou mais classes ou categorias predefinidas com base em seu conteúdo visual.

- [Caltech 101](../../datasets/classify/caltech101.md): Um conjunto de dados contendo imagens de 101 categorias de objetos para tarefas de classificação de imagens.
- [Caltech 256](../../datasets/classify/caltech256.md): Uma versão estendida do Caltech 101 com 256 categorias de objetos e imagens mais desafiadoras.
- [CIFAR-10](../../datasets/classify/cifar10.md): Um conjunto de dados de 60 mil imagens coloridas de 32x32 em 10 classes, com 6 mil imagens por classe.
- [CIFAR-100](../../datasets/classify/cifar100.md): Uma versão estendida do CIFAR-10 com 100 categorias de objetos e 600 imagens por classe.
- [Fashion-MNIST](../../datasets/classify/fashion-mnist.md): Um conjunto de dados consistindo de 70 mil imagens em escala de cinza de 10 categorias de moda para tarefas de classificação de imagens.
- [ImageNet](../../datasets/classify/imagenet.md): Um conjunto de dados em grande escala para detecção de objetos e classificação de imagens com mais de 14 milhões de imagens e 20 mil categorias.
- [ImageNet-10](../../datasets/classify/imagenet10.md): Um subconjunto menor do ImageNet com 10 categorias para experimentação e teste mais rápidos.
- [Imagenette](../../datasets/classify/imagenette.md): Um subconjunto menor do ImageNet que contém 10 classes facilmente distinguíveis para treinamento e teste mais rápidos.
- [Imagewoof](../../datasets/classify/imagewoof.md): Um subconjunto do ImageNet mais desafiador contendo 10 categorias de raças de cães para tarefas de classificação de imagens.
- [MNIST](../../datasets/classify/mnist.md): Um conjunto de dados de 70 mil imagens em escala de cinza de dígitos manuscritos para tarefas de classificação de imagens.

## [Caixas Delimitadoras Orientadas (OBB)](../../datasets/obb/index.md)

As Caixas Delimitadoras Orientadas (OBB) é um método em visão computacional para detectar objetos angulados em imagens usando caixas delimitadoras rotacionadas, muitas vezes aplicado em imagens aéreas e de satélite.

- [DOTAv2](../../datasets/obb/dota-v2.md): Um popular conjunto de dados de imagens aéreas OBB com 1,7 milhão de instâncias e 11.268 imagens.

## [Rastreamento de Múltiplos Objetos](../../datasets/track/index.md)

O rastreamento de múltiplos objetos é uma técnica de visão computacional que envolve detectar e rastrear vários objetos ao longo do tempo em uma sequência de vídeo.

- [Argoverse](../../datasets/detect/argoverse.md): Um conjunto de dados contendo dados de rastreamento 3D e previsão de movimento de ambientes urbanos com anotações ricas para tarefas de rastreamento de múltiplos objetos.
- [VisDrone](../../datasets/detect/visdrone.md): Um conjunto de dados que contém informação de detecção de objetos e rastreamento de múltiplos objetos a partir de imagens capturadas por drones com mais de 10 mil imagens e sequências de vídeo.

## Contribuir com Novos Conjuntos de Dados

Contribuir com um novo conjunto de dados envolve várias etapas para garantir que ele se alinhe bem com a infraestrutura existente. Abaixo estão as etapas necessárias:

### Etapas para Contribuir com um Novo Conjunto de Dados

1. **Coletar Imagens**: Reúna as imagens que pertencem ao conjunto de dados. Estas podem ser coletadas de várias fontes, como bancos de dados públicos ou sua própria coleção.

2. **Anotar Imagens**: Anote essas imagens com caixas delimitadoras, segmentos ou pontos-chave, dependendo da tarefa.

3. **Exportar Anotações**: Converta essas anotações no formato de arquivo `*.txt` YOLO que a Ultralytics suporta.

4. **Organizar Conjunto de Dados**: Organize seu conjunto de dados na estrutura de pastas correta. Você deve ter diretórios de topo `train/` e `val/`, e dentro de cada um, um subdiretório `images/` e `labels/`.

    ```
    conjunto_de_dados/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

5. **Criar um Arquivo `data.yaml`**: No diretório raiz do seu conjunto de dados, crie um arquivo `data.yaml` que descreva o conjunto de dados, as classes e outras informações necessárias.

6. **Otimizar Imagens (Opcional)**: Se você quiser reduzir o tamanho do conjunto de dados para um processamento mais eficiente, pode otimizar as imagens usando o código abaixo. Isso não é obrigatório, mas recomendado para tamanhos menores de conjunto de dados e velocidades de download mais rápidas.

7. **Compactar Conjunto de Dados**: Compacte toda a pasta do conjunto de dados em um arquivo zip.

8. **Documentar e PR**: Crie uma página de documentação descrevendo seu conjunto de dados e como ele se encaixa no framework existente. Depois disso, submeta um Pull Request (PR). Consulte [Diretrizes de Contribuição da Ultralytics](https://docs.ultralytics.com/help/contributing) para mais detalhes sobre como submeter um PR.

### Exemplo de Código para Otimizar e Compactar um Conjunto de Dados

!!! Example "Otimizar e Compactar um Conjunto de Dados"

    === "Python"

    ```python
    from pathlib import Path
    from ultralytics.data.utils import compress_one_image
    from ultralytics.utils.downloads import zip_directory

    # Definir diretório do conjunto de dados
    path = Path('caminho/para/conjunto_de_dados')

    # Otimizar imagens no conjunto de dados (opcional)
    for f in path.rglob('*.jpg'):
        compress_one_image(f)

    # Compactar conjunto de dados em 'caminho/para/conjunto_de_dados.zip'
    zip_directory(path)
    ```

Seguindo esses passos, você poderá contribuir com um novo conjunto de dados que se integra bem com a estrutura existente da Ultralytics.
